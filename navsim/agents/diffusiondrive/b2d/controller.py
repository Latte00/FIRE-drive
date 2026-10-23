"""Trajectory-only closed-loop controller for FireDrive Bench2Drive agents."""

from collections import deque
import os

import numpy as np
from scipy.interpolate import PchipInterpolator


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


class _ScalarPID:
    def __init__(self, k_p: float, k_i: float, k_d: float, window: int):
        self.k_p = float(k_p)
        self.k_i = float(k_i)
        self.k_d = float(k_d)
        self._window = deque(maxlen=max(int(window), 1))

    def step(self, error: float) -> float:
        self._window.append(float(error))
        integral = float(np.mean(self._window))
        derivative = (
            self._window[-1] - self._window[-2]
            if len(self._window) >= 2
            else 0.0
        )
        return self.k_p * error + self.k_i * integral + self.k_d * derivative


class AutoMoTTrajectoryController:
    """AutoMoT-style controller that follows the predicted trajectory only.

    The parent Bench2Drive agent supplies waypoints either as PAD ``(right,
    front)`` coordinates or native FireDrive ``(front, left)`` coordinates.
    Internally this controller always uses ``(front, left)``. The route target
    argument is retained for API compatibility and diagnostics, but never
    replaces the trajectory-derived steering target.
    """

    def __init__(self, waypoint_frame: str = "pad"):
        self.waypoint_frame = str(waypoint_frame).lower()
        self.turn_k_p = _env_float("DIFFUSIONDRIVE_B2D_AUTOMOT_TURN_KP", 3.118)
        self.turn_k_i = _env_float(
            "DIFFUSIONDRIVE_B2D_AUTOMOT_TURN_KI", 0.6406067986034124
        )
        self.turn_k_d = _env_float(
            "DIFFUSIONDRIVE_B2D_AUTOMOT_TURN_KD", 1.3782508892109167
        )
        self.speed_scale = _env_float(
            "DIFFUSIONDRIVE_B2D_AUTOMOT_LOOKAHEAD_SPEED_SCALE",
            0.9755321901954155,
        )
        self.speed_offset = _env_float(
            "DIFFUSIONDRIVE_B2D_AUTOMOT_LOOKAHEAD_SPEED_OFFSET", 1.195
        )
        self.default_lookahead = _env_float(
            "DIFFUSIONDRIVE_B2D_AUTOMOT_MIN_LOOKAHEAD_DM", 24.0
        )
        self.max_lookahead = _env_float(
            "DIFFUSIONDRIVE_B2D_AUTOMOT_MAX_LOOKAHEAD_DM", 105.0
        )
        self.interpolation_step = max(
            _env_float("DIFFUSIONDRIVE_B2D_AUTOMOT_INTERPOLATION_M", 0.1),
            1e-3,
        )
        self.brake_speed = _env_float(
            "DIFFUSIONDRIVE_B2D_AUTOMOT_BRAKE_SPEED", 0.4
        )
        self.brake_ratio = _env_float(
            "DIFFUSIONDRIVE_B2D_AUTOMOT_BRAKE_RATIO", 1.1
        )
        self.clip_delta = _env_float(
            "DIFFUSIONDRIVE_B2D_AUTOMOT_CLIP_DELTA", 1.0
        )
        self.max_throttle = _env_float(
            "DIFFUSIONDRIVE_B2D_AUTOMOT_MAX_THROTTLE", 0.75
        )
        # FireDrive uses left-positive trajectory coordinates, while the
        # current CARLA control chain steers left with a negative command.
        self.steer_sign = _env_float(
            "DIFFUSIONDRIVE_B2D_AUTOMOT_STEER_SIGN", -1.0
        )
        self.turn_controller = _ScalarPID(
            self.turn_k_p, self.turn_k_i, self.turn_k_d, window=6
        )
        self.speed_controller = _ScalarPID(1.75, 1.0, 2.0, window=20)

    def _to_front_left(self, waypoints: np.ndarray) -> np.ndarray:
        waypoints = np.asarray(waypoints, dtype=np.float32)
        if waypoints.ndim != 2 or waypoints.shape[1] < 2:
            raise ValueError(
                f"Expected waypoints shaped (N, 2+), got {waypoints.shape}"
            )
        waypoints = waypoints[:, :2]
        if self.waypoint_frame in ("pad", "b2d", "right_front"):
            return np.stack([waypoints[:, 1], -waypoints[:, 0]], axis=-1)
        if self.waypoint_frame in ("raw", "diffusiondrive", "front_left"):
            return waypoints.copy()
        raise ValueError(f"Unsupported waypoint frame: {self.waypoint_frame}")

    def _from_front_left_point(self, point: np.ndarray) -> np.ndarray:
        if self.waypoint_frame in ("pad", "b2d", "right_front"):
            return np.asarray([-point[1], point[0]], dtype=np.float32)
        return np.asarray(point, dtype=np.float32)

    def _interpolate_waypoints(self, waypoints: np.ndarray) -> np.ndarray:
        points = np.concatenate(
            [np.zeros((1, 2), dtype=np.float32), waypoints], axis=0
        )
        segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        distances = np.concatenate(
            [np.zeros((1,), dtype=np.float64), np.cumsum(segment_lengths)]
        )
        # AutoMoT adds a tiny monotonic offset so stationary points remain
        # valid PCHIP knots.
        distances += np.arange(len(distances), dtype=np.float64) * 1e-4
        end_distance = float(distances[-1])
        if end_distance <= self.interpolation_step:
            return points[-1:].copy()
        samples = np.arange(
            self.interpolation_step,
            end_distance,
            self.interpolation_step,
            dtype=np.float64,
        )
        if samples.size == 0:
            return points[-1:].copy()
        return np.asarray(
            PchipInterpolator(distances, points, axis=0)(samples),
            dtype=np.float32,
        )

    @staticmethod
    def _normalized_heading(point: np.ndarray) -> float:
        angle = float(np.arctan2(float(point[1]), float(point[0])))
        angle = (angle + np.pi) % (2.0 * np.pi) - np.pi
        return float(np.degrees(angle) / 90.0)

    def control_pid(self, waypoints, speed, target):
        trajectory = self._to_front_left(waypoints)
        if len(trajectory) == 0 or not np.isfinite(trajectory).all():
            raise ValueError("AutoMoT controller received an invalid trajectory")

        speed_mps = float(np.asarray(speed).reshape(-1)[0])
        if len(trajectory) >= 2:
            # FireDrive predicts points every 0.5 s. AutoMoT estimates speed
            # from its 0.5-to-1.0 s segment rather than from the route target.
            desired_speed = float(np.linalg.norm(trajectory[1] - trajectory[0]) * 2.0)
        else:
            desired_speed = float(np.linalg.norm(trajectory[0]) * 2.0)

        route = self._interpolate_waypoints(trajectory)
        speed_kph = max(speed_mps, 0.0) * 3.6
        lookahead_dm = float(
            np.clip(
                self.speed_scale * speed_kph + self.speed_offset,
                self.default_lookahead,
                self.max_lookahead,
            )
        )
        lookahead_index = min(
            max(int(lookahead_dm), 0), max(len(route) - 1, 0)
        )
        aim_front_left = route[lookahead_index]
        heading_front_left = self._normalized_heading(aim_front_left)
        angle = self.steer_sign * heading_front_left
        steer = float(np.clip(self.turn_controller.step(angle), -1.0, 1.0))

        brake = bool(
            desired_speed < self.brake_speed
            or speed_mps / max(desired_speed, 1e-5) > self.brake_ratio
        )
        delta = float(np.clip(desired_speed - speed_mps, 0.0, self.clip_delta))
        throttle = float(
            np.clip(self.speed_controller.step(delta), 0.0, self.max_throttle)
        )
        if brake:
            throttle = 0.0

        target_array = np.asarray(target, dtype=np.float32).reshape(-1)
        target_angle = (
            self.steer_sign
            * self._normalized_heading(self._to_front_left(target_array[None])[0])
            if target_array.size >= 2 and np.isfinite(target_array[:2]).all()
            else 0.0
        )
        aim_native = self._from_front_left_point(aim_front_left)
        metadata = {
            "controller_mode": "automot",
            "route_target_used_for_steering": False,
            "speed": speed_mps,
            "steer": steer,
            "throttle": throttle,
            "brake": brake,
            "aim": tuple(aim_native.astype(np.float64)),
            "aim_front_left": tuple(aim_front_left.astype(np.float64)),
            "aim_distance": float(np.linalg.norm(aim_front_left)),
            "lookahead_index": int(lookahead_index),
            "interpolated_points": int(len(route)),
            "target": tuple(target_array[:2].astype(np.float64)),
            "desired_speed": desired_speed,
            "heading_front_left": heading_front_left,
            "angle": angle,
            "angle_target": target_angle,
            "angle_final": angle,
            "use_target_to_aim": False,
            "delta": delta,
        }
        for index in range(4):
            source_index = min(index, len(np.asarray(waypoints)) - 1)
            metadata[f"wp_{index + 1}"] = tuple(
                np.asarray(waypoints)[source_index, :2].astype(np.float64)
            )
        return steer, throttle, brake, metadata


def configure_b2d_controller(agent) -> str:
    """Install an opt-in FireDrive controller without changing other agents."""
    mode = os.environ.get("DIFFUSIONDRIVE_B2D_CONTROLLER", "legacy").strip().lower()
    if mode in ("legacy", "pad"):
        mode = "legacy"
    elif mode in ("automot", "trajectory_only"):
        mode = "automot"
        agent.pidcontroller = AutoMoTTrajectoryController(
            waypoint_frame=getattr(agent, "pid_frame", "pad")
        )
    else:
        raise ValueError(
            "DIFFUSIONDRIVE_B2D_CONTROLLER must be legacy or automot, "
            f"got {mode!r}"
        )
    agent.firedrive_controller_mode = mode
    print(f"FireDriveB2D controller: mode={mode}", flush=True)
    return mode
