import datetime
import json
import math
import os
import pathlib
import re
import sys
import time
from collections import deque
from typing import Optional, Tuple

import carla
import cv2
import numpy as np
import torch
from scipy.optimize import fsolve

from hydra.utils import instantiate
from omegaconf import OmegaConf

from leaderboard.autoagents import autonomous_agent
from team_code.pid_controller import PIDController
from team_code.planner import RoutePlanner


SAVE_PATH = os.environ.get("SAVE_PATH", None)
IS_BENCH2DRIVE = os.environ.get("IS_BENCH2DRIVE", None)
TEAM_AGENT = os.environ.get("TEAM_AGENT", "")


def get_entry_point():
    return "DiffusionDriveB2DAgent"


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return float(default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except Exception:
        return int(default)


def _sanitize_name(value: str) -> str:
    value = pathlib.Path(str(value)).name or "diffusiondrive"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _normalize_angle(angle: float) -> float:
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


def _command_value(command) -> int:
    if hasattr(command, "value"):
        command = command.value
    try:
        return int(command)
    except Exception:
        return 4


def _command_to_navsim_4d(command) -> np.ndarray:
    # CARLA/B2D route commands: 1 left, 2 right, 3 straight, 4 follow,
    # 5 change-left, 6 change-right. DiffusionDrive uses
    # [left, straight, right, follow].
    cmd = _command_value(command)
    out = np.zeros(4, dtype=np.float32)
    if cmd in (1, 5):
        out[0] = 1.0
    elif cmd == 3:
        out[1] = 1.0
    elif cmd in (2, 6):
        out[2] = 1.0
    else:
        out[3] = 1.0
    return out


def _resolve_config_path(config_path: Optional[str]) -> Optional[str]:
    if not config_path or str(config_path).lower() in {"none", "null", "-"}:
        return None
    path = pathlib.Path(config_path)
    if path.is_dir():
        candidates = (
            path / "code" / "hydra" / "config.yaml",
            path / ".hydra" / "config.yaml",
            path / "config.yaml",
        )
        for candidate in candidates:
            if candidate.is_file():
                return str(candidate)
    return str(path)


def _split_team_config(path_to_conf_file: str) -> Tuple[Optional[str], str]:
    parts = str(path_to_conf_file).split("+")
    if len(parts) >= 2:
        config_path, ckpt_path = parts[0], parts[1]
    else:
        config_path = os.environ.get("DIFFUSIONDRIVE_B2D_HYDRA_CONFIG")
        ckpt_path = parts[0]
    ckpt_path = ckpt_path or os.environ.get("DIFFUSIONDRIVE_B2D_CKPT", "")
    if not ckpt_path:
        raise ValueError(
            "Missing checkpoint. Use TEAM_CONFIG=<hydra_config.yaml>+<ckpt> "
            "or set DIFFUSIONDRIVE_B2D_CKPT."
        )
    return _resolve_config_path(config_path), ckpt_path


def _extract_leaderboard_save_name(path_to_conf_file: str, ckpt_path: str) -> str:
    parts = str(path_to_conf_file).split("+")
    if len(parts) >= 3 and parts[2]:
        return parts[2]
    return pathlib.Path(str(ckpt_path)).stem or "diffusiondrive"


class DiffusionDriveB2DAgent(autonomous_agent.AutonomousAgent):
    def setup(self, path_to_conf_file):
        self.track = autonomous_agent.Track.SENSORS
        self.pidcontroller = PIDController()
        self.step = -1
        self.wall_start = time.time()
        self.initialized = False
        self.metric_info = {}
        self.prev_control_cache = []
        self._pose_history = deque(maxlen=256)
        self._lidar_history = deque(maxlen=1)

        self.config_path, self.ckpt_path = _split_team_config(path_to_conf_file)
        self.leaderboard_save_name = _extract_leaderboard_save_name(
            path_to_conf_file, self.ckpt_path
        )
        self.save_name = _sanitize_name(self.leaderboard_save_name)

        self.device = torch.device(
            os.environ.get(
                "DIFFUSIONDRIVE_B2D_DEVICE",
                "cuda" if torch.cuda.is_available() else "cpu",
            )
        )
        self.scale = _env_float("DIFFUSIONDRIVE_B2D_CAMERA_SCALE", 0.64)
        self.stitch_mode = os.environ.get("DIFFUSIONDRIVE_B2D_STITCH_MODE", "b2d_mid")
        self.side_crop_ratio = _env_float("DIFFUSIONDRIVE_B2D_SIDE_CROP_RATIO", 0.30)
        self.side_overlap_ratio = _env_float("DIFFUSIONDRIVE_B2D_SIDE_OVERLAP_RATIO", 0.17)
        self.jpeg_quality = _env_int("DIFFUSIONDRIVE_B2D_JPEG_QUALITY", 20)
        self.lidar_mode = os.environ.get("DIFFUSIONDRIVE_B2D_LIDAR_MODE", "zeros")
        self.lidar_coordinate_mode = os.environ.get(
            "DIFFUSIONDRIVE_B2D_LIDAR_COORDINATE_MODE", "front_left"
        )
        self.lidar_z_offset = _env_float("DIFFUSIONDRIVE_B2D_LIDAR_Z_OFFSET", 2.0)
        self.lidar_min_z = _env_float("DIFFUSIONDRIVE_B2D_LIDAR_MIN_Z", 0.2)
        self.lidar_max_z = _env_float("DIFFUSIONDRIVE_B2D_LIDAR_MAX_Z", 100.0)
        self.lidar_accum_frames = max(1, _env_int("DIFFUSIONDRIVE_B2D_LIDAR_ACCUM_FRAMES", 1))
        self.lidar_rotation_frequency = _env_float("DIFFUSIONDRIVE_B2D_LIDAR_ROTATION_FREQUENCY", 20.0)
        self.lidar_points_per_second = _env_int("DIFFUSIONDRIVE_B2D_LIDAR_POINTS_PER_SECOND", 100000)
        self.lidar_sensor_tick = _env_float("DIFFUSIONDRIVE_B2D_LIDAR_SENSOR_TICK", 0.05)
        self._lidar_history = deque(maxlen=self.lidar_accum_frames)
        self._last_lidar_stats = {}
        self.history_frames = _env_int("DIFFUSIONDRIVE_B2D_HISTORY_FRAMES", 4)
        self.history_interval = _env_int("DIFFUSIONDRIVE_B2D_HISTORY_INTERVAL", 5)
        self.save_every = max(1, _env_int("DIFFUSIONDRIVE_B2D_SAVE_EVERY", 10))
        self.save_images = os.environ.get("DIFFUSIONDRIVE_B2D_SAVE_IMAGES", "0") == "1"
        self.save_lidar = os.environ.get("DIFFUSIONDRIVE_B2D_SAVE_LIDAR", "0") == "1"
        self.pid_frame = os.environ.get("DIFFUSIONDRIVE_B2D_PID_FRAME", "pad").lower()
        self.traj_scale = _env_float("DIFFUSIONDRIVE_B2D_TRAJ_SCALE", 1.0)
        self.traj_x_scale = _env_float("DIFFUSIONDRIVE_B2D_TRAJ_X_SCALE", 1.0)
        self.traj_y_scale = _env_float("DIFFUSIONDRIVE_B2D_TRAJ_Y_SCALE", 1.0)
        self.traj_forward_bias = _env_float("DIFFUSIONDRIVE_B2D_TRAJ_FORWARD_BIAS", 0.0)
        self.target_lateral_scale = _env_float(
            "DIFFUSIONDRIVE_B2D_TARGET_LATERAL_SCALE", 1.0
        )

        self.model = self._build_model(self.config_path, self.ckpt_path)
        self.model.to(self.device)
        self.model.eval()
        self.model.requires_grad_(False)

        self.lat_ref, self.lon_ref = 42.0, 2.0
        self.save_path = None
        if SAVE_PATH is not None:
            route_stem = pathlib.Path(os.environ.get("ROUTES", "route")).stem
            run_name = f"{route_stem}_{self.save_name}"
            self.save_path = pathlib.Path(SAVE_PATH) / run_name
            self.save_path.mkdir(parents=True, exist_ok=True)
            (self.save_path / "meta").mkdir(exist_ok=True)
            if self.save_images:
                (self.save_path / "rgb_stitched").mkdir(exist_ok=True)
            if self.save_lidar:
                (self.save_path / "lidar_bev").mkdir(exist_ok=True)
                (self.save_path / "lidar_feature").mkdir(exist_ok=True)

        control = carla.VehicleControl()
        control.steer = 0.0
        control.throttle = 0.0
        control.brake = 0.0
        self.prev_control = control

        print(
            "DiffusionDriveB2D setup: "
            f"config={self.config_path}, ckpt={self.ckpt_path}, device={self.device}, "
            f"stitch={self.stitch_mode}, lidar={self.lidar_mode}, "
            f"lidar_coord={self.lidar_coordinate_mode}, pid_frame={self.pid_frame}",
            flush=True,
        )

    def _build_model(self, config_path: Optional[str], ckpt_path: str):
        from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig
        from navsim.agents.diffusiondrive.transfuser_model_v2 import (
            V2TransfuserModel,
        )

        if config_path is not None and pathlib.Path(config_path).is_file():
            cfg = OmegaConf.load(config_path)
            if "agent" in cfg and "config" in cfg.agent:
                config = instantiate(cfg.agent.config)
            elif "config" in cfg:
                config = instantiate(cfg.config)
            else:
                config = instantiate(cfg)
        else:
            config = TransfuserConfig()
        if os.environ.get("DIFFUSIONDRIVE_B2D_BKB_PATH"):
            config.bkb_path = os.environ["DIFFUSIONDRIVE_B2D_BKB_PATH"]
        if os.environ.get("DIFFUSIONDRIVE_B2D_PLAN_ANCHOR_PATH"):
            config.plan_anchor_path = os.environ["DIFFUSIONDRIVE_B2D_PLAN_ANCHOR_PATH"]

        model = V2TransfuserModel(config)
        checkpoint = self._load_checkpoint(ckpt_path)
        raw_state = checkpoint.get("state_dict", checkpoint)
        model_state = model.state_dict()
        filtered_state = {}
        skipped = []
        for key, value in raw_state.items():
            model_key = key
            for prefix in (
                "agent._transfuser_model.",
                "_transfuser_model.",
                "model.",
            ):
                if model_key.startswith(prefix):
                    model_key = model_key[len(prefix) :]
                    break
            if model_key not in model_state:
                continue
            if tuple(model_state[model_key].shape) != tuple(value.shape):
                skipped.append(model_key)
                continue
            filtered_state[model_key] = value
        missing, unexpected = model.load_state_dict(filtered_state, strict=False)
        print(
            "DiffusionDriveB2D loaded model: "
            f"matched={len(filtered_state)}, missing={len(missing)}, "
            f"unexpected={len(unexpected)}, skipped_shape={len(skipped)}",
            flush=True,
        )
        if skipped[:10]:
            print(f"Skipped shape mismatch keys sample: {skipped[:10]}", flush=True)
        return model

    @staticmethod
    def _load_checkpoint(ckpt_path: str):
        map_location = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            return torch.load(ckpt_path, map_location=map_location)
        except TypeError:
            return torch.load(ckpt_path, map_location=map_location, weights_only=False)
        except Exception:
            return torch.load(ckpt_path, map_location=map_location, weights_only=False)

    def _init(self):
        try:
            locx = self._global_plan_world_coord[0][0].location.x
            locy = self._global_plan_world_coord[0][0].location.y
            lon = self._global_plan[0][0]["lon"]
            lat = self._global_plan[0][0]["lat"]
            earth_radius = 6378137.0

            def equations(vars_):
                x, y = vars_
                eq1 = lon * math.cos(x * math.pi / 180) - (locx * x * 180) / (
                    math.pi * earth_radius
                ) - math.cos(x * math.pi / 180) * y
                eq2 = (
                    math.log(math.tan((lat + 90) * math.pi / 360))
                    * earth_radius
                    * math.cos(x * math.pi / 180)
                    + locy
                    - math.cos(x * math.pi / 180)
                    * earth_radius
                    * math.log(math.tan((90 + x) * math.pi / 360))
                )
                return [eq1, eq2]

            solution = fsolve(equations, [0, 0])
            self.lat_ref, self.lon_ref = solution[0], solution[1]
        except Exception as exc:
            print(f"Failed to solve CARLA lat/lon reference: {exc}", flush=True)
            self.lat_ref, self.lon_ref = 0, 0

        self._route_planner = RoutePlanner(4.0, 50.0, lat_ref=self.lat_ref, lon_ref=self.lon_ref)
        self._route_planner.set_route(self._global_plan, True)
        self.initialized = True
        self.metric_info = {}

    def sensors(self):
        width = int(round(1600 * self.scale))
        height = int(round(900 * self.scale))
        sensors = [
            {
                "type": "sensor.camera.rgb",
                "x": 0.80,
                "y": 0.0,
                "z": 1.60,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "width": width,
                "height": height,
                "fov": 70,
                "id": "CAM_FRONT",
            },
            {
                "type": "sensor.camera.rgb",
                "x": 0.27,
                "y": -0.55,
                "z": 1.60,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": -55.0,
                "width": width,
                "height": height,
                "fov": 70,
                "id": "CAM_FRONT_LEFT",
            },
            {
                "type": "sensor.camera.rgb",
                "x": 0.27,
                "y": 0.55,
                "z": 1.60,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 55.0,
                "width": width,
                "height": height,
                "fov": 70,
                "id": "CAM_FRONT_RIGHT",
            },
            {
                "type": "sensor.other.imu",
                "x": -1.4,
                "y": 0.0,
                "z": 0.0,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "sensor_tick": 0.05,
                "id": "IMU",
            },
            {
                "type": "sensor.other.gnss",
                "x": -1.4,
                "y": 0.0,
                "z": 0.0,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "sensor_tick": 0.01,
                "id": "GPS",
            },
            {
                "type": "sensor.speedometer",
                "reading_frequency": 20,
                "id": "SPEED",
            },
        ]
        if self.lidar_mode == "histogram":
            sensors.append(
                {
                    "type": "sensor.lidar.ray_cast",
                    "x": 0.0,
                    "y": 0.0,
                    "z": 2.0,
                    "roll": 0.0,
                    "pitch": 0.0,
                    "yaw": 0.0,
                    "range": 64,
                    "rotation_frequency": self.lidar_rotation_frequency,
                    "points_per_second": self.lidar_points_per_second,
                    "sensor_tick": self.lidar_sensor_tick,
                    "id": "LIDAR_TOP",
                }
            )
        return sensors

    def _read_rgb(self, input_data, cam: str) -> np.ndarray:
        img = cv2.cvtColor(input_data[cam][1][:, :, :3], cv2.COLOR_BGR2RGB)
        if self.jpeg_quality > 0:
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(self.jpeg_quality)]
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            _, encoded = cv2.imencode(".jpg", bgr, encode_param)
            bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return img

    def tick(self, input_data):
        self.step += 1
        imgs = {
            cam: self._read_rgb(input_data, cam)
            for cam in ("CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT")
        }
        gps = input_data["GPS"][1][:2]
        speed = input_data["SPEED"][1]["speed"]
        compass = input_data["IMU"][1][-1]
        acceleration = input_data["IMU"][1][:3]
        angular_velocity = input_data["IMU"][1][3:6]

        if math.isnan(compass):
            compass = 0.0
            acceleration = np.zeros(3)
            angular_velocity = np.zeros(3)

        pos = self.gps_to_location(gps)
        near_node, near_command = self._route_planner.run_step(pos)
        return {
            "imgs": imgs,
            "gps": gps,
            "pos": pos,
            "speed": speed,
            "compass": compass,
            "acceleration": acceleration,
            "angular_velocity": angular_velocity,
            "command_near": near_command,
            "command_near_xy": near_node,
        }

    @staticmethod
    def _crop_front_stitch_patch(
        image: np.ndarray,
        keep_width: Optional[float] = None,
        horizontal_anchor: str = "center",
        start_x: Optional[int] = None,
    ) -> np.ndarray:
        height, width = image.shape[:2]
        crop_y = int(round(height * (28.0 / 256.0)))
        if crop_y > 0 and height > 2 * crop_y:
            image = image[crop_y:-crop_y]
        if keep_width is None:
            return image
        keep_w = int(round(width * keep_width))
        keep_w = max(1, min(width, keep_w))
        if start_x is not None:
            x0 = int(start_x)
        elif horizontal_anchor == "left":
            x0 = 0
        elif horizontal_anchor == "right":
            x0 = max(0, width - keep_w)
        else:
            x0 = max(0, (width - keep_w) // 2)
        x0 = max(0, min(x0, width - keep_w))
        return image[:, x0 : x0 + keep_w]

    def _build_camera_feature(self, imgs) -> torch.Tensor:
        if self.stitch_mode == "front_only":
            stitched = self._crop_front_stitch_patch(imgs["CAM_FRONT"], keep_width=None)
        else:
            front = self._crop_front_stitch_patch(imgs["CAM_FRONT"], keep_width=None)
            left_img = imgs["CAM_FRONT_LEFT"]
            right_img = imgs["CAM_FRONT_RIGHT"]
            if self.stitch_mode == "navsim_center":
                left = self._crop_front_stitch_patch(
                    left_img, keep_width=self.side_crop_ratio, horizontal_anchor="center"
                )
                right = self._crop_front_stitch_patch(
                    right_img, keep_width=self.side_crop_ratio, horizontal_anchor="center"
                )
            elif self.stitch_mode == "b2d_inner":
                left = self._crop_front_stitch_patch(
                    left_img, keep_width=self.side_crop_ratio, horizontal_anchor="right"
                )
                right = self._crop_front_stitch_patch(
                    right_img, keep_width=self.side_crop_ratio, horizontal_anchor="left"
                )
            elif self.stitch_mode == "b2d_outer":
                left = self._crop_front_stitch_patch(
                    left_img, keep_width=self.side_crop_ratio, horizontal_anchor="left"
                )
                right = self._crop_front_stitch_patch(
                    right_img, keep_width=self.side_crop_ratio, horizontal_anchor="right"
                )
            elif self.stitch_mode == "b2d_mid":
                left_w = left_img.shape[1]
                right_w = right_img.shape[1]
                left_keep = max(1, int(round(left_w * self.side_crop_ratio)))
                right_keep = max(1, int(round(right_w * self.side_crop_ratio)))
                left_overlap = max(0, int(round(left_w * self.side_overlap_ratio)))
                right_overlap = max(0, int(round(right_w * self.side_overlap_ratio)))
                left = self._crop_front_stitch_patch(
                    left_img,
                    keep_width=self.side_crop_ratio,
                    start_x=left_w - left_overlap - left_keep,
                )
                right = self._crop_front_stitch_patch(
                    right_img,
                    keep_width=self.side_crop_ratio,
                    start_x=right_overlap,
                )
            else:
                raise ValueError(f"Unsupported stitch mode: {self.stitch_mode}")
            stitched = np.concatenate([left, front, right], axis=1)
        stitched = cv2.resize(stitched, (1024, 256), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(stitched.astype(np.float32) / 255.0)
        return tensor.permute(2, 0, 1).contiguous()

    def _build_lidar_feature(self, input_data) -> torch.Tensor:
        if self.lidar_mode != "histogram" or "LIDAR_TOP" not in input_data:
            self._last_lidar_stats = {"mode": self.lidar_mode, "raw_points": 0}
            return torch.zeros((1, 256, 256), dtype=torch.float32)
        points = np.asarray(input_data["LIDAR_TOP"][1], dtype=np.float32)
        if points.ndim != 2 or points.shape[1] < 3:
            self._last_lidar_stats = {"mode": self.lidar_mode, "raw_points": 0}
            return torch.zeros((1, 256, 256), dtype=torch.float32)
        points = points[:, :3]
        points = points[np.isfinite(points).all(axis=1)]
        if points.size == 0:
            self._last_lidar_stats = {"mode": self.lidar_mode, "raw_points": 0}
            return torch.zeros((1, 256, 256), dtype=torch.float32)
        raw_points = int(points.shape[0])
        self._lidar_history.append(points.copy())
        if self.lidar_accum_frames > 1 and len(self._lidar_history) > 1:
            points = np.concatenate(list(self._lidar_history), axis=0)
        if self.lidar_coordinate_mode == "front_left":
            xy = np.stack([points[:, 0], -points[:, 1]], axis=1).astype(np.float32)
        elif self.lidar_coordinate_mode == "b2d_lidar":
            xy = np.stack([points[:, 1], -points[:, 0]], axis=1).astype(np.float32)
        elif self.lidar_coordinate_mode == "identity":
            xy = points[:, :2].astype(np.float32, copy=True)
        else:
            raise ValueError(f"Unsupported lidar coordinate mode: {self.lidar_coordinate_mode}")
        # CARLA lidar points are in the sensor frame. Shift z by the sensor
        # mounting height so the NAVSIM/TransFuser split-height filter is
        # applied in an ego-ground-like frame.
        z = points[:, 2:3] + self.lidar_z_offset
        point_cloud = np.concatenate([xy, z], axis=1)
        point_cloud = point_cloud[point_cloud[:, 2] < self.lidar_max_z]
        point_cloud = point_cloud[point_cloud[:, 2] > self.lidar_min_z]
        if point_cloud.size == 0:
            self._last_lidar_stats = {
                "mode": self.lidar_mode,
                "raw_points": raw_points,
                "accum_points": int(points.shape[0]),
                "filtered_points": 0,
                "accum_frames": int(len(self._lidar_history)),
            }
            return torch.zeros((1, 256, 256), dtype=torch.float32)
        self._last_lidar_stats = {
            "mode": self.lidar_mode,
            "raw_points": raw_points,
            "accum_points": int(points.shape[0]),
            "filtered_points": int(point_cloud.shape[0]),
            "accum_frames": int(len(self._lidar_history)),
            "front_min": float(point_cloud[:, 0].min()),
            "front_max": float(point_cloud[:, 0].max()),
            "left_min": float(point_cloud[:, 1].min()),
            "left_max": float(point_cloud[:, 1].max()),
            "z_min": float(point_cloud[:, 2].min()),
            "z_max": float(point_cloud[:, 2].max()),
        }
        bins = np.linspace(-32.0, 32.0, 257)
        hist = np.histogramdd(point_cloud[:, :2], bins=(bins, bins))[0].astype(np.float32)
        hist = np.clip(hist, 0.0, 5.0) / 5.0
        return torch.from_numpy(hist[None]).contiguous()

    def _build_status_feature(self, tick_data) -> torch.Tensor:
        command = _command_to_navsim_4d(tick_data["command_near"])
        speed = float(tick_data["speed"])
        acceleration = np.asarray(tick_data["acceleration"], dtype=np.float32)
        status = np.concatenate(
            [
                command,
                np.array([speed, 0.0], dtype=np.float32),
                np.array([acceleration[0], -acceleration[1]], dtype=np.float32),
            ],
            axis=0,
        )
        status = np.nan_to_num(status, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.from_numpy(status.astype(np.float32))

    def _current_pose_and_command(self, tick_data):
        raw_theta = tick_data["compass"] if not np.isnan(tick_data["compass"]) else 0.0
        ego_theta = -raw_theta + np.pi / 2
        can_bus_xy = np.array([tick_data["pos"][0], -tick_data["pos"][1]], dtype=np.float32)
        command_near_xy = np.array(
            [
                tick_data["command_near_xy"][0] - can_bus_xy[0],
                -tick_data["command_near_xy"][1] - can_bus_xy[1],
            ],
            dtype=np.float32,
        )
        rotation_matrix = np.array(
            [[np.cos(raw_theta), -np.sin(raw_theta)], [np.sin(raw_theta), np.cos(raw_theta)]],
            dtype=np.float32,
        )
        local_command_xy = rotation_matrix @ command_near_xy
        if not np.isfinite(local_command_xy).all() or np.linalg.norm(local_command_xy) < 1e-3:
            local_command_xy = np.array([1.0, 0.0], dtype=np.float32)
        return can_bus_xy, ego_theta, local_command_xy

    def _build_ego_history(self, cur_xy: np.ndarray, cur_theta: float) -> torch.Tensor:
        if len(self._pose_history) == 0:
            return torch.zeros((self.history_frames, 3), dtype=torch.float32)
        history = list(self._pose_history)
        poses = []
        rot = np.array(
            [[math.cos(cur_theta), -math.sin(cur_theta)], [math.sin(cur_theta), math.cos(cur_theta)]],
            dtype=np.float32,
        )
        world_to_ego = rot.T
        for offset in reversed(range(self.history_frames)):
            target_step = self.step - offset * self.history_interval
            candidates = [item for item in history if item[0] <= target_step]
            if len(candidates) == 0:
                _, past_xy, past_theta = history[0]
            else:
                _, past_xy, past_theta = candidates[-1]
            local_xy = world_to_ego @ (past_xy - cur_xy)
            local_heading = _normalize_angle(past_theta - cur_theta)
            poses.append(np.array([local_xy[0], local_xy[1], local_heading], dtype=np.float32))
        poses[-1] = np.zeros(3, dtype=np.float32)
        return torch.from_numpy(np.stack(poses, axis=0).astype(np.float32))

    def _build_features(self, input_data, tick_data, cur_xy: np.ndarray, cur_theta: float):
        camera_feature = self._build_camera_feature(tick_data["imgs"])
        lidar_feature = self._build_lidar_feature(input_data)
        status_feature = self._build_status_feature(tick_data)
        ego_history = self._build_ego_history(cur_xy, cur_theta)
        self._last_lidar_feature = lidar_feature.detach().cpu().clone()
        return {
            "camera_feature": camera_feature.unsqueeze(0).to(self.device),
            "lidar_feature": lidar_feature.unsqueeze(0).to(self.device),
            "status_feature": status_feature.unsqueeze(0).to(self.device),
            "ego_history": ego_history.unsqueeze(0).to(self.device),
        }

    def _model_plan_to_pid_waypoints(self, model_xy: np.ndarray) -> np.ndarray:
        dd_xy = np.asarray(model_xy, dtype=np.float32).copy()
        dd_xy[:, 0] = dd_xy[:, 0] * self.traj_scale * self.traj_x_scale + self.traj_forward_bias
        dd_xy[:, 1] = dd_xy[:, 1] * self.traj_scale * self.traj_y_scale
        if self.pid_frame in ("pad", "b2d", "right_front"):
            # DiffusionDrive predicts (front, left). PAD's PID expects
            # (right, front), so invert the cache-time conversion.
            return np.stack([-dd_xy[:, 1], dd_xy[:, 0]], axis=-1).astype(np.float32)
        if self.pid_frame in ("raw", "diffusiondrive", "front_left"):
            return dd_xy.astype(np.float32)
        raise ValueError(f"Unsupported PID frame: {self.pid_frame}")

    def _adjust_pid_target(self, local_command_xy: np.ndarray) -> np.ndarray:
        target = np.asarray(local_command_xy, dtype=np.float32).copy()
        if self.pid_frame in ("pad", "b2d", "right_front"):
            target[0] *= self.target_lateral_scale
        elif self.pid_frame in ("raw", "diffusiondrive", "front_left"):
            target[1] *= self.target_lateral_scale
        return target

    @torch.no_grad()
    def run_step(self, input_data, timestamp):
        if not self.initialized:
            self._init()
        tick_data = self.tick(input_data)
        cur_xy, cur_theta, local_command_xy = self._current_pose_and_command(tick_data)
        raw_local_command_xy = local_command_xy.copy()
        local_command_xy = self._adjust_pid_target(local_command_xy)
        self._pose_history.append((self.step, cur_xy.copy(), float(cur_theta)))

        features = self._build_features(input_data, tick_data, cur_xy, cur_theta)
        output = self.model(features)
        out_traj = output["trajectory"].detach().float().cpu().numpy()[0]
        model_plan_dd = out_traj[:, :2].astype(np.float32)
        out_truck = self._model_plan_to_pid_waypoints(model_plan_dd)
        if out_truck.shape[0] < 4:
            pad = np.repeat(out_truck[-1:], 4 - out_truck.shape[0], axis=0)
            out_truck = np.concatenate([out_truck, pad], axis=0)
        if not np.isfinite(out_truck).all():
            model_plan_dd = np.array(
                [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]], dtype=np.float32
            )
            out_truck = self._model_plan_to_pid_waypoints(model_plan_dd)

        steer_traj, throttle_traj, brake_traj, metadata_traj = self.pidcontroller.control_pid(
            out_truck, np.array(tick_data["speed"], dtype=np.float32), local_command_xy
        )
        if brake_traj < 0.05:
            brake_traj = 0.0
        if throttle_traj > brake_traj:
            brake_traj = 0.0

        control = carla.VehicleControl()
        control.steer = np.clip(float(steer_traj), -1.0, 1.0)
        control.throttle = np.clip(float(throttle_traj), 0.0, 0.75)
        control.brake = np.clip(float(brake_traj), 0.0, 1.0)

        self.pid_metadata = metadata_traj
        self.pid_metadata["agent"] = "diffusiondrive"
        self.pid_metadata["steer"] = control.steer
        self.pid_metadata["throttle"] = control.throttle
        self.pid_metadata["brake"] = control.brake
        self.pid_metadata["plan"] = out_truck.tolist()
        self.pid_metadata["model_plan_diffusiondrive"] = model_plan_dd.tolist()
        self.pid_metadata["pid_frame"] = self.pid_frame
        self.pid_metadata["traj_scale"] = self.traj_scale
        self.pid_metadata["traj_x_scale"] = self.traj_x_scale
        self.pid_metadata["traj_y_scale"] = self.traj_y_scale
        self.pid_metadata["traj_forward_bias"] = self.traj_forward_bias
        self.pid_metadata["target_lateral_scale"] = self.target_lateral_scale
        self.pid_metadata["lidar_coordinate_mode"] = self.lidar_coordinate_mode
        self.pid_metadata["lidar_stats"] = self._last_lidar_stats
        self.pid_metadata["leaderboard_save_name"] = self.leaderboard_save_name
        self.pid_metadata["command"] = _command_value(tick_data["command_near"])
        self.pid_metadata["step"] = int(self.step)
        self.pid_metadata["timestamp"] = float(timestamp)
        self.pid_metadata["ego_xy_navsim"] = [float(cur_xy[0]), float(cur_xy[1])]
        self.pid_metadata["ego_theta_navsim"] = float(cur_theta)
        self.pid_metadata["ego_carla_xy"] = [
            float(tick_data["pos"][0]),
            float(tick_data["pos"][1]),
        ]
        self.pid_metadata["compass"] = float(tick_data["compass"])
        self.pid_metadata["speed_mps"] = float(tick_data["speed"])
        self.pid_metadata["local_command_xy"] = [
            float(local_command_xy[0]),
            float(local_command_xy[1]),
        ]
        self.pid_metadata["raw_local_command_xy"] = [
            float(raw_local_command_xy[0]),
            float(raw_local_command_xy[1]),
        ]
        self.pid_metadata["command_near_xy_carla"] = [
            float(tick_data["command_near_xy"][0]),
            float(tick_data["command_near_xy"][1]),
        ]
        if output.get("pdm_score") is not None:
            score = output["pdm_score"].detach().float().cpu().numpy()[0]
            self.pid_metadata["pdm_score"] = score.tolist()

        self.metric_info[self.step] = self.get_metric_info()
        if self.save_path is not None and self.step % self.save_every == 0:
            self.save(tick_data, out_truck)

        self.prev_control = control
        if len(self.prev_control_cache) == 10:
            self.prev_control_cache.pop(0)
        self.prev_control_cache.append(control)
        return control

    def save(self, tick_data, ego_traj):
        frame = self.step
        with open(self.save_path / "meta" / (f"{frame:06d}.json"), "w") as outfile:
            json.dump(self.pid_metadata, outfile, indent=4)
        with open(self.save_path / "metric_info.json", "w") as outfile:
            json.dump(self.metric_info, outfile, indent=4)
        if self.save_images:
            stitched = self._build_camera_feature(tick_data["imgs"]).permute(1, 2, 0).numpy()
            stitched = np.clip(stitched * 255.0, 0, 255).astype(np.uint8)
            cv2.imwrite(
                str(self.save_path / "rgb_stitched" / (f"{frame:06d}.jpg")),
                cv2.cvtColor(stitched, cv2.COLOR_RGB2BGR),
            )
        if self.save_lidar and hasattr(self, "_last_lidar_feature"):
            lidar = self._last_lidar_feature.numpy().squeeze().astype(np.float32)
            np.save(self.save_path / "lidar_feature" / (f"{frame:06d}.npy"), lidar)
            lidar_img = np.clip(lidar, 0.0, 1.0)
            lidar_img = (lidar_img * 255.0).astype(np.uint8)
            cv2.imwrite(
                str(self.save_path / "lidar_bev" / (f"{frame:06d}.png")),
                lidar_img,
            )

    def destroy(self):
        if hasattr(self, "model"):
            del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def gps_to_location(self, gps):
        earth_radius = 6378137.0
        lat, lon = gps
        scale = math.cos(self.lat_ref * math.pi / 180.0)
        my = math.log(math.tan((lat + 90) * math.pi / 360.0)) * (earth_radius * scale)
        mx = (lon * (math.pi * earth_radius * scale)) / 180.0
        y = scale * earth_radius * math.log(math.tan((90.0 + self.lat_ref) * math.pi / 360.0)) - my
        x = mx - scale * self.lon_ref * math.pi * earth_radius / 180.0
        return np.array([x, y])
