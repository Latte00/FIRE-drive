"""Bench2Drive adapter for a fair FIREDrive/DiffusionDrive evaluation."""

from collections import deque
import math
import os

import numpy as np

from navsim.agents.diffusiondrive.b2d.base_agent import DiffusionDriveB2DAgent

from navsim.agents.diffusiondrive.b2d.controller import (
    configure_b2d_controller,
)


def get_entry_point():
    return "FireDriveB2DAgent"


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().lower() in {"1", "true", "yes", "on"}


class FireDriveB2DAgent(DiffusionDriveB2DAgent):
    """Run the original FIREDrive checkpoint with a matched online interface.

    Bench2Drive emits half of a 10 Hz LiDAR revolution at each 20 Hz simulator
    tick. This adapter aligns the preceding half-scan to the current ego pose
    before rasterizing both halves as one DiffusionDrive LiDAR feature.
    """

    def setup(self, path_to_conf_file):
        crop_is_explicit = "DIFFUSIONDRIVE_B2D_SIDE_CROP_RATIO" in os.environ
        overlap_is_explicit = (
            "DIFFUSIONDRIVE_B2D_SIDE_OVERLAP_RATIO" in os.environ
        )
        if crop_is_explicit != overlap_is_explicit:
            raise ValueError(
                "Set DIFFUSIONDRIVE_B2D_SIDE_CROP_RATIO and "
                "DIFFUSIONDRIVE_B2D_SIDE_OVERLAP_RATIO together"
            )
        camera_contract = "explicit" if crop_is_explicit else "base_default"

        os.environ.setdefault("DIFFUSIONDRIVE_B2D_STITCH_MODE", "b2d_mid")
        os.environ.setdefault("DIFFUSIONDRIVE_B2D_LIDAR_MODE", "histogram")
        os.environ.setdefault(
            "DIFFUSIONDRIVE_B2D_LIDAR_COORDINATE_MODE", "front_left"
        )
        os.environ.setdefault("DIFFUSIONDRIVE_B2D_LIDAR_ACCUM_FRAMES", "2")
        os.environ.setdefault("DIFFUSIONDRIVE_B2D_CONTROLLER", "automot")

        super().setup(path_to_conf_file)

        if self.stitch_mode != "b2d_mid":
            raise ValueError(
                "FireDriveB2DAgent requires "
                "DIFFUSIONDRIVE_B2D_STITCH_MODE=b2d_mid"
            )
        if self.lidar_mode != "histogram":
            raise ValueError(
                "FireDriveB2DAgent requires "
                "DIFFUSIONDRIVE_B2D_LIDAR_MODE=histogram"
            )
        if self.lidar_coordinate_mode != "front_left":
            raise ValueError(
                "FireDriveB2DAgent requires "
                "DIFFUSIONDRIVE_B2D_LIDAR_COORDINATE_MODE=front_left"
            )
        if self.lidar_accum_frames != 2:
            raise ValueError(
                "FireDriveB2DAgent requires "
                "DIFFUSIONDRIVE_B2D_LIDAR_ACCUM_FRAMES=2"
            )

        self._firedrive_lidar_motion_compensation = _env_flag(
            "DIFFUSIONDRIVE_B2D_LIDAR_MOTION_COMPENSATION", True
        )
        self._firedrive_lidar_history = deque(maxlen=self.lidar_accum_frames)
        self._firedrive_current_pose = None
        self._firedrive_passthrough_history = deque(maxlen=1)
        controller_mode = configure_b2d_controller(self)
        print(
            "FireDriveB2D interface: "
            f"stitch={self.stitch_mode}, "
            f"side_crop={self.side_crop_ratio}, "
            f"side_overlap={self.side_overlap_ratio}, "
            f"camera_contract={camera_contract}, "
            f"lidar_frames={self.lidar_accum_frames}, "
            "lidar_motion_compensation="
            f"{self._firedrive_lidar_motion_compensation}, "
            f"controller={controller_mode}",
            flush=True,
        )

    @staticmethod
    def _rotation(theta: float) -> np.ndarray:
        return np.asarray(
            [
                [math.cos(theta), -math.sin(theta)],
                [math.sin(theta), math.cos(theta)],
            ],
            dtype=np.float32,
        )

    @classmethod
    def _align_raw_lidar_to_current_pose(
        cls,
        points: np.ndarray,
        source_xy: np.ndarray,
        source_theta: float,
        current_xy: np.ndarray,
        current_theta: float,
    ) -> np.ndarray:
        """Transform CARLA ``(front, right, z)`` points between ego poses."""
        aligned = np.asarray(points, dtype=np.float32).copy()
        source_front_left = np.stack(
            [aligned[:, 0], -aligned[:, 1]], axis=1
        )
        world_xy = (
            source_front_left @ cls._rotation(source_theta).T
            + np.asarray(source_xy, dtype=np.float32)[None]
        )
        current_front_left = (
            world_xy - np.asarray(current_xy, dtype=np.float32)[None]
        ) @ cls._rotation(current_theta)
        aligned[:, 0] = current_front_left[:, 0]
        aligned[:, 1] = -current_front_left[:, 1]
        return aligned

    def _build_features(self, input_data, tick_data, cur_xy, cur_theta):
        self._firedrive_current_pose = (
            np.asarray(cur_xy, dtype=np.float32).copy(),
            float(cur_theta),
        )
        return super()._build_features(
            input_data, tick_data, cur_xy, cur_theta
        )

    def _build_lidar_feature(self, input_data):
        payload = input_data.get("LIDAR_TOP")
        if (
            self.lidar_mode != "histogram"
            or payload is None
            or self._firedrive_current_pose is None
        ):
            return super()._build_lidar_feature(input_data)

        raw_points = np.asarray(payload[1], dtype=np.float32)
        if raw_points.ndim != 2 or raw_points.shape[1] < 3:
            return super()._build_lidar_feature(input_data)
        raw_points = raw_points[:, :3]
        raw_points = raw_points[np.isfinite(raw_points).all(axis=1)]
        if raw_points.size == 0:
            return super()._build_lidar_feature(input_data)

        current_xy, current_theta = self._firedrive_current_pose
        self._firedrive_lidar_history.append(
            (
                raw_points.copy(),
                current_xy.copy(),
                float(current_theta),
            )
        )

        scans = []
        entries = list(self._firedrive_lidar_history)
        alignment_translation_m = 0.0
        alignment_yaw_rad = 0.0
        for scan_index, (points, source_xy, source_theta) in enumerate(
            reversed(entries)
        ):
            is_current = scan_index == 0
            if (
                self._firedrive_lidar_motion_compensation
                and not is_current
            ):
                points = self._align_raw_lidar_to_current_pose(
                    points,
                    source_xy,
                    source_theta,
                    current_xy,
                    current_theta,
                )
                alignment_translation_m = float(
                    np.linalg.norm(current_xy - source_xy)
                )
                alignment_yaw_rad = float(
                    (current_theta - source_theta + math.pi)
                    % (2.0 * math.pi)
                    - math.pi
                )
            scans.append(points)
        combined = np.concatenate(scans, axis=0)

        adjusted_input = dict(input_data)
        adjusted_input["LIDAR_TOP"] = (payload[0], combined)

        # The scan has already been accumulated and aligned above. Temporarily
        # bypass the parent's unaligned history concatenation for rasterization.
        parent_history = self._lidar_history
        parent_accum_frames = self.lidar_accum_frames
        self._lidar_history = self._firedrive_passthrough_history
        self.lidar_accum_frames = 1
        self._firedrive_passthrough_history.clear()
        try:
            feature = super()._build_lidar_feature(adjusted_input)
        finally:
            self._lidar_history = parent_history
            self.lidar_accum_frames = parent_accum_frames

        self._last_lidar_stats.update(
            {
                "raw_points": int(raw_points.shape[0]),
                "accum_points": int(combined.shape[0]),
                "accum_frames": int(len(entries)),
                "motion_compensated": bool(
                    self._firedrive_lidar_motion_compensation
                    and len(entries) > 1
                ),
                "alignment_translation_m": alignment_translation_m,
                "alignment_yaw_rad": alignment_yaw_rad,
            }
        )
        return feature
