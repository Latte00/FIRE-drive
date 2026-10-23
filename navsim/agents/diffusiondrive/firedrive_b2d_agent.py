"""Navigation-enabled FireDrive entry point for Bench2Drive."""
import os
import math
import numpy as np
import torch
from leaderboard.utils.route_manipulation import downsample_route
from team_code.planner import RoutePlanner

from hydra.utils import instantiate
from omegaconf import OmegaConf
from navsim.agents.diffusiondrive.b2d.motion_agent import FireDriveB2DAgent


def get_entry_point():
    return "FireDriveNavigationB2DAgent"


class FireDriveNavigationB2DAgent(FireDriveB2DAgent):
    def setup(self, path_to_conf_file):
        FireDriveB2DAgent.setup(self, path_to_conf_file)
        print(
            'FireDriveNavigationB2D perception: '
            f'camera_scale={self.scale}, jpeg_quality={self.jpeg_quality}, '
            f'lidar_z_offset={self.lidar_z_offset}, '
            f'lidar_min_z={self.lidar_min_z}, lidar_max_z={self.lidar_max_z}, '
            f'lidar_rotation_frequency={self.lidar_rotation_frequency}, '
            f'lidar_points_per_second={self.lidar_points_per_second}, '
            f'lidar_sensor_tick={self.lidar_sensor_tick}', flush=True,
        )

    def _build_model(self, config_path, ckpt_path):
        from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig
        from navsim.agents.diffusiondrive.transfuser_model_v2 import V2TransfuserModel

        cfg = OmegaConf.load(config_path)
        node = cfg.agent.config if "agent" in cfg else cfg.config
        config = instantiate(node)
        if not isinstance(config, TransfuserConfig) or not config.b2d_navigation_enable:
            raise ValueError("Use a FireDrive config with b2d_navigation_enable=true")
        for env, field in (
            ("DIFFUSIONDRIVE_B2D_BKB_PATH", "bkb_path"),
            ("DIFFUSIONDRIVE_B2D_PLAN_ANCHOR_PATH", "plan_anchor_path"),
        ):
            if os.environ.get(env):
                setattr(config, field, os.environ[env])
        model = V2TransfuserModel(config)
        checkpoint = self._load_checkpoint(ckpt_path)
        raw = checkpoint.get("state_dict", checkpoint)
        weights = {}
        for key, value in raw.items():
            for prefix in ("agent._transfuser_model.", "_transfuser_model.", "model."):
                if key.startswith(prefix):
                    key = key[len(prefix):]
                    break
            if key in weights:
                raise ValueError(f"Duplicate model checkpoint key: {key}")
            weights[key] = value
        # Missing navigation weights must never silently run randomly initialized.
        model.load_state_dict(weights, strict=True)
        print(f"FireDriveNavigationB2D loaded all {len(weights)} tensors (strict)", flush=True)
        return model

    def set_global_plan(self, global_plan_gps, global_plan_world_coord):
        """Keep independent near and far routes matching B2D training labels."""
        near_ids = downsample_route(global_plan_world_coord, 2)
        far_ids = downsample_route(global_plan_world_coord, 50)
        self._firedrive_global_plan_near = [global_plan_gps[idx] for idx in near_ids]
        self._firedrive_global_plan_far = [global_plan_gps[idx] for idx in far_ids]
        return super().set_global_plan(global_plan_gps, global_plan_world_coord)

    def _init(self):
        super()._init()
        near_plan = getattr(self, "_firedrive_global_plan_near", None)
        far_plan = getattr(self, "_firedrive_global_plan_far", None)
        if not near_plan or not far_plan:
            raise RuntimeError("FireDrive near/far global plans were not initialized")

        # Keep the parent's 50 m planner untouched because its command point is
        # also consumed by the legacy PID controller. Navigation conditioning
        # uses independent planners and must not change the control baseline.
        self._navigation_route_planner_near = RoutePlanner(
            4.0, 50.0, lat_ref=self.lat_ref, lon_ref=self.lon_ref
        )
        self._navigation_route_planner_near.set_route(near_plan, True)
        self._navigation_route_planner_far = RoutePlanner(
            4.0, 50.0, lat_ref=self.lat_ref, lon_ref=self.lon_ref
        )
        self._navigation_route_planner_far.set_route(far_plan, True)
        print(
            "FireDriveB2D route planners: "
            f"near_points={len(near_plan)}, far_points={len(far_plan)}",
            flush=True,
        )

    def tick(self, input_data):
        tick_data = super().tick(input_data)
        navigation_near, _ = self._navigation_route_planner_near.run_step(
            tick_data["pos"]
        )
        navigation_far, _ = self._navigation_route_planner_far.run_step(
            tick_data["pos"]
        )
        tick_data["navigation_near_xy"] = navigation_near
        tick_data["navigation_far_xy"] = navigation_far
        return tick_data

    @staticmethod
    def _route_point_to_local(point, cur_xy, raw_theta):
        """Convert a CARLA route point to DiffusionDrive (front, left) XY."""
        point_xy = np.asarray([point[0], -point[1]], dtype=np.float32)
        delta = point_xy - cur_xy
        rotation = np.asarray(
            [
                [math.cos(raw_theta), -math.sin(raw_theta)],
                [math.sin(raw_theta), math.cos(raw_theta)],
            ],
            dtype=np.float32,
        )
        # The CARLA/PAD local frame produced here is (right, front), while
        # cached navigation targets and FireDrive trajectories use
        # DiffusionDrive's (front, left) convention.
        right_front = rotation @ delta
        return np.asarray([right_front[1], -right_front[0]], dtype=np.float32)

    def _build_features(self, input_data, tick_data, cur_xy, cur_theta):
        features = super()._build_features(
            input_data, tick_data, cur_xy, cur_theta
        )
        raw_theta = float(tick_data["compass"])
        near = self._route_point_to_local(
            tick_data["navigation_near_xy"], cur_xy, raw_theta
        )
        far = self._route_point_to_local(
            tick_data["navigation_far_xy"], cur_xy, raw_theta
        )
        near_valid = bool(np.isfinite(near).all())
        far_valid = bool(np.isfinite(far).all())
        if not near_valid:
            near = np.zeros(2, dtype=np.float32)
        if not far_valid:
            far = np.zeros(2, dtype=np.float32)

        features.update(
            {
                "tp_near": torch.from_numpy(near).unsqueeze(0).to(self.device),
                "tp_far": torch.from_numpy(far).unsqueeze(0).to(self.device),
                "tp_near_valid": torch.tensor(
                    [near_valid], dtype=torch.bool, device=self.device
                ),
                "tp_far_valid": torch.tensor(
                    [far_valid], dtype=torch.bool, device=self.device
                ),
            }
        )
        if self.step == 0:
            pid_route_target = self._route_point_to_local(
                tick_data["command_near_xy"], cur_xy, raw_theta
            )
            print(
                "FireDriveB2D navigation sample (front,left): "
                f"near={near.tolist()}, far={far.tolist()}, "
                f"pid_route_target={pid_route_target.tolist()}, "
                f"near_valid={near_valid}, far_valid={far_valid}",
                flush=True,
            )
        return features
