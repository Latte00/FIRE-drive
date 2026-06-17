from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
import numpy.typing as npt

import torch
import torch.nn.functional as F
from torchvision import transforms

from shapely import affinity
from shapely.geometry import Polygon, LineString, Point

from nuplan.common.maps.abstract_map import AbstractMap, SemanticMapLayer, MapObject
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType

from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig
from navsim.common.dataclasses import AgentInput, Scene, Annotations, NAVSIM_INTERVAL_LENGTH
from navsim.common.enums import BoundingBoxIndex, LidarIndex
from navsim.planning.scenario_builder.navsim_scenario_utils import (
    normalize_angle,
    tracked_object_types,
)
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder


class TransfuserFeatureBuilder(AbstractFeatureBuilder):
    """Input feature builder for TransFuser."""

    def __init__(self, config: TransfuserConfig):
        """
        Initializes feature builder.
        :param config: global config dataclass of TransFuser
        """
        self._config = config

    def get_unique_name(self) -> str:
        """Inherited, see superclass."""
        return "transfuser_feature"

    def compute_features(self, agent_input: AgentInput) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""
        features = {}

        features["camera_feature"] = self._get_camera_feature(agent_input)
        features["lidar_feature"] = self._get_lidar_feature(agent_input)
        features["status_feature"] = torch.concatenate(
            [
                torch.tensor(agent_input.ego_statuses[-1].driving_command, dtype=torch.float32),
                torch.tensor(agent_input.ego_statuses[-1].ego_velocity, dtype=torch.float32),
                torch.tensor(agent_input.ego_statuses[-1].ego_acceleration, dtype=torch.float32),
            ],
        )
        if self._config.include_ego_history:
            ego_history = np.stack(
                [status.ego_pose for status in agent_input.ego_statuses], axis=0
            ).astype(np.float32)
            features["ego_history"] = torch.tensor(ego_history)

        return features

    def _get_camera_feature(self, agent_input: AgentInput) -> torch.Tensor:
        """
        Extract stitched camera from AgentInput
        :param agent_input: input dataclass
        :return: stitched front view image as torch tensor
        """

        cameras = agent_input.cameras[-1]

        # Crop to ensure 4:1 aspect ratio
        l0 = cameras.cam_l0.image[28:-28, 416:-416]
        f0 = cameras.cam_f0.image[28:-28]
        r0 = cameras.cam_r0.image[28:-28, 416:-416]

        # stitch l0, f0, r0 images
        stitched_image = np.concatenate([l0, f0, r0], axis=1)
        resized_image = cv2.resize(stitched_image, (1024, 256))
        # resized_image = cv2.resize(stitched_image, (2048, 512))
        tensor_image = transforms.ToTensor()(resized_image)

        return tensor_image

    def _get_lidar_feature(self, agent_input: AgentInput) -> torch.Tensor:
        """
        Compute LiDAR feature as 2D histogram, according to Transfuser
        :param agent_input: input dataclass
        :return: LiDAR histogram as torch tensors
        """

        # only consider (x,y,z) & swap axes for (N,3) numpy array
        lidar_pc = agent_input.lidars[-1].lidar_pc[LidarIndex.POSITION].T

        # NOTE: Code from
        # https://github.com/autonomousvision/carla_garage/blob/main/team_code/data.py#L873
        def splat_points(point_cloud):
            # 256 x 256 grid
            xbins = np.linspace(
                self._config.lidar_min_x,
                self._config.lidar_max_x,
                (self._config.lidar_max_x - self._config.lidar_min_x) * int(self._config.pixels_per_meter) + 1,
            )
            ybins = np.linspace(
                self._config.lidar_min_y,
                self._config.lidar_max_y,
                (self._config.lidar_max_y - self._config.lidar_min_y) * int(self._config.pixels_per_meter) + 1,
            )
            hist = np.histogramdd(point_cloud[:, :2], bins=(xbins, ybins))[0]
            hist[hist > self._config.hist_max_per_pixel] = self._config.hist_max_per_pixel
            overhead_splat = hist / self._config.hist_max_per_pixel
            return overhead_splat

        # Remove points above the vehicle
        lidar_pc = lidar_pc[lidar_pc[..., 2] < self._config.max_height_lidar]
        below = lidar_pc[lidar_pc[..., 2] <= self._config.lidar_split_height]
        above = lidar_pc[lidar_pc[..., 2] > self._config.lidar_split_height]
        above_features = splat_points(above)
        if self._config.use_ground_plane:
            below_features = splat_points(below)
            features = np.stack([below_features, above_features], axis=-1)
        else:
            features = np.stack([above_features], axis=-1)
        features = np.transpose(features, (2, 0, 1)).astype(np.float32)

        return torch.tensor(features)


class TransfuserTargetBuilder(AbstractTargetBuilder):
    """Output target builder for TransFuser."""

    def __init__(self, config: TransfuserConfig):
        """
        Initializes target builder.
        :param config: global config dataclass of TransFuser
        """
        self._config = config

    def get_unique_name(self) -> str:
        """Inherited, see superclass."""
        return "transfuser_target"

    def compute_targets(self, scene: Scene) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""

        trajectory = torch.tensor(
            scene.get_future_trajectory(num_trajectory_frames=self._config.trajectory_sampling.num_poses).poses
        )
        frame_idx = scene.scene_metadata.num_history_frames - 1
        annotations = scene.frames[frame_idx].annotations
        ego_pose = StateSE2(*scene.frames[frame_idx].ego_status.ego_pose)

        agent_states, agent_labels = self._compute_agent_targets(annotations)
        bev_semantic_map = self._compute_bev_semantic_map(annotations, scene.map_api, ego_pose)

        targets = {
            "trajectory": trajectory,
            "agent_states": agent_states,
            "agent_labels": agent_labels,
            "bev_semantic_map": bev_semantic_map,
            "token": scene.scene_metadata.initial_token,
        }
        if self._config.include_future_bev_semantic_map:
            future_bev_semantic_map = self._compute_future_bev_semantic_map(
                scene, ego_pose, frame_idx
            )
            targets["future_bev_semantic_map"] = future_bev_semantic_map

        feasible_area_mask = None
        feasible_lane_mask = None
        if self._config.extract_feasible_lane or self._config.trajectory_candidates_filter_by_feasible:
            feasible_area_mask, feasible_lane_mask = self._extract_feasible_lane_from_map(
                bev_semantic_map
            )

        dynamic_obstacle_mask = None
        if self._config.generate_trajectory_candidates and self._config.trajectory_candidates_filter_by_dynamic:
            dynamic_obstacle_mask = self._compute_dynamic_obstacle_mask(annotations)

        if self._config.generate_trajectory_candidates:
            candidates, candidates_mask = self._generate_candidate_trajectories(
                trajectory, scene.map_api, ego_pose, feasible_area_mask, dynamic_obstacle_mask
            )
            targets["trajectory_candidates"] = candidates
            targets["trajectory_candidates_mask"] = candidates_mask
        if self._config.extract_feasible_lane:
            targets["feasible_area_mask"] = feasible_area_mask
            targets["feasible_lane_mask"] = feasible_lane_mask
        if self._config.include_velocity_bev:
            prev_frame_idx = max(frame_idx - 1, 0)
            if prev_frame_idx != frame_idx:
                prev_frame = scene.frames[prev_frame_idx]
                prev_annotations = prev_frame.annotations
                prev_ego_pose = StateSE2(*prev_frame.ego_status.ego_pose)
            else:
                prev_annotations = None
                prev_ego_pose = None
            velocity_bev = self._compute_velocity_bev(
                annotations, prev_annotations, ego_pose, prev_ego_pose
            )
            targets["velocity_bev"] = velocity_bev
        return targets

    def _generate_candidate_trajectories(
        self,
        trajectory: torch.Tensor,
        map_api: AbstractMap,
        ego_pose: StateSE2,
        feasible_area_mask: Optional[torch.Tensor] = None,
        dynamic_obstacle_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mode = getattr(self._config, "trajectory_candidates_mode", "centerline")
        if mode == "semantic":
            candidates, candidates_mask = self._generate_semantic_candidate_trajectories(trajectory)
        else:
            candidates, candidates_mask = self._generate_lane_centerline_candidates(
                trajectory, map_api, ego_pose
            )
        if self._config.trajectory_candidates_filter_by_feasible and feasible_area_mask is not None:
            candidates, candidates_mask = self._filter_candidates_by_feasible_area(
                candidates, candidates_mask, feasible_area_mask
            )
        if self._config.trajectory_candidates_filter_by_dynamic and dynamic_obstacle_mask is not None:
            candidates, candidates_mask = self._filter_candidates_by_dynamic_obstacles(
                candidates, candidates_mask, dynamic_obstacle_mask
            )
        return candidates, candidates_mask

    def _generate_semantic_candidate_trajectories(
        self, trajectory: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        gt_traj = trajectory.detach().cpu().numpy().astype(np.float32)
        max_count = max(1, self._config.trajectory_candidates_max_count)
        num_steps = gt_traj.shape[0]
        candidates = np.zeros((max_count, num_steps, 3), dtype=np.float32)
        candidates_mask = np.zeros((max_count,), dtype=bool)
        if gt_traj.ndim != 2 or gt_traj.shape[1] < 2:
            if gt_traj.ndim == 2 and gt_traj.shape[1] == 3 and max_count > 0:
                candidates[0] = gt_traj
                candidates_mask[0] = True
            return torch.tensor(candidates), torch.tensor(candidates_mask)

        gt_xy = gt_traj[:, :2]
        if gt_xy.shape[0] == 0:
            return torch.tensor(candidates), torch.tensor(candidates_mask)
        if gt_traj.shape[1] >= 3:
            gt_heading = gt_traj[:, 2]
            gt_full = gt_traj[:, :3]
        else:
            gt_heading = self._compute_headings(gt_xy)
            gt_full = np.concatenate([gt_xy, gt_heading[:, None]], axis=1)

        dt = float(self._config.trajectory_sampling.interval_length)
        yaw_rate = self._estimate_initial_yaw_rate(gt_heading, dt)
        speed_profile = self._get_speed_profile(gt_xy, dt)
        base_xy, base_heading = self._build_constant_curvature_path(
            gt_xy[0], gt_heading[0], speed_profile, yaw_rate, dt
        )
        keep_lane = np.concatenate([base_xy, base_heading[:, None]], axis=1)

        offset_mag = float(self._config.trajectory_candidates_lane_change_offset_m)
        if self._config.trajectory_candidates_lane_change_use_gt_offset:
            offset_est = self._estimate_lane_change_offset(gt_xy, base_xy, base_heading)
            if abs(offset_est) > 1e-3:
                min_m = float(self._config.trajectory_candidates_lane_change_min_m)
                max_m = float(self._config.trajectory_candidates_lane_change_max_m)
                offset_mag = float(np.clip(abs(offset_est), min_m, max_m))

        lane_left_xy = self._apply_lateral_offset(base_xy, base_heading, offset_mag)
        lane_right_xy = self._apply_lateral_offset(base_xy, base_heading, -offset_mag)
        lane_left = np.concatenate(
            [lane_left_xy, self._compute_headings(lane_left_xy)[:, None]], axis=1
        )
        lane_right = np.concatenate(
            [lane_right_xy, self._compute_headings(lane_right_xy)[:, None]], axis=1
        )

        idx = 0
        if self._config.trajectory_candidates_include_gt and idx < max_count:
            candidates[idx] = gt_full
            candidates_mask[idx] = True
            idx += 1
        if idx < max_count:
            candidates[idx] = keep_lane
            candidates_mask[idx] = True
            idx += 1
        if idx < max_count:
            candidates[idx] = lane_left
            candidates_mask[idx] = True
            idx += 1
        if idx < max_count:
            candidates[idx] = lane_right
            candidates_mask[idx] = True

        return torch.tensor(candidates), torch.tensor(candidates_mask)

    def _estimate_initial_yaw_rate(self, headings: np.ndarray, dt: float) -> float:
        if headings.shape[0] < 2 or dt <= 0.0:
            return 0.0
        deltas = np.array(
            [normalize_angle(h1 - h0) for h0, h1 in zip(headings[:-1], headings[1:])],
            dtype=np.float32,
        )
        window = min(
            int(self._config.trajectory_candidates_yaw_rate_window), deltas.shape[0]
        )
        if window <= 0:
            return 0.0
        return float(np.median(deltas[:window]) / dt)

    def _get_speed_profile(self, gt_xy: np.ndarray, dt: float) -> np.ndarray:
        if gt_xy.shape[0] < 2 or dt <= 0.0:
            return np.zeros((gt_xy.shape[0],), dtype=np.float32)
        step_dist = np.linalg.norm(gt_xy[1:] - gt_xy[:-1], axis=1)
        speeds = step_dist / dt
        speeds = np.concatenate([speeds, speeds[-1:]], axis=0)
        return speeds.astype(np.float32)

    def _build_constant_curvature_path(
        self,
        start_xy: np.ndarray,
        start_heading: float,
        speeds: np.ndarray,
        yaw_rate: float,
        dt: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        num_steps = speeds.shape[0]
        base_xy = np.zeros((num_steps, 2), dtype=np.float32)
        base_heading = np.zeros((num_steps,), dtype=np.float32)
        if num_steps == 0:
            return base_xy, base_heading
        base_xy[0] = start_xy.astype(np.float32)
        base_heading[0] = float(start_heading)
        for i in range(1, num_steps):
            heading = base_heading[i - 1]
            v = float(speeds[i - 1])
            base_xy[i, 0] = base_xy[i - 1, 0] + v * np.cos(heading) * dt
            base_xy[i, 1] = base_xy[i - 1, 1] + v * np.sin(heading) * dt
            base_heading[i] = float(normalize_angle(heading + yaw_rate * dt))
        return base_xy, base_heading

    def _estimate_lane_change_offset(
        self, gt_xy: np.ndarray, base_xy: np.ndarray, base_heading: np.ndarray
    ) -> float:
        if gt_xy.shape[0] == 0:
            return 0.0
        delta = gt_xy - base_xy
        lateral = -np.sin(base_heading) * delta[:, 0] + np.cos(base_heading) * delta[:, 1]
        return float(np.median(lateral))

    def _apply_lateral_offset(
        self, base_xy: np.ndarray, base_heading: np.ndarray, offset: float
    ) -> np.ndarray:
        if base_xy.shape[0] == 0:
            return base_xy
        s = np.linspace(0.0, 1.0, base_xy.shape[0], dtype=np.float32)
        smooth = 3.0 * s ** 2 - 2.0 * s ** 3
        offset_profile = offset * smooth
        nx = -np.sin(base_heading)
        ny = np.cos(base_heading)
        offset_xy = np.stack([nx * offset_profile, ny * offset_profile], axis=1)
        return (base_xy + offset_xy).astype(np.float32)

    def _align_candidate_start(
        self, samples: np.ndarray, start_xy: np.ndarray, blend_steps: int
    ) -> np.ndarray:
        if samples.shape[0] == 0:
            return samples
        if blend_steps <= 0 or samples.shape[0] < 2:
            return samples
        points = np.concatenate([start_xy[None, :], samples], axis=0)
        k = min(blend_steps, points.shape[0] - 1)
        if k > 1:
            anchor = points[k].copy()
            denom = float(k - 1)
            for idx in range(1, k):
                t = idx / denom
                points[idx] = start_xy * (1.0 - t) + anchor * t
        return points[1:]

    def _generate_lane_centerline_candidates(
        self,
        trajectory: torch.Tensor,
        map_api: AbstractMap,
        ego_pose: StateSE2,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        gt_traj = trajectory.detach().cpu().numpy().astype(np.float32)
        max_count = max(1, self._config.trajectory_candidates_max_count)
        if gt_traj.ndim != 2 or gt_traj.shape[1] < 2:
            candidates = np.zeros((max_count, gt_traj.shape[0], 3), dtype=np.float32)
            candidates_mask = np.zeros((max_count,), dtype=bool)
            if gt_traj.ndim == 2 and gt_traj.shape[1] == 3:
                candidates[0] = gt_traj
                candidates_mask[0] = True
            return torch.tensor(candidates), torch.tensor(candidates_mask)

        gt_xy = gt_traj[:, :2]
        ref_heading = self._get_reference_heading(gt_xy)
        step_dist = np.linalg.norm(gt_xy[1:] - gt_xy[:-1], axis=1)
        cum_dist = np.concatenate([[0.0], np.cumsum(step_dist)], axis=0)
        total_dist = float(cum_dist[-1]) if cum_dist.size > 0 else 0.0

        lines = self._get_centerlines(map_api, ego_pose)
        max_heading_err = np.deg2rad(self._config.trajectory_candidates_max_heading_error_deg)
        max_lateral_err = self._config.trajectory_candidates_max_lateral_error
        min_length_ratio = self._config.trajectory_candidates_min_length_ratio
        tangent_eps = self._config.trajectory_candidates_tangent_eps

        lane_candidates = []
        start_xy = np.array([0.0, 0.0], dtype=np.float32)
        start_point = Point(float(start_xy[0]), float(start_xy[1]))
        blend_steps = int(self._config.trajectory_candidates_start_blend_steps)
        heading_norm = max(max_heading_err, 1e-6)
        lateral_norm = max(max_lateral_err, 1e-6)
        for line in lines:
            if line.length <= 0:
                continue
            s0 = float(line.project(start_point))
            if total_dist > 1e-3 and (line.length - s0) < total_dist * min_length_ratio:
                continue

            tangent = self._line_tangent(line, s0, tangent_eps)
            if tangent is None:
                continue
            line_heading = np.arctan2(tangent[1], tangent[0])
            heading_err = abs(normalize_angle(line_heading - ref_heading))
            if heading_err > max_heading_err:
                continue

            lateral_err = float(line.distance(start_point))
            if lateral_err > max_lateral_err:
                continue

            samples = []
            for s in s0 + cum_dist:
                s_clamped = min(max(s, 0.0), line.length)
                pt = line.interpolate(s_clamped)
                samples.append([pt.x, pt.y])
            samples = np.asarray(samples, dtype=np.float32)
            samples = self._align_candidate_start(samples, start_xy, blend_steps)
            headings = self._compute_headings(samples)
            candidate = np.concatenate([samples, headings[:, None]], axis=1)
            score = heading_err / heading_norm + lateral_err / lateral_norm
            lane_candidates.append((score, candidate))

        lane_candidates.sort(key=lambda item: item[0])
        selected = [cand for _, cand in lane_candidates[: max_count - 1]]

        candidates = np.repeat(gt_traj[None, ...], max_count, axis=0).astype(np.float32)
        candidates_mask = np.zeros((max_count,), dtype=bool)
        candidates[0] = gt_traj
        candidates_mask[0] = True
        for idx, cand in enumerate(selected, start=1):
            if idx >= max_count:
                break
            candidates[idx] = cand
            candidates_mask[idx] = True

        return torch.tensor(candidates), torch.tensor(candidates_mask)

    def _filter_candidates_by_feasible_area(
        self,
        candidates: torch.Tensor,
        candidates_mask: torch.Tensor,
        feasible_area_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if candidates.numel() == 0:
            return candidates, candidates_mask
        mask = feasible_area_mask
        if mask.dim() == 3 and mask.shape[0] == 1:
            mask = mask[0]
        mask_np = mask.detach().cpu().numpy().astype(bool)
        cand_np = candidates.detach().cpu().numpy()
        cand_mask_np = candidates_mask.detach().cpu().numpy().astype(bool)
        min_ratio = float(self._config.trajectory_candidates_feasible_ratio)
        step_m = float(self._config.trajectory_candidates_feasible_step_m)
        for idx in range(cand_np.shape[0]):
            if not cand_mask_np[idx]:
                continue
            if idx == 0:
                continue
            ratio = self._trajectory_feasible_ratio(cand_np[idx, :, :2], mask_np, step_m)
            if ratio < min_ratio:
                cand_mask_np[idx] = False
        return torch.tensor(cand_np), torch.tensor(cand_mask_np)

    def _filter_candidates_by_dynamic_obstacles(
        self,
        candidates: torch.Tensor,
        candidates_mask: torch.Tensor,
        dynamic_obstacle_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if candidates.numel() == 0:
            return candidates, candidates_mask
        mask = dynamic_obstacle_mask
        if mask.dim() == 3 and mask.shape[0] == 1:
            mask = mask[0]
        mask_np = mask.detach().cpu().numpy().astype(bool)
        cand_np = candidates.detach().cpu().numpy()
        cand_mask_np = candidates_mask.detach().cpu().numpy().astype(bool)
        steps = int(self._config.trajectory_candidates_dynamic_filter_steps)
        steps = max(0, min(steps, cand_np.shape[1]))
        if steps == 0:
            return candidates, candidates_mask
        height, width = mask_np.shape
        ego_row = 0
        ego_col = (width - 1) // 2
        for idx in range(cand_np.shape[0]):
            if not cand_mask_np[idx]:
                continue
            if idx == 0:
                continue
            pts = cand_np[idx, :steps, :2]
            rows = ego_row + pts[:, 0] / float(self._config.bev_pixel_size)
            cols = ego_col + pts[:, 1] / float(self._config.bev_pixel_size)
            rows_i = np.round(rows).astype(np.int32)
            cols_i = np.round(cols).astype(np.int32)
            valid = (rows_i >= 0) & (rows_i < height) & (cols_i >= 0) & (cols_i < width)
            if not valid.any():
                continue
            if mask_np[rows_i[valid], cols_i[valid]].any():
                cand_mask_np[idx] = False
        return torch.tensor(cand_np), torch.tensor(cand_mask_np)

    def _trajectory_feasible_ratio(
        self, points: np.ndarray, feasible_mask: np.ndarray, step_m: float
    ) -> float:
        if points.shape[0] == 0:
            return 0.0
        samples = self._sample_polyline(points, step_m)
        if samples.shape[0] == 0:
            return 0.0
        height, width = feasible_mask.shape
        ego_row = 0
        ego_col = (width - 1) // 2
        rows = ego_row + samples[:, 0] / float(self._config.bev_pixel_size)
        cols = ego_col + samples[:, 1] / float(self._config.bev_pixel_size)
        rows_i = np.round(rows).astype(np.int32)
        cols_i = np.round(cols).astype(np.int32)
        valid = (rows_i >= 0) & (rows_i < height) & (cols_i >= 0) & (cols_i < width)
        if not valid.any():
            return 0.0
        inside = np.zeros((rows_i.shape[0],), dtype=bool)
        inside[valid] = feasible_mask[rows_i[valid], cols_i[valid]]
        return float(inside.mean())

    @staticmethod
    def _sample_polyline(points: np.ndarray, step_m: float) -> np.ndarray:
        if points.shape[0] <= 1:
            return points.astype(np.float32)
        if step_m <= 0.0:
            return points.astype(np.float32)
        samples = [points[0].astype(np.float32)]
        for idx in range(1, points.shape[0]):
            p0 = points[idx - 1].astype(np.float32)
            p1 = points[idx].astype(np.float32)
            seg = p1 - p0
            dist = float(np.linalg.norm(seg))
            if dist < 1e-6:
                continue
            count = max(1, int(np.floor(dist / step_m)))
            for step in range(1, count + 1):
                t = step / float(count)
                samples.append(p0 + seg * t)
        return np.asarray(samples, dtype=np.float32)

    def _get_reference_heading(self, gt_xy: np.ndarray) -> float:
        if gt_xy.shape[0] < 2:
            return 0.0
        deltas = gt_xy[1:] - gt_xy[:-1]
        for dx, dy in deltas:
            if abs(dx) > 1e-3 or abs(dy) > 1e-3:
                return float(np.arctan2(dy, dx))
        return 0.0

    def _compute_headings(self, points: np.ndarray) -> np.ndarray:
        if points.shape[0] <= 1:
            return np.zeros((points.shape[0],), dtype=np.float32)
        deltas = points[1:] - points[:-1]
        headings = np.arctan2(deltas[:, 1], deltas[:, 0])
        headings = np.concatenate([headings, headings[-1:]], axis=0)
        return headings.astype(np.float32)

    def _line_tangent(self, line: LineString, s: float, eps: float) -> Optional[np.ndarray]:
        s0 = max(0.0, s - eps)
        s1 = min(line.length, s + eps)
        if s1 - s0 < 1e-3:
            return None
        p0 = line.interpolate(s0)
        p1 = line.interpolate(s1)
        return np.array([p1.x - p0.x, p1.y - p0.y], dtype=np.float32)

    def _get_centerlines(self, map_api: AbstractMap, ego_pose: StateSE2) -> List[LineString]:
        radius = self._config.trajectory_candidates_search_radius
        map_object_dict = map_api.get_proximal_map_objects(
            point=ego_pose.point,
            radius=radius,
            layers=[SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR],
        )
        lines: List[LineString] = []
        for layer in [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]:
            for map_object in map_object_dict.get(layer, []):
                linestring = self._geometry_local_coords(
                    map_object.baseline_path.linestring, ego_pose
                )
                if linestring.length > 0:
                    lines.append(linestring)
        return lines

    def _extract_feasible_lane_from_map(
        self, bev_semantic_map: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        single_sample = bev_semantic_map.dim() == 2 or (
            bev_semantic_map.dim() == 3
            and (
                bev_semantic_map.shape[0] == self._config.num_bev_classes
                or bev_semantic_map.shape[-1] == self._config.num_bev_classes
            )
        )
        class_map = self._to_class_map(bev_semantic_map)
        road_label = self._config.bev_road_label
        centerline_label = self._config.bev_centerline_label

        drivable_mask = (class_map == road_label) | (class_map == centerline_label)
        centerline_mask = class_map == centerline_label

        batch_size, height, width = class_map.shape
        ego_row = 0
        ego_col = (width - 1) // 2

        seed_rows = max(1, min(self._config.feasible_lane_seed_rows, height))   
        row_start = 0
        row_end = min(height, seed_rows)
        seed_cols = max(0, min(self._config.feasible_lane_seed_cols, width // 2))
        col_start = max(0, ego_col - seed_cols)
        col_end = min(width, ego_col + seed_cols + 1)
        if col_end <= col_start:
            col_start = min(ego_col, width - 1)
            col_end = min(width, col_start + 1)

        row_coords = torch.arange(row_start, row_end, device=class_map.device)   
        col_coords = torch.arange(col_start, col_end, device=class_map.device)  
        row_grid = row_coords[:, None]
        col_grid = col_coords[None, :]
        base_dist = (row_grid - ego_row).abs() + (col_grid - ego_col).abs()     
        base_score = -base_dist.float()

        seed_mask = torch.zeros_like(drivable_mask)
        for batch_idx in range(batch_size):
            region_mask = drivable_mask[batch_idx, row_start:row_end, col_start:col_end]
            if region_mask.any().item():
                region_scores = base_score.masked_fill(~region_mask, float("-inf"))
                flat_idx = region_scores.view(-1).argmax().item()
                row_offset = flat_idx // (col_end - col_start)
                col_offset = flat_idx % (col_end - col_start)
                seed_mask[batch_idx, row_start + row_offset, col_start + col_offset] = True
            else:
                full_mask = drivable_mask[batch_idx]
                if full_mask.any().item():
                    coords = full_mask.nonzero(as_tuple=False)
                    ego_index = torch.tensor(
                        [ego_row, ego_col], device=coords.device, dtype=coords.dtype
                    )
                    deltas = coords - ego_index
                    dist2 = deltas[:, 0] * deltas[:, 0] + deltas[:, 1] * deltas[:, 1]
                    best_idx = dist2.argmin().item()
                    seed_mask[batch_idx, coords[best_idx, 0], coords[best_idx, 1]] = True
                else:
                    seed_mask[batch_idx, min(ego_row, height - 1), min(ego_col, width - 1)] = True

        feasible_area_mask = self._flood_fill_mask(drivable_mask, seed_mask)
        feasible_lane_mask = feasible_area_mask & centerline_mask
        if single_sample and feasible_area_mask.shape[0] == 1:
            return feasible_area_mask[0], feasible_lane_mask[0]
        return feasible_area_mask, feasible_lane_mask

    def _to_class_map(self, bev_semantic_map: torch.Tensor) -> torch.Tensor:    
        if bev_semantic_map.dim() == 2:
            return bev_semantic_map.long().unsqueeze(0)
        if bev_semantic_map.dim() == 3:
            if bev_semantic_map.shape[0] == self._config.num_bev_classes:       
                class_map = bev_semantic_map.argmax(dim=0).long()
            elif bev_semantic_map.shape[-1] == self._config.num_bev_classes:    
                class_map = bev_semantic_map.argmax(dim=-1).long()
            else:
                class_map = bev_semantic_map.long()
            if class_map.dim() == 2:
                class_map = class_map.unsqueeze(0)
            return class_map
        if bev_semantic_map.dim() == 4:
            return bev_semantic_map.argmax(dim=1).long()
        raise ValueError("Unsupported bev_semantic_map shape for target extraction.")

    def _flood_fill_mask(self, mask: torch.Tensor, seed: torch.Tensor) -> torch.Tensor:
        max_iters = self._config.feasible_lane_max_iters
        if max_iters <= 0:
            max_iters = mask.shape[-2] + mask.shape[-1]

        current = seed
        for _ in range(max_iters):
            expanded = F.max_pool2d(
                current.float().unsqueeze(1), kernel_size=3, stride=1, padding=1
            ).squeeze(1) > 0
            next_mask = expanded & mask
            if torch.equal(next_mask, current):
                break
            current = next_mask

        return current

    def _compute_agent_targets(self, annotations: Annotations) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extracts 2D agent bounding boxes in ego coordinates
        :param annotations: annotation dataclass
        :return: tuple of bounding box values and labels (binary)
        """

        max_agents = self._config.num_bounding_boxes
        agent_states_list: List[npt.NDArray[np.float32]] = []

        def _xy_in_lidar(x: float, y: float, config: TransfuserConfig) -> bool:
            return (config.lidar_min_x <= x <= config.lidar_max_x) and (config.lidar_min_y <= y <= config.lidar_max_y)

        for box, name in zip(annotations.boxes, annotations.names):
            box_x, box_y, box_heading, box_length, box_width = (
                box[BoundingBoxIndex.X],
                box[BoundingBoxIndex.Y],
                box[BoundingBoxIndex.HEADING],
                box[BoundingBoxIndex.LENGTH],
                box[BoundingBoxIndex.WIDTH],
            )

            if name == "vehicle" and _xy_in_lidar(box_x, box_y, self._config):
                agent_states_list.append(np.array([box_x, box_y, box_heading, box_length, box_width], dtype=np.float32))

        agents_states_arr = np.array(agent_states_list)

        # filter num_instances nearest
        agent_states = np.zeros((max_agents, BoundingBox2DIndex.size()), dtype=np.float32)
        agent_labels = np.zeros(max_agents, dtype=bool)

        if len(agents_states_arr) > 0:
            distances = np.linalg.norm(agents_states_arr[..., BoundingBox2DIndex.POINT], axis=-1)
            argsort = np.argsort(distances)[:max_agents]

            # filter detections
            agents_states_arr = agents_states_arr[argsort]
            agent_states[: len(agents_states_arr)] = agents_states_arr
            agent_labels[: len(agents_states_arr)] = True

        return torch.tensor(agent_states), torch.tensor(agent_labels)

    def _compute_bev_semantic_map(
        self, annotations: Annotations, map_api: AbstractMap, ego_pose: StateSE2
    ) -> torch.Tensor:
        """
        Creates sematic map in BEV
        :param annotations: annotation dataclass
        :param map_api: map interface of nuPlan
        :param ego_pose: ego pose in global frame
        :return: 2D torch tensor of semantic labels
        """

        bev_semantic_map = np.zeros(self._config.bev_semantic_frame, dtype=np.int64)    # bev的尺寸
        for label, (entity_type, layers) in self._config.bev_semantic_classes.items():
            if entity_type == "polygon":
                entity_mask = self._compute_map_polygon_mask(map_api, ego_pose, layers)
            elif entity_type == "linestring":
                entity_mask = self._compute_map_linestring_mask(map_api, ego_pose, layers)
            else:
                entity_mask = self._compute_box_mask(annotations, layers)
            bev_semantic_map[entity_mask] = label

        return torch.Tensor(bev_semantic_map)

    def _compute_future_bev_semantic_map(
        self, scene: Scene, ego_pose: StateSE2, frame_idx: int
    ) -> torch.Tensor:
        future_offset = int(self._config.trajectory_sampling.num_poses)
        future_idx = min(frame_idx + future_offset, len(scene.frames) - 1)
        future_frame = scene.frames[future_idx]
        future_pose = StateSE2(*future_frame.ego_status.ego_pose)
        aligned_annotations = self._transform_annotations_between_poses(
            future_frame.annotations, future_pose, ego_pose
        )
        return self._compute_bev_semantic_map(
            aligned_annotations, scene.map_api, ego_pose
        )

    def _transform_annotations_between_poses(
        self, annotations: Annotations, src_pose: StateSE2, dst_pose: StateSE2
    ) -> Annotations:
        if annotations is None:
            return annotations
        boxes = np.array(annotations.boxes, dtype=np.float32, copy=True)
        if boxes.size > 0:
            for idx in range(boxes.shape[0]):
                x = float(boxes[idx, BoundingBoxIndex.X])
                y = float(boxes[idx, BoundingBoxIndex.Y])
                heading = float(boxes[idx, BoundingBoxIndex.HEADING])
                xy_dst = self._transform_point_between_poses(
                    np.array([x, y], dtype=np.float32), src_pose, dst_pose
                )
                heading_global = normalize_angle(heading + float(src_pose.heading))
                heading_dst = normalize_angle(heading_global - float(dst_pose.heading))
                boxes[idx, BoundingBoxIndex.X] = xy_dst[0]
                boxes[idx, BoundingBoxIndex.Y] = xy_dst[1]
                boxes[idx, BoundingBoxIndex.HEADING] = heading_dst
        return Annotations(
            boxes=boxes,
            names=list(annotations.names),
            velocity_3d=np.array(annotations.velocity_3d, copy=True),
            instance_tokens=list(annotations.instance_tokens),
            track_tokens=list(annotations.track_tokens),
        )

    def _compute_dynamic_obstacle_mask(self, annotations: Annotations) -> torch.Tensor:
        dynamic_layers = [
            TrackedObjectType.VEHICLE,
            TrackedObjectType.PEDESTRIAN,
            TrackedObjectType.BICYCLE,
        ]
        dynamic_mask = self._compute_box_mask(annotations, dynamic_layers)
        return torch.tensor(dynamic_mask)

    def _compute_velocity_bev(
        self,
        current_annotations: Annotations,
        prev_annotations: Optional[Annotations],
        current_ego_pose: StateSE2,
        prev_ego_pose: Optional[StateSE2],
    ) -> torch.Tensor:
        vel_x = np.zeros(self._config.bev_semantic_frame[::-1], dtype=np.float32)
        vel_y = np.zeros(self._config.bev_semantic_frame[::-1], dtype=np.float32)
        if prev_annotations is None or prev_ego_pose is None:
            vel_x = np.rot90(vel_x)[::-1]
            vel_y = np.rot90(vel_y)[::-1]
            return torch.tensor(np.stack([vel_x, vel_y], axis=0))

        prev_by_token = {
            token: box
            for token, box in zip(prev_annotations.track_tokens, prev_annotations.boxes)
        }
        dynamic_types = {
            TrackedObjectType.VEHICLE,
            TrackedObjectType.PEDESTRIAN,
            TrackedObjectType.BICYCLE,
        }
        dt = max(float(NAVSIM_INTERVAL_LENGTH), 1e-6)

        def _xy_in_lidar(x: float, y: float) -> bool:
            return (self._config.lidar_min_x <= x <= self._config.lidar_max_x) and (
                self._config.lidar_min_y <= y <= self._config.lidar_max_y
            )

        for name, box, track_token in zip(
            current_annotations.names,
            current_annotations.boxes,
            current_annotations.track_tokens,
        ):
            track_type = tracked_object_types.get(name)
            if track_type not in dynamic_types:
                continue
            if track_token not in prev_by_token:
                continue
            x, y = float(box[BoundingBoxIndex.X]), float(box[BoundingBoxIndex.Y])
            if not _xy_in_lidar(x, y):
                continue
            prev_box = prev_by_token[track_token]
            prev_xy = np.array(
                [prev_box[BoundingBoxIndex.X], prev_box[BoundingBoxIndex.Y]],
                dtype=np.float32,
            )
            prev_xy_local = self._transform_point_between_poses(
                prev_xy, prev_ego_pose, current_ego_pose
            )
            cur_xy = np.array([x, y], dtype=np.float32)
            vel = (cur_xy - prev_xy_local) / dt
            vx, vy = float(vel[0]), float(vel[1])

            heading = float(box[BoundingBoxIndex.HEADING])
            box_length = float(box[BoundingBoxIndex.LENGTH])
            box_width = float(box[BoundingBoxIndex.WIDTH])
            box_height = float(box[BoundingBoxIndex.HEIGHT])
            agent_box = OrientedBox(
                StateSE2(x, y, heading), box_length, box_width, box_height
            )
            exterior = np.array(agent_box.geometry.exterior.coords).reshape((-1, 1, 2))
            exterior = self._coords_to_pixel(exterior)
            cv2.fillPoly(vel_x, [exterior], color=vx)
            cv2.fillPoly(vel_y, [exterior], color=vy)

        vel_x = np.rot90(vel_x)[::-1]
        vel_y = np.rot90(vel_y)[::-1]
        return torch.tensor(np.stack([vel_x, vel_y], axis=0))

    @staticmethod
    def _transform_point_between_poses(
        point: np.ndarray, src_pose: StateSE2, dst_pose: StateSE2
    ) -> np.ndarray:
        cos_s, sin_s = np.cos(src_pose.heading), np.sin(src_pose.heading)
        x_global = src_pose.x + cos_s * point[0] - sin_s * point[1]
        y_global = src_pose.y + sin_s * point[0] + cos_s * point[1]

        dx = x_global - dst_pose.x
        dy = y_global - dst_pose.y
        cos_d, sin_d = np.cos(dst_pose.heading), np.sin(dst_pose.heading)
        x_local = cos_d * dx + sin_d * dy
        y_local = -sin_d * dx + cos_d * dy
        return np.array([x_local, y_local], dtype=np.float32)

    def _compute_map_polygon_mask(
        self, map_api: AbstractMap, ego_pose: StateSE2, layers: List[SemanticMapLayer]
    ) -> npt.NDArray[np.bool_]:
        """
        Compute binary mask given a map layer class
        :param map_api: map interface of nuPlan
        :param ego_pose: ego pose in global frame
        :param layers: map layers
        :return: binary mask as numpy array
        """

        map_object_dict = map_api.get_proximal_map_objects(
            point=ego_pose.point, radius=self._config.bev_radius, layers=layers
        )
        map_polygon_mask = np.zeros(self._config.bev_semantic_frame[::-1], dtype=np.uint8)
        for layer in layers:
            for map_object in map_object_dict[layer]:
                polygon: Polygon = self._geometry_local_coords(map_object.polygon, ego_pose)
                exterior = np.array(polygon.exterior.coords).reshape((-1, 1, 2))
                exterior = self._coords_to_pixel(exterior)
                cv2.fillPoly(map_polygon_mask, [exterior], color=255)
        # OpenCV has origin on top-left corner
        map_polygon_mask = np.rot90(map_polygon_mask)[::-1]
        return map_polygon_mask > 0

    def _compute_map_linestring_mask(
        self, map_api: AbstractMap, ego_pose: StateSE2, layers: List[SemanticMapLayer]
    ) -> npt.NDArray[np.bool_]:
        """
        Compute binary of linestring given a map layer class
        :param map_api: map interface of nuPlan
        :param ego_pose: ego pose in global frame
        :param layers: map layers
        :return: binary mask as numpy array
        """
        map_object_dict = map_api.get_proximal_map_objects(
            point=ego_pose.point, radius=self._config.bev_radius, layers=layers
        )
        map_linestring_mask = np.zeros(self._config.bev_semantic_frame[::-1], dtype=np.uint8)
        for layer in layers:
            for map_object in map_object_dict[layer]:
                linestring: LineString = self._geometry_local_coords(map_object.baseline_path.linestring, ego_pose)
                points = np.array(linestring.coords).reshape((-1, 1, 2))
                points = self._coords_to_pixel(points)
                cv2.polylines(map_linestring_mask, [points], isClosed=False, color=255, thickness=2)
        # OpenCV has origin on top-left corner
        map_linestring_mask = np.rot90(map_linestring_mask)[::-1]
        return map_linestring_mask > 0

    def _compute_box_mask(self, annotations: Annotations, layers: TrackedObjectType) -> npt.NDArray[np.bool_]:
        """
        Compute binary of bounding boxes in BEV space
        :param annotations: annotation dataclass
        :param layers: bounding box labels to include
        :return: binary mask as numpy array
        """
        box_polygon_mask = np.zeros(self._config.bev_semantic_frame[::-1], dtype=np.uint8)
        for name_value, box_value in zip(annotations.names, annotations.boxes):
            agent_type = tracked_object_types[name_value]
            if agent_type in layers:
                # box_value = (x, y, z, length, width, height, yaw) TODO: add intenum
                x, y, heading = box_value[0], box_value[1], box_value[-1]
                box_length, box_width, box_height = box_value[3], box_value[4], box_value[5]
                agent_box = OrientedBox(StateSE2(x, y, heading), box_length, box_width, box_height)
                exterior = np.array(agent_box.geometry.exterior.coords).reshape((-1, 1, 2))
                exterior = self._coords_to_pixel(exterior)
                cv2.fillPoly(box_polygon_mask, [exterior], color=255)
        # OpenCV has origin on top-left corner
        box_polygon_mask = np.rot90(box_polygon_mask)[::-1]
        return box_polygon_mask > 0

    @staticmethod
    def _query_map_objects(
        self, map_api: AbstractMap, ego_pose: StateSE2, layers: List[SemanticMapLayer]
    ) -> List[MapObject]:
        """
        Queries map objects
        :param map_api: map interface of nuPlan
        :param ego_pose: ego pose in global frame
        :param layers: map layers
        :return: list of map objects
        """

        # query map api with interesting layers
        map_object_dict = map_api.get_proximal_map_objects(point=ego_pose.point, radius=self, layers=layers)
        map_objects: List[MapObject] = []
        for layer in layers:
            map_objects += map_object_dict[layer]
        return map_objects

    @staticmethod
    def _geometry_local_coords(geometry: Any, origin: StateSE2) -> Any:
        """
        Transform shapely geometry in local coordinates of origin.
        :param geometry: shapely geometry
        :param origin: pose dataclass
        :return: shapely geometry
        """

        a = np.cos(origin.heading)
        b = np.sin(origin.heading)
        d = -np.sin(origin.heading)
        e = np.cos(origin.heading)
        xoff = -origin.x
        yoff = -origin.y

        translated_geometry = affinity.affine_transform(geometry, [1, 0, 0, 1, xoff, yoff])
        rotated_geometry = affinity.affine_transform(translated_geometry, [a, b, d, e, 0, 0])

        return rotated_geometry

    def _coords_to_pixel(self, coords):
        """
        Transform local coordinates in pixel indices of BEV map
        :param coords: _description_
        :return: _description_
        """

        # NOTE: remove half in backward direction
        pixel_center = np.array([[0, self._config.bev_pixel_width / 2.0]])
        coords_idcs = (coords / self._config.bev_pixel_size) + pixel_center

        return coords_idcs.astype(np.int32)


class BoundingBox2DIndex(IntEnum):
    """Intenum for bounding boxes in TransFuser."""

    _X = 0
    _Y = 1
    _HEADING = 2
    _LENGTH = 3
    _WIDTH = 4

    @classmethod
    def size(cls):
        valid_attributes = [
            attribute
            for attribute in dir(cls)
            if attribute.startswith("_") and not attribute.startswith("__") and not callable(getattr(cls, attribute))
        ]
        return len(valid_attributes)

    @classmethod
    @property
    def X(cls):
        return cls._X

    @classmethod
    @property
    def Y(cls):
        return cls._Y

    @classmethod
    @property
    def HEADING(cls):
        return cls._HEADING

    @classmethod
    @property
    def LENGTH(cls):
        return cls._LENGTH

    @classmethod
    @property
    def WIDTH(cls):
        return cls._WIDTH

    @classmethod
    @property
    def POINT(cls):
        # assumes X, Y have subsequent indices
        return slice(cls._X, cls._Y + 1)

    @classmethod
    @property
    def STATE_SE2(cls):
        # assumes X, Y, HEADING have subsequent indices
        return slice(cls._X, cls._HEADING + 1)
