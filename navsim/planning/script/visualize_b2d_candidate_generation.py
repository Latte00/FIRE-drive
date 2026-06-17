"""Preview high-quality Bench2Drive trajectory candidates without writing cache.

The script builds 40 map-based candidates per sample by default:
  - keep-lane candidates on the current centerline
  - normal lane-change candidates to an adjacent left/right centerline
  - early lane-change candidates that start immediately but still transition smoothly

It then scores them with the same B2D pseudo-PDM component definitions used by
DiffusionDrive training and renders raw candidates plus the selected top-k.
Nothing is written back to transfuser_target.gz.
"""

from __future__ import annotations

import argparse
import logging
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import torch
from tqdm import tqdm

from navsim.planning.script.run_b2d_diffusiondrive_caching import (
    B2D_WRONG_DIRECTION_DISABLED_SCENARIOS,
    BEV_CENTERLINE_LABEL,
    BEV_PEDESTRIAN_LABEL,
    BEV_PIXEL_HEIGHT,
    BEV_PIXEL_SIZE,
    BEV_PIXEL_WIDTH,
    BEV_ROAD_LABEL,
    BEV_STATIC_LABEL,
    BEV_VEHICLE_LABEL,
    build_future_agent_boxes,
    build_future_trajectory,
    build_lane_direction_bev,
    build_static_map_bev,
    compute_feasible_masks,
    compute_headings,
    infer_wrong_direction_enabled,
    interpolate_polyline_by_distance,
    is_same_log,
    load_pickle,
    load_route_scenarios,
    local_points_in_window,
    localize_map_points,
    normalize_angle,
    overlay_agents_on_bev,
    safe_token,
)

logger = logging.getLogger(__name__)


SEMANTIC_COLORS = [
    "#111827",
    "#9ca3af",
    "#ef4444",
    "#facc15",
    "#60a5fa",
    "#22c55e",
    "#a78bfa",
]

COMPONENT_NAMES = (
    "no_collision",
    "drivable",
    "progress",
    "ttc",
    "comfort",
    "wrong_lane",
)


@dataclass
class CandidateBatch:
    trajectories: np.ndarray
    kinds: List[str]
    side: List[str]


@dataclass
class ScoredCandidates:
    scores: np.ndarray
    components: np.ndarray
    progress_raw: np.ndarray
    gt_score: float
    gt_components: np.ndarray
    selected_idx: np.ndarray
    selected_mask: np.ndarray


def _to_numpy(value) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("_") or "sample"


def _xy_to_image(xy: np.ndarray, pixel_size: float, width: int) -> Tuple[np.ndarray, np.ndarray]:
    row = xy[:, 0] / max(pixel_size, 1e-6)
    col = xy[:, 1] / max(pixel_size, 1e-6) + (width - 1) / 2.0
    return row, col


def _polyline_length(polyline: np.ndarray) -> float:
    if polyline.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(polyline[1:] - polyline[:-1], axis=-1).sum())


def _polyline_tangent(polyline: np.ndarray, distance: float) -> np.ndarray:
    if polyline.shape[0] < 2:
        return np.array([1.0, 0.0], dtype=np.float32)
    seg = polyline[1:] - polyline[:-1]
    seg_len = np.linalg.norm(seg, axis=-1)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)], axis=0)
    j = int(np.searchsorted(cum, np.clip(distance, 0.0, float(cum[-1])), side="right") - 1)
    j = max(0, min(j, len(seg_len) - 1))
    tangent = seg[j]
    norm = float(np.linalg.norm(tangent))
    if norm < 1e-6:
        return np.array([1.0, 0.0], dtype=np.float32)
    return (tangent / norm).astype(np.float32)


def _trim_and_orient_centerline(local: np.ndarray) -> Optional[np.ndarray]:
    if local.shape[0] < 2:
        return None
    nearest_idx = int(np.argmin(np.linalg.norm(local, axis=-1)))
    if nearest_idx >= local.shape[0] - 1:
        local = local[::-1].copy()
        nearest_idx = int(np.argmin(np.linalg.norm(local, axis=-1)))
    tangent = local[min(nearest_idx + 1, local.shape[0] - 1)] - local[max(nearest_idx - 1, 0)]
    if float(np.linalg.norm(tangent)) < 1e-3:
        return None
    if tangent[0] < -0.1:
        local = local[::-1].copy()
        nearest_idx = int(np.argmin(np.linalg.norm(local, axis=-1)))
    if nearest_idx > 0:
        local = local[nearest_idx:].copy()
    if local.shape[0] < 2 or _polyline_length(local) < 5.0:
        return None
    return local.astype(np.float32)


def _command_point_local(info: Dict[str, object], key: str) -> Optional[np.ndarray]:
    value = info.get(key)
    if value is None:
        return None
    point = np.asarray(value, dtype=np.float32).reshape(-1)
    if point.shape[0] < 2 or not np.isfinite(point[:2]).all():
        return None
    world2lidar = np.asarray(info["sensors"]["LIDAR_TOP"]["world2lidar"], dtype=np.float32)
    local = localize_map_points(point[:2][None], world2lidar)
    if local.shape[0] == 0 or not np.isfinite(local[0]).all():
        return None
    return local[0].astype(np.float32)


def is_near_traffic_light_trigger(
    info: Dict[str, object],
    map_infos: Optional[Dict[str, object]],
    distance_m: float,
    include_stop_sign: bool = False,
) -> bool:
    if map_infos is None or float(distance_m) <= 0.0:
        return False
    town_name = str(info.get("town_name", ""))
    map_info = map_infos.get(town_name)
    if map_info is None:
        return False
    target_types = {"TrafficLight"}
    if include_stop_sign:
        target_types.add("StopSign")
    world2lidar = np.asarray(info["sensors"]["LIDAR_TOP"]["world2lidar"], dtype=np.float32)
    trigger_points = map_info.get("trigger_volumes_points", [])
    trigger_types = map_info.get("trigger_volumes_types", [])
    for points, trigger_type in zip(trigger_points, trigger_types):
        if str(trigger_type) not in target_types:
            continue
        local = localize_map_points(points, world2lidar)
        if local.shape[0] == 0:
            continue
        xy = local[:, :2].astype(np.float32, copy=False)
        centroid = xy.mean(axis=0, keepdims=True)
        probe = np.concatenate([xy, centroid], axis=0)
        if float(np.linalg.norm(probe, axis=-1).min()) <= float(distance_m):
            return True
    return False


def collect_centerlines(
    info: Dict[str, object],
    map_infos: Dict[str, object],
    search_radius: float,
) -> List[np.ndarray]:
    town_name = str(info.get("town_name", ""))
    map_info = map_infos.get(town_name)
    if map_info is None:
        return []
    world2lidar = np.asarray(info["sensors"]["LIDAR_TOP"]["world2lidar"], dtype=np.float32)
    centerlines: List[np.ndarray] = []
    lane_points = map_info.get("lane_points", [])
    lane_sample_points = map_info.get("lane_sample_points", [])
    lane_types = map_info.get("lane_types", [])
    for lane_idx, (points, lane_type) in enumerate(zip(lane_points, lane_types)):
        if str(lane_type) != "Center":
            continue
        sample_points = lane_sample_points[lane_idx] if lane_idx < len(lane_sample_points) else points
        sample_local = localize_map_points(sample_points, world2lidar)
        if not local_points_in_window(
            sample_local,
            front_min=-5.0,
            front_max=search_radius,
            lateral_abs=search_radius,
        ):
            continue
        local = localize_map_points(points, world2lidar)
        trimmed = _trim_and_orient_centerline(local)
        if trimmed is not None:
            centerlines.append(trimmed)
    return centerlines


def choose_current_centerline(
    centerlines: Sequence[np.ndarray],
    info: Dict[str, object],
    command_weight: float,
) -> Optional[np.ndarray]:
    if not centerlines:
        return None
    command = _command_point_local(info, "command_far_xy")
    if command is None:
        command = _command_point_local(info, "command_near_xy")
    command_norm = None
    if command is not None and np.linalg.norm(command) > 1e-3:
        command_norm = command / max(float(np.linalg.norm(command)), 1e-6)

    best_score = float("inf")
    best = None
    for line in centerlines:
        tangent = _polyline_tangent(line, 2.0)
        heading_penalty = max(0.0, 1.0 - float(tangent[0])) * 10.0
        command_penalty = 0.0
        if command_norm is not None:
            lane_vec = line[min(len(line) - 1, max(1, int(len(line) * 0.25)))] - line[0]
            if np.linalg.norm(lane_vec) > 1e-3:
                lane_vec = lane_vec / max(float(np.linalg.norm(lane_vec)), 1e-6)
                command_penalty = command_weight * max(0.0, 1.0 - float(np.dot(lane_vec, command_norm)))
        score = float(np.linalg.norm(line[0])) + heading_penalty + command_penalty
        if score < best_score:
            best_score = score
            best = line
    return best


def find_adjacent_centerlines(
    base: np.ndarray,
    centerlines: Sequence[np.ndarray],
    min_offset_m: float,
    max_offset_m: float,
    max_longitudinal_delta_m: float,
    min_parallel_cos: float,
    probe_distance_m: float,
) -> Dict[str, Optional[np.ndarray]]:
    base_point = interpolate_polyline_by_distance(base, np.asarray([probe_distance_m], dtype=np.float32))[0]
    tangent = _polyline_tangent(base, probe_distance_m)
    normal_left = np.array([-tangent[1], tangent[0]], dtype=np.float32)

    best: Dict[str, Tuple[float, np.ndarray]] = {}
    for line in centerlines:
        if line is base:
            continue
        line_point = interpolate_polyline_by_distance(line, np.asarray([probe_distance_m], dtype=np.float32))[0]
        delta = line_point - base_point
        lateral = float(np.dot(delta, normal_left))
        longitudinal = float(np.dot(delta, tangent))
        abs_lat = abs(lateral)
        if abs_lat < min_offset_m or abs_lat > max_offset_m:
            continue
        if abs(longitudinal) > max_longitudinal_delta_m:
            continue
        line_tangent = _polyline_tangent(line, probe_distance_m)
        if float(np.dot(tangent, line_tangent)) < min_parallel_cos:
            continue
        side = "left" if lateral > 0.0 else "right"
        quality = abs(abs_lat - 3.5) + 0.15 * abs(longitudinal)
        if side not in best or quality < best[side][0]:
            best[side] = (quality, line)
    return {"left": best.get("left", (float("inf"), None))[1], "right": best.get("right", (float("inf"), None))[1]}


def gt_distances_and_speeds(gt: np.ndarray, dt: float, min_speed: float) -> Tuple[np.ndarray, np.ndarray]:
    xy = gt[:, :2]
    if xy.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    prev = np.concatenate([np.zeros((1, 2), dtype=np.float32), xy[:-1]], axis=0)
    step = np.linalg.norm(xy - prev, axis=-1)
    if float(step.sum()) < 1e-3:
        speeds = np.full((xy.shape[0],), max(float(min_speed), 0.1), dtype=np.float32)
    else:
        speeds = np.maximum(step / max(dt, 1e-6), float(min_speed)).astype(np.float32)
    return np.cumsum(speeds * max(dt, 1e-6)).astype(np.float32), speeds


def accelerated_distances(
    gt_distances: np.ndarray,
    gt_speeds: np.ndarray,
    dt: float,
    rng: random.Random,
    accel_extra_min: float,
    accel_extra_max: float,
    speed_scale_min: float,
    speed_scale_max: float,
    max_speed: float,
) -> np.ndarray:
    if gt_distances.size == 0:
        return gt_distances
    extra_accel = rng.uniform(accel_extra_min, accel_extra_max)
    speed_scale = rng.uniform(speed_scale_min, speed_scale_max)
    t = (np.arange(1, gt_distances.shape[0] + 1, dtype=np.float32) * max(dt, 1e-6))
    speeds = np.minimum(gt_speeds * speed_scale + extra_accel * t, max_speed)
    distances = np.cumsum(speeds * max(dt, 1e-6)).astype(np.float32)
    return np.maximum(distances, gt_distances).astype(np.float32)


def sample_route_candidate(route: np.ndarray, distances: np.ndarray, gt_heading: np.ndarray) -> np.ndarray:
    xy = interpolate_polyline_by_distance(route, distances)
    if xy.shape[0] > 0:
        blend = np.linspace(0.0, 1.0, xy.shape[0], dtype=np.float32)
        blend = 3.0 * blend**2 - 2.0 * blend**3
        xy = xy - route[:1] * (1.0 - blend[:, None])
    heading = compute_headings(xy, fallback=gt_heading)
    return np.concatenate([xy.astype(np.float32), heading[:, None]], axis=1)


def sample_lane_change_candidate(
    base_route: np.ndarray,
    target_route: np.ndarray,
    distances: np.ndarray,
    gt_heading: np.ndarray,
    start_step: int,
    duration_steps: int,
) -> np.ndarray:
    base_xy = interpolate_polyline_by_distance(base_route, distances)
    target_xy = interpolate_polyline_by_distance(target_route, distances)
    if base_xy.shape[0] > 0:
        start_blend = np.linspace(0.0, 1.0, base_xy.shape[0], dtype=np.float32)
        start_blend = 3.0 * start_blend**2 - 2.0 * start_blend**3
        base_xy = base_xy - base_route[:1] * (1.0 - start_blend[:, None])
        target_xy = target_xy - target_route[:1] * (1.0 - start_blend[:, None])

    steps = base_xy.shape[0]
    alpha = np.zeros((steps,), dtype=np.float32)
    start = max(0, min(int(start_step), max(steps - 1, 0)))
    end = max(start + 1, min(steps, start + max(1, int(duration_steps))))
    if start < steps:
        local = np.linspace(0.0, 1.0, end - start, dtype=np.float32)
        alpha[start:end] = 3.0 * local**2 - 2.0 * local**3
        alpha[end:] = 1.0
    xy = base_xy * (1.0 - alpha[:, None]) + target_xy * alpha[:, None]
    heading = compute_headings(xy, fallback=gt_heading)
    return np.concatenate([xy.astype(np.float32), heading[:, None]], axis=1)


def append_unique(candidates: List[np.ndarray], candidate: np.ndarray, min_mean_distance: float) -> bool:
    if not np.isfinite(candidate).all():
        return False
    for existing in candidates:
        dist = np.linalg.norm(existing[:, :2] - candidate[:, :2], axis=-1).mean()
        if float(dist) < min_mean_distance:
            return False
    candidates.append(candidate.astype(np.float32))
    return True


def build_preview_candidates(
    gt: np.ndarray,
    info: Dict[str, object],
    map_infos: Dict[str, object],
    args: argparse.Namespace,
    rng: random.Random,
    lane_direction_bev: Optional[torch.Tensor] = None,
    lane_direction_mask: Optional[torch.Tensor] = None,
    wrong_direction_enabled: bool = True,
) -> CandidateBatch:
    centerlines = collect_centerlines(info, map_infos, search_radius=args.candidate_route_search_radius)
    base = choose_current_centerline(centerlines, info, command_weight=args.candidate_command_weight)
    gt_heading = gt[:, 2] if gt.shape[1] >= 3 else compute_headings(gt[:, :2])
    if base is None:
        logger.warning("No route centerline found; falling back to GT acceleration only.")
        base = np.concatenate([np.zeros((1, 2), dtype=np.float32), gt[:, :2]], axis=0)
    if bool(getattr(args, "wrong_lane_allow_from_gt", True)) and wrong_direction_enabled:
        gt_wrong_lane_score = candidate_wrong_lane_score(
            gt,
            lane_direction_bev=lane_direction_bev,
            lane_direction_mask=lane_direction_mask,
            threshold=float(args.wrong_lane_cos_threshold),
        )
        if gt_wrong_lane_score < float(args.gt_wrong_lane_allow_threshold):
            wrong_direction_enabled = False

    adjacent = find_adjacent_centerlines(
        base,
        centerlines,
        min_offset_m=args.adjacent_min_offset_m,
        max_offset_m=args.adjacent_max_offset_m,
        max_longitudinal_delta_m=args.adjacent_max_longitudinal_delta_m,
        min_parallel_cos=args.adjacent_min_parallel_cos,
        probe_distance_m=args.adjacent_probe_distance_m,
    )

    gt_distances, gt_speeds = gt_distances_and_speeds(gt, args.trajectory_dt, args.candidate_min_speed)
    candidates: List[np.ndarray] = []
    kinds: List[str] = []
    sides: List[str] = []

    keep_attempts = max(args.num_keep_lane * 4, args.num_keep_lane + 4)
    for _ in range(keep_attempts):
        if len([k for k in kinds if k == "keep"]) >= args.num_keep_lane:
            break
        distances = accelerated_distances(
            gt_distances,
            gt_speeds,
            dt=args.trajectory_dt,
            rng=rng,
            accel_extra_min=args.accel_extra_min,
            accel_extra_max=args.accel_extra_max,
            speed_scale_min=args.speed_scale_min,
            speed_scale_max=args.speed_scale_max,
            max_speed=args.max_speed,
        )
        cand = sample_route_candidate(base, distances, gt_heading)
        if should_reject_wrong_lane_candidate(
            cand,
            lane_direction_bev=lane_direction_bev,
            lane_direction_mask=lane_direction_mask,
            wrong_direction_enabled=wrong_direction_enabled,
            args=args,
        ):
            continue
        if append_unique(candidates, cand, args.candidate_unique_distance):
            kinds.append("keep")
            sides.append("center")

    lane_targets = [(side, line) for side, line in adjacent.items() if line is not None]

    def _append_lane_change_group(
        target_count: int,
        kind: str,
        start_min: int,
        start_max: int,
        duration_min: int,
        duration_max: int,
        attempt_multiplier: int = 8,
    ) -> None:
        if target_count <= 0 or not lane_targets:
            return
        start_lo, start_hi = sorted((int(start_min), int(start_max)))
        duration_lo, duration_hi = sorted((int(duration_min), int(duration_max)))
        attempts = max(int(target_count) * attempt_multiplier, int(target_count) + 8)
        accepted = 0
        for attempt in range(attempts):
            if accepted >= target_count:
                break
            side, target = lane_targets[attempt % len(lane_targets)]
            distances = accelerated_distances(
                gt_distances,
                gt_speeds,
                dt=args.trajectory_dt,
                rng=rng,
                accel_extra_min=args.accel_extra_min,
                accel_extra_max=args.accel_extra_max,
                speed_scale_min=args.speed_scale_min,
                speed_scale_max=args.speed_scale_max,
                max_speed=args.max_speed,
            )
            start_step = rng.randint(start_lo, start_hi)
            duration = rng.randint(max(1, duration_lo), max(1, duration_hi))
            cand = sample_lane_change_candidate(base, target, distances, gt_heading, start_step, duration)
            if should_reject_wrong_lane_candidate(
                cand,
                lane_direction_bev=lane_direction_bev,
                lane_direction_mask=lane_direction_mask,
                wrong_direction_enabled=wrong_direction_enabled,
                args=args,
            ):
                continue
            if append_unique(candidates, cand, args.candidate_unique_distance):
                kinds.append(kind)
                sides.append(side)
                accepted += 1

    target_early_lane_change = min(
        max(int(getattr(args, "num_early_lane_change", 0)), 0),
        max(int(args.num_lane_change), 0),
    )
    target_normal_lane_change = max(int(args.num_lane_change) - target_early_lane_change, 0)
    _append_lane_change_group(
        target_early_lane_change,
        "early_lane_change",
        getattr(args, "early_lane_change_start_min", 0),
        getattr(args, "early_lane_change_start_max", 1),
        getattr(args, "early_lane_change_duration_min", 4),
        getattr(args, "early_lane_change_duration_max", 6),
        attempt_multiplier=10,
    )
    _append_lane_change_group(
        target_normal_lane_change,
        "lane_change",
        args.lane_change_start_min,
        args.lane_change_start_max,
        args.lane_change_duration_min,
        args.lane_change_duration_max,
    )

    # If there are no adjacent lanes nearby, fill remaining slots with keep-lane
    # variants so the visualizer still shows a complete candidate set.
    while len(candidates) < args.num_candidates and len(candidates) < args.num_candidates * 2:
        distances = accelerated_distances(
            gt_distances,
            gt_speeds,
            dt=args.trajectory_dt,
            rng=rng,
            accel_extra_min=args.accel_extra_min,
            accel_extra_max=args.accel_extra_max,
            speed_scale_min=args.speed_scale_min,
            speed_scale_max=args.speed_scale_max,
            max_speed=args.max_speed,
        )
        cand = sample_route_candidate(base, distances, gt_heading)
        if should_reject_wrong_lane_candidate(
            cand,
            lane_direction_bev=lane_direction_bev,
            lane_direction_mask=lane_direction_mask,
            wrong_direction_enabled=wrong_direction_enabled,
            args=args,
        ):
            break
        if append_unique(candidates, cand, args.candidate_unique_distance * 0.5):
            kinds.append("keep_fill")
            sides.append("center")
        else:
            break

    if not candidates:
        candidates = [gt.astype(np.float32)]
        kinds = ["fallback_gt"]
        sides = ["center"]

    out = np.stack(candidates[: args.num_candidates], axis=0).astype(np.float32)
    return CandidateBatch(trajectories=out, kinds=kinds[: out.shape[0]], side=sides[: out.shape[0]])


def gather_bev_values(bev: np.ndarray, xy: np.ndarray, default: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    bev = np.asarray(bev)
    h, w = bev.shape[-2:]
    row = np.round(xy[..., 0] / BEV_PIXEL_SIZE).astype(np.int64)
    col = np.round(xy[..., 1] / BEV_PIXEL_SIZE + (w - 1) / 2.0).astype(np.int64)
    valid = (row >= 0) & (row < h) & (col >= 0) & (col < w)
    row_clip = np.clip(row, 0, h - 1)
    col_clip = np.clip(col, 0, w - 1)
    values = np.full(xy.shape[:-1], default, dtype=np.float32)
    values[valid] = bev[row_clip[valid], col_clip[valid]].astype(np.float32)
    return values, valid


def candidate_wrong_lane_score(
    candidate: np.ndarray,
    lane_direction_bev: Optional[torch.Tensor],
    lane_direction_mask: Optional[torch.Tensor],
    threshold: float,
) -> float:
    if lane_direction_bev is None or lane_direction_mask is None:
        return 1.0
    lane_dir_np = _to_numpy(lane_direction_bev).astype(np.float32)
    lane_mask_np = _to_numpy(lane_direction_mask).astype(bool)
    if lane_dir_np.shape[0] != 2:
        return 1.0
    xy = candidate[:, :2]
    heading = candidate[:, 2] if candidate.shape[-1] >= 3 else compute_headings(xy)
    lane_x, _ = gather_bev_values(lane_dir_np[0], xy)
    lane_y, _ = gather_bev_values(lane_dir_np[1], xy)
    lane_valid, _ = gather_bev_values(lane_mask_np.astype(np.float32), xy)
    lane_valid_bool = lane_valid > 0.5
    if not lane_valid_bool.any():
        return 1.0
    lane_vec = np.stack([lane_x, lane_y], axis=-1)
    lane_unit = lane_vec / np.maximum(np.linalg.norm(lane_vec, axis=-1, keepdims=True), 1e-6)
    heading_vec = np.stack([np.cos(heading), np.sin(heading)], axis=-1)
    cos_sim = np.sum(heading_vec * lane_unit, axis=-1)
    ok = (cos_sim > float(threshold)) | (~lane_valid_bool)
    return float((ok.astype(np.float32) * lane_valid_bool.astype(np.float32)).sum() / max(float(lane_valid_bool.sum()), 1.0))


def should_reject_wrong_lane_candidate(
    candidate: np.ndarray,
    lane_direction_bev: Optional[torch.Tensor],
    lane_direction_mask: Optional[torch.Tensor],
    wrong_direction_enabled: bool,
    args: argparse.Namespace,
) -> bool:
    if not bool(getattr(args, "forbid_wrong_lane_candidates", True)):
        return False
    # wrong_direction_enabled=False means this B2D route is allowed to occupy
    # the opposite lane for obstacle/overtake scenarios.
    if not bool(wrong_direction_enabled):
        return False
    score = candidate_wrong_lane_score(
        candidate,
        lane_direction_bev=lane_direction_bev,
        lane_direction_mask=lane_direction_mask,
        threshold=float(args.wrong_lane_cos_threshold),
    )
    return score < float(args.candidate_wrong_lane_min_score)


def effective_wrong_direction_enabled_for_gt(
    gt: np.ndarray,
    lane_direction_bev: Optional[torch.Tensor],
    lane_direction_mask: Optional[torch.Tensor],
    wrong_direction_enabled: bool,
    args: argparse.Namespace,
) -> bool:
    if not bool(wrong_direction_enabled):
        return False
    if not bool(getattr(args, "wrong_lane_allow_from_gt", True)):
        return True
    gt_wrong_lane_score = candidate_wrong_lane_score(
        gt,
        lane_direction_bev=lane_direction_bev,
        lane_direction_mask=lane_direction_mask,
        threshold=float(args.wrong_lane_cos_threshold),
    )
    return gt_wrong_lane_score >= float(args.gt_wrong_lane_allow_threshold)


def ego_corners_from_trajectory(traj: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    xy = traj[..., :2]
    heading = traj[..., 2] if traj.shape[-1] >= 3 else compute_headings(xy)
    half_l = float(args.ego_half_length)
    half_w = float(args.ego_half_width)
    rear_to_center = float(args.ego_rear_axle_to_center)
    local = np.asarray([[half_l, half_w], [-half_l, half_w], [-half_l, -half_w], [half_l, -half_w]], dtype=np.float32)
    cos_yaw = np.cos(heading)
    sin_yaw = np.sin(heading)
    center = xy + rear_to_center * np.stack([cos_yaw, sin_yaw], axis=-1)
    rot_x = cos_yaw[..., None] * local[:, 0] - sin_yaw[..., None] * local[:, 1]
    rot_y = sin_yaw[..., None] * local[:, 0] + cos_yaw[..., None] * local[:, 1]
    return np.stack([rot_x + center[..., 0:1], rot_y + center[..., 1:2]], axis=-1).astype(np.float32)


def rect_intersects_np(rect_a: np.ndarray, rect_b: np.ndarray) -> bool:
    axes = []
    for rect in (rect_a, rect_b):
        for edge in (rect[1] - rect[0], rect[3] - rect[0]):
            norm = float(np.linalg.norm(edge))
            if norm > 1e-6:
                axes.append(edge / norm)
    for axis in axes:
        proj_a = rect_a @ axis
        proj_b = rect_b @ axis
        if float(proj_a.max()) < float(proj_b.min()) or float(proj_b.max()) < float(proj_a.min()):
            return False
    return True


def pad_style_collision_ttc(
    candidates: np.ndarray,
    gt: np.ndarray,
    future_agent_boxes: torch.Tensor,
    future_agent_boxes_mask: torch.Tensor,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, np.ndarray]:
    boxes = _to_numpy(future_agent_boxes).astype(np.float32)
    boxes_mask = _to_numpy(future_agent_boxes_mask).astype(bool)
    num_candidates, num_steps = candidates.shape[:2]
    no_collision = np.ones((num_candidates,), dtype=np.float32)
    ttc = np.ones((num_candidates,), dtype=np.float32)
    if boxes.size == 0 or boxes_mask.size == 0 or num_steps <= 0:
        return no_collision, ttc

    steps = min(num_steps, boxes.shape[1], boxes_mask.shape[1])
    if steps <= 0:
        return no_collision, ttc

    dt = max(float(args.trajectory_dt), 1e-3)

    def expanded_corners(traj: np.ndarray) -> np.ndarray:
        traj = traj[:steps].astype(np.float32, copy=True)
        xy = traj[:, :2]
        vel = np.concatenate([xy[:1], xy[1:] - xy[:-1]], axis=0) / dt
        expanded = []
        for horizon in (0.0, dt, 2.0 * dt):
            shifted = traj.copy()
            shifted[:, :2] = xy + vel * horizon
            expanded.append(ego_corners_from_trajectory(shifted, args))
        return np.stack(expanded, axis=1)

    gt_corners = expanded_corners(gt)
    gt_collision_by_step = np.zeros((steps,), dtype=bool)
    gt_ttc_by_step = np.zeros((steps,), dtype=bool)
    for step in range(steps):
        for agent_idx in np.flatnonzero(boxes_mask[:, step]):
            agent_box = boxes[agent_idx, step]
            if rect_intersects_np(gt_corners[step, 0], agent_box):
                gt_collision_by_step[step] = True
            if rect_intersects_np(gt_corners[step, 1], agent_box) or rect_intersects_np(gt_corners[step, 2], agent_box):
                gt_ttc_by_step[step] = True

    for cand_idx in range(num_candidates):
        cand_corners = expanded_corners(candidates[cand_idx])
        collision_by_step = np.zeros((steps,), dtype=bool)
        ttc_by_step = np.zeros((steps,), dtype=bool)
        for step in range(steps):
            for agent_idx in np.flatnonzero(boxes_mask[:, step]):
                agent_box = boxes[agent_idx, step]
                if rect_intersects_np(cand_corners[step, 0], agent_box):
                    collision_by_step[step] = True
                if rect_intersects_np(cand_corners[step, 1], agent_box) or rect_intersects_np(cand_corners[step, 2], agent_box):
                    ttc_by_step[step] = True
        collision_by_step &= ~gt_collision_by_step
        ttc_by_step &= ~gt_ttc_by_step
        no_collision[cand_idx] = 0.0 if collision_by_step.any() else 1.0
        ttc[cand_idx] = 0.0 if ttc_by_step.any() else 1.0
    return no_collision, ttc


def score_candidates(
    candidates: np.ndarray,
    gt: np.ndarray,
    feasible_area: torch.Tensor,
    future_agent_boxes: torch.Tensor,
    future_agent_boxes_mask: torch.Tensor,
    lane_direction_bev: torch.Tensor,
    lane_direction_mask: torch.Tensor,
    wrong_direction_enabled: bool,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_candidates, num_steps = candidates.shape[:2]
    xy = candidates[..., :2]
    feasible_np = _to_numpy(feasible_area).astype(bool)
    drivable = np.zeros((num_candidates,), dtype=np.float32)
    for i in range(num_candidates):
        values, _ = gather_bev_values(feasible_np.astype(np.float32), xy[i])
        drivable[i] = float(np.mean(values)) if values.size else 0.0

    ref_progress = max(abs(float(gt[-1, 0])) if gt.shape[0] else 0.0, 1.0)
    progress_raw = xy[:, -1, 0] / ref_progress
    progress_mode = str(getattr(args, "progress_score_mode", "gt_uncapped"))
    if progress_mode == "absolute":
        ref_m = max(float(getattr(args, "progress_reference_m", 20.0)), 1e-3)
        progress = xy[:, -1, 0] / ref_m
    elif progress_mode == "gt_clamped":
        progress = np.clip(progress_raw, 0.0, 1.0)
    elif progress_mode == "gt_uncapped":
        progress = np.maximum(progress_raw, 0.0)
    else:
        raise ValueError(f"Unsupported progress_score_mode: {progress_mode}")
    progress_cap = float(getattr(args, "progress_score_cap", 1.3))
    if progress_cap > 0.0:
        progress = np.minimum(progress, progress_cap)
    progress = progress.astype(np.float32)

    no_collision = np.ones((num_candidates,), dtype=np.float32)
    ttc = np.ones((num_candidates,), dtype=np.float32)
    boxes = _to_numpy(future_agent_boxes).astype(np.float32)
    boxes_mask = _to_numpy(future_agent_boxes_mask).astype(bool)
    if str(getattr(args, "ttc_mode", "pad")).lower() == "pad":
        no_collision, ttc = pad_style_collision_ttc(
            candidates,
            gt,
            future_agent_boxes=future_agent_boxes,
            future_agent_boxes_mask=future_agent_boxes_mask,
            args=args,
        )
    elif boxes.size and boxes_mask.size:
        steps = min(num_steps, boxes.shape[1], boxes_mask.shape[1])
        if steps > 0:
            centers = boxes[:, :steps].mean(axis=-2)
            valid = boxes_mask[:, :steps]
            min_dist = np.full((num_candidates,), np.inf, dtype=np.float32)
            valid_indices = np.argwhere(valid)
            for agent_idx, step in valid_indices:
                dist = np.linalg.norm(xy[:, step] - centers[agent_idx, step][None], axis=-1)
                min_dist = np.minimum(min_dist, dist.astype(np.float32))
            collision_dist = float(args.collision_distance)
            safe_dist = max(float(args.ttc_safe_distance), collision_dist + 1e-3)
            no_collision = (min_dist > collision_dist).astype(np.float32)
            ttc = np.clip((min_dist - collision_dist) / (safe_dist - collision_dist), 0.0, 1.0)
            ttc[~np.isfinite(ttc)] = 1.0

    if candidates.shape[-1] >= 3:
        heading = candidates[..., 2]
    else:
        heading = np.stack([compute_headings(traj[:, :2]) for traj in candidates], axis=0)

    lane_dir_np = _to_numpy(lane_direction_bev).astype(np.float32)
    lane_mask_np = _to_numpy(lane_direction_mask).astype(bool)
    raw_wrong_lane = np.ones((num_candidates,), dtype=np.float32)
    if lane_dir_np.shape[0] == 2:
        wrong_values = []
        for i in range(num_candidates):
            lane_x, _ = gather_bev_values(lane_dir_np[0], xy[i])
            lane_y, _ = gather_bev_values(lane_dir_np[1], xy[i])
            lane_valid, _ = gather_bev_values(lane_mask_np.astype(np.float32), xy[i])
            lane_valid_bool = lane_valid > 0.5
            lane_vec = np.stack([lane_x, lane_y], axis=-1)
            lane_norm = np.linalg.norm(lane_vec, axis=-1, keepdims=True)
            lane_unit = lane_vec / np.maximum(lane_norm, 1e-6)
            heading_vec = np.stack([np.cos(heading[i]), np.sin(heading[i])], axis=-1)
            cos_sim = np.sum(heading_vec * lane_unit, axis=-1)
            ok = (cos_sim > float(args.wrong_lane_cos_threshold)) | (~lane_valid_bool)
            if lane_valid_bool.any():
                wrong_values.append(float((ok.astype(np.float32) * lane_valid_bool.astype(np.float32)).sum() / max(float(lane_valid_bool.sum()), 1.0)))
            else:
                wrong_values.append(1.0)
        raw_wrong_lane = np.asarray(wrong_values, dtype=np.float32)
    allow_wrong_lane = not bool(wrong_direction_enabled)
    if bool(getattr(args, "wrong_lane_allow_from_gt", True)) and lane_dir_np.shape[0] == 2:
        gt_wrong_lane = candidate_wrong_lane_score(
            gt,
            lane_direction_bev=lane_direction_bev,
            lane_direction_mask=lane_direction_mask,
            threshold=float(args.wrong_lane_cos_threshold),
        )
        if gt_wrong_lane < float(args.gt_wrong_lane_allow_threshold):
            allow_wrong_lane = True
    if allow_wrong_lane:
        floor = min(max(float(args.wrong_lane_allowed_floor), 0.0), 1.0)
        wrong_lane = floor + (1.0 - floor) * raw_wrong_lane
    else:
        wrong_lane = np.power(np.clip(raw_wrong_lane, 0.0, 1.0), max(float(args.wrong_lane_strict_power), 1e-3)).astype(np.float32)

    comfort = np.ones((num_candidates,), dtype=np.float32)
    if num_steps >= 3:
        dt = max(float(args.trajectory_dt), 1e-3)
        vel = (xy[:, 1:] - xy[:, :-1]) / dt
        accel = (vel[:, 1:] - vel[:, :-1]) / dt
        max_accel = np.linalg.norm(accel, axis=-1).max(axis=-1)
        yaw_delta = np.arctan2(np.sin(heading[:, 1:] - heading[:, :-1]), np.cos(heading[:, 1:] - heading[:, :-1]))
        yaw_rate = np.abs(yaw_delta / dt).max(axis=-1)
        comfort_mode = str(getattr(args, "comfort_mode", "relative")).lower()
        if comfort_mode in ("relative", "pad_relative", "gt_relative"):
            gt_xy = gt[:num_steps, :2]
            gt_heading = gt[:num_steps, 2] if gt.shape[-1] >= 3 else compute_headings(gt_xy)
            gt_vel = (gt_xy[1:] - gt_xy[:-1]) / dt
            gt_accel = (gt_vel[1:] - gt_vel[:-1]) / dt
            gt_max_accel = float(np.linalg.norm(gt_accel, axis=-1).max()) if gt_accel.size else 0.0
            gt_yaw_delta = np.arctan2(
                np.sin(gt_heading[1:] - gt_heading[:-1]),
                np.cos(gt_heading[1:] - gt_heading[:-1]),
            )
            gt_yaw_rate = float(np.abs(gt_yaw_delta / dt).max()) if gt_yaw_delta.size else 0.0
            accel_ref = max(gt_max_accel, float(args.comfort_accel_floor), 1e-3)
            yaw_ref = max(gt_yaw_rate, float(args.comfort_yaw_rate_floor), 1e-3)
            comfort = np.minimum(
                accel_ref / np.maximum(max_accel, accel_ref),
                yaw_ref / np.maximum(yaw_rate, yaw_ref),
            ).astype(np.float32)
        elif comfort_mode in ("pad", "pad_binary", "binary"):
            gt_xy = gt[:num_steps, :2]
            gt_heading = gt[:num_steps, 2] if gt.shape[-1] >= 3 else compute_headings(gt_xy)
            gt_vel = (gt_xy[1:] - gt_xy[:-1]) / dt
            gt_accel = (gt_vel[1:] - gt_vel[:-1]) / dt
            gt_max_accel = float(np.linalg.norm(gt_accel, axis=-1).max()) if gt_accel.size else 0.0
            gt_yaw_delta = np.arctan2(
                np.sin(gt_heading[1:] - gt_heading[:-1]),
                np.cos(gt_heading[1:] - gt_heading[:-1]),
            )
            gt_yaw_rate = float(np.abs(gt_yaw_delta / dt).max()) if gt_yaw_delta.size else 0.0
            comfort = ((max_accel <= gt_max_accel) & (yaw_rate <= gt_yaw_rate)).astype(np.float32)
        else:
            accel_th = max(float(args.comfort_accel_threshold), 1e-3)
            yaw_th = max(float(args.comfort_yaw_rate_threshold), 1e-3)
            comfort = np.minimum(
                np.clip(1.0 - max_accel / accel_th, 0.0, 1.0),
                np.clip(1.0 - yaw_rate / yaw_th, 0.0, 1.0),
            ).astype(np.float32)

    route_quality = (5.0 * progress + 5.0 * ttc + 2.0 * comfort) / 12.0
    scores = np.maximum(no_collision * drivable * wrong_lane * route_quality, 0.0).astype(np.float32)
    score_cap = float(getattr(args, "score_cap", 0.0))
    if score_cap > 0.0:
        scores = np.minimum(scores, score_cap).astype(np.float32)
    components = np.stack([no_collision, drivable, progress, ttc, comfort, wrong_lane], axis=-1).astype(np.float32)
    return scores, components, progress_raw.astype(np.float32)


def select_topk(
    scores: np.ndarray,
    progress_raw: np.ndarray,
    components: np.ndarray,
    gt_score: float,
    topk: int,
    eps: float,
    strict_above_gt: bool,
    require_wrong_lane: bool,
    wrong_lane_min_score: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if scores.size == 0 or topk <= 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=bool)
    finite = np.isfinite(scores)
    if strict_above_gt:
        eligible = finite & (scores > gt_score + eps)
    else:
        eligible = finite & (scores >= gt_score - eps)
    lane_ok = np.ones_like(eligible, dtype=bool)
    if require_wrong_lane and components.shape[-1] > 5:
        lane_ok = components[:, 5] >= float(wrong_lane_min_score)
        eligible = eligible & lane_ok
    # Sort by score, then raw progress, then comfort, then wrong-lane.
    order = np.lexsort(
        (
            components[:, 5],
            components[:, 4],
            progress_raw,
            scores,
        )
    )[::-1]
    selected = [idx for idx in order if eligible[idx]][:topk]
    if len(selected) < topk:
        for idx in order:
            if idx not in selected and finite[idx] and lane_ok[idx]:
                selected.append(int(idx))
            if len(selected) >= topk:
                break
    selected_idx = np.asarray(selected[:topk], dtype=np.int64)
    selected_mask = eligible[selected_idx] if selected_idx.size else np.zeros((0,), dtype=bool)
    return selected_idx, selected_mask.astype(bool)


def build_scored_candidates(
    batch: CandidateBatch,
    gt: np.ndarray,
    feasible_area: torch.Tensor,
    future_agent_boxes: torch.Tensor,
    future_agent_boxes_mask: torch.Tensor,
    lane_direction_bev: torch.Tensor,
    lane_direction_mask: torch.Tensor,
    wrong_direction_enabled: bool,
    args: argparse.Namespace,
) -> ScoredCandidates:
    effective_wrong_direction_enabled = effective_wrong_direction_enabled_for_gt(
        gt,
        lane_direction_bev=lane_direction_bev,
        lane_direction_mask=lane_direction_mask,
        wrong_direction_enabled=wrong_direction_enabled,
        args=args,
    )
    scores, components, progress_raw = score_candidates(
        batch.trajectories,
        gt,
        feasible_area,
        future_agent_boxes,
        future_agent_boxes_mask,
        lane_direction_bev,
        lane_direction_mask,
        effective_wrong_direction_enabled,
        args,
    )
    gt_scores, gt_components, _ = score_candidates(
        gt[None],
        gt,
        feasible_area,
        future_agent_boxes,
        future_agent_boxes_mask,
        lane_direction_bev,
        lane_direction_mask,
        effective_wrong_direction_enabled,
        args,
    )
    gt_score = float(gt_scores[0]) if gt_scores.size else float("nan")
    selected_idx, selected_mask = select_topk(
        scores,
        progress_raw,
        components,
        gt_score=gt_score,
        topk=args.topk,
        eps=args.gt_score_eps,
        strict_above_gt=args.strict_above_gt,
        require_wrong_lane=bool(args.forbid_wrong_lane_candidates) and bool(effective_wrong_direction_enabled),
        wrong_lane_min_score=float(args.candidate_wrong_lane_min_score),
    )
    return ScoredCandidates(
        scores=scores,
        components=components,
        progress_raw=progress_raw,
        gt_score=gt_score,
        gt_components=gt_components[0] if gt_components.size else np.full((6,), np.nan, dtype=np.float32),
        selected_idx=selected_idx,
        selected_mask=selected_mask,
    )


def plot_candidates(
    split: str,
    token: str,
    idx: int,
    bev_semantic_map: np.ndarray,
    gt: np.ndarray,
    batch: CandidateBatch,
    scored: ScoredCandidates,
    output_dir: Path,
    args: argparse.Namespace,
) -> Path:
    class_map = bev_semantic_map.astype(np.int64)
    height, width = class_map.shape
    selected_set = set(int(i) for i in scored.selected_idx.tolist())

    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 0.85])
    ax = fig.add_subplot(gs[0, 0])
    ax_info = fig.add_subplot(gs[0, 1])

    cmap = ListedColormap(SEMANTIC_COLORS)
    ax.imshow(class_map, origin="upper", interpolation="nearest", cmap=cmap, vmin=0, vmax=len(SEMANTIC_COLORS) - 1)

    for cand_idx, traj in enumerate(batch.trajectories):
        kind = batch.kinds[cand_idx] if cand_idx < len(batch.kinds) else "candidate"
        side = batch.side[cand_idx] if cand_idx < len(batch.side) else ""
        if kind.startswith("keep"):
            color = "#2563eb"
        elif side == "left":
            color = "#f97316"
        elif side == "right":
            color = "#a855f7"
        else:
            color = "#f59e0b"
        linewidth = 2.7 if cand_idx in selected_set else 1.0
        alpha = 0.95 if cand_idx in selected_set else 0.32
        row, col = _xy_to_image(traj[:, :2], BEV_PIXEL_SIZE, width)
        ax.plot(col, row, color=color, linewidth=linewidth, alpha=alpha)
        if cand_idx in selected_set:
            ax.scatter(col[-1], row[-1], s=36, color=color, marker="x", linewidths=1.4)
            ax.text(
                col[-1],
                row[-1],
                f"{cand_idx}",
                color="white",
                fontsize=8,
                ha="center",
                va="center",
                bbox={"boxstyle": "round,pad=0.16", "fc": "black", "ec": "none", "alpha": 0.70},
            )

    gt_row, gt_col = _xy_to_image(gt[:, :2], BEV_PIXEL_SIZE, width)
    ax.plot(gt_col, gt_row, color="black", linewidth=2.8, label=f"GT {scored.gt_score:.3f}")
    ax.scatter([(width - 1) / 2.0], [0], s=52, color="white", marker="x", linewidths=1.5)
    ax.scatter(gt_col[-1], gt_row[-1], s=42, color="black", marker="x")
    ax.set_xlim([-0.5, width - 0.5])
    ax.set_ylim([height - 0.5, -0.5])
    ax.axis("off")
    ax.set_title(
        f"{split}/{token}\nraw={batch.trajectories.shape[0]}, topk={len(scored.selected_idx)}, idx={idx}",
        fontsize=10,
    )

    ax_info.axis("off")
    keep_count = sum(1 for kind in batch.kinds if kind.startswith("keep"))
    early_lc_count = sum(1 for kind in batch.kinds if kind == "early_lane_change")
    lc_count = sum(1 for kind in batch.kinds if kind == "lane_change")
    selected_rows = []
    for rank, cand_idx in enumerate(scored.selected_idx.tolist()):
        active = "Y" if rank < scored.selected_mask.size and bool(scored.selected_mask[rank]) else "fallback"
        comps = scored.components[cand_idx]
        selected_rows.append(
            f"{rank + 1:>2}. #{cand_idx:<2} {batch.kinds[cand_idx]:<11} {batch.side[cand_idx]:<6} "
            f"s={scored.scores[cand_idx]:.3f} raw_prog={scored.progress_raw[cand_idx]:.2f} {active}"
        )
        selected_rows.append(
            "    "
            + " ".join(f"{name[:4]}={value:.2f}" for name, value in zip(COMPONENT_NAMES, comps))
        )
    if not selected_rows:
        finite = np.isfinite(scored.scores)
        if bool(args.forbid_wrong_lane_candidates) and scored.components.ndim == 2 and scored.components.shape[-1] > 5:
            lane_ok = scored.components[:, 5] >= float(args.candidate_wrong_lane_min_score)
        else:
            lane_ok = np.ones_like(finite, dtype=bool)
        selected_rows.append(
            f"    none: finite={int(finite.sum())}/{finite.size}, "
            f"lane_ok={int((finite & lane_ok).sum())}/{finite.size}"
        )
        if finite.any():
            order = np.argsort(np.nan_to_num(scored.scores, nan=-np.inf))[::-1]
            for cand_idx in [int(i) for i in order if finite[int(i)]][:5]:
                comps = scored.components[cand_idx]
                selected_rows.append(
                    f"    best #{cand_idx:<2} {batch.kinds[cand_idx]:<17} {batch.side[cand_idx]:<6} "
                    f"s={scored.scores[cand_idx]:.3f} raw_prog={scored.progress_raw[cand_idx]:.2f}"
                )
                selected_rows.append(
                    "      "
                    + " ".join(f"{name[:4]}={value:.2f}" for name, value in zip(COMPONENT_NAMES, comps))
                )
    text = "\n".join(
        [
            f"GT score: {scored.gt_score:.3f}",
            "GT components: " + " ".join(f"{name[:4]}={value:.2f}" for name, value in zip(COMPONENT_NAMES, scored.gt_components)),
            f"candidate count: keep={keep_count}, early_lc={early_lc_count}, lane_change={lc_count}, total={batch.trajectories.shape[0]}",
            f"selection rule: {'score > GT + eps' if args.strict_above_gt else 'score >= GT - eps'}, eps={args.gt_score_eps}",
            f"progress mode: {args.progress_score_mode}, progress cap={args.progress_score_cap}, score cap={args.score_cap}",
            f"ttc mode: {args.ttc_mode}, comfort mode: {args.comfort_mode}",
            f"wrong-lane filter: {'on' if args.forbid_wrong_lane_candidates else 'off'}, route wrong-direction check={'on' if args.route_wrong_direction_enabled else 'off'}, min={args.candidate_wrong_lane_min_score}",
            f"wrong-lane score: allow_from_gt={args.wrong_lane_allow_from_gt}, gt_thr={args.gt_wrong_lane_allow_threshold}, allowed_floor={args.wrong_lane_allowed_floor}, strict_power={args.wrong_lane_strict_power}",
            "",
            "Top candidates:",
            *selected_rows,
            "",
            "Colors: blue=keep lane, orange=left lane-change, purple=right lane-change, black=GT",
        ]
    )
    ax_info.text(0.0, 1.0, text, ha="left", va="top", family="monospace", fontsize=9)
    fig.tight_layout()

    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    out_path = split_dir / f"{idx:08d}__{_safe_name(token)}.png"
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def iter_indices(
    infos: Sequence[Dict[str, object]],
    split: str,
    args: argparse.Namespace,
) -> Iterable[int]:
    if args.indices:
        for idx in args.indices:
            if 0 <= idx < len(infos):
                yield idx
        return
    if args.tokens:
        token_set = set(args.tokens)
        for idx, info in enumerate(infos):
            if safe_token(split, idx, info) in token_set:
                yield idx
        return

    yielded = 0
    for idx in range(max(0, args.start_index), len(infos), max(1, args.stride)):
        if args.num_samples > 0 and yielded >= args.num_samples:
            break
        yield idx
        yielded += 1


def process_split(split: str, args: argparse.Namespace) -> None:
    infos_root = Path(args.infos_root)
    info_path = infos_root / f"b2d_infos_{split}.pkl"
    map_path = infos_root / "b2d_map_infos.pkl"
    if not info_path.is_file():
        raise FileNotFoundError(f"Missing B2D infos: {info_path}")
    if not map_path.is_file():
        raise FileNotFoundError(f"Missing B2D map infos: {map_path}")

    infos = load_pickle(info_path)
    map_infos = load_pickle(map_path)
    output_dir = Path(args.output_dir)
    rng = random.Random(args.seed + (0 if split == "train" else 1000003))
    saved = 0
    skipped = 0

    indices = list(iter_indices(infos, split, args))
    for idx in tqdm(indices, desc=f"preview {split}"):
        info = infos[idx]
        if bool(args.skip_candidates_near_traffic_light) and is_near_traffic_light_trigger(
            info,
            map_infos,
            distance_m=float(args.traffic_light_skip_distance_m),
            include_stop_sign=bool(args.traffic_light_skip_include_stop_sign),
        ):
            skipped += 1
            continue
        gt = build_future_trajectory(
            infos,
            idx=idx,
            sample_interval=args.sample_interval,
            future_frames=args.future_frames,
        )
        if gt is None:
            skipped += 1
            continue
        future_idx = idx + args.future_frames * args.sample_interval
        future_info = infos[future_idx] if future_idx < len(infos) and is_same_log(info, infos[future_idx]) else info

        static_bev = build_static_map_bev(info, map_infos)
        bev_semantic_map = overlay_agents_on_bev(static_bev, source_info=info, current_info=info)
        feasible_area_mask, _ = compute_feasible_masks(bev_semantic_map)
        future_agent_boxes, future_agent_boxes_mask = build_future_agent_boxes(
            infos,
            idx=idx,
            sample_interval=args.sample_interval,
            future_frames=args.future_frames,
            max_future_agents=args.max_future_agents,
            range_m=args.future_agent_range,
        )
        lane_direction_bev, lane_direction_mask = build_lane_direction_bev(
            info,
            map_infos,
            radius_m=args.lane_direction_radius_m,
            search_radius=args.candidate_route_search_radius,
        )
        wrong_direction_enabled, _ = infer_wrong_direction_enabled(info, args)
        token = safe_token(split, idx, info)

        sample_rng = random.Random(rng.randint(0, 2**31 - 1) + idx)
        args.route_wrong_direction_enabled = bool(wrong_direction_enabled)
        batch = build_preview_candidates(
            _to_numpy(gt).astype(np.float32),
            info,
            map_infos,
            args,
            sample_rng,
            lane_direction_bev=lane_direction_bev,
            lane_direction_mask=lane_direction_mask,
            wrong_direction_enabled=wrong_direction_enabled,
        )
        scored = build_scored_candidates(
            batch=batch,
            gt=_to_numpy(gt).astype(np.float32),
            feasible_area=feasible_area_mask,
            future_agent_boxes=future_agent_boxes,
            future_agent_boxes_mask=future_agent_boxes_mask,
            lane_direction_bev=lane_direction_bev,
            lane_direction_mask=lane_direction_mask,
            wrong_direction_enabled=wrong_direction_enabled,
            args=args,
        )
        plot_candidates(
            split=split,
            token=token,
            idx=idx,
            bev_semantic_map=bev_semantic_map,
            gt=_to_numpy(gt).astype(np.float32),
            batch=batch,
            scored=scored,
            output_dir=output_dir,
            args=args,
        )
        saved += 1
    logger.info("%s: saved=%d skipped=%d output_dir=%s", split, saved, skipped, output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--infos-root", required=True)
    parser.add_argument("--output-dir", default="b2d_candidate_previews")
    parser.add_argument("--splits", nargs="+", default=["train"], choices=["train", "val"])
    parser.add_argument("--tokens", nargs="*", default=None)
    parser.add_argument("--indices", nargs="*", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--sample-interval", type=int, default=5)
    parser.add_argument("--future-frames", type=int, default=6)
    parser.add_argument("--trajectory-dt", type=float, default=0.5)
    parser.add_argument("--max-future-agents", type=int, default=64)
    parser.add_argument("--future-agent-range", type=float, default=48.0)
    parser.add_argument("--skip-candidates-near-traffic-light", action="store_true")
    parser.add_argument("--traffic-light-skip-distance-m", type=float, default=12.0)
    parser.add_argument("--traffic-light-skip-include-stop-sign", action="store_true")

    parser.add_argument("--num-candidates", type=int, default=40)
    parser.add_argument("--num-keep-lane", type=int, default=20)
    parser.add_argument("--num-lane-change", type=int, default=20)
    parser.add_argument(
        "--num-early-lane-change",
        type=int,
        default=6,
        help="Subset of num-lane-change that starts changing lane at the first/second future step.",
    )
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--candidate-route-search-radius", type=float, default=80.0)
    parser.add_argument("--candidate-command-weight", type=float, default=8.0)
    parser.add_argument("--candidate-min-speed", type=float, default=1.0)
    parser.add_argument("--candidate-unique-distance", type=float, default=0.45)
    parser.add_argument("--speed-scale-min", type=float, default=1.0)
    parser.add_argument("--speed-scale-max", type=float, default=1.25)
    parser.add_argument("--accel-extra-min", type=float, default=0.0)
    parser.add_argument("--accel-extra-max", type=float, default=1.2)
    parser.add_argument("--max-speed", type=float, default=18.0)

    parser.add_argument("--lane-change-start-min", type=int, default=1)
    parser.add_argument("--lane-change-start-max", type=int, default=4)
    parser.add_argument("--lane-change-duration-min", type=int, default=3)
    parser.add_argument("--lane-change-duration-max", type=int, default=6)
    parser.add_argument("--early-lane-change-start-min", type=int, default=0)
    parser.add_argument("--early-lane-change-start-max", type=int, default=1)
    parser.add_argument("--early-lane-change-duration-min", type=int, default=4)
    parser.add_argument("--early-lane-change-duration-max", type=int, default=6)
    parser.add_argument("--adjacent-min-offset-m", type=float, default=2.0)
    parser.add_argument("--adjacent-max-offset-m", type=float, default=5.5)
    parser.add_argument("--adjacent-max-longitudinal-delta-m", type=float, default=8.0)
    parser.add_argument("--adjacent-min-parallel-cos", type=float, default=0.70)
    parser.add_argument("--adjacent-probe-distance-m", type=float, default=8.0)

    parser.add_argument(
        "--progress-score-mode",
        choices=["gt_uncapped", "gt_clamped", "absolute"],
        default="gt_uncapped",
        help="gt_uncapped lets trajectories beyond GT score above 1; absolute uses final x / progress-reference-m.",
    )
    parser.add_argument("--progress-score-cap", type=float, default=1.3, help="<=0 disables the progress component cap.")
    parser.add_argument("--progress-reference-m", type=float, default=20.0)
    parser.add_argument("--score-cap", type=float, default=0.0, help="<=0 keeps selection score uncapped.")
    parser.add_argument("--gt-score-eps", type=float, default=0.02)
    parser.add_argument("--strict-above-gt", action="store_true")
    parser.add_argument("--ttc-mode", choices=["pad", "distance"], default="pad")
    parser.add_argument("--collision-distance", type=float, default=2.5)
    parser.add_argument("--ttc-safe-distance", type=float, default=8.0)
    parser.add_argument("--wrong-lane-cos-threshold", type=float, default=0.0)
    parser.add_argument("--forbid-wrong-lane-candidates", action="store_true", default=True)
    parser.add_argument("--allow-wrong-lane-candidates", dest="forbid_wrong_lane_candidates", action="store_false")
    parser.add_argument("--candidate-wrong-lane-min-score", type=float, default=0.80)
    parser.add_argument("--wrong-lane-allow-from-gt", action="store_true", default=True)
    parser.add_argument("--no-wrong-lane-allow-from-gt", dest="wrong_lane_allow_from_gt", action="store_false")
    parser.add_argument("--gt-wrong-lane-allow-threshold", type=float, default=0.95)
    parser.add_argument("--wrong-lane-allowed-floor", type=float, default=0.70)
    parser.add_argument("--wrong-lane-strict-power", type=float, default=2.0)
    parser.add_argument("--ego-half-length", type=float, default=2.042)
    parser.add_argument("--ego-half-width", type=float, default=0.925)
    parser.add_argument("--ego-rear-axle-to-center", type=float, default=0.39)
    parser.add_argument("--comfort-mode", choices=["relative", "pad_binary", "threshold"], default="relative")
    parser.add_argument("--comfort-accel-floor", type=float, default=0.5)
    parser.add_argument("--comfort-yaw-rate-floor", type=float, default=0.1)
    parser.add_argument("--comfort-accel-threshold", type=float, default=4.5)
    parser.add_argument("--comfort-yaw-rate-threshold", type=float, default=1.2)
    parser.add_argument("--lane-direction-radius-m", type=float, default=1.0)
    parser.add_argument("--b2d-routes-file", type=Path, default=None)
    parser.add_argument(
        "--wrong-direction-disabled-scenarios",
        nargs="*",
        default=list(B2D_WRONG_DIRECTION_DISABLED_SCENARIOS),
    )
    parser.add_argument("--dpi", type=int, default=140)
    args = parser.parse_args()

    if args.num_keep_lane + args.num_lane_change != args.num_candidates:
        logger.warning(
            "num_keep_lane + num_lane_change != num_candidates; requested %d + %d vs %d. "
            "The script will generate up to num_candidates trajectories.",
            args.num_keep_lane,
            args.num_lane_change,
            args.num_candidates,
        )
    if args.num_early_lane_change > args.num_lane_change:
        logger.warning(
            "num_early_lane_change=%d is larger than num_lane_change=%d; clamping early lane-change count.",
            args.num_early_lane_change,
            args.num_lane_change,
        )
        args.num_early_lane_change = args.num_lane_change
    args.route_scenario_by_id = load_route_scenarios(args.b2d_routes_file)
    return args


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = parse_args()
    for split in args.splits:
        process_split(split, args)


if __name__ == "__main__":
    main()
