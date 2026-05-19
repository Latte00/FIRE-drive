import argparse
import logging
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

from navsim.agents.diffusiondrive.modules.pdm_supervision import (
    PDMSupervision,
    PDMScoreConfig,
)
from navsim.common.dataloader import MetricCacheLoader
from navsim.common.dataclasses import Trajectory
from navsim.evaluate.pdm_score import get_trajectory_as_array, transform_trajectory
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import (
    PDMSimulator,
)
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import (
    PDMScorer,
)
from navsim.planning.training.dataset import (
    load_feature_target_from_pickle,
    dump_feature_target_to_pickle,
)
from nuplan.planning.simulation.trajectory.trajectory_sampling import (
    TrajectorySampling,
)

try:
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import (
        MultiMetricIndex,
        WeightedMetricIndex,
    )
except ImportError:  # pragma: no cover - optional in some environments
    MultiMetricIndex = None
    WeightedMetricIndex = None
try:
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import (
        PDMScorerConfig,
    )
except ImportError:  # pragma: no cover - optional in some environments
    PDMScorerConfig = None

logger = logging.getLogger(__name__)


def _to_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _to_numpy_or_none(value):
    if value is None:
        return None
    try:
        return _to_numpy(value)
    except Exception:
        return None


def _cached_pdm_scores_are_invalid(data: Dict[str, object]) -> bool:
    scores = _to_numpy_or_none(data.get("pdm_score_targets"))
    if scores is None:
        return True
    try:
        scores_flat = np.asarray(scores, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        return True
    if scores_flat.size == 0:
        return True

    mask = _to_numpy_or_none(data.get("trajectory_candidates_mask"))
    if mask is not None:
        try:
            mask_arr = np.asarray(mask, dtype=bool)
        except (TypeError, ValueError):
            mask_arr = None
        scores_squeezed = np.squeeze(np.asarray(scores, dtype=np.float32))
        mask_squeezed = np.squeeze(mask_arr) if mask_arr is not None else None
        if mask_squeezed is not None and mask_squeezed.shape == scores_squeezed.shape:
            scores_to_check = scores_squeezed.reshape(-1)[mask_squeezed.reshape(-1)]
        elif mask_arr is not None and mask_arr.size == scores_flat.size:
            scores_to_check = scores_flat[mask_arr.reshape(-1)]
        else:
            scores_to_check = scores_flat
    else:
        scores_to_check = scores_flat

    if scores_to_check.size > 0 and not np.isfinite(scores_to_check).all():
        return True

    gt_score = _to_numpy_or_none(data.get("gt_pdm_score"))
    if gt_score is None:
        return True
    try:
        gt_flat = np.asarray(gt_score, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        return True
    return gt_flat.size == 0 or not np.isfinite(gt_flat).all()


def _bev_to_class_map(bev: np.ndarray) -> np.ndarray:
    if bev.ndim == 2:
        return bev.astype(np.int64)
    if bev.ndim == 3:
        if bev.shape[0] <= 16:
            return bev.argmax(axis=0).astype(np.int64)
        if bev.shape[-1] <= 16:
            return bev.argmax(axis=-1).astype(np.int64)
    raise ValueError(f"Unsupported BEV shape {bev.shape}")


def _build_drivable_mask(
    bev: np.ndarray,
    road_label: int,
    centerline_label: int,
) -> np.ndarray:
    class_map = _bev_to_class_map(bev)
    return (class_map == road_label) | (class_map == centerline_label)


def _build_centerline_mask(bev: np.ndarray, centerline_label: int) -> np.ndarray:
    class_map = _bev_to_class_map(bev)
    return class_map == centerline_label


def _xy_to_indices(
    xy: np.ndarray,
    pixel_size: float,
    width: int,
) -> Tuple[np.ndarray, np.ndarray]:
    row = np.rint(xy[:, 0] / pixel_size).astype(np.int64)
    col = np.rint(xy[:, 1] / pixel_size + (width - 1) / 2.0).astype(np.int64)
    return row, col


def _mask_ratio(
    traj_xy: np.ndarray,
    mask_current: np.ndarray,
    mask_future: Optional[np.ndarray],
    pixel_size: float,
    strict_steps: int,
) -> float:
    height, width = mask_current.shape
    rows, cols = _xy_to_indices(traj_xy, pixel_size, width)
    in_bounds = (
        (rows >= 0)
        & (rows < height)
        & (cols >= 0)
        & (cols < width)
    )
    ok = np.zeros(traj_xy.shape[0], dtype=bool)
    for i in range(traj_xy.shape[0]):
        if not in_bounds[i]:
            continue
        use_mask = mask_current
        if mask_future is not None and i >= strict_steps:
            use_mask = mask_future
        ok[i] = bool(use_mask[rows[i], cols[i]])
    return float(ok.mean()) if ok.size else 0.0


def _resample_traj(
    traj: np.ndarray,
    src_dt: float,
    dst_dt: float,
    dst_steps: int,
) -> np.ndarray:
    if traj.ndim != 2 or traj.shape[1] < 2:
        raise ValueError(f"Invalid traj shape {traj.shape}")
    src_steps = traj.shape[0]
    src_t = np.arange(src_steps, dtype=np.float32) * src_dt
    dst_t = np.arange(dst_steps, dtype=np.float32) * dst_dt
    xy = traj[:, :2]
    x = np.interp(dst_t, src_t, xy[:, 0], left=xy[0, 0], right=xy[-1, 0])
    y = np.interp(dst_t, src_t, xy[:, 1], left=xy[0, 1], right=xy[-1, 1])
    if traj.shape[1] >= 3:
        heading = np.unwrap(traj[:, 2])
        yaw = np.interp(dst_t, src_t, heading, left=heading[0], right=heading[-1])
    else:
        yaw = np.zeros_like(x)
    return np.stack([x, y, yaw], axis=-1).astype(np.float32)


def _extract_components_from_scorer(
    scorer: PDMScorer, num_modes: int, skip_first: bool
) -> Optional[np.ndarray]:
    if MultiMetricIndex is None or WeightedMetricIndex is None:
        return None
    no_collision = np.asarray(
        scorer._multi_metrics[MultiMetricIndex.NO_COLLISION], dtype=np.float32
    ).reshape(-1)
    drivable = np.asarray(
        scorer._multi_metrics[MultiMetricIndex.DRIVABLE_AREA], dtype=np.float32
    ).reshape(-1)
    ego_progress = np.asarray(
        scorer._weighted_metrics[WeightedMetricIndex.PROGRESS], dtype=np.float32
    ).reshape(-1)
    ttc = np.asarray(
        scorer._weighted_metrics[WeightedMetricIndex.TTC], dtype=np.float32
    ).reshape(-1)
    comfort = np.asarray(
        scorer._weighted_metrics[WeightedMetricIndex.COMFORTABLE], dtype=np.float32
    ).reshape(-1)
    driving_dir = np.asarray(
        scorer._weighted_metrics[WeightedMetricIndex.DRIVING_DIRECTION],
        dtype=np.float32,
    ).reshape(-1)
    components = np.stack(
        [no_collision, drivable, ego_progress, ttc, comfort, driving_dir],
        axis=-1,
    )
    if skip_first:
        components = components[1:]
    if components.shape[0] != num_modes:
        components = components[:num_modes]
    return components


def _score_with_baseline(
    metric_cache,
    trajectories: np.ndarray,
    traj_sampling: TrajectorySampling,
    pdm_sampling: TrajectorySampling,
    simulator: PDMSimulator,
    scorer: PDMScorer,
    return_components: bool,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if trajectories.size == 0:
        return np.zeros((0,), dtype=np.float32), None
    initial_ego_state = metric_cache.ego_state
    pdm_states = get_trajectory_as_array(
        metric_cache.trajectory, pdm_sampling, initial_ego_state.time_point
    )
    pred_states = []
    for traj_local in trajectories:
        traj = Trajectory(
            traj_local.astype(np.float32), trajectory_sampling=traj_sampling
        )
        pred_traj = transform_trajectory(traj, initial_ego_state)
        pred_states.append(
            get_trajectory_as_array(
                pred_traj, pdm_sampling, initial_ego_state.time_point
            )
        )
    pred_states = np.stack(pred_states, axis=0)
    trajectory_states = np.concatenate(
        [pdm_states[None, ...], pred_states], axis=0
    )
    simulated_states = simulator.simulate_proposals(
        trajectory_states, initial_ego_state
    )
    pdm_progress = getattr(metric_cache, "pdm_progress", None)
    try:
        scores = scorer.score_proposals(
            simulated_states,
            metric_cache.observation,
            metric_cache.centerline,
            metric_cache.route_lane_ids,
            metric_cache.drivable_area_map,
            pdm_progress,
        )
    except TypeError:
        scores = scorer.score_proposals(
            simulated_states,
            metric_cache.observation,
            metric_cache.centerline,
            metric_cache.route_lane_ids,
            metric_cache.drivable_area_map,
        )
    scores = np.asarray(scores, dtype=np.float32)
    scores = scores[1:]
    components = None
    if return_components:
        components = _extract_components_from_scorer(
            scorer, scores.size, skip_first=True
        )
    return scores, components


def _simulate_bicycle(
    num_steps: int,
    dt: float,
    v0: float,
    wheel_base: float,
    steer_seq: np.ndarray,
    accel_seq: np.ndarray,
) -> np.ndarray:
    x = 0.0
    y = 0.0
    yaw = 0.0
    v = max(float(v0), 0.0)
    traj = np.zeros((num_steps, 3), dtype=np.float32)
    for i in range(num_steps):
        v = max(0.0, v + float(accel_seq[i]) * dt)
        steer = float(steer_seq[i])
        yaw += v / max(wheel_base, 1e-3) * math.tan(steer) * dt
        x += v * math.cos(yaw) * dt
        y += v * math.sin(yaw) * dt
        traj[i, 0] = x
        traj[i, 1] = y
        traj[i, 2] = yaw
    return traj


def _accelerate_gt_candidates(
    traj_gt: np.ndarray,
    dt: float,
    factors: List[float],
) -> np.ndarray:
    if traj_gt.ndim != 2 or traj_gt.shape[0] == 0:
        return np.zeros((0, 0, 3), dtype=np.float32)
    if not factors:
        return np.zeros((0, traj_gt.shape[0], 3), dtype=np.float32)
    base_xy = traj_gt[:, :2].astype(np.float32)
    num_steps = base_xy.shape[0]
    times = (np.arange(num_steps, dtype=np.float32) + 1) * float(dt)
    last_time = float(times[-1])
    if num_steps >= 2:
        vel = (base_xy[-1] - base_xy[-2]) / max(float(dt), 1e-3)
    else:
        vel = np.array([0.0, 0.0], dtype=np.float32)
    candidates = []
    for factor in factors:
        if factor <= 0:
            continue
        t_new = times * float(factor)
        xy = np.zeros((num_steps, 2), dtype=np.float32)
        in_range = t_new <= last_time
        if in_range.any():
            xy[in_range, 0] = np.interp(
                t_new[in_range], times, base_xy[:, 0]
            ).astype(np.float32)
            xy[in_range, 1] = np.interp(
                t_new[in_range], times, base_xy[:, 1]
            ).astype(np.float32)
        if (~in_range).any():
            extra = (t_new[~in_range] - last_time).astype(np.float32)
            xy[~in_range] = base_xy[-1] + extra[:, None] * vel[None, :]
        deltas = np.diff(xy, axis=0, prepend=xy[:1])
        yaw = np.arctan2(deltas[:, 1], deltas[:, 0]).astype(np.float32)
        cand = np.zeros((num_steps, 3), dtype=np.float32)
        cand[:, :2] = xy
        cand[:, 2] = yaw
        candidates.append(cand)
    if not candidates:
        return np.zeros((0, traj_gt.shape[0], 3), dtype=np.float32)
    return np.stack(candidates, axis=0)


def _count_sign_flips(values: np.ndarray, eps: float) -> int:
    flips = 0
    last = 0
    for v in values:
        if abs(float(v)) <= eps:
            continue
        sign = 1 if v > 0 else -1
        if last == 0:
            last = sign
        elif sign != last:
            flips += 1
            last = sign
    return flips


def _filter_random_candidates(
    candidates: np.ndarray,
    max_dyaw: float,
    max_flips: int,
    yaw_eps: float,
) -> np.ndarray:
    if candidates.size == 0:
        return candidates
    keep = []
    for traj in candidates:
        yaw = np.unwrap(traj[:, 2])
        dyaw = np.diff(yaw)
        if max_dyaw > 0 and np.max(np.abs(dyaw)) > max_dyaw:
            keep.append(False)
            continue
        if max_flips >= 0:
            flips = _count_sign_flips(dyaw, yaw_eps)
            if flips > max_flips:
                keep.append(False)
                continue
        keep.append(True)
    keep = np.array(keep, dtype=bool)
    return candidates[keep]


def _sample_candidates(
    rng: random.Random,
    num_candidates: int,
    num_steps: int,
    dt: float,
    v0: float,
    wheel_base: float,
    max_steer_deg: float,
    min_accel: float,
    max_accel: float,
) -> np.ndarray:
    candidates = np.zeros((num_candidates, num_steps, 3), dtype=np.float32)
    max_steer = math.radians(max_steer_deg)
    for i in range(num_candidates):
        split = rng.randint(1, max(1, num_steps - 1))
        steer_a = rng.uniform(-max_steer, max_steer)
        steer_b = rng.uniform(-max_steer, max_steer)
        accel_a = rng.uniform(min_accel, max_accel)
        accel_b = rng.uniform(min_accel, max_accel)
        if rng.random() < 0.5:
            steer_seq = np.full(num_steps, steer_a, dtype=np.float32)
        else:
            steer_seq = np.concatenate(
                [
                    np.full(split, steer_a, dtype=np.float32),
                    np.full(num_steps - split, steer_b, dtype=np.float32),
                ],
                axis=0,
            )
        if rng.random() < 0.5:
            accel_seq = np.full(num_steps, accel_a, dtype=np.float32)
        else:
            accel_seq = np.concatenate(
                [
                    np.full(split, accel_a, dtype=np.float32),
                    np.full(num_steps - split, accel_b, dtype=np.float32),
                ],
                axis=0,
            )
        candidates[i] = _simulate_bicycle(
            num_steps=num_steps,
            dt=dt,
            v0=v0,
            wheel_base=wheel_base,
            steer_seq=steer_seq,
            accel_seq=accel_seq,
        )
    return candidates


def _row_cluster_centers(
    centerline_mask: np.ndarray,
    cluster_gap: int = 2,
) -> Dict[int, np.ndarray]:
    rows, cols = np.where(centerline_mask)
    row_map: Dict[int, List[int]] = {}
    for r, c in zip(rows.tolist(), cols.tolist()):
        row_map.setdefault(r, []).append(c)
    centers: Dict[int, np.ndarray] = {}
    for r, cols_list in row_map.items():
        cols_arr = np.array(sorted(cols_list), dtype=np.int64)
        if cols_arr.size == 0:
            continue
        groups = []
        current = [cols_arr[0]]
        for col in cols_arr[1:]:
            if int(col) - int(current[-1]) <= cluster_gap:
                current.append(int(col))
            else:
                groups.append(current)
                current = [int(col)]
        groups.append(current)
        centers[r] = np.array([int(round(np.mean(g))) for g in groups], dtype=np.int64)
    return centers


def _pick_centerline_seed(
    centerline_mask: np.ndarray,
    seed_rows: int,
    seed_cols: int,
    rng: random.Random,
) -> Optional[Tuple[int, int]]:
    height, width = centerline_mask.shape
    ego_row = 0
    ego_col = (width - 1) // 2
    row_end = min(height, max(1, seed_rows))
    col_start = max(0, ego_col - seed_cols)
    col_end = min(width, ego_col + seed_cols + 1)
    window = centerline_mask[ego_row:row_end, col_start:col_end]
    coords = np.argwhere(window)
    if coords.size == 0:
        coords = np.argwhere(centerline_mask)
        if coords.size == 0:
            return None
        idx = rng.randrange(coords.shape[0])
        return int(coords[idx, 0]), int(coords[idx, 1])
    idx = rng.randrange(coords.shape[0])
    return int(coords[idx, 0] + ego_row), int(coords[idx, 1] + col_start)


def _build_centerline_tracks(
    centerline_mask: np.ndarray,
    seed_rows: int,
    seed_cols: int,
    max_jump_px: int,
    rng: random.Random,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    height, width = centerline_mask.shape
    ego_col = (width - 1) // 2
    centers_by_row = _row_cluster_centers(centerline_mask)
    seed_row = None
    best_centers = None
    best_row_centers = None
    row_limit = min(height, max(1, seed_rows))
    col_min = max(0, ego_col - seed_cols)
    col_max = min(width - 1, ego_col + seed_cols)
    for r in range(row_limit):
        centers = centers_by_row.get(r)
        if centers is None or centers.size == 0:
            continue
        window = centers[(centers >= col_min) & (centers <= col_max)]
        if window.size == 0:
            continue
        if best_centers is None or window.size > best_centers.size:
            best_centers = window
            best_row_centers = centers
            seed_row = r
    if best_centers is None:
        for r in range(row_limit):
            centers = centers_by_row.get(r)
            if centers is None or centers.size == 0:
                continue
            if best_centers is None or centers.size > best_centers.size:
                best_centers = centers
                best_row_centers = centers
                seed_row = r
    if best_centers is None:
        for r, centers in centers_by_row.items():
            if centers.size > 0:
                best_centers = centers
                best_row_centers = centers
                seed_row = r
                break
    if best_centers is None:
        return (
            np.full((height,), ego_col, dtype=np.int64),
            None,
            None,
        )
    row_centers = best_row_centers if best_row_centers is not None else best_centers
    base_col = int(best_centers[np.abs(best_centers - ego_col).argmin()])
    left_cols = row_centers[row_centers < base_col]
    right_cols = row_centers[row_centers > base_col]
    left_seed = int(left_cols[-1]) if left_cols.size > 0 else None
    right_seed = int(right_cols[0]) if right_cols.size > 0 else None

    base_track = np.full((height,), base_col, dtype=np.int64)
    left_track = np.full((height,), left_seed if left_seed is not None else base_col, dtype=np.int64)
    right_track = np.full((height,), right_seed if right_seed is not None else base_col, dtype=np.int64)
    have_left = left_seed is not None
    have_right = right_seed is not None
    prev_base = base_col
    prev_left = left_seed if left_seed is not None else base_col
    prev_right = right_seed if right_seed is not None else base_col
    min_gap = 2
    for r in range(height):
        centers = centers_by_row.get(r)
        if centers is None or centers.size == 0:
            base_track[r] = prev_base
            if have_left:
                left_track[r] = prev_left
            if have_right:
                right_track[r] = prev_right
            continue
        base_idx = np.abs(centers - prev_base).argmin()
        base_col = int(centers[base_idx])
        if abs(base_col - prev_base) <= max_jump_px:
            prev_base = base_col
        base_track[r] = prev_base

        if have_left:
            left_candidates = centers[centers < (prev_base - min_gap)]
            if left_candidates.size > 0:
                left_idx = np.abs(left_candidates - prev_left).argmin()
                prev_left = int(left_candidates[left_idx])
            left_track[r] = prev_left
        if have_right:
            right_candidates = centers[centers > (prev_base + min_gap)]
            if right_candidates.size > 0:
                right_idx = np.abs(right_candidates - prev_right).argmin()
                prev_right = int(right_candidates[right_idx])
            right_track[r] = prev_right

    if not have_left:
        left_track = None
    if not have_right:
        right_track = None
    return base_track, left_track, right_track


def _stabilize_adjacent_track(
    base_track: np.ndarray,
    target_track: Optional[np.ndarray],
    side: int,
    width: int,
    max_offset_px: int,
    min_gap_px: int = 2,
) -> Optional[np.ndarray]:
    if target_track is None:
        return None
    signed_offset = side * (target_track.astype(np.int64) - base_track.astype(np.int64))
    valid = signed_offset >= min_gap_px
    if max_offset_px > 0:
        valid &= signed_offset <= max(max_offset_px * 2, min_gap_px)
    if not valid.any():
        return None

    fallback = int(round(float(np.median(signed_offset[valid]))))
    fallback = max(fallback, min_gap_px)
    if max_offset_px > 0:
        fallback = min(fallback, max_offset_px)

    stable_offset = np.full_like(signed_offset, fallback, dtype=np.int64)
    stable_offset[valid] = signed_offset[valid]
    stable_offset = np.maximum(stable_offset, min_gap_px)
    if max_offset_px > 0:
        stable_offset = np.minimum(stable_offset, max_offset_px)
    stable_track = base_track.astype(np.int64) + side * stable_offset
    return np.clip(stable_track, 0, width - 1).astype(np.int64)


def _follow_centerline(
    centerline_mask: np.ndarray,
    seed: Tuple[int, int],
    num_steps: int,
    step_px: int,
    rng: random.Random,
) -> np.ndarray:
    height, width = centerline_mask.shape
    row_map: Dict[int, np.ndarray] = {}
    rows, cols = np.where(centerline_mask)
    for r, c in zip(rows.tolist(), cols.tolist()):
        row_map.setdefault(r, []).append(c)
    for r, cols_list in row_map.items():
        row_map[r] = np.array(cols_list, dtype=np.int64)
    cur_row, cur_col = seed
    path = np.zeros((num_steps, 2), dtype=np.int64)
    for i in range(num_steps):
        target_row = cur_row + max(1, step_px)
        best_row = None
        best_cols = None
        for delta in range(0, 4):
            for candidate_row in (target_row + delta, target_row - delta):
                cols_arr = row_map.get(candidate_row)
                if cols_arr is None:
                    continue
                best_row = candidate_row
                best_cols = cols_arr
                break
            if best_cols is not None:
                break
        if best_cols is None:
            cur_row = min(height - 1, target_row)
        else:
            cur_row = best_row
            nearest_idx = np.abs(best_cols - cur_col).argmin()
            cur_col = int(best_cols[nearest_idx])
        path[i, 0] = cur_row
        path[i, 1] = cur_col
    return path


def _centerline_candidates(
    centerline_mask: np.ndarray,
    num_candidates: int,
    num_steps: int,
    step_px_base: int,
    pixel_size: float,
    max_offset_m: float,
    lanechange_steps: int,
    seed_rows: int,
    seed_cols: int,
    rng: random.Random,
) -> np.ndarray:
    if centerline_mask is None or not centerline_mask.any():
        return np.zeros((0, num_steps, 3), dtype=np.float32)
    height, width = centerline_mask.shape
    ego_row = 0
    ego_col = (width - 1) // 2
    base_track, left_track, right_track = _build_centerline_tracks(
        centerline_mask=centerline_mask,
        seed_rows=seed_rows,
        seed_cols=seed_cols,
        max_jump_px=max(4, step_px_base * 2),
        rng=rng,
    )
    shift = ego_col - int(base_track[0])
    if shift != 0:
        base_track = np.clip(base_track + shift, 0, width - 1)
        if left_track is not None:
            left_track = np.clip(left_track + shift, 0, width - 1)
        if right_track is not None:
            right_track = np.clip(right_track + shift, 0, width - 1)
    max_offset_px = (
        max(1, int(round(max_offset_m / max(pixel_size, 1e-3))))
        if max_offset_m > 0
        else width
    )
    left_track = _stabilize_adjacent_track(
        base_track=base_track,
        target_track=left_track,
        side=-1,
        width=width,
        max_offset_px=max_offset_px,
    )
    right_track = _stabilize_adjacent_track(
        base_track=base_track,
        target_track=right_track,
        side=1,
        width=width,
        max_offset_px=max_offset_px,
    )

    def _rows_from_step(step_px: float) -> np.ndarray:
        rows = (np.arange(num_steps, dtype=np.float32) + 1) * step_px
        rows = np.round(rows).astype(np.int64)
        rows = np.clip(rows, 0, height - 1)
        rows = np.maximum.accumulate(rows)
        return rows

    def _track_to_xy(track: np.ndarray, rows: np.ndarray) -> np.ndarray:
        cols = track[rows]
        xy = np.zeros((num_steps, 2), dtype=np.float32)
        xy[:, 0] = (rows - ego_row) * pixel_size
        xy[:, 1] = (cols - ego_col) * pixel_size
        return xy

    targets_track: List[np.ndarray] = []
    if left_track is not None:
        targets_track.append(left_track)
    if right_track is not None:
        targets_track.append(right_track)
    if not targets_track:
        targets_track.append(base_track)

    candidates = np.zeros((num_candidates, num_steps, 3), dtype=np.float32)
    half = int(math.ceil(num_candidates / 2))
    strengths = np.ones((half,), dtype=np.float32)
    lanechange_max = max(1, min(lanechange_steps, num_steps))
    lanechange_min = min(4, lanechange_max)
    speed_jitter = 0.1
    made = 0
    for _ in range(num_candidates * 2):
        target_track = targets_track[made % len(targets_track)]
        strength = float(strengths[made % strengths.size])
        lane_steps = rng.randint(lanechange_min, lanechange_max)
        start_max = max(0, num_steps - lane_steps)
        start_step = rng.randint(0, start_max) if start_max > 0 else 0
        ramp = np.linspace(0.0, 1.0, num=lane_steps, dtype=np.float32)
        ramp = 0.5 - 0.5 * np.cos(np.pi * ramp)
        profile = np.zeros((num_steps,), dtype=np.float32)
        profile[start_step : start_step + lane_steps] = ramp
        profile[start_step + lane_steps :] = 1.0
        speed_scale = 1.0 + rng.uniform(-speed_jitter, speed_jitter)
        rows = _rows_from_step(max(1.0, step_px_base * speed_scale))
        base_xy = _track_to_xy(base_track, rows)
        target_xy = _track_to_xy(target_track, rows)
        base_y0 = float(base_xy[0, 1])
        if abs(base_y0) > 1e-3:
            base_xy[:, 1] -= base_y0
            target_xy[:, 1] -= base_y0
        delta = target_xy - base_xy
        if max_offset_m > 0:
            lateral = np.abs(delta[:, 1]) + 1e-3
            scale = np.minimum(1.0, max_offset_m / lateral)
            delta = delta * scale[:, None]
        blend = (strength * profile)[:, None]
        xy = base_xy + delta * blend
        candidates[made, :, 0:2] = xy
        deltas = np.diff(xy, axis=0, prepend=xy[:1])
        candidates[made, :, 2] = np.arctan2(deltas[:, 1], deltas[:, 0])
        made += 1
        if made >= num_candidates:
            break
    return candidates[:made]


def _select_topk(
    candidates: np.ndarray,
    scores: np.ndarray,
    topk: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if scores.size == 0:
        order = np.arange(min(topk, candidates.shape[0]))
        return (
            candidates[:topk],
            np.full((topk,), np.nan, dtype=np.float32),
            order,
        )
    safe_scores = np.nan_to_num(scores, nan=-np.inf)
    order = np.argsort(-safe_scores)
    order = order[: min(topk, order.size)]
    return candidates[order], scores[order], order


def _iter_target_files(cache_path: Path, log_names: Optional[List[str]]) -> Iterable[Path]:
    if log_names:
        roots = [cache_path / name for name in log_names]
    else:
        roots = [p for p in cache_path.iterdir() if p.is_dir()]
    for root in roots:
        for token_dir in root.iterdir():
            target_path = token_dir / "transfuser_target.gz"
            if target_path.is_file():
                yield target_path


def _safe_vis_name(value: object) -> str:
    text = str(value)
    safe = [
        ch
        if ch.isascii() and (ch.isalnum() or ch in "._-")
        else "_"
        for ch in text
    ]
    return "".join(safe).strip("_") or "sample"


def _vis_path(target_path: Path, token: object, vis_dir: str, stage: str) -> Path:
    log_name = target_path.parent.parent.name
    token_name = token if isinstance(token, str) else target_path.parent.name
    file_name = f"{_safe_vis_name(log_name)}__{_safe_vis_name(token_name)}__{stage}.png"
    return Path(vis_dir) / stage / file_name


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-path", required=True)
    ap.add_argument("--metric-cache-path", required=True)
    ap.add_argument("--log-names", nargs="*", default=None)
    ap.add_argument("--num-candidates", type=int, default=30)
    ap.add_argument("--num-random", type=int, default=15)
    ap.add_argument("--num-centerline", type=int, default=15)
    ap.add_argument("--keep-topk", type=int, default=5)
    ap.add_argument("--num-poses", type=int, default=8)
    ap.add_argument("--dt", type=float, default=0.5)
    ap.add_argument("--pdm-num-poses", type=int, default=40)
    ap.add_argument("--pdm-interval", type=float, default=0.1)
    ap.add_argument("--wheel-base", type=float, default=3.089)
    ap.add_argument("--max-steer-deg", type=float, default=20.0)
    ap.add_argument("--min-accel", type=float, default=-1.0)
    ap.add_argument("--max-accel", type=float, default=1.0)
    ap.add_argument("--random-max-dyaw", type=float, default=0.3)
    ap.add_argument("--random-max-yaw-flips", type=int, default=1)
    ap.add_argument("--random-yaw-eps", type=float, default=1e-3)
    ap.add_argument("--gt-accel-factors", type=float, nargs="*", default=[1.05, 1.1])
    ap.add_argument("--road-label", type=int, default=1)
    ap.add_argument("--centerline-label", type=int, default=3)
    ap.add_argument("--bev-pixel-size", type=float, default=0.25)
    ap.add_argument("--centerline-offset-m", type=float, default=4.0)
    ap.add_argument("--centerline-lanechange-steps", type=int, default=6)
    ap.add_argument("--centerline-seed-rows", type=int, default=12)
    ap.add_argument("--centerline-seed-cols", type=int, default=16)
    ap.add_argument("--strict-steps", type=int, default=3)
    ap.add_argument("--feasible-thresh", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--use-ray", action="store_true")
    ap.add_argument("--ray-threads", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument(
        "--overwrite-invalid-only",
        action="store_true",
        help=(
            "For existing candidate caches, recompute only when active "
            "pdm_score_targets or gt_pdm_score are missing/NaN/Inf."
        ),
    )
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--require-above-gt", action="store_true")
    ap.add_argument("--gt-score-margin", type=float, default=0.0)
    ap.add_argument("--store-components", action="store_true")
    ap.add_argument("--score-with-baseline", action="store_true")
    ap.add_argument("--vis-dir", type=str, default="")
    ap.add_argument("--vis-max", type=int, default=0)
    ap.add_argument("--vis-every", type=int, default=1)
    ap.add_argument("--vis-raw-candidates", action="store_true")
    ap.add_argument("--raw-vis-only", action="store_true")
    ap.add_argument("--max-files", type=int, default=0)
    ap.add_argument("--no-dump", action="store_true")
    args = ap.parse_args()

    cache_root = Path(args.cache_path)
    if args.num_shards < 1:
        raise ValueError(f"--num-shards must be >= 1, got {args.num_shards}")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError(
            f"--shard-index must be in [0, {args.num_shards}), got {args.shard_index}"
        )

    all_target_files = sorted(_iter_target_files(cache_root, args.log_names))
    target_files = [
        (idx, path)
        for idx, path in enumerate(all_target_files)
        if idx % args.num_shards == args.shard_index
    ]
    if args.max_files > 0:
        target_files = target_files[: args.max_files]
    logger.info(
        "Candidate cache shard %d/%d selected %d/%d target files",
        args.shard_index,
        args.num_shards,
        len(target_files),
        len(all_target_files),
    )

    rng = random.Random(args.seed + args.shard_index * 1000003)
    if args.num_random + args.num_centerline != args.num_candidates:
        accel_count = len([f for f in args.gt_accel_factors if f > 0])
        if args.num_candidates <= args.num_centerline + accel_count:
            args.num_centerline = max(args.num_candidates - accel_count, 0)
            args.num_random = 0
        else:
            args.num_random = args.num_candidates - args.num_centerline - accel_count
        logger.info(
            "Adjusted candidate counts to num_centerline=%d num_random=%d",
            args.num_centerline,
            args.num_random,
        )
    metric_loader = None
    traj_sampling = None
    pdm_sampling = None
    simulator = None
    scorer = None
    pdm = None
    if args.raw_vis_only:
        logger.info("raw-vis-only enabled; skipping PDM scorer initialization")
    elif args.score_with_baseline:
        metric_loader = MetricCacheLoader(Path(args.metric_cache_path))
        traj_sampling = TrajectorySampling(
            num_poses=args.num_poses, interval_length=args.dt
        )
        pdm_sampling = TrajectorySampling(
            num_poses=args.pdm_num_poses, interval_length=args.pdm_interval
        )
        simulator = PDMSimulator(pdm_sampling)
        if PDMScorerConfig is None:
            scorer = PDMScorer(pdm_sampling)
        else:
            scorer = PDMScorer(pdm_sampling, PDMScorerConfig())
    else:
        pdm = PDMSupervision(
            PDMScoreConfig(
                cache_path=args.metric_cache_path,
                num_poses=args.pdm_num_poses,
                interval_length=args.pdm_interval,
                use_ray=args.use_ray,
                ray_threads=args.ray_threads,
            )
        )

    total = 0
    updated = 0
    skipped = 0
    invalid_existing = 0
    vis_saved = 0
    raw_vis_saved = 0
    progress_desc = (
        f"shard {args.shard_index + 1}/{args.num_shards}"
        if args.num_shards > 1
        else None
    )
    for _, target_path in tqdm(target_files, desc=progress_desc):
        total += 1
        data = load_feature_target_from_pickle(target_path)
        if "trajectory_candidates" in data:
            if args.overwrite_invalid_only:
                if _cached_pdm_scores_are_invalid(data):
                    invalid_existing += 1
                else:
                    skipped += 1
                    continue
            elif args.skip_existing and not args.overwrite:
                skipped += 1
                continue

        token = data.get("token")
        if token is None:
            skipped += 1
            continue

        bev = data.get("bev_semantic_map")
        if bev is None:
            skipped += 1
            continue
        future_bev = data.get("future_bev_semantic_map")

        bev_np = _to_numpy(bev)
        cur_mask = _build_drivable_mask(
            bev_np, args.road_label, args.centerline_label
        )
        future_mask = None
        if future_bev is not None:
            future_mask = _build_drivable_mask(
                _to_numpy(future_bev), args.road_label, args.centerline_label
            )

        traj_gt = _to_numpy(data["trajectory"]).astype(np.float32)
        if traj_gt.ndim == 3:
            traj_gt = traj_gt[0]
        v0 = float(max(np.linalg.norm(traj_gt[0, :2]) / max(args.dt, 1e-3), 0.0))

        step_px_base = max(
            1,
            int(
                round(
                    (v0 * args.dt) / max(args.bev_pixel_size, 1e-3)
                )
            ),
        )
        centerline_src = future_bev if future_bev is not None else bev
        centerline_mask = _build_centerline_mask(
            _to_numpy(centerline_src), args.centerline_label
        )
        centerline_candidates = _centerline_candidates(
            centerline_mask=centerline_mask,
            num_candidates=args.num_centerline,
            num_steps=args.num_poses,
            step_px_base=step_px_base,
            pixel_size=args.bev_pixel_size,
            max_offset_m=args.centerline_offset_m,
            lanechange_steps=args.centerline_lanechange_steps,
            seed_rows=args.centerline_seed_rows,
            seed_cols=args.centerline_seed_cols,
            rng=rng,
        )
        gt_accel_candidates = _accelerate_gt_candidates(
            traj_gt,
            dt=args.dt,
            factors=[f for f in args.gt_accel_factors if f > 0],
        )
        num_random = max(
            args.num_candidates
            - centerline_candidates.shape[0]
            - gt_accel_candidates.shape[0],
            0,
        )
        random_candidates = _sample_candidates(
            rng=rng,
            num_candidates=max(num_random, 0),
            num_steps=args.num_poses,
            dt=args.dt,
            v0=v0,
            wheel_base=args.wheel_base,
            max_steer_deg=args.max_steer_deg,
            min_accel=args.min_accel,
            max_accel=args.max_accel,
        )
        random_candidates = _filter_random_candidates(
            random_candidates,
            max_dyaw=float(args.random_max_dyaw),
            max_flips=int(args.random_max_yaw_flips),
            yaw_eps=float(args.random_yaw_eps),
        )
        candidates = np.concatenate(
            [centerline_candidates, gt_accel_candidates, random_candidates],
            axis=0,
        )
        if candidates.shape[0] == 0:
            skipped += 1
            continue

        if (
            args.vis_raw_candidates
            and args.vis_dir
            and (args.vis_max <= 0 or raw_vis_saved < args.vis_max)
        ):
            if args.vis_every > 0 and (total - 1) % args.vis_every == 0:
                _visualize_bev_candidates(
                    bev=_to_numpy(bev),
                    candidates=candidates,
                    pixel_size=args.bev_pixel_size,
                    save_path=_vis_path(target_path, token, args.vis_dir, "raw"),
                    scores=None,
                    gt_traj=traj_gt,
                )
                raw_vis_saved += 1

        if args.raw_vis_only:
            skipped += 1
            continue

        pixel_size = float(args.bev_pixel_size)
        ratios = np.array(
            [
                _mask_ratio(
                    cand[:, :2],
                    cur_mask,
                    future_mask,
                    pixel_size,
                    args.strict_steps,
                )
                for cand in candidates
            ],
            dtype=np.float32,
        )
        valid_mask = ratios >= args.feasible_thresh
        if not valid_mask.any():
            valid_mask = np.ones_like(valid_mask, dtype=bool)

        candidates_valid = candidates[valid_mask]
        if candidates_valid.shape[0] == 0:
            skipped += 1
            continue

        comps = None
        gt_components = None
        if args.score_with_baseline:
            metric_cache = metric_loader.get_from_token(token)
            scores, comps = _score_with_baseline(
                metric_cache,
                np.asarray(candidates_valid, dtype=np.float32),
                traj_sampling,
                pdm_sampling,
                simulator,
                scorer,
                args.store_components,
            )
            gt_scores, gt_components = _score_with_baseline(
                metric_cache,
                np.asarray(traj_gt, dtype=np.float32)[None, ...],
                traj_sampling,
                pdm_sampling,
                simulator,
                scorer,
                args.store_components,
            )
        else:
            trajs_pdm = np.asarray(candidates_valid, dtype=np.float32)[None, ...]
            scores = pdm.score_batch([token], trajs_pdm)[0]
            if args.store_components:
                comps = pdm.score_batch_components([token], trajs_pdm)[0]

            gt_scores = pdm.score_batch(
                [token], np.asarray(traj_gt, dtype=np.float32)[None, None, ...]
            )[0]
            if args.store_components:
                gt_components = pdm.score_batch_components(
                    [token], np.asarray(traj_gt, dtype=np.float32)[None, None, ...]
                )[0]
        gt_score = float(gt_scores[0]) if gt_scores.size else float("nan")

        drop_all = False
        if args.require_above_gt:
            thresh = gt_score + float(args.gt_score_margin)
            keep = scores >= thresh
            if keep.any():
                candidates_valid = candidates_valid[keep]
                scores = scores[keep]
                if comps is not None:
                    comps = comps[keep]
            else:
                drop_all = True

        top_traj, top_scores, top_idx = _select_topk(
            candidates_valid, scores, args.keep_topk
        )            
        top_mask = np.ones((top_traj.shape[0],), dtype=bool)
        top_components = None
        if comps is not None:
            top_components = comps[top_idx]
        if drop_all:
            top_mask = np.zeros((top_traj.shape[0],), dtype=bool)
            top_scores = np.full((top_traj.shape[0],), np.nan, dtype=np.float32)
            if top_components is not None:
                top_components = np.full_like(top_components, np.nan, dtype=np.float32)

        if top_traj.shape[0] < args.keep_topk:
            pad_count = args.keep_topk - top_traj.shape[0]
            pad = np.repeat(top_traj[-1:, ...], pad_count, axis=0)
            top_traj = np.concatenate([top_traj, pad], axis=0)
            top_scores = np.concatenate(
                [top_scores, np.full((pad_count,), np.nan, dtype=np.float32)],
                axis=0,
            )
            top_mask = np.concatenate(
                [top_mask, np.zeros((pad_count,), dtype=bool)], axis=0
            )
            if top_components is not None:
                pad_comps = np.full((pad_count, top_components.shape[1]), np.nan, dtype=np.float32)
                top_components = np.concatenate([top_components, pad_comps], axis=0)

        data["trajectory_candidates"] = torch.tensor(top_traj, dtype=torch.float32)
        data["trajectory_candidates_mask"] = torch.tensor(top_mask, dtype=torch.bool)
        data["pdm_score_targets"] = torch.tensor(top_scores, dtype=torch.float32)
        data["gt_pdm_score"] = torch.tensor(gt_scores[0], dtype=torch.float32)
        if args.store_components:
            data["pdm_score_components_candidates"] = torch.tensor(
                top_components if top_components is not None else np.full((top_traj.shape[0], 6), np.nan, dtype=np.float32),
                dtype=torch.float32,
            )
            data["gt_pdm_components"] = torch.tensor(
                gt_components[0], dtype=torch.float32
            )

        if args.vis_dir and (args.vis_max <= 0 or vis_saved < args.vis_max):
            if args.vis_every > 0 and (total - 1) % args.vis_every == 0:
                if top_mask.any():
                    vis_traj = top_traj[top_mask]
                    vis_scores = top_scores[top_mask]
                else:
                    vis_traj = top_traj
                    vis_scores = top_scores
                _visualize_bev_candidates(
                    bev=_to_numpy(bev),
                    candidates=vis_traj,
                    pixel_size=args.bev_pixel_size,
                    save_path=_vis_path(target_path, token, args.vis_dir, "topk"),
                    scores=vis_scores,
                    gt_traj=traj_gt,
                )
                vis_saved += 1
            # T_top_traj = torch.tensor(top_traj, dtype=torch.float32).unsqueeze(0)
            # T_GT = torch.tensor(traj_gt, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            # traj_check = torch.cat([T_top_traj, T_GT], dim=1)
            # plot_trajectory_anchors(traj_check[:,:,:,:2], None, 'top_candi_traj')
            # plot_trajectory_anchors(torch.tensor(candidates_valid).unsqueeze(0),None,'origin_candi')
            # import ipdb; ipdb.set_trace()

        if args.no_dump:
            skipped += 1
        else:
            dump_feature_target_to_pickle(target_path, data)
            updated += 1

    logger.info(
        "Candidate cache done. shard=%d/%d total=%d updated=%d skipped=%d invalid_existing=%d raw_vis_saved=%d topk_vis_saved=%d",
        args.shard_index,
        args.num_shards,
        total,
        updated,
        skipped,
        invalid_existing,
        raw_vis_saved,
        vis_saved,
    )

def plot_trajectory_anchors(trajectory_anchors, pi=None, file_name="trajectory_anchors.png"):
    """
    可视化轨迹锚点tensor并保存为PNG图像
    
    Args:
        trajectory_anchors: 轨迹锚点tensor，形状为 (bs, modes, poses, 2)
        file_name: 保存的文件名
    """
    # 确保输出目录存在
    output_dir = "/home/xqf/DiffusionDrive-main/free_anchors"
    os.makedirs(output_dir, exist_ok=True)
    
    # 将tensor转换为numpy数组
    if isinstance(trajectory_anchors, torch.Tensor):
        trajectory_anchors = trajectory_anchors.detach().cpu().numpy()
    
    # 获取维度信息
    bs, modes, poses, _ = trajectory_anchors.shape
    # import pdb; pdb.set_trace()
    if pi is not None:
        max_prob_indices = pi.argmax(dim=1)
        max_prob = pi[torch.arange(bs), max_prob_indices]
    
    # 为每个batch创建一个图像
    for batch_idx in range(1):
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        
        # 为每个mode绘制轨迹
        for mode_idx in range(modes):
            # 获取当前batch和mode的轨迹数据
            trajectory = trajectory_anchors[batch_idx, mode_idx, :, :]  # [poses, 2]
            
            # 绘制轨迹点
            ax.plot(trajectory[:, 0], trajectory[:, 1], marker='o', markersize=4, linewidth=2, 
                    label=f'Mode {mode_idx}')
            
            # 标记起始点和结束点
            ax.scatter(trajectory[0, 0], trajectory[0, 1], color='green', s=50, marker='s', 
                      label='Start' if mode_idx == 0 else "")
            ax.scatter(trajectory[-1, 0], trajectory[-1, 1], color='red', s=50, marker='s',
                      label='End' if mode_idx == 0 else "")
        
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        if pi is not None:
            ax.set_title(f'Trajectory Anchors - Batch {batch_idx} - best_Mode {max_prob_indices[batch_idx]} - prob {max_prob[batch_idx]:.3f}')
        else:
            ax.set_title(f'Trajectory Anchors - Batch {batch_idx}')
        ax.legend()
        ax.grid(True)
        
        # 保存图像
        save_path = os.path.join(output_dir, f"batch_{batch_idx}_{file_name}")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
    print(f"Saved trajectory anchors visualization to {output_dir}")


def _visualize_bev_candidates(
    bev: np.ndarray,
    candidates: np.ndarray,
    pixel_size: float,
    save_path: Path,
    scores: Optional[np.ndarray] = None,
    gt_traj: Optional[np.ndarray] = None,
) -> None:
    class_map = _bev_to_class_map(bev)
    height, width = class_map.shape
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(class_map, origin="upper", interpolation="nearest")

    if scores is not None and scores.size == candidates.shape[0]:
        finite = np.isfinite(scores)
        if finite.any():
            min_score = float(np.nanmin(scores))
            max_score = float(np.nanmax(scores))
            denom = max(max_score - min_score, 1e-6)
            norm = (scores - min_score) / denom
        else:
            norm = np.zeros_like(scores, dtype=np.float32)
        colors = plt.cm.viridis(np.clip(norm, 0.0, 1.0))
    else:
        colors = plt.cm.tab20(np.linspace(0, 1, candidates.shape[0]))

    for idx, traj in enumerate(candidates):
        xy = traj[:, :2]
        row = xy[:, 0] / pixel_size
        col = xy[:, 1] / pixel_size + (width - 1) / 2.0
        color = colors[idx] if idx < len(colors) else (1.0, 0.0, 0.0, 1.0)
        ax.plot(col, row, linewidth=1.5, color=color)
        ax.scatter(col[0], row[0], s=10, color=color)

    if gt_traj is not None:
        xy = gt_traj[:, :2]
        row = xy[:, 0] / pixel_size
        col = xy[:, 1] / pixel_size + (width - 1) / 2.0
        ax.plot(col, row, linewidth=2.0, color="black", label="GT")

    ax.scatter([(width - 1) / 2.0], [0], s=40, color="white", marker="x")
    ax.set_xlim([-0.5, width - 0.5])
    ax.set_ylim([height - 0.5, -0.5])
    ax.set_title(save_path.stem)
    ax.axis("off")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    main()
