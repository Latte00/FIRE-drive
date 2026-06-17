"""
Compare fast cache fields against official NAVSIM PDM safety metrics.

This script is read-only by default. It can derive the same fields produced by
patch_navsim_fast_pdm_cache.py directly from metric_cache.pkl without writing them
to training_cache, then uses those OBB fields to approximate the PDM
no-at-fault-collision and TTC components. It also approximates DAC from
feasible_area_mask / bev_semantic_map when available.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import lzma
import math
import pickle
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

from navsim.common.dataclasses import Trajectory
from navsim.evaluate.pdm_score import get_trajectory_as_array, transform_trajectory
from navsim.planning.script.patch_navsim_fast_pdm_cache import (
    _compute_feasible_masks,
    _collect_future_agents,
    _load_metric_cache_index,
)
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import (
    PDMScorer,
)
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import (
    PDMSimulator,
)
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import (
    EgoAreaIndex,
    MultiMetricIndex,
    StateIndex,
    WeightedMetricIndex,
)
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.planning.simulation.observation.idm.utils import is_agent_ahead, is_agent_behind
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import (
    state_array_to_coords_array,
)


PDM_COMPONENT_NAMES = (
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "ego_progress",
    "time_to_collision_within_bound",
    "comfort",
    "driving_direction_compliance",
)


def _load_metric_cache_indices(metric_cache_paths: Sequence[str]) -> Dict[str, str]:
    merged: Dict[str, str] = {}
    for metric_cache_path in metric_cache_paths:
        current = _load_metric_cache_index(Path(metric_cache_path))
        merged.update(current)
    return merged


def _load_gzip_pickle(path: Path) -> Dict[str, Any]:
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def _load_metric_cache(path: Path) -> Any:
    with lzma.open(path, "rb") as f:
        return pickle.load(f)


def _to_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _normalize_angle(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _ensure_heading(traj: np.ndarray) -> np.ndarray:
    arr = np.asarray(traj, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[-1] < 2:
        raise ValueError(f"Expected trajectory [T, D>=2], got {arr.shape}")
    if arr.shape[-1] >= 3:
        return arr[:, :3]
    delta = np.diff(arr[:, :2], axis=0, prepend=np.zeros((1, 2), dtype=np.float32))
    heading = np.arctan2(delta[:, 1], delta[:, 0]).astype(np.float32)
    return np.concatenate([arr[:, :2], heading[:, None]], axis=-1)


def _iter_target_paths(cache_root: Path, roots: Optional[Sequence[str]], limit: int) -> List[Path]:
    root_paths = [cache_root / root for root in roots] if roots else [cache_root]
    out: List[Path] = []
    for root in root_paths:
        if not root.exists():
            continue
        for path in root.rglob("transfuser_target.gz"):
            out.append(path)
            if limit > 0 and len(out) >= limit:
                return sorted(out)
    return sorted(out)


def _as_candidates(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    arr = _to_numpy(value).astype(np.float32)
    while arr.ndim > 0 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 3 or arr.shape[-1] < 2:
        return None
    return arr[..., : min(arr.shape[-1], 3)]


def _as_trajectory(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    arr = _to_numpy(value).astype(np.float32)
    while arr.ndim > 0 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2 or arr.shape[-1] < 2:
        return None
    return arr[:, : min(arr.shape[-1], 3)]


def _candidate_mask(data: Dict[str, Any], count: int) -> np.ndarray:
    mask = data.get("trajectory_candidates_mask")
    if mask is None:
        return np.ones(count, dtype=bool)
    arr = _to_numpy(mask).astype(bool).reshape(-1)
    if arr.size < count:
        arr = np.pad(arr, (0, count - arr.size), constant_values=False)
    return arr[:count]


def _best_candidate_index(data: Dict[str, Any], candidates: np.ndarray) -> int:
    count = int(candidates.shape[0])
    mask = _candidate_mask(data, count)
    scores_raw = data.get("pdm_score_targets")
    if scores_raw is not None:
        scores = _to_numpy(scores_raw).astype(np.float32).reshape(-1)
        if scores.size >= count:
            valid = mask & np.isfinite(scores[:count])
            if valid.any():
                return int(np.nanargmax(np.where(valid, scores[:count], np.nan)))
    valid_idx = np.where(mask)[0]
    return int(valid_idx[0]) if valid_idx.size else -1


def _select_trajectories(data: Dict[str, Any], args: argparse.Namespace) -> Tuple[List[str], np.ndarray]:
    labels: List[str] = []
    trajs: List[np.ndarray] = []
    gt = _as_trajectory(data.get("trajectory"))
    candidates = _as_candidates(data.get("trajectory_candidates"))

    source = str(args.trajectory_set)
    if source in {"gt", "gt_best", "gt_all"} and gt is not None:
        labels.append("gt")
        trajs.append(_ensure_heading(gt))

    if candidates is not None and source in {"best", "gt_best"}:
        best_idx = _best_candidate_index(data, candidates)
        if best_idx >= 0:
            labels.append(f"candidate_{best_idx}")
            trajs.append(_ensure_heading(candidates[best_idx]))

    if candidates is not None and source in {"all", "gt_all"}:
        mask = _candidate_mask(data, int(candidates.shape[0]))
        indices = np.where(mask)[0]
        max_candidates = int(args.max_candidates)
        if max_candidates > 0:
            indices = indices[:max_candidates]
        for idx in indices:
            labels.append(f"candidate_{int(idx)}")
            trajs.append(_ensure_heading(candidates[int(idx)]))

    if not trajs:
        return [], np.zeros((0, 0, 3), dtype=np.float32)

    horizon = min(traj.shape[0] for traj in trajs)
    trajs = [traj[:horizon] for traj in trajs if traj.shape[0] >= horizon]
    return labels, np.stack(trajs, axis=0).astype(np.float32)


def _simulated_states(
    metric_cache: Any,
    trajectories: np.ndarray,
    simulator: PDMSimulator,
    trajectory_interval: float,
) -> np.ndarray:
    initial_ego_state = metric_cache.ego_state
    proposal_states = []
    for traj in trajectories:
        traj_sampling = TrajectorySampling(
            num_poses=int(traj.shape[0]),
            interval_length=float(trajectory_interval),
        )
        pred_trajectory = transform_trajectory(
            Trajectory(
                np.nan_to_num(traj, nan=0.0).astype(np.float32),
                trajectory_sampling=traj_sampling,
            ),
            initial_ego_state,
        )
        proposal_states.append(
            get_trajectory_as_array(
                pred_trajectory,
                simulator.proposal_sampling,
                initial_ego_state.time_point,
            )
        )
    proposal_states_np = np.stack(proposal_states, axis=0)
    return simulator.simulate_proposals(proposal_states_np, initial_ego_state)


def _official_components(
    metric_cache: Any,
    simulated_states: np.ndarray,
    scorer: PDMScorer,
) -> Tuple[np.ndarray, np.ndarray]:
    try:
        scores = scorer.score_proposals(
            simulated_states,
            metric_cache.observation,
            metric_cache.centerline,
            metric_cache.route_lane_ids,
            metric_cache.drivable_area_map,
            getattr(metric_cache, "pdm_progress", None),
        )
    except TypeError:
        scores = scorer.score_proposals(
            simulated_states,
            metric_cache.observation,
            metric_cache.centerline,
            metric_cache.route_lane_ids,
            metric_cache.drivable_area_map,
        )
    components = np.stack(
        [
            np.asarray(scorer._multi_metrics[MultiMetricIndex.NO_COLLISION], dtype=np.float32),
            np.asarray(scorer._multi_metrics[MultiMetricIndex.DRIVABLE_AREA], dtype=np.float32),
            np.asarray(scorer._weighted_metrics[WeightedMetricIndex.PROGRESS], dtype=np.float32),
            np.asarray(scorer._weighted_metrics[WeightedMetricIndex.TTC], dtype=np.float32),
            np.asarray(scorer._weighted_metrics[WeightedMetricIndex.COMFORTABLE], dtype=np.float32),
            np.asarray(scorer._weighted_metrics[WeightedMetricIndex.DRIVING_DIRECTION], dtype=np.float32),
        ],
        axis=-1,
    )
    return np.asarray(scores, dtype=np.float32), components


def _official_components_profiled(
    metric_cache: Any,
    simulated_states: np.ndarray,
    scorer: PDMScorer,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    timings: Dict[str, float] = {}

    def timed(name: str, fn: Any) -> Any:
        t0 = time.perf_counter()
        result = fn()
        timings[name] = timings.get(name, 0.0) + time.perf_counter() - t0
        return result

    timed(
        "reset",
        lambda: scorer._reset(
            simulated_states,
            metric_cache.observation,
            metric_cache.centerline,
            metric_cache.route_lane_ids,
            metric_cache.drivable_area_map,
        ),
    )
    timed("ego_area", scorer._calculate_ego_area)
    timed("no_collision", scorer._calculate_no_at_fault_collision)
    timed("drivable_area", scorer._calculate_drivable_area_compliance)
    timed("driving_direction", scorer._calculate_driving_direction_compliance)
    timed("progress", scorer._calculate_progress)
    timed("ttc", scorer._calculate_ttc)
    timed("comfort", scorer._calculate_is_comfortable)
    scores = timed("aggregate", scorer._aggregate_scores)

    components = np.stack(
        [
            np.asarray(scorer._multi_metrics[MultiMetricIndex.NO_COLLISION], dtype=np.float32),
            np.asarray(scorer._multi_metrics[MultiMetricIndex.DRIVABLE_AREA], dtype=np.float32),
            np.asarray(scorer._weighted_metrics[WeightedMetricIndex.PROGRESS], dtype=np.float32),
            np.asarray(scorer._weighted_metrics[WeightedMetricIndex.TTC], dtype=np.float32),
            np.asarray(scorer._weighted_metrics[WeightedMetricIndex.COMFORTABLE], dtype=np.float32),
            np.asarray(scorer._weighted_metrics[WeightedMetricIndex.DRIVING_DIRECTION], dtype=np.float32),
        ],
        axis=-1,
    )
    return np.asarray(scores, dtype=np.float32), components, timings


def _cached_dac_from_ego_area(scorer: PDMScorer) -> Tuple[Optional[np.ndarray], Dict[str, float]]:
    ego_areas = getattr(scorer, "_ego_areas", None)
    if ego_areas is None:
        return None, {}
    ego_areas_np = np.asarray(ego_areas, dtype=np.bool_)
    if ego_areas_np.ndim != 3 or ego_areas_np.shape[-1] <= int(EgoAreaIndex.NON_DRIVABLE_AREA):
        return None, {}

    non_drivable = ego_areas_np[:, :, EgoAreaIndex.NON_DRIVABLE_AREA]
    scores = np.ones(non_drivable.shape[0], dtype=np.float32)
    scores[non_drivable.any(axis=-1)] = 0.0

    stats = {
        "files": 1.0,
        "num_proposals": float(ego_areas_np.shape[0]),
        "num_timesteps": float(ego_areas_np.shape[1]),
        "num_channels": float(ego_areas_np.shape[2]),
        "full_ego_area_bool_bytes": float(ego_areas_np.nbytes),
        "non_drivable_mask_bool_bytes": float(non_drivable.nbytes),
        "dac_label_uint8_bytes": float(scores.astype(np.uint8).nbytes),
        "full_ego_area_bitpacked_bytes": float(math.ceil(ego_areas_np.size / 8.0)),
        "non_drivable_mask_bitpacked_bytes": float(math.ceil(non_drivable.size / 8.0)),
    }
    return scores, stats


def _official_dac_only_scores(metric_cache: Any, simulated_states: np.ndarray) -> np.ndarray:
    ego_coords = state_array_to_coords_array(simulated_states, get_pacifica_parameters())
    in_polygons = metric_cache.drivable_area_map.points_in_polygons(ego_coords)
    in_polygons = in_polygons.transpose(1, 2, 0, 3)
    drivable_area_idcs = metric_cache.drivable_area_map.get_indices_of_map_type(
        [
            SemanticMapLayer.ROADBLOCK,
            SemanticMapLayer.INTERSECTION,
            SemanticMapLayer.DRIVABLE_AREA,
            SemanticMapLayer.CARPARK_AREA,
        ]
    )
    corners_in_polygon = in_polygons[..., :-1]
    nondrivable_mask = (
        (corners_in_polygon[:, :, drivable_area_idcs].sum(axis=-2) > 0).sum(axis=-1) < 4
    )
    scores = np.ones(simulated_states.shape[0], dtype=np.float32)
    scores[nondrivable_mask.any(axis=-1)] = 0.0
    return scores


def _global_states_to_local(states: np.ndarray, metric_cache: Any) -> np.ndarray:
    ego = metric_cache.ego_state.center
    ego_x, ego_y, ego_h = float(ego.x), float(ego.y), float(ego.heading)
    out = states.astype(np.float32, copy=True)
    dx = states[..., StateIndex.X] - ego_x
    dy = states[..., StateIndex.Y] - ego_y
    cos_h = float(np.cos(ego_h))
    sin_h = float(np.sin(ego_h))
    out[..., StateIndex.X] = cos_h * dx + sin_h * dy
    out[..., StateIndex.Y] = -sin_h * dx + cos_h * dy
    out[..., StateIndex.HEADING] = _normalize_angle(states[..., StateIndex.HEADING] - ego_h)
    vx = states[..., StateIndex.VELOCITY_X]
    vy = states[..., StateIndex.VELOCITY_Y]
    out[..., StateIndex.VELOCITY_X] = cos_h * vx + sin_h * vy
    out[..., StateIndex.VELOCITY_Y] = -sin_h * vx + cos_h * vy
    return out


def _obb_overlap(
    c1: np.ndarray,
    h1: float,
    l1: float,
    w1: float,
    c2: np.ndarray,
    h2: float,
    l2: float,
    w2: float,
    margin: float,
) -> bool:
    axes = []
    for heading in (h1, h2):
        ux = np.array([math.cos(heading), math.sin(heading)], dtype=np.float32)
        uy = np.array([-math.sin(heading), math.cos(heading)], dtype=np.float32)
        axes.extend([ux, uy])
    u1 = axes[0]
    v1 = axes[1]
    u2 = axes[2]
    v2 = axes[3]
    d = c2 - c1
    hl1 = 0.5 * float(l1) + margin
    hw1 = 0.5 * float(w1) + margin
    hl2 = 0.5 * float(l2)
    hw2 = 0.5 * float(w2)
    for axis in axes:
        r1 = hl1 * abs(float(np.dot(axis, u1))) + hw1 * abs(float(np.dot(axis, v1)))
        r2 = hl2 * abs(float(np.dot(axis, u2))) + hw2 * abs(float(np.dot(axis, v2)))
        if abs(float(np.dot(d, axis))) > r1 + r2:
            return False
    return True


def _approx_collision_scores(
    local_states: np.ndarray,
    future_fields: Dict[str, Any],
    args: argparse.Namespace,
) -> Tuple[np.ndarray, np.ndarray]:
    obb = _to_numpy(future_fields["future_agent_obb"]).astype(np.float32)
    mask = _to_numpy(future_fields["future_agent_mask"]).astype(bool)
    is_agent = _to_numpy(future_fields["future_agent_is_agent"]).astype(bool)
    ignore = _to_numpy(future_fields.get("future_agent_ignore", np.zeros(len(obb), dtype=bool))).astype(bool)

    vehicle = get_pacifica_parameters()
    ego_length = float(vehicle.half_length) * 2.0
    ego_width = float(vehicle.half_width) * 2.0
    rear_axle_to_center = float(vehicle.rear_axle_to_center)
    margin = float(args.collision_margin_m)
    mode = str(args.approx_collision_mode)

    num_modes, horizon = local_states.shape[:2]
    scores = np.ones(num_modes, dtype=np.float32)
    first_time = np.full(num_modes, np.nan, dtype=np.float32)
    max_t = min(horizon, obb.shape[1], mask.shape[1])
    for mode_idx in range(num_modes):
        for t_idx in range(max_t):
            state = local_states[mode_idx, t_idx]
            heading = float(state[StateIndex.HEADING])
            rear = np.array([state[StateIndex.X], state[StateIndex.Y]], dtype=np.float32)
            ego_center = rear + rear_axle_to_center * np.array(
                [math.cos(heading), math.sin(heading)], dtype=np.float32
            )
            for agent_idx in range(obb.shape[0]):
                if ignore[agent_idx] or not mask[agent_idx, t_idx]:
                    continue
                other = obb[agent_idx, t_idx]
                if not _obb_overlap(
                    ego_center,
                    heading,
                    ego_length,
                    ego_width,
                    other[:2],
                    float(other[2]),
                    float(other[3]),
                    float(other[4]),
                    margin,
                ):
                    continue
                if mode == "front":
                    rel = other[:2] - ego_center
                    lon = rel[0] * math.cos(heading) + rel[1] * math.sin(heading)
                    if lon < -0.25 * ego_length:
                        continue
                score = 0.0 if bool(is_agent[agent_idx]) else 0.5
                scores[mode_idx] = min(float(scores[mode_idx]), score)
                first_time[mode_idx] = float(t_idx) * float(args.interval_length)
                break
            if scores[mode_idx] < 1.0:
                break
    return scores, first_time


def _pairwise_obb_overlap_np(
    ego_center: np.ndarray,
    ego_heading: np.ndarray,
    ego_length: float,
    ego_width: float,
    other_obb: np.ndarray,
    margin: float,
) -> np.ndarray:
    num_modes = ego_center.shape[0]
    num_agents = other_obb.shape[0]
    if num_modes == 0 or num_agents == 0:
        return np.zeros((num_modes, num_agents), dtype=bool)

    ego_u = np.stack([np.cos(ego_heading), np.sin(ego_heading)], axis=-1).astype(np.float32)
    ego_v = np.stack([-np.sin(ego_heading), np.cos(ego_heading)], axis=-1).astype(np.float32)
    other_heading = other_obb[:, 2].astype(np.float32)
    other_u = np.stack([np.cos(other_heading), np.sin(other_heading)], axis=-1).astype(np.float32)
    other_v = np.stack([-np.sin(other_heading), np.cos(other_heading)], axis=-1).astype(np.float32)
    delta = other_obb[None, :, :2] - ego_center[:, None, :]

    ego_hl = 0.5 * float(ego_length) + float(margin)
    ego_hw = 0.5 * float(ego_width) + float(margin)
    other_hl = 0.5 * other_obb[:, 3].astype(np.float32)
    other_hw = 0.5 * other_obb[:, 4].astype(np.float32)

    axes = (
        ego_u[:, None, :],
        ego_v[:, None, :],
        other_u[None, :, :],
        other_v[None, :, :],
    )
    overlap = np.ones((num_modes, num_agents), dtype=bool)
    for axis in axes:
        sep = np.abs(np.sum(delta * axis, axis=-1))
        r_ego = ego_hl * np.abs(np.sum(ego_u[:, None, :] * axis, axis=-1))
        r_ego += ego_hw * np.abs(np.sum(ego_v[:, None, :] * axis, axis=-1))
        r_other = other_hl[None, :] * np.abs(np.sum(other_u[None, :, :] * axis, axis=-1))
        r_other += other_hw[None, :] * np.abs(np.sum(other_v[None, :, :] * axis, axis=-1))
        overlap &= sep <= (r_ego + r_other)
        if not overlap.any():
            break
    return overlap


def _approx_collision_scores_fast(
    local_states: np.ndarray,
    future_fields: Dict[str, Any],
    args: argparse.Namespace,
) -> Tuple[np.ndarray, np.ndarray]:
    obb = _to_numpy(future_fields["future_agent_obb"]).astype(np.float32)
    mask = _to_numpy(future_fields["future_agent_mask"]).astype(bool)
    is_agent = _to_numpy(future_fields["future_agent_is_agent"]).astype(bool)
    ignore = _to_numpy(future_fields.get("future_agent_ignore", np.zeros(len(obb), dtype=bool))).astype(bool)

    vehicle = get_pacifica_parameters()
    ego_length = float(vehicle.half_length) * 2.0
    ego_width = float(vehicle.half_width) * 2.0
    rear_axle_to_center = float(vehicle.rear_axle_to_center)
    margin = float(args.collision_margin_m)
    mode = str(args.approx_collision_mode)

    num_modes, horizon = local_states.shape[:2]
    scores = np.ones(num_modes, dtype=np.float32)
    first_time = np.full(num_modes, np.nan, dtype=np.float32)
    unresolved = np.ones(num_modes, dtype=bool)
    max_t = min(horizon, obb.shape[1], mask.shape[1])

    for t_idx in range(max_t):
        mode_indices = np.where(unresolved)[0]
        if mode_indices.size == 0:
            break
        agent_valid = mask[:, t_idx].astype(bool) & ~ignore
        if not agent_valid.any():
            continue

        states_t = local_states[mode_indices, t_idx]
        heading = states_t[:, StateIndex.HEADING].astype(np.float32)
        rear = states_t[:, [StateIndex.X, StateIndex.Y]].astype(np.float32)
        ego_center = rear + rear_axle_to_center * np.stack(
            [np.cos(heading), np.sin(heading)],
            axis=-1,
        ).astype(np.float32)
        other = obb[agent_valid, t_idx]
        overlap = _pairwise_obb_overlap_np(
            ego_center=ego_center,
            ego_heading=heading,
            ego_length=ego_length,
            ego_width=ego_width,
            other_obb=other,
            margin=margin,
        )
        if mode == "front" and overlap.any():
            rel = other[None, :, :2] - ego_center[:, None, :]
            lon = rel[..., 0] * np.cos(heading)[:, None] + rel[..., 1] * np.sin(heading)[:, None]
            overlap &= lon >= -0.25 * ego_length
        if not overlap.any():
            continue

        valid_agent_indices = np.where(agent_valid)[0]
        local_hit_modes = np.where(overlap.any(axis=1))[0]
        for local_idx in local_hit_modes:
            global_mode_idx = int(mode_indices[local_idx])
            hit_agent_indices = valid_agent_indices[np.where(overlap[local_idx])[0]]
            hit_scores = np.where(is_agent[hit_agent_indices], 0.0, 0.5).astype(np.float32)
            scores[global_mode_idx] = float(hit_scores.min())
            first_time[global_mode_idx] = float(t_idx) * float(args.interval_length)
            unresolved[global_mode_idx] = False
    return scores, first_time


def _nc_torch_device(args: argparse.Namespace) -> torch.device:
    requested = str(getattr(args, "nc_torch_device", "auto")).lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _approx_collision_scores_torch(
    local_states: np.ndarray,
    future_fields: Dict[str, Any],
    args: argparse.Namespace,
) -> Tuple[np.ndarray, np.ndarray]:
    with torch.inference_mode():
        device = _nc_torch_device(args)
        states = torch.as_tensor(local_states, dtype=torch.float32, device=device)
        obb = torch.as_tensor(_to_numpy(future_fields["future_agent_obb"]), dtype=torch.float32, device=device)
        mask = torch.as_tensor(_to_numpy(future_fields["future_agent_mask"]), dtype=torch.bool, device=device)
        is_agent = torch.as_tensor(_to_numpy(future_fields["future_agent_is_agent"]), dtype=torch.bool, device=device)
        ignore = torch.as_tensor(
            _to_numpy(future_fields.get("future_agent_ignore", np.zeros(int(obb.shape[0]), dtype=bool))),
            dtype=torch.bool,
            device=device,
        )

        if states.numel() == 0 or obb.numel() == 0:
            num_modes = int(states.shape[0]) if states.dim() >= 1 else 0
            return np.ones(num_modes, dtype=np.float32), np.full(num_modes, np.nan, dtype=np.float32)

        num_agents = int(obb.shape[0])
        if is_agent.numel() < num_agents:
            is_agent = torch.cat(
                [is_agent.reshape(-1), torch.zeros(num_agents - is_agent.numel(), dtype=torch.bool, device=device)],
                dim=0,
            )
        if ignore.numel() < num_agents:
            ignore = torch.cat(
                [ignore.reshape(-1), torch.zeros(num_agents - ignore.numel(), dtype=torch.bool, device=device)],
                dim=0,
            )
        is_agent = is_agent.reshape(-1)[:num_agents]
        ignore = ignore.reshape(-1)[:num_agents]

        vehicle = get_pacifica_parameters()
        ego_length = float(vehicle.half_length) * 2.0
        ego_width = float(vehicle.half_width) * 2.0
        rear_axle_to_center = float(vehicle.rear_axle_to_center)
        margin = float(args.collision_margin_m)
        mode = str(args.approx_collision_mode)

        num_modes = int(states.shape[0])
        max_t = min(int(states.shape[1]), int(obb.shape[1]), int(mask.shape[1]))
        if max_t <= 0:
            return np.ones(num_modes, dtype=np.float32), np.full(num_modes, np.nan, dtype=np.float32)

        states = states[:, :max_t]
        obb = obb[:, :max_t]
        mask = mask[:, :max_t]

        heading = states[..., StateIndex.HEADING]
        rear = states[..., [StateIndex.X, StateIndex.Y]]
        ego_u = torch.stack([torch.cos(heading), torch.sin(heading)], dim=-1)
        ego_v = torch.stack([-torch.sin(heading), torch.cos(heading)], dim=-1)
        ego_center = rear + rear_axle_to_center * ego_u

        other_heading = obb[..., 2]
        other_u = torch.stack([torch.cos(other_heading), torch.sin(other_heading)], dim=-1)
        other_v = torch.stack([-torch.sin(other_heading), torch.cos(other_heading)], dim=-1)

        ego_center_e = ego_center[:, :, None, :]
        ego_u_e = ego_u[:, :, None, :]
        ego_v_e = ego_v[:, :, None, :]
        other_center = obb[:, :, :2].permute(1, 0, 2)[None, :, :, :]
        other_u_e = other_u.permute(1, 0, 2)[None, :, :, :]
        other_v_e = other_v.permute(1, 0, 2)[None, :, :, :]
        delta = other_center - ego_center_e

        ego_hl = 0.5 * ego_length + margin
        ego_hw = 0.5 * ego_width + margin
        other_hl = (0.5 * obb[:, :, 3]).transpose(0, 1)[None, :, :]
        other_hw = (0.5 * obb[:, :, 4]).transpose(0, 1)[None, :, :]

        overlap = torch.ones((num_modes, max_t, num_agents), dtype=torch.bool, device=device)
        for axis in (ego_u_e, ego_v_e, other_u_e, other_v_e):
            sep = (delta * axis).sum(dim=-1).abs()
            r_ego = ego_hl * (ego_u_e * axis).sum(dim=-1).abs()
            r_ego = r_ego + ego_hw * (ego_v_e * axis).sum(dim=-1).abs()
            r_other = other_hl * (other_u_e * axis).sum(dim=-1).abs()
            r_other = r_other + other_hw * (other_v_e * axis).sum(dim=-1).abs()
            overlap = overlap & (sep <= (r_ego + r_other))

        agent_valid = mask.transpose(0, 1)[None, :, :] & (~ignore)[None, None, :]
        overlap = overlap & agent_valid

        if mode == "front":
            lon = (delta * ego_u_e).sum(dim=-1)
            overlap = overlap & (lon >= -0.25 * ego_length)

        mode_time_hit = overlap.any(dim=-1)
        has_hit = mode_time_hit.any(dim=-1)
        first_idx = mode_time_hit.to(torch.int64).argmax(dim=-1)
        mode_idx = torch.arange(num_modes, device=device)
        hit_at_first = overlap[mode_idx, first_idx]
        agent_score = torch.where(
            is_agent,
            torch.zeros(num_agents, dtype=torch.float32, device=device),
            torch.full((num_agents,), 0.5, dtype=torch.float32, device=device),
        )
        hit_scores = torch.where(hit_at_first, agent_score[None, :], torch.ones_like(hit_at_first, dtype=torch.float32))
        scores_t = torch.where(
            has_hit,
            hit_scores.min(dim=-1).values,
            torch.ones(num_modes, dtype=torch.float32, device=device),
        )
        first_time_t = torch.where(
            has_hit,
            first_idx.to(torch.float32) * float(args.interval_length),
            torch.full((num_modes,), float("nan"), dtype=torch.float32, device=device),
        )
        return scores_t.cpu().numpy().astype(np.float32), first_time_t.cpu().numpy().astype(np.float32)


def _parse_int_list(value: str) -> List[int]:
    return [int(part) for part in str(value).replace(",", " ").split() if part.strip()]


def _projected_ego_center(
    state: np.ndarray,
    rear_axle_to_center: float,
    delta_t: float = 0.0,
) -> Tuple[np.ndarray, float]:
    heading = float(state[StateIndex.HEADING])
    rear = np.array([state[StateIndex.X], state[StateIndex.Y]], dtype=np.float32)
    if abs(delta_t) > 1e-9:
        rear = rear + np.array(
            [state[StateIndex.VELOCITY_X], state[StateIndex.VELOCITY_Y]],
            dtype=np.float32,
        ) * float(delta_t)
    ego_center = rear + rear_axle_to_center * np.array(
        [math.cos(heading), math.sin(heading)], dtype=np.float32
    )
    return ego_center, heading


def _approx_ttc_scores(
    local_states: np.ndarray,
    future_fields: Dict[str, Any],
    args: argparse.Namespace,
) -> Tuple[np.ndarray, np.ndarray]:
    obb = _to_numpy(future_fields["future_agent_obb"]).astype(np.float32)
    mask = _to_numpy(future_fields["future_agent_mask"]).astype(bool)
    ignore = _to_numpy(future_fields.get("future_agent_ignore", np.zeros(len(obb), dtype=bool))).astype(bool)
    token_hash = _to_numpy(future_fields.get("future_agent_token_hash", np.arange(len(obb)))).reshape(-1)
    if token_hash.shape[0] == 0:
        token_hash = np.arange(len(obb))
    if token_hash.shape[0] < obb.shape[0]:
        token_hash = np.pad(token_hash, (0, obb.shape[0] - token_hash.shape[0]), mode="edge")

    vehicle = get_pacifica_parameters()
    ego_length = float(vehicle.half_length) * 2.0
    ego_width = float(vehicle.half_width) * 2.0
    rear_axle_to_center = float(vehicle.rear_axle_to_center)
    margin = float(args.ttc_margin_m)
    ttc_future_indices = _parse_int_list(str(args.ttc_future_time_indices))
    mode = str(args.approx_ttc_mode)
    stopped_speed_threshold = float(args.ttc_stopped_speed_threshold)

    num_modes, horizon = local_states.shape[:2]
    scores = np.ones(num_modes, dtype=np.float32)
    first_time = np.full(num_modes, np.nan, dtype=np.float32)
    for mode_idx in range(num_modes):
        temp_collided: set[int] = set(
            int(token_hash[agent_idx])
            for agent_idx in range(min(len(ignore), len(token_hash)))
            if bool(ignore[agent_idx])
        )
        for t_idx in range(horizon):
            state = local_states[mode_idx, t_idx]
            speed = float(np.hypot(state[StateIndex.VELOCITY_X], state[StateIndex.VELOCITY_Y]))
            if speed < stopped_speed_threshold:
                continue
            for future_idx in ttc_future_indices:
                current_t = t_idx + int(future_idx)
                if current_t < 0 or current_t >= obb.shape[1] or current_t >= mask.shape[1]:
                    continue
                delta_t = float(future_idx) * float(args.interval_length)
                ego_center, heading = _projected_ego_center(state, rear_axle_to_center, delta_t=delta_t)
                for agent_idx in range(obb.shape[0]):
                    agent_hash = int(token_hash[agent_idx]) if agent_idx < len(token_hash) else int(agent_idx)
                    if agent_hash in temp_collided or not mask[agent_idx, current_t]:
                        continue
                    other = obb[agent_idx, current_t]
                    if not _obb_overlap(
                        ego_center,
                        heading,
                        ego_length,
                        ego_width,
                        other[:2],
                        float(other[2]),
                        float(other[3]),
                        float(other[4]),
                        margin,
                    ):
                        continue
                    if mode == "front":
                        rel = other[:2] - ego_center
                        lon = rel[0] * math.cos(heading) + rel[1] * math.sin(heading)
                        if lon < -0.25 * ego_length:
                            continue
                    elif mode == "official_like":
                        ego_rear = StateSE2(
                            float(state[StateIndex.X]),
                            float(state[StateIndex.Y]),
                            float(state[StateIndex.HEADING]),
                        )
                        track_state = StateSE2(
                            float(other[0]),
                            float(other[1]),
                            float(other[2]),
                        )
                        if not is_agent_ahead(ego_rear, track_state):
                            if bool(args.ttc_official_like_allow_not_behind):
                                if is_agent_behind(ego_rear, track_state):
                                    temp_collided.add(agent_hash)
                                    continue
                            else:
                                temp_collided.add(agent_hash)
                                continue
                    scores[mode_idx] = 0.0
                    first_time[mode_idx] = float(t_idx) * float(args.interval_length)
                    break
                if scores[mode_idx] < 1.0:
                    break
            if scores[mode_idx] < 1.0:
                break
    return scores, first_time


def _ego_corners_local(state: np.ndarray) -> np.ndarray:
    vehicle = get_pacifica_parameters()
    center, heading = _projected_ego_center(state, float(vehicle.rear_axle_to_center))
    ux = np.array([math.cos(heading), math.sin(heading)], dtype=np.float32)
    uy = np.array([-math.sin(heading), math.cos(heading)], dtype=np.float32)
    half_length = float(vehicle.half_length)
    half_width = float(vehicle.half_width)
    return np.stack(
        [
            center + half_length * ux + half_width * uy,
            center - half_length * ux + half_width * uy,
            center - half_length * ux - half_width * uy,
            center + half_length * ux - half_width * uy,
        ],
        axis=0,
    )


def _target_feasible_area_mask(data: Dict[str, Any], args: argparse.Namespace) -> Optional[np.ndarray]:
    mask = data.get("feasible_area_mask")
    if mask is not None:
        arr = np.squeeze(_to_numpy(mask)).astype(bool)
        if arr.ndim == 2:
            return arr
    if not bool(args.compute_dac_mask_from_bev):
        return None
    try:
        feasible = _compute_feasible_masks(data, args)
    except Exception:
        feasible = None
    if feasible is None:
        return None
    return _to_numpy(feasible[0]).astype(bool)


def _points_in_bev_mask(points_xy: np.ndarray, mask: np.ndarray, pixel_size: float) -> np.ndarray:
    height, width = mask.shape
    row = points_xy[:, 0] / max(float(pixel_size), 1e-6)
    col = points_xy[:, 1] / max(float(pixel_size), 1e-6) + (width - 1) / 2.0
    row_i = np.round(row).astype(np.int64)
    col_i = np.round(col).astype(np.int64)
    in_bounds = (row_i >= 0) & (row_i < height) & (col_i >= 0) & (col_i < width)
    out = np.zeros(points_xy.shape[0], dtype=bool)
    valid_idx = np.where(in_bounds)[0]
    if valid_idx.size:
        out[valid_idx] = mask[row_i[valid_idx], col_i[valid_idx]]
    return out


def _approx_dac_scores(
    local_states: np.ndarray,
    data: Dict[str, Any],
    args: argparse.Namespace,
) -> Tuple[np.ndarray, bool]:
    mask = _target_feasible_area_mask(data, args)
    if mask is None:
        return np.full(local_states.shape[0], np.nan, dtype=np.float32), False
    pixel_size = float(args.map_pixel_size)
    scores = np.ones(local_states.shape[0], dtype=np.float32)
    for mode_idx in range(local_states.shape[0]):
        for t_idx in range(local_states.shape[1]):
            corners = _ego_corners_local(local_states[mode_idx, t_idx])
            if not _points_in_bev_mask(corners, mask, pixel_size).all():
                scores[mode_idx] = 0.0
                break
    return scores, True


def _summarize(values: np.ndarray) -> Dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"n": 0}
    return {
        "n": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def _binary_stats(true_collision: np.ndarray, pred_collision: np.ndarray) -> Dict[str, float]:
    true_collision = true_collision.astype(bool)
    pred_collision = pred_collision.astype(bool)
    tp = int(np.logical_and(true_collision, pred_collision).sum())
    fp = int(np.logical_and(~true_collision, pred_collision).sum())
    fn = int(np.logical_and(true_collision, ~pred_collision).sum())
    tn = int(np.logical_and(~true_collision, ~pred_collision).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float((tp + tn) / max(tp + fp + fn + tn, 1)),
    }


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _build_nc_case_tables(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    score_mismatch: List[Dict[str, Any]] = []
    false_positive: List[Dict[str, Any]] = []
    false_negative: List[Dict[str, Any]] = []
    grouped: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for row in rows:
        official_collision = int(row.get("official_collision", 0))
        approx_collision = int(row.get("approx_collision", 0))
        abs_error = float(row.get("abs_no_collision_error", 0.0))
        if official_collision == 0 and approx_collision == 1:
            error_type = "false_positive"
        elif official_collision == 1 and approx_collision == 0:
            error_type = "false_negative"
        elif abs_error > 1e-6:
            error_type = "score_mismatch"
        else:
            continue

        case = dict(row)
        case["nc_error_type"] = error_type
        score_mismatch.append(case)
        if error_type == "false_positive":
            false_positive.append(case)
        elif error_type == "false_negative":
            false_negative.append(case)

        key = (str(row.get("token", "")), str(row.get("path", "")))
        entry = grouped.setdefault(
            key,
            {
                "token": key[0],
                "path": key[1],
                "nc_error_rows": 0,
                "false_positive": 0,
                "false_negative": 0,
                "score_mismatch": 0,
                "max_abs_no_collision_error": 0.0,
                "min_official_no_collision": 1.0,
                "min_approx_no_collision": 1.0,
                "mode_labels": [],
            },
        )
        entry["nc_error_rows"] += 1
        entry[error_type] += 1
        entry["max_abs_no_collision_error"] = max(
            float(entry["max_abs_no_collision_error"]), abs_error
        )
        entry["min_official_no_collision"] = min(
            float(entry["min_official_no_collision"]), float(row.get("official_no_collision", 1.0))
        )
        entry["min_approx_no_collision"] = min(
            float(entry["min_approx_no_collision"]), float(row.get("approx_no_collision", 1.0))
        )
        mode_labels = entry["mode_labels"]
        if len(mode_labels) < 12:
            mode_labels.append(str(row.get("mode_label", row.get("mode_idx", ""))))

    for entry in grouped.values():
        entry["mode_labels"] = ",".join(entry["mode_labels"])

    score_mismatch.sort(key=lambda item: float(item.get("abs_no_collision_error", 0.0)), reverse=True)
    false_positive.sort(key=lambda item: str(item.get("token", "")))
    false_negative.sort(key=lambda item: str(item.get("token", "")))
    by_token = sorted(
        grouped.values(),
        key=lambda item: (
            -int(item["nc_error_rows"]),
            -float(item["max_abs_no_collision_error"]),
            str(item["token"]),
        ),
    )
    return {
        "score_mismatch": score_mismatch,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "by_token": by_token,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument(
        "--metric-cache-path",
        required=True,
        nargs="+",
        help="One or more metric cache roots. Each root must contain metadata/*.csv.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--roots", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument(
        "--check-cache-overlap",
        action="store_true",
        help="Only print target/metric token overlap diagnostics, then exit.",
    )
    parser.add_argument(
        "--profile-official-components",
        action="store_true",
        help="Time official PDM scorer substeps: reset, ego_area, NC, DAC, TTC, progress, comfort, etc.",
    )
    parser.add_argument(
        "--trajectory-set",
        choices=["gt", "best", "gt_best", "all", "gt_all"],
        default="gt_best",
    )
    parser.add_argument("--max-candidates", type=int, default=16)
    parser.add_argument(
        "--use-existing-fast-fields",
        action="store_true",
        help="Use future_agent_* fields from transfuser_target.gz if present.",
    )
    parser.add_argument("--num-poses", type=int, default=40)
    parser.add_argument("--interval-length", type=float, default=0.1)
    parser.add_argument("--trajectory-interval", type=float, default=0.5)
    parser.add_argument("--observation-interval", type=float, default=0.1)
    parser.add_argument(
        "--future-agent-extra-poses",
        type=int,
        default=9,
        help="Extra cached observation steps for TTC lookahead. Collision only needs 0; TTC needs 9 for 0.9s lookahead.",
    )
    parser.add_argument("--max-agents", type=int, default=64)
    parser.add_argument("--future-range-m", type=float, default=90.0)
    parser.add_argument("--include-red-lights", action="store_true")
    parser.add_argument("--collision-margin-m", type=float, default=0.0)
    parser.add_argument(
        "--approx-nc-impl",
        choices=["torch", "fast", "slow"],
        default="torch",
        help="torch uses batched tensor OBB overlap; fast uses NumPy; slow keeps the original Python-loop reference.",
    )
    parser.add_argument(
        "--nc-torch-device",
        default="auto",
        help="Device for --approx-nc-impl torch. Use auto, cuda, cuda:0, or cpu.",
    )
    parser.add_argument(
        "--approx-collision-mode",
        choices=["any", "front"],
        default="any",
        help="'any' is an occupancy upper bound; 'front' ignores likely rear impacts.",
    )
    parser.add_argument("--ttc-margin-m", type=float, default=0.0)
    parser.add_argument("--ttc-future-time-indices", default="0,3,6,9")
    parser.add_argument("--ttc-stopped-speed-threshold", type=float, default=5e-3)
    parser.add_argument(
        "--approx-ttc-mode",
        choices=["any", "front", "official_like"],
        default="official_like",
        help="official_like adds ahead/behind filtering and temporary track de-duplication.",
    )
    parser.add_argument(
        "--ttc-official-like-allow-not-behind",
        action="store_true",
        help=(
            "Approximate the official special-area branch by accepting lateral/non-behind "
            "overlaps. This usually improves recall but can increase false positives."
        ),
    )
    parser.add_argument("--map-pixel-size", type=float, default=0.25)
    parser.add_argument("--compute-dac-mask-from-bev", action="store_true", default=True)
    parser.add_argument("--no-compute-dac-mask-from-bev", dest="compute_dac_mask_from_bev", action="store_false")
    parser.add_argument("--num-bev-classes", type=int, default=7)
    parser.add_argument("--road-label", type=int, default=1)
    parser.add_argument("--centerline-label", type=int, default=3)
    parser.add_argument("--feasible-seed-rows", type=int, default=12)
    parser.add_argument("--feasible-seed-cols", type=int, default=16)
    parser.add_argument("--feasible-max-iters", type=int, default=0)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_paths = _load_metric_cache_indices(args.metric_cache_path)
    target_paths = _iter_target_paths(Path(args.cache_root), args.roots, int(args.limit))
    if not target_paths:
        raise FileNotFoundError(f"No transfuser_target.gz found under {args.cache_root}")
    target_tokens = [path.parent.name for path in target_paths]
    matched_tokens = [token for token in target_tokens if token in metric_paths]
    missing_tokens = [token for token in target_tokens if token not in metric_paths]
    print(
        "cache_overlap "
        f"target_files={len(target_paths)} "
        f"metric_tokens={len(metric_paths)} "
        f"matched={len(matched_tokens)} "
        f"missing_metric={len(missing_tokens)}"
    )
    if missing_tokens:
        print("missing_metric_examples: " + ", ".join(missing_tokens[:10]))
    if bool(args.check_cache_overlap):
        return

    proposal_sampling = TrajectorySampling(
        num_poses=int(args.num_poses),
        interval_length=float(args.interval_length),
    )
    simulator = PDMSimulator(proposal_sampling)
    scorer = PDMScorer(proposal_sampling)

    rows: List[Dict[str, Any]] = []
    sample_rows: List[Dict[str, Any]] = []
    timing: Dict[str, float] = {
        "simulate_s": 0.0,
        "official_full_pdm_s": 0.0,
        "official_dac_only_s": 0.0,
        "approx_nc_s": 0.0,
        "approx_ttc_s": 0.0,
        "approx_dac_s": 0.0,
    }
    official_component_timing: Dict[str, float] = {}
    dac_cache_storage: Dict[str, float] = {}
    counters: Dict[str, int] = {
        "files": 0,
        "scored_files": 0,
        "missing_metric": 0,
        "missing_trajectory": 0,
        "errors": 0,
    }

    for target_path in tqdm(target_paths, desc="compare fast collision"):
        counters["files"] += 1
        token = target_path.parent.name
        metric_path = metric_paths.get(token)
        if metric_path is None:
            counters["missing_metric"] += 1
            continue
        try:
            data = _load_gzip_pickle(target_path)
            labels, trajectories = _select_trajectories(data, args)
            if trajectories.size == 0:
                counters["missing_trajectory"] += 1
                continue
            metric_cache = _load_metric_cache(Path(metric_path))
            t0 = time.perf_counter()
            states = _simulated_states(
                metric_cache,
                trajectories,
                simulator,
                trajectory_interval=float(args.trajectory_interval),
            )
            timing["simulate_s"] += time.perf_counter() - t0
            t0 = time.perf_counter()
            if bool(args.profile_official_components):
                official_scores, official_components, component_timing = _official_components_profiled(
                    metric_cache, states, scorer
                )
                for key, value in component_timing.items():
                    official_component_timing[key] = official_component_timing.get(key, 0.0) + float(value)
            else:
                official_scores, official_components = _official_components(metric_cache, states, scorer)
            timing["official_full_pdm_s"] += time.perf_counter() - t0
            cached_ego_area_dac, cached_ego_area_stats = _cached_dac_from_ego_area(scorer)
            for key, value in cached_ego_area_stats.items():
                dac_cache_storage[key] = dac_cache_storage.get(key, 0.0) + float(value)
            t0 = time.perf_counter()
            official_dac_only = _official_dac_only_scores(metric_cache, states)
            timing["official_dac_only_s"] += time.perf_counter() - t0
            if (
                bool(args.use_existing_fast_fields)
                and "future_agent_obb" in data
                and "future_agent_mask" in data
                and "future_agent_is_agent" in data
            ):
                future_fields = data
                future_field_source = "target_cache"
            else:
                future_args = argparse.Namespace(**vars(args))
                future_args.num_poses = int(args.num_poses) + max(0, int(args.future_agent_extra_poses))
                future_fields = _collect_future_agents(metric_cache, future_args)
                future_field_source = "metric_cache_online"
            local_states = _global_states_to_local(states, metric_cache)
            t0 = time.perf_counter()
            if str(args.approx_nc_impl) == "slow":
                approx_collision, approx_collision_time = _approx_collision_scores(
                    local_states, future_fields, args
                )
            elif str(args.approx_nc_impl) == "fast":
                approx_collision, approx_collision_time = _approx_collision_scores_fast(
                    local_states, future_fields, args
                )
            else:
                approx_collision, approx_collision_time = _approx_collision_scores_torch(
                    local_states, future_fields, args
                )
            timing["approx_nc_s"] += time.perf_counter() - t0
            t0 = time.perf_counter()
            approx_ttc, approx_ttc_time = _approx_ttc_scores(local_states, future_fields, args)
            timing["approx_ttc_s"] += time.perf_counter() - t0
            t0 = time.perf_counter()
            approx_dac, approx_dac_valid = _approx_dac_scores(local_states, data, args)
            timing["approx_dac_s"] += time.perf_counter() - t0
            counters["scored_files"] += 1
        except Exception as exc:  # pragma: no cover - depends on local data/runtime
            counters["errors"] += 1
            sample_rows.append(
                {
                    "token": token,
                    "path": str(target_path),
                    "status": f"error:{type(exc).__name__}:{exc}",
                }
            )
            continue

        true_collision_score = official_components[:, 0]
        true_dac = official_components[:, 1]
        true_ttc = official_components[:, 3]
        for mode_idx, label in enumerate(labels):
            exact_no_collision = float(true_collision_score[mode_idx])
            approx_no_collision = float(approx_collision[mode_idx])
            exact_ttc = float(true_ttc[mode_idx])
            approx_ttc_value = float(approx_ttc[mode_idx])
            exact_dac = float(true_dac[mode_idx])
            approx_dac_value = float(approx_dac[mode_idx]) if approx_dac_valid else math.nan
            cached_ego_area_dac_value = (
                float(cached_ego_area_dac[mode_idx])
                if cached_ego_area_dac is not None and mode_idx < len(cached_ego_area_dac)
                else math.nan
            )
            official_dac_only_value = (
                float(official_dac_only[mode_idx])
                if mode_idx < len(official_dac_only)
                else math.nan
            )
            rows.append(
                {
                    "token": token,
                    "path": str(target_path),
                    "mode_label": label,
                    "mode_idx": int(mode_idx),
                    "future_field_source": future_field_source,
                    "official_score": float(official_scores[mode_idx]),
                    "official_no_collision": exact_no_collision,
                    "official_dac": exact_dac,
                    "official_dac_only": official_dac_only_value,
                    "official_ttc": float(true_ttc[mode_idx]),
                    "cached_ego_area_dac": cached_ego_area_dac_value,
                    "approx_no_collision": approx_no_collision,
                    "approx_collision_time_s": float(approx_collision_time[mode_idx])
                    if math.isfinite(float(approx_collision_time[mode_idx]))
                    else math.nan,
                    "approx_ttc": approx_ttc_value,
                    "approx_ttc_time_s": float(approx_ttc_time[mode_idx])
                    if math.isfinite(float(approx_ttc_time[mode_idx]))
                    else math.nan,
                    "approx_dac": approx_dac_value,
                    "abs_no_collision_error": abs(approx_no_collision - exact_no_collision),
                    "abs_ttc_error": abs(approx_ttc_value - exact_ttc),
                    "abs_dac_error": abs(approx_dac_value - exact_dac)
                    if math.isfinite(approx_dac_value)
                    else math.nan,
                    "abs_cached_ego_area_dac_error": abs(cached_ego_area_dac_value - exact_dac)
                    if math.isfinite(cached_ego_area_dac_value)
                    else math.nan,
                    "abs_official_dac_only_error": abs(official_dac_only_value - exact_dac)
                    if math.isfinite(official_dac_only_value)
                    else math.nan,
                    "official_collision": int(exact_no_collision < 1.0),
                    "approx_collision": int(approx_no_collision < 1.0),
                    "official_agent_collision": int(exact_no_collision == 0.0),
                    "approx_agent_collision": int(approx_no_collision == 0.0),
                    "official_ttc_violation": int(exact_ttc < 1.0),
                    "approx_ttc_violation": int(approx_ttc_value < 1.0),
                    "official_dac_violation": int(exact_dac < 1.0),
                    "official_dac_only_violation": int(official_dac_only_value < 1.0)
                    if math.isfinite(official_dac_only_value)
                    else "",
                    "cached_ego_area_dac_violation": int(cached_ego_area_dac_value < 1.0)
                    if math.isfinite(cached_ego_area_dac_value)
                    else "",
                    "approx_dac_violation": int(approx_dac_value < 1.0)
                    if math.isfinite(approx_dac_value)
                    else "",
                }
            )

    if not rows:
        raise RuntimeError(f"No comparable rows. counters={counters}")

    exact = np.asarray([row["official_no_collision"] for row in rows], dtype=np.float32)
    approx = np.asarray([row["approx_no_collision"] for row in rows], dtype=np.float32)
    exact_ttc = np.asarray([row["official_ttc"] for row in rows], dtype=np.float32)
    approx_ttc = np.asarray([row["approx_ttc"] for row in rows], dtype=np.float32)
    exact_dac = np.asarray([row["official_dac"] for row in rows], dtype=np.float32)
    official_dac_only = np.asarray([row["official_dac_only"] for row in rows], dtype=np.float32)
    approx_dac = np.asarray([row["approx_dac"] for row in rows], dtype=np.float32)
    cached_ego_area_dac = np.asarray([row["cached_ego_area_dac"] for row in rows], dtype=np.float32)
    abs_err = np.abs(approx - exact)
    ttc_abs_err = np.abs(approx_ttc - exact_ttc)
    dac_valid = np.isfinite(approx_dac) & np.isfinite(exact_dac)
    dac_abs_err = np.abs(approx_dac[dac_valid] - exact_dac[dac_valid])
    official_dac_only_valid = np.isfinite(official_dac_only) & np.isfinite(exact_dac)
    official_dac_only_abs_err = np.abs(
        official_dac_only[official_dac_only_valid] - exact_dac[official_dac_only_valid]
    )
    cached_ego_area_dac_valid = np.isfinite(cached_ego_area_dac) & np.isfinite(exact_dac)
    cached_ego_area_dac_abs_err = np.abs(
        cached_ego_area_dac[cached_ego_area_dac_valid] - exact_dac[cached_ego_area_dac_valid]
    )
    nc_case_tables = _build_nc_case_tables(rows)
    scored_files = max(int(counters.get("scored_files", 0)), 1)
    rows_count = max(int(len(rows)), 1)
    timing_summary = {
        **{key: float(value) for key, value in timing.items()},
        "official_full_pdm_ms_per_file": 1000.0 * timing["official_full_pdm_s"] / scored_files,
        "official_full_pdm_ms_per_mode": 1000.0 * timing["official_full_pdm_s"] / rows_count,
        "official_dac_only_ms_per_file": 1000.0 * timing["official_dac_only_s"] / scored_files,
        "official_dac_only_ms_per_mode": 1000.0 * timing["official_dac_only_s"] / rows_count,
        "approx_nc_ms_per_file": 1000.0 * timing["approx_nc_s"] / scored_files,
        "approx_nc_ms_per_mode": 1000.0 * timing["approx_nc_s"] / rows_count,
        "approx_nc_vs_official_full_speedup": timing["official_full_pdm_s"] / max(timing["approx_nc_s"], 1e-12),
        "approx_ttc_ms_per_file": 1000.0 * timing["approx_ttc_s"] / scored_files,
        "approx_ttc_ms_per_mode": 1000.0 * timing["approx_ttc_s"] / rows_count,
        "approx_dac_ms_per_file": 1000.0 * timing["approx_dac_s"] / scored_files,
        "approx_dac_ms_per_mode": 1000.0 * timing["approx_dac_s"] / rows_count,
    }
    official_component_total = sum(official_component_timing.values())
    if official_component_total > 0.0:
        timing_summary["official_component_ms_per_file"] = {
            key: 1000.0 * value / scored_files for key, value in official_component_timing.items()
        }
        timing_summary["official_component_percent"] = {
            key: 100.0 * value / official_component_total for key, value in official_component_timing.items()
        }
    dac_cache_files = max(float(dac_cache_storage.get("files", 0.0)), 1.0)
    dac_cache_storage_summary = {
        key: float(value) / dac_cache_files
        for key, value in dac_cache_storage.items()
        if key != "files"
    }
    dac_cache_storage_summary["files"] = int(dac_cache_storage.get("files", 0.0))
    summary = {
        "counters": counters,
        "timing": timing_summary,
        "settings": {
            "trajectory_set": args.trajectory_set,
            "max_candidates": int(args.max_candidates),
            "num_poses": int(args.num_poses),
            "interval_length": float(args.interval_length),
            "trajectory_interval": float(args.trajectory_interval),
            "metric_cache_path": list(args.metric_cache_path),
            "profile_official_components": bool(args.profile_official_components),
            "future_agent_extra_poses": int(args.future_agent_extra_poses),
            "max_agents": int(args.max_agents),
            "future_range_m": float(args.future_range_m),
            "collision_margin_m": float(args.collision_margin_m),
            "approx_nc_impl": str(args.approx_nc_impl),
            "nc_torch_device": str(_nc_torch_device(args)) if str(args.approx_nc_impl) == "torch" else None,
            "approx_collision_mode": args.approx_collision_mode,
            "ttc_margin_m": float(args.ttc_margin_m),
            "approx_ttc_mode": args.approx_ttc_mode,
            "ttc_official_like_allow_not_behind": bool(args.ttc_official_like_allow_not_behind),
            "map_pixel_size": float(args.map_pixel_size),
        },
        "rows": int(len(rows)),
        "official_collision_rate": float(np.mean(exact < 1.0)),
        "approx_collision_rate": float(np.mean(approx < 1.0)),
        "no_collision_mae": float(np.mean(abs_err)),
        "no_collision_error": _summarize(abs_err),
        "collision_binary": _binary_stats(exact < 1.0, approx < 1.0),
        "agent_collision_binary": _binary_stats(exact == 0.0, approx == 0.0),
        "nc_error_cases": {
            "score_mismatch_rows": len(nc_case_tables["score_mismatch"]),
            "false_positive_rows": len(nc_case_tables["false_positive"]),
            "false_negative_rows": len(nc_case_tables["false_negative"]),
            "tokens_with_errors": len(nc_case_tables["by_token"]),
        },
        "official_ttc_violation_rate": float(np.mean(exact_ttc < 1.0)),
        "approx_ttc_violation_rate": float(np.mean(approx_ttc < 1.0)),
        "ttc_mae": float(np.mean(ttc_abs_err)),
        "ttc_error": _summarize(ttc_abs_err),
        "ttc_binary": _binary_stats(exact_ttc < 1.0, approx_ttc < 1.0),
        "dac_valid_rows": int(dac_valid.sum()),
        "official_dac_violation_rate": float(np.mean(exact_dac[dac_valid] < 1.0)) if dac_valid.any() else math.nan,
        "approx_dac_violation_rate": float(np.mean(approx_dac[dac_valid] < 1.0)) if dac_valid.any() else math.nan,
        "dac_mae": float(np.mean(dac_abs_err)) if dac_abs_err.size else math.nan,
        "dac_error": _summarize(dac_abs_err),
        "dac_binary": _binary_stats(exact_dac[dac_valid] < 1.0, approx_dac[dac_valid] < 1.0)
        if dac_valid.any()
        else {},
        "official_dac_only_valid_rows": int(official_dac_only_valid.sum()),
        "official_dac_only_mae": float(np.mean(official_dac_only_abs_err))
        if official_dac_only_abs_err.size
        else math.nan,
        "official_dac_only_error": _summarize(official_dac_only_abs_err),
        "official_dac_only_binary": _binary_stats(
            exact_dac[official_dac_only_valid] < 1.0,
            official_dac_only[official_dac_only_valid] < 1.0,
        )
        if official_dac_only_valid.any()
        else {},
        "cached_ego_area_dac_valid_rows": int(cached_ego_area_dac_valid.sum()),
        "cached_ego_area_dac_mae": float(np.mean(cached_ego_area_dac_abs_err))
        if cached_ego_area_dac_abs_err.size
        else math.nan,
        "cached_ego_area_dac_error": _summarize(cached_ego_area_dac_abs_err),
        "cached_ego_area_dac_binary": _binary_stats(
            exact_dac[cached_ego_area_dac_valid] < 1.0,
            cached_ego_area_dac[cached_ego_area_dac_valid] < 1.0,
        )
        if cached_ego_area_dac_valid.any()
        else {},
        "cached_ego_area_dac_storage_per_file": dac_cache_storage_summary,
    }

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    _write_csv(output_dir / "per_mode.csv", rows)
    cached_dac_mismatch_rows = [
        row
        for row in rows
        if math.isfinite(float(row.get("abs_cached_ego_area_dac_error", math.nan)))
        and float(row.get("abs_cached_ego_area_dac_error", 0.0)) > 1e-6
    ]
    official_dac_only_mismatch_rows = [
        row
        for row in rows
        if math.isfinite(float(row.get("abs_official_dac_only_error", math.nan)))
        and float(row.get("abs_official_dac_only_error", 0.0)) > 1e-6
    ]
    if official_dac_only_mismatch_rows:
        _write_csv(output_dir / "dac_official_only_mismatch.csv", official_dac_only_mismatch_rows)
    if cached_dac_mismatch_rows:
        _write_csv(output_dir / "dac_cached_ego_area_mismatch.csv", cached_dac_mismatch_rows)
    if nc_case_tables["score_mismatch"]:
        _write_csv(output_dir / "nc_score_mismatch.csv", nc_case_tables["score_mismatch"])
    if nc_case_tables["false_positive"]:
        _write_csv(output_dir / "nc_false_positive.csv", nc_case_tables["false_positive"])
    if nc_case_tables["false_negative"]:
        _write_csv(output_dir / "nc_false_negative.csv", nc_case_tables["false_negative"])
    if nc_case_tables["by_token"]:
        _write_csv(output_dir / "nc_error_by_token.csv", nc_case_tables["by_token"])
    if sample_rows:
        _write_csv(output_dir / "errors.csv", sample_rows)

    rows_per_scored_file = len(rows) / max(counters["scored_files"], 1)
    print(
        f"rows={len(rows)} files={counters['files']} scored_files={counters['scored_files']} "
        f"rows_per_scored_file={rows_per_scored_file:.2f}"
    )
    print(
        "counters "
        f"missing_metric={counters['missing_metric']} "
        f"missing_trajectory={counters['missing_trajectory']} "
        f"errors={counters['errors']}"
    )
    print(
        "settings "
        f"trajectory_set={args.trajectory_set} max_candidates={int(args.max_candidates)} "
        f"approx_nc_impl={args.approx_nc_impl} nc_torch_device="
        f"{str(_nc_torch_device(args)) if str(args.approx_nc_impl) == 'torch' else 'n/a'}"
    )
    if sample_rows:
        print("error_samples:")
        for sample in sample_rows[:5]:
            print(f"  {sample.get('token')}: {sample.get('status')}")
    print(
        "no_collision "
        f"mae={summary['no_collision_mae']:.6f} "
        f"official_collision_rate={summary['official_collision_rate']:.4f} "
        f"approx_collision_rate={summary['approx_collision_rate']:.4f}"
    )
    binary = summary["collision_binary"]
    print(
        "collision_binary "
        f"precision={binary['precision']:.4f} recall={binary['recall']:.4f} "
        f"f1={binary['f1']:.4f} fp={binary['fp']} fn={binary['fn']}"
    )
    print(
        "nc_error_cases "
        f"score_mismatch={len(nc_case_tables['score_mismatch'])} "
        f"false_positive={len(nc_case_tables['false_positive'])} "
        f"false_negative={len(nc_case_tables['false_negative'])} "
        f"tokens={len(nc_case_tables['by_token'])}"
    )
    print(
        "timing "
        f"official_full_pdm={timing_summary['official_full_pdm_ms_per_file']:.3f}ms/file "
        f"approx_nc={timing_summary['approx_nc_ms_per_file']:.3f}ms/file "
        f"speedup={timing_summary['approx_nc_vs_official_full_speedup']:.1f}x"
    )
    if official_component_total > 0.0:
        print("official_component_timing_ms_per_file:")
        for key, value in timing_summary["official_component_ms_per_file"].items():
            percent = timing_summary["official_component_percent"][key]
            print(f"  {key}: {value:.3f}ms/file ({percent:.1f}%)")
    ttc_binary = summary["ttc_binary"]
    print(
        "ttc_binary "
        f"precision={ttc_binary['precision']:.4f} recall={ttc_binary['recall']:.4f} "
        f"f1={ttc_binary['f1']:.4f} fp={ttc_binary['fp']} fn={ttc_binary['fn']}"
    )
    dac_binary = summary.get("dac_binary", {})
    if dac_binary:
        print(
            "dac_binary "
            f"precision={dac_binary['precision']:.4f} recall={dac_binary['recall']:.4f} "
            f"f1={dac_binary['f1']:.4f} fp={dac_binary['fp']} fn={dac_binary['fn']}"
        )
    official_dac_only_binary = summary.get("official_dac_only_binary", {})
    if official_dac_only_binary:
        print(
            "official_dac_only "
            f"mae={summary['official_dac_only_mae']:.6f} "
            f"precision={official_dac_only_binary['precision']:.4f} "
            f"recall={official_dac_only_binary['recall']:.4f} "
            f"f1={official_dac_only_binary['f1']:.4f} "
            f"fp={official_dac_only_binary['fp']} fn={official_dac_only_binary['fn']} "
            f"mismatch={len(official_dac_only_mismatch_rows)} "
            f"time={timing_summary['official_dac_only_ms_per_file']:.3f}ms/file"
        )
    cached_dac_binary = summary.get("cached_ego_area_dac_binary", {})
    if cached_dac_binary:
        print(
            "cached_ego_area_dac "
            f"mae={summary['cached_ego_area_dac_mae']:.6f} "
            f"precision={cached_dac_binary['precision']:.4f} "
            f"recall={cached_dac_binary['recall']:.4f} "
            f"f1={cached_dac_binary['f1']:.4f} "
            f"fp={cached_dac_binary['fp']} fn={cached_dac_binary['fn']} "
            f"mismatch={len(cached_dac_mismatch_rows)}"
        )
        storage = summary["cached_ego_area_dac_storage_per_file"]
        print(
            "cached_dac_storage_per_file "
            f"full_ego_area_bool={storage.get('full_ego_area_bool_bytes', math.nan):.1f}B "
            f"non_drivable_mask_bool={storage.get('non_drivable_mask_bool_bytes', math.nan):.1f}B "
            f"dac_label_uint8={storage.get('dac_label_uint8_bytes', math.nan):.1f}B "
            f"full_ego_area_bitpacked={storage.get('full_ego_area_bitpacked_bytes', math.nan):.1f}B "
            f"non_drivable_bitpacked={storage.get('non_drivable_mask_bitpacked_bytes', math.nan):.1f}B"
        )
    print(f"wrote: {output_dir / 'summary.json'}")
    print(f"wrote: {output_dir / 'per_mode.csv'}")


if __name__ == "__main__":
    main()
