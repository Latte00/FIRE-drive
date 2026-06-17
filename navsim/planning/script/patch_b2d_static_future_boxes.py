"""
Patch existing Bench2Drive DiffusionDrive targets with collidable static obstacle boxes.

This script is intentionally narrow: by default it only updates these fields in
existing transfuser_target.gz files:

  - future_agent_boxes
  - future_agent_boxes_mask

With --patch-target-fields it fills target-side fields introduced by the full
B2D cache builder when they are missing. With --patch-bev it also recomputes the
BEV semantic/feasible target fields using the current B2D rasterization logic.

With --patch-candidates it also updates:

  - trajectory_candidates
  - trajectory_candidates_mask

It does not write fake candidate PDM scores unless --write-debug-pdm-placeholders
is explicitly passed.
Full lane-direction BEV targets are only written when --write-lane-direction-bev
is passed; otherwise only compact adjacent-lane occupancy flags are cached.

It does not rebuild image/lidar features.
"""

from __future__ import annotations

import argparse
import gzip
import multiprocessing as mp
import pickle
import random
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

from navsim.planning.script.run_b2d_diffusiondrive_caching import (
    B2D_WRONG_DIRECTION_DISABLED_SCENARIOS,
    build_agent_targets,
    build_debug_pdm_placeholders,
    build_future_agent_boxes,
    build_future_trajectory,
    build_lane_direction_bev,
    build_route_lattice_candidates,
    build_simple_candidates,
    build_static_map_bev,
    build_velocity_bev,
    compute_feasible_masks,
    diffusiondrive_lidar2world,
    infer_wrong_direction_enabled,
    is_same_log,
    load_pickle,
    load_route_scenarios,
    overlay_agents_on_bev,
    safe_token,
)
from navsim.planning.script.visualize_b2d_candidate_generation import (
    build_preview_candidates,
    build_scored_candidates,
    candidate_wrong_lane_score,
    is_near_traffic_light_trigger,
)


_WORKER_INFOS = None
_WORKER_MAP_INFOS = None
_WORKER_ARGS = None
_WORKER_SPLIT = None


def load_target(path: Path) -> Dict[str, object]:
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def dump_target(path: Path, target: Dict[str, object]) -> None:
    with gzip.open(path, "wb", compresslevel=1) as f:
        pickle.dump(target, f)


def _infer_box_shape(
    target: Dict[str, object],
    default_future_frames: int,
    default_max_future_agents: int,
) -> Tuple[int, int]:
    boxes = target.get("future_agent_boxes")
    if torch.is_tensor(boxes) and boxes.dim() >= 2:
        return int(boxes.shape[1]), int(boxes.shape[0])
    return int(default_future_frames), int(default_max_future_agents)


def _tensor_equal(old_value: object, new_value: torch.Tensor) -> bool:
    if not torch.is_tensor(old_value):
        return False
    if tuple(old_value.shape) != tuple(new_value.shape):
        return False
    if old_value.dtype != new_value.dtype:
        old_value = old_value.to(dtype=new_value.dtype)
    if new_value.dtype == torch.bool:
        return bool(torch.equal(old_value.bool(), new_value.bool()))
    if not new_value.dtype.is_floating_point:
        return bool(torch.equal(old_value, new_value))
    return bool(torch.allclose(old_value, new_value))


def _infer_candidate_count(target: Dict[str, object], requested: int) -> int:
    if requested > 0:
        return int(requested)
    candidates = target.get("trajectory_candidates")
    if torch.is_tensor(candidates) and candidates.dim() >= 1:
        return int(candidates.shape[0])
    return 0


def _visualize_lane_occupancy_allowed(
    gt: np.ndarray,
    lane_direction_bev: torch.Tensor,
    lane_direction_mask: torch.Tensor,
    wrong_direction_enabled: bool,
    args: argparse.Namespace,
) -> bool:
    if not bool(wrong_direction_enabled):
        return True
    if not bool(getattr(args, "wrong_lane_allow_from_gt", True)):
        return False
    gt_wrong_lane_score = candidate_wrong_lane_score(
        gt,
        lane_direction_bev=lane_direction_bev,
        lane_direction_mask=lane_direction_mask,
        threshold=float(args.wrong_lane_cos_threshold),
    )
    return gt_wrong_lane_score < float(args.gt_wrong_lane_allow_threshold)


def _build_visualize_candidates(
    target: Dict[str, object],
    info: Dict[str, object],
    map_infos: Optional[Dict[str, object]],
    future_agent_boxes: torch.Tensor,
    future_agent_boxes_mask: torch.Tensor,
    args: argparse.Namespace,
    rng: random.Random,
) -> Optional[Dict[str, object]]:
    if map_infos is None:
        return None
    trajectory = target.get("trajectory")
    feasible_area = target.get("feasible_area_mask")
    if not torch.is_tensor(trajectory) or not torch.is_tensor(feasible_area):
        return None
    if bool(getattr(args, "skip_candidates_near_traffic_light", False)) and is_near_traffic_light_trigger(
        info,
        map_infos,
        distance_m=float(getattr(args, "traffic_light_skip_distance_m", 12.0)),
        include_stop_sign=bool(getattr(args, "traffic_light_skip_include_stop_sign", False)),
    ):
        return _build_empty_candidate_payload(
            target=target,
            info=info,
            args=args,
            reason="near_traffic_light",
        )

    lane_direction_bev, lane_direction_mask = build_lane_direction_bev(
        info,
        map_infos,
        radius_m=args.lane_direction_radius_m,
        search_radius=args.candidate_route_search_radius,
    )
    wrong_direction_enabled, scenario_type = infer_wrong_direction_enabled(info, args)
    gt = trajectory.detach().cpu().numpy().astype(np.float32)
    adjacent_lane_occupancy_allowed = _visualize_lane_occupancy_allowed(
        gt=gt,
        lane_direction_bev=lane_direction_bev,
        lane_direction_mask=lane_direction_mask,
        wrong_direction_enabled=bool(wrong_direction_enabled),
        args=args,
    )
    batch = build_preview_candidates(
        gt,
        info,
        map_infos,
        args,
        rng,
        lane_direction_bev=lane_direction_bev,
        lane_direction_mask=lane_direction_mask,
        wrong_direction_enabled=bool(wrong_direction_enabled),
    )
    scored = build_scored_candidates(
        batch=batch,
        gt=gt,
        feasible_area=feasible_area,
        future_agent_boxes=future_agent_boxes,
        future_agent_boxes_mask=future_agent_boxes_mask,
        lane_direction_bev=lane_direction_bev,
        lane_direction_mask=lane_direction_mask,
        wrong_direction_enabled=bool(wrong_direction_enabled),
        args=args,
    )
    selected_idx = scored.selected_idx.astype(np.int64, copy=False)
    topk = max(int(args.topk), 0)
    num_selected = min(int(selected_idx.shape[0]), topk)
    if num_selected <= 0 or topk <= 0:
        return _build_empty_candidate_payload(
            target=target,
            info=info,
            args=args,
            reason="no_topk_candidate",
        )

    num_steps = int(batch.trajectories.shape[1])
    candidate_arr = np.zeros((topk, num_steps, 3), dtype=np.float32)
    candidate_mask = np.zeros((topk,), dtype=bool)
    score_arr = np.full((topk,), np.nan, dtype=np.float32)
    component_arr = np.full((topk, 6), np.nan, dtype=np.float32)
    selected_idx_arr = np.full((topk,), -1, dtype=np.int64)
    fill_idx = selected_idx[:num_selected]
    candidate_arr[:num_selected] = batch.trajectories[fill_idx].astype(np.float32, copy=False)
    candidate_mask[:num_selected] = scored.selected_mask[:num_selected].astype(bool, copy=False)
    score_arr[:num_selected] = scored.scores[fill_idx].astype(np.float32, copy=False)
    component_arr[:num_selected] = scored.components[fill_idx].astype(np.float32, copy=False)
    selected_idx_arr[:num_selected] = fill_idx

    candidates = torch.from_numpy(candidate_arr)
    candidates_mask = torch.from_numpy(candidate_mask)
    score_targets = torch.from_numpy(score_arr)
    component_targets = torch.from_numpy(component_arr)

    left_available = any(side == "left" for side in batch.side)
    right_available = any(side == "right" for side in batch.side)
    selected_left = any(batch.side[int(idx)] == "left" for idx in fill_idx)
    selected_right = any(batch.side[int(idx)] == "right" for idx in fill_idx)

    payload = {
        "trajectory_candidates": candidates,
        "trajectory_candidates_mask": candidates_mask,
        "pdm_score_targets": score_targets,
        "gt_pdm_score": torch.tensor(float(scored.gt_score), dtype=torch.float32),
        "pdm_score_components_candidates": component_targets,
        "gt_pdm_components": torch.from_numpy(scored.gt_components.astype(np.float32, copy=False)),
        "b2d_candidate_raw_count": torch.tensor(int(batch.trajectories.shape[0]), dtype=torch.int32),
        "b2d_candidate_selected_indices": torch.from_numpy(selected_idx_arr),
        "b2d_adjacent_lane_occupancy_allowed": torch.tensor(
            bool(adjacent_lane_occupancy_allowed),
            dtype=torch.bool,
        ),
        "b2d_left_adjacent_candidate_available": torch.tensor(bool(left_available), dtype=torch.bool),
        "b2d_right_adjacent_candidate_available": torch.tensor(bool(right_available), dtype=torch.bool),
        "b2d_selected_left_lane_change": torch.tensor(bool(selected_left), dtype=torch.bool),
        "b2d_selected_right_lane_change": torch.tensor(bool(selected_right), dtype=torch.bool),
        "b2d_effective_wrong_direction_check_enabled": torch.tensor(
            not bool(adjacent_lane_occupancy_allowed),
            dtype=torch.bool,
        ),
        "wrong_direction_check_enabled": torch.tensor(bool(wrong_direction_enabled), dtype=torch.bool),
        "b2d_scenario_type": scenario_type,
        "b2d_candidate_skip_reason": "",
        "b2d_candidate_skipped": torch.tensor(False, dtype=torch.bool),
    }
    if bool(getattr(args, "write_lane_direction_bev", False)):
        payload["lane_direction_bev"] = lane_direction_bev
        payload["lane_direction_mask"] = lane_direction_mask.bool()
    return payload


def _build_empty_candidate_payload(
    target: Dict[str, object],
    info: Dict[str, object],
    args: argparse.Namespace,
    reason: str,
) -> Optional[Dict[str, object]]:
    trajectory = target.get("trajectory")
    if not torch.is_tensor(trajectory) or trajectory.ndim < 2:
        return None
    topk = max(int(args.topk), 0)
    if topk <= 0:
        return None
    num_steps = int(trajectory.shape[-2])
    candidates = torch.zeros((topk, num_steps, 3), dtype=torch.float32)
    candidates_mask = torch.zeros((topk,), dtype=torch.bool)
    score_targets = torch.zeros((topk,), dtype=torch.float32)
    component_targets = torch.zeros((topk, 6), dtype=torch.float32)
    selected_idx_arr = torch.full((topk,), -1, dtype=torch.int64)
    wrong_direction_enabled, scenario_type = infer_wrong_direction_enabled(info, args)
    return {
        "trajectory_candidates": candidates,
        "trajectory_candidates_mask": candidates_mask,
        "pdm_score_targets": score_targets,
        "gt_pdm_score": torch.tensor(0.0, dtype=torch.float32),
        "pdm_score_components_candidates": component_targets,
        "gt_pdm_components": torch.zeros((6,), dtype=torch.float32),
        "b2d_candidate_raw_count": torch.tensor(0, dtype=torch.int32),
        "b2d_candidate_selected_indices": selected_idx_arr,
        "b2d_adjacent_lane_occupancy_allowed": torch.tensor(False, dtype=torch.bool),
        "b2d_left_adjacent_candidate_available": torch.tensor(False, dtype=torch.bool),
        "b2d_right_adjacent_candidate_available": torch.tensor(False, dtype=torch.bool),
        "b2d_selected_left_lane_change": torch.tensor(False, dtype=torch.bool),
        "b2d_selected_right_lane_change": torch.tensor(False, dtype=torch.bool),
        "b2d_effective_wrong_direction_check_enabled": torch.tensor(
            bool(wrong_direction_enabled),
            dtype=torch.bool,
        ),
        "wrong_direction_check_enabled": torch.tensor(bool(wrong_direction_enabled), dtype=torch.bool),
        "b2d_scenario_type": scenario_type,
        "b2d_candidate_skip_reason": reason,
        "b2d_candidate_skipped": torch.tensor(True, dtype=torch.bool),
    }


def _build_target_field_payload(
    target: Dict[str, object],
    infos: Sequence[Dict[str, object]],
    idx: int,
    info: Dict[str, object],
    map_infos: Optional[Dict[str, object]],
    split: str,
    future_agent_boxes: torch.Tensor,
    future_agent_boxes_mask: torch.Tensor,
    args: argparse.Namespace,
) -> Optional[Dict[str, object]]:
    if map_infos is None:
        return None

    trajectory = target.get("trajectory")
    if not torch.is_tensor(trajectory):
        trajectory = build_future_trajectory(
            infos,
            idx=idx,
            sample_interval=args.sample_interval,
            future_frames=args.future_frames,
            max_translation_m=args.max_trajectory_translation,
            max_step_m=args.max_trajectory_step,
        )
        if trajectory is None:
            return None

    static_bev = build_static_map_bev(info, map_infos)
    bev_semantic_map = overlay_agents_on_bev(static_bev, source_info=info, current_info=info)
    feasible_area_mask, feasible_lane_mask = compute_feasible_masks(bev_semantic_map)
    agent_states, agent_labels = build_agent_targets(info, max_agents=args.max_agents)
    wrong_direction_enabled, scenario_type = infer_wrong_direction_enabled(info, args)
    token = safe_token(split, idx, info)

    payload: Dict[str, object] = {
        "trajectory": trajectory,
        "agent_states": agent_states,
        "agent_labels": agent_labels,
        "future_agent_boxes": future_agent_boxes,
        "future_agent_boxes_mask": future_agent_boxes_mask,
        "bev_semantic_map": torch.from_numpy(bev_semantic_map.astype(np.float32)),
        "token": token,
        "feasible_area_mask": feasible_area_mask.bool(),
        "feasible_lane_mask": feasible_lane_mask.bool(),
        "velocity_bev": build_velocity_bev(info),
        "wrong_direction_check_enabled": torch.tensor(wrong_direction_enabled, dtype=torch.bool),
        "b2d_scenario_type": scenario_type,
        "b2d_index": idx,
        "b2d_split": split,
        "folder": str(info.get("folder", "")),
        "frame_idx": int(info.get("frame_idx", idx)),
        "town_name": str(info.get("town_name", "")),
        "lidar2world": torch.from_numpy(diffusiondrive_lidar2world(info)),
        "b2d_lidar2world": torch.from_numpy(
            np.linalg.inv(np.asarray(info["sensors"]["LIDAR_TOP"]["world2lidar"], dtype=np.float32)).astype(np.float32)
        ),
    }
    if bool(getattr(args, "patch_future_bev", True)):
        future_idx = idx + int(args.future_frames) * int(args.sample_interval)
        future_info = infos[future_idx] if future_idx < len(infos) and is_same_log(info, infos[future_idx]) else info
        future_bev_semantic_map = overlay_agents_on_bev(static_bev, source_info=future_info, current_info=info)
        payload["future_bev_semantic_map"] = torch.from_numpy(future_bev_semantic_map.astype(np.float32))
    if bool(getattr(args, "write_lane_direction_bev", False)):
        lane_direction_bev, lane_direction_mask = build_lane_direction_bev(
            info,
            map_infos,
            radius_m=args.lane_direction_radius_m,
            search_radius=args.candidate_route_search_radius,
        )
        payload["lane_direction_bev"] = lane_direction_bev
        payload["lane_direction_mask"] = lane_direction_mask.bool()
    return payload


def _build_candidates(
    target: Dict[str, object],
    info: Dict[str, object],
    map_infos: Optional[Dict[str, object]],
    future_agent_boxes: torch.Tensor,
    future_agent_boxes_mask: torch.Tensor,
    args: argparse.Namespace,
) -> Optional[Dict[str, object]]:
    if args.candidate_mode == "visualize":
        seed_offset = 0 if str(getattr(args, "_split", "train")) == "train" else 1000003
        rng = random.Random(int(args.seed) + seed_offset + int(info.get("frame_idx", 0)))
        return _build_visualize_candidates(
            target=target,
            info=info,
            map_infos=map_infos,
            future_agent_boxes=future_agent_boxes,
            future_agent_boxes_mask=future_agent_boxes_mask,
            args=args,
            rng=rng,
        )
    if bool(getattr(args, "skip_candidates_near_traffic_light", False)):
        if map_infos is None:
            return None
        if is_near_traffic_light_trigger(
            info,
            map_infos,
            distance_m=float(getattr(args, "traffic_light_skip_distance_m", 12.0)),
            include_stop_sign=bool(getattr(args, "traffic_light_skip_include_stop_sign", False)),
        ):
            return _build_empty_candidate_payload(
                target=target,
                info=info,
                args=args,
                reason="near_traffic_light",
            )

    max_candidates = _infer_candidate_count(target, int(args.num_candidates))
    if max_candidates <= 0:
        return None

    trajectory = target.get("trajectory")
    if not torch.is_tensor(trajectory):
        return None
    feasible_area = target.get("feasible_area_mask")
    if not torch.is_tensor(feasible_area):
        return None

    if args.candidate_mode == "route_lattice":
        if map_infos is None:
            return None
        candidates, candidates_mask = build_route_lattice_candidates(
            trajectory=trajectory,
            max_candidates=max_candidates,
            lateral_offset_m=args.candidate_lateral_offset,
            dt=args.trajectory_dt,
            info=info,
            map_infos=map_infos,
            feasible_area=feasible_area,
            future_agent_boxes=future_agent_boxes,
            future_agent_boxes_mask=future_agent_boxes_mask,
            args=args,
        )
        return {
            "trajectory_candidates": candidates,
            "trajectory_candidates_mask": candidates_mask,
        }

    candidates, candidates_mask = build_simple_candidates(
        trajectory=trajectory,
        max_candidates=max_candidates,
        lateral_offset_m=args.candidate_lateral_offset,
        dt=args.trajectory_dt,
    )
    return {
        "trajectory_candidates": candidates,
        "trajectory_candidates_mask": candidates_mask,
    }


def _candidate_targets_equal(
    target: Dict[str, object],
    candidate_payload: Dict[str, object],
) -> bool:
    for key, value in candidate_payload.items():
        if torch.is_tensor(value) and not _tensor_equal(target.get(key), value):
            return False
        if not torch.is_tensor(value) and target.get(key) != value:
            return False
    return True


def _target_payload_force_keys(args: argparse.Namespace) -> set:
    force_keys = set()
    if bool(getattr(args, "patch_bev", False)):
        force_keys.update(
            {
                "bev_semantic_map",
                "feasible_area_mask",
                "feasible_lane_mask",
            }
        )
        if bool(getattr(args, "patch_future_bev", True)):
            force_keys.add("future_bev_semantic_map")
    if bool(getattr(args, "overwrite_target_fields", False)):
        force_keys.update(
            {
                "trajectory",
                "agent_states",
                "agent_labels",
                "velocity_bev",
                "wrong_direction_check_enabled",
                "b2d_scenario_type",
                "b2d_index",
                "b2d_split",
                "folder",
                "frame_idx",
                "town_name",
                "lidar2world",
                "b2d_lidar2world",
                "lane_direction_bev",
                "lane_direction_mask",
            }
        )
    return force_keys


def _payload_value_equal(old_value: object, new_value: object) -> bool:
    if torch.is_tensor(new_value):
        return _tensor_equal(old_value, new_value)
    return old_value == new_value


def _target_payload_changed(
    target: Dict[str, object],
    payload: Optional[Dict[str, object]],
    args: argparse.Namespace,
) -> bool:
    if not payload:
        return False
    force_keys = _target_payload_force_keys(args)
    for key, value in payload.items():
        should_check = key in force_keys or (bool(args.patch_target_fields) and key not in target)
        if not should_check:
            continue
        if not _payload_value_equal(target.get(key), value):
            return True
    return False


def _apply_target_payload(
    target: Dict[str, object],
    payload: Optional[Dict[str, object]],
    args: argparse.Namespace,
) -> None:
    if not payload:
        return
    force_keys = _target_payload_force_keys(args)
    for key, value in payload.items():
        should_write = key in force_keys or (bool(args.patch_target_fields) and key not in target)
        if should_write:
            target[key] = value


def _box_targets_equal(
    target: Dict[str, object],
    boxes: torch.Tensor,
    mask: torch.Tensor,
) -> bool:
    old_boxes = target.get("future_agent_boxes")
    old_mask = target.get("future_agent_boxes_mask")
    return _tensor_equal(old_boxes, boxes) and _tensor_equal(old_mask, mask)


def _maybe_backup(target_path: Path, args: argparse.Namespace) -> None:
    if not args.backup:
        return
    backup_path = target_path.with_suffix(target_path.suffix + ".bak")
    if not backup_path.exists():
        backup_path.write_bytes(target_path.read_bytes())


def _update_debug_score_placeholders(
    target: Dict[str, object],
    candidates: torch.Tensor,
    candidates_mask: torch.Tensor,
) -> None:
    trajectory = target.get("trajectory")
    if not torch.is_tensor(trajectory):
        return
    target.update(build_debug_pdm_placeholders(trajectory, candidates, candidates_mask))


def _candidate_payload_has_scores(candidate_payload: Dict[str, object]) -> bool:
    return (
        "pdm_score_targets" in candidate_payload
        and "gt_pdm_score" in candidate_payload
        and "pdm_score_components_candidates" in candidate_payload
        and "gt_pdm_components" in candidate_payload
    )


def _write_needed(
    boxes_changed: bool,
    target_fields_changed: bool,
    candidates_changed: bool,
    args: argparse.Namespace,
) -> bool:
    if boxes_changed:
        return True
    if target_fields_changed:
        return True
    if candidates_changed:
        return True
    if args.write_debug_pdm_placeholders and args.patch_candidates:
        return True
    return False


def patch_one_index(idx: int) -> Tuple[str, int, str]:
    infos = _WORKER_INFOS
    map_infos = _WORKER_MAP_INFOS
    args = _WORKER_ARGS
    split = _WORKER_SPLIT
    assert infos is not None and args is not None and split is not None

    info = infos[idx]
    token = safe_token(split, idx, info)
    target_path = Path(args.cache_root) / split / token / "transfuser_target.gz"
    if not target_path.is_file():
        return "missing", idx, token

    target = load_target(target_path)
    future_frames, max_future_agents = _infer_box_shape(
        target,
        default_future_frames=args.future_frames,
        default_max_future_agents=args.max_future_agents,
    )
    new_boxes, new_mask = build_future_agent_boxes(
        infos=infos,
        idx=idx,
        sample_interval=args.sample_interval,
        future_frames=future_frames,
        max_future_agents=max_future_agents,
        range_m=args.future_agent_range,
    )

    boxes_changed = not _box_targets_equal(target, new_boxes, new_mask)

    target_payload = None
    target_fields_changed = False
    if args.patch_target_fields or args.patch_bev:
        target_payload = _build_target_field_payload(
            target=target,
            infos=infos,
            idx=idx,
            info=info,
            map_infos=map_infos,
            split=split,
            future_agent_boxes=new_boxes,
            future_agent_boxes_mask=new_mask,
            args=args,
        )
        if target_payload is None:
            return "target_skipped", idx, token
        target_fields_changed = _target_payload_changed(target, target_payload, args)

    candidates_changed = False
    candidate_payload = None
    if args.patch_candidates:
        args._split = split
        candidate_source = dict(target)
        if target_payload:
            candidate_source.update(target_payload)
        candidate_payload = _build_candidates(
            target=candidate_source,
            info=info,
            map_infos=map_infos,
            future_agent_boxes=new_boxes,
            future_agent_boxes_mask=new_mask,
            args=args,
        )
        if candidate_payload is None:
            candidate_payload = _build_empty_candidate_payload(
                target=candidate_source,
                info=info,
                args=args,
                reason="candidate_build_failed",
            )
        if candidate_payload is None:
            return "candidate_skipped", idx, token
        candidates_changed = not _candidate_targets_equal(target, candidate_payload)

    if not _write_needed(boxes_changed, target_fields_changed, candidates_changed, args):
        return "unchanged", idx, token

    if not args.dry_run:
        _maybe_backup(target_path, args)
        if boxes_changed:
            target["future_agent_boxes"] = new_boxes
            target["future_agent_boxes_mask"] = new_mask
        if target_fields_changed:
            _apply_target_payload(target, target_payload, args)
        if args.patch_candidates and candidate_payload is not None:
            target.update(candidate_payload)
            candidates = candidate_payload.get("trajectory_candidates")
            candidates_mask = candidate_payload.get("trajectory_candidates_mask")
            if (
                args.write_debug_pdm_placeholders
                and torch.is_tensor(candidates)
                and torch.is_tensor(candidates_mask)
                and not _candidate_payload_has_scores(candidate_payload)
            ):
                _update_debug_score_placeholders(target, candidates, candidates_mask)
        dump_target(target_path, target)
    return "updated", idx, token


def init_worker(
    infos: Sequence[Dict[str, object]],
    map_infos: Optional[Dict[str, object]],
    args: argparse.Namespace,
    split: str,
) -> None:
    global _WORKER_INFOS
    global _WORKER_MAP_INFOS
    global _WORKER_ARGS
    global _WORKER_SPLIT
    _WORKER_INFOS = infos
    _WORKER_MAP_INFOS = map_infos
    _WORKER_ARGS = args
    _WORKER_SPLIT = split


def _load_token_filter(args: argparse.Namespace) -> set:
    tokens = set(str(token).strip() for token in (args.tokens or []) if str(token).strip())
    token_file = getattr(args, "token_file", None)
    if token_file:
        path = Path(token_file)
        if not path.is_file():
            raise FileNotFoundError(f"Missing token file: {path}")
        for line in path.read_text().splitlines():
            value = line.strip()
            if not value or value.startswith("#"):
                continue
            value = value.split()[0].strip()
            if value.endswith("transfuser_target.gz") or value.endswith("transfuser_feature.gz"):
                value = Path(value).parent.name
            elif "/" in value or "\\" in value:
                value = Path(value).name
            if value:
                tokens.add(value)
    return tokens


def iter_indices(infos: Sequence[Dict[str, object]], split: str, args: argparse.Namespace):
    selected = set()
    if args.indices:
        for idx in args.indices:
            if 0 <= int(idx) < len(infos):
                selected.add(int(idx))
    token_filter = _load_token_filter(args)
    if token_filter:
        for idx, info in enumerate(infos):
            if safe_token(split, idx, info) in token_filter:
                selected.add(idx)
    if selected:
        for count, idx in enumerate(sorted(selected)):
            if args.limit > 0 and count >= args.limit:
                break
            yield idx
        return

    length = len(infos)
    count = 0
    for idx in range(0, length, max(1, args.stride)):
        if args.limit > 0 and count >= args.limit:
            break
        yield idx
        count += 1


def patch_split(split: str, args: argparse.Namespace) -> None:
    infos_root = Path(args.infos_root)
    info_path = infos_root / f"b2d_infos_{split}.pkl"
    if not info_path.is_file():
        raise FileNotFoundError(f"Missing B2D infos: {info_path}")
    infos = load_pickle(info_path)
    map_infos = None
    need_map_infos = (
        bool(args.patch_target_fields)
        or bool(args.patch_bev)
        or bool(args.write_lane_direction_bev)
        or (args.patch_candidates and bool(args.skip_candidates_near_traffic_light))
        or (args.patch_candidates and args.candidate_mode in ("route_lattice", "visualize"))
    )
    if need_map_infos:
        map_path = infos_root / "b2d_map_infos.pkl"
        if not map_path.is_file():
            raise FileNotFoundError(f"Missing B2D map infos: {map_path}")
        map_infos = load_pickle(map_path)
    indices = list(iter_indices(infos, split, args))

    counts = {"updated": 0, "unchanged": 0, "missing": 0, "candidate_skipped": 0, "target_skipped": 0}
    if args.workers > 1:
        with mp.Pool(
            processes=args.workers,
            initializer=init_worker,
            initargs=(infos, map_infos, args, split),
        ) as pool:
            iterator = pool.imap_unordered(patch_one_index, indices)
            for status, _, _ in tqdm(iterator, total=len(indices), desc=f"patch {split}"):
                counts[status] = counts.get(status, 0) + 1
    else:
        init_worker(infos, map_infos, args, split)
        for idx in tqdm(indices, desc=f"patch {split}"):
            status, _, _ = patch_one_index(idx)
            counts[status] = counts.get(status, 0) + 1

    print(
        f"{split}: updated={counts.get('updated', 0)} "
        f"unchanged={counts.get('unchanged', 0)} missing={counts.get('missing', 0)} "
        f"target_skipped={counts.get('target_skipped', 0)} "
        f"candidate_skipped={counts.get('candidate_skipped', 0)}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--infos-root", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val"], choices=["train", "val"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample-interval", type=int, default=5)
    parser.add_argument("--future-frames", type=int, default=6)
    parser.add_argument("--trajectory-dt", type=float, default=0.5)
    parser.add_argument("--max-trajectory-translation", type=float, default=120.0)
    parser.add_argument("--max-trajectory-step", type=float, default=60.0)
    parser.add_argument("--max-agents", type=int, default=30)
    parser.add_argument("--max-future-agents", type=int, default=64)
    parser.add_argument("--future-agent-range", type=float, default=48.0)
    parser.add_argument(
        "--patch-target-fields",
        action="store_true",
        help="Fill missing target-side fields from the current full B2D cache schema without rebuilding camera/lidar features.",
    )
    parser.add_argument(
        "--overwrite-target-fields",
        action="store_true",
        help="Overwrite non-BEV target metadata/auxiliary fields instead of only filling missing keys.",
    )
    parser.add_argument(
        "--patch-bev",
        action="store_true",
        help="Force recomputation of bev_semantic_map, future_bev_semantic_map, feasible_area_mask, and feasible_lane_mask.",
    )
    parser.add_argument(
        "--no-patch-future-bev",
        dest="patch_future_bev",
        action="store_false",
        help="When --patch-bev is set, skip recomputing future_bev_semantic_map.",
    )
    parser.set_defaults(patch_future_bev=True)
    parser.add_argument("--skip-candidates-near-traffic-light", action="store_true")
    parser.add_argument("--traffic-light-skip-distance-m", type=float, default=12.0)
    parser.add_argument("--traffic-light-skip-include-stop-sign", action="store_true")
    parser.add_argument("--patch-candidates", action="store_true")
    parser.add_argument("--num-candidates", type=int, default=40)
    parser.add_argument("--candidate-mode", choices=["simple", "route_lattice", "visualize"], default="visualize")
    parser.add_argument("--candidate-include-gt", action="store_true", default=True)
    parser.add_argument("--no-candidate-include-gt", dest="candidate_include_gt", action="store_false")
    parser.add_argument("--candidate-lateral-offset", type=float, default=3.5)
    parser.add_argument("--candidate-route-search-radius", type=float, default=80.0)
    parser.add_argument("--candidate-command-weight", type=float, default=8.0)
    parser.add_argument("--candidate-min-speed", type=float, default=1.0)
    parser.add_argument("--candidate-unique-distance", type=float, default=0.45)
    parser.add_argument("--candidate-filter-feasible", action="store_true", default=True)
    parser.add_argument("--no-candidate-filter-feasible", dest="candidate_filter_feasible", action="store_false")
    parser.add_argument("--candidate-feasible-ratio", type=float, default=0.5)
    parser.add_argument("--candidate-filter-dynamic", action="store_true", default=False)
    parser.add_argument("--candidate-dynamic-margin", type=float, default=1.0)
    parser.add_argument("--num-keep-lane", type=int, default=20)
    parser.add_argument("--num-lane-change", type=int, default=20)
    parser.add_argument("--num-early-lane-change", type=int, default=6)
    parser.add_argument("--topk", type=int, default=5)
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
    )
    parser.add_argument("--progress-score-cap", type=float, default=1.3)
    parser.add_argument("--progress-reference-m", type=float, default=20.0)
    parser.add_argument("--score-cap", type=float, default=0.0)
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
    parser.add_argument("--write-lane-direction-bev", action="store_true")
    parser.add_argument("--b2d-routes-file", type=Path, default=None)
    parser.add_argument(
        "--wrong-direction-disabled-scenarios",
        nargs="*",
        default=None,
    )
    parser.add_argument("--write-debug-pdm-placeholders", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--indices", nargs="*", type=int, default=None)
    parser.add_argument("--tokens", nargs="*", default=None)
    parser.add_argument("--token-file", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--backup", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.wrong_direction_disabled_scenarios is None:
        args.wrong_direction_disabled_scenarios = list(B2D_WRONG_DIRECTION_DISABLED_SCENARIOS)
    args.route_scenario_by_id = load_route_scenarios(args.b2d_routes_file)
    for split in args.splits:
        patch_split(split, args)


if __name__ == "__main__":
    main()
