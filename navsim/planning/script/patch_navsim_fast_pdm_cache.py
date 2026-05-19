"""
Patch existing NAVSIM DiffusionDrive target caches with fast-PDM helper fields.

The script only updates existing ``transfuser_target.gz`` files. It does not
rebuild image/lidar features and does not replace official PDM scoring.

Added fields:

  - future_agent_obb: [A, T, 5] in current ego frame, (x, y, heading, length, width)
  - future_agent_mask: [A, T]
  - future_agent_is_agent: [A], true for VEHICLE/PEDESTRIAN/BICYCLE-like objects
  - future_agent_ignore: [A], true for tracks already collided at t=0
  - future_agent_token_hash: [A], stable debug ids
  - feasible_area_mask / feasible_lane_mask, when missing or --overwrite-feasible
  - progress_centerline: [M, 2], local route-centerline polyline
  - progress_centerline_mask: [M]
  - progress_reference: scalar GT route progress in meters
  - pdm_scene_* exact scene objects, when --patch-exact-pdm-scene is set

The helper fields are intended for a tensorized scorer/attention target. Keep
official online PDM for validation/monitoring if exact leaderboard alignment is
required.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import lzma
import multiprocessing as mp
import pickle
from collections import deque
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

from shapely.geometry import Point

try:
    from nuplan.common.actor_state.tracked_objects_types import AGENT_TYPES
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

    from navsim.evaluate.pdm_score import get_trajectory_as_array
except Exception:  # pragma: no cover - import errors should be surfaced at runtime.
    AGENT_TYPES = tuple()
    TrajectorySampling = None
    get_trajectory_as_array = None


_WORKER_ARGS: Optional[argparse.Namespace] = None
_WORKER_METRIC_PATHS: Optional[Dict[str, str]] = None


def _load_gzip_pickle(path: Path) -> Dict[str, object]:
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def _dump_gzip_pickle(path: Path, data: Dict[str, object]) -> None:
    with gzip.open(path, "wb", compresslevel=1) as f:
        pickle.dump(data, f)


def _load_metric_cache(path: Path) -> object:
    with lzma.open(path, "rb") as f:
        return pickle.load(f)


def _token_from_metric_path(path_str: str) -> str:
    normalized = path_str.strip().replace("\\", "/")
    parts = [part for part in normalized.split("/") if part]
    if len(parts) >= 2:
        return parts[-2]
    return Path(path_str).parent.name


def _load_metric_cache_index(metric_cache_path: Path) -> Dict[str, str]:
    metadata_dir = metric_cache_path / "metadata"
    if not metadata_dir.is_dir():
        raise FileNotFoundError(f"Missing metric cache metadata dir: {metadata_dir}")
    metadata_files = sorted(
        p for p in metadata_dir.iterdir() if p.is_file() and p.suffix == ".csv"
    )
    if not metadata_files:
        raise FileNotFoundError(f"No metric cache metadata csv found in {metadata_dir}")

    metric_paths: Dict[str, str] = {}
    for metadata_file in metadata_files:
        with open(metadata_file, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue
            metric_paths[_token_from_metric_path(line)] = line
    return metric_paths


def _check_metric_cache_coverage(
    target_paths: Sequence[Path],
    metric_paths: Dict[str, str],
    log_errors: int,
) -> Dict[str, int]:
    target_tokens = [path.parent.name for path in target_paths]
    target_token_set = set(target_tokens)
    metric_token_set = set(metric_paths.keys())
    missing_metric = sorted(target_token_set - metric_token_set)
    extra_metric = sorted(metric_token_set - target_token_set)
    missing_metric_files = sorted(
        token for token, path_str in metric_paths.items() if not Path(path_str).is_file()
    )
    duplicate_targets = sorted(
        token for token in target_token_set if target_tokens.count(token) > 1
    )

    print(
        "metric_cache_coverage "
        f"targets={len(target_tokens)} "
        f"unique_targets={len(target_token_set)} "
        f"metric_tokens={len(metric_token_set)} "
        f"matched={len(target_token_set & metric_token_set)} "
        f"missing_metric={len(missing_metric)} "
        f"extra_metric={len(extra_metric)} "
        f"missing_metric_files={len(missing_metric_files)} "
        f"duplicate_targets={len(duplicate_targets)}"
    )
    if missing_metric:
        print("missing_metric_examples: " + ", ".join(missing_metric[: int(log_errors)]))
    if missing_metric_files:
        print("missing_metric_file_examples: " + ", ".join(missing_metric_files[: int(log_errors)]))
    if duplicate_targets:
        print("duplicate_target_examples: " + ", ".join(duplicate_targets[: int(log_errors)]))

    return {
        "targets": len(target_tokens),
        "unique_targets": len(target_token_set),
        "metric_tokens": len(metric_token_set),
        "matched": len(target_token_set & metric_token_set),
        "missing_metric": len(missing_metric),
        "extra_metric": len(extra_metric),
        "missing_metric_files": len(missing_metric_files),
        "duplicate_targets": len(duplicate_targets),
    }


def _validate_metric_cache_load(
    target_paths: Sequence[Path],
    metric_paths: Dict[str, str],
    log_errors: int,
) -> Dict[str, int]:
    target_tokens = sorted({path.parent.name for path in target_paths})
    matched_tokens = [token for token in target_tokens if token in metric_paths]
    errors: List[str] = []
    for token in tqdm(matched_tokens, desc="validate metric cache"):
        try:
            _load_metric_cache(Path(metric_paths[token]))
        except Exception as exc:
            errors.append(f"{token}:{type(exc).__name__}:{exc}")
    print(
        "metric_cache_load_validation "
        f"checked={len(matched_tokens)} "
        f"errors={len(errors)}"
    )
    if errors:
        print("metric_cache_load_error_examples:")
        for error in errors[: int(log_errors)]:
            print(f"  {error}")
    return {"checked": len(matched_tokens), "errors": len(errors)}


def _iter_target_paths(cache_path: Path, tokens: Optional[Sequence[str]]) -> List[Path]:
    token_set = set(tokens) if tokens else None
    paths: List[Path] = []
    for target_path in cache_path.rglob("transfuser_target.gz"):
        token = target_path.parent.name
        if token_set is not None and token not in token_set:
            continue
        paths.append(target_path)
    return sorted(paths)


def _read_token_file(path: Optional[str]) -> Optional[List[str]]:
    if not path:
        return None
    token_file = Path(path)
    tokens = []
    with open(token_file, "r", encoding="utf-8") as f:
        for line in f:
            token = line.strip()
            if token:
                tokens.append(token)
    return tokens


def _stable_hash64(text: str) -> np.int64:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return np.int64(int.from_bytes(digest, byteorder="little", signed=False) & ((1 << 63) - 1))


def _as_numpy(value: object) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _to_class_map(bev_semantic_map: object, num_bev_classes: int) -> np.ndarray:
    bev = _as_numpy(bev_semantic_map)
    if bev.ndim == 2:
        return bev.astype(np.int64, copy=False)
    if bev.ndim == 3:
        if bev.shape[0] == num_bev_classes or bev.shape[0] <= 32:
            return bev.argmax(axis=0).astype(np.int64, copy=False)
        if bev.shape[-1] == num_bev_classes or bev.shape[-1] <= 32:
            return bev.argmax(axis=-1).astype(np.int64, copy=False)
    if bev.ndim == 4 and bev.shape[0] == 1:
        return _to_class_map(bev[0], num_bev_classes)
    raise ValueError(f"Unsupported bev_semantic_map shape: {bev.shape}")


def _flood_fill(mask: np.ndarray, seed: np.ndarray, max_iters: int) -> np.ndarray:
    height, width = mask.shape
    current = seed.astype(bool, copy=True) & mask
    if not current.any():
        return current
    queue: deque[Tuple[int, int]] = deque(map(tuple, np.argwhere(current)))
    visited = current.copy()
    steps = 0
    while queue:
        row, col = queue.popleft()
        for nr in (row - 1, row, row + 1):
            for nc in (col - 1, col, col + 1):
                if nr < 0 or nr >= height or nc < 0 or nc >= width:
                    continue
                if visited[nr, nc] or not mask[nr, nc]:
                    continue
                visited[nr, nc] = True
                queue.append((nr, nc))
        steps += 1
        if max_iters > 0 and steps > max_iters * max(height, width):
            break
    return visited


def _compute_feasible_masks(
    target: Dict[str, object],
    args: argparse.Namespace,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    bev = target.get("bev_semantic_map")
    if bev is None:
        return None
    class_map = _to_class_map(bev, int(args.num_bev_classes))
    road_label = int(args.road_label)
    centerline_label = int(args.centerline_label)
    drivable_mask = (class_map == road_label) | (class_map == centerline_label)
    centerline_mask = class_map == centerline_label

    height, width = drivable_mask.shape
    ego_row = 0
    ego_col = (width - 1) // 2
    seed_rows = max(1, min(int(args.feasible_seed_rows), height))
    seed_cols = max(0, min(int(args.feasible_seed_cols), width // 2))
    row_start, row_end = 0, min(height, seed_rows)
    col_start = max(0, ego_col - seed_cols)
    col_end = min(width, ego_col + seed_cols + 1)

    seed = np.zeros_like(drivable_mask, dtype=bool)
    region = drivable_mask[row_start:row_end, col_start:col_end]
    if region.any():
        coords = np.argwhere(region)
        ego = np.array([ego_row, ego_col - col_start], dtype=np.int64)
        dist2 = ((coords - ego[None]) ** 2).sum(axis=1)
        best = coords[int(dist2.argmin())]
        seed[row_start + int(best[0]), col_start + int(best[1])] = True
    elif drivable_mask.any():
        coords = np.argwhere(drivable_mask)
        ego = np.array([ego_row, ego_col], dtype=np.int64)
        dist2 = ((coords - ego[None]) ** 2).sum(axis=1)
        best = coords[int(dist2.argmin())]
        seed[int(best[0]), int(best[1])] = True
    else:
        seed[min(ego_row, height - 1), min(ego_col, width - 1)] = True

    feasible_area = _flood_fill(drivable_mask, seed, int(args.feasible_max_iters))
    feasible_lane = feasible_area & centerline_mask
    return torch.from_numpy(feasible_area.astype(bool)), torch.from_numpy(feasible_lane.astype(bool))


def _ego_pose_xyh(metric_cache: object) -> Tuple[float, float, float]:
    ego = metric_cache.ego_state.center
    return float(ego.x), float(ego.y), float(ego.heading)


def _global_to_local_xy(points: np.ndarray, ego_x: float, ego_y: float, ego_h: float) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    dx = pts[..., 0] - ego_x
    dy = pts[..., 1] - ego_y
    cos_h = float(np.cos(ego_h))
    sin_h = float(np.sin(ego_h))
    local_x = cos_h * dx + sin_h * dy
    local_y = -sin_h * dx + cos_h * dy
    return np.stack([local_x, local_y], axis=-1)


def _normalize_angle(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def _polygon_to_local_obb(
    polygon: object,
    ego_x: float,
    ego_y: float,
    ego_h: float,
) -> Optional[np.ndarray]:
    if polygon is None or getattr(polygon, "is_empty", False):
        return None
    try:
        rect = polygon.minimum_rotated_rectangle
        coords = np.asarray(rect.exterior.coords, dtype=np.float64)
    except Exception:
        return None
    if coords.shape[0] < 4:
        return None
    if np.allclose(coords[0], coords[-1]):
        coords = coords[:-1]
    if coords.shape[0] != 4:
        coords = coords[:4]

    local = _global_to_local_xy(coords, ego_x, ego_y, ego_h)
    edges = np.roll(local, -1, axis=0) - local
    edge_lengths = np.linalg.norm(edges, axis=-1)
    if not np.isfinite(edge_lengths).all() or edge_lengths.max() <= 1e-4:
        return None
    length_idx = int(edge_lengths.argmax())
    width_idx = (length_idx + 1) % 4
    length = float(edge_lengths[length_idx])
    width = float(edge_lengths[width_idx])
    heading = _normalize_angle(float(np.arctan2(edges[length_idx, 1], edges[length_idx, 0])))
    center = local.mean(axis=0)
    if length < width:
        length, width = width, length
        heading = _normalize_angle(heading + np.pi / 2.0)
    return np.array([center[0], center[1], heading, length, width], dtype=np.float32)


def _token_is_agent(metric_cache: object, token: str) -> bool:
    obj = getattr(metric_cache.observation, "unique_objects", {}).get(token)
    if obj is None:
        return False
    return bool(getattr(obj, "tracked_object_type", None) in AGENT_TYPES)


def _collect_future_agents(
    metric_cache: object,
    args: argparse.Namespace,
) -> Dict[str, torch.Tensor]:
    ego_x, ego_y, ego_h = _ego_pose_xyh(metric_cache)
    num_poses = int(args.num_poses)
    stride = max(1, int(round(float(args.interval_length) / max(float(args.observation_interval), 1e-6))))
    time_indices = [idx * stride for idx in range(num_poses + 1)]
    red_light_token = getattr(metric_cache.observation, "red_light_token", "red_light")
    collided = set(getattr(metric_cache.observation, "collided_track_ids", []))

    per_token: Dict[str, Dict[str, object]] = {}
    for cache_t, obs_t in enumerate(time_indices):
        try:
            occupancy = metric_cache.observation[obs_t]
        except Exception:
            continue
        for token in getattr(occupancy, "tokens", []):
            if (not bool(args.include_red_lights)) and red_light_token in token:
                continue
            try:
                polygon = occupancy[token]
            except Exception:
                continue
            obb = _polygon_to_local_obb(polygon, ego_x, ego_y, ego_h)
            if obb is None:
                continue
            dist = float(np.hypot(obb[0], obb[1]))
            if dist > float(args.future_range_m):
                continue
            entry = per_token.setdefault(
                token,
                {
                    "obbs": {},
                    "min_dist": dist,
                    "is_agent": _token_is_agent(metric_cache, token),
                    "ignore": token in collided,
                },
            )
            entry["obbs"][cache_t] = obb
            entry["min_dist"] = min(float(entry["min_dist"]), dist)

    selected_tokens = sorted(
        per_token.keys(),
        key=lambda t: (bool(per_token[t]["ignore"]), float(per_token[t]["min_dist"]), t),
    )[: int(args.max_agents)]
    max_agents = int(args.max_agents)
    horizon = num_poses + 1
    obbs = np.zeros((max_agents, horizon, 5), dtype=np.float32)
    masks = np.zeros((max_agents, horizon), dtype=bool)
    is_agent = np.zeros((max_agents,), dtype=bool)
    ignore = np.zeros((max_agents,), dtype=bool)
    token_hash = np.zeros((max_agents,), dtype=np.int64)

    for agent_idx, token in enumerate(selected_tokens):
        entry = per_token[token]
        is_agent[agent_idx] = bool(entry["is_agent"])
        ignore[agent_idx] = bool(entry["ignore"])
        token_hash[agent_idx] = _stable_hash64(token)
        for t_idx, obb in entry["obbs"].items():
            obbs[agent_idx, int(t_idx)] = obb
            masks[agent_idx, int(t_idx)] = True

    return {
        "future_agent_obb": torch.from_numpy(obbs),
        "future_agent_mask": torch.from_numpy(masks),
        "future_agent_is_agent": torch.from_numpy(is_agent),
        "future_agent_ignore": torch.from_numpy(ignore),
        "future_agent_token_hash": torch.from_numpy(token_hash),
    }


def _centerline_local(metric_cache: object, args: argparse.Namespace) -> Dict[str, torch.Tensor]:
    ego_x, ego_y, ego_h = _ego_pose_xyh(metric_cache)
    max_points = int(args.progress_points)
    centerline = getattr(metric_cache, "centerline", None)
    coords = np.zeros((0, 2), dtype=np.float64)
    try:
        length = float(getattr(centerline, "length", 0.0) or 0.0)
        if length > 0.0 and max_points > 1:
            distances = np.linspace(0.0, length, max_points, dtype=np.float64)
            coords = np.asarray(centerline.interpolate(distances, as_array=True), dtype=np.float64)[:, :2]
    except Exception:
        coords = np.zeros((0, 2), dtype=np.float64)
    if coords.size == 0:
        states = getattr(centerline, "_states_se2_array", None)
        if states is not None:
            coords = np.asarray(states, dtype=np.float64)[:, :2]
        else:
            linestring = getattr(centerline, "linestring", None)
            if linestring is not None:
                coords = np.asarray(linestring.coords, dtype=np.float64)

        if coords.shape[0] > max_points and max_points > 1:
            sample_idx = np.linspace(0, coords.shape[0] - 1, max_points).round().astype(np.int64)
            coords = coords[sample_idx]
    coords_local = _global_to_local_xy(coords, ego_x, ego_y, ego_h) if coords.size else coords
    out = np.zeros((max_points, 2), dtype=np.float32)
    mask = np.zeros((max_points,), dtype=bool)
    n = min(max_points, coords_local.shape[0])
    if n > 0:
        out[:n] = coords_local[:n].astype(np.float32)
        mask[:n] = True

    return {
        "progress_centerline": torch.from_numpy(out),
        "progress_centerline_mask": torch.from_numpy(mask),
        "progress_reference": torch.tensor(_gt_progress_reference(metric_cache, args), dtype=torch.float32),
        "progress_centerline_length": torch.tensor(
            float(getattr(centerline, "length", 0.0) or 0.0),
            dtype=torch.float32,
        ),
    }


def _gt_progress_reference(metric_cache: object, args: argparse.Namespace) -> float:
    centerline = getattr(metric_cache, "centerline", None)
    if centerline is None:
        return 0.0
    try:
        if TrajectorySampling is not None and get_trajectory_as_array is not None:
            sampling = TrajectorySampling(
                num_poses=int(args.num_poses),
                interval_length=float(args.interval_length),
            )
            states = get_trajectory_as_array(
                metric_cache.trajectory,
                sampling,
                metric_cache.ego_state.time_point,
            )
            points = states[:, :2]
            start_proj, end_proj = centerline.project([Point(*points[0]), Point(*points[-1])])
            return float(max(float(end_proj) - float(start_proj), 0.0))
    except Exception:
        pass
    try:
        sampled_states = metric_cache.trajectory.get_sampled_trajectory()
        if len(sampled_states) < 2:
            return 0.0
        start = sampled_states[0].center.point
        end = sampled_states[-1].center.point
        start_proj, end_proj = centerline.project([start, end])
        return float(max(float(end_proj) - float(start_proj), 0.0))
    except Exception:
        return 0.0


def _exact_pdm_scene(metric_cache: object) -> Dict[str, object]:
    return {
        "pdm_scene_ego_state": metric_cache.ego_state,
        "pdm_scene_observation": metric_cache.observation,
        "pdm_scene_drivable_area_map": metric_cache.drivable_area_map,
        "pdm_scene_centerline": metric_cache.centerline,
        "pdm_scene_route_lane_ids": list(metric_cache.route_lane_ids),
    }


def _exact_dac_scene(metric_cache: object) -> Dict[str, object]:
    return {
        "pdm_scene_ego_state": metric_cache.ego_state,
        "pdm_scene_drivable_area_map": metric_cache.drivable_area_map,
    }


def _needs_patch(target: Dict[str, object], args: argparse.Namespace) -> bool:
    if bool(args.overwrite):
        return True
    required = []
    if bool(args.patch_fast_fields):
        required.extend(
            [
                "future_agent_obb",
                "future_agent_mask",
                "future_agent_is_agent",
                "future_agent_ignore",
            ]
        )
        if bool(args.patch_progress):
            required.extend(
                [
                    "progress_centerline",
                    "progress_centerline_mask",
                    "progress_reference",
                ]
            )
    if bool(args.patch_feasible) and ("feasible_area_mask" not in target or "feasible_lane_mask" not in target):
        return True
    if bool(args.patch_exact_dac_scene):
        required_dac_scene = [
            "pdm_scene_ego_state",
            "pdm_scene_drivable_area_map",
        ]
        if bool(args.overwrite_exact_dac_scene) or any(key not in target for key in required_dac_scene):
            return True
    if bool(args.patch_exact_pdm_scene):
        required_scene = [
            "pdm_scene_ego_state",
            "pdm_scene_observation",
            "pdm_scene_drivable_area_map",
            "pdm_scene_centerline",
            "pdm_scene_route_lane_ids",
        ]
        if bool(args.overwrite_exact_pdm_scene) or any(key not in target for key in required_scene):
            return True
    return any(key not in target for key in required)


def _patch_one(target_path_str: str) -> Tuple[str, str, str]:
    assert _WORKER_ARGS is not None
    assert _WORKER_METRIC_PATHS is not None
    args = _WORKER_ARGS
    target_path = Path(target_path_str)
    token = target_path.parent.name
    metric_path_str = _WORKER_METRIC_PATHS.get(token)
    if metric_path_str is None:
        return token, "missing_metric", str(target_path)

    try:
        target = _load_gzip_pickle(target_path)
    except Exception as exc:
        return token, f"load_target_error:{type(exc).__name__}", str(target_path)

    if not _needs_patch(target, args):
        return token, "skipped_existing", str(target_path)

    try:
        metric_cache = _load_metric_cache(Path(metric_path_str))
        updates: Dict[str, object] = {}

        if bool(args.patch_fast_fields):
            if bool(args.overwrite) or "future_agent_obb" not in target or "future_agent_mask" not in target:
                updates.update(_collect_future_agents(metric_cache, args))
            else:
                for key in [
                    "future_agent_is_agent",
                    "future_agent_ignore",
                    "future_agent_token_hash",
                ]:
                    if bool(args.overwrite) or key not in target:
                        updates.update(_collect_future_agents(metric_cache, args))
                        break

            if bool(args.patch_progress) and (
                bool(args.overwrite)
                or "progress_centerline" not in target
                or "progress_reference" not in target
            ):
                updates.update(_centerline_local(metric_cache, args))

        if bool(args.patch_exact_dac_scene) and (
            bool(args.overwrite)
            or bool(args.overwrite_exact_dac_scene)
            or "pdm_scene_ego_state" not in target
            or "pdm_scene_drivable_area_map" not in target
        ):
            updates.update(_exact_dac_scene(metric_cache))

        if bool(args.patch_exact_pdm_scene) and (
            bool(args.overwrite)
            or bool(args.overwrite_exact_pdm_scene)
            or "pdm_scene_observation" not in target
            or "pdm_scene_drivable_area_map" not in target
        ):
            updates.update(_exact_pdm_scene(metric_cache))

        if bool(args.patch_feasible) and (
            bool(args.overwrite_feasible)
            or "feasible_area_mask" not in target
            or "feasible_lane_mask" not in target
        ):
            feasible = _compute_feasible_masks(target, args)
            if feasible is not None:
                updates["feasible_area_mask"], updates["feasible_lane_mask"] = feasible

        if not updates:
            return token, "skipped_existing", str(target_path)

        if bool(args.dry_run):
            return token, "would_patch", str(target_path)

        target.update(updates)
        _dump_gzip_pickle(target_path, target)
        return token, "patched", str(target_path)
    except Exception as exc:
        return token, f"error:{type(exc).__name__}:{exc}", str(target_path)


def _init_worker(args: argparse.Namespace, metric_paths: Dict[str, str]) -> None:
    global _WORKER_ARGS, _WORKER_METRIC_PATHS
    _WORKER_ARGS = args
    _WORKER_METRIC_PATHS = metric_paths


def _summarize(results: Iterable[Tuple[str, str, str]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for _, status, _ in results:
        key = status.split(":", 1)[0]
        counts[key] = counts.get(key, 0) + 1
    return counts


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-path", required=True, help="NAVSIM training_cache root.")
    parser.add_argument("--metric-cache-path", required=True, help="NAVSIM metric_cache root.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--token-file", default=None, help="Optional newline-separated token list.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--check-only", action="store_true", help="Only check training/metric cache token coverage.")
    parser.add_argument(
        "--validate-metric-load",
        action="store_true",
        help="With --check-only, load matched metric_cache.pkl files to catch corrupt caches.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite all fast-PDM helper fields.")
    parser.add_argument("--patch-fast-fields", action="store_true", default=True)
    parser.add_argument("--no-patch-fast-fields", dest="patch_fast_fields", action="store_false")
    parser.add_argument("--patch-progress", action="store_true", default=True)
    parser.add_argument("--no-patch-progress", dest="patch_progress", action="store_false")

    parser.add_argument("--num-poses", type=int, default=40, help="PDM proposal horizon steps. Default matches 40x0.1s.")
    parser.add_argument("--interval-length", type=float, default=0.1, help="PDM proposal interval in seconds.")
    parser.add_argument("--observation-interval", type=float, default=0.1, help="Metric-cache observation interval in seconds.")
    parser.add_argument("--max-agents", type=int, default=64)
    parser.add_argument("--future-range-m", type=float, default=90.0)
    parser.add_argument("--include-red-lights", action="store_true", help="Keep red-light pseudo-polygons in future_agent_obb.")

    parser.add_argument("--patch-feasible", action="store_true", default=True)
    parser.add_argument("--no-patch-feasible", dest="patch_feasible", action="store_false")
    parser.add_argument("--overwrite-feasible", action="store_true")
    parser.add_argument("--num-bev-classes", type=int, default=7)
    parser.add_argument("--road-label", type=int, default=1)
    parser.add_argument("--centerline-label", type=int, default=3)
    parser.add_argument("--feasible-seed-rows", type=int, default=12)
    parser.add_argument("--feasible-seed-cols", type=int, default=16)
    parser.add_argument("--feasible-max-iters", type=int, default=0)

    parser.add_argument("--progress-points", type=int, default=128)
    parser.add_argument(
        "--patch-exact-dac-scene",
        action="store_true",
        help="Cache only exact DAC scene objects: pdm_scene_ego_state and pdm_scene_drivable_area_map.",
    )
    parser.add_argument("--overwrite-exact-dac-scene", action="store_true")
    parser.add_argument(
        "--patch-exact-pdm-scene",
        action="store_true",
        help=(
            "Cache exact scene objects needed for official NC/DAC scoring: ego_state, "
            "PDMObservation, PDMDrivableMap, centerline, and route_lane_ids."
        ),
    )
    parser.add_argument("--overwrite-exact-pdm-scene", action="store_true")
    parser.add_argument("--log-errors", type=int, default=20)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    cache_path = Path(args.cache_path)
    metric_cache_path = Path(args.metric_cache_path)
    if not cache_path.is_dir():
        raise FileNotFoundError(f"Missing cache path: {cache_path}")

    metric_paths = _load_metric_cache_index(metric_cache_path)
    tokens = _read_token_file(args.token_file)
    target_paths = _iter_target_paths(cache_path, tokens)
    if not target_paths:
        raise FileNotFoundError(f"No transfuser_target.gz files found under {cache_path}")

    print(f"targets={len(target_paths)} metric_cache={len(metric_paths)} dry_run={bool(args.dry_run)}")
    coverage = _check_metric_cache_coverage(target_paths, metric_paths, int(args.log_errors))
    if bool(args.check_only):
        load_validation = {"errors": 0}
        if bool(args.validate_metric_load):
            load_validation = _validate_metric_cache_load(
                target_paths, metric_paths, int(args.log_errors)
            )
        if (
            coverage["missing_metric"]
            or coverage["missing_metric_files"]
            or coverage["duplicate_targets"]
            or load_validation["errors"]
        ):
            raise SystemExit(2)
        return
    target_path_strs = [str(path) for path in target_paths]

    if int(args.workers) <= 1:
        _init_worker(args, metric_paths)
        results = [_patch_one(path) for path in tqdm(target_path_strs, desc="patch fast pdm")]
    else:
        try:
            ctx = mp.get_context("fork")
        except ValueError:  # Windows fallback.
            ctx = mp.get_context("spawn")
        with ctx.Pool(
            processes=int(args.workers),
            initializer=_init_worker,
            initargs=(args, metric_paths),
        ) as pool:
            results = list(
                tqdm(
                    pool.imap_unordered(_patch_one, target_path_strs, chunksize=16),
                    total=len(target_path_strs),
                    desc="patch fast pdm",
                )
            )

    counts = _summarize(results)
    print("summary:", " ".join(f"{key}={value}" for key, value in sorted(counts.items())))
    error_results = [item for item in results if item[1].startswith("error") or item[1].startswith("load")]
    for token, status, path in error_results[: max(0, int(args.log_errors))]:
        print(f"{status} token={token} path={path}")


if __name__ == "__main__":
    main()
