"""Add near/far B2D navigation points to existing DiffusionDrive caches.

The patch writes a small ``b2d_navigation_feature.gz`` sidecar per token. Large
camera/lidar feature files and all target fields remain untouched.
"""

from __future__ import annotations

import argparse
import gzip
import json
import multiprocessing as mp
import os
import pickle
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

from navsim.planning.script.run_b2d_diffusiondrive_caching import (
    build_navigation_features,
    b2d_xy_to_diffusiondrive,
    load_pickle,
    safe_token,
)


_WORKER_INFOS = None
_WORKER_ARGS = None
_WORKER_SPLIT = None


def _dump_gzip_pickle_atomic(path: Path, payload: Dict[str, object]) -> None:
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with gzip.open(temp_path, "wb", compresslevel=1) as stream:
            pickle.dump(payload, stream)
        temp_path.replace(path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _value_equal(old_value: object, new_value: torch.Tensor) -> bool:
    if not torch.is_tensor(old_value):
        return False
    if tuple(old_value.shape) != tuple(new_value.shape):
        return False
    old_value = old_value.to(dtype=new_value.dtype)
    if new_value.dtype == torch.bool:
        return bool(torch.equal(old_value.bool(), new_value.bool()))
    return bool(torch.allclose(old_value, new_value, rtol=0.0, atol=1e-6))


def _write_navigation_sidecar(
    token_path: Path,
    navigation: Dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> Tuple[str, str]:
    navigation_path = token_path / "b2d_navigation_feature.gz"
    if navigation_path.is_file():
        try:
            with gzip.open(navigation_path, "rb") as stream:
                existing = pickle.load(stream)
        except Exception as exc:
            return "invalid", f"{token_path.name}: {type(exc).__name__}: {exc}"
        changed = any(
            not _value_equal(existing.get(key), value)
            for key, value in navigation.items()
        )
        if not changed:
            return "unchanged", token_path.name
    if args.dry_run:
        return "would_update", token_path.name

    if args.backup and navigation_path.is_file():
        backup_path = navigation_path.with_suffix(navigation_path.suffix + ".bak")
        if not backup_path.exists():
            backup_path.write_bytes(navigation_path.read_bytes())
    _dump_gzip_pickle_atomic(navigation_path, navigation)
    return "updated", token_path.name


def _tensor_numpy(value: object) -> Optional[np.ndarray]:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    if value is None:
        return None
    return np.asarray(value)


def _find_annotation_path(b2d_root: Path, folder: str, frame_idx: int) -> Optional[Path]:
    anno_dir = b2d_root / folder / "anno"
    for name in (
        f"{frame_idx:05d}.json.gz",
        f"{frame_idx:05d}.gz",
        f"{frame_idx}.json.gz",
        f"{frame_idx}.gz",
    ):
        path = anno_dir / name
        if path.is_file():
            return path
    matches = sorted(anno_dir.glob(f"{frame_idx:05d}.*")) if anno_dir.is_dir() else []
    return matches[0] if matches else None


def _raw_command_point(anno: Dict[str, object], suffix: str) -> Optional[np.ndarray]:
    x = anno.get(f"x_command_{suffix}")
    y = anno.get(f"y_command_{suffix}")
    if x is None or y is None:
        return None
    point = np.asarray([x, -float(y)], dtype=np.float32)
    return point if np.isfinite(point).all() else None


def _localize_raw_command(
    point: Optional[np.ndarray],
    target: Dict[str, object],
) -> Optional[np.ndarray]:
    if point is None:
        return None
    homogeneous = np.asarray([point[0], point[1], 0.0, 1.0], dtype=np.float64)

    dd_lidar2world = _tensor_numpy(target.get("lidar2world"))
    if dd_lidar2world is not None and dd_lidar2world.shape == (4, 4):
        local = np.linalg.inv(dd_lidar2world.astype(np.float64)) @ homogeneous
        result = local[:2]
    else:
        b2d_lidar2world = _tensor_numpy(target.get("b2d_lidar2world"))
        if b2d_lidar2world is None or b2d_lidar2world.shape != (4, 4):
            return None
        local = np.linalg.inv(b2d_lidar2world.astype(np.float64)) @ homogeneous
        result = b2d_xy_to_diffusiondrive(local[:2])
    if not np.isfinite(result).all():
        return None
    return np.asarray(result, dtype=np.float32)


def _navigation_from_raw_cache(
    token_path: Path,
    b2d_root: Path,
) -> Tuple[Optional[Dict[str, torch.Tensor]], str]:
    target_path = token_path / "transfuser_target.gz"
    if not target_path.is_file():
        return None, "missing_target"
    try:
        with gzip.open(target_path, "rb") as stream:
            target = pickle.load(stream)
    except Exception as exc:
        return None, f"invalid_target:{type(exc).__name__}:{exc}"

    folder = str(target.get("folder", "")).strip("/\\")
    frame_idx = int(target.get("frame_idx", -1))
    if not folder or frame_idx < 0:
        return None, "missing_target_metadata"
    annotation_path = _find_annotation_path(b2d_root, folder, frame_idx)
    if annotation_path is None:
        return None, f"missing_annotation:{folder}:{frame_idx}"
    try:
        with gzip.open(annotation_path, "rt", encoding="utf-8") as stream:
            anno = json.load(stream)
    except Exception as exc:
        return None, f"invalid_annotation:{type(exc).__name__}:{exc}"

    payload: Dict[str, torch.Tensor] = {}
    for key, suffix in (("tp_near", "near"), ("tp_far", "far")):
        local = _localize_raw_command(_raw_command_point(anno, suffix), target)
        valid = local is not None
        if local is None:
            local = np.zeros((2,), dtype=np.float32)
        payload[key] = torch.from_numpy(local.astype(np.float32, copy=False))
        payload[f"{key}_valid"] = torch.tensor(valid, dtype=torch.bool)
    return payload, ""


def _init_worker(infos: Sequence[Dict[str, object]], args: argparse.Namespace, split: str) -> None:
    global _WORKER_INFOS
    global _WORKER_ARGS
    global _WORKER_SPLIT
    _WORKER_INFOS = infos
    _WORKER_ARGS = args
    _WORKER_SPLIT = split


def _patch_one(idx: int) -> Tuple[str, int, str]:
    infos = _WORKER_INFOS
    args = _WORKER_ARGS
    split = _WORKER_SPLIT
    assert infos is not None and args is not None and split is not None

    info = infos[idx]
    token = safe_token(split, idx, info)
    token_path = Path(args.cache_root) / split / token
    feature_path = token_path / "transfuser_feature.gz"
    if not feature_path.is_file():
        return "missing", idx, token

    status, detail = _write_navigation_sidecar(
        token_path, build_navigation_features(info), args
    )
    return status, idx, detail


def _patch_one_raw(token_path_value: str) -> Tuple[str, int, str]:
    args = _WORKER_ARGS
    assert args is not None
    token_path = Path(token_path_value)
    navigation, reason = _navigation_from_raw_cache(token_path, Path(args.b2d_root))
    if navigation is None:
        return "invalid", -1, f"{token_path.name}: {reason}"
    status, detail = _write_navigation_sidecar(token_path, navigation, args)
    return status, -1, detail


def _iter_indices(length: int, args: argparse.Namespace) -> Iterable[int]:
    count = 0
    for idx in range(0, length, max(int(args.stride), 1)):
        if args.limit > 0 and count >= args.limit:
            break
        yield idx
        count += 1


def _patch_split(split: str, args: argparse.Namespace) -> None:
    info_path = Path(args.infos_root) / f"b2d_infos_{split}.pkl" if args.infos_root else None
    use_infos = info_path is not None and info_path.is_file()
    infos = load_pickle(info_path) if use_infos else None
    if use_infos:
        work_items = list(_iter_indices(len(infos), args))
        worker = _patch_one
        init_args = (infos, args, split)
    else:
        if not args.b2d_root:
            missing = str(info_path) if info_path is not None else "not provided"
            raise FileNotFoundError(
                f"Missing B2D infos ({missing}) and --b2d-root was not provided"
            )
        split_dir = Path(args.cache_root) / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Missing cache split: {split_dir}")
        token_paths = sorted(
            path for path in split_dir.iterdir()
            if path.is_dir() and (path / "transfuser_feature.gz").is_file()
        )
        work_items = [str(token_paths[idx]) for idx in _iter_indices(len(token_paths), args)]
        worker = _patch_one_raw
        init_args = ([], args, split)
    counts: Dict[str, int] = {}

    if args.workers > 1:
        with mp.Pool(
            processes=args.workers,
            initializer=_init_worker,
            initargs=init_args,
        ) as pool:
            iterator = pool.imap_unordered(worker, work_items)
            for status, _, detail in tqdm(iterator, total=len(work_items), desc=f"navigation {split}"):
                counts[status] = counts.get(status, 0) + 1
                if status == "invalid" and counts[status] <= args.max_error_print:
                    print(detail)
    else:
        _init_worker(*init_args)
        for item in tqdm(work_items, desc=f"navigation {split}"):
            status, _, detail = worker(item)
            counts[status] = counts.get(status, 0) + 1
            if status == "invalid" and counts[status] <= args.max_error_print:
                print(detail)

    summary = " ".join(f"{key}={value}" for key, value in sorted(counts.items()))
    print(f"{split}: {summary}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--infos-root",
        default=None,
        help="Optional prepared infos directory. If absent/missing, read raw annotations via --b2d-root.",
    )
    parser.add_argument(
        "--b2d-root",
        default=None,
        help="Raw Bench2Drive dataset root containing v1/*/anno/*.json.gz.",
    )
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val"], choices=["train", "val"])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--backup", action="store_true")
    parser.add_argument("--max-error-print", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for split in args.splits:
        _patch_split(split, args)


if __name__ == "__main__":
    main()
