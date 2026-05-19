"""Check Bench2Drive DiffusionDrive cache for non-finite and outlier targets."""

from __future__ import annotations

import argparse
import gzip
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from tqdm import tqdm


def load_pickle_gz(path: Path) -> Dict[str, object]:
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach") and callable(value.detach):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def iter_token_dirs(cache_root: Path, splits: List[str], limit: int) -> Iterable[Tuple[str, Path]]:
    count = 0
    for split in splits:
        split_dir = cache_root / split
        if not split_dir.is_dir():
            continue
        for token_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            if limit > 0 and count >= limit:
                return
            yield split, token_dir
            count += 1


def check_array(name: str, value, reasons: List[str]) -> None:
    if value is None:
        reasons.append(f"missing:{name}")
        return
    try:
        arr = to_numpy(value)
    except Exception as exc:
        reasons.append(f"load:{name}:{type(exc).__name__}")
        return
    if arr.size == 0:
        return
    if not np.isfinite(arr).all():
        reasons.append(f"nonfinite:{name}")


def check_token(token_dir: Path, args: argparse.Namespace) -> List[str]:
    reasons: List[str] = []
    target_path = token_dir / "transfuser_target.gz"
    feature_path = token_dir / "transfuser_feature.gz"
    if not target_path.is_file():
        return ["missing:transfuser_target.gz"]
    try:
        target = load_pickle_gz(target_path)
    except Exception as exc:
        return [f"load_target:{type(exc).__name__}:{exc}"]

    for key in (
        "trajectory",
        "agent_states",
        "agent_labels",
        "future_agent_boxes",
        "future_agent_boxes_mask",
        "bev_semantic_map",
        "future_bev_semantic_map",
        "feasible_area_mask",
        "feasible_lane_mask",
    ):
        if key in target:
            check_array(key, target.get(key), reasons)

    traj_value = target.get("trajectory")
    if traj_value is not None:
        try:
            traj = to_numpy(traj_value).astype(np.float32)
            if traj.ndim == 2 and traj.shape[-1] >= 2 and np.isfinite(traj[:, :2]).all():
                max_translation = float(np.linalg.norm(traj[:, :2], axis=-1).max())
                if args.max_trajectory_translation > 0 and max_translation > args.max_trajectory_translation:
                    reasons.append(f"outlier:trajectory_translation:{max_translation:.2f}")
                if traj.shape[0] > 1:
                    max_step = float(np.linalg.norm(traj[1:, :2] - traj[:-1, :2], axis=-1).max())
                    if args.max_trajectory_step > 0 and max_step > args.max_trajectory_step:
                        reasons.append(f"outlier:trajectory_step:{max_step:.2f}")
        except Exception as exc:
            reasons.append(f"inspect:trajectory:{type(exc).__name__}")

    if feature_path.is_file():
        try:
            feature = load_pickle_gz(feature_path)
            if "status_feature" in feature:
                check_array("status_feature", feature.get("status_feature"), reasons)
                status = to_numpy(feature.get("status_feature")).astype(np.float32)
                if status.size and np.isfinite(status).all():
                    max_status = float(np.abs(status).max())
                    if args.max_status_abs > 0 and max_status > args.max_status_abs:
                        reasons.append(f"outlier:status_feature:{max_status:.2f}")
        except Exception as exc:
            reasons.append(f"load_feature:{type(exc).__name__}:{exc}")
    else:
        reasons.append("missing:transfuser_feature.gz")
    return reasons


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-bad-print", type=int, default=100)
    parser.add_argument("--max-trajectory-translation", type=float, default=120.0)
    parser.add_argument("--max-trajectory-step", type=float, default=60.0)
    parser.add_argument("--max-status-abs", type=float, default=100.0)
    args = parser.parse_args()

    cache_root = Path(args.cache_root)
    total = 0
    bad = 0
    printed = 0
    by_reason: Dict[str, int] = {}
    for split, token_dir in tqdm(list(iter_token_dirs(cache_root, args.splits, args.limit)), desc="checking"):
        total += 1
        reasons = check_token(token_dir, args)
        if not reasons:
            continue
        bad += 1
        for reason in reasons:
            key = reason.split(":", 2)[0] + ":" + reason.split(":", 2)[1] if ":" in reason else reason
            by_reason[key] = by_reason.get(key, 0) + 1
        if printed < args.max_bad_print:
            print(f"{split}/{token_dir.name}: " + ", ".join(reasons))
            printed += 1

    print(f"checked={total} bad={bad}")
    for reason, count in sorted(by_reason.items(), key=lambda item: (-item[1], item[0])):
        print(f"{reason}: {count}")


if __name__ == "__main__":
    main()
