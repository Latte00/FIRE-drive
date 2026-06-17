import argparse
import logging
import os
import random
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import torch
from tqdm import tqdm

from navsim.planning.training.dataset import load_feature_target_from_pickle

logger = logging.getLogger(__name__)


SEMANTIC_COLORS = [
    "#111827",
    "#9ca3af",
    "#ef4444",
    "#facc15",
    "#60a5fa",
    "#22c55e",
    "#a78bfa",
    "#f97316",
    "#14b8a6",
    "#e879f9",
    "#f9fafb",
    "#6b7280",
    "#84cc16",
    "#06b6d4",
    "#fb7185",
    "#c084fc",
]

LIDAR_MIN_X = -32.0
LIDAR_MAX_X = 32.0
LIDAR_MIN_Y = -32.0
LIDAR_MAX_Y = 32.0


def _to_numpy(value) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _bev_to_class_map(bev: np.ndarray) -> np.ndarray:
    bev = _to_numpy(bev)
    if bev.ndim == 2:
        return bev.astype(np.int64)
    if bev.ndim == 3:
        if bev.shape[0] <= 32:
            return bev.argmax(axis=0).astype(np.int64)
        if bev.shape[-1] <= 32:
            return bev.argmax(axis=-1).astype(np.int64)
    raise ValueError(f"Unsupported BEV shape: {bev.shape}")


def _normalize_traj(value: np.ndarray) -> np.ndarray:
    traj = _to_numpy(value).astype(np.float32)
    traj = np.squeeze(traj)
    if traj.ndim != 2 or traj.shape[-1] < 2:
        raise ValueError(f"Unsupported trajectory shape: {traj.shape}")
    return traj


def _normalize_candidates(value: np.ndarray) -> np.ndarray:
    candidates = _to_numpy(value).astype(np.float32)
    candidates = np.squeeze(candidates)
    if candidates.ndim != 3 or candidates.shape[-1] < 2:
        raise ValueError(f"Unsupported candidates shape: {candidates.shape}")
    return candidates


def _normalize_vector(
    value,
    length: int,
    dtype,
    default,
) -> np.ndarray:
    if value is None:
        return np.full((length,), default, dtype=dtype)
    arr = np.squeeze(_to_numpy(value)).reshape(-1)
    if arr.size != length:
        return np.full((length,), default, dtype=dtype)
    return arr.astype(dtype)


def _iter_target_files(cache_path: Path, log_names: Optional[List[str]]) -> Iterable[Path]:
    if log_names:
        roots = [cache_path / name for name in log_names]
    else:
        roots = [p for p in cache_path.iterdir() if p.is_dir()]
    for root in roots:
        if not root.is_dir():
            continue
        for token_dir in root.iterdir():
            target_path = token_dir / "transfuser_target.gz"
            if target_path.is_file():
                yield target_path


def _safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("_") or "sample"


def _token_from_data(data: Dict[str, object], target_path: Path) -> str:
    token = data.get("token")
    if isinstance(token, str):
        return token
    if isinstance(token, bytes):
        return token.decode("utf-8", errors="replace")
    return target_path.parent.name


def _xy_to_image(xy: np.ndarray, pixel_size: float, width: int) -> Tuple[np.ndarray, np.ndarray]:
    row = xy[:, 0] / max(pixel_size, 1e-6)
    col = xy[:, 1] / max(pixel_size, 1e-6) + (width - 1) / 2.0
    return row, col


def _xy_to_lidar_image(
    xy: np.ndarray,
    pixel_size: float,
    display_mode: str,
) -> Tuple[np.ndarray, np.ndarray]:
    if display_mode == "ego":
        row = (LIDAR_MAX_X - xy[:, 0]) / max(pixel_size, 1e-6)
        col = (LIDAR_MAX_Y - xy[:, 1]) / max(pixel_size, 1e-6)
    elif display_mode == "tensor":
        row = (xy[:, 0] - LIDAR_MIN_X) / max(pixel_size, 1e-6)
        col = (xy[:, 1] - LIDAR_MIN_Y) / max(pixel_size, 1e-6)
    else:
        raise ValueError(f"Unsupported lidar display mode: {display_mode}")
    return row, col


def _box_corners_xy(box_state: np.ndarray) -> np.ndarray:
    x, y, heading, length, width = [float(v) for v in box_state[:5]]
    half_l = length / 2.0
    half_w = width / 2.0
    corners = np.array(
        [[half_l, half_w], [-half_l, half_w], [-half_l, -half_w], [half_l, -half_w], [half_l, half_w]],
        dtype=np.float32,
    )
    c = np.cos(heading)
    s = np.sin(heading)
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    return corners @ rot.T + np.array([x, y], dtype=np.float32)


def _normalize_lidar(value) -> np.ndarray:
    lidar = _to_numpy(value).astype(np.float32)
    lidar = np.squeeze(lidar)
    if lidar.ndim == 3:
        lidar = lidar[0]
    if lidar.ndim != 2:
        raise ValueError(f"Unsupported lidar shape: {lidar.shape}")
    return lidar


def _load_feature_for_target(target_path: Path) -> Optional[Dict[str, object]]:
    feature_path = target_path.parent / "transfuser_feature.gz"
    if not feature_path.is_file():
        logger.warning("Missing feature file for lidar overlay: %s", feature_path)
        return None
    return load_feature_target_from_pickle(feature_path)


def _candidate_colors(scores: np.ndarray, count: int) -> np.ndarray:
    if scores.size == count and np.isfinite(scores).any():
        finite_scores = scores[np.isfinite(scores)]
        min_score = float(finite_scores.min())
        max_score = float(finite_scores.max())
        denom = max(max_score - min_score, 1e-6)
        norm = np.nan_to_num((scores - min_score) / denom, nan=0.0)
        return plt.cm.viridis(np.clip(norm, 0.0, 1.0))
    return plt.cm.tab10(np.linspace(0.0, 1.0, count))


def _draw_lidar_axes(ax, display_mode: str, height: int, width: int) -> None:
    center_col = (width - 1) / 2.0
    center_row = (height - 1) / 2.0
    arrow_len = min(height, width) * 0.16
    if display_mode == "ego":
        front = (0.0, -arrow_len)
        left = (-arrow_len, 0.0)
        front_label_xy = (center_col, center_row - arrow_len - 8)
        left_label_xy = (center_col - arrow_len - 12, center_row)
        title_suffix = "ego display: top=front, left=left"
    else:
        front = (0.0, arrow_len)
        left = (arrow_len, 0.0)
        front_label_xy = (center_col, center_row + arrow_len + 14)
        left_label_xy = (center_col + arrow_len + 12, center_row)
        title_suffix = "tensor display: down=front, right=left"

    ax.scatter([center_col], [center_row], s=28, color="white", edgecolors="black", linewidths=0.5)
    ax.arrow(center_col, center_row, front[0], front[1], color="cyan", width=0.7, head_width=5.0, length_includes_head=True)
    ax.arrow(center_col, center_row, left[0], left[1], color="lime", width=0.7, head_width=5.0, length_includes_head=True)
    ax.text(*front_label_xy, "front", color="cyan", fontsize=8, ha="center", va="center")
    ax.text(*left_label_xy, "left", color="lime", fontsize=8, ha="center", va="center")
    ax.set_title(f"lidar_feature overlay\n{title_suffix}", fontsize=10)


def _plot_lidar_overlay(
    ax,
    target_path: Path,
    data: Dict[str, object],
    candidates: np.ndarray,
    mask: np.ndarray,
    scores: np.ndarray,
    gt_traj: Optional[np.ndarray],
    lidar_key: str,
    lidar_pixel_size: float,
    lidar_display_mode: str,
) -> None:
    features = _load_feature_for_target(target_path)
    if features is None or lidar_key not in features:
        ax.text(0.5, 0.5, f"missing {lidar_key}", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return

    lidar = _normalize_lidar(features[lidar_key])
    if lidar_display_mode == "ego":
        image = np.flip(lidar, axis=(0, 1))
    elif lidar_display_mode == "tensor":
        image = lidar
    else:
        raise ValueError(f"Unsupported lidar display mode: {lidar_display_mode}")

    finite = image[np.isfinite(image)]
    if finite.size and float(finite.max() - finite.min()) > 1e-6:
        vmin, vmax = np.percentile(finite, [1.0, 99.0])
    else:
        vmin, vmax = 0.0, 1.0
    ax.imshow(image, origin="upper", interpolation="nearest", cmap="magma", vmin=vmin, vmax=vmax)

    height, width = image.shape
    colors = _candidate_colors(scores, candidates.shape[0])
    for idx, traj in enumerate(candidates):
        if not bool(mask[idx]):
            continue
        xy = traj[:, :2]
        in_range = (
            (xy[:, 0] >= LIDAR_MIN_X)
            & (xy[:, 0] <= LIDAR_MAX_X)
            & (xy[:, 1] >= LIDAR_MIN_Y)
            & (xy[:, 1] <= LIDAR_MAX_Y)
        )
        if not in_range.any():
            continue
        row, col = _xy_to_lidar_image(xy[in_range], lidar_pixel_size, lidar_display_mode)
        ax.plot(col, row, color=colors[idx], linewidth=1.2, alpha=0.75)

    if gt_traj is not None:
        xy = gt_traj[:, :2]
        in_range = (
            (xy[:, 0] >= LIDAR_MIN_X)
            & (xy[:, 0] <= LIDAR_MAX_X)
            & (xy[:, 1] >= LIDAR_MIN_Y)
            & (xy[:, 1] <= LIDAR_MAX_Y)
        )
        if in_range.any():
            row, col = _xy_to_lidar_image(xy[in_range], lidar_pixel_size, lidar_display_mode)
            ax.plot(col, row, color="white", linewidth=2.2, label="GT trajectory")
            ax.scatter(col[0], row[0], s=24, color="white", edgecolors="black", linewidths=0.5)
            ax.scatter(col[-1], row[-1], s=28, color="white", marker="x")

    if "agent_states" in data:
        agent_states = _to_numpy(data["agent_states"]).astype(np.float32)
        agent_states = np.squeeze(agent_states)
        if agent_states.ndim == 2 and agent_states.shape[-1] >= 5:
            labels = _normalize_vector(
                data.get("agent_labels"),
                length=agent_states.shape[0],
                dtype=bool,
                default=True,
            )
            for state, valid in zip(agent_states, labels):
                if not bool(valid):
                    continue
                corners = _box_corners_xy(state)
                in_range = (
                    (corners[:, 0] >= LIDAR_MIN_X)
                    & (corners[:, 0] <= LIDAR_MAX_X)
                    & (corners[:, 1] >= LIDAR_MIN_Y)
                    & (corners[:, 1] <= LIDAR_MAX_Y)
                )
                if not in_range.any():
                    continue
                row, col = _xy_to_lidar_image(corners, lidar_pixel_size, lidar_display_mode)
                ax.plot(col, row, color="deepskyblue", linewidth=1.0, alpha=0.9)

    _draw_lidar_axes(ax, lidar_display_mode, height=height, width=width)
    ax.set_xlim([-0.5, width - 0.5])
    ax.set_ylim([height - 0.5, -0.5])
    ax.axis("off")


def _plot_sample(
    target_path: Path,
    data: Dict[str, object],
    output_dir: Path,
    pixel_size: float,
    bev_key: str,
    show_inactive: bool,
    include_lidar: bool,
    lidar_key: str,
    lidar_pixel_size: float,
    lidar_display_mode: str,
    dpi: int,
) -> Optional[Path]:
    if bev_key not in data:
        logger.warning("Skipping %s: missing %s", target_path, bev_key)
        return None
    if "trajectory_candidates" not in data:
        logger.warning("Skipping %s: missing trajectory_candidates", target_path)
        return None

    token = _token_from_data(data, target_path)
    class_map = _bev_to_class_map(data[bev_key])
    height, width = class_map.shape
    candidates = _normalize_candidates(data["trajectory_candidates"])
    num_candidates = candidates.shape[0]
    mask = _normalize_vector(
        data.get("trajectory_candidates_mask"),
        length=num_candidates,
        dtype=bool,
        default=True,
    )
    scores = _normalize_vector(
        data.get("pdm_score_targets"),
        length=num_candidates,
        dtype=np.float32,
        default=np.nan,
    )
    gt_score_arr = None
    if "gt_pdm_score" in data:
        gt_score_arr = np.squeeze(_to_numpy(data["gt_pdm_score"])).reshape(-1)
    gt_score = float(gt_score_arr[0]) if gt_score_arr is not None and gt_score_arr.size else np.nan

    cmap = ListedColormap(SEMANTIC_COLORS)
    if include_lidar:
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        ax = axes[0]
        ax_lidar = axes[1]
    else:
        fig, ax = plt.subplots(1, 1, figsize=(9, 9))
        ax_lidar = None
    ax.imshow(
        class_map,
        origin="upper",
        interpolation="nearest",
        cmap=cmap,
        vmin=0,
        vmax=len(SEMANTIC_COLORS) - 1,
    )

    colors = _candidate_colors(scores, num_candidates)
    for idx, traj in enumerate(candidates):
        active = bool(mask[idx])
        if not active and not show_inactive:
            continue
        xy = traj[:, :2]
        row, col = _xy_to_image(xy, pixel_size, width)
        color = colors[idx]
        if not active:
            color = (0.85, 0.85, 0.85, 0.75)
        linestyle = "-" if active else "--"
        linewidth = 2.0 if active else 1.2
        label = f"cand {idx}: {scores[idx]:.3f}" if np.isfinite(scores[idx]) else f"cand {idx}: nan"
        if not active:
            label += " inactive"
        ax.plot(col, row, color=color, linewidth=linewidth, linestyle=linestyle, label=label)
        ax.scatter(col[0], row[0], s=18, color=color, edgecolors="black", linewidths=0.3)
        ax.scatter(col[-1], row[-1], s=24, color=color, marker="x")
        if active:
            ax.text(
                col[-1],
                row[-1],
                f"{idx}",
                color="white",
                fontsize=8,
                ha="center",
                va="center",
                bbox={"boxstyle": "round,pad=0.15", "fc": "black", "ec": "none", "alpha": 0.65},
            )

    gt_traj = None
    if "trajectory" in data:
        try:
            gt_traj = _normalize_traj(data["trajectory"])
            gt_row, gt_col = _xy_to_image(gt_traj[:, :2], pixel_size, width)
            ax.plot(gt_col, gt_row, color="black", linewidth=2.5, label=f"GT: {gt_score:.3f}")
            ax.scatter(gt_col[0], gt_row[0], s=28, color="white", edgecolors="black", linewidths=0.6)
            ax.scatter(gt_col[-1], gt_row[-1], s=32, color="black", marker="x")
        except ValueError as exc:
            logger.warning("Could not plot GT for %s: %s", target_path, exc)

    ax.scatter([(width - 1) / 2.0], [0], s=52, color="white", marker="x", linewidths=1.5)
    ax.set_xlim([-0.5, width - 0.5])
    ax.set_ylim([height - 0.5, -0.5])
    ax.axis("off")
    ax.set_title(
        f"{target_path.parent.parent.name}/{token}\n"
        f"active={int(mask.sum())}/{num_candidates}, gt_pdm={gt_score:.3f}, bev={bev_key}",
        fontsize=10,
    )
    ax.legend(loc="lower left", fontsize=7, framealpha=0.78, ncol=1)
    if include_lidar and ax_lidar is not None:
        _plot_lidar_overlay(
            ax=ax_lidar,
            target_path=target_path,
            data=data,
            candidates=candidates,
            mask=mask,
            scores=scores,
            gt_traj=gt_traj,
            lidar_key=lidar_key,
            lidar_pixel_size=lidar_pixel_size,
            lidar_display_mode=lidar_display_mode,
        )
    fig.tight_layout()

    out_name = f"{_safe_name(target_path.parent.parent.name)}__{_safe_name(token)}.png"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / out_name
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _select_files(
    files: Sequence[Path],
    tokens: Optional[List[str]],
    num_samples: int,
    seed: int,
    start_index: int,
) -> List[Path]:
    if tokens:
        token_set = set(tokens)
        return [path for path in files if path.parent.name in token_set]
    files = list(files)
    if start_index > 0:
        files = files[start_index:]
    if num_samples > 0 and len(files) > num_samples:
        rng = random.Random(seed)
        files = rng.sample(files, num_samples)
        files.sort()
    return files


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    navsim_root = os.getenv("NAVSIM_EXP_ROOT")
    default_cache = str(Path(navsim_root) / "training_cache") if navsim_root else None
    default_output = (
        str(Path(navsim_root) / "candidate_visualizations")
        if navsim_root
        else "candidate_visualizations"
    )
    parser.add_argument("--cache-path", default=default_cache, required=default_cache is None)
    parser.add_argument("--output-dir", default=default_output)
    parser.add_argument("--log-names", nargs="*", default=None)
    parser.add_argument("--tokens", nargs="*", default=None)
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--bev-key", choices=["bev_semantic_map", "future_bev_semantic_map"], default="bev_semantic_map")
    parser.add_argument("--bev-pixel-size", type=float, default=0.25)
    parser.add_argument("--show-inactive", action="store_true")
    parser.add_argument("--include-lidar", action="store_true")
    parser.add_argument("--lidar-key", default="lidar_feature")
    parser.add_argument("--lidar-pixel-size", type=float, default=0.25)
    parser.add_argument(
        "--lidar-display-mode",
        choices=["ego", "tensor"],
        default="ego",
        help="ego is human-oriented top=front/left=left; tensor shows the raw feature orientation consumed by the model.",
    )
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()

    cache_path = Path(args.cache_path)
    output_dir = Path(args.output_dir)
    target_files = sorted(_iter_target_files(cache_path, args.log_names))
    selected_files = _select_files(
        target_files,
        tokens=args.tokens,
        num_samples=args.num_samples,
        seed=args.seed,
        start_index=args.start_index,
    )
    logger.info("Selected %d/%d target files", len(selected_files), len(target_files))

    saved = 0
    skipped = 0
    for target_path in tqdm(selected_files):
        try:
            data = load_feature_target_from_pickle(target_path)
            out_path = _plot_sample(
                target_path=target_path,
                data=data,
                output_dir=output_dir,
                pixel_size=args.bev_pixel_size,
                bev_key=args.bev_key,
                show_inactive=args.show_inactive,
                include_lidar=args.include_lidar,
                lidar_key=args.lidar_key,
                lidar_pixel_size=args.lidar_pixel_size,
                lidar_display_mode=args.lidar_display_mode,
                dpi=args.dpi,
            )
        except Exception as exc:
            skipped += 1
            logger.warning("Skipping %s: %s", target_path, exc)
            continue
        if out_path is None:
            skipped += 1
        else:
            saved += 1

    logger.info("Candidate visualization done. saved=%d skipped=%d output_dir=%s", saved, skipped, output_dir)


if __name__ == "__main__":
    main()
