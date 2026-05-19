import argparse
import logging
import os
import random
import re
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from navsim.planning.training.dataset import load_feature_target_from_pickle

logger = logging.getLogger(__name__)


def _to_numpy(value) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("_") or "sample"


def _iter_feature_files(cache_path: Path, log_names: Optional[List[str]]) -> Iterable[Path]:
    roots = [cache_path / name for name in log_names] if log_names else [p for p in cache_path.iterdir() if p.is_dir()]
    for root in roots:
        if not root.is_dir():
            continue
        for token_dir in sorted(root.iterdir()):
            feature_path = token_dir / "transfuser_feature.gz"
            if feature_path.is_file():
                yield feature_path


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


def _format_stats(name: str, value) -> str:
    if value is None:
        return f"{name}: missing"
    arr = _to_numpy(value)
    if arr.size == 0:
        return f"{name}: shape={arr.shape}, empty"
    finite = arr[np.isfinite(arr)] if np.issubdtype(arr.dtype, np.number) else np.asarray([])
    if finite.size:
        return (
            f"{name}: shape={arr.shape}, dtype={arr.dtype}, "
            f"min={float(finite.min()):.4f}, max={float(finite.max()):.4f}, mean={float(finite.mean()):.4f}"
        )
    return f"{name}: shape={arr.shape}, dtype={arr.dtype}"


def _camera_to_image(value) -> np.ndarray:
    arr = _to_numpy(value).astype(np.float32)
    arr = np.squeeze(arr)
    if arr.ndim == 4 and arr.shape[1] in (1, 3):
        images = [_camera_to_image(arr[idx]) for idx in range(arr.shape[0])]
        return np.concatenate(images, axis=1)
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.moveaxis(arr, 0, -1)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    if arr.ndim != 3 or arr.shape[-1] not in (1, 3):
        raise ValueError(f"Unsupported camera shape: {arr.shape}")
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if float(np.nanmin(arr)) < 0.0 or float(np.nanmax(arr)) > 1.0:
        lo, hi = np.nanpercentile(arr, [1.0, 99.0])
        arr = (arr - lo) / max(float(hi - lo), 1e-6)
    return np.clip(arr, 0.0, 1.0)


def _lidar_to_image(value) -> np.ndarray:
    arr = _to_numpy(value).astype(np.float32)
    arr = np.squeeze(arr)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Unsupported lidar shape: {arr.shape}")
    return arr


def _plot_sample(
    feature_path: Path,
    data,
    output_dir: Path,
    camera_key: str,
    lidar_key: str,
    lidar_origin: str,
    dpi: int,
) -> Optional[Path]:
    if camera_key not in data and lidar_key not in data:
        logger.warning("Skipping %s: missing %s and %s", feature_path, camera_key, lidar_key)
        return None

    split_name = feature_path.parent.parent.name
    token = feature_path.parent.name
    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(3, 2, height_ratios=[2.2, 2.0, 0.9], width_ratios=[1.0, 0.05])

    ax_cam = fig.add_subplot(gs[0, 0])
    if camera_key in data:
        camera_img = _camera_to_image(data[camera_key])
        ax_cam.imshow(camera_img)
        ax_cam.set_title(f"{split_name}/{token} - {camera_key}")
    else:
        ax_cam.text(0.5, 0.5, f"missing {camera_key}", ha="center", va="center")
    ax_cam.axis("off")

    ax_lidar = fig.add_subplot(gs[1, 0])
    cbar_ax = fig.add_subplot(gs[1, 1])
    if lidar_key in data:
        lidar_img = _lidar_to_image(data[lidar_key])
        finite = lidar_img[np.isfinite(lidar_img)]
        if finite.size and float(finite.max() - finite.min()) > 1e-6:
            vmin, vmax = np.percentile(finite, [1.0, 99.0])
        else:
            vmin, vmax = 0.0, 1.0
        im = ax_lidar.imshow(lidar_img, cmap="magma", origin=lidar_origin, interpolation="nearest", vmin=vmin, vmax=vmax)
        fig.colorbar(im, cax=cbar_ax)
        ax_lidar.set_title(f"{lidar_key} heatmap")
        if finite.size and float(finite.max() - finite.min()) <= 1e-6:
            ax_lidar.text(
                0.5,
                0.5,
                "constant lidar feature",
                color="white",
                ha="center",
                va="center",
                transform=ax_lidar.transAxes,
                bbox={"boxstyle": "round,pad=0.25", "fc": "black", "ec": "none", "alpha": 0.65},
            )
    else:
        cbar_ax.axis("off")
        ax_lidar.text(0.5, 0.5, f"missing {lidar_key}", ha="center", va="center")
    ax_lidar.axis("off")

    ax_text = fig.add_subplot(gs[2, :])
    lines = [
        _format_stats(camera_key, data.get(camera_key)),
        _format_stats(lidar_key, data.get(lidar_key)),
        _format_stats("status_feature", data.get("status_feature")),
        _format_stats("ego_history", data.get("ego_history")),
    ]
    if "status_feature" in data:
        status = np.squeeze(_to_numpy(data["status_feature"])).astype(np.float32)
        lines.append("status_feature values: " + np.array2string(status, precision=3, separator=", "))
    ax_text.text(0.01, 0.98, "\n".join(lines), ha="left", va="top", family="monospace", fontsize=8)
    ax_text.axis("off")

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{_safe_name(split_name)}__{_safe_name(token)}__inputs.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _log_sample_stats(feature_path: Path, data, camera_key: str, lidar_key: str) -> None:
    logger.info(
        "%s\n  %s\n  %s\n  %s\n  %s",
        feature_path.parent.name,
        _format_stats(camera_key, data.get(camera_key)),
        _format_stats(lidar_key, data.get(lidar_key)),
        _format_stats("status_feature", data.get("status_feature")),
        _format_stats("ego_history", data.get("ego_history")),
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    navsim_root = os.getenv("NAVSIM_EXP_ROOT")
    default_cache = str(Path(navsim_root) / "training_cache") if navsim_root else None
    default_output = str(Path(navsim_root) / "input_visualizations") if navsim_root else "input_visualizations"
    parser.add_argument("--cache-path", default=default_cache, required=default_cache is None)
    parser.add_argument("--output-dir", default=default_output)
    parser.add_argument("--log-names", nargs="*", default=None)
    parser.add_argument("--tokens", nargs="*", default=None)
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--camera-key", default="camera_feature")
    parser.add_argument("--lidar-key", default="lidar_feature")
    parser.add_argument("--lidar-origin", choices=["upper", "lower"], default="upper")
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument("--print-stats", action="store_true")
    args = parser.parse_args()

    cache_path = Path(args.cache_path)
    output_dir = Path(args.output_dir)
    feature_files = sorted(_iter_feature_files(cache_path, args.log_names))
    selected_files = _select_files(
        feature_files,
        tokens=args.tokens,
        num_samples=args.num_samples,
        seed=args.seed,
        start_index=args.start_index,
    )
    logger.info("Selected %d/%d feature files", len(selected_files), len(feature_files))

    saved = 0
    skipped = 0
    for feature_path in tqdm(selected_files):
        try:
            data = load_feature_target_from_pickle(feature_path)
            if args.print_stats:
                _log_sample_stats(feature_path, data, args.camera_key, args.lidar_key)
            out_path = _plot_sample(
                feature_path=feature_path,
                data=data,
                output_dir=output_dir,
                camera_key=args.camera_key,
                lidar_key=args.lidar_key,
                lidar_origin=args.lidar_origin,
                dpi=args.dpi,
            )
        except Exception as exc:
            skipped += 1
            logger.warning("Skipping %s: %s", feature_path, exc)
            continue
        if out_path is None:
            skipped += 1
        else:
            saved += 1

    logger.info("Input visualization done. saved=%d skipped=%d output_dir=%s", saved, skipped, output_dir)


if __name__ == "__main__":
    main()
