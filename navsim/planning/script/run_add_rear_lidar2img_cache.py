from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import Camera, SceneFilter, SensorConfig
from navsim.planning.training.dataset import (
    dump_feature_target_to_pickle,
    load_feature_target_from_pickle,
)

logger = logging.getLogger(__name__)


def _collect_cache_entries(cache_path: Path, feature_filename: str) -> Dict[str, List[Tuple[str, Path]]]:
    """Collect token -> feature path entries grouped by log name."""
    entries_by_log: Dict[str, List[Tuple[str, Path]]] = defaultdict(list)
    for log_dir in sorted(cache_path.iterdir()):
        if not log_dir.is_dir():
            continue
        for token_dir in sorted(log_dir.iterdir()):
            if not token_dir.is_dir():
                continue
            feature_path = token_dir / feature_filename
            if feature_path.is_file():
                entries_by_log[log_dir.name].append((token_dir.name, feature_path))
    return entries_by_log


def _camera_to_lidar2img(camera: Camera) -> np.ndarray:
    """Compute lidar2img matrix from camera calibration."""
    if (
        camera.intrinsics is None
        or camera.sensor2lidar_rotation is None
        or camera.sensor2lidar_translation is None
    ):
        raise ValueError("Camera calibration is incomplete.")

    lidar2cam_r = np.linalg.inv(np.asarray(camera.sensor2lidar_rotation, dtype=np.float32))
    lidar2cam_t = np.asarray(camera.sensor2lidar_translation, dtype=np.float32) @ lidar2cam_r.T

    lidar2cam_rt = np.eye(4, dtype=np.float32)
    lidar2cam_rt[:3, :3] = lidar2cam_r.T
    lidar2cam_rt[3, :3] = -lidar2cam_t

    intrinsic = np.asarray(camera.intrinsics, dtype=np.float32)
    viewpad = np.eye(4, dtype=np.float32)
    viewpad[: intrinsic.shape[0], : intrinsic.shape[1]] = intrinsic

    return (viewpad @ lidar2cam_rt.T).astype(np.float32)


def _build_image_transform(
    src_h: int,
    src_w: int,
    crop_top: int,
    crop_bottom: int,
    crop_left: int,
    crop_right: int,
    out_h: Optional[int],
    out_w: Optional[int],
) -> Tuple[np.ndarray, int, int]:
    """Build 4x4 image-plane transform for crop+resize."""
    h1 = src_h - crop_top - crop_bottom
    w1 = src_w - crop_left - crop_right
    if h1 <= 0 or w1 <= 0:
        raise ValueError(
            f"Invalid crop: src=({src_h},{src_w}), crop=(top={crop_top}, bottom={crop_bottom}, "
            f"left={crop_left}, right={crop_right})"
        )

    h2 = h1 if out_h is None else int(out_h)
    w2 = w1 if out_w is None else int(out_w)
    if h2 <= 0 or w2 <= 0:
        raise ValueError(f"Invalid target size: ({h2}, {w2})")

    sx = float(w2) / float(w1)
    sy = float(h2) / float(h1)

    transform = np.eye(4, dtype=np.float32)
    transform[0, 0] = sx
    transform[1, 1] = sy
    transform[0, 2] = -float(crop_left) * sx
    transform[1, 2] = -float(crop_top) * sy
    return transform, h2, w2


def _to_rear_tensor(
    image: np.ndarray,
    *,
    crop_top: int,
    crop_bottom: int,
    crop_left: int,
    crop_right: int,
    out_h: Optional[int],
    out_w: Optional[int],
    dtype: str,
) -> Tuple[torch.Tensor, Tuple[int, int], np.ndarray]:
    """
    Convert rear image (HWC uint8/float) to tensor CHW and return image transform.
    """
    if image is None:
        raise ValueError("Rear camera image is missing.")

    src_h, src_w = image.shape[:2]
    transform, dst_h, dst_w = _build_image_transform(
        src_h=src_h,
        src_w=src_w,
        crop_top=crop_top,
        crop_bottom=crop_bottom,
        crop_left=crop_left,
        crop_right=crop_right,
        out_h=out_h,
        out_w=out_w,
    )

    h_start = crop_top
    h_end = src_h - crop_bottom if crop_bottom > 0 else src_h
    w_start = crop_left
    w_end = src_w - crop_right if crop_right > 0 else src_w
    cropped = image[h_start:h_end, w_start:w_end]

    if (dst_h, dst_w) != cropped.shape[:2]:
        import cv2  # lazy import; only needed when resize is requested

        cropped = cv2.resize(cropped, (dst_w, dst_h))

    chw = np.transpose(cropped, (2, 0, 1)).copy()
    if dtype == "uint8":
        tensor = torch.from_numpy(chw).to(torch.uint8)
    elif dtype == "float32":
        tensor = torch.from_numpy(chw).to(torch.float32)
        if tensor.max().item() > 1.0:
            tensor = tensor / 255.0
    else:
        raise ValueError(f"Unsupported rear dtype: {dtype}")

    return tensor, (dst_h, dst_w), transform


def _build_sensor_config(history_idx: int) -> SensorConfig:
    """Load only required cameras for the current/history frame."""
    return SensorConfig(
        cam_f0=[history_idx],
        cam_l0=[history_idx],
        cam_l1=[],
        cam_l2=[],
        cam_r0=[history_idx],
        cam_r1=[],
        cam_r2=[],
        cam_b0=[history_idx],
        lidar_pc=False,
    )


def _iter_cam_order(cam_order: Sequence[str]) -> Iterable[str]:
    for cam_name in cam_order:
        c = cam_name.strip()
        if c:
            yield c


def _patch_log(
    *,
    scene_loader: SceneLoader,
    entries: Sequence[Tuple[str, Path]],
    cam_order: Sequence[str],
    rear_cam_name: str,
    rear_key: str,
    lidar2img_key: str,
    img_shape_key: str,
    valid_mask_key: str,
    overwrite: bool,
    dry_run: bool,
    rear_crop_top: int,
    rear_crop_bottom: int,
    rear_crop_left: int,
    rear_crop_right: int,
    rear_out_h: Optional[int],
    rear_out_w: Optional[int],
    rear_dtype: str,
    apply_rear_transform_to_lidar2img: bool,
) -> Dict[str, int]:
    stats = {
        "total": 0,
        "updated": 0,
        "skipped_exists": 0,
        "missing_scene": 0,
        "missing_rear": 0,
    }

    token_set = set(scene_loader.tokens)

    for token, feature_path in entries:
        stats["total"] += 1

        if token not in token_set:
            stats["missing_scene"] += 1
            continue

        data_dict = load_feature_target_from_pickle(feature_path)
        has_all_keys = (
            rear_key in data_dict
            and lidar2img_key in data_dict
            and img_shape_key in data_dict
            and valid_mask_key in data_dict
        )
        if has_all_keys and not overwrite:
            stats["skipped_exists"] += 1
            continue

        agent_input = scene_loader.get_agent_input_from_token(token)
        cameras = agent_input.cameras[-1]

        rear_cam: Camera = getattr(cameras, rear_cam_name)
        if rear_cam.image is None:
            stats["missing_rear"] += 1
            continue

        rear_tensor, rear_hw, rear_transform = _to_rear_tensor(
            rear_cam.image,
            crop_top=rear_crop_top,
            crop_bottom=rear_crop_bottom,
            crop_left=rear_crop_left,
            crop_right=rear_crop_right,
            out_h=rear_out_h,
            out_w=rear_out_w,
            dtype=rear_dtype,
        )

        lidar2img_list: List[np.ndarray] = []
        img_shape_list: List[List[int]] = []
        valid_list: List[bool] = []

        for cam_name in _iter_cam_order(cam_order):
            cam: Camera = getattr(cameras, cam_name)
            valid = (
                cam.image is not None
                and cam.intrinsics is not None
                and cam.sensor2lidar_rotation is not None
                and cam.sensor2lidar_translation is not None
            )
            valid_list.append(bool(valid))

            if not valid:
                lidar2img_list.append(np.eye(4, dtype=np.float32))
                img_shape_list.append([0, 0])
                continue

            lidar2img = _camera_to_lidar2img(cam)
            h, w = cam.image.shape[:2]
            if cam_name == rear_cam_name:
                h, w = rear_hw
                if apply_rear_transform_to_lidar2img:
                    lidar2img = (rear_transform @ lidar2img).astype(np.float32)

            lidar2img_list.append(lidar2img)
            img_shape_list.append([int(h), int(w)])

        data_dict[rear_key] = rear_tensor
        data_dict[lidar2img_key] = torch.from_numpy(np.stack(lidar2img_list, axis=0)).to(torch.float32)
        data_dict[img_shape_key] = torch.from_numpy(np.asarray(img_shape_list, dtype=np.int32))
        data_dict[valid_mask_key] = torch.from_numpy(np.asarray(valid_list, dtype=np.bool_))

        if not dry_run:
            dump_feature_target_to_pickle(feature_path, data_dict)
        stats["updated"] += 1

    return stats


def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Patch existing transfuser feature cache with rear camera tensor and lidar2img metadata."
    )
    ap.add_argument("--cache-path", type=Path, required=True, help="Path to training cache root.")
    ap.add_argument("--data-path", type=Path, required=True, help="Path to navsim logs (*.pkl).")
    ap.add_argument("--sensor-blobs-path", type=Path, required=True, help="Path to sensor blobs root.")
    ap.add_argument("--feature-filename", default="transfuser_feature.gz")
    ap.add_argument("--num-history-frames", type=int, default=4)
    ap.add_argument("--num-future-frames", type=int, default=10)
    ap.add_argument("--frame-interval", type=int, default=1)
    ap.add_argument("--no-route-filter", action="store_true", help="Disable has_route filter in SceneFilter.")
    ap.add_argument(
        "--cam-order",
        default="cam_l0,cam_f0,cam_r0,cam_b0",
        help="Comma-separated camera order used for lidar2img/img_shape.",
    )
    ap.add_argument("--rear-cam-name", default="cam_b0")
    ap.add_argument("--rear-key", default="camera_rear_feature")
    ap.add_argument("--lidar2img-key", default="lidar2img_4cam")
    ap.add_argument("--img-shape-key", default="img_shape_4cam")
    ap.add_argument("--valid-mask-key", default="camera_valid_mask_4cam")
    ap.add_argument("--rear-dtype", choices=("uint8", "float32"), default="uint8")
    ap.add_argument("--rear-out-h", type=int, default=None)
    ap.add_argument("--rear-out-w", type=int, default=None)
    ap.add_argument("--rear-crop-top", type=int, default=0)
    ap.add_argument("--rear-crop-bottom", type=int, default=0)
    ap.add_argument("--rear-crop-left", type=int, default=0)
    ap.add_argument("--rear-crop-right", type=int, default=0)
    ap.add_argument(
        "--disable-rear-lidar2img-transform",
        action="store_true",
        help="Do not apply rear crop/resize transform to rear lidar2img matrix.",
    )
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing keys if present.")
    ap.add_argument("--dry-run", action="store_true")
    return ap


def main() -> None:
    args = _build_argparser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    cache_path: Path = args.cache_path
    if not cache_path.is_dir():
        raise FileNotFoundError(f"Cache path does not exist: {cache_path}")
    if not args.data_path.is_dir():
        raise FileNotFoundError(f"Data path does not exist: {args.data_path}")
    if not args.sensor_blobs_path.is_dir():
        raise FileNotFoundError(f"Sensor blobs path does not exist: {args.sensor_blobs_path}")

    cam_order = [c.strip() for c in args.cam_order.split(",") if c.strip()]
    if args.rear_cam_name not in cam_order:
        raise ValueError(f"--rear-cam-name={args.rear_cam_name} must be included in --cam-order={cam_order}")

    entries_by_log = _collect_cache_entries(cache_path, args.feature_filename)
    if not entries_by_log:
        logger.warning("No cache entries found for feature filename: %s", args.feature_filename)
        return

    logger.info("Found %d logs with cache entries.", len(entries_by_log))
    history_idx = args.num_history_frames - 1
    sensor_config = _build_sensor_config(history_idx=history_idx)

    total_stats = {
        "total": 0,
        "updated": 0,
        "skipped_exists": 0,
        "missing_scene": 0,
        "missing_rear": 0,
    }

    for log_name, entries in tqdm(entries_by_log.items(), desc="Logs"):
        tokens = [token for token, _ in entries]
        scene_filter = SceneFilter(
            num_history_frames=args.num_history_frames,
            num_future_frames=args.num_future_frames,
            frame_interval=args.frame_interval,
            has_route=not args.no_route_filter,
            max_scenes=None,
            log_names=[log_name],
            tokens=tokens,
        )
        scene_loader = SceneLoader(
            data_path=args.data_path,
            sensor_blobs_path=args.sensor_blobs_path,
            scene_filter=scene_filter,
            sensor_config=sensor_config,
        )

        stats = _patch_log(
            scene_loader=scene_loader,
            entries=entries,
            cam_order=cam_order,
            rear_cam_name=args.rear_cam_name,
            rear_key=args.rear_key,
            lidar2img_key=args.lidar2img_key,
            img_shape_key=args.img_shape_key,
            valid_mask_key=args.valid_mask_key,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
            rear_crop_top=args.rear_crop_top,
            rear_crop_bottom=args.rear_crop_bottom,
            rear_crop_left=args.rear_crop_left,
            rear_crop_right=args.rear_crop_right,
            rear_out_h=args.rear_out_h,
            rear_out_w=args.rear_out_w,
            rear_dtype=args.rear_dtype,
            apply_rear_transform_to_lidar2img=not args.disable_rear_lidar2img_transform,
        )

        for k, v in stats.items():
            total_stats[k] += v

        logger.info(
            "log=%s total=%d updated=%d skipped_exists=%d missing_scene=%d missing_rear=%d",
            log_name,
            stats["total"],
            stats["updated"],
            stats["skipped_exists"],
            stats["missing_scene"],
            stats["missing_rear"],
        )

    logger.info(
        "DONE total=%d updated=%d skipped_exists=%d missing_scene=%d missing_rear=%d dry_run=%s",
        total_stats["total"],
        total_stats["updated"],
        total_stats["skipped_exists"],
        total_stats["missing_scene"],
        total_stats["missing_rear"],
        args.dry_run,
    )


if __name__ == "__main__":
    main()
