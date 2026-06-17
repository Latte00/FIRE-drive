from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
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


def _collect_cache_entries(
    cache_path: Path, feature_filename: str
) -> Dict[str, List[Tuple[str, Path]]]:
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


def _build_sensor_config(history_idx: int) -> SensorConfig:
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


def _camera_to_lidar2img(camera: Camera) -> np.ndarray:
    if (
        camera.intrinsics is None
        or camera.sensor2lidar_rotation is None
        or camera.sensor2lidar_translation is None
    ):
        raise ValueError("Camera calibration is incomplete.")

    rotation = np.asarray(camera.sensor2lidar_rotation, dtype=np.float32)
    translation = np.asarray(camera.sensor2lidar_translation, dtype=np.float32)
    intrinsics = np.asarray(camera.intrinsics, dtype=np.float32)

    lidar2cam_r = np.linalg.inv(rotation)
    lidar2cam_t = translation @ lidar2cam_r.T
    lidar2cam_rt = np.eye(4, dtype=np.float32)
    lidar2cam_rt[:3, :3] = lidar2cam_r.T
    lidar2cam_rt[3, :3] = -lidar2cam_t

    viewpad = np.eye(4, dtype=np.float32)
    viewpad[: intrinsics.shape[0], : intrinsics.shape[1]] = intrinsics
    return (viewpad @ lidar2cam_rt.T).astype(np.float32)


def _parse_triplet(values: str) -> np.ndarray:
    parts = [v.strip() for v in values.split(",") if v.strip()]
    if len(parts) != 3:
        raise ValueError(f"Expect 3 comma separated values, got: {values}")
    return np.asarray([float(v) for v in parts], dtype=np.float32)


def _pad_to_divisor(img: np.ndarray, size_divisor: int) -> np.ndarray:
    h, w = img.shape[:2]
    target_h = int(np.ceil(h / size_divisor) * size_divisor)
    target_w = int(np.ceil(w / size_divisor) * size_divisor)
    if target_h == h and target_w == w:
        return img
    out = np.zeros((target_h, target_w, img.shape[2]), dtype=img.dtype)
    out[:h, :w] = img
    return out


def _build_zero_image(height: int, width: int) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.float32)


def _process_pad_camera(
    *,
    camera: Optional[Camera],
    image: Optional[np.ndarray],
    fallback_hw: Tuple[int, int],
    mean: np.ndarray,
    std: np.ndarray,
    to_rgb: bool,
    scale: float,
    size_divisor: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """
    Return: padded image(HWC), lidar2img(4x4), img_shape(3,), valid_flag
    """
    valid = (
        camera is not None
        and image is not None
        and camera.intrinsics is not None
        and camera.sensor2lidar_rotation is not None
        and camera.sensor2lidar_translation is not None
    )

    if image is None:
        img = _build_zero_image(fallback_hw[0], fallback_hw[1])
    else:
        img = image.astype(np.float32)

    if to_rgb:
        img = img[..., ::-1]
    img = (img - mean.reshape(1, 1, 3)) / np.maximum(std.reshape(1, 1, 3), 1e-6)

    lidar2img = np.eye(4, dtype=np.float32)
    if valid:
        try:
            lidar2img = _camera_to_lidar2img(camera)
        except (ValueError, np.linalg.LinAlgError):
            valid = False
            lidar2img = np.eye(4, dtype=np.float32)

    if abs(scale - 1.0) > 1e-6:
        h, w = img.shape[:2]
        out_w = max(1, int(round(w * scale)))
        out_h = max(1, int(round(h * scale)))
        img = cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        if valid:
            scale_mat = np.eye(4, dtype=np.float32)
            scale_mat[0, 0] = scale
            scale_mat[1, 1] = scale
            lidar2img = (scale_mat @ lidar2img).astype(np.float32)

    img = _pad_to_divisor(img, size_divisor=size_divisor)
    img_shape = np.asarray([img.shape[0], img.shape[1], img.shape[2]], dtype=np.float32)
    return img, lidar2img, img_shape, valid


def _patch_log(
    *,
    scene_loader: SceneLoader,
    entries: Sequence[Tuple[str, Path]],
    cam_order: Sequence[str],
    camera_feature_key: str,
    lidar2img_key: str,
    img_shape_key: str,
    valid_mask_key: str,
    mean: np.ndarray,
    std: np.ndarray,
    to_rgb: bool,
    scale: float,
    size_divisor: int,
    default_image_h: int,
    default_image_w: int,
    camera_dtype: str,
    overwrite: bool,
    dry_run: bool,
) -> Dict[str, int]:
    stats = {
        "total": 0,
        "updated": 0,
        "skipped_exists": 0,
        "missing_scene": 0,
    }

    token_set = set(scene_loader.tokens)
    for token, feature_path in entries:
        stats["total"] += 1
        if token not in token_set:
            stats["missing_scene"] += 1
            continue

        data_dict = load_feature_target_from_pickle(feature_path)
        has_all_keys = (
            camera_feature_key in data_dict
            and lidar2img_key in data_dict
            and img_shape_key in data_dict
            and valid_mask_key in data_dict
        )
        if has_all_keys and not overwrite:
            stats["skipped_exists"] += 1
            continue

        agent_input = scene_loader.get_agent_input_from_token(token)
        cameras = agent_input.cameras[-1]

        fallback_hw = (default_image_h, default_image_w)
        for cam_name in cam_order:
            cam_obj = getattr(cameras, cam_name, None)
            cam_img = None if cam_obj is None else cam_obj.image
            if cam_img is not None:
                fallback_hw = (int(cam_img.shape[0]), int(cam_img.shape[1]))
                break

        imgs_hwc: List[np.ndarray] = []
        lidar2img_list: List[np.ndarray] = []
        valid_list: List[bool] = []

        for cam_name in cam_order:
            cam_obj = getattr(cameras, cam_name, None)
            cam_img = None if cam_obj is None else cam_obj.image
            padded_img, lidar2img, img_shape, valid = _process_pad_camera(
                camera=cam_obj,
                image=cam_img,
                fallback_hw=fallback_hw,
                mean=mean,
                std=std,
                to_rgb=to_rgb,
                scale=scale,
                size_divisor=size_divisor,
            )
            imgs_hwc.append(padded_img)
            lidar2img_list.append(lidar2img)
            valid_list.append(valid)

        # Safety: align to same padded shape if camera native sizes differ.
        max_h = max(int(img.shape[0]) for img in imgs_hwc)
        max_w = max(int(img.shape[1]) for img in imgs_hwc)
        imgs_chw: List[np.ndarray] = []
        aligned_shapes: List[np.ndarray] = []
        for img in imgs_hwc:
            if img.shape[0] != max_h or img.shape[1] != max_w:
                tmp = np.zeros((max_h, max_w, img.shape[2]), dtype=img.dtype)
                tmp[: img.shape[0], : img.shape[1]] = img
                img = tmp
            imgs_chw.append(np.ascontiguousarray(img.transpose(2, 0, 1)))
            aligned_shapes.append(np.asarray([max_h, max_w, img.shape[2]], dtype=np.float32))

        camera_tensor = torch.from_numpy(np.stack(imgs_chw, axis=0))
        if camera_dtype == "float16":
            camera_tensor = camera_tensor.to(torch.float16)
        else:
            camera_tensor = camera_tensor.to(torch.float32)
        data_dict[camera_feature_key] = camera_tensor
        data_dict[lidar2img_key] = torch.from_numpy(np.stack(lidar2img_list, axis=0)).to(torch.float32)
        data_dict[img_shape_key] = torch.from_numpy(np.stack(aligned_shapes, axis=0)).to(torch.float32)
        data_dict[valid_mask_key] = torch.from_numpy(np.asarray(valid_list, dtype=np.bool_))

        if not dry_run:
            dump_feature_target_to_pickle(feature_path, data_dict)
        stats["updated"] += 1

    return stats


def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Patch existing feature cache with PAD-style 4-camera tensors and projection metadata."
    )
    ap.add_argument("--cache-path", type=Path, required=True, help="Path to training cache root.")
    ap.add_argument("--data-path", type=Path, required=True, help="Path to navsim logs (*.pkl) root.")
    ap.add_argument("--sensor-blobs-path", type=Path, required=True, help="Path to sensor blobs root.")
    ap.add_argument("--feature-filename", default="transfuser_feature.gz")
    ap.add_argument("--num-history-frames", type=int, default=4)
    ap.add_argument("--num-future-frames", type=int, default=10)
    ap.add_argument("--frame-interval", type=int, default=1)
    ap.add_argument("--no-route-filter", action="store_true", help="Disable has_route in SceneFilter.")
    ap.add_argument(
        "--cam-order",
        default="cam_b0,cam_f0,cam_l0,cam_r0",
        help="Comma-separated camera order (PAD default: b0,f0,l0,r0).",
    )
    ap.add_argument("--camera-feature-key", default="camera_feature_4cam_pad")
    ap.add_argument("--lidar2img-key", default="lidar2img_4cam_pad")
    ap.add_argument("--img-shape-key", default="img_shape_4cam_pad")
    ap.add_argument("--valid-mask-key", default="camera_valid_mask_4cam_pad")
    ap.add_argument("--mean", default="123.675,116.28,103.53")
    ap.add_argument("--std", default="58.395,57.12,57.375")
    ap.add_argument("--no-to-rgb", action="store_true", help="Disable BGR->RGB before normalization.")
    ap.add_argument("--scale", type=float, default=0.4, help="PAD default scale.")
    ap.add_argument("--size-divisor", type=int, default=32, help="PAD default divisor.")
    ap.add_argument("--default-image-h", type=int, default=1080)
    ap.add_argument("--default-image-w", type=int, default=1920)
    ap.add_argument("--camera-dtype", choices=("float32", "float16"), default="float32")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    return ap


def main() -> None:
    args = _build_argparser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    if not args.cache_path.is_dir():
        raise FileNotFoundError(f"Cache path does not exist: {args.cache_path}")
    if not args.data_path.is_dir():
        raise FileNotFoundError(f"Data path does not exist: {args.data_path}")
    if not args.sensor_blobs_path.is_dir():
        raise FileNotFoundError(f"Sensor blobs path does not exist: {args.sensor_blobs_path}")
    if args.size_divisor <= 0:
        raise ValueError("--size-divisor must be > 0")

    cam_order = [c.strip() for c in args.cam_order.split(",") if c.strip()]
    if not cam_order:
        raise ValueError("Empty --cam-order")

    mean = _parse_triplet(args.mean)
    std = _parse_triplet(args.std)
    to_rgb = not args.no_to_rgb

    entries_by_log = _collect_cache_entries(args.cache_path, args.feature_filename)
    if not entries_by_log:
        logger.warning("No cache entries found for feature filename: %s", args.feature_filename)
        return

    logger.info("Found %d logs with cache entries", len(entries_by_log))
    history_idx = args.num_history_frames - 1
    sensor_config = _build_sensor_config(history_idx=history_idx)

    total = {
        "total": 0,
        "updated": 0,
        "skipped_exists": 0,
        "missing_scene": 0,
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
            camera_feature_key=args.camera_feature_key,
            lidar2img_key=args.lidar2img_key,
            img_shape_key=args.img_shape_key,
            valid_mask_key=args.valid_mask_key,
            mean=mean,
            std=std,
            to_rgb=to_rgb,
            scale=float(args.scale),
            size_divisor=int(args.size_divisor),
            default_image_h=int(args.default_image_h),
            default_image_w=int(args.default_image_w),
            camera_dtype=args.camera_dtype,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )

        for key, value in stats.items():
            total[key] += value

        logger.info(
            "log=%s total=%d updated=%d skipped_exists=%d missing_scene=%d",
            log_name,
            stats["total"],
            stats["updated"],
            stats["skipped_exists"],
            stats["missing_scene"],
        )

    logger.info(
        "DONE total=%d updated=%d skipped_exists=%d missing_scene=%d dry_run=%s",
        total["total"],
        total["updated"],
        total["skipped_exists"],
        total["missing_scene"],
        args.dry_run,
    )


if __name__ == "__main__":
    main()
