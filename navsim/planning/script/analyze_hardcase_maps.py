from __future__ import annotations

import argparse
import lzma
import pickle
import re
from pathlib import Path
from typing import Dict, Iterable, Optional, Any
import traceback
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf, open_dict
from nuplan.common.actor_state.car_footprint import CarFootprint
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters

# Ensure repo root is importable for pickle deserialization of navsim classes.
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.diffusiondrive.transfuser_callback import semantic_map_to_rgb
from navsim.common.dataclasses import Annotations
from navsim.common.dataloader import SceneLoader
from navsim.common.enums import BoundingBoxIndex
from navsim.planning.scenario_builder.navsim_scenario_utils import tracked_object_types
from navsim.visualization.bev import add_annotations_to_bev_ax
from navsim.visualization.config import AGENT_CONFIG

MAP_BASE_FACE = "#f4f5f7"
MAP_PANEL_FACE = "#f7f8fb"
MAP_LANE_FILL = "#d8dce3"
MAP_LANE_EDGE = "#adb5c2"
MAP_ROUTE_FILL = "#d9e3ff"
MAP_ROUTE_EDGE = "#5c78d6"
MAP_CENTERLINE = "#4f6cd6"
MAP_ALL_MODES = "#8ea4ff"
MAP_SELECTED = "#3156d3"
MAP_GT = "#7a4bc2"
MAP_EGO = "#111111"
MAP_EGO_BOX = "#2859d8"
MAP_AGENT_BOX = "#3f444d"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze hard cases from run_pdm_score CSV and render BEV maps from metric cache."
        )
    )
    parser.add_argument("--score-csv", type=Path, required=True, help="Path to pdm score csv.")
    parser.add_argument(
        "--metric-cache-path",
        type=Path,
        required=True,
        help="Root directory of metric cache.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory. Default: <score-csv-dir>/hardcase_maps",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.0,
        help="Select scenes with score <= threshold for map export.",
    )
    parser.add_argument(
        "--score-compare",
        choices=("le", "ge"),
        default="le",
        help="Score selection direction: le keeps score <= threshold, ge keeps score >= threshold.",
    )
    parser.add_argument(
        "--only-hard",
        action="store_true",
        help="If set, score-threshold export only keeps hard scenes.",
    )
    parser.add_argument(
        "--hard-threshold",
        type=float,
        default=None,
        help=(
            "Hard threshold for hc_u_adapter_variance. "
            "Ignored when hc_is_hardcase exists in CSV."
        ),
    )
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=200,
        help="Maximum number of map figures to export.",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=80.0,
        help="BEV visualization half range (meters).",
    )
    parser.add_argument("--dpi", type=int, default=180, help="Figure dpi.")
    parser.add_argument(
        "--path-rewrite-old",
        type=str,
        default=None,
        help="Optional old prefix in metadata path (for moved cache).",
    )
    parser.add_argument(
        "--path-rewrite-new",
        type=str,
        default=None,
        help="Optional new prefix in metadata path (for moved cache).",
    )
    parser.add_argument(
        "--hydra-config",
        type=Path,
        default=None,
        help=(
            "Optional hydra config.yaml from a run directory. When provided, the script also "
            "loads the model and renders predicted BEV and trajectory diagnostics."
        ),
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help="Optional checkpoint override used together with --hydra-config.",
    )
    parser.add_argument(
        "--navsim-log-path",
        type=Path,
        default=None,
        help="Optional override for cfg.navsim_log_path when loading scenes for model rendering.",
    )
    parser.add_argument(
        "--sensor-blobs-path",
        type=Path,
        default=None,
        help="Optional override for cfg.sensor_blobs_path when loading scenes for model rendering.",
    )
    parser.add_argument(
        "--ignore-config-log-names",
        action="store_true",
        help=(
            "Ignore cfg.train_test_split.scene_filter.log_names when loading scenes. "
            "Useful when visualizing a score CSV from a different split than the hydra config."
        ),
    )
    parser.add_argument(
        "--no-vector-attention",
        action="store_true",
        help="Disable selected-mode attention heatmap overlay on vector_map_viz.",
    )
    parser.add_argument(
        "--vector-attention-alpha",
        type=float,
        default=0.38,
        help="Alpha for selected-mode attention heatmap overlay on vector_map_viz.",
    )
    parser.add_argument(
        "--vector-attention-min",
        type=float,
        default=0.12,
        help="Normalized attention values below this threshold are fully transparent on vector_map_viz.",
    )
    parser.add_argument(
        "--vector-attention-cmap-min",
        type=float,
        default=0.25,
        help="Lower bound sampled from magma colormap, skipping the black low end.",
    )
    parser.add_argument(
        "--vector-attention-alpha-gamma",
        type=float,
        default=0.8,
        help="Gamma for attention alpha ramp. Smaller values make low-mid heat regions more visible.",
    )
    return parser.parse_args()


def _load_score_df(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "token" not in df.columns:
        raise ValueError(f"CSV missing token column: {csv_path}")
    df = df[df["token"].astype(str) != "average"].copy()
    if "score" not in df.columns:
        raise ValueError(f"CSV missing score column: {csv_path}")
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    if "hc_u_adapter_variance" in df.columns:
        df["hc_u_adapter_variance"] = pd.to_numeric(df["hc_u_adapter_variance"], errors="coerce")
    return df


def _hard_mask(df: pd.DataFrame, hard_threshold: Optional[float]) -> pd.Series:
    if "hc_is_hardcase" in df.columns:
        return pd.to_numeric(df["hc_is_hardcase"], errors="coerce").fillna(0.0) >= 0.5
    if "hc_u_adapter_variance" in df.columns and hard_threshold is not None:
        return df["hc_u_adapter_variance"].fillna(-np.inf) >= float(hard_threshold)
    return pd.Series(False, index=df.index)


def _extract_token_from_cache_path(path_str: str) -> Optional[str]:
    parts = [p for p in re.split(r"[\\/]+", path_str.strip()) if p]
    if len(parts) < 2:
        return None
    return parts[-2]


def _rewrite_path(path_str: str, old: Optional[str], new: Optional[str]) -> str:
    if not old or not new:
        return path_str
    if path_str.startswith(old):
        return new + path_str[len(old) :]
    return path_str


def _load_metric_cache_paths_from_metadata(
    metric_cache_path: Path,
    rewrite_old: Optional[str],
    rewrite_new: Optional[str],
) -> Dict[str, Path]:
    token_to_path: Dict[str, Path] = {}
    metadata_dir = metric_cache_path / "metadata"
    if not metadata_dir.exists():
        return token_to_path

    metadata_files = sorted(metadata_dir.glob("*.csv"))
    for file in metadata_files:
        lines = file.read_text(encoding="utf-8", errors="ignore").splitlines()
        if len(lines) <= 1:
            continue
        for line in lines[1:]:
            # Metadata files in this project usually have one path per line.
            # If there are commas, use the last field as path.
            raw = line.split(",")[-1].strip()
            if not raw:
                continue
            raw = _rewrite_path(raw, rewrite_old, rewrite_new)
            token = _extract_token_from_cache_path(raw)
            if token is None:
                continue
            token_to_path[token] = Path(raw)
    return token_to_path


def _index_metric_cache_recursively(metric_cache_path: Path) -> Dict[str, Path]:
    token_to_path: Dict[str, Path] = {}
    for cache_file in metric_cache_path.rglob("metric_cache.pkl"):
        token = cache_file.parent.name
        token_to_path[token] = cache_file
    return token_to_path


def _iter_polygons(geometry) -> Iterable:
    if geometry is None or getattr(geometry, "is_empty", True):
        return []
    geom_type = getattr(geometry, "geom_type", "")
    if geom_type == "Polygon":
        return [geometry]
    if geom_type == "MultiPolygon":
        return list(geometry.geoms)
    return []


def _draw_metric_cache_map(
    metric_cache,
    score_row: pd.Series,
    save_path: Path,
    radius: float,
    dpi: int,
    annotations: Optional[Annotations] = None,
) -> None:
    drivable_map = metric_cache.drivable_area_map
    route_lane_ids = set(metric_cache.route_lane_ids)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    fig.patch.set_facecolor(MAP_BASE_FACE)
    ax.set_facecolor(MAP_PANEL_FACE)

    for token, layer, geom in zip(
        drivable_map.tokens,
        drivable_map.map_types,
        drivable_map._geometries,
    ):
        layer_name = getattr(layer, "name", str(layer))
        if token in route_lane_ids:
            face_color = MAP_ROUTE_FILL
            edge_color = MAP_ROUTE_EDGE
            alpha = 0.55
            lw = 0.65
        elif "LANE_CONNECTOR" in layer_name:
            face_color = "#e4e8f1"
            edge_color = "#b4bdcc"
            alpha = 0.45
            lw = 0.35
        elif "LANE" in layer_name:
            face_color = MAP_LANE_FILL
            edge_color = MAP_LANE_EDGE
            alpha = 0.40
            lw = 0.30
        elif "INTERSECTION" in layer_name:
            face_color = "#e2e6ee"
            edge_color = "#b8c0cc"
            alpha = 0.35
            lw = 0.30
        else:
            face_color = MAP_LANE_FILL
            edge_color = MAP_LANE_EDGE
            alpha = 0.28
            lw = 0.30

        for poly in _iter_polygons(geom):
            x, y = poly.exterior.xy
            ax.fill(x, y, facecolor=face_color, edgecolor=edge_color, alpha=alpha, linewidth=lw)

    centerline = metric_cache.centerline.linestring
    if centerline is not None:
        x, y = centerline.xy
        ax.plot(x, y, color=MAP_CENTERLINE, linewidth=1.4, alpha=0.95, linestyle="--", label="centerline")

    if hasattr(metric_cache.trajectory, "get_sampled_trajectory"):
        sampled_states = metric_cache.trajectory.get_sampled_trajectory()
        traj_x = [state.rear_axle.x for state in sampled_states]
        traj_y = [state.rear_axle.y for state in sampled_states]
        ax.plot(traj_x, traj_y, color=MAP_GT, linewidth=1.8, alpha=0.95, label="reference")

    if annotations is not None:
        add_annotations_to_bev_ax(ax, annotations, add_ego=False)

    ego = metric_cache.ego_state.rear_axle
    ex, ey, eh = ego.x, ego.y, ego.heading
    ax.scatter([ex], [ey], s=30, color=MAP_EGO, zorder=10)
    ax.arrow(
        ex,
        ey,
        float(np.cos(eh) * 4.0),
        float(np.sin(eh) * 4.0),
        width=0.22,
        head_width=1.1,
        color=MAP_EGO,
        length_includes_head=True,
        zorder=10,
    )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(ex - radius, ex + radius)
    ax.set_ylim(ey - radius, ey + radius)
    ax.grid(False)
    ax.tick_params(labelsize=8, colors="#3c4353")
    for spine in ax.spines.values():
        spine.set_color("#c4cad6")
        spine.set_linewidth(0.8)

    token = str(score_row["token"])
    score = float(score_row["score"]) if pd.notna(score_row["score"]) else np.nan
    hard = int(score_row["is_hard"])
    u_value = (
        float(score_row["hc_u_adapter_variance"])
        if "hc_u_adapter_variance" in score_row and pd.notna(score_row["hc_u_adapter_variance"])
        else np.nan
    )
    ax.set_title(f"token={token} | score={score:.4f} | hard={hard} | u={u_value:.4f}", fontsize=9, color=MAP_EGO)
    ax.legend(loc="upper right", fontsize=7, framealpha=0.92, facecolor="white", edgecolor="#d1d6e0")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def _batchify_feature_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    return {k: (v.unsqueeze(0) if torch.is_tensor(v) else v) for k, v in data.items()}


def _batchify_target_dict(data: Dict[str, Any], token_override: Optional[str] = None) -> Dict[str, Any]:
    batched: Dict[str, Any] = {}
    for k, v in data.items():
        if torch.is_tensor(v):
            batched[k] = v.unsqueeze(0)
        elif isinstance(v, (str, bytes)):
            batched[k] = [v]
        else:
            batched[k] = v
    if token_override is not None:
        batched["token"] = [token_override]
    return batched


def _resolve_cfg_path(cfg, key: str, override: Optional[Path]) -> Path:
    if override is not None:
        return override
    value = OmegaConf.select(cfg, key)
    if value is None:
        raise ValueError(f"Missing config value: {key}")
    resolved = OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value
    return Path(str(resolved))


def _build_model_context(args: argparse.Namespace, tokens: Iterable[str]) -> Optional[Dict[str, Any]]:
    if args.hydra_config is None:
        return None

    cfg = OmegaConf.load(args.hydra_config)
    with open_dict(cfg):
        if args.checkpoint_path is not None:
            cfg.agent.checkpoint_path = str(args.checkpoint_path)

    agent: AbstractAgent = instantiate(cfg.agent)
    agent.initialize()
    agent.eval()

    scene_filter = instantiate(cfg.train_test_split.scene_filter)
    token_list = list(tokens)
    scene_filter.tokens = token_list
    if args.ignore_config_log_names or args.navsim_log_path is not None:
        # When a training hydra config is reused for navtest/navtrain visualization,
        # its log_names can silently filter out valid selected tokens.
        scene_filter.log_names = None
    scene_loader = SceneLoader(
        sensor_blobs_path=_resolve_cfg_path(cfg, "sensor_blobs_path", args.sensor_blobs_path),
        data_path=_resolve_cfg_path(cfg, "navsim_log_path", args.navsim_log_path),
        scene_filter=scene_filter,
        sensor_config=agent.get_sensor_config(),
    )
    available_tokens = set(scene_loader.tokens)
    missing_tokens = [token for token in token_list if token not in available_tokens]
    if missing_tokens:
        print(
            "[warn] SceneLoader missing "
            f"{len(missing_tokens)}/{len(token_list)} selected tokens. "
            "Check --navsim-log-path, hydra train_test_split, and scene_filter.log_names. "
            f"examples={missing_tokens[:5]}"
        )

    return {
        "cfg": cfg,
        "agent": agent,
        "scene_loader": scene_loader,
        "feature_builders": list(agent.get_feature_builders()),
        "target_builders": list(agent.get_target_builders()),
    }


def _make_lidar_traj_rgb(
    lidar_map: np.ndarray,
    gt_trajectory: np.ndarray,
    pred_trajectory: np.ndarray,
    config,
    annotations: Optional[Annotations] = None,
    all_pred_trajectories: Optional[np.ndarray] = None,
) -> np.ndarray:
    gt_color, pred_color = (0, 255, 0), (255, 0, 0)
    point_size = 4
    height, width = lidar_map.shape[:2]

    def coords_to_pixel(coords: np.ndarray) -> np.ndarray:
        pixel_center = np.array([[height / 2.0, width / 2.0]], dtype=np.float32)
        coords_idcs = (coords / float(config.bev_pixel_size)) + pixel_center
        return coords_idcs.astype(np.int32)

    rgb_map = np.clip(lidar_map * 255.0, 0.0, 255.0).astype(np.uint8)
    rgb_map = rgb_map[..., None].repeat(3, axis=-1)

    def draw_traj_line(canvas: np.ndarray, traj: np.ndarray, color, thickness: int) -> None:
        if traj is None or len(traj) == 0:
            return
        traj_xy = np.asarray(traj, dtype=np.float32)[..., :2]
        pixels = coords_to_pixel(traj_xy)
        pixels = np.flip(pixels, axis=-1).reshape((-1, 1, 2))
        cv2.polylines(canvas, [pixels], isClosed=False, color=color, thickness=thickness)

    if all_pred_trajectories is not None and len(all_pred_trajectories) > 0:
        overlay = rgb_map.copy()
        pale_blue = (255, 180, 90)
        for traj in np.asarray(all_pred_trajectories, dtype=np.float32):
            draw_traj_line(overlay, traj, pale_blue, thickness=1)
        rgb_map = cv2.addWeighted(overlay, 0.35, rgb_map, 0.65, 0.0)

    for color, traj in ((gt_color, gt_trajectory), (pred_color, pred_trajectory)):
        if traj is None or len(traj) == 0:
            continue
        draw_traj_line(rgb_map, traj, color, thickness=2)
        traj_xy = np.asarray(traj, dtype=np.float32)[..., :2]
        trajectory_indices = coords_to_pixel(traj_xy)
        for x, y in trajectory_indices:
            if 0 <= x < height and 0 <= y < width:
                cv2.circle(rgb_map, (y, x), point_size, color, -1)

    if annotations is not None:
        for name_value, box_value in zip(annotations.names, annotations.boxes):
            x = float(box_value[BoundingBoxIndex.X])
            y = float(box_value[BoundingBoxIndex.Y])
            heading = float(box_value[BoundingBoxIndex.HEADING])
            box_length = float(box_value[BoundingBoxIndex.LENGTH])
            box_width = float(box_value[BoundingBoxIndex.WIDTH])
            box_height = float(box_value[BoundingBoxIndex.HEIGHT])
            agent_box = OrientedBox(StateSE2(x, y, heading), box_length, box_width, box_height)
            corners = np.asarray(agent_box.geometry.exterior.coords, dtype=np.float32)
            pixels = coords_to_pixel(corners[:, :2])
            pixels = np.flip(pixels, axis=-1).reshape((-1, 1, 2))
            agent_type = tracked_object_types.get(name_value, TrackedObjectType.VEHICLE)
            hex_color = AGENT_CONFIG[agent_type]["fill_color"]
            rgb = tuple(int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
            bgr = (rgb[2], rgb[1], rgb[0])
            cv2.polylines(rgb_map, [pixels], isClosed=True, color=bgr, thickness=2)

    return rgb_map[::-1, ::-1]


def _make_bev_traj_rgb(
    bev_rgb: np.ndarray,
    gt_trajectory: np.ndarray,
    pred_trajectory: np.ndarray,
    config,
    all_pred_trajectories: Optional[np.ndarray] = None,
) -> np.ndarray:
    gt_color, pred_color = (0, 255, 0), (255, 0, 0)
    point_size = 4
    height, width = bev_rgb.shape[:2]

    def coords_to_pixel(coords: np.ndarray) -> np.ndarray:
        pixel_center = np.array([[height / 2.0, width / 2.0]], dtype=np.float32)
        coords_idcs = (coords / float(config.bev_pixel_size)) + pixel_center
        return coords_idcs.astype(np.int32)

    def draw_traj_line(canvas: np.ndarray, traj: np.ndarray, color, thickness: int) -> None:
        if traj is None or len(traj) == 0:
            return
        traj_xy = np.asarray(traj, dtype=np.float32)[..., :2]
        pixels = coords_to_pixel(traj_xy)
        pixels = np.flip(pixels, axis=-1).reshape((-1, 1, 2))
        cv2.polylines(canvas, [pixels], isClosed=False, color=color, thickness=thickness)

    # semantic_map_to_rgb already flips both axes; unflip first so trajectory drawing
    # uses the same raw BEV coordinate convention as lidar / metric-cache rendering.
    rgb_map = bev_rgb[::-1, ::-1].copy()

    if all_pred_trajectories is not None and len(all_pred_trajectories) > 0:
        overlay = rgb_map.copy()
        pale_blue = (255, 180, 90)
        for traj in np.asarray(all_pred_trajectories, dtype=np.float32):
            draw_traj_line(overlay, traj, pale_blue, thickness=1)
        rgb_map = cv2.addWeighted(overlay, 0.35, rgb_map, 0.65, 0.0)

    for color, traj in ((gt_color, gt_trajectory), (pred_color, pred_trajectory)):
        if traj is None or len(traj) == 0:
            continue
        draw_traj_line(rgb_map, traj, color, thickness=2)
        traj_xy = np.asarray(traj, dtype=np.float32)[..., :2]
        trajectory_indices = coords_to_pixel(traj_xy)
        for x, y in trajectory_indices:
            if 0 <= x < height and 0 <= y < width:
                cv2.circle(rgb_map, (y, x), point_size, color, -1)

    return rgb_map[::-1, ::-1]


def _normalize_trajectory_array(traj: np.ndarray) -> np.ndarray:
    traj_arr = np.asarray(traj, dtype=np.float32)
    while traj_arr.ndim > 2 and traj_arr.shape[0] == 1:
        traj_arr = traj_arr[0]
    if traj_arr.ndim != 2:
        raise ValueError(f"Expected trajectory with shape [T, D], got {traj_arr.shape}")
    if traj_arr.shape[-1] < 2:
        raise ValueError(f"Trajectory last dim must be >=2, got {traj_arr.shape}")
    return traj_arr


def _normalize_proposals_array(traj: np.ndarray) -> np.ndarray:
    traj_arr = np.asarray(traj, dtype=np.float32)
    while traj_arr.ndim > 3 and traj_arr.shape[0] == 1:
        traj_arr = traj_arr[0]
    if traj_arr.ndim != 3:
        raise ValueError(f"Expected proposal trajectories with shape [M, T, D], got {traj_arr.shape}")
    if traj_arr.shape[-1] < 2:
        raise ValueError(f"Proposal trajectory last dim must be >=2, got {traj_arr.shape}")
    return traj_arr


def _extract_all_pred_trajectories(predictions: Dict[str, Any]) -> Optional[np.ndarray]:
    for key in ("poses_reg", "poses_reg_specialist", "trajectory_modes", "pred_trajectories"):
        value = predictions.get(key)
        if value is None:
            continue
        if torch.is_tensor(value):
            value = value.detach().cpu().numpy()
        try:
            return _normalize_proposals_array(value)
        except Exception:
            continue
    return None


def _normalize_semantic_map_array(semantic_map: np.ndarray) -> np.ndarray:
    sem = np.asarray(semantic_map)
    while sem.ndim > 2 and sem.shape[0] == 1:
        sem = sem[0]
    if sem.ndim == 2:
        return sem
    if sem.ndim == 3:
        return np.argmax(sem, axis=0)
    raise ValueError(f"Expected semantic map with shape [H, W] or [C, H, W], got {sem.shape}")


def _normalize_attention_map_array(attn_map: np.ndarray) -> np.ndarray:
    attn = np.asarray(attn_map, dtype=np.float32)
    while attn.ndim > 3 and attn.shape[0] == 1:
        attn = attn[0]
    if attn.ndim == 2:
        return attn
    if attn.ndim == 3:
        raise ValueError(
            f"Attention map still has mode dimension; expected selected [H, W], got {attn.shape}"
        )
    raise ValueError(f"Expected attention map with shape [H, W], got {attn.shape}")


def _select_mode_index_from_predictions(predictions: Dict[str, Any]) -> int:
    if "pdm_score" in predictions and torch.is_tensor(predictions["pdm_score"]):
        pdm_score = predictions["pdm_score"].detach().cpu()
        if pdm_score.dim() == 2 and pdm_score.shape[0] >= 1:
            return int(pdm_score[0].argmax().item())
    if "poses_cls" in predictions and torch.is_tensor(predictions["poses_cls"]):
        poses_cls = predictions["poses_cls"].detach().cpu()
        if poses_cls.dim() == 2 and poses_cls.shape[0] >= 1:
            return int(poses_cls[0].argmax().item())
    return 0


def _transform_local_to_global(traj: np.ndarray, ego_pose_global: np.ndarray) -> np.ndarray:
    traj_arr = np.asarray(traj, dtype=np.float32)
    ego_pose = np.asarray(ego_pose_global, dtype=np.float32).reshape(3)
    transformed = traj_arr.copy()

    cos_h = float(np.cos(ego_pose[2]))
    sin_h = float(np.sin(ego_pose[2]))
    local_x = traj_arr[..., 0]
    local_y = traj_arr[..., 1]

    transformed[..., 0] = ego_pose[0] + cos_h * local_x - sin_h * local_y
    transformed[..., 1] = ego_pose[1] + sin_h * local_x + cos_h * local_y
    if transformed.shape[-1] >= 3:
        transformed[..., 2] = traj_arr[..., 2] + ego_pose[2]
    return transformed


def _local_box_to_global_box(box_value: np.ndarray, ego_pose_global: np.ndarray) -> OrientedBox:
    box = np.asarray(box_value, dtype=np.float32)
    ego_pose = np.asarray(ego_pose_global, dtype=np.float32).reshape(3)
    local_pose = np.zeros(3, dtype=np.float32)
    local_pose[:3] = box[:3]
    global_pose = _transform_local_to_global(local_pose[None, :], ego_pose)[0]
    return OrientedBox(
        StateSE2(float(global_pose[0]), float(global_pose[1]), float(global_pose[2])),
        float(box[BoundingBoxIndex.LENGTH]),
        float(box[BoundingBoxIndex.WIDTH]),
        float(box[BoundingBoxIndex.HEIGHT]),
    )


def _draw_oriented_box_xy(
    ax: plt.Axes,
    box: OrientedBox,
    edge_color: str,
    face_color: str,
    alpha: float,
    linewidth: float,
    label: Optional[str] = None,
    zorder: int = 7,
) -> None:
    corners = box.all_corners()
    xy = np.asarray([[corner.x, corner.y] for corner in corners] + [[corners[0].x, corners[0].y]])
    ax.fill(
        xy[:, 0],
        xy[:, 1],
        facecolor=face_color,
        edgecolor=edge_color,
        alpha=alpha,
        linewidth=linewidth,
        label=label,
        zorder=zorder,
    )
    ax.plot(
        xy[:, 0],
        xy[:, 1],
        color=edge_color,
        alpha=min(1.0, alpha + 0.2),
        linewidth=linewidth,
        zorder=zorder + 0.1,
    )


def _draw_attention_heatmap_on_vector_map(
    ax: plt.Axes,
    attention_map: np.ndarray,
    ego_pose_global: np.ndarray,
    config,
    alpha: float,
    min_value: float,
    cmap_min: float,
    alpha_gamma: float,
) -> None:
    attn = np.asarray(attention_map, dtype=np.float32)
    if attn.ndim != 2 or attn.size == 0:
        return
    attn = attn - float(np.nanmin(attn))
    vmax = float(np.nanmax(attn))
    if vmax <= 0.0:
        return
    attn = np.clip(attn / vmax, 0.0, 1.0)
    min_value = float(np.clip(min_value, 0.0, 1.0))

    # NAVSIM BEV semantic map uses ego at row=0 and col=width/2:
    # local x extends forward from ego, while local y is centered left/right.
    full_height = int(getattr(config, "bev_pixel_height", attn.shape[0]))
    full_width = int(getattr(config, "bev_pixel_width", attn.shape[1]))
    pixel_size = float(getattr(config, "bev_pixel_size", 0.25))
    target_height = max(full_height, attn.shape[0])
    target_width = max(full_width, attn.shape[1])
    if attn.shape != (target_height, target_width):
        attn = cv2.resize(
            attn,
            (target_width, target_height),
            interpolation=cv2.INTER_CUBIC,
        )
        attn = np.clip(attn, 0.0, 1.0)
        attn = cv2.GaussianBlur(attn, (0, 0), sigmaX=1.2, sigmaY=1.2)
        attn = np.clip(attn, 0.0, 1.0)

    height, width = attn.shape
    x_min = 0.0
    x_max = height * pixel_size
    y_min = -0.5 * width * pixel_size
    y_max = 0.5 * width * pixel_size
    row_edges = np.linspace(x_min, x_max, height + 1, dtype=np.float32)
    col_edges = np.linspace(y_min, y_max, width + 1, dtype=np.float32)
    row_grid, col_grid = np.meshgrid(row_edges, col_edges, indexing="ij")
    local_edges = np.stack([row_grid, col_grid], axis=-1)
    global_edges = _transform_local_to_global(local_edges, ego_pose_global)
    alpha_max = float(np.clip(alpha, 0.0, 1.0))
    alpha_gamma = max(float(alpha_gamma), 1e-3)
    alpha_map = np.clip((attn - min_value) / max(1.0 - min_value, 1e-6), 0.0, 1.0)
    alpha_map = np.power(alpha_map, alpha_gamma)
    attn_masked = np.ma.masked_where(alpha_map <= 0.0, attn)
    cmap_min = float(np.clip(cmap_min, 0.0, 0.95))
    base_cmap = plt.get_cmap("magma")
    heat_colors = base_cmap(np.linspace(cmap_min, 1.0, 256))
    heat_colors[:, 3] = np.linspace(0.0, alpha_max, 256)
    heat_cmap = matplotlib.colors.ListedColormap(heat_colors, name="magma_alpha_ramp")
    heat_cmap.set_bad((0.0, 0.0, 0.0, 0.0))

    ax.pcolormesh(
        global_edges[..., 0],
        global_edges[..., 1],
        np.ma.masked_where(alpha_map <= 0.0, alpha_map),
        cmap=heat_cmap,
        vmin=0.0,
        vmax=1.0,
        shading="auto",
        zorder=6,
        linewidth=0.0,
        rasterized=True,
    )


def _draw_vector_prediction_map(
    metric_cache,
    ego_pose_global: np.ndarray,
    gt_trajectory: np.ndarray,
    pred_trajectory: np.ndarray,
    all_pred_trajectories: Optional[np.ndarray],
    score_row: pd.Series,
    save_path: Path,
    radius: float,
    dpi: int,
    annotations: Optional[Annotations] = None,
    attention_map: Optional[np.ndarray] = None,
    config: Optional[Any] = None,
    attention_alpha: float = 0.38,
    attention_min: float = 0.12,
    attention_cmap_min: float = 0.25,
    attention_alpha_gamma: float = 0.8,
) -> None:
    route_lane_ids = set(metric_cache.route_lane_ids)
    drivable_map = metric_cache.drivable_area_map

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    fig.patch.set_facecolor(MAP_BASE_FACE)
    ax.set_facecolor(MAP_PANEL_FACE)

    for token, layer, geom in zip(
        drivable_map.tokens,
        drivable_map.map_types,
        drivable_map._geometries,
    ):
        layer_name = getattr(layer, "name", str(layer))
        if token in route_lane_ids:
            face_color = MAP_ROUTE_FILL
            edge_color = MAP_ROUTE_EDGE
            alpha = 0.55
            lw = 0.65
        elif "LANE_CONNECTOR" in layer_name:
            face_color = "#e4e8f1"
            edge_color = "#b4bdcc"
            alpha = 0.45
            lw = 0.35
        else:
            face_color = MAP_LANE_FILL
            edge_color = MAP_LANE_EDGE
            alpha = 0.40
            lw = 0.30

        for poly in _iter_polygons(geom):
            x, y = poly.exterior.xy
            ax.fill(x, y, facecolor=face_color, edgecolor=edge_color, alpha=alpha, linewidth=lw)

    centerline = getattr(metric_cache, "centerline", None)
    centerline_linestring = getattr(centerline, "linestring", None)
    if centerline_linestring is not None:
        x, y = centerline_linestring.xy
        ax.plot(
            x,
            y,
            color=MAP_CENTERLINE,
            linewidth=1.4,
            alpha=0.95,
            linestyle="--",
            label="Route Centerline",
            zorder=5,
        )

    if attention_map is not None and config is not None:
        _draw_attention_heatmap_on_vector_map(
            ax=ax,
            attention_map=attention_map,
            ego_pose_global=ego_pose_global,
            config=config,
            alpha=attention_alpha,
            min_value=attention_min,
            cmap_min=attention_cmap_min,
            alpha_gamma=attention_alpha_gamma,
        )

    gt_global = _transform_local_to_global(gt_trajectory, ego_pose_global)
    pred_global = _transform_local_to_global(pred_trajectory, ego_pose_global)
    all_global = None
    if all_pred_trajectories is not None and len(all_pred_trajectories) > 0:
        all_global = _transform_local_to_global(all_pred_trajectories, ego_pose_global)

    if all_global is not None:
        drew_label = False
        for traj in np.asarray(all_global, dtype=np.float32):
            ax.plot(
                traj[:, 0],
                traj[:, 1],
                color=MAP_ALL_MODES,
                linewidth=1.35,
                alpha=0.55,
                solid_capstyle="round",
                label=f"All Modes ({len(all_global)})" if not drew_label else None,
                zorder=7,
            )
            drew_label = True

    if annotations is not None:
        drew_agents = False
        for box_value in annotations.boxes:
            try:
                agent_box = _local_box_to_global_box(box_value, ego_pose_global)
            except Exception:
                continue
            _draw_oriented_box_xy(
                ax,
                agent_box,
                edge_color=MAP_AGENT_BOX,
                face_color=MAP_AGENT_BOX,
                alpha=0.32,
                linewidth=0.75,
                label="Other Agents" if not drew_agents else None,
                zorder=8,
            )
            drew_agents = True

    ax.plot(
        gt_global[:, 0],
        gt_global[:, 1],
        color=MAP_GT,
        linewidth=2.3,
        alpha=0.98,
        marker="o",
        markersize=3.2,
        markerfacecolor=MAP_GT,
        markeredgewidth=0.0,
        label="GT",
        zorder=10,
    )
    ax.plot(
        pred_global[:, 0],
        pred_global[:, 1],
        color=MAP_SELECTED,
        linewidth=2.5,
        alpha=0.98,
        marker="o",
        markersize=3.2,
        markerfacecolor=MAP_SELECTED,
        markeredgecolor=MAP_EGO,
        markeredgewidth=0.25,
        label="Selected Mode",
        zorder=11,
    )

    ex, ey, eh = map(float, np.asarray(ego_pose_global, dtype=np.float32))
    ego_footprint = CarFootprint.build_from_rear_axle(
        rear_axle_pose=StateSE2(ex, ey, eh),
        vehicle_parameters=get_pacifica_parameters(),
    )
    _draw_oriented_box_xy(
        ax,
        ego_footprint.oriented_box,
        edge_color=MAP_EGO_BOX,
        face_color=MAP_EGO_BOX,
        alpha=0.45,
        linewidth=1.0,
        label="Ego",
        zorder=12,
    )
    ax.scatter([ex], [ey], s=24, color=MAP_EGO, zorder=13)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(ex - radius, ex + radius)
    ax.set_ylim(ey - radius, ey + radius)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    for spine in ax.spines.values():
        spine.set_visible(False)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0.02, facecolor=fig.get_facecolor())
    plt.close(fig)


def _make_attention_rgb(attn_map: np.ndarray) -> np.ndarray:
    attn = np.asarray(attn_map, dtype=np.float32)
    attn = attn - float(np.nanmin(attn))
    vmax = float(np.nanmax(attn))
    if vmax > 0.0:
        attn = attn / vmax
    attn = np.clip(attn, 0.0, 1.0)
    rgba = plt.get_cmap("magma")(attn)
    rgb = (rgba[..., :3] * 255.0).astype(np.uint8)
    # Match semantic_map_to_rgb / lidar panel orientation for visual alignment.
    return rgb[::-1, ::-1]


def _draw_model_prediction_panel(
    model_ctx: Dict[str, Any],
    token: str,
    score_row: pd.Series,
    save_path: Path,
    dpi: int,
    metric_cache: Optional[Any] = None,
    radius: float = 80.0,
    vector_map_save_path: Optional[Path] = None,
    vector_attention: bool = True,
    vector_attention_alpha: float = 0.38,
    vector_attention_min: float = 0.12,
    vector_attention_cmap_min: float = 0.25,
    vector_attention_alpha_gamma: float = 0.8,
) -> None:
    scene_loader: SceneLoader = model_ctx["scene_loader"]
    if token not in scene_loader.tokens:
        raise KeyError(f"Token not found in SceneLoader: {token}")

    scene = scene_loader.get_scene_from_token(token)
    agent_input = scene.get_agent_input()
    current_frame = scene.frames[scene.scene_metadata.num_history_frames - 1]
    ego_pose_global = np.asarray(current_frame.ego_status.ego_pose, dtype=np.float32)

    features: Dict[str, Any] = {}
    for builder in model_ctx["feature_builders"]:
        features.update(builder.compute_features(agent_input))

    targets: Dict[str, Any] = {}
    for builder in model_ctx["target_builders"]:
        targets.update(builder.compute_targets(scene))

    batched_features = _batchify_feature_dict(features)
    batched_targets = _batchify_target_dict(targets, token_override=token)

    agent: AbstractAgent = model_ctx["agent"]
    with torch.no_grad():
        try:
            predictions = agent.forward(batched_features, batched_targets)
        except TypeError:
            predictions = agent.forward(batched_features)
        hook = getattr(agent, "_on_inference_predictions", None)
        if callable(hook):
            try:
                hook(predictions)
            except Exception:
                pass

    if "bev_semantic_map" not in predictions:
        raise ValueError("Predictions missing bev_semantic_map")
    if "trajectory" not in predictions:
        raise ValueError("Predictions missing trajectory")
    if "lidar_feature" not in features:
        raise ValueError("Features missing lidar_feature")
    if "trajectory" not in targets:
        raise ValueError("Targets missing trajectory")
    if "bev_semantic_map" not in targets:
        raise ValueError("Targets missing bev_semantic_map")

    pred_bev_np = _normalize_semantic_map_array(
        predictions["bev_semantic_map"].detach().cpu().numpy()
    )
    gt_bev_np = _normalize_semantic_map_array(
        targets["bev_semantic_map"].detach().cpu().numpy()
    )

    pred_trajectory = _normalize_trajectory_array(
        predictions["trajectory"].detach().cpu().numpy()
    )
    gt_trajectory = _normalize_trajectory_array(targets["trajectory"].detach().cpu().numpy())
    all_pred_trajectories = _extract_all_pred_trajectories(predictions)
    attention_rgb = None
    selected_attention_np = None
    if "mode_bev_attention_map" in predictions and torch.is_tensor(predictions["mode_bev_attention_map"]):
        try:
            mode_idx = _select_mode_index_from_predictions(predictions)
            attn_tensor = predictions["mode_bev_attention_map"].detach().cpu()
            if attn_tensor.dim() == 4 and attn_tensor.shape[0] >= 1:
                mode_idx = max(0, min(mode_idx, int(attn_tensor.shape[1]) - 1))
                attn_np = _normalize_attention_map_array(attn_tensor[0, mode_idx].numpy())
                selected_attention_np = attn_np
                attention_rgb = _make_attention_rgb(attn_np)
        except Exception:
            attention_rgb = None
            selected_attention_np = None
    lidar_map = features["lidar_feature"].detach().cpu().numpy().squeeze(0)

    cfg = model_ctx["agent"]._config
    gt_bev_rgb = _make_bev_traj_rgb(
        bev_rgb=semantic_map_to_rgb(gt_bev_np, cfg),
        gt_trajectory=gt_trajectory,
        pred_trajectory=pred_trajectory,
        config=cfg,
        all_pred_trajectories=all_pred_trajectories,
    )
    pred_bev_rgb = semantic_map_to_rgb(pred_bev_np, cfg)
    lidar_rgb = _make_lidar_traj_rgb(
        lidar_map=lidar_map,
        gt_trajectory=gt_trajectory,
        pred_trajectory=pred_trajectory,
        config=cfg,
        annotations=current_frame.annotations,
        all_pred_trajectories=all_pred_trajectories,
    )

    if vector_map_save_path is not None and metric_cache is not None:
        _draw_vector_prediction_map(
            metric_cache=metric_cache,
            ego_pose_global=ego_pose_global,
            gt_trajectory=gt_trajectory,
            pred_trajectory=pred_trajectory,
            all_pred_trajectories=all_pred_trajectories,
            score_row=score_row,
            save_path=vector_map_save_path,
            radius=float(radius),
            dpi=int(dpi),
            annotations=current_frame.annotations,
            attention_map=selected_attention_np if vector_attention else None,
            config=cfg,
            attention_alpha=float(vector_attention_alpha),
            attention_min=float(vector_attention_min),
            attention_cmap_min=float(vector_attention_cmap_min),
            attention_alpha_gamma=float(vector_attention_alpha_gamma),
        )

    panels = [
        ("GT BEV", gt_bev_rgb),
        ("Pred BEV", pred_bev_rgb),
    ]
    if attention_rgb is not None:
        panels.append(("Selected Attention", attention_rgb))
    panels.append(("Lidar + GT/Pred Traj", lidar_rgb))

    fig, axes = plt.subplots(1, len(panels), figsize=(4.5 * len(panels), 4.5))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    for ax, (title, image) in zip(axes, panels):
        ax.imshow(image)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    token_str = str(score_row["token"])
    score = float(score_row["score"]) if pd.notna(score_row["score"]) else np.nan
    ax_title = f"token={token_str} | score={score:.4f}"
    fig.suptitle(ax_title, fontsize=11)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = _parse_args()

    score_csv = args.score_csv
    metric_cache_path = args.metric_cache_path
    out_dir = args.out_dir or (score_csv.parent / "hardcase_maps")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _load_score_df(score_csv)
    df["is_hard"] = _hard_mask(df, args.hard_threshold).astype(int)

    sort_cols = ["score"]
    ascending = [True]
    if "hc_u_adapter_variance" in df.columns:
        sort_cols.append("hc_u_adapter_variance")
        ascending.append(False)

    hard_df = df[df["is_hard"] == 1].copy().sort_values(sort_cols, ascending=ascending)
    hard_df.to_csv(out_dir / "hard_scenes_all.csv", index=False)

    score_threshold = float(args.score_threshold)
    if args.score_compare == "ge":
        selected = df[df["score"] >= score_threshold].copy()
    else:
        selected = df[df["score"] <= score_threshold].copy()
    if args.only_hard:
        selected = selected[selected["is_hard"] == 1]
    selected = selected.sort_values(sort_cols, ascending=ascending).head(int(args.max_scenes))
    selected_tag = f"score_{args.score_compare}_{str(args.score_threshold).replace('.', 'p')}"
    selected.to_csv(
        out_dir / f"selected_{selected_tag}.csv",
        index=False,
    )

    model_ctx = _build_model_context(args, selected["token"].astype(str).tolist())

    # Robust default: recursively index all metric_cache.pkl files.
    token_to_path = _index_metric_cache_recursively(metric_cache_path)
    if len(token_to_path) == 0:
        token_to_path = _load_metric_cache_paths_from_metadata(
            metric_cache_path=metric_cache_path,
            rewrite_old=args.path_rewrite_old,
            rewrite_new=args.path_rewrite_new,
        )
    print(f"[info] indexed metric cache files: {len(token_to_path)}")

    rendered = 0
    model_rendered = 0
    skipped = 0
    model_skipped = 0
    miss_examples = []
    fail_rows = []
    model_fail_rows = []
    scene_cache: Dict[str, Any] = {}
    for rank, (_, row) in enumerate(selected.iterrows()):
        token = str(row["token"])
        cache_file = token_to_path.get(token, None)
        if cache_file is None or not cache_file.exists():
            skipped += 1
            if len(miss_examples) < 20:
                miss_examples.append({"token": token, "cache_file": str(cache_file)})
            continue
        try:
            with lzma.open(cache_file, "rb") as f:
                metric_cache = pickle.load(f)
            score_tag = float(row["score"]) if pd.notna(row["score"]) else np.nan
            file_name = f"{rank:04d}_{token}_score_{score_tag:.4f}.png"
            save_path = out_dir / "maps" / file_name
            annotations = None
            if model_ctx is not None:
                if token not in scene_cache:
                    try:
                        scene_cache[token] = model_ctx["scene_loader"].get_scene_from_token(token)
                    except Exception:
                        scene_cache[token] = None
                scene_obj = scene_cache.get(token)
                if scene_obj is not None:
                    annotations = scene_obj.frames[
                        scene_obj.scene_metadata.num_history_frames - 1
                    ].annotations
            _draw_metric_cache_map(
                metric_cache=metric_cache,
                score_row=row,
                save_path=save_path,
                radius=float(args.radius),
                dpi=int(args.dpi),
                annotations=annotations,
            )
            rendered += 1
        except Exception as e:
            skipped += 1
            fail_rows.append(
                {
                    "token": token,
                    "cache_file": str(cache_file),
                    "error": repr(e),
                    "traceback": traceback.format_exc(limit=2),
                }
            )
            if len(fail_rows) <= 10:
                print(f"[warn] render failed token={token} file={cache_file}: {repr(e)}")

        if model_ctx is not None:
            try:
                score_tag = float(row["score"]) if pd.notna(row["score"]) else np.nan
                file_name = f"{rank:04d}_{token}_score_{score_tag:.4f}.png"
                save_path = out_dir / "model_viz" / file_name
                vector_map_save_path = out_dir / "vector_map_viz" / file_name
                _draw_model_prediction_panel(
                    model_ctx=model_ctx,
                    token=token,
                    score_row=row,
                    save_path=save_path,
                    dpi=int(args.dpi),
                    metric_cache=metric_cache,
                    radius=float(args.radius),
                    vector_map_save_path=vector_map_save_path,
                    vector_attention=not bool(args.no_vector_attention),
                    vector_attention_alpha=float(args.vector_attention_alpha),
                    vector_attention_min=float(args.vector_attention_min),
                    vector_attention_cmap_min=float(args.vector_attention_cmap_min),
                    vector_attention_alpha_gamma=float(args.vector_attention_alpha_gamma),
                )
                model_rendered += 1
            except Exception as e:
                model_skipped += 1
                model_fail_rows.append(
                    {
                        "token": token,
                        "error": repr(e),
                        "traceback": traceback.format_exc(limit=2),
                    }
                )
                if len(model_fail_rows) <= 10:
                    print(f"[warn] model render failed token={token}: {repr(e)}")

    summary = {
        "score_csv": str(score_csv),
        "metric_cache_path": str(metric_cache_path),
        "num_rows": int(len(df)),
        "num_hard": int((df["is_hard"] == 1).sum()),
        "num_selected": int(len(selected)),
        "rendered_maps": int(rendered),
        "skipped_maps": int(skipped),
        "model_rendered": int(model_rendered),
        "model_skipped": int(model_skipped),
        "missing_cache_examples": int(len(miss_examples)),
        "render_failures": int(len(fail_rows)),
        "model_render_failures": int(len(model_fail_rows)),
    }
    pd.DataFrame([summary]).to_csv(out_dir / "summary.csv", index=False)
    if len(miss_examples) > 0:
        pd.DataFrame(miss_examples).to_csv(out_dir / "missing_cache_examples.csv", index=False)
    if len(fail_rows) > 0:
        pd.DataFrame(fail_rows).to_csv(out_dir / "render_failures.csv", index=False)
    if len(model_fail_rows) > 0:
        pd.DataFrame(model_fail_rows).to_csv(out_dir / "model_render_failures.csv", index=False)
    print(pd.Series(summary).to_string())


if __name__ == "__main__":
    main()
