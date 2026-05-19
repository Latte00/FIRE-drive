import json
import logging
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import hydra
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra.utils import instantiate
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from omegaconf import DictConfig, ListConfig, OmegaConf
from PIL import Image
from shapely import affinity
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Polygon
from tqdm import tqdm

from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.common.maps.abstract_map import SemanticMapLayer
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataloader import MetricCacheLoader, SceneLoader
from navsim.common.dataclasses import Camera, Scene, SceneFilter, Trajectory
from navsim.common.enums import BoundingBoxIndex, LidarIndex
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/training"
CONFIG_NAME = "default_training"


def _safe_name(value: str, limit: int = 140) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return (safe or "sample")[:limit]


def _cfg_to_dict(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, DictConfig):
        return OmegaConf.to_container(value, resolve=True)  # type: ignore[return-value]
    return dict(value)


def _list_from_cfg(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (ListConfig, list, tuple)):
        return [str(v) for v in value]
    if isinstance(value, str):
        return [value] if value else []
    return [str(value)]


def _get_split_logs(cfg: DictConfig, split_name: str) -> Optional[List[str]]:
    key = f"{split_name}_logs"
    if hasattr(cfg, key):
        return _list_from_cfg(getattr(cfg, key))
    return None


def _to_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _to_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def _move_to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device=device, non_blocking=True)
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    return value


def _batch_features(features: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    batched: Dict[str, Any] = {}
    for key, value in features.items():
        if torch.is_tensor(value):
            batched[key] = value.unsqueeze(0).to(device=device, non_blocking=True)
        else:
            batched[key] = _move_to_device(value, device)
    return batched


def _nested_attr(obj: Any, path: str) -> Any:
    current = obj
    for name in path.split("."):
        if not hasattr(current, name):
            return None
        current = getattr(current, name)
    return current


def _maybe_attach_model_arrays(agent: AbstractAgent, predictions: Dict[str, Any]) -> Dict[str, Any]:
    if any(key in predictions for key in ("plan_anchor", "anchor", "anchor_mu")):
        return predictions

    for attr_path in (
        "_transfuser_model._trajectory_head.plan_anchor",
        "_trajectory_head.plan_anchor",
        "plan_anchor",
    ):
        anchor = _nested_attr(agent, attr_path)
        if anchor is None:
            continue
        if torch.is_tensor(anchor):
            anchor_tensor = anchor.detach()
            if anchor_tensor.dim() == 3:
                anchor_tensor = anchor_tensor.unsqueeze(0)
            predictions = dict(predictions)
            predictions["plan_anchor"] = anchor_tensor
            return predictions
    return predictions


def _compute_predictions(agent: AbstractAgent, scene: Scene, device: torch.device) -> Dict[str, Any]:
    agent_input = scene.get_agent_input()
    features: Dict[str, Any] = {}
    for builder in agent.get_feature_builders():
        features.update(builder.compute_features(agent_input))
    features = _batch_features(features, device)
    agent.eval()
    with torch.no_grad():
        predictions = agent.forward(features)
    return _maybe_attach_model_arrays(agent, predictions)


def _get_cfg_path(cfg: DictConfig, path: str) -> Any:
    current: Any = cfg
    for name in path.split("."):
        if current is None or not hasattr(current, name):
            return None
        current = getattr(current, name)
    return current


def _metric_cache_path_from_cfg(cfg: DictConfig, vis: Dict[str, Any]) -> Optional[str]:
    for value in (
        vis.get("metric_cache_path"),
        _get_cfg_path(cfg, "metric_cache_path"),
        _get_cfg_path(cfg, "agent.config.pdm_metric_cache_path"),
    ):
        if value:
            return str(value)
    return None


def _build_pdm_context(cfg: DictConfig, vis: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not bool(vis.get("pdm_enable", True)):
        return None
    metric_cache_path = _metric_cache_path_from_cfg(cfg, vis)
    if not metric_cache_path:
        return None

    proposal_sampling = TrajectorySampling(
        num_poses=int(vis.get("pdm_num_poses", 40)),
        interval_length=float(vis.get("pdm_interval_length", 0.1)),
    )
    return {
        "metric_cache_path": metric_cache_path,
        "metric_cache_loader": MetricCacheLoader(Path(metric_cache_path)),
        "simulator": PDMSimulator(proposal_sampling),
        "scorer": PDMScorer(proposal_sampling),
    }


def _trajectory_sampling_from_array(traj: np.ndarray, vis: Dict[str, Any]) -> TrajectorySampling:
    interval = float(vis.get("trajectory_interval_length", 0.5))
    return TrajectorySampling(num_poses=int(traj.shape[0]), interval_length=interval)


def _compute_pdm_for_selected(
    token: str,
    predictions: Dict[str, Any],
    pdm_context: Optional[Dict[str, Any]],
    vis: Dict[str, Any],
) -> Dict[str, Any]:
    if pdm_context is None:
        return {"pdm_valid": None}
    loader: MetricCacheLoader = pdm_context["metric_cache_loader"]
    if token not in loader.metric_cache_paths:
        return {
            "pdm_valid": False,
            "pdm_error": "missing_metric_cache",
            "pdm_metric_cache_path": pdm_context["metric_cache_path"],
        }
    if "trajectory" not in predictions:
        return {"pdm_valid": False, "pdm_error": "missing_prediction_trajectory"}

    try:
        traj = _normalize_trajectory(predictions["trajectory"]).astype(np.float32)
        result = pdm_score(
            metric_cache=loader.get_from_token(token),
            model_trajectory=Trajectory(traj, _trajectory_sampling_from_array(traj, vis)),
            future_sampling=pdm_context["simulator"].proposal_sampling,
            simulator=pdm_context["simulator"],
            scorer=pdm_context["scorer"],
        )
        pdm_dict = _to_jsonable(asdict(result))
        return {
            "pdm_valid": True,
            "pdm_score": pdm_dict.get("score"),
            "pdm": pdm_dict,
            "pdm_metric_cache_path": pdm_context["metric_cache_path"],
        }
    except Exception as exc:
        return {
            "pdm_valid": False,
            "pdm_error": str(exc),
            "pdm_metric_cache_path": pdm_context["metric_cache_path"],
        }


def _squeeze_batch(value: Any) -> np.ndarray:
    arr = _to_numpy(value).astype(np.float32)
    while arr.ndim > 0 and arr.shape[0] == 1:
        arr = arr[0]
    return arr


def _normalize_trajectory(value: Any) -> np.ndarray:
    arr = _squeeze_batch(value)
    if arr.ndim != 2 or arr.shape[-1] < 2:
        raise ValueError(f"Expected trajectory [T, D>=2], got {arr.shape}")
    return arr[:, : min(arr.shape[-1], 3)]


def _normalize_modes(value: Any) -> np.ndarray:
    arr = _squeeze_batch(value)
    if arr.ndim == 2 and arr.shape[-1] >= 2:
        arr = arr[None]
    if arr.ndim != 3 or arr.shape[-1] < 2:
        raise ValueError(f"Expected modes [M, T, D>=2], got {arr.shape}")
    return arr[..., :2]


def _first_prediction_array(predictions: Dict[str, Any], names: Sequence[str]) -> Tuple[Optional[str], Optional[np.ndarray]]:
    for name in names:
        if name in predictions:
            return name, _to_numpy(predictions[name])
    for key, value in predictions.items():
        key_l = key.lower()
        if any(key_l == name.lower() or key_l.endswith("." + name.lower()) for name in names):
            return key, _to_numpy(value)
    return None, None


def _geometry_local_coords(geometry: Any, origin: StateSE2) -> Any:
    a = np.cos(origin.heading)
    b = np.sin(origin.heading)
    d = -np.sin(origin.heading)
    e = np.cos(origin.heading)
    translated = affinity.affine_transform(geometry, [1, 0, 0, 1, -origin.x, -origin.y])
    return affinity.affine_transform(translated, [a, b, d, e, 0, 0])


def _iter_polygons(geometry: Any) -> Iterable[Polygon]:
    if isinstance(geometry, Polygon):
        yield geometry
    elif isinstance(geometry, MultiPolygon):
        yield from geometry.geoms
    elif hasattr(geometry, "geoms"):
        for item in geometry.geoms:
            if isinstance(item, Polygon):
                yield item


def _iter_lines(geometry: Any) -> Iterable[LineString]:
    if isinstance(geometry, LineString):
        yield geometry
    elif isinstance(geometry, MultiLineString):
        yield from geometry.geoms
    elif hasattr(geometry, "geoms"):
        for item in geometry.geoms:
            if isinstance(item, LineString):
                yield item


def _plot_polygon(ax: plt.Axes, polygon: Polygon, face: str, edge: str, alpha: float, lw: float, zorder: int) -> None:
    xs, ys = polygon.exterior.xy
    ax.fill(ys, xs, facecolor=face, edgecolor=edge, alpha=alpha, linewidth=lw, zorder=zorder)
    for interior in polygon.interiors:
        ix, iy = interior.xy
        ax.fill(iy, ix, facecolor="#f8fafc", edgecolor=edge, alpha=1.0, linewidth=lw, zorder=zorder + 1)


def _draw_vector_map(ax: plt.Axes, scene: Scene, frame_idx: int, vis: Dict[str, Any]) -> None:
    origin = StateSE2(*scene.frames[frame_idx].ego_status.ego_pose)
    radius = float(vis.get("map_radius", 80.0))
    polygon_layers = [
        SemanticMapLayer.LANE,
        SemanticMapLayer.LANE_CONNECTOR,
        SemanticMapLayer.INTERSECTION,
        SemanticMapLayer.CROSSWALK,
        SemanticMapLayer.WALKWAYS,
        SemanticMapLayer.CARPARK_AREA,
        SemanticMapLayer.STOP_LINE,
    ]
    line_layers = [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]
    layers = list(dict.fromkeys(polygon_layers + line_layers))
    map_objects = scene.map_api.get_proximal_map_objects(point=origin.point, radius=radius, layers=layers)

    polygon_style = {
        SemanticMapLayer.LANE: ("#e5e7eb", "#cbd5e1", 0.88, 0.45, 1),
        SemanticMapLayer.LANE_CONNECTOR: ("#eef2f7", "#d1d5db", 0.82, 0.42, 1),
        SemanticMapLayer.INTERSECTION: ("#f1f5f9", "#d1d5db", 0.78, 0.4, 0),
        SemanticMapLayer.CROSSWALK: ("#fde68a", "#f59e0b", 0.55, 0.38, 2),
        SemanticMapLayer.WALKWAYS: ("#dcfce7", "#86efac", 0.40, 0.30, 0),
        SemanticMapLayer.CARPARK_AREA: ("#e0e7ff", "#a5b4fc", 0.35, 0.30, 0),
        SemanticMapLayer.STOP_LINE: ("#fecaca", "#ef4444", 0.70, 0.36, 3),
    }

    for layer in polygon_layers:
        for map_object in map_objects.get(layer, []):
            polygon = getattr(map_object, "polygon", None)
            if polygon is None:
                continue
            local_polygon = _geometry_local_coords(polygon, origin)
            face, edge, alpha, lw, zorder = polygon_style[layer]
            for poly in _iter_polygons(local_polygon):
                _plot_polygon(ax, poly, face, edge, alpha, lw, zorder)

    for layer in line_layers:
        for map_object in map_objects.get(layer, []):
            baseline = getattr(map_object, "baseline_path", None)
            linestring = getattr(baseline, "linestring", None)
            if linestring is None:
                continue
            local_line = _geometry_local_coords(linestring, origin)
            for line in _iter_lines(local_line):
                xs, ys = line.xy
                ax.plot(ys, xs, color="#ffffff", alpha=0.75, linewidth=1.3, zorder=5)
                ax.plot(ys, xs, color="#64748b", alpha=0.35, linewidth=0.45, zorder=6)


def _box_corners_xy(x: float, y: float, heading: float, length: float, width: float) -> np.ndarray:
    half_l = max(float(length), 0.1) / 2.0
    half_w = max(float(width), 0.1) / 2.0
    corners = np.array(
        [[half_l, half_w], [half_l, -half_w], [-half_l, -half_w], [-half_l, half_w], [half_l, half_w]],
        dtype=np.float32,
    )
    c = np.cos(float(heading))
    s = np.sin(float(heading))
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    return corners @ rot.T + np.array([x, y], dtype=np.float32)


def _draw_boxes(ax: plt.Axes, scene: Scene, frame_idx: int) -> None:
    annotations = scene.frames[frame_idx].annotations
    for box in annotations.boxes:
        corners = _box_corners_xy(
            box[BoundingBoxIndex.X],
            box[BoundingBoxIndex.Y],
            box[BoundingBoxIndex.HEADING],
            box[BoundingBoxIndex.LENGTH],
            box[BoundingBoxIndex.WIDTH],
        )
        ax.fill(corners[:, 1], corners[:, 0], facecolor="#475569", edgecolor="#0f172a", alpha=0.38, linewidth=0.7, zorder=12)

    ego = _box_corners_xy(0.0, 0.0, 0.0, 4.9, 2.1)
    ax.fill(ego[:, 1], ego[:, 0], facecolor="#111827", edgecolor="#020617", alpha=0.92, linewidth=1.0, zorder=20)


def _anchor_time_colors(num_segments: int, vis: Dict[str, Any], fallback_alpha: float) -> List[Tuple[float, float, float, float]]:
    if num_segments <= 0:
        return []
    colors = [
        str(vis.get("anchor_gradient_start_color", "#7c2d12")),
        str(vis.get("anchor_gradient_mid_color", "#f97316")),
        str(vis.get("anchor_gradient_end_color", "#fed7aa")),
    ]
    cmap = LinearSegmentedColormap.from_list("anchor_time_gradient", colors)
    alpha_start = float(vis.get("anchor_alpha_early", 0.62))
    alpha_end = float(vis.get("anchor_alpha_late", fallback_alpha))
    values = np.linspace(0.0, 1.0, num_segments, dtype=np.float32)
    rgba_list = []
    for value in values:
        rgba = list(cmap(float(value)))
        rgba[3] = alpha_start * (1.0 - float(value)) + alpha_end * float(value)
        rgba_list.append(tuple(rgba))
    return rgba_list


def _plot_modes(
    ax: plt.Axes,
    modes_xy: np.ndarray,
    color: str,
    alpha: float,
    lw: float,
    max_modes: int,
    *,
    time_gradient: bool = False,
    vis: Optional[Dict[str, Any]] = None,
) -> None:
    modes = _normalize_modes(modes_xy)
    modes = modes[: max(1, int(max_modes))]
    vis = vis or {}
    for mode in modes:
        pts = np.concatenate([np.zeros((1, 2), dtype=np.float32), mode[:, :2]], axis=0)
        plot_pts = np.stack([pts[:, 1], pts[:, 0]], axis=1)
        if time_gradient and len(plot_pts) >= 2:
            segments = np.stack([plot_pts[:-1], plot_pts[1:]], axis=1)
            collection = LineCollection(
                segments,
                colors=_anchor_time_colors(len(segments), vis, fallback_alpha=alpha),
                linewidths=lw,
                capstyle="round",
                joinstyle="round",
                zorder=30,
            )
            ax.add_collection(collection)
        else:
            ax.plot(plot_pts[:, 0], plot_pts[:, 1], color=color, alpha=alpha, linewidth=lw, zorder=30)

        early_steps = int(vis.get("anchor_gradient_early_steps", 2))
        if time_gradient and early_steps > 0:
            marker_pts = plot_pts[1 : min(len(plot_pts), early_steps + 1)]
            if len(marker_pts):
                ax.scatter(
                    marker_pts[:, 0],
                    marker_pts[:, 1],
                    s=float(vis.get("anchor_early_marker_size", 7.0)),
                    color=str(vis.get("anchor_gradient_start_color", "#7c2d12")),
                    alpha=float(vis.get("anchor_early_marker_alpha", 0.42)),
                    linewidths=0.0,
                    zorder=31,
                )


def _select_anchor_for_plot(predictions: Dict[str, Any], vis: Dict[str, Any]) -> Tuple[Optional[str], Optional[np.ndarray]]:
    source = str(vis.get("anchor_source", "plan_anchor")).lower()
    if source in {"noisy", "noisy_anchor", "noisy_plan_anchor", "initial_noisy"}:
        key, value = _first_prediction_array(
            predictions,
            ["noisy_plan_anchor", "initial_noisy_trajectory", "noisy_anchor"],
        )
        if value is not None:
            return key, value
        logger.warning("anchor_source=noisy requested, but predictions do not contain noisy_plan_anchor.")
    elif source in {"auto"}:
        key, value = _first_prediction_array(
            predictions,
            ["noisy_plan_anchor", "initial_noisy_trajectory", "noisy_anchor", "plan_anchor", "anchor", "anchor_mu"],
        )
        if value is not None:
            return key, value

    key, value = _first_prediction_array(predictions, ["plan_anchor"])
    if value is not None:
        return key, value
    return _first_prediction_array(predictions, ["anchor", "anchor_mu"])


def _plot_bev_anchor(scene: Scene, predictions: Dict[str, Any], out_path: Path, vis: Dict[str, Any]) -> Dict[str, Any]:
    frame_idx = scene.scene_metadata.num_history_frames - 1
    anchor_key, anchor = _select_anchor_for_plot(predictions, vis)
    if anchor is None:
        raise ValueError("Predictions do not contain a plottable anchor/noisy anchor.")

    fig, ax = plt.subplots(figsize=(7.2, 7.2), dpi=int(vis.get("dpi", 180)))
    ax.set_facecolor("#f8fafc")
    _draw_vector_map(ax, scene, frame_idx, vis)
    if bool(vis.get("draw_boxes", True)):
        _draw_boxes(ax, scene, frame_idx)
    _plot_modes(
        ax,
        anchor,
        color=str(vis.get("anchor_color", "#f97316")),
        alpha=float(vis.get("anchor_alpha", 0.22)),
        lw=float(vis.get("anchor_linewidth", 1.15)),
        max_modes=int(vis.get("max_anchor_modes", 64)),
        time_gradient=bool(vis.get("anchor_time_gradient", True)),
        vis=vis,
    )
    side = float(vis.get("bev_side", 32.0))
    forward = float(vis.get("bev_forward", 56.0))
    backward = float(vis.get("bev_backward", 12.0))
    ax.set_xlim(-side, side)
    ax.set_ylim(-backward, forward)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return {
        "anchor_key": anchor_key,
        "anchor_source": str(vis.get("anchor_source", "plan_anchor")),
        "anchor_shape": list(_to_numpy(anchor).shape),
        "anchor_png": str(out_path),
        "anchor_time_gradient": bool(vis.get("anchor_time_gradient", True)),
    }


def _prediction_mask_array(predictions: Dict[str, Any], names: Sequence[str]) -> Tuple[Optional[str], Optional[np.ndarray]]:
    for name in names:
        if name not in predictions:
            continue
        arr = _to_numpy(predictions[name])
        arr = np.squeeze(arr)
        if arr.ndim == 2:
            return name, arr.astype(bool)
        if arr.ndim == 3:
            return name, arr[0].astype(bool)
    return None, None


def _anchor_mask_for_plot(predictions: Dict[str, Any], vis: Dict[str, Any]) -> Tuple[Optional[str], Optional[np.ndarray]]:
    mask_key = str(vis.get("feasible_anchor_mask_key", "auto"))
    if mask_key and mask_key.lower() != "auto":
        return _prediction_mask_array(predictions, [mask_key])
    return _prediction_mask_array(
        predictions,
        [
            "physical_feasible_area_mask",
            "feasible_area_mask",
            "reachability_mask",
        ],
    )


def _anchor_lane_mask_for_plot(predictions: Dict[str, Any], area_key: Optional[str]) -> Tuple[Optional[str], Optional[np.ndarray]]:
    if area_key == "physical_feasible_area_mask":
        key, mask = _prediction_mask_array(predictions, ["physical_feasible_lane_mask"])
        if mask is not None:
            return key, mask
    return _prediction_mask_array(predictions, ["feasible_lane_mask"])


def _anchor_points_to_bev_pixels(points_xy: np.ndarray, mask_shape: Tuple[int, int], vis: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = mask_shape
    pixel_size = float(vis.get("feasible_anchor_pixel_size", vis.get("bev_pixel_size", 0.25)))
    ego_row = float(vis.get("feasible_anchor_ego_row", 0.0))
    ego_col = float(vis.get("feasible_anchor_ego_col", (width - 1) / 2.0))
    row = points_xy[:, 0] / max(pixel_size, 1e-6) + ego_row
    col = points_xy[:, 1] / max(pixel_size, 1e-6) + ego_col
    valid = np.isfinite(row) & np.isfinite(col) & (row >= 0) & (row < height) & (col >= 0) & (col < width)
    return row, col, valid


def _draw_anchor_feasible_mask(ax: plt.Axes, area_mask: np.ndarray, lane_mask: Optional[np.ndarray], title: str) -> None:
    height, width = area_mask.shape
    ax.imshow(np.ones((height, width, 3), dtype=np.float32), origin="upper", interpolation="nearest", zorder=0)
    area_rgba = np.zeros((height, width, 4), dtype=np.float32)
    area_rgba[area_mask, :3] = np.array([125, 211, 252], dtype=np.float32) / 255.0
    area_rgba[area_mask, 3] = 0.72
    ax.imshow(area_rgba, origin="upper", interpolation="nearest", zorder=1)
    if lane_mask is not None and lane_mask.shape == area_mask.shape:
        lane_rgba = np.zeros((height, width, 4), dtype=np.float32)
        lane_rgba[lane_mask, :3] = np.array([37, 99, 235], dtype=np.float32) / 255.0
        lane_rgba[lane_mask, 3] = 0.88
        ax.imshow(lane_rgba, origin="upper", interpolation="nearest", zorder=2)
    ax.scatter([(width - 1) / 2.0], [0.0], s=30, color="#111827", marker="x", linewidths=1.0, zorder=5)
    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(height - 0.5, -0.5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=10, color="#111827", pad=4)
    ax.axis("off")


def _anchor_sample_area_mask(
    row: np.ndarray,
    col: np.ndarray,
    mask_shape: Tuple[int, int],
    area_mask: Optional[np.ndarray],
    vis: Dict[str, Any],
) -> Optional[np.ndarray]:
    points = np.stack([col, row], axis=1).astype(np.float32)
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    if len(points) < int(vis.get("feasible_anchor_sample_area_min_points", 3)):
        return None

    percentile = float(vis.get("feasible_anchor_sample_area_percentile", 95.0))
    if 0.0 < percentile < 100.0 and len(points) > 3:
        center = np.median(points, axis=0)
        dist = np.linalg.norm(points - center[None], axis=1)
        keep = dist <= np.percentile(dist, percentile)
        if int(keep.sum()) >= 3:
            points = points[keep]

    hull = cv2.convexHull(points).reshape(-1, 2)
    if len(hull) < 3:
        return None

    height, width = mask_shape
    hull_px = np.round(hull).astype(np.int32)
    hull_px[:, 0] = np.clip(hull_px[:, 0], 0, width - 1)
    hull_px[:, 1] = np.clip(hull_px[:, 1], 0, height - 1)

    sample_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(sample_mask, hull_px, 1)

    dilate_px = int(vis.get("feasible_anchor_sample_area_dilate_px", 2))
    if dilate_px > 0:
        kernel_size = 2 * dilate_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        sample_mask = cv2.dilate(sample_mask, kernel, iterations=1)

    if bool(vis.get("feasible_anchor_sample_area_clip_to_feasible", False)) and area_mask is not None:
        sample_mask = np.logical_and(sample_mask.astype(bool), area_mask.astype(bool)).astype(np.uint8)

    return sample_mask.astype(bool)


def _draw_anchor_sample_area(
    ax: plt.Axes,
    row: np.ndarray,
    col: np.ndarray,
    mask_shape: Tuple[int, int],
    area_mask: Optional[np.ndarray],
    vis: Dict[str, Any],
) -> int:
    if not bool(vis.get("feasible_anchor_sample_area_enable", True)):
        return 0
    sample_mask = _anchor_sample_area_mask(row, col, mask_shape, area_mask, vis)
    if sample_mask is None or not sample_mask.any():
        return 0

    rgb = np.asarray(to_rgb(str(vis.get("feasible_anchor_sample_area_color", "#2563eb"))), dtype=np.float32)
    rgba = np.zeros((*sample_mask.shape, 4), dtype=np.float32)
    rgba[sample_mask, :3] = rgb
    rgba[sample_mask, 3] = float(vis.get("feasible_anchor_sample_area_alpha", 0.28))
    ax.imshow(rgba, origin="upper", interpolation="nearest", zorder=6)

    edge_alpha = float(vis.get("feasible_anchor_sample_area_edge_alpha", 0.72))
    edge_width = float(vis.get("feasible_anchor_sample_area_edge_width", 1.0))
    if edge_alpha > 0.0 and edge_width > 0.0:
        ax.contour(
            sample_mask.astype(np.float32),
            levels=[0.5],
            colors=[str(vis.get("feasible_anchor_sample_area_edge_color", "#1d4ed8"))],
            linewidths=edge_width,
            alpha=edge_alpha,
            zorder=7,
        )
    return int(sample_mask.sum())


def _anchor_point_cmap(vis: Dict[str, Any]):
    cmap_name = str(vis.get("feasible_anchor_point_cmap", "anchor_blue_gradient"))
    if cmap_name != "anchor_blue_gradient":
        return cmap_name
    return LinearSegmentedColormap.from_list(
        "anchor_blue_gradient",
        [
            str(vis.get("feasible_anchor_point_start_color", "#bfdbfe")),
            str(vis.get("feasible_anchor_point_mid_color", "#3b82f6")),
            str(vis.get("feasible_anchor_point_end_color", "#1e3a8a")),
        ],
    )


def _plot_feasible_anchor_points(
    predictions: Dict[str, Any],
    out_path: Path,
    vis: Dict[str, Any],
) -> Dict[str, Any]:
    if not bool(vis.get("feasible_anchor_points_enable", False)):
        return {}

    area_key, area_mask = _anchor_mask_for_plot(predictions, vis)
    if area_mask is None:
        raise ValueError("Predictions do not contain feasible/physical/reachability mask for anchor point visualization.")
    lane_key, lane_mask = _anchor_lane_mask_for_plot(predictions, area_key)

    source_names = _list_from_cfg(vis.get("feasible_anchor_points_source"))
    if not source_names:
        source_names = ["plan_anchor"]
    anchor_key, anchor = _first_prediction_array(predictions, source_names)
    if anchor is None:
        raise ValueError(f"Predictions do not contain requested anchor point source: {source_names}")
    modes = _normalize_modes(anchor)
    modes = modes[: max(1, int(vis.get("feasible_anchor_max_modes", vis.get("max_anchor_modes", 64))))]
    points = modes.reshape(-1, 2)
    steps = np.tile(np.arange(modes.shape[1], dtype=np.float32), modes.shape[0])
    max_points = int(vis.get("feasible_anchor_max_points", 0))
    if max_points > 0 and points.shape[0] > max_points:
        keep = np.linspace(0, points.shape[0] - 1, max_points).astype(np.int64)
        points = points[keep]
        steps = steps[keep]

    row, col, valid = _anchor_points_to_bev_pixels(points, area_mask.shape, vis)
    if not valid.any():
        raise ValueError("Anchor-free sample points are outside feasible mask image bounds.")
    row = row[valid]
    col = col[valid]
    steps = steps[valid]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    region_path = out_path.parent.parent / "feasible_anchor_region" / out_path.name
    region_path.parent.mkdir(parents=True, exist_ok=True)

    region_fig, region_ax = plt.subplots(1, 1, figsize=(4.8, 4.8), dpi=int(vis.get("dpi", 180)))
    _draw_anchor_feasible_mask(region_ax, area_mask, lane_mask, "Feasible Region + Sampling Range")
    sample_area_pixels = _draw_anchor_sample_area(region_ax, row, col, area_mask.shape, area_mask, vis)
    region_fig.tight_layout(pad=0.20)
    region_fig.savefig(region_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(region_fig)

    fig, ax = plt.subplots(1, 1, figsize=(4.8, 4.8), dpi=int(vis.get("dpi", 180)))
    _draw_anchor_feasible_mask(ax, area_mask, lane_mask, "Anchor-Free Gaussian Samples")
    if bool(vis.get("feasible_anchor_sample_area_on_points", False)):
        _draw_anchor_sample_area(ax, row, col, area_mask.shape, area_mask, vis)
    scatter = ax.scatter(
        col,
        row,
        c=steps,
        cmap=_anchor_point_cmap(vis),
        s=float(vis.get("feasible_anchor_point_size", 9.0)),
        alpha=float(vis.get("feasible_anchor_point_alpha", 0.82)),
        edgecolors="none",
        zorder=8,
    )
    if bool(vis.get("feasible_anchor_show_colorbar", False)):
        cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.02)
        cbar.set_label("Trajectory step", fontsize=8)
        cbar.ax.tick_params(labelsize=7)
    fig.tight_layout(pad=0.20)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return {
        "feasible_anchor_points_png": str(out_path),
        "feasible_anchor_region_png": str(region_path),
        "feasible_anchor_points_key": anchor_key,
        "feasible_anchor_points_source": source_names,
        "feasible_anchor_points_shape": list(_to_numpy(anchor).shape),
        "feasible_anchor_mask_key": area_key,
        "feasible_anchor_lane_mask_key": lane_key,
        "feasible_anchor_points_visible": int(valid.sum()),
        "feasible_anchor_sample_area_pixels": sample_area_pixels,
    }


def _image_uint8(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.dtype == np.uint8:
        return arr.copy()
    arr = arr.astype(np.float32)
    if arr.size and float(np.nanmax(arr)) <= 1.0:
        arr = arr * 255.0
    return np.clip(arr, 0.0, 255.0).astype(np.uint8)


def _lidar_points_to_image(points_xyz: np.ndarray, camera: Camera, image_shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    if camera.sensor2lidar_rotation is None or camera.sensor2lidar_translation is None or camera.intrinsics is None:
        raise ValueError("Front camera calibration is missing.")
    sensor2lidar_rotation = np.asarray(camera.sensor2lidar_rotation, dtype=np.float32)
    sensor2lidar_translation = np.asarray(camera.sensor2lidar_translation, dtype=np.float32)
    intrinsic = np.asarray(camera.intrinsics, dtype=np.float32)

    lidar2cam_r = np.linalg.inv(sensor2lidar_rotation)
    lidar2cam_t = sensor2lidar_translation @ lidar2cam_r.T
    lidar2cam_rt = np.eye(4, dtype=np.float32)
    lidar2cam_rt[:3, :3] = lidar2cam_r.T
    lidar2cam_rt[3, :3] = -lidar2cam_t

    viewpad = np.eye(4, dtype=np.float32)
    viewpad[: intrinsic.shape[0], : intrinsic.shape[1]] = intrinsic
    lidar2img_rt = viewpad @ lidar2cam_rt.T

    points_h = np.concatenate([points_xyz.astype(np.float32), np.ones((len(points_xyz), 1), dtype=np.float32)], axis=-1)
    points_img = (lidar2img_rt @ points_h.T).T
    depth = points_img[:, 2]
    eps = 1e-3
    pixels = points_img[:, :2] / np.maximum(depth[:, None], eps)
    img_h, img_w = image_shape
    valid = (
        (depth > eps)
        & (pixels[:, 0] > 0)
        & (pixels[:, 0] < img_w - 1)
        & (pixels[:, 1] > 0)
        & (pixels[:, 1] < img_h - 1)
    )
    return pixels, valid


def _densify_xy(xy: np.ndarray, points_per_segment: int) -> np.ndarray:
    if len(xy) < 2 or points_per_segment <= 1:
        return xy
    dense = [xy[0]]
    for start, end in zip(xy[:-1], xy[1:]):
        for frac in np.linspace(0.0, 1.0, points_per_segment + 1, dtype=np.float32)[1:]:
            dense.append(start * (1.0 - frac) + end * frac)
    return np.asarray(dense, dtype=np.float32)


def _estimate_heading_from_xy(xy: np.ndarray) -> np.ndarray:
    if len(xy) == 0:
        return np.zeros((0,), dtype=np.float32)
    if len(xy) == 1:
        return np.zeros((1,), dtype=np.float32)
    deltas = np.diff(xy[:, :2], axis=0)
    headings = np.arctan2(deltas[:, 1], deltas[:, 0]).astype(np.float32)
    return np.concatenate([headings[:1], headings], axis=0)


def _front_projection_offset_m(vis: Dict[str, Any]) -> float:
    reference = str(vis.get("front_projection_reference", "center")).lower()
    if "front_projection_offset_m" in vis:
        return float(vis["front_projection_offset_m"])
    vehicle = get_pacifica_parameters()
    if reference in {"rear", "rear_axle", "rear-axle"}:
        return 0.0
    if reference in {"center", "vehicle_center", "ego_center"}:
        return float(vehicle.rear_axle_to_center)
    if reference in {"front", "front_bumper", "front-center", "front_center"}:
        return float(vehicle.rear_axle_to_center + vehicle.half_length)
    raise ValueError(
        "Unsupported front_projection_reference="
        f"{reference}. Use rear_axle, center, or front_bumper."
    )


def _trajectory_xy_for_projection(
    trajectory: np.ndarray,
    include_origin: bool,
    vis: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    vis = vis or {}
    traj = trajectory.astype(np.float32)
    xy = traj[:, :2]
    heading = traj[:, 2] if traj.ndim == 2 and traj.shape[-1] >= 3 else _estimate_heading_from_xy(xy)
    if include_origin:
        xy = np.concatenate([np.zeros((1, 2), dtype=np.float32), xy], axis=0)
        heading = np.concatenate([np.zeros((1,), dtype=np.float32), heading.astype(np.float32)], axis=0)
    offset_m = _front_projection_offset_m(vis)
    if abs(offset_m) > 1e-6:
        xy = xy + offset_m * np.stack([np.cos(heading), np.sin(heading)], axis=-1).astype(np.float32)
    return xy


def _trajectory_length_m(xy: np.ndarray) -> float:
    if len(xy) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(xy[:, :2], axis=0), axis=1).sum())


def _estimate_ground_z_from_lidar(scene: Scene, frame_idx: int, xy: np.ndarray, vis: Dict[str, Any]) -> np.ndarray:
    default_z = float(vis.get("trajectory_z", -1.2))
    frame_lidar = scene.frames[frame_idx].lidar
    if frame_lidar.lidar_pc is None:
        return np.full((len(xy), 1), default_z, dtype=np.float32)

    lidar_xyz = np.asarray(frame_lidar.lidar_pc[LidarIndex.POSITION].T, dtype=np.float32)
    if lidar_xyz.ndim != 2 or lidar_xyz.shape[1] < 3 or len(lidar_xyz) == 0:
        return np.full((len(xy), 1), default_z, dtype=np.float32)

    min_z = float(vis.get("trajectory_ground_min_z", -4.0))
    max_z = float(vis.get("trajectory_ground_max_z", 0.5))
    valid = np.isfinite(lidar_xyz).all(axis=1) & (lidar_xyz[:, 2] >= min_z) & (lidar_xyz[:, 2] <= max_z)
    lidar_xyz = lidar_xyz[valid]
    if len(lidar_xyz) == 0:
        return np.full((len(xy), 1), default_z, dtype=np.float32)

    radius = float(vis.get("trajectory_ground_radius", 1.5))
    percentile = float(vis.get("trajectory_ground_percentile", 20.0))
    fallback_nearest = bool(vis.get("trajectory_ground_fallback_nearest", True))
    z_values: List[float] = []
    pc_xy = lidar_xyz[:, :2]
    for point_xy in xy.astype(np.float32):
        dist2 = np.sum((pc_xy - point_xy[None]) ** 2, axis=1)
        nearby = lidar_xyz[dist2 <= radius * radius, 2]
        if len(nearby):
            z_values.append(float(np.percentile(nearby, percentile)))
        elif fallback_nearest:
            nearest_idx = int(np.argmin(dist2))
            z_values.append(float(lidar_xyz[nearest_idx, 2]))
        else:
            z_values.append(default_z)
    return np.asarray(z_values, dtype=np.float32).reshape(-1, 1)


def _trajectory_xyz_for_projection(
    scene: Scene,
    frame_idx: int,
    xy: np.ndarray,
    vis: Dict[str, Any],
) -> np.ndarray:
    z_mode = str(vis.get("trajectory_z_mode", "constant")).lower()
    if z_mode in {"lidar_ground", "ground", "pointcloud"}:
        z = _estimate_ground_z_from_lidar(scene, frame_idx, xy, vis)
    else:
        z = np.full((len(xy), 1), float(vis.get("trajectory_z", -1.2)), dtype=np.float32)
    return np.concatenate([xy.astype(np.float32), z.astype(np.float32)], axis=1)


def _draw_selected_on_front(scene: Scene, predictions: Dict[str, Any], out_path: Path, vis: Dict[str, Any]) -> Dict[str, Any]:
    frame_idx = scene.scene_metadata.num_history_frames - 1
    camera = scene.frames[frame_idx].cameras.cam_f0
    if camera.image is None:
        raise ValueError("cam_f0 image is missing.")

    traj_key, selected = _first_prediction_array(predictions, ["trajectory"])
    if selected is None:
        raise ValueError("Predictions do not contain selected trajectory.")
    traj = _normalize_trajectory(selected)

    image = _image_uint8(camera.image)
    image_h, image_w = image.shape[:2]
    offsets = np.array(
        [
            float(vis.get("trajectory_x_offset", 0.0)),
            float(vis.get("trajectory_y_offset", 0.0)),
        ],
        dtype=np.float32,
    )

    def _draw_projected_trajectory(
        trajectory: np.ndarray,
        *,
        line_color: Sequence[int],
        shadow_color: Sequence[int],
        thickness: int,
        point_radius: int,
    ) -> Dict[str, Any]:
        include_origin = bool(vis.get("front_include_origin", True))
        trajectory_xy = _trajectory_xy_for_projection(trajectory, include_origin=include_origin, vis=vis)
        xy = _densify_xy(
            trajectory_xy,
            int(vis.get("projection_points_per_segment", 16)),
        )
        xy = xy + offsets[None]
        points_xyz = _trajectory_xyz_for_projection(scene, frame_idx, xy, vis)
        pixels, valid = _lidar_points_to_image(points_xyz, camera, (image_h, image_w))

        color = tuple(int(c) for c in line_color)
        shadow = tuple(int(c) for c in shadow_color)
        shadow_thickness = max(int(thickness) + 4, 7)

        for p0, p1, v0, v1 in zip(pixels[:-1], pixels[1:], valid[:-1], valid[1:]):
            if not (bool(v0) and bool(v1)):
                continue
            pt0 = (int(round(p0[0])), int(round(p0[1])))
            pt1 = (int(round(p1[0])), int(round(p1[1])))
            cv2.line(image, pt0, pt1, shadow, shadow_thickness, cv2.LINE_AA)
            cv2.line(image, pt0, pt1, color, int(thickness), cv2.LINE_AA)

        # Mark future samples only; the current origin is usually hidden by the ego hood.
        raw_xy = _trajectory_xy_for_projection(trajectory, include_origin=False, vis=vis) + offsets[None]
        raw_xyz = _trajectory_xyz_for_projection(scene, frame_idx, raw_xy, vis)
        raw_pixels, raw_valid = _lidar_points_to_image(raw_xyz, camera, (image_h, image_w))
        for pixel, is_valid in zip(raw_pixels, raw_valid):
            if not bool(is_valid):
                continue
            center = (int(round(pixel[0])), int(round(pixel[1])))
            cv2.circle(image, center, int(point_radius) + 2, shadow, -1, cv2.LINE_AA)
            cv2.circle(image, center, int(point_radius), color, -1, cv2.LINE_AA)
        endpoint_pixel = raw_pixels[-1].tolist() if len(raw_pixels) else None
        endpoint_valid = bool(raw_valid[-1]) if len(raw_valid) else False
        return {
            "projected_points": int(valid.sum()),
            "sample_projected_points": int(raw_valid.sum()),
            "endpoint_xy": trajectory[-1, :2].astype(float).tolist() if len(trajectory) else None,
            "endpoint_projection_xy": raw_xy[-1].astype(float).tolist() if len(raw_xy) else None,
            "endpoint_pixel": endpoint_pixel,
            "endpoint_valid": endpoint_valid,
        }

    gt_projection: Optional[Dict[str, Any]] = None
    gt_length_m: Optional[float] = None
    if bool(vis.get("front_draw_gt", True)):
        gt = scene.get_future_trajectory(num_trajectory_frames=traj.shape[0]).poses.astype(np.float32)
        gt_length_m = _trajectory_length_m(_trajectory_xy_for_projection(gt, include_origin=True, vis=vis))
        gt_projection = _draw_projected_trajectory(
            gt,
            line_color=vis.get("gt_rgb_color", [239, 68, 68]),
            shadow_color=vis.get("gt_shadow_color", [127, 29, 29]),
            thickness=int(vis.get("gt_projection_thickness", max(3, int(vis.get("projection_thickness", 5)) - 1))),
            point_radius=int(vis.get("gt_projection_point_radius", 4)),
        )

    selected_projection = _draw_projected_trajectory(
        traj,
        line_color=vis.get("selected_rgb_color", [37, 99, 235]),
        shadow_color=vis.get("selected_shadow_color", [15, 23, 42]),
        thickness=int(vis.get("projection_thickness", 5)),
        point_radius=int(vis.get("projection_point_radius", 5)),
    )
    selected_length_m = _trajectory_length_m(_trajectory_xy_for_projection(traj, include_origin=True, vis=vis))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(out_path)
    return {
        "trajectory_key": traj_key,
        "trajectory_shape": list(_to_numpy(selected).shape),
        "front_png": str(out_path),
        "projected_points": selected_projection["projected_points"],
        "gt_projected_points": None if gt_projection is None else gt_projection["projected_points"],
        "selected_length_m": selected_length_m,
        "gt_length_m": gt_length_m,
        "selected_projection": selected_projection,
        "gt_projection": gt_projection,
        "front_include_origin": bool(vis.get("front_include_origin", True)),
        "front_projection_reference": str(vis.get("front_projection_reference", "center")),
        "front_projection_offset_m": _front_projection_offset_m(vis),
        "trajectory_z": float(vis.get("trajectory_z", -1.2)),
        "trajectory_z_mode": str(vis.get("trajectory_z_mode", "constant")),
    }


def _build_scene_loader(cfg: DictConfig, agent: AbstractAgent, vis: Dict[str, Any]) -> SceneLoader:
    split_name = str(vis.get("log_split", "test"))
    scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    log_names = _get_split_logs(cfg, split_name)
    if log_names is not None:
        scene_filter.log_names = log_names

    tokens = _list_from_cfg(vis.get("tokens"))
    if tokens:
        scene_filter.tokens = tokens
        scene_filter.max_scenes = None
    else:
        start_index = max(0, int(vis.get("start_index", 0)))
        start_token = str(vis.get("start_token", "") or "")
        max_scenes = int(vis.get("max_scenes", 8))
        if start_token:
            scene_filter.max_scenes = None
        elif start_index > 0 and max_scenes > 0:
            scene_filter.max_scenes = start_index + max_scenes
        else:
            scene_filter.max_scenes = max_scenes if max_scenes > 0 else None

    return SceneLoader(
        sensor_blobs_path=Path(cfg.sensor_blobs_path),
        data_path=Path(cfg.navsim_log_path),
        scene_filter=scene_filter,
        sensor_config=agent.get_sensor_config(),
    )


def _select_token_window(tokens: List[str], vis: Dict[str, Any]) -> List[str]:
    if _list_from_cfg(vis.get("tokens")):
        return tokens

    start_index = max(0, int(vis.get("start_index", 0)))
    start_token = str(vis.get("start_token", "") or "")
    if start_token:
        if start_token not in tokens:
            raise ValueError(f"start_token not found in loaded scenes: {start_token}")
        start_index = tokens.index(start_token)

    max_scenes = int(vis.get("max_scenes", 8))
    selected = tokens[start_index:]
    if max_scenes > 0:
        selected = selected[:max_scenes]
    return selected


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)
    vis = _cfg_to_dict(getattr(cfg, "vis", {}))
    output_dir = Path(str(vis.get("output_dir", Path(cfg.output_dir) / "navsim_test_vis")))
    device_name = str(vis.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    device = torch.device(device_name)

    logger.info("Building agent")
    agent: AbstractAgent = instantiate(cfg.agent)
    agent.to(device)
    agent.eval()

    logger.info("Building SceneLoader")
    scene_loader = _build_scene_loader(cfg, agent, vis)
    loaded_tokens = scene_loader.tokens
    tokens = _select_token_window(loaded_tokens, vis)
    logger.info("Loaded %d scenes, selected %d scenes for visualization", len(loaded_tokens), len(tokens))
    if not tokens:
        raise RuntimeError("No scenes matched the requested split/tokens.")

    pdm_context = _build_pdm_context(cfg, vis)
    if pdm_context is None:
        logger.info("PDM scoring disabled or metric cache path not provided.")
    else:
        logger.info("PDM scoring enabled with metric cache: %s", pdm_context["metric_cache_path"])

    manifest_path = output_dir / "manifest.jsonl"
    if manifest_path.exists() and bool(vis.get("overwrite_manifest", True)):
        manifest_path.unlink()

    saved = 0
    skipped = 0
    for index, token in enumerate(tqdm(tokens, desc="Rendering NAVSIM visualizations")):
        try:
            scene = scene_loader.get_scene_from_token(token)
            predictions = _compute_predictions(agent, scene, device)
            prefix = f"{index:04d}_{_safe_name(token)}"
            anchor_info = _plot_bev_anchor(
                scene,
                predictions,
                output_dir / "bev_anchor" / f"{prefix}.png",
                vis,
            )
            feasible_anchor_info = _plot_feasible_anchor_points(
                predictions,
                output_dir / "feasible_anchor_points" / f"{prefix}.png",
                vis,
            )
            front_info = _draw_selected_on_front(
                scene,
                predictions,
                output_dir / "front_selected" / f"{prefix}.png",
                vis,
            )
            pdm_info = _compute_pdm_for_selected(token, predictions, pdm_context, vis)
            meta = {
                "token": token,
                "log_name": scene.scene_metadata.log_name,
                "map_name": scene.scene_metadata.map_name,
                **anchor_info,
                **feasible_anchor_info,
                **front_info,
                **pdm_info,
            }
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            with manifest_path.open("a", encoding="utf-8") as fp:
                fp.write(json.dumps(_to_jsonable(meta), ensure_ascii=False) + "\n")
            saved += 1
        except Exception as exc:
            skipped += 1
            logger.warning("Skipping token=%s: %s", token, exc)

    logger.info("Done. saved=%d skipped=%d output_dir=%s", saved, skipped, output_dir)


if __name__ == "__main__":
    main()
