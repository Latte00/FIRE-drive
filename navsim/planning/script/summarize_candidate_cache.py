"""
Summarize high-quality trajectory candidates stored in DiffusionDrive caches.

The script scans existing ``transfuser_target.gz`` files and reports dataset-level
statistics useful for candidate supervision analysis:

  - candidate availability and valid top-k count
  - candidate score improvement over GT
  - PDM component distributions
  - optional online PDM re-scoring for GT and the cached best candidate
  - best-candidate vs GT six-component quality tables
  - fixed-bin candidate / best-candidate / GT score distributions
  - trajectory diversity and GT coverage proxies
  - lane-change proxy counts based on lateral displacement

It is read-only.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import lzma
import math
import pickle
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm


DEFAULT_COMPONENT_NAMES = [
    "no_collision",
    "drivable",
    "progress",
    "ttc",
    "comfort",
    "direction_or_wrong_lane",
]

PREFERRED_COMPONENT_ORDER = [
    "no_collision",
    "drivable",
    "progress",
    "ttc",
    "comfort",
    "driving_direction",
    "direction_or_wrong_lane",
    "wrong_lane",
]

DEFAULT_SCORE_BINS = "0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0"

ONLINE_PDM_COMPONENT_NAMES = [
    "no_collision",
    "drivable",
    "progress",
    "ttc",
    "comfort",
    "driving_direction",
]

COMPONENT_DISPLAY_NAMES = {
    "score": "PDM",
    "no_collision": "NC",
    "drivable": "DAC",
    "progress": "EP",
    "ttc": "TTC",
    "comfort": "C",
    "driving_direction": "DDC",
    "direction_or_wrong_lane": "DDC/WL",
    "wrong_lane": "WL",
}

GT_COLOR = "#7c3aed"
CANDIDATE_COLOR = "#16a34a"
DARK_COLOR = "#111827"
BEST_CANDIDATE_COLOR = "#15803d"
FEASIBLE_OVERLAY_COLOR = "#93c5fd"
MAP_BASE_FACE = "#f4f5f7"
MAP_PANEL_FACE = "#f8fafc"
MAP_SPINE = "#cbd5e1"
MAP_LANE_FILL = "#d8dce3"
MAP_LANE_EDGE = "#adb5c2"
MAP_ROUTE_FILL = "#d9e3ff"
MAP_ROUTE_EDGE = "#5c78d6"
MAP_CENTERLINE = "#4f6cd6"
MAP_AGENT_BOX = "#3f444d"
EGO_BOX_COLOR = "#2563eb"

SEMANTIC_COLORS = [
    "#111827",
    "#d1d5db",
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


def load_pickle_gz(path: Path) -> Dict[str, object]:
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def load_lzma_pickle(path: Path) -> object:
    with lzma.open(path, "rb") as f:
        return pickle.load(f)


def to_numpy(value) -> np.ndarray:
    if value is None:
        return np.asarray([])
    if hasattr(value, "detach") and callable(value.detach):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def set_matplotlib_style(plt) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
        }
    )


def normalize_plot_formats(formats: Optional[Sequence[str]]) -> List[str]:
    normalized: List[str] = []
    for fmt in formats or ["png"]:
        clean = str(fmt).strip().lower().lstrip(".")
        if not clean:
            continue
        if clean not in normalized:
            normalized.append(clean)
    return normalized or ["png"]


def extend_plot_formats(formats: Sequence[str], extra_formats: Sequence[str]) -> List[str]:
    merged = normalize_plot_formats(formats)
    for fmt in normalize_plot_formats(extra_formats):
        if fmt not in merged:
            merged.append(fmt)
    return merged


def save_figure(fig, base_path: Path, formats: Sequence[str], **kwargs) -> Path:
    base = base_path.with_suffix("") if base_path.suffix else base_path
    first_path: Optional[Path] = None
    for fmt in normalize_plot_formats(formats):
        out_path = base.with_suffix(f".{fmt}")
        fig.savefig(out_path, **kwargs)
        if first_path is None:
            first_path = out_path
    assert first_path is not None
    return first_path


def token_from_data(data: Dict[str, object], path: Path) -> str:
    token = data.get("token")
    if isinstance(token, str):
        return token
    if isinstance(token, bytes):
        return token.decode("utf-8", errors="replace")
    return path.parent.name


def safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("_") or "sample"


def display_metric_label(name: str) -> str:
    return COMPONENT_DISPLAY_NAMES.get(name, name)


def token_from_metric_path(path_str: str) -> str:
    normalized = path_str.strip().replace("\\", "/")
    parts = [part for part in normalized.split("/") if part]
    if len(parts) >= 2:
        return parts[-2]
    return Path(path_str).parent.name


def load_metric_cache_index(metric_cache_path: Optional[str]) -> Dict[str, str]:
    if not metric_cache_path:
        return {}
    root = Path(metric_cache_path)
    metadata_dir = root / "metadata"
    if not metadata_dir.is_dir():
        print(f"metric map vis disabled: missing metadata dir {metadata_dir}")
        return {}
    metadata_files = sorted(p for p in metadata_dir.iterdir() if p.is_file() and p.suffix == ".csv")
    if not metadata_files:
        print(f"metric map vis disabled: no csv in {metadata_dir}")
        return {}
    out: Dict[str, str] = {}
    for metadata_file in metadata_files:
        with open(metadata_file, "r", encoding="utf-8") as f:
            for line in f.read().splitlines()[1:]:
                line = line.strip()
                if not line:
                    continue
                out[token_from_metric_path(line)] = line
    return out


def iter_target_files(
    cache_root: Path,
    roots: Optional[Sequence[str]],
    limit: int,
) -> Iterable[Path]:
    count = 0
    if roots:
        search_roots = [cache_root / root for root in roots]
    else:
        search_roots = [cache_root]
    for root in search_roots:
        if not root.is_dir():
            continue
        for path in root.rglob("transfuser_target.gz"):
            if limit > 0 and count >= limit:
                return
            count += 1
            yield path


def as_candidates(value) -> Optional[np.ndarray]:
    arr = to_numpy(value).astype(np.float32, copy=False)
    arr = np.squeeze(arr)
    if arr.ndim != 3 or arr.shape[-1] < 2:
        return None
    return arr


def as_trajectory(value) -> Optional[np.ndarray]:
    arr = to_numpy(value).astype(np.float32, copy=False)
    arr = np.squeeze(arr)
    if arr.ndim != 2 or arr.shape[-1] < 2:
        return None
    return arr


def bev_to_class_map(value: object) -> np.ndarray:
    bev = to_numpy(value)
    bev = np.squeeze(bev)
    if bev.ndim == 2:
        return bev.astype(np.int64, copy=False)
    if bev.ndim == 3:
        if bev.shape[0] <= 32:
            return bev.argmax(axis=0).astype(np.int64, copy=False)
        if bev.shape[-1] <= 32:
            return bev.argmax(axis=-1).astype(np.int64, copy=False)
    raise ValueError(f"Unsupported BEV shape: {bev.shape}")


def as_vector(value, length: int, default: float, dtype=np.float32) -> np.ndarray:
    if value is None:
        return np.full((length,), default, dtype=dtype)
    arr = np.squeeze(to_numpy(value)).reshape(-1)
    if arr.size != length:
        return np.full((length,), default, dtype=dtype)
    return arr.astype(dtype, copy=False)


def as_components(value, length: int) -> Optional[np.ndarray]:
    if value is None:
        return None
    arr = to_numpy(value).astype(np.float32, copy=False)
    arr = np.squeeze(arr)
    if arr.ndim != 2 or arr.shape[0] != length:
        return None
    return arr


def as_component_vector(value) -> Optional[np.ndarray]:
    if value is None:
        return None
    arr = to_numpy(value).astype(np.float32, copy=False)
    arr = np.squeeze(arr)
    if arr.ndim != 1:
        return None
    return arr


def format_float_list(values: np.ndarray) -> str:
    vals = np.asarray(values, dtype=np.float32).reshape(-1)
    vals = vals[np.isfinite(vals)]
    return ";".join(f"{float(v):.6g}" for v in vals)


def parse_float_list(value: object) -> List[float]:
    if value is None:
        return []
    text = str(value)
    if not text:
        return []
    out: List[float] = []
    for item in text.split(";"):
        try:
            val = float(item)
        except ValueError:
            continue
        if math.isfinite(val):
            out.append(val)
    return out


def finite_quantiles(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": math.nan, "std": math.nan, "p05": math.nan, "p25": math.nan, "p50": math.nan, "p75": math.nan, "p95": math.nan}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "p05": float(np.quantile(arr, 0.05)),
        "p25": float(np.quantile(arr, 0.25)),
        "p50": float(np.quantile(arr, 0.50)),
        "p75": float(np.quantile(arr, 0.75)),
        "p95": float(np.quantile(arr, 0.95)),
    }


def finite_stats(values: Sequence[float]) -> Dict[str, object]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    stats: Dict[str, object] = finite_quantiles(arr)
    stats["count"] = int(arr.size)
    return stats


def parse_score_bins(value: object) -> np.ndarray:
    text = str(value or DEFAULT_SCORE_BINS).strip()
    parts = [part for part in re.split(r"[,;\s]+", text) if part]
    edges = np.asarray([float(part) for part in parts], dtype=np.float64)
    if edges.size < 2:
        raise ValueError("--score-bins must contain at least two ascending edges")
    if not np.all(np.isfinite(edges)):
        raise ValueError("--score-bins contains non-finite values")
    if np.any(np.diff(edges) <= 0.0):
        raise ValueError("--score-bins must be strictly increasing")
    return edges


def score_bin_label(left: float, right: float, is_last: bool) -> str:
    left_text = f"{left:.3g}"
    right_text = f"{right:.3g}"
    return f"[{left_text}, {right_text}{']' if is_last else ')'}"


def component_sort_key(name: str) -> Tuple[int, str]:
    if name in PREFERRED_COMPONENT_ORDER:
        return PREFERRED_COMPONENT_ORDER.index(name), name
    return len(PREFERRED_COMPONENT_ORDER), name


def shared_component_names(rows: Sequence[Dict[str, object]], max_count: int = 6) -> List[str]:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    best_names = {
        key.replace("best_component_", "")
        for row in ok_rows
        for key in row
        if key.startswith("best_component_")
    }
    gt_names = {
        key.replace("gt_component_", "")
        for row in ok_rows
        for key in row
        if key.startswith("gt_component_")
    }
    names = sorted(best_names & gt_names, key=component_sort_key)
    return names[:max_count]


def finite_row_values(rows: Sequence[Dict[str, object]], key: str) -> List[float]:
    values: List[float] = []
    for row in rows:
        try:
            val = float(row.get(key, math.nan))
        except Exception:
            continue
        if math.isfinite(val):
            values.append(val)
    return values


def collect_score_values(rows: Sequence[Dict[str, object]]) -> Dict[str, np.ndarray]:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    candidate_scores = np.asarray(
        [score for row in ok_rows for score in parse_float_list(row.get("candidate_scores", ""))],
        dtype=np.float64,
    )
    best_candidate_scores = np.asarray(
        finite_row_values(ok_rows, "best_candidate_score"),
        dtype=np.float64,
    )
    gt_scores = np.asarray(finite_row_values(ok_rows, "gt_score"), dtype=np.float64)
    return {
        "candidate": candidate_scores[np.isfinite(candidate_scores)],
        "best_candidate": best_candidate_scores[np.isfinite(best_candidate_scores)],
        "gt": gt_scores[np.isfinite(gt_scores)],
    }


def build_quality_summary(rows: Sequence[Dict[str, object]]) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    component_names = shared_component_names(ok_rows, max_count=6)
    sources = [
        ("best_candidate", "Best candidate", "best_component_", "best_candidate_score"),
        ("gt", "GT", "gt_component_", "gt_score"),
        ("all_valid_candidates_mean", "All valid candidates mean", "component_mean_", "mean_candidate_score"),
    ]

    csv_rows: List[Dict[str, object]] = []
    json_sources: Dict[str, object] = {}
    for source_key, source_label, prefix, score_key in sources:
        score_stats = finite_stats(finite_row_values(ok_rows, score_key))
        csv_rows.append(
            {
                "source": source_key,
                "source_label": source_label,
                "metric": "score",
                "display_name": display_metric_label("score"),
                **score_stats,
            }
        )

        component_stats: Dict[str, object] = {}
        component_means: List[float] = []
        for component_name in component_names:
            stats = finite_stats(finite_row_values(ok_rows, f"{prefix}{component_name}"))
            component_stats[component_name] = stats
            if math.isfinite(float(stats["mean"])):
                component_means.append(float(stats["mean"]))
            csv_rows.append(
                {
                    "source": source_key,
                    "source_label": source_label,
                    "metric": component_name,
                    "display_name": display_metric_label(component_name),
                    **stats,
                }
            )

        json_sources[source_key] = {
            "label": source_label,
            "score": score_stats,
            "components": component_stats,
            "component_mean_of_means": float(np.mean(component_means)) if component_means else math.nan,
        }

    best_components = json_sources.get("best_candidate", {}).get("components", {})
    gt_components = json_sources.get("gt", {}).get("components", {})
    best_minus_gt: Dict[str, float] = {}
    for name in component_names:
        best_mean = float(best_components.get(name, {}).get("mean", math.nan))
        gt_mean = float(gt_components.get(name, {}).get("mean", math.nan))
        best_minus_gt[name] = best_mean - gt_mean if math.isfinite(best_mean) and math.isfinite(gt_mean) else math.nan

    return (
        {
            "num_ok_samples": len(ok_rows),
            "component_names": component_names,
            "component_display_names": {name: display_metric_label(name) for name in component_names},
            "sources": json_sources,
            "best_minus_gt_component_mean": best_minus_gt,
        },
        csv_rows,
    )


def build_score_distributions(
    rows: Sequence[Dict[str, object]],
    bins: np.ndarray,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    score_values = collect_score_values(rows)
    csv_rows: List[Dict[str, object]] = []
    json_sources: Dict[str, object] = {}
    source_labels = {
        "candidate": "All valid candidates",
        "best_candidate": "Best candidate per sample",
        "gt": "GT",
    }

    for source_key, values in score_values.items():
        hist, edges = np.histogram(values, bins=bins)
        total = int(values.size)
        source_bins: List[Dict[str, object]] = []
        for idx, count in enumerate(hist.tolist()):
            left = float(edges[idx])
            right = float(edges[idx + 1])
            item = {
                "source": source_key,
                "source_label": source_labels[source_key],
                "bin_index": idx,
                "bin_left": left,
                "bin_right": right,
                "bin_label": score_bin_label(left, right, idx == len(hist) - 1),
                "count": int(count),
                "fraction": float(count / total) if total > 0 else math.nan,
            }
            source_bins.append(item)
            csv_rows.append(item)

        json_sources[source_key] = {
            "label": source_labels[source_key],
            "total": total,
            "underflow_count": int((values < bins[0]).sum()) if total else 0,
            "overflow_count": int((values > bins[-1]).sum()) if total else 0,
            "bins": source_bins,
        }

    return (
        {
            "bin_edges": [float(v) for v in bins.tolist()],
            "sources": json_sources,
        },
        csv_rows,
    )


def write_quality_outputs(out_dir: Path, rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    quality_summary, quality_rows = build_quality_summary(rows)
    quality_json_path = out_dir / "quality_summary.json"
    quality_csv_path = out_dir / "quality_summary.csv"
    with open(quality_json_path, "w", encoding="utf-8") as f:
        json.dump(quality_summary, f, ensure_ascii=False, indent=2)
    write_csv(quality_csv_path, quality_rows)
    return {
        "json": str(quality_json_path),
        "csv": str(quality_csv_path),
        "summary": quality_summary,
    }


def write_score_distribution_outputs(
    out_dir: Path,
    rows: Sequence[Dict[str, object]],
    args: argparse.Namespace,
) -> Dict[str, object]:
    bins = parse_score_bins(args.score_bins)
    distribution_summary, distribution_rows = build_score_distributions(rows, bins)
    distribution_json_path = out_dir / "score_distributions.json"
    distribution_csv_path = out_dir / "score_distributions.csv"
    with open(distribution_json_path, "w", encoding="utf-8") as f:
        json.dump(distribution_summary, f, ensure_ascii=False, indent=2)
    write_csv(distribution_csv_path, distribution_rows)
    write_score_distribution_chart_csvs(out_dir, distribution_summary)
    return {
        "json": str(distribution_json_path),
        "csv": str(distribution_csv_path),
        "summary": distribution_summary,
    }


def read_roots_file(path: Optional[str]) -> List[str]:
    if not path:
        return []
    roots: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            roots.append(text)
    return roots


def safe_float(value: object, default: float = math.nan) -> float:
    arr = np.squeeze(to_numpy(value)).reshape(-1)
    if arr.size == 0:
        return default
    val = float(arr[0])
    return val if math.isfinite(val) else default


def path_length_xy(traj: np.ndarray) -> float:
    xy = traj[:, :2]
    if xy.shape[0] < 2:
        return float(np.linalg.norm(xy[-1]))
    start = np.zeros((1, 2), dtype=np.float32)
    pts = np.concatenate([start, xy], axis=0)
    return float(np.linalg.norm(pts[1:] - pts[:-1], axis=-1).sum())


def pairwise_diversity(candidates: np.ndarray, valid: np.ndarray) -> Tuple[float, float]:
    traj = candidates[valid, :, :2]
    n = traj.shape[0]
    if n < 2:
        return math.nan, math.nan
    diff = traj[:, None] - traj[None]
    dist = np.linalg.norm(diff, axis=-1)
    upper = np.triu_indices(n, k=1)
    pair_ade = dist.mean(axis=-1)[upper]
    pair_fde = dist[:, :, -1][upper]
    return float(pair_ade.mean()), float(pair_fde.mean())


def gt_coverage(candidates: np.ndarray, valid: np.ndarray, gt: Optional[np.ndarray]) -> Tuple[float, float]:
    if gt is None:
        return math.nan, math.nan
    traj = candidates[valid, :, :2]
    if traj.size == 0:
        return math.nan, math.nan
    steps = min(traj.shape[1], gt.shape[0])
    if steps <= 0:
        return math.nan, math.nan
    dist = np.linalg.norm(traj[:, :steps] - gt[None, :steps, :2], axis=-1)
    ade = dist.mean(axis=-1)
    fde = dist[:, -1]
    return float(ade.min()), float(fde.min())


def xy_to_image(xy: np.ndarray, pixel_size: float, width: int) -> Tuple[np.ndarray, np.ndarray]:
    row = xy[:, 0] / max(pixel_size, 1e-6)
    col = xy[:, 1] / max(pixel_size, 1e-6) + (width - 1) / 2.0
    return row, col


def box_corners_xy(box_state: np.ndarray) -> np.ndarray:
    x, y, heading, length, width = [float(v) for v in box_state[:5]]
    half_l = length / 2.0
    half_w = width / 2.0
    corners = np.array(
        [[half_l, half_w], [-half_l, half_w], [-half_l, -half_w], [half_l, -half_w], [half_l, half_w]],
        dtype=np.float32,
    )
    c = float(np.cos(heading))
    s = float(np.sin(heading))
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    return corners @ rot.T + np.array([x, y], dtype=np.float32)


def transform_local_to_global(traj: np.ndarray, ego_pose: np.ndarray) -> np.ndarray:
    arr = np.asarray(traj, dtype=np.float32)
    pose = np.asarray(ego_pose, dtype=np.float32).reshape(3)
    out = arr.copy()
    c = float(np.cos(pose[2]))
    s = float(np.sin(pose[2]))
    lx = arr[..., 0]
    ly = arr[..., 1]
    out[..., 0] = pose[0] + c * lx - s * ly
    out[..., 1] = pose[1] + s * lx + c * ly
    if out.shape[-1] >= 3:
        out[..., 2] = arr[..., 2] + pose[2]
    return out


def iter_polygons(geometry) -> List[object]:
    if geometry is None or getattr(geometry, "is_empty", True):
        return []
    geom_type = getattr(geometry, "geom_type", "")
    if geom_type == "Polygon":
        return [geometry]
    if geom_type == "MultiPolygon":
        return list(geometry.geoms)
    return []


def summarize_one(path: Path, args: argparse.Namespace) -> Optional[Dict[str, object]]:
    try:
        data = load_pickle_gz(path)
    except Exception as exc:
        return {
            "token": path.parent.name,
            "path": str(path),
            "status": f"load_error:{type(exc).__name__}",
        }

    candidates = as_candidates(data.get("trajectory_candidates"))
    if candidates is None:
        return {
            "token": token_from_data(data, path),
            "path": str(path),
            "status": "missing_candidates",
        }

    num_candidates = int(candidates.shape[0])
    mask = as_vector(data.get("trajectory_candidates_mask"), num_candidates, 1, dtype=bool)
    scores = as_vector(data.get("pdm_score_targets"), num_candidates, math.nan, dtype=np.float32)
    finite_scores = np.isfinite(scores)
    valid = mask.astype(bool) & finite_scores

    gt_score = safe_float(data.get("gt_pdm_score"))
    valid_scores = scores[valid]
    best_score = float(valid_scores.max()) if valid_scores.size else math.nan
    mean_score = float(valid_scores.mean()) if valid_scores.size else math.nan
    score_gain = best_score - gt_score if math.isfinite(best_score) and math.isfinite(gt_score) else math.nan
    margin = float(args.gt_margin)
    above_gt = int(((scores > gt_score + margin) & valid).sum()) if math.isfinite(gt_score) else 0
    high_quality = int(above_gt > 0)

    valid_candidates = candidates[valid]
    if valid_candidates.size:
        final_xy = valid_candidates[:, -1, :2]
        final_x = final_xy[:, 0]
        final_y = final_xy[:, 1]
        path_lengths = np.asarray([path_length_xy(traj) for traj in valid_candidates], dtype=np.float32)
        lane_change = np.abs(final_y) >= float(args.lane_change_y_thresh)
        left_change = final_y >= float(args.lane_change_y_thresh)
        right_change = final_y <= -float(args.lane_change_y_thresh)
        final_y_span = float(final_y.max() - final_y.min())
        final_x_span = float(final_x.max() - final_x.min())
        mean_path_length = float(path_lengths.mean())
        max_path_length = float(path_lengths.max())
    else:
        final_y_span = final_x_span = mean_path_length = max_path_length = math.nan
        lane_change = left_change = right_change = np.asarray([], dtype=bool)

    pair_ade, pair_fde = pairwise_diversity(candidates, valid)
    gt = as_trajectory(data.get("trajectory"))
    min_ade_to_gt, min_fde_to_gt = gt_coverage(candidates, valid, gt)

    best_idx = int(np.nanargmax(np.where(valid, scores, np.nan))) if valid.any() else -1

    components = as_components(data.get("pdm_score_components_candidates"), num_candidates)
    gt_components = as_component_vector(data.get("gt_pdm_components"))
    component_means: Dict[str, float] = {}
    component_mins: Dict[str, float] = {}
    if components is not None and valid.any():
        names = component_names(args, components.shape[-1])
        for comp_idx, name in enumerate(names):
            vals = components[valid, comp_idx]
            vals = vals[np.isfinite(vals)]
            component_means[f"component_mean_{name}"] = float(vals.mean()) if vals.size else math.nan
            component_mins[f"component_min_{name}"] = float(vals.min()) if vals.size else math.nan
    best_component_values: Dict[str, float] = {}
    if components is not None and best_idx >= 0:
        names = component_names(args, components.shape[-1])
        for comp_idx, name in enumerate(names):
            val = float(components[best_idx, comp_idx])
            best_component_values[f"best_component_{name}"] = val if math.isfinite(val) else math.nan
    gt_component_values: Dict[str, float] = {}
    if gt_components is not None:
        names = component_names(args, int(gt_components.shape[-1]))
        for comp_idx, name in enumerate(names):
            val = float(gt_components[comp_idx])
            gt_component_values[f"gt_component_{name}"] = val if math.isfinite(val) else math.nan

    reason = data.get("b2d_candidate_selection_reason")
    if hasattr(reason, "item"):
        try:
            reason = reason.item()
        except Exception:
            pass
    if isinstance(reason, bytes):
        reason = reason.decode("utf-8", errors="replace")
    if reason is None:
        reason = ""

    row: Dict[str, object] = {
        "token": token_from_data(data, path),
        "path": str(path),
        "status": "ok",
        "num_candidates": num_candidates,
        "valid_candidates": int(valid.sum()),
        "gt_score": gt_score,
        "best_candidate_score": best_score,
        "mean_candidate_score": mean_score,
        "best_minus_gt": score_gain,
        "num_above_gt": above_gt,
        "has_high_quality": high_quality,
        "best_candidate_rank": best_idx,
        "candidate_scores": format_float_list(scores[valid]),
        "lane_change_candidates": int(lane_change.sum()),
        "left_change_candidates": int(left_change.sum()),
        "right_change_candidates": int(right_change.sum()),
        "lane_change_rate": float(lane_change.mean()) if lane_change.size else math.nan,
        "final_y_span": final_y_span,
        "final_x_span": final_x_span,
        "mean_path_length": mean_path_length,
        "max_path_length": max_path_length,
        "pairwise_ade": pair_ade,
        "pairwise_fde": pair_fde,
        "min_ade_to_gt": min_ade_to_gt,
        "min_fde_to_gt": min_fde_to_gt,
        "b2d_adjacent_lane_occupancy_allowed": bool(np.squeeze(to_numpy(data.get("b2d_adjacent_lane_occupancy_allowed"))).item())
        if "b2d_adjacent_lane_occupancy_allowed" in data
        else "",
        "candidate_selection_reason": str(reason),
    }
    row.update(component_means)
    row.update(component_mins)
    row.update(best_component_values)
    row.update(gt_component_values)
    return row


def component_names(args: argparse.Namespace, count: int) -> List[str]:
    names = list(args.component_names or DEFAULT_COMPONENT_NAMES)
    if len(names) < count:
        names += [f"component_{idx}" for idx in range(len(names), count)]
    return names[:count]


def online_component_names(count: int) -> List[str]:
    names = list(ONLINE_PDM_COMPONENT_NAMES)
    if len(names) < count:
        names += [f"component_{idx}" for idx in range(len(names), count)]
    return names[:count]


def ensure_trajectory3(traj: np.ndarray) -> Optional[np.ndarray]:
    arr = np.asarray(traj, dtype=np.float32)
    arr = np.squeeze(arr)
    if arr.ndim != 2 or arr.shape[-1] < 2 or arr.shape[0] == 0:
        return None
    out = np.zeros((arr.shape[0], 3), dtype=np.float32)
    out[:, : min(3, arr.shape[-1])] = arr[:, : min(3, arr.shape[-1])]
    out[~np.isfinite(out)] = 0.0
    return out


def pdm_scores_from_components(components: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    comps = np.asarray(components, dtype=np.float32)
    if comps.shape[-1] < 5:
        return np.full(comps.shape[:-1], np.nan, dtype=np.float32)

    weighted = np.zeros(comps.shape[:-1], dtype=np.float32)
    denom = 0.0
    for comp_idx, weight in (
        (2, float(args.online_progress_weight)),
        (3, float(args.online_ttc_weight)),
        (4, float(args.online_comfort_weight)),
        (5, float(args.online_driving_direction_weight)),
    ):
        if weight <= 0.0:
            continue
        if comp_idx >= comps.shape[-1]:
            return np.full(comps.shape[:-1], np.nan, dtype=np.float32)
        weighted += comps[..., comp_idx] * weight
        denom += weight

    if denom <= 0.0:
        return np.full(comps.shape[:-1], np.nan, dtype=np.float32)

    scores = comps[..., 0] * comps[..., 1] * (weighted / denom)
    return np.where(np.isfinite(scores), scores, np.nan).astype(np.float32)


def stash_cached_gt_best_fields(row: Dict[str, object]) -> None:
    for key in ("gt_score", "best_candidate_score", "best_minus_gt", "has_high_quality"):
        if key in row and f"cached_{key}" not in row:
            row[f"cached_{key}"] = row[key]

    for key in list(row.keys()):
        if key.startswith("gt_component_") or key.startswith("best_component_"):
            cached_key = f"cached_{key}"
            if cached_key not in row:
                row[cached_key] = row[key]
            del row[key]


def choose_best_candidate_index(data: Dict[str, object], candidates: np.ndarray, row: Dict[str, object]) -> int:
    try:
        best_idx = int(row.get("best_candidate_rank", -1))
    except Exception:
        best_idx = -1
    if 0 <= best_idx < candidates.shape[0]:
        return best_idx

    num_candidates = int(candidates.shape[0])
    mask = as_vector(data.get("trajectory_candidates_mask"), num_candidates, 1, dtype=bool)
    scores = as_vector(data.get("pdm_score_targets"), num_candidates, math.nan, dtype=np.float32)
    valid = mask.astype(bool) & np.isfinite(scores)
    if not valid.any():
        return -1
    return int(np.nanargmax(np.where(valid, scores, np.nan)))


def apply_online_score_result(
    row: Dict[str, object],
    components: np.ndarray,
    scores: np.ndarray,
    best_idx: int,
    args: argparse.Namespace,
) -> bool:
    comps = np.asarray(components, dtype=np.float32)
    score_arr = np.asarray(scores, dtype=np.float32).reshape(-1)
    if comps.ndim != 2 or comps.shape[0] < 2 or score_arr.size < 2:
        row["online_score_status"] = "invalid_result_shape"
        return False

    gt_score = float(score_arr[0])
    best_score = float(score_arr[1])
    if not (math.isfinite(gt_score) and math.isfinite(best_score)):
        row["online_score_status"] = "nonfinite_score"
        return False

    stash_cached_gt_best_fields(row)
    margin = float(args.gt_margin)
    online_above_gt = int(best_score > gt_score + margin)
    row.update(
        {
            "online_score_status": "ok",
            "online_score_reference_progress": not bool(args.online_score_no_reference_progress),
            "online_best_candidate_rank": int(best_idx),
            "online_gt_pdm_score": gt_score,
            "online_best_candidate_pdm_score": best_score,
            "online_best_minus_gt": best_score - gt_score,
            "online_best_above_gt": online_above_gt,
            "gt_score": gt_score,
            "best_candidate_score": best_score,
            "best_minus_gt": best_score - gt_score,
            "has_high_quality": online_above_gt,
        }
    )

    names = online_component_names(comps.shape[-1])
    for comp_idx, name in enumerate(names):
        gt_val = float(comps[0, comp_idx])
        best_val = float(comps[1, comp_idx])
        row[f"online_gt_component_{name}"] = gt_val if math.isfinite(gt_val) else math.nan
        row[f"online_best_component_{name}"] = best_val if math.isfinite(best_val) else math.nan
        row[f"gt_component_{name}"] = gt_val if math.isfinite(gt_val) else math.nan
        row[f"best_component_{name}"] = best_val if math.isfinite(best_val) else math.nan
    return True


def online_score_gt_and_best(rows: List[Dict[str, object]], args: argparse.Namespace) -> Dict[str, int]:
    if not bool(args.online_score_best_gt):
        return {"enabled": 0}
    if not args.metric_cache_path:
        raise ValueError("--metric-cache-path is required when --online-score-best-gt is enabled")

    from navsim.agents.diffusiondrive.modules.pdm_supervision import (  # pylint: disable=import-outside-toplevel
        PDMScoreConfig,
        PDMSupervision,
    )

    pdm = PDMSupervision(
        PDMScoreConfig(
            cache_path=str(args.metric_cache_path),
            num_poses=int(args.online_pdm_num_poses),
            interval_length=float(args.online_pdm_interval),
            use_ray=bool(args.online_use_ray),
            ray_threads=int(args.online_ray_threads),
            cache_lru_size=int(args.online_cache_lru_size),
            progress_use_reference_baseline=not bool(args.online_score_no_reference_progress),
        )
    )

    counters = {
        "enabled": 1,
        "queued": 0,
        "scored": 0,
        "skipped": 0,
        "errors": 0,
    }
    batch_tokens: List[str] = []
    batch_trajs: List[np.ndarray] = []
    batch_rows: List[Dict[str, object]] = []
    batch_best_idx: List[int] = []
    batch_horizon = -1
    batch_size = max(int(args.online_score_batch_size), 1)

    def flush() -> None:
        nonlocal batch_tokens, batch_trajs, batch_rows, batch_best_idx, batch_horizon
        if not batch_tokens:
            return
        try:
            traj_batch = np.stack(batch_trajs, axis=0).astype(np.float32, copy=False)
            components = pdm.score_batch_components(batch_tokens, traj_batch)
            scores = pdm_scores_from_components(components, args)
            for idx, row in enumerate(batch_rows):
                if apply_online_score_result(row, components[idx], scores[idx], batch_best_idx[idx], args):
                    counters["scored"] += 1
                else:
                    counters["errors"] += 1
        except Exception as exc:  # pragma: no cover - depends on local metric cache/runtime
            counters["errors"] += len(batch_rows)
            for row in batch_rows:
                row["online_score_status"] = f"score_error:{type(exc).__name__}"
        batch_tokens = []
        batch_trajs = []
        batch_rows = []
        batch_best_idx = []
        batch_horizon = -1

    for row in tqdm(rows, desc="online scoring GT/best"):
        if row.get("status") != "ok" or int(row.get("valid_candidates", 0)) <= 0:
            row["online_score_status"] = "skip:no_valid_candidate"
            counters["skipped"] += 1
            continue

        try:
            data = load_pickle_gz(Path(str(row["path"])))
            candidates = as_candidates(data.get("trajectory_candidates"))
            gt = as_trajectory(data.get("trajectory"))
            if candidates is None or gt is None:
                row["online_score_status"] = "skip:missing_trajectory"
                counters["skipped"] += 1
                continue
            best_idx = choose_best_candidate_index(data, candidates, row)
            if best_idx < 0:
                row["online_score_status"] = "skip:no_best_candidate"
                counters["skipped"] += 1
                continue
            gt3 = ensure_trajectory3(gt)
            best3 = ensure_trajectory3(candidates[best_idx])
            if gt3 is None or best3 is None:
                row["online_score_status"] = "skip:invalid_trajectory"
                counters["skipped"] += 1
                continue
            horizon = min(gt3.shape[0], best3.shape[0])
            if horizon <= 0:
                row["online_score_status"] = "skip:empty_trajectory"
                counters["skipped"] += 1
                continue
            traj_pair = np.stack([gt3[:horizon], best3[:horizon]], axis=0)
        except Exception as exc:
            row["online_score_status"] = f"prepare_error:{type(exc).__name__}"
            counters["errors"] += 1
            continue

        if batch_tokens and (horizon != batch_horizon or len(batch_tokens) >= batch_size):
            flush()
        batch_horizon = horizon
        batch_tokens.append(str(row["token"]))
        batch_trajs.append(traj_pair)
        batch_rows.append(row)
        batch_best_idx.append(best_idx)
        counters["queued"] += 1

    flush()
    return counters


def aggregate(rows: List[Dict[str, object]]) -> Dict[str, object]:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    total = len(rows)
    with_candidates = len(ok_rows)
    valid_rows = [row for row in ok_rows if int(row.get("valid_candidates", 0)) > 0]

    summary: Dict[str, object] = {
        "total_files": total,
        "with_candidates": with_candidates,
        "without_candidates": total - with_candidates,
        "with_valid_candidates": len(valid_rows),
        "candidate_cache_coverage": with_candidates / max(total, 1),
        "valid_candidate_coverage": len(valid_rows) / max(total, 1),
    }

    scalar_keys = [
        "valid_candidates",
        "gt_score",
        "best_candidate_score",
        "mean_candidate_score",
        "best_minus_gt",
        "num_above_gt",
        "lane_change_rate",
        "final_y_span",
        "final_x_span",
        "mean_path_length",
        "pairwise_ade",
        "pairwise_fde",
        "min_ade_to_gt",
        "min_fde_to_gt",
        "online_gt_pdm_score",
        "online_best_candidate_pdm_score",
        "online_best_minus_gt",
        "online_best_above_gt",
    ]
    for key in scalar_keys:
        summary[key] = finite_quantiles([float(row.get(key, math.nan)) for row in ok_rows])

    summary["high_quality_sample_rate"] = (
        float(np.mean([int(row.get("has_high_quality", 0)) for row in ok_rows])) if ok_rows else math.nan
    )
    summary["mean_num_above_gt"] = (
        float(np.mean([int(row.get("num_above_gt", 0)) for row in ok_rows])) if ok_rows else math.nan
    )
    summary["empty_valid_candidate_rate"] = (
        float(np.mean([int(row.get("valid_candidates", 0)) == 0 for row in ok_rows])) if ok_rows else math.nan
    )
    online_rows = [row for row in ok_rows if row.get("online_score_status") == "ok"]
    if online_rows:
        summary["online_scored_samples"] = len(online_rows)
        summary["online_score_success_rate"] = len(online_rows) / max(len(ok_rows), 1)
        summary["online_best_above_gt_rate"] = float(
            np.mean([int(row.get("online_best_above_gt", 0)) for row in online_rows])
        )

    reason_counts: Dict[str, int] = {}
    for row in ok_rows:
        reason = str(row.get("candidate_selection_reason", ""))
        if not reason:
            continue
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    summary["candidate_selection_reason_counts"] = reason_counts

    for key in sorted({key for row in ok_rows for key in row if key.startswith("component_mean_")}):
        summary[key] = finite_quantiles([float(row.get(key, math.nan)) for row in ok_rows])
    for key in sorted({key for row in ok_rows for key in row if key.startswith("component_min_")}):
        summary[key] = finite_quantiles([float(row.get(key, math.nan)) for row in ok_rows])
    for key in sorted({key for row in ok_rows for key in row if key.startswith("best_component_")}):
        summary[key] = finite_quantiles([float(row.get(key, math.nan)) for row in ok_rows])
    for key in sorted({key for row in ok_rows for key in row if key.startswith("gt_component_")}):
        summary[key] = finite_quantiles([float(row.get(key, math.nan)) for row in ok_rows])

    return summary


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_csv(path: Path) -> List[Dict[str, object]]:
    with open(path, "r", newline="", encoding="utf-8") as f:
        return [dict(row) for row in csv.DictReader(f)]


def draw_box_on_bev(ax, box_state: np.ndarray, pixel_size: float, width: int, **kwargs) -> None:
    corners = box_corners_xy(box_state)
    row, col = xy_to_image(corners, pixel_size, width)
    ax.plot(col, row, **kwargs)


def draw_boxes_from_target(ax, data: Dict[str, object], pixel_size: float, width: int) -> None:
    if "agent_states" in data:
        states = np.squeeze(to_numpy(data["agent_states"])).astype(np.float32)
        if states.ndim == 2 and states.shape[-1] >= 5:
            labels = as_vector(data.get("agent_labels"), states.shape[0], 1, dtype=bool)
            for state, valid in zip(states, labels):
                if not bool(valid):
                    continue
                draw_box_on_bev(
                    ax,
                    state,
                    pixel_size,
                    width,
                    color="#334155",
                    linewidth=0.75,
                    alpha=0.55,
                    zorder=5,
                )
        return

    if "future_agent_obb" in data and "future_agent_mask" in data:
        obb = np.squeeze(to_numpy(data["future_agent_obb"])).astype(np.float32)
        mask = np.squeeze(to_numpy(data["future_agent_mask"])).astype(bool)
        if obb.ndim == 3 and mask.ndim == 2:
            for state, valid in zip(obb[:, 0], mask[:, 0]):
                if not bool(valid):
                    continue
                draw_box_on_bev(
                    ax,
                    state,
                    pixel_size,
                    width,
                    color="#334155",
                    linewidth=0.75,
                    alpha=0.50,
                    zorder=5,
                )
        return

    if "future_agent_boxes" in data and "future_agent_boxes_mask" in data:
        boxes = np.squeeze(to_numpy(data["future_agent_boxes"])).astype(np.float32)
        mask = np.squeeze(to_numpy(data["future_agent_boxes_mask"])).astype(bool)
        if boxes.ndim == 4 and mask.ndim == 2:
            for corners, valid in zip(boxes[:, 0], mask[:, 0]):
                if not bool(valid):
                    continue
                row, col = xy_to_image(np.concatenate([corners, corners[:1]], axis=0), pixel_size, width)
                ax.plot(col, row, color="#334155", linewidth=0.75, alpha=0.50, zorder=5)


def draw_metric_cache_map_base(ax, metric_cache: object) -> None:
    drivable_map = getattr(metric_cache, "drivable_area_map", None)
    if drivable_map is None:
        return
    route_lane_ids = set(getattr(metric_cache, "route_lane_ids", []) or [])
    for token, layer, geom in zip(
        getattr(drivable_map, "tokens", []),
        getattr(drivable_map, "map_types", []),
        getattr(drivable_map, "_geometries", []),
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

        for poly in iter_polygons(geom):
            x, y = poly.exterior.xy
            ax.fill(x, y, facecolor=face_color, edgecolor=edge_color, alpha=alpha, linewidth=lw, zorder=1)

    centerline = getattr(getattr(metric_cache, "centerline", None), "linestring", None)
    if centerline is not None:
        x, y = centerline.xy
        ax.plot(x, y, color=MAP_CENTERLINE, linewidth=1.4, alpha=0.95, linestyle="--", label="Route centerline", zorder=4)


def draw_metric_cache_obstacles(ax, metric_cache: object) -> None:
    try:
        occupancy = metric_cache.observation[0]
    except Exception:
        return
    for token in getattr(occupancy, "tokens", []):
        if "red_light" in str(token):
            continue
        try:
            geom = occupancy[token]
        except Exception:
            continue
        for poly in iter_polygons(geom):
            x, y = poly.exterior.xy
            ax.fill(
                x,
                y,
                facecolor=MAP_AGENT_BOX,
                edgecolor=MAP_AGENT_BOX,
                alpha=0.26,
                linewidth=0.65,
                zorder=5,
            )


def draw_metric_cache_ego_box(ax, metric_cache: object) -> None:
    footprint = getattr(getattr(metric_cache, "ego_state", None), "car_footprint", None)
    geometry = getattr(footprint, "geometry", None)
    drew = False
    for poly in iter_polygons(geometry):
        x, y = poly.exterior.xy
        ax.fill(
            x,
            y,
            facecolor=EGO_BOX_COLOR,
            edgecolor=EGO_BOX_COLOR,
            alpha=0.32,
            linewidth=1.1,
            label="Ego footprint",
            zorder=12,
        )
        drew = True
        break
    if drew:
        return
    ego = getattr(getattr(metric_cache, "ego_state", None), "rear_axle", None)
    if ego is None:
        return
    box = np.array([0.0, 0.0, 0.0, 4.8, 2.0], dtype=np.float32)
    corners = transform_local_to_global(box_corners_xy(box), np.array([ego.x, ego.y, ego.heading], dtype=np.float32))
    ax.fill(
        corners[:, 0],
        corners[:, 1],
        facecolor=EGO_BOX_COLOR,
        edgecolor=EGO_BOX_COLOR,
        alpha=0.32,
        linewidth=1.1,
        label="Ego footprint",
        zorder=12,
    )


def plot_candidate_vector_map_sample(
    target_path: Path,
    row: Dict[str, object],
    metric_cache_path: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> Optional[Path]:
    try:
        data = load_pickle_gz(target_path)
        metric_cache = load_lzma_pickle(metric_cache_path)
    except Exception as exc:
        print(f"skip vector map vis {target_path}: {type(exc).__name__}: {exc}")
        return None
    candidates = as_candidates(data.get("trajectory_candidates"))
    gt = as_trajectory(data.get("trajectory"))
    if candidates is None:
        return None

    num_candidates = int(candidates.shape[0])
    mask = as_vector(data.get("trajectory_candidates_mask"), num_candidates, 1, dtype=bool)
    scores = as_vector(data.get("pdm_score_targets"), num_candidates, math.nan, dtype=np.float32)
    valid = mask.astype(bool) & np.isfinite(scores)
    best_idx = int(np.nanargmax(np.where(valid, scores, np.nan))) if valid.any() else -1
    display_best_score = float(row.get("best_candidate_score", scores[best_idx] if best_idx >= 0 else math.nan))
    display_gt_score = float(row.get("gt_score", safe_float(data.get("gt_pdm_score"))))

    ego = metric_cache.ego_state.rear_axle
    ego_pose = np.array([float(ego.x), float(ego.y), float(ego.heading)], dtype=np.float32)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    set_matplotlib_style(plt)
    fig, ax = plt.subplots(1, 1, figsize=(6.0, 6.0), dpi=int(args.map_vis_dpi))
    fig.patch.set_facecolor(MAP_BASE_FACE)
    ax.set_facecolor(MAP_PANEL_FACE)

    draw_metric_cache_map_base(ax, metric_cache)
    draw_metric_cache_obstacles(ax, metric_cache)
    draw_metric_cache_ego_box(ax, metric_cache)

    if str(args.map_vis_candidates) in ("all", "valid"):
        for idx, traj in enumerate(candidates):
            if str(args.map_vis_candidates) == "valid" and not bool(valid[idx]):
                continue
            if idx == best_idx:
                continue
            global_traj = transform_local_to_global(traj, ego_pose)
            ax.plot(
                global_traj[:, 0],
                global_traj[:, 1],
                color=CANDIDATE_COLOR,
                linewidth=1.15 if bool(valid[idx]) else 0.8,
                alpha=0.34 if bool(valid[idx]) else 0.14,
                solid_capstyle="round",
                zorder=8,
            )

    if best_idx >= 0:
        best_global = transform_local_to_global(candidates[best_idx], ego_pose)
        ax.plot(
            best_global[:, 0],
            best_global[:, 1],
            color=BEST_CANDIDATE_COLOR,
            linewidth=2.8,
            alpha=0.98,
            solid_capstyle="round",
            label=f"Best candidate ({display_best_score:.3f})",
            zorder=11,
        )
        ax.scatter(best_global[-1, 0], best_global[-1, 1], s=38, color=BEST_CANDIDATE_COLOR, marker="x", linewidths=1.6, zorder=12)

    if gt is not None:
        gt_global = transform_local_to_global(gt, ego_pose)
        ax.plot(
            gt_global[:, 0],
            gt_global[:, 1],
            color=GT_COLOR,
            linewidth=2.8,
            alpha=0.98,
            solid_capstyle="round",
            label=f"GT ({display_gt_score:.3f})",
            zorder=10,
        )
        ax.scatter(gt_global[-1, 0], gt_global[-1, 1], s=40, color=GT_COLOR, marker="x", linewidths=1.6, zorder=12)

    radius = float(args.map_vis_radius)
    ax.set_xlim(ego_pose[0] - radius, ego_pose[0] + radius)
    ax.set_ylim(ego_pose[1] - radius, ego_pose[1] + radius)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)
    ax.tick_params(labelsize=8, colors="#475569")
    for spine in ax.spines.values():
        spine.set_color(MAP_SPINE)
        spine.set_linewidth(0.8)

    token = str(row.get("token", target_path.parent.name))
    ax.set_title(
        f"{token}\nGT={display_gt_score:.3f}, Best={display_best_score:.3f}, "
        f"Gain={float(row.get('best_minus_gt', math.nan)):.3f}",
        fontsize=10,
        color=DARK_COLOR,
    )
    ax.legend(loc="upper right", fontsize=7, frameon=True, framealpha=0.92, edgecolor=MAP_SPINE)
    fig.tight_layout(pad=0.5)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_figure(
        fig,
        output_dir / f"{safe_name(token)}__map_best{best_idx}",
        args.plot_formats,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)
    return out_path


def hex_to_rgb01(color: str) -> np.ndarray:
    text = str(color).strip().lstrip("#")
    if len(text) != 6:
        return np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
    return np.asarray([int(text[idx : idx + 2], 16) / 255.0 for idx in (0, 2, 4)], dtype=np.float32)


def bool_mask_from_target(data: Dict[str, object], key: str) -> Optional[np.ndarray]:
    if key not in data:
        return None
    mask = np.squeeze(to_numpy(data.get(key))).astype(bool)
    if mask.ndim != 2:
        return None
    return mask


def overlay_bool_mask(ax, mask: Optional[np.ndarray], color: str, alpha: float, zorder: int) -> None:
    if mask is None:
        return
    rgba = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.float32)
    rgba[mask, :3] = hex_to_rgb01(color)
    rgba[mask, 3] = float(alpha)
    ax.imshow(rgba, origin="upper", interpolation="nearest", zorder=zorder)


def overlay_feasible_area(ax, data: Dict[str, object]) -> None:
    overlay_bool_mask(ax, bool_mask_from_target(data, "feasible_area_mask"), FEASIBLE_OVERLAY_COLOR, 0.16, 2)


def select_rows_for_map_vis(rows: List[Dict[str, object]], args: argparse.Namespace) -> List[Dict[str, object]]:
    ok_rows = [
        row
        for row in rows
        if row.get("status") == "ok"
        and int(row.get("valid_candidates", 0)) > 0
        and isinstance(row.get("path"), str)
    ]
    if not ok_rows:
        return []
    sort_key = str(args.map_vis_sort)
    if sort_key == "best_score":
        ok_rows.sort(key=lambda row: float(row.get("best_candidate_score", -np.inf)), reverse=True)
    elif sort_key == "worst_gt":
        ok_rows.sort(key=lambda row: float(row.get("gt_score", np.inf)))
    elif sort_key == "first":
        pass
    else:
        ok_rows.sort(key=lambda row: float(row.get("best_minus_gt", -np.inf)), reverse=True)
    return ok_rows[: max(0, int(args.map_vis_max))]


def plot_candidate_map_sample(
    target_path: Path,
    row: Dict[str, object],
    output_dir: Path,
    args: argparse.Namespace,
) -> Optional[Path]:
    try:
        data = load_pickle_gz(target_path)
    except Exception as exc:
        print(f"skip map vis {target_path}: {type(exc).__name__}: {exc}")
        return None
    bev_key = str(args.map_vis_bev_key)
    if bev_key not in data or "trajectory_candidates" not in data:
        return None
    try:
        class_map = bev_to_class_map(data[bev_key])
        candidates = as_candidates(data.get("trajectory_candidates"))
        gt = as_trajectory(data.get("trajectory"))
    except Exception as exc:
        print(f"skip map vis {target_path}: {type(exc).__name__}: {exc}")
        return None
    if candidates is None:
        return None

    num_candidates = int(candidates.shape[0])
    mask = as_vector(data.get("trajectory_candidates_mask"), num_candidates, 1, dtype=bool)
    scores = as_vector(data.get("pdm_score_targets"), num_candidates, math.nan, dtype=np.float32)
    valid = mask.astype(bool) & np.isfinite(scores)
    best_idx = int(np.nanargmax(np.where(valid, scores, np.nan))) if valid.any() else -1
    display_best_score = float(row.get("best_candidate_score", scores[best_idx] if best_idx >= 0 else math.nan))
    display_gt_score = float(row.get("gt_score", safe_float(data.get("gt_pdm_score"))))
    height, width = class_map.shape
    pixel_size = float(args.map_vis_pixel_size)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    set_matplotlib_style(plt)
    fig, ax = plt.subplots(1, 1, figsize=(6.0, 6.0), dpi=int(args.map_vis_dpi))
    fig.patch.set_facecolor("white")
    ax.set_facecolor(MAP_PANEL_FACE)
    cmap = ListedColormap(SEMANTIC_COLORS)
    ax.imshow(
        class_map,
        origin="upper",
        interpolation="nearest",
        cmap=cmap,
        vmin=0,
        vmax=len(SEMANTIC_COLORS) - 1,
        alpha=0.82,
        zorder=1,
    )
    overlay_feasible_area(ax, data)
    draw_boxes_from_target(ax, data, pixel_size=pixel_size, width=width)
    draw_box_on_bev(
        ax,
        np.array([0.0, 0.0, 0.0, 4.8, 2.0], dtype=np.float32),
        pixel_size,
        width,
        color=EGO_BOX_COLOR,
        linewidth=1.2,
        alpha=0.95,
        zorder=12,
    )

    if str(args.map_vis_candidates) in ("all", "valid"):
        for idx, traj in enumerate(candidates):
            if str(args.map_vis_candidates) == "valid" and not bool(valid[idx]):
                continue
            if idx == best_idx:
                continue
            alpha = 0.34 if bool(valid[idx]) else 0.16
            linewidth = 1.15 if bool(valid[idx]) else 0.8
            row_px, col_px = xy_to_image(traj[:, :2], pixel_size, width)
            ax.plot(
                col_px,
                row_px,
                color=CANDIDATE_COLOR,
                linewidth=linewidth,
                alpha=alpha,
                solid_capstyle="round",
                zorder=8,
            )

    if best_idx >= 0:
        best_traj = candidates[best_idx]
        row_px, col_px = xy_to_image(best_traj[:, :2], pixel_size, width)
        ax.plot(
            col_px,
            row_px,
            color=BEST_CANDIDATE_COLOR,
            linewidth=2.8,
            alpha=0.98,
            solid_capstyle="round",
            label=f"Best candidate ({display_best_score:.3f})",
            zorder=11,
        )
        ax.scatter(col_px[-1], row_px[-1], s=38, color=BEST_CANDIDATE_COLOR, marker="x", linewidths=1.6, zorder=12)

    if gt is not None:
        row_px, col_px = xy_to_image(gt[:, :2], pixel_size, width)
        ax.plot(
            col_px,
            row_px,
            color=GT_COLOR,
            linewidth=2.8,
            alpha=0.98,
            solid_capstyle="round",
            label=f"GT ({display_gt_score:.3f})",
            zorder=10,
        )
        ax.scatter(col_px[-1], row_px[-1], s=40, color=GT_COLOR, marker="x", linewidths=1.6, zorder=12)

    ax.scatter([(width - 1) / 2.0], [0], s=48, color="white", marker="x", linewidths=1.4, zorder=13)
    ax.set_xlim([-0.5, width - 0.5])
    ax.set_ylim([height - 0.5, -0.5])
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(labelsize=8, colors="#475569")
    for spine in ax.spines.values():
        spine.set_color(MAP_SPINE)
        spine.set_linewidth(0.8)
    token = str(row.get("token", target_path.parent.name))
    ax.set_title(
        f"{token}\nGT={display_gt_score:.3f}, Best={display_best_score:.3f}, "
        f"Gain={float(row.get('best_minus_gt', math.nan)):.3f}",
        fontsize=10,
        color=DARK_COLOR,
    )
    ax.legend(loc="lower left", fontsize=8, frameon=True, framealpha=0.90, edgecolor=MAP_SPINE)
    fig.tight_layout(pad=0.5)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_figure(
        fig,
        output_dir / f"{safe_name(token)}__best{best_idx}",
        args.plot_formats,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)
    return out_path


def draw_feasible_vis_base(
    ax,
    class_map: Optional[np.ndarray],
    data: Dict[str, object],
    height: int,
    width: int,
    pixel_size: float,
) -> None:
    ax.set_facecolor(MAP_PANEL_FACE)
    if class_map is not None:
        from matplotlib.colors import ListedColormap

        ax.imshow(
            class_map,
            origin="upper",
            interpolation="nearest",
            cmap=ListedColormap(SEMANTIC_COLORS),
            vmin=0,
            vmax=len(SEMANTIC_COLORS) - 1,
            alpha=0.78,
            zorder=1,
        )
    else:
        ax.imshow(
            np.zeros((height, width), dtype=np.float32),
            origin="upper",
            interpolation="nearest",
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
            alpha=0.0,
            zorder=1,
        )
    overlay_bool_mask(ax, bool_mask_from_target(data, "feasible_area_mask"), FEASIBLE_OVERLAY_COLOR, 0.30, 3)
    overlay_bool_mask(ax, bool_mask_from_target(data, "feasible_lane_mask"), "#22c55e", 0.42, 4)
    draw_boxes_from_target(ax, data, pixel_size=pixel_size, width=width)
    draw_box_on_bev(
        ax,
        np.array([0.0, 0.0, 0.0, 4.8, 2.0], dtype=np.float32),
        pixel_size,
        width,
        color=EGO_BOX_COLOR,
        linewidth=1.2,
        alpha=0.95,
        zorder=12,
    )
    ax.scatter([(width - 1) / 2.0], [0], s=42, color="white", marker="x", linewidths=1.3, zorder=13)
    ax.set_xlim([-0.5, width - 0.5])
    ax.set_ylim([height - 0.5, -0.5])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)
    ax.set_axis_off()


def choose_candidate_indices_for_points(
    data: Dict[str, object],
    row: Dict[str, object],
    num_candidates: int,
    args: argparse.Namespace,
) -> np.ndarray:
    mask = as_vector(data.get("trajectory_candidates_mask"), num_candidates, 1, dtype=bool)
    scores = as_vector(data.get("pdm_score_targets"), num_candidates, math.nan, dtype=np.float32)
    finite_scores = np.isfinite(scores)
    valid = mask.astype(bool) & finite_scores if finite_scores.any() else mask.astype(bool)
    source = str(args.feasible_vis_point_source)
    if source == "all":
        return np.arange(num_candidates, dtype=np.int64)
    if source == "best":
        best_idx = int(np.nanargmax(np.where(valid, scores, np.nan))) if valid.any() and finite_scores.any() else -1
        if best_idx < 0:
            try:
                best_idx = int(row.get("best_candidate_rank", -1))
            except Exception:
                best_idx = -1
        return np.asarray([best_idx], dtype=np.int64) if 0 <= best_idx < num_candidates else np.asarray([], dtype=np.int64)
    return np.where(valid)[0].astype(np.int64)


def plot_feasible_point_sample(
    target_path: Path,
    row: Dict[str, object],
    output_dir: Path,
    args: argparse.Namespace,
) -> Optional[Path]:
    try:
        data = load_pickle_gz(target_path)
        candidates = as_candidates(data.get("trajectory_candidates"))
        gt = as_trajectory(data.get("trajectory"))
    except Exception as exc:
        print(f"skip feasible vis {target_path}: {type(exc).__name__}: {exc}")
        return None
    if candidates is None:
        return None

    class_map: Optional[np.ndarray] = None
    bev_key = str(args.map_vis_bev_key)
    if bev_key in data:
        try:
            class_map = bev_to_class_map(data[bev_key])
        except Exception:
            class_map = None
    feasible_mask = bool_mask_from_target(data, "feasible_area_mask")
    if class_map is None and feasible_mask is None:
        return None
    if class_map is not None:
        height, width = class_map.shape
    else:
        assert feasible_mask is not None
        height, width = feasible_mask.shape

    candidate_indices = choose_candidate_indices_for_points(data, row, int(candidates.shape[0]), args)
    if candidate_indices.size == 0:
        return None
    selected = candidates[candidate_indices, :, :2]
    points = selected.reshape(-1, 2)
    steps = np.tile(np.arange(selected.shape[1], dtype=np.float32), selected.shape[0])
    max_points = int(getattr(args, "feasible_vis_max_points", 0))
    if max_points > 0 and points.shape[0] > max_points:
        keep = np.linspace(0, points.shape[0] - 1, max_points).astype(np.int64)
        points = points[keep]
        steps = steps[keep]

    pixel_size = float(args.map_vis_pixel_size)
    row_px, col_px = xy_to_image(points, pixel_size, width)
    in_bounds = (row_px >= 0) & (row_px < height) & (col_px >= 0) & (col_px < width)
    row_px = row_px[in_bounds]
    col_px = col_px[in_bounds]
    steps = steps[in_bounds]
    if row_px.size == 0:
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    set_matplotlib_style(plt)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.2), dpi=int(args.map_vis_dpi))
    fig.patch.set_facecolor("white")
    for ax in axes:
        draw_feasible_vis_base(ax, class_map, data, height, width, pixel_size)

    axes[0].set_title("Feasible Region", fontsize=11, color=DARK_COLOR, pad=4)
    axes[1].set_title("Sampled Trajectory Points", fontsize=11, color=DARK_COLOR, pad=4)
    scatter = axes[1].scatter(
        col_px,
        row_px,
        c=steps,
        cmap="viridis",
        s=float(args.feasible_vis_point_size),
        alpha=float(args.feasible_vis_point_alpha),
        edgecolors="none",
        zorder=14,
    )
    if bool(args.feasible_vis_draw_gt) and gt is not None:
        gt_row, gt_col = xy_to_image(gt[:, :2], pixel_size, width)
        gt_in_bounds = (gt_row >= 0) & (gt_row < height) & (gt_col >= 0) & (gt_col < width)
        axes[1].scatter(
            gt_col[gt_in_bounds],
            gt_row[gt_in_bounds],
            s=float(args.feasible_vis_point_size) * 1.15,
            color=GT_COLOR,
            marker="x",
            linewidths=1.1,
            alpha=0.92,
            zorder=15,
        )
    cbar = fig.colorbar(scatter, ax=axes[1], fraction=0.046, pad=0.02)
    cbar.set_label("Trajectory step", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    token = str(row.get("token", target_path.parent.name))
    fig.suptitle(token, fontsize=10, color=DARK_COLOR, y=0.985)
    fig.tight_layout(pad=0.5)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_figure(
        fig,
        output_dir / f"{safe_name(token)}__feasible_points",
        args.plot_formats,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)
    return out_path


def maybe_write_map_visualizations(
    out_dir: Path,
    rows: List[Dict[str, object]],
    args: argparse.Namespace,
    metric_cache_index: Optional[Dict[str, str]] = None,
) -> None:
    selected = select_rows_for_map_vis(rows, args)
    if not selected:
        return
    vis_dir = out_dir / "candidate_map_visualizations"
    manifest_rows: List[Dict[str, object]] = []
    for row in tqdm(selected, desc="candidate map vis"):
        path = Path(str(row["path"]))
        token = str(row.get("token", path.parent.name))
        metric_path = None
        if metric_cache_index:
            metric_path_str = metric_cache_index.get(token) or metric_cache_index.get(path.parent.name)
            if metric_path_str:
                metric_path = Path(metric_path_str)
        if metric_path is not None:
            out_path = plot_candidate_vector_map_sample(path, row, metric_path, vis_dir, args)
        else:
            if bool(getattr(args, "map_vis_require_metric_cache", False)):
                continue
            out_path = plot_candidate_map_sample(path, row, vis_dir, args)
        if out_path is None:
            continue
        manifest_rows.append(
            {
                "token": row.get("token", ""),
                "source_path": str(path),
                "image_path": str(out_path),
                "gt_score": row.get("gt_score", ""),
                "best_candidate_score": row.get("best_candidate_score", ""),
                "best_minus_gt": row.get("best_minus_gt", ""),
                "best_candidate_rank": row.get("best_candidate_rank", ""),
            }
        )
    if manifest_rows:
        write_csv(vis_dir / "manifest.csv", manifest_rows)


def select_rows_for_feasible_vis(rows: List[Dict[str, object]], args: argparse.Namespace) -> List[Dict[str, object]]:
    ok_rows = [
        row
        for row in rows
        if row.get("status") == "ok"
        and int(row.get("valid_candidates", 0)) > 0
        and isinstance(row.get("path"), str)
    ]
    if not ok_rows:
        return []
    sort_key = str(args.map_vis_sort)
    if sort_key == "best_score":
        ok_rows.sort(key=lambda row: float(row.get("best_candidate_score", -np.inf)), reverse=True)
    elif sort_key == "worst_gt":
        ok_rows.sort(key=lambda row: float(row.get("gt_score", np.inf)))
    elif sort_key == "first":
        pass
    else:
        ok_rows.sort(key=lambda row: float(row.get("best_minus_gt", -np.inf)), reverse=True)
    return ok_rows[: max(0, int(args.feasible_vis_max))]


def maybe_write_feasible_point_visualizations(
    out_dir: Path,
    rows: List[Dict[str, object]],
    args: argparse.Namespace,
) -> None:
    selected = select_rows_for_feasible_vis(rows, args)
    if not selected:
        return
    vis_dir = out_dir / "feasible_point_visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: List[Dict[str, object]] = []
    for row in tqdm(selected, desc="writing feasible point visualizations"):
        path = Path(str(row["path"]))
        out_path = plot_feasible_point_sample(path, row, vis_dir, args)
        if out_path is None:
            continue
        manifest_rows.append(
            {
                "token": row.get("token", ""),
                "source_path": str(path),
                "image_path": str(out_path),
                "point_source": str(args.feasible_vis_point_source),
                "valid_candidates": row.get("valid_candidates", ""),
                "gt_score": row.get("gt_score", ""),
                "best_candidate_score": row.get("best_candidate_score", ""),
                "best_minus_gt": row.get("best_minus_gt", ""),
            }
        )
    if manifest_rows:
        write_csv(vis_dir / "manifest.csv", manifest_rows)


def maybe_write_plots(out_dir: Path, rows: List[Dict[str, object]], args: argparse.Namespace) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"skip plots: matplotlib unavailable: {exc}")
        return

    set_matplotlib_style(plt)
    plot_formats = normalize_plot_formats(args.plot_formats)
    score_distribution_formats = extend_plot_formats(plot_formats, ["svg"])
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    candidate_scores = np.asarray(
        [score for row in ok_rows for score in parse_float_list(row.get("candidate_scores", ""))],
        dtype=np.float64,
    )
    candidate_scores = candidate_scores[np.isfinite(candidate_scores)]
    if candidate_scores.size:
        fig = plt.figure(figsize=(4.2, 4.2), dpi=180)
        plt.hist(candidate_scores, bins=50, color=CANDIDATE_COLOR, alpha=0.86)
        plt.xlabel("Candidate PDM score")
        plt.ylabel("Count")
        plt.title("Candidate Score Distribution")
        plt.tight_layout()
        save_figure(fig, out_dir / "candidate_score_distribution", score_distribution_formats)
        plt.close()

    gt_scores = np.asarray([float(row.get("gt_score", math.nan)) for row in ok_rows], dtype=np.float64)
    gt_scores = gt_scores[np.isfinite(gt_scores)]
    if gt_scores.size:
        fig = plt.figure(figsize=(4.2, 4.2), dpi=180)
        plt.hist(gt_scores, bins=50, color=GT_COLOR, alpha=0.86)
        plt.xlabel("GT PDM score")
        plt.ylabel("Count")
        plt.title("GT Score Distribution")
        plt.tight_layout()
        save_figure(fig, out_dir / "gt_score_distribution", score_distribution_formats)
        plt.close()

    best_scores = np.asarray([float(row.get("best_candidate_score", math.nan)) for row in ok_rows], dtype=np.float64)
    best_scores = best_scores[np.isfinite(best_scores)]
    if best_scores.size:
        fig = plt.figure(figsize=(4.2, 4.2), dpi=180)
        plt.hist(best_scores, bins=50, color=CANDIDATE_COLOR, alpha=0.86)
        plt.xlabel("Best candidate PDM score")
        plt.ylabel("Count")
        plt.title("Best Candidate Score Distribution")
        plt.tight_layout()
        save_figure(fig, out_dir / "best_candidate_score_distribution", plot_formats)
        plt.close()

    try:
        bins = parse_score_bins(args.score_bins)
    except ValueError as exc:
        print(f"skip score bin distribution plot: {exc}")
        bins = np.asarray([], dtype=np.float64)
    if bins.size >= 2:
        distribution_summary, _ = build_score_distributions(rows, bins)
        source_bins = distribution_summary["sources"]
        labels = [item["bin_label"] for item in source_bins["candidate"]["bins"]]
        if labels and any(source_bins[name]["total"] > 0 for name in ("candidate", "best_candidate", "gt")):
            x = np.arange(len(labels), dtype=np.float32)
            width = 0.26
            fig_width = max(7.2, 0.68 * len(labels))
            fig, ax = plt.subplots(1, 1, figsize=(fig_width, 3.9), dpi=180)
            specs = [
                ("candidate", "All candidates", CANDIDATE_COLOR, -width),
                ("best_candidate", "Best candidate", BEST_CANDIDATE_COLOR, 0.0),
                ("gt", "GT", GT_COLOR, width),
            ]
            for source_key, label, color, offset in specs:
                fractions = [float(item["fraction"]) for item in source_bins[source_key]["bins"]]
                ax.bar(x + offset, fractions, width=width, color=color, alpha=0.86, label=label)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=35, ha="right")
            ax.set_xlabel("PDM score bin")
            ax.set_ylabel("Fraction")
            ax.set_title("Score Bin Distribution")
            ax.grid(axis="y", color="#cbd5e1", linewidth=0.7, alpha=0.65)
            ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.01))
            plt.tight_layout()
            save_figure(fig, out_dir / "score_bin_distribution", plot_formats)
            plt.close()

    metric_labels = [display_metric_label("score")]
    gt_means = [float(np.nanmean(gt_scores)) if gt_scores.size else math.nan]
    cand_score_values = np.asarray(
        [float(row.get("best_candidate_score", math.nan)) for row in ok_rows],
        dtype=np.float64,
    )
    cand_score_values = cand_score_values[np.isfinite(cand_score_values)]
    cand_means = [float(np.nanmean(cand_score_values)) if cand_score_values.size else math.nan]

    component_keys = sorted(
        {
            key.replace("best_component_", "")
            for row in ok_rows
            for key in row
            if key.startswith("best_component_")
        }
        & {
            key.replace("gt_component_", "")
            for row in ok_rows
            for key in row
            if key.startswith("gt_component_")
        }
    )
    component_keys = sorted(component_keys, key=component_sort_key)
    if not component_keys:
        has_best = any(any(key.startswith("best_component_") for key in row) for row in ok_rows)
        has_gt = any(any(key.startswith("gt_component_") for key in row) for row in ok_rows)
        print(
            "component comparison: only PDM score will be plotted; "
            f"best_candidate_components={'yes' if has_best else 'no'}, "
            f"gt_components={'yes' if has_gt else 'no'}. "
            "Rebuild/patch candidates with --store-components to plot NC/DAC/EP/TTC/C/DDC."
        )
    for name in component_keys:
        gt_vals = np.asarray([float(row.get(f"gt_component_{name}", math.nan)) for row in ok_rows], dtype=np.float64)
        cand_vals = np.asarray([float(row.get(f"best_component_{name}", math.nan)) for row in ok_rows], dtype=np.float64)
        gt_vals = gt_vals[np.isfinite(gt_vals)]
        cand_vals = cand_vals[np.isfinite(cand_vals)]
        if gt_vals.size == 0 or cand_vals.size == 0:
            continue
        metric_labels.append(display_metric_label(name))
        gt_means.append(float(gt_vals.mean()))
        cand_means.append(float(cand_vals.mean()))

    if len(metric_labels) > 1 or all(math.isfinite(v) for v in gt_means + cand_means):
        x = np.arange(len(metric_labels), dtype=np.float32)
        fig_width = max(8.5, 1.05 * len(metric_labels))
        fig, ax = plt.subplots(1, 1, figsize=(fig_width, 3.8), dpi=180)
        gt_arr = np.asarray(gt_means, dtype=np.float64)
        cand_arr = np.asarray(cand_means, dtype=np.float64)
        finite_pair = np.isfinite(gt_arr) & np.isfinite(cand_arr)
        finite_vals = np.concatenate([gt_arr[np.isfinite(gt_arr)], cand_arr[np.isfinite(cand_arr)]])
        y_max = max(1.0, float(finite_vals.max()) * 1.08) if finite_vals.size else 1.0
        high_score_threshold = 0.8
        high_band_top = max(y_max, high_score_threshold + 0.05)
        ax.axhspan(high_score_threshold, high_band_top, color="#dbeafe", alpha=0.36, zorder=0)
        ax.axhline(
            high_score_threshold,
            color="#334155",
            linestyle="--",
            linewidth=1.0,
            alpha=0.8,
            zorder=1,
        )
        if finite_pair.any():
            ax.fill_between(
                x,
                gt_arr,
                cand_arr,
                where=finite_pair,
                color="#38bdf8",
                alpha=0.22,
                interpolate=True,
                label="Gap",
                zorder=2,
            )
        ax.plot(
            x,
            gt_arr,
            color=GT_COLOR,
            linewidth=2.4,
            marker="o",
            markersize=5.5,
            label="GT mean",
            zorder=4,
        )
        ax.plot(
            x,
            cand_arr,
            color=CANDIDATE_COLOR,
            linewidth=2.4,
            marker="o",
            markersize=5.5,
            label="Best candidate mean",
            zorder=5,
        )
        for idx, (gt_val, cand_val) in enumerate(zip(gt_arr, cand_arr)):
            if not (math.isfinite(float(gt_val)) and math.isfinite(float(cand_val))):
                continue
            if max(float(gt_val), float(cand_val)) < high_score_threshold:
                continue
            delta = float(cand_val - gt_val)
            label_y = min(high_band_top - 0.015, max(float(gt_val), float(cand_val)) + 0.025)
            ax.annotate(
                f"{delta:+.3f}",
                xy=(idx, label_y),
                xytext=(0, 0),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                color=DARK_COLOR,
                zorder=6,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels, rotation=0, ha="center")
        ax.set_ylim(high_score_threshold, high_band_top)
        ax.set_ylabel("Score")
        ax.set_xlabel("PDM metric")
        ax.set_title("Mean PDM Metrics")
        ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.01))
        ax.grid(axis="y", color="#cbd5e1", linewidth=0.7, alpha=0.65)
        break_kwargs = dict(transform=ax.transAxes, color=DARK_COLOR, clip_on=False, linewidth=0.9)
        ax.plot((-0.012, 0.012), (-0.018, 0.018), **break_kwargs)
        ax.plot((0.988, 1.012), (-0.018, 0.018), **break_kwargs)
        plt.tight_layout()
        save_figure(fig, out_dir / "pdm_metrics_gt_vs_best_candidate", plot_formats)
        plt.close()


def load_json_file(path: Path) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def stats_mean(value: object) -> float:
    if not isinstance(value, dict):
        return math.nan
    try:
        val = float(value.get("mean", math.nan))
    except Exception:
        return math.nan
    return val if math.isfinite(val) else math.nan


def plot_single_score_distribution_from_json(
    plt,
    out_dir: Path,
    distribution_summary: Dict[str, object],
    source_key: str,
    filename: str,
    color: str,
    xlabel: str,
    formats: Sequence[str],
) -> None:
    sources = distribution_summary.get("sources", {})
    if not isinstance(sources, dict) or source_key not in sources:
        return
    source = sources[source_key]
    if not isinstance(source, dict):
        return
    bins = source.get("bins", [])
    if not isinstance(bins, list) or not bins:
        return
    labels = [str(item.get("bin_label", "")) for item in bins if isinstance(item, dict)]
    counts = [float(item.get("count", 0.0)) for item in bins if isinstance(item, dict)]
    if not labels or not any(count > 0 for count in counts):
        return

    x = np.arange(len(labels), dtype=np.float32)
    fig_width = max(4.6, 0.46 * len(labels))
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, 4.2), dpi=180)
    ax.bar(x, counts, width=0.82, color=color, alpha=0.86)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.set_title(xlabel.replace("PDM score", "Score") + " Distribution")
    ax.grid(axis="y", color="#cbd5e1", linewidth=0.7, alpha=0.65)
    plt.tight_layout()
    save_figure(fig, out_dir / filename, formats)
    plt.close()


def score_distribution_source_rows(
    distribution_summary: Dict[str, object],
    source_key: str,
) -> List[Dict[str, object]]:
    sources = distribution_summary.get("sources", {})
    if not isinstance(sources, dict):
        return []
    source = sources.get(source_key, {})
    if not isinstance(source, dict):
        return []
    bins = source.get("bins", [])
    if not isinstance(bins, list):
        return []
    rows: List[Dict[str, object]] = []
    for item in bins:
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "source": source_key,
                "source_label": source.get("label", item.get("source_label", source_key)),
                "bin_index": item.get("bin_index", ""),
                "bin_left": item.get("bin_left", ""),
                "bin_right": item.get("bin_right", ""),
                "bin_label": item.get("bin_label", ""),
                "count": item.get("count", ""),
                "fraction": item.get("fraction", ""),
                "total": source.get("total", ""),
                "underflow_count": source.get("underflow_count", ""),
                "overflow_count": source.get("overflow_count", ""),
            }
        )
    return rows


def write_score_distribution_chart_csvs(
    out_dir: Path,
    distribution_summary: Dict[str, object],
) -> None:
    for source_key, filename in (
        ("candidate", "candidate_score_distribution.csv"),
        ("gt", "gt_score_distribution.csv"),
        ("best_candidate", "best_candidate_score_distribution.csv"),
    ):
        rows = score_distribution_source_rows(distribution_summary, source_key)
        if rows:
            write_csv(out_dir / filename, rows)


def plot_score_bin_distribution_from_json(
    plt,
    out_dir: Path,
    distribution_summary: Dict[str, object],
    formats: Sequence[str],
) -> None:
    sources = distribution_summary.get("sources", {})
    if not isinstance(sources, dict) or "candidate" not in sources:
        return
    candidate_source = sources.get("candidate", {})
    if not isinstance(candidate_source, dict):
        return
    candidate_bins = candidate_source.get("bins", [])
    if not isinstance(candidate_bins, list) or not candidate_bins:
        return

    labels = [str(item.get("bin_label", "")) for item in candidate_bins if isinstance(item, dict)]
    if not labels:
        return
    x = np.arange(len(labels), dtype=np.float32)
    width = 0.26
    fig_width = max(7.2, 0.68 * len(labels))
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, 3.9), dpi=180)
    specs = [
        ("candidate", "All candidates", CANDIDATE_COLOR, -width),
        ("best_candidate", "Best candidate", BEST_CANDIDATE_COLOR, 0.0),
        ("gt", "GT", GT_COLOR, width),
    ]
    drawn = False
    for source_key, label, color, offset in specs:
        source = sources.get(source_key, {})
        if not isinstance(source, dict):
            continue
        bins = source.get("bins", [])
        if not isinstance(bins, list) or len(bins) != len(labels):
            continue
        fractions = [float(item.get("fraction", 0.0)) for item in bins if isinstance(item, dict)]
        if len(fractions) != len(labels):
            continue
        ax.bar(x + offset, fractions, width=width, color=color, alpha=0.86, label=label)
        drawn = True
    if not drawn:
        plt.close()
        return
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_xlabel("PDM score bin")
    ax.set_ylabel("Fraction")
    ax.set_title("Score Bin Distribution")
    ax.grid(axis="y", color="#cbd5e1", linewidth=0.7, alpha=0.65)
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.01))
    plt.tight_layout()
    save_figure(fig, out_dir / "score_bin_distribution", formats)
    plt.close()


def plot_pdm_metrics_from_quality_json(
    plt,
    out_dir: Path,
    quality_summary: Dict[str, object],
    formats: Sequence[str],
) -> None:
    sources = quality_summary.get("sources", {})
    if not isinstance(sources, dict):
        return
    gt_source = sources.get("gt", {})
    cand_source = sources.get("best_candidate", {})
    if not isinstance(gt_source, dict) or not isinstance(cand_source, dict):
        return

    component_names = quality_summary.get("component_names", [])
    if not isinstance(component_names, list):
        component_names = []
    metric_labels = [display_metric_label("score")]
    gt_means = [stats_mean(gt_source.get("score"))]
    cand_means = [stats_mean(cand_source.get("score"))]
    gt_components = gt_source.get("components", {})
    cand_components = cand_source.get("components", {})
    if not isinstance(gt_components, dict):
        gt_components = {}
    if not isinstance(cand_components, dict):
        cand_components = {}
    for name in component_names:
        metric_labels.append(display_metric_label(str(name)))
        gt_means.append(stats_mean(gt_components.get(str(name))))
        cand_means.append(stats_mean(cand_components.get(str(name))))

    gt_arr = np.asarray(gt_means, dtype=np.float64)
    cand_arr = np.asarray(cand_means, dtype=np.float64)
    if not (np.isfinite(gt_arr).any() or np.isfinite(cand_arr).any()):
        return

    x = np.arange(len(metric_labels), dtype=np.float32)
    fig_width = max(8.5, 1.05 * len(metric_labels))
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, 3.8), dpi=180)
    finite_pair = np.isfinite(gt_arr) & np.isfinite(cand_arr)
    finite_vals = np.concatenate([gt_arr[np.isfinite(gt_arr)], cand_arr[np.isfinite(cand_arr)]])
    high_score_threshold = 0.8
    y_max = max(1.0, float(finite_vals.max()) * 1.08) if finite_vals.size else 1.0
    high_band_top = max(y_max, high_score_threshold + 0.05)
    ax.axhspan(high_score_threshold, high_band_top, color="#dbeafe", alpha=0.36, zorder=0)
    ax.axhline(
        high_score_threshold,
        color="#334155",
        linestyle="--",
        linewidth=1.0,
        alpha=0.8,
        zorder=1,
    )
    if finite_pair.any():
        ax.fill_between(
            x,
            gt_arr,
            cand_arr,
            where=finite_pair,
            color="#38bdf8",
            alpha=0.22,
            interpolate=True,
            label="Gap",
            zorder=2,
        )
    ax.plot(x, gt_arr, color=GT_COLOR, linewidth=2.4, marker="o", markersize=5.5, label="GT mean", zorder=4)
    ax.plot(
        x,
        cand_arr,
        color=CANDIDATE_COLOR,
        linewidth=2.4,
        marker="o",
        markersize=5.5,
        label="Best candidate mean",
        zorder=5,
    )
    for idx, (gt_val, cand_val) in enumerate(zip(gt_arr, cand_arr)):
        if not (math.isfinite(float(gt_val)) and math.isfinite(float(cand_val))):
            continue
        if max(float(gt_val), float(cand_val)) < high_score_threshold:
            continue
        delta = float(cand_val - gt_val)
        label_y = min(high_band_top - 0.015, max(float(gt_val), float(cand_val)) + 0.025)
        ax.annotate(
            f"{delta:+.3f}",
            xy=(idx, label_y),
            xytext=(0, 0),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            color=DARK_COLOR,
            zorder=6,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, rotation=0, ha="center")
    ax.set_ylim(high_score_threshold, high_band_top)
    ax.set_ylabel("Score")
    ax.set_xlabel("PDM metric")
    ax.set_title("Mean PDM Metrics")
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.01))
    ax.grid(axis="y", color="#cbd5e1", linewidth=0.7, alpha=0.65)
    break_kwargs = dict(transform=ax.transAxes, color=DARK_COLOR, clip_on=False, linewidth=0.9)
    ax.plot((-0.012, 0.012), (-0.018, 0.018), **break_kwargs)
    ax.plot((0.988, 1.012), (-0.018, 0.018), **break_kwargs)
    plt.tight_layout()
    save_figure(fig, out_dir / "pdm_metrics_gt_vs_best_candidate", formats)
    plt.close()


def maybe_write_plots_from_json(out_dir: Path, json_dir: Path, args: argparse.Namespace) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"skip plots: matplotlib unavailable: {exc}")
        return

    set_matplotlib_style(plt)
    plot_formats = normalize_plot_formats(args.plot_formats)
    score_distribution_formats = extend_plot_formats(plot_formats, ["svg"])

    quality_path = json_dir / "quality_summary.json"
    distribution_path = json_dir / "score_distributions.json"
    if quality_path.is_file():
        plot_pdm_metrics_from_quality_json(
            plt,
            out_dir,
            load_json_file(quality_path),
            plot_formats,
        )
    else:
        print(f"skip PDM metrics plot: missing {quality_path}")

    if distribution_path.is_file():
        distribution_summary = load_json_file(distribution_path)
        write_score_distribution_chart_csvs(out_dir, distribution_summary)
        plot_single_score_distribution_from_json(
            plt,
            out_dir,
            distribution_summary,
            "candidate",
            "candidate_score_distribution",
            CANDIDATE_COLOR,
            "Candidate PDM score",
            score_distribution_formats,
        )
        plot_single_score_distribution_from_json(
            plt,
            out_dir,
            distribution_summary,
            "gt",
            "gt_score_distribution",
            GT_COLOR,
            "GT PDM score",
            score_distribution_formats,
        )
        plot_single_score_distribution_from_json(
            plt,
            out_dir,
            distribution_summary,
            "best_candidate",
            "best_candidate_score_distribution",
            BEST_CANDIDATE_COLOR,
            "Best candidate PDM score",
            plot_formats,
        )
        plot_score_bin_distribution_from_json(plt, out_dir, distribution_summary, plot_formats)
    else:
        print(f"skip score distribution plots: missing {distribution_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", default=None)
    parser.add_argument(
        "--plot-from-output-dir",
        default=None,
        help="Read existing quality_summary.json and score_distributions.json from this folder and redraw plots only.",
    )
    parser.add_argument(
        "--roots",
        nargs="*",
        default=None,
        help="Optional first-level roots to scan, e.g. train val or selected log names.",
    )
    parser.add_argument(
        "--roots-file",
        default=None,
        help="Optional text file containing one first-level cache root/log name per line.",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--gt-margin", type=float, default=0.0)
    parser.add_argument("--lane-change-y-thresh", type=float, default=2.0)
    parser.add_argument("--component-names", nargs="*", default=None)
    parser.add_argument(
        "--score-bins",
        default=DEFAULT_SCORE_BINS,
        help="Comma/space separated PDM score bin edges for score_distributions.* outputs.",
    )
    parser.add_argument("--plots", action="store_true")
    parser.add_argument(
        "--plot-formats",
        nargs="+",
        default=["png"],
        help="Figure formats to save, e.g. --plot-formats png pdf svg.",
    )
    parser.add_argument("--map-vis-max", type=int, default=0, help="Number of candidate/GT map visualizations to save.")
    parser.add_argument(
        "--map-vis-sort",
        choices=["best_gain", "best_score", "worst_gt", "first"],
        default="best_gain",
        help="How to choose samples for map visualizations.",
    )
    parser.add_argument("--map-vis-bev-key", default="bev_semantic_map")
    parser.add_argument("--map-vis-pixel-size", type=float, default=0.25)
    parser.add_argument("--map-vis-dpi", type=int, default=220)
    parser.add_argument(
        "--metric-cache-path",
        default=None,
        help="NAVSIM metric_cache root. Used by vector map visualizations and optional online PDM scoring.",
    )
    parser.add_argument(
        "--online-score-best-gt",
        action="store_true",
        help="Re-score only GT and the cached best candidate online, then use those values in CSV/summary/plots.",
    )
    parser.add_argument(
        "--online-score-no-reference-progress",
        action="store_true",
        help="Disable metric-cache reference trajectory baseline for online EP/progress.",
    )
    parser.add_argument("--online-pdm-num-poses", type=int, default=40)
    parser.add_argument("--online-pdm-interval", type=float, default=0.1)
    parser.add_argument("--online-score-batch-size", type=int, default=64)
    parser.add_argument("--online-cache-lru-size", type=int, default=128)
    parser.add_argument("--online-use-ray", action="store_true")
    parser.add_argument("--online-ray-threads", type=int, default=0)
    parser.add_argument("--online-progress-weight", type=float, default=5.0)
    parser.add_argument("--online-ttc-weight", type=float, default=5.0)
    parser.add_argument("--online-comfort-weight", type=float, default=2.0)
    parser.add_argument("--online-driving-direction-weight", type=float, default=0.0)
    parser.add_argument(
        "--map-vis-require-metric-cache",
        action="store_true",
        help="Skip map visualizations without a matching metric cache instead of falling back to BEV.",
    )
    parser.add_argument("--map-vis-radius", type=float, default=55.0)
    parser.add_argument(
        "--map-vis-candidates",
        choices=["valid", "all", "best"],
        default="valid",
        help="Which candidates to draw on map visualizations.",
    )
    parser.add_argument(
        "--feasible-vis-max",
        type=int,
        default=0,
        help="Number of feasible-region / sampled-point comparison figures to save.",
    )
    parser.add_argument(
        "--feasible-vis-point-source",
        choices=["valid", "all", "best"],
        default="valid",
        help="Which candidate trajectories supply sampled points in feasible visualizations.",
    )
    parser.add_argument("--feasible-vis-max-points", type=int, default=0, help="Optional cap on plotted sample points.")
    parser.add_argument("--feasible-vis-point-size", type=float, default=13.0)
    parser.add_argument("--feasible-vis-point-alpha", type=float, default=0.82)
    parser.add_argument("--feasible-vis-draw-gt", action="store_true", help="Overlay GT points on sampled-point panel.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    json_dir = Path(args.plot_from_output_dir) if args.plot_from_output_dir else None
    if args.output_dir:
        out_dir = Path(args.output_dir)
    elif json_dir is not None:
        out_dir = json_dir
    else:
        raise ValueError("--output-dir is required unless --plot-from-output-dir is used")
    out_dir.mkdir(parents=True, exist_ok=True)

    if json_dir is not None:
        maybe_write_plots_from_json(out_dir, json_dir, args)
        if int(args.feasible_vis_max) > 0:
            per_sample_path = json_dir / "per_sample.csv"
            if per_sample_path.is_file():
                maybe_write_feasible_point_visualizations(out_dir, read_csv(per_sample_path), args)
            else:
                print(f"skip feasible visualizations: missing {per_sample_path}")
        print(f"redrew plots from json: {json_dir}")
        print(f"wrote plots to: {out_dir}")
        return

    if not args.cache_root:
        raise ValueError("--cache-root is required unless --plot-from-output-dir is used")
    cache_root = Path(args.cache_root)
    roots = list(args.roots or [])
    roots.extend(read_roots_file(args.roots_file))
    target_files = list(iter_target_files(cache_root, roots or None, int(args.limit)))
    if not target_files:
        raise FileNotFoundError(f"No transfuser_target.gz found under {cache_root}")

    rows = [summarize_one(path, args) for path in tqdm(target_files, desc="summarizing candidates")]
    rows = [row for row in rows if row is not None]
    online_counters = online_score_gt_and_best(rows, args)
    summary = aggregate(rows)
    if online_counters.get("enabled", 0):
        summary["online_score_counters"] = online_counters
    quality_outputs = write_quality_outputs(out_dir, rows)
    distribution_outputs = write_score_distribution_outputs(out_dir, rows, args)
    summary["quality_summary_file"] = quality_outputs["json"]
    summary["quality_summary_csv"] = quality_outputs["csv"]
    summary["score_distributions_file"] = distribution_outputs["json"]
    summary["score_distributions_csv"] = distribution_outputs["csv"]

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    write_csv(out_dir / "per_sample.csv", rows)
    if bool(args.plots):
        maybe_write_plots(out_dir, rows, args)
    if int(args.map_vis_max) > 0:
        metric_cache_index = load_metric_cache_index(args.metric_cache_path)
        maybe_write_map_visualizations(out_dir, rows, args, metric_cache_index)
    if int(args.feasible_vis_max) > 0:
        maybe_write_feasible_point_visualizations(out_dir, rows, args)

    print(f"files={summary['total_files']} with_candidates={summary['with_candidates']} valid={summary['with_valid_candidates']}")
    print(f"high_quality_sample_rate={summary['high_quality_sample_rate']:.4f} mean_num_above_gt={summary['mean_num_above_gt']:.3f}")
    if online_counters.get("enabled", 0):
        print(
            "online_score "
            f"queued={online_counters['queued']} scored={online_counters['scored']} "
            f"skipped={online_counters['skipped']} errors={online_counters['errors']}"
        )
    print(f"wrote: {out_dir / 'summary.json'}")
    print(f"wrote: {out_dir / 'per_sample.csv'}")
    print(f"wrote: {out_dir / 'quality_summary.json'}")
    print(f"wrote: {out_dir / 'score_distributions.json'}")


if __name__ == "__main__":
    main()
