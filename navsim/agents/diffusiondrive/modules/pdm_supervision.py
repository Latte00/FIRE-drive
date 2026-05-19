from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import lzma
import pickle
from collections import OrderedDict

import numpy as np

from navsim.common.dataclasses import Trajectory
from navsim.common.dataloader import MetricCacheLoader
from navsim.evaluate.pdm_score import (
    get_trajectory_as_array,
    transform_trajectory,
)
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import (
    PDMSimulator,
)
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import (
    PDMScorer,
)
try:
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import (
        MultiMetricIndex,
        WeightedMetricIndex,
    )
except ImportError:  # pragma: no cover - optional in some environments
    MultiMetricIndex = None
    WeightedMetricIndex = None
from nuplan.planning.simulation.trajectory.trajectory_sampling import (
    TrajectorySampling,
)

try:
    from navsim.planning.utils.multithreading.worker_ray_no_torch import (
        RayDistributedNoTorch,
    )
    from nuplan.planning.utils.multithreading.worker_utils import worker_map
except ImportError:  # pragma: no cover - optional ray deps
    RayDistributedNoTorch = None
    worker_map = None

try:
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import (
        PDMScorerConfig,
    )
except ImportError:  # pragma: no cover - optional in some environments
    PDMScorerConfig = None


@dataclass
class PDMScoreConfig:
    cache_path: str
    num_poses: int = 40
    interval_length: float = 0.1
    use_ray: bool = False
    ray_threads: int = 0
    ray_debug: bool = False
    cache_lru_size: int = 0
    progress_use_reference_baseline: bool = False


_PDM_WORKERS: Dict[Tuple[str, int, float, int, bool], "_PDMWorker"] = {}
_INFRACTION_NUMERIC_KEYS = (
    "collision_time_idx",
    "collision_time_s",
    "collision_position_xy",
    "collision_actor_position_xy",
    "collision_agent_position_xy",
    "collision_is_agent",
    "collision_valid",
    "offroad_time_idx",
    "offroad_time_s",
    "offroad_position_xy",
    "offroad_valid",
)


class _PDMWorker:
    def __init__(
        self,
        cache_path: str,
        num_poses: int,
        interval_length: float,
        cache_lru_size: int,
        progress_use_reference_baseline: bool,
    ) -> None:
        cache_root = Path(cache_path)
        self._metric_cache_paths = MetricCacheLoader(cache_root).metric_cache_paths
        self._proposal_sampling = TrajectorySampling(
            num_poses=num_poses, interval_length=interval_length
        )
        self._simulator = PDMSimulator(self._proposal_sampling)
        self._scorer = self._build_scorer()
        self._reference_scorer = self._build_scorer()
        self._progress_use_reference_baseline = bool(progress_use_reference_baseline)
        self._cache_lru_size = max(int(cache_lru_size), 0)
        self._metric_cache_lru: Optional[OrderedDict[str, Any]] = (
            OrderedDict() if self._cache_lru_size > 0 else None
        )

    def _build_scorer(self) -> PDMScorer:
        if PDMScorerConfig is None:
            return PDMScorer(self._proposal_sampling)
        return PDMScorer(self._proposal_sampling, PDMScorerConfig())

    @staticmethod
    def _extract_ego_pose(initial_ego_state: Any) -> Optional[Tuple[float, float, float]]:
        if initial_ego_state is None:
            return None
        center = getattr(initial_ego_state, "center", None)
        if center is None:
            return None
        x = getattr(center, "x", None)
        y = getattr(center, "y", None)
        heading = getattr(center, "heading", None)
        if x is None or y is None or heading is None:
            return None
        return float(x), float(y), float(heading)

    @staticmethod
    def _global_to_local_xy(
        points: Any,
        ego_x: float,
        ego_y: float,
        ego_heading: float,
    ) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[-1] != 2:
            return pts
        dx = pts[:, 0] - float(ego_x)
        dy = pts[:, 1] - float(ego_y)
        cos_h = float(np.cos(ego_heading))
        sin_h = float(np.sin(ego_heading))
        local_x = cos_h * dx + sin_h * dy
        local_y = -sin_h * dx + cos_h * dy
        return np.stack([local_x, local_y], axis=-1).astype(np.float32)

    def _convert_infraction_to_local(
        self,
        infraction_data: Optional[Dict[str, Any]],
        initial_ego_state: Any,
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(infraction_data, dict):
            return infraction_data
        ego_pose = self._extract_ego_pose(initial_ego_state)
        if ego_pose is None:
            return infraction_data
        ego_x, ego_y, ego_heading = ego_pose
        out = dict(infraction_data)
        for key in (
            "collision_position_xy",
            "collision_actor_position_xy",
            "collision_agent_position_xy",
            "offroad_position_xy",
        ):
            if key not in out:
                continue
            out[key] = self._global_to_local_xy(
                out[key], ego_x=ego_x, ego_y=ego_y, ego_heading=ego_heading
            )
        return out

    def _load_metric_cache(self, token: str) -> Optional[Any]:
        cache_path = self._metric_cache_paths.get(token)
        if cache_path is None:
            return None
        if self._metric_cache_lru is not None:
            cached = self._metric_cache_lru.get(token)
            if cached is not None:
                self._metric_cache_lru.move_to_end(token)
                return cached
        try:
            with lzma.open(cache_path, "rb") as f:
                metric_cache = pickle.load(f)
        except (OSError, EOFError, FileNotFoundError):
            return None
        if self._metric_cache_lru is not None:
            self._metric_cache_lru[token] = metric_cache
            if len(self._metric_cache_lru) > self._cache_lru_size:
                self._metric_cache_lru.popitem(last=False)
        return metric_cache

    def _extract_components(
        self,
        num_modes: int,
        progress_override: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if MultiMetricIndex is None or WeightedMetricIndex is None:
            raise ValueError("PDM component enums are unavailable")
        no_collision = np.asarray(
            self._scorer._multi_metrics[MultiMetricIndex.NO_COLLISION],
            dtype=np.float32,
        ).reshape(-1)
        drivable = np.asarray(
            self._scorer._multi_metrics[MultiMetricIndex.DRIVABLE_AREA],
            dtype=np.float32,
        ).reshape(-1)
        if progress_override is None:
            ego_progress = np.asarray(
                self._scorer._weighted_metrics[WeightedMetricIndex.PROGRESS],
                dtype=np.float32,
            ).reshape(-1)
        else:
            ego_progress = np.asarray(progress_override, dtype=np.float32).reshape(-1)
        ttc = np.asarray(
            self._scorer._weighted_metrics[WeightedMetricIndex.TTC],
            dtype=np.float32,
        ).reshape(-1)
        comfort = np.asarray(
            self._scorer._weighted_metrics[WeightedMetricIndex.COMFORTABLE],
            dtype=np.float32,
        ).reshape(-1)
        driving_dir = np.asarray(
            self._scorer._weighted_metrics[WeightedMetricIndex.DRIVING_DIRECTION],
            dtype=np.float32,
        ).reshape(-1)
        components = np.stack(
            [no_collision, drivable, ego_progress, ttc, comfort, driving_dir],
            axis=-1,
        )
        if components.shape[0] != num_modes:
            components = components[:num_modes]
        return components

    @staticmethod
    def _extract_effective_raw_progress_from_scorer(
        scorer: PDMScorer,
    ) -> Tuple[np.ndarray, np.ndarray]:
        multi_metrics = np.asarray(scorer._multi_metrics, dtype=np.float64)
        progress_raw = np.asarray(scorer._progress_raw, dtype=np.float64).reshape(-1)
        multiplicative = multi_metrics.prod(axis=0).reshape(-1)
        effective_raw_progress = progress_raw * multiplicative
        return effective_raw_progress, multiplicative

    def _compute_reference_baseline_progress(
        self,
        metric_cache: Any,
        initial_ego_state: Any,
    ) -> float:
        baseline_states = get_trajectory_as_array(
            metric_cache.trajectory,
            self._proposal_sampling,
            initial_ego_state.time_point,
        )
        simulated_baseline_states = self._simulator.simulate_proposals(
            baseline_states[None, ...],
            initial_ego_state,
        )
        self._reference_scorer.score_proposals(
            simulated_baseline_states,
            metric_cache.observation,
            metric_cache.centerline,
            metric_cache.route_lane_ids,
            metric_cache.drivable_area_map,
        )
        baseline_effective_progress, _ = self._extract_effective_raw_progress_from_scorer(
            self._reference_scorer
        )
        if baseline_effective_progress.size == 0:
            return 0.0
        return float(np.nan_to_num(baseline_effective_progress[0], nan=0.0))

    def _compute_reference_progress_override(
        self,
        metric_cache: Any,
        initial_ego_state: Any,
    ) -> np.ndarray:
        candidate_effective_progress, candidate_multiplicative = (
            self._extract_effective_raw_progress_from_scorer(self._scorer)
        )
        baseline_effective_progress = self._compute_reference_baseline_progress(
            metric_cache, initial_ego_state
        )
        threshold = float(
            getattr(self._scorer._config, "progress_distance_threshold", 0.0)
        )
        pairwise_max_progress = np.maximum(
            candidate_effective_progress,
            baseline_effective_progress,
        )
        progress = np.ones_like(candidate_effective_progress, dtype=np.float32)
        over_threshold = pairwise_max_progress > threshold
        safe_denom = np.clip(pairwise_max_progress, a_min=1e-6, a_max=None)
        progress[over_threshold] = (
            candidate_effective_progress[over_threshold] / safe_denom[over_threshold]
        ).astype(np.float32)
        progress[~over_threshold] = (candidate_multiplicative[~over_threshold] > 0.0).astype(
            np.float32
        )
        return progress

    def score_tokens(
        self,
        tokens: List[str],
        trajectories: np.ndarray,
        return_components: bool = False,
        return_infraction_data: bool = False,
    ) -> Any:
        if trajectories.ndim != 4:
            raise ValueError(
                f"Expected trajectories [B, K, T, 3], got shape {trajectories.shape}"
            )
        batch_size, num_modes = trajectories.shape[:2]
        scores = np.full((batch_size, num_modes), np.nan, dtype=np.float32)
        components = None
        infraction_data: Optional[List[Optional[Dict[str, Any]]]] = None
        if return_components:
            components = np.full((batch_size, num_modes, 6), np.nan, dtype=np.float32)
        if return_infraction_data:
            infraction_data = [None for _ in range(batch_size)]
        for idx, token in enumerate(tokens):
            metric_cache = self._load_metric_cache(token)
            if metric_cache is None:
                continue

            initial_ego_state = metric_cache.ego_state
            traj_states = []
            for mode_traj in trajectories[idx]:
                traj = Trajectory(
                    np.nan_to_num(mode_traj, nan=0.0).astype(np.float32)
                )
                pred_trajectory = transform_trajectory(traj, initial_ego_state)
                pred_states = get_trajectory_as_array(
                    pred_trajectory,
                    self._proposal_sampling,
                    initial_ego_state.time_point,
                )
                traj_states.append(pred_states)
            if not traj_states:
                continue
            traj_states = np.stack(traj_states, axis=0)
            simulated_states = self._simulator.simulate_proposals(
                traj_states, initial_ego_state
            )
            pdm_progress = getattr(metric_cache, "pdm_progress", None)
            try:
                scores[idx] = self._scorer.score_proposals(
                    simulated_states,
                    metric_cache.observation,
                    metric_cache.centerline,
                    metric_cache.route_lane_ids,
                    metric_cache.drivable_area_map,
                    pdm_progress,
                )
            except TypeError:
                scores[idx] = self._scorer.score_proposals(
                    simulated_states,
                    metric_cache.observation,
                    metric_cache.centerline,
                    metric_cache.route_lane_ids,
                    metric_cache.drivable_area_map,
                )
            if components is not None:
                progress_override = None
                if self._progress_use_reference_baseline:
                    try:
                        progress_override = self._compute_reference_progress_override(
                            metric_cache=metric_cache,
                            initial_ego_state=initial_ego_state,
                        )
                    except Exception:
                        progress_override = None
                components[idx] = self._extract_components(
                    num_modes,
                    progress_override=progress_override,
                )
            if infraction_data is not None:
                raw_infraction = self._scorer.export_infraction_features()
                infraction_data[idx] = self._convert_infraction_to_local(
                    raw_infraction, initial_ego_state
                )
        if components is not None and infraction_data is not None:
            return scores, components, infraction_data
        if components is not None:
            return scores, components
        if infraction_data is not None:
            return scores, infraction_data
        return scores


def _get_pdm_worker(
    cache_path: str,
    num_poses: int,
    interval_length: float,
    cache_lru_size: int,
    progress_use_reference_baseline: bool,
) -> _PDMWorker:
    key = (
        str(cache_path),
        int(num_poses),
        float(interval_length),
        int(cache_lru_size),
        bool(progress_use_reference_baseline),
    )
    worker = _PDM_WORKERS.get(key)
    if worker is None:
        worker = _PDMWorker(
            cache_path=cache_path,
            num_poses=num_poses,
            interval_length=interval_length,
            cache_lru_size=cache_lru_size,
            progress_use_reference_baseline=progress_use_reference_baseline,
        )
        _PDM_WORKERS[key] = worker
    return worker


def _pdm_score_worker(args: List[Dict[str, Any]]) -> List[Dict[str, Any]]:      
    if not args:
        return []
    cache_path = args[0]["cache_path"]
    num_poses = int(args[0]["num_poses"])
    interval_length = float(args[0]["interval_length"])
    cache_lru_size = int(args[0]["cache_lru_size"])
    progress_use_reference_baseline = bool(
        args[0].get("progress_use_reference_baseline", False)
    )
    return_components = bool(args[0].get("return_components", False))
    return_infraction_data = bool(args[0].get("return_infraction_data", False))
    worker = _get_pdm_worker(
        cache_path,
        num_poses,
        interval_length,
        cache_lru_size,
        progress_use_reference_baseline,
    )
    tokens = [item["token"] for item in args]
    trajectories = np.stack([item["trajectory"] for item in args], axis=0)
    if return_components and return_infraction_data:
        scores, components, infraction_data = worker.score_tokens(
            tokens,
            trajectories,
            return_components=True,
            return_infraction_data=True,
        )
    elif return_components:
        scores, components = worker.score_tokens(
            tokens, trajectories, return_components=True
        )
    elif return_infraction_data:
        scores, infraction_data = worker.score_tokens(
            tokens, trajectories, return_infraction_data=True
        )
    else:
        scores = worker.score_tokens(tokens, trajectories)
    results: List[Dict[str, Any]] = []
    for idx, (item, score) in enumerate(zip(args, scores)):
        payload = {"index": item["index"], "scores": score}
        if return_components:
            payload["components"] = components[idx]
        if return_infraction_data:
            payload["infraction_data"] = infraction_data[idx]
        results.append(payload)
    return results


class PDMSupervision:
    """Compute PDM scores for a batch of candidate trajectories."""

    def __init__(self, config: PDMScoreConfig) -> None:
        self._config = config
        self._worker_state = _PDMWorker(
            cache_path=config.cache_path,
            num_poses=config.num_poses,
            interval_length=config.interval_length,
            cache_lru_size=config.cache_lru_size,
            progress_use_reference_baseline=config.progress_use_reference_baseline,
        )
        self._use_ray = bool(config.use_ray)
        self._ray_worker = None
        if self._use_ray:
            if RayDistributedNoTorch is None or worker_map is None:
                self._use_ray = False
            else:
                threads = config.ray_threads if config.ray_threads > 0 else None
                self._ray_worker = RayDistributedNoTorch(
                    threads_per_node=threads,
                    debug_mode=config.ray_debug,
                    log_to_driver=False,
                )

    @staticmethod
    def _init_infraction_data(batch_size: int, num_modes: int) -> Dict[str, Any]:
        return {
            "collision_time_idx": np.full((batch_size, num_modes), np.nan, dtype=np.float32),
            "collision_time_s": np.full((batch_size, num_modes), np.nan, dtype=np.float32),
            "collision_position_xy": np.full((batch_size, num_modes, 2), np.nan, dtype=np.float32),
            "collision_actor_position_xy": np.full((batch_size, num_modes, 2), np.nan, dtype=np.float32),
            "collision_agent_position_xy": np.full((batch_size, num_modes, 2), np.nan, dtype=np.float32),
            "collision_is_agent": np.zeros((batch_size, num_modes), dtype=np.bool_),
            "collision_valid": np.zeros((batch_size, num_modes), dtype=np.bool_),
            "offroad_time_idx": np.full((batch_size, num_modes), np.nan, dtype=np.float32),
            "offroad_time_s": np.full((batch_size, num_modes), np.nan, dtype=np.float32),
            "offroad_position_xy": np.full((batch_size, num_modes, 2), np.nan, dtype=np.float32),
            "offroad_valid": np.zeros((batch_size, num_modes), dtype=np.bool_),
            "collision_track_token": np.full((batch_size, num_modes), None, dtype=object),
        }

    @staticmethod
    def _fill_infraction_row(
        infraction_data: Dict[str, Any],
        batch_idx: int,
        row_data: Optional[Dict[str, Any]],
    ) -> None:
        if not isinstance(row_data, dict):
            return
        for key in _INFRACTION_NUMERIC_KEYS:
            value = row_data.get(key)
            if value is None:
                continue
            dst = infraction_data.get(key)
            if dst is None:
                continue
            arr = np.asarray(value, dtype=dst.dtype)
            if arr.shape != dst[batch_idx].shape:
                continue
            dst[batch_idx] = arr
        token_row = row_data.get("collision_track_token")
        if token_row is None:
            return
        dst_token = infraction_data.get("collision_track_token")
        if dst_token is None:
            return
        token_arr = np.asarray(token_row, dtype=object)
        if token_arr.shape != dst_token[batch_idx].shape:
            return
        dst_token[batch_idx] = token_arr

    def _score_batch_internal(
        self,
        tokens: List[str],
        trajectories: np.ndarray,
        return_components: bool = False,
        return_infraction_data: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[Dict[str, Any]]]:
        if trajectories.ndim != 4:
            raise ValueError(
                f"Expected trajectories [B, K, T, 3], got shape {trajectories.shape}"
            )
        batch_size, num_modes = trajectories.shape[:2]
        components = (
            np.full((batch_size, num_modes, 6), np.nan, dtype=np.float32)
            if return_components
            else None
        )
        infraction_data = (
            self._init_infraction_data(batch_size, num_modes)
            if return_infraction_data
            else None
        )

        if (
            self._use_ray
            and self._ray_worker is not None
            and worker_map is not None
            and batch_size > 0
        ):
            data_points = [
                {
                    "index": idx,
                    "token": token,
                    "trajectory": trajectories[idx],
                    "cache_path": self._config.cache_path,
                    "num_poses": self._config.num_poses,
                    "interval_length": self._config.interval_length,
                    "cache_lru_size": self._config.cache_lru_size,
                    "progress_use_reference_baseline": self._config.progress_use_reference_baseline,
                    "return_components": return_components,
                    "return_infraction_data": return_infraction_data,
                }
                for idx, token in enumerate(tokens)
            ]
            results = worker_map(self._ray_worker, _pdm_score_worker, data_points)
            flat_results: List[Dict[str, Any]] = []
            for item in results:
                if isinstance(item, list):
                    flat_results.extend([entry for entry in item if isinstance(entry, dict)])
                elif isinstance(item, dict):
                    flat_results.append(item)
            scores = np.full((batch_size, num_modes), np.nan, dtype=np.float32)
            for item in flat_results:
                idx = item.get("index")
                if idx is None:
                    continue
                scores[idx] = item.get("scores", scores[idx])
                if components is not None:
                    comps = item.get("components")
                    if comps is not None:
                        components[idx] = comps
                if infraction_data is not None:
                    self._fill_infraction_row(
                        infraction_data,
                        idx,
                        item.get("infraction_data"),
                    )
            return scores, components, infraction_data

        if return_components and return_infraction_data:
            scores, local_components, local_infraction = self._worker_state.score_tokens(
                tokens,
                trajectories,
                return_components=True,
                return_infraction_data=True,
            )
            components = local_components
            if infraction_data is not None:
                for idx, row_data in enumerate(local_infraction):
                    self._fill_infraction_row(infraction_data, idx, row_data)
            return scores, components, infraction_data

        if return_components:
            scores, local_components = self._worker_state.score_tokens(
                tokens,
                trajectories,
                return_components=True,
            )
            return scores, local_components, infraction_data

        if return_infraction_data:
            scores, local_infraction = self._worker_state.score_tokens(
                tokens,
                trajectories,
                return_infraction_data=True,
            )
            if infraction_data is not None:
                for idx, row_data in enumerate(local_infraction):
                    self._fill_infraction_row(infraction_data, idx, row_data)
            return scores, components, infraction_data

        scores = self._worker_state.score_tokens(tokens, trajectories)
        return scores, components, infraction_data

    def score_batch(
        self, tokens: List[str], trajectories: np.ndarray
    ) -> np.ndarray:
        """
        Args:
            tokens: list of scenario tokens, length B
            trajectories: array [B, K, T, 3] in ego frame
        Returns:
            scores: array [B, K], NaN for missing tokens
        """
        scores, _, _ = self._score_batch_internal(
            tokens=tokens,
            trajectories=trajectories,
            return_components=False,
            return_infraction_data=False,
        )
        return scores

    def score_batch_components(
        self, tokens: List[str], trajectories: np.ndarray
    ) -> np.ndarray:
        """
        Args:
            tokens: list of scenario tokens, length B
            trajectories: array [B, K, T, 3] in ego frame
        Returns:
            components: array [B, K, 6], NaN for missing tokens
        """
        _, components, _ = self._score_batch_internal(
            tokens=tokens,
            trajectories=trajectories,
            return_components=True,
            return_infraction_data=False,
        )
        if components is None:
            batch_size, num_modes = trajectories.shape[:2]
            return np.full((batch_size, num_modes, 6), np.nan, dtype=np.float32)
        return components

    def score_batch_with_infraction_data(
        self, tokens: List[str], trajectories: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        scores, _, infraction_data = self._score_batch_internal(
            tokens=tokens,
            trajectories=trajectories,
            return_components=False,
            return_infraction_data=True,
        )
        if infraction_data is None:
            batch_size, num_modes = trajectories.shape[:2]
            infraction_data = self._init_infraction_data(batch_size, num_modes)
        return scores, infraction_data

    def score_batch_components_with_infraction_data(
        self, tokens: List[str], trajectories: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        _, components, infraction_data = self._score_batch_internal(
            tokens=tokens,
            trajectories=trajectories,
            return_components=True,
            return_infraction_data=True,
        )
        if components is None:
            batch_size, num_modes = trajectories.shape[:2]
            components = np.full((batch_size, num_modes, 6), np.nan, dtype=np.float32)
        if infraction_data is None:
            batch_size, num_modes = trajectories.shape[:2]
            infraction_data = self._init_infraction_data(batch_size, num_modes)
        return components, infraction_data
