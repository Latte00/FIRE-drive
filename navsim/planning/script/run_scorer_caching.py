from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import logging
import os
import lzma
import pickle

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataloader import MetricCacheLoader
from navsim.common.dataclasses import Trajectory
from navsim.evaluate.pdm_score import (
    get_trajectory_as_array,
    transform_trajectory,
    pdm_score,
)
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import (
    PDMSimulator,
)
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import (
    PDMScorer,
)
from nuplan.planning.simulation.trajectory.trajectory_sampling import (
    TrajectorySampling,
)
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import (
    MultiMetricIndex,
    WeightedMetricIndex,
)
from navsim.planning.training.dataset import (
    CacheOnlyDataset,
    dump_feature_target_to_pickle,
)

try:
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import (
        PDMScorerConfig,
    )
except ImportError:  # pragma: no cover - optional in some environments
    PDMScorerConfig = None

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/training"
CONFIG_NAME = "default_training"

PDM_COMPONENT_KEYS = (
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "ego_progress",
    "time_to_collision_within_bound",
    "comfort",
    "driving_direction_compliance",
)


def _aggregate_drivor_score(
    components: np.ndarray, weights: np.ndarray, eps: float = 1e-6
) -> np.ndarray:
    """Aggregate PDM components using DrivoR-style log formulation."""
    if components.ndim != 2 or components.shape[-1] != len(PDM_COMPONENT_KEYS):
        raise ValueError(
            f"Expected components [K, {len(PDM_COMPONENT_KEYS)}], got {components.shape}"
        )
    if weights.shape[-1] != len(PDM_COMPONENT_KEYS):
        weights = np.ones(len(PDM_COMPONENT_KEYS), dtype=np.float32)
    comp = np.clip(components.astype(np.float32), 0.0, 1.0)
    w = weights.astype(np.float32)
    noc = comp[:, 0]
    dac = comp[:, 1]
    ep = comp[:, 2]
    ttc = comp[:, 3]
    comfort = comp[:, 4]
    ddc = comp[:, 5]
    w_noc, w_dac, w_ep, w_ttc, w_comfort, w_ddc = w.tolist()
    log_terms = (
        w_noc * np.log(noc + eps)
        + w_dac * np.log(dac + eps)
        + w_ddc * np.log(ddc + eps)
    )
    sum_term = w_ttc * ttc + w_ep * ep + w_comfort * comfort
    return log_terms + np.log(sum_term + eps)


def _normalize_token(value: Any, fallback: str) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if torch.is_tensor(value):
        try:
            return str(value.item())
        except Exception:
            return fallback
    return fallback


def _ensure_heading(traj: np.ndarray) -> np.ndarray:
    if traj.shape[-1] == 3:
        return traj
    if traj.shape[-1] != 2:
        raise ValueError(f"Expected trajectory last dim 2 or 3, got {traj.shape}")
    delta = np.diff(traj, axis=-2, prepend=traj[..., :1, :])
    heading = np.arctan2(delta[..., 1], delta[..., 0])
    return np.concatenate([traj, heading[..., None]], axis=-1)


def _get_metric_cache_path(cfg: DictConfig) -> Optional[str]:
    path = None
    agent_cfg = getattr(cfg, "agent", None)
    if agent_cfg is not None:
        cfg_config = getattr(agent_cfg, "config", None)
        if cfg_config is not None:
            path = getattr(cfg_config, "pdm_metric_cache_path", None)
    if not path:
        navsim_root = os.getenv("NAVSIM_EXP_ROOT")
        if navsim_root:
            path = str(Path(navsim_root) / "metric_cache")
    return path


def _to_device(
    data: Dict[str, Any], device: torch.device
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in data.items():
        if torch.is_tensor(value):
            out[key] = value.to(device)
        else:
            out[key] = value
    return out


def _add_batch_dim(data: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in data.items():
        if torch.is_tensor(value):
            out[key] = value.unsqueeze(0)
        else:
            out[key] = value
    return out


def _score_pdm_components(
    metric_cache: Any,
    trajectories: np.ndarray,
    traj_sampling: TrajectorySampling,
    simulator: PDMSimulator,
    scorer: PDMScorer,
) -> Dict[str, np.ndarray]:
    if trajectories.ndim != 3:
        raise ValueError(
            f"Expected trajectories [K, T, 3], got shape {trajectories.shape}"
        )
    trajectories = _ensure_heading(trajectories)
    initial_ego_state = metric_cache.ego_state

    pdm_states = get_trajectory_as_array(
        metric_cache.trajectory,
        simulator.proposal_sampling,
        initial_ego_state.time_point,
    )
    pred_states: List[np.ndarray] = []
    for traj in trajectories:
        traj_obj = Trajectory(
            np.nan_to_num(traj, nan=0.0).astype(np.float32),
            trajectory_sampling=traj_sampling,
        )
        pred_trajectory = transform_trajectory(traj_obj, initial_ego_state)
        pred_states.append(
            get_trajectory_as_array(
                pred_trajectory,
                simulator.proposal_sampling,
                initial_ego_state.time_point,
            )
        )
    if not pred_states:
        raise ValueError("No trajectories to score")
    trajectory_states = np.concatenate(
        [pdm_states[None, ...], np.stack(pred_states, axis=0)], axis=0
    )
    simulated_states = simulator.simulate_proposals(
        trajectory_states, initial_ego_state
    )
    pdm_progress = getattr(metric_cache, "pdm_progress", None)
    try:
        scores = scorer.score_proposals(
            simulated_states,
            metric_cache.observation,
            metric_cache.centerline,
            metric_cache.route_lane_ids,
            metric_cache.drivable_area_map,
            pdm_progress,
        )
    except TypeError:
        scores = scorer.score_proposals(
            simulated_states,
            metric_cache.observation,
            metric_cache.centerline,
            metric_cache.route_lane_ids,
            metric_cache.drivable_area_map,
        )

    pred_slice = slice(1, trajectories.shape[0] + 1)
    no_collision = scorer._multi_metrics[MultiMetricIndex.NO_COLLISION, pred_slice]
    drivable = scorer._multi_metrics[MultiMetricIndex.DRIVABLE_AREA, pred_slice]
    ego_progress = scorer._weighted_metrics[WeightedMetricIndex.PROGRESS, pred_slice]
    ttc = scorer._weighted_metrics[WeightedMetricIndex.TTC, pred_slice]
    comfort = scorer._weighted_metrics[WeightedMetricIndex.COMFORTABLE, pred_slice]
    driving_dir = scorer._weighted_metrics[
        WeightedMetricIndex.DRIVING_DIRECTION, pred_slice
    ]
    return {
        "score": np.asarray(scores[pred_slice], dtype=np.float32),
        "no_at_fault_collisions": np.asarray(no_collision, dtype=np.float32),
        "drivable_area_compliance": np.asarray(drivable, dtype=np.float32),
        "ego_progress": np.asarray(ego_progress, dtype=np.float32),
        "time_to_collision_within_bound": np.asarray(ttc, dtype=np.float32),
        "comfort": np.asarray(comfort, dtype=np.float32),
        "driving_direction_compliance": np.asarray(driving_dir, dtype=np.float32),
    }


def _score_pdm_components_via_pdm_score(
    metric_cache: Any,
    trajectories: np.ndarray,
    traj_sampling: TrajectorySampling,
    simulator: PDMSimulator,
    scorer: PDMScorer,
) -> Dict[str, np.ndarray]:
    """Compute PDM components using the same pdm_score() routine as evaluation."""
    if trajectories.ndim != 3:
        raise ValueError(
            f"Expected trajectories [K, T, 3], got shape {trajectories.shape}"
        )
    trajectories = _ensure_heading(trajectories)
    results = []
    for traj in trajectories:
        traj_obj = Trajectory(
            np.nan_to_num(traj, nan=0.0).astype(np.float32),
            trajectory_sampling=traj_sampling,
        )
        results.append(
            pdm_score(
                metric_cache=metric_cache,
                model_trajectory=traj_obj,
                future_sampling=simulator.proposal_sampling,
                simulator=simulator,
                scorer=scorer,
            )
        )
    scores = np.asarray([res.score for res in results], dtype=np.float32)
    return {
        "score": scores,
        "no_at_fault_collisions": np.asarray(
            [res.no_at_fault_collisions for res in results], dtype=np.float32
        ),
        "drivable_area_compliance": np.asarray(
            [res.drivable_area_compliance for res in results], dtype=np.float32
        ),
        "ego_progress": np.asarray(
            [res.ego_progress for res in results], dtype=np.float32
        ),
        "time_to_collision_within_bound": np.asarray(
            [res.time_to_collision_within_bound for res in results], dtype=np.float32
        ),
        "comfort": np.asarray([res.comfort for res in results], dtype=np.float32),
        "driving_direction_compliance": np.asarray(
            [res.driving_direction_compliance for res in results], dtype=np.float32
        ),
    }


def _select_keys(
    data: Dict[str, Any], keys: Iterable[str]
) -> Dict[str, Any]:
    return {key: value for key, value in data.items() if key in keys}


def _select_pred_keys(
    data: Dict[str, Any], keys: Iterable[str]
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key in keys:
        if key not in data:
            continue
        value = data[key]
        if torch.is_tensor(value):
            value = value.detach().cpu()
            if value.dim() >= 1 and value.shape[0] == 1:
                value = value.squeeze(0)
        out[key] = value
    return out


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    cache_root = Path(cfg.cache_path)
    if not cache_root.is_dir():
        raise FileNotFoundError(f"Cache path does not exist: {cache_root}")

    output_root = Path(
        getattr(cfg, "scorer_cache_path", None)
        or str(cache_root) + "_scorer"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    force = bool(getattr(cfg, "scorer_cache_force", False))

    metric_cache_path = _get_metric_cache_path(cfg)
    if not metric_cache_path:
        raise ValueError("pdm_metric_cache_path is required to build scorer cache")
    metric_cache_loader = MetricCacheLoader(Path(metric_cache_path))

    agent: AbstractAgent = instantiate(cfg.agent)
    agent.initialize()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent.to(device)
    use_train_mode = bool(getattr(cfg, "scorer_cache_use_train_mode", False))
    force_train_forward = bool(
        getattr(cfg, "scorer_cache_force_train_forward", False)
    )
    if force_train_forward:
        try:
            agent._config.force_train_forward = True
        except Exception:
            pass
        try:
            agent._transfuser_model._config.force_train_forward = True
        except Exception:
            pass
    if use_train_mode:
        agent.train()
        logger.info("Scorer cache using train mode to expose all modes.")
        if force_train_forward:
            logger.info(
                "scorer_cache_force_train_forward ignored because train mode is enabled."
            )
    else:
        agent.eval()
        if force_train_forward:
            logger.info(
                "Scorer cache using eval mode with train forward path (dropout disabled)."
            )

    traj_sampling = agent._config.trajectory_sampling
    pdm_num_poses = int(getattr(cfg, "scorer_cache_pdm_num_poses", 40))
    pdm_interval = float(getattr(cfg, "scorer_cache_pdm_interval_length", 0.1))
    pdm_sampling = TrajectorySampling(
        num_poses=pdm_num_poses, interval_length=pdm_interval
    )
    logger.info(
        "Scorer cache sampling: traj=%s@%s, pdm=%s@%s",
        traj_sampling.num_poses,
        traj_sampling.interval_length,
        pdm_sampling.num_poses,
        pdm_sampling.interval_length,
    )
    simulator = PDMSimulator(pdm_sampling)
    if PDMScorerConfig is None:
        scorer = PDMScorer(pdm_sampling)
    else:
        scorer = PDMScorer(pdm_sampling, PDMScorerConfig())
    component_weights = np.asarray(
        getattr(agent._config, "pdm_score_component_weights", (1.0,) * len(PDM_COMPONENT_KEYS)),
        dtype=np.float32,
    )
    use_pdm_score = bool(getattr(cfg, "scorer_cache_use_pdm_score", False))
    validate_gt = bool(getattr(cfg, "scorer_cache_validate_gt", False))
    drop_invalid_gt = bool(getattr(cfg, "scorer_cache_drop_invalid_gt", False))

    # NOTE: This script writes a minimal cache (token + pdm_score_components).
    # Additional feature/target keys are intentionally ignored to keep the cache small.

    dataset = CacheOnlyDataset(
        cache_path=str(cache_root),
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
    )
    total = 0
    skipped = 0
    missing_metric = 0
    saved = 0
    gt_invalid_count = 0
    logged_shapes = False

    for token in dataset.tokens:
        token_path = dataset._valid_cache_paths[token]
        log_name = token_path.parent.name
        out_dir = output_root / log_name / token
        out_path = out_dir / "scorer_cache.gz"

        if out_path.exists() and not force:
            skipped += 1
            continue

        metric_cache_path = metric_cache_loader.metric_cache_paths.get(token)
        if metric_cache_path is None:
            missing_metric += 1
            continue

        total += 1
        features_raw, targets_raw = dataset._load_scene_with_token(token)
        features = _add_batch_dim(features_raw)
        targets = _add_batch_dim(targets_raw)
        features = _to_device(features, device)
        targets = _to_device(targets, device)

        with torch.no_grad():
            predictions = agent.forward(features, targets)

        poses_reg = predictions.get("poses_reg")
        poses_cls = predictions.get("poses_cls")
        if poses_reg is None:
            traj = predictions.get("trajectory")
            if traj is None:
                logger.warning("Missing poses_reg/trajectory for token %s", token)
                continue
            poses_reg = traj.unsqueeze(1)
            if not logged_shapes:
                logger.info(
                    "Scorer cache using fallback trajectory; poses_reg shape=%s",
                    tuple(poses_reg.shape),
                )
                logger.info(
                    "Scorer cache config ego_fut_mode=%s",
                    getattr(agent._config, "ego_fut_mode", None),
                )
                logged_shapes = True
        elif not logged_shapes:
            logger.info(
                "Scorer cache poses_reg shape=%s (num_modes=%s)",
                tuple(poses_reg.shape),
                poses_reg.shape[1] if poses_reg.dim() > 1 else "n/a",
            )
            logger.info(
                "Scorer cache config ego_fut_mode=%s",
                getattr(agent._config, "ego_fut_mode", None),
            )
            logged_shapes = True
        poses_reg_cpu = poses_reg.detach().cpu().numpy()
        if poses_reg_cpu.ndim == 4:
            poses_reg_cpu = poses_reg_cpu[0]
        poses_reg_cpu = _ensure_heading(poses_reg_cpu.astype(np.float32))

        with lzma.open(metric_cache_path, "rb") as f:
            metric_cache = pickle.load(f)
        if validate_gt and "trajectory" in targets_raw:
            gt_traj = targets_raw["trajectory"]
            if torch.is_tensor(gt_traj):
                gt_traj = gt_traj.detach().cpu().numpy()
            if gt_traj.ndim == 2:
                gt_traj = gt_traj[None, ...]
            gt_traj = _ensure_heading(gt_traj.astype(np.float32))
            if use_pdm_score:
                gt_components = _score_pdm_components_via_pdm_score(
                    metric_cache, gt_traj, traj_sampling, simulator, scorer
                )
            else:
                gt_components = _score_pdm_components(
                    metric_cache, gt_traj, traj_sampling, simulator, scorer
                )
            gt_component_array = np.stack(
                [gt_components[key] for key in PDM_COMPONENT_KEYS], axis=-1
            ).astype(np.float32)
            gt_invalid_flag = not np.isfinite(gt_component_array).all()
            if not gt_invalid_flag:
                # Treat zero on key safety components as invalid GT.
                gt_critical = gt_component_array[:, [0, 1]]
                gt_invalid_flag = np.any(gt_critical <= 0.0)
            if gt_invalid_flag:
                logger.warning("Scorer cache GT PDM invalid for token %s", token)
                gt_invalid_count += 1
                if drop_invalid_gt:
                    skipped += 1
                    continue
            if total % 100 == 0:
                logger.info(
                    "Scorer cache progress=%s, gt_invalid=%s (%.2f%%)",
                    total,
                    gt_invalid_count,
                    100.0 * gt_invalid_count / max(total, 1),
                )
        if use_pdm_score:
            pdm_components = _score_pdm_components_via_pdm_score(
                metric_cache, poses_reg_cpu, traj_sampling, simulator, scorer
            )
        else:
            pdm_components = _score_pdm_components(
                metric_cache, poses_reg_cpu, traj_sampling, simulator, scorer
            )

        poses_cls_cpu = None
        if poses_cls is not None:
            poses_cls_cpu = poses_cls.detach().cpu()
            if poses_cls_cpu.dim() == 2:
                poses_cls_cpu = poses_cls_cpu[0]
        component_array = np.stack(
            [pdm_components[key] for key in PDM_COMPONENT_KEYS], axis=-1
        ).astype(np.float32)
        component_valid = np.isfinite(component_array).all(axis=-1)
        component_array[~component_valid] = np.nan
        drivor_score = _aggregate_drivor_score(component_array, component_weights)
        drivor_score[~component_valid] = np.nan
        save_payload: Dict[str, Any] = {
            "token": _normalize_token(targets_raw.get("token"), token),
            "poses_reg": torch.tensor(poses_reg_cpu),
            "pdm_score_components": torch.tensor(component_array),
            "pdm_score": torch.tensor(drivor_score),
            "pdm_score_valid": torch.tensor(component_valid),
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        dump_feature_target_to_pickle(out_path, save_payload)
        saved += 1

    logger.info(
        "Scorer cache done. total=%s saved=%s skipped=%s missing_metric=%s",
        total,
        saved,
        skipped,
        missing_metric,
    )
    if total > 0:
        logger.info(
            "Scorer cache GT invalid=%s (%.2f%% of processed)",
            gt_invalid_count,
            100.0 * gt_invalid_count / total,
        )


if __name__ == "__main__":
    main()
