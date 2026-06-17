from __future__ import annotations

from pathlib import Path
from typing import Any, Optional
import logging
import os

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from navsim.agents.diffusiondrive.modules.pdm_supervision import (
    PDMSupervision,
    PDMScoreConfig,
)
from navsim.planning.training.dataset import (
    load_feature_target_from_pickle,
    dump_feature_target_to_pickle,
)

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/training"
CONFIG_NAME = "default_training"


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


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    cache_root = Path(cfg.cache_path)
    if not cache_root.is_dir():
        raise FileNotFoundError(f"Cache path does not exist: {cache_root}")

    metric_cache_path = _get_metric_cache_path(cfg)
    if not metric_cache_path:
        raise ValueError("pdm_metric_cache_path is required to build offline labels")

    force = bool(getattr(cfg, "pdm_label_force", False))
    agent_cfg = getattr(cfg, "agent", None)
    cfg_config = getattr(agent_cfg, "config", None) if agent_cfg is not None else None
    use_ray = bool(getattr(cfg_config, "pdm_score_use_ray", False)) if cfg_config else False
    ray_threads = int(getattr(cfg_config, "pdm_score_ray_threads", 0)) if cfg_config else 0
    ray_debug = bool(getattr(cfg_config, "pdm_score_ray_debug", False)) if cfg_config else False
    cache_lru_size = int(getattr(cfg_config, "pdm_score_cache_lru_size", 0)) if cfg_config else 0

    supervisor = PDMSupervision(
        PDMScoreConfig(
            cache_path=metric_cache_path,
            use_ray=use_ray,
            ray_threads=ray_threads,
            ray_debug=ray_debug,
            cache_lru_size=cache_lru_size,
        )
    )

    total = 0
    updated = 0
    skipped = 0
    missing = 0

    for log_dir in cache_root.iterdir():
        if not log_dir.is_dir():
            continue
        for token_dir in log_dir.iterdir():
            if not token_dir.is_dir():
                continue
            target_path = token_dir / "transfuser_target.gz"
            if not target_path.is_file():
                continue
            total += 1
            targets = load_feature_target_from_pickle(target_path)
            if "trajectory_candidates" not in targets:
                missing += 1
                continue
            if "pdm_score_targets" in targets and not force:
                skipped += 1
                continue

            candidates = targets["trajectory_candidates"]
            cand_mask = targets.get("trajectory_candidates_mask")
            if torch.is_tensor(candidates):
                candidates = candidates.detach().cpu().numpy()
            if candidates.ndim == 3:
                candidates = candidates[None, ...]
            candidates = _ensure_heading(candidates.astype(np.float32))

            token = _normalize_token(targets.get("token"), token_dir.name)
            scores = supervisor.score_batch([token], candidates)
            score_vec = scores[0]
            if cand_mask is not None:
                if torch.is_tensor(cand_mask):
                    cand_mask = cand_mask.detach().cpu().numpy()
                if cand_mask.ndim == 2:
                    cand_mask = cand_mask[0]
                score_vec = np.where(cand_mask.astype(bool), score_vec, np.nan)

            targets["pdm_score_targets"] = torch.tensor(
                score_vec.astype(np.float32)
            )
            dump_feature_target_to_pickle(target_path, targets)
            updated += 1

    logger.info(
        "PDM label caching done. total=%s updated=%s skipped=%s missing=%s",
        total,
        updated,
        skipped,
        missing,
    )


if __name__ == "__main__":
    main()
