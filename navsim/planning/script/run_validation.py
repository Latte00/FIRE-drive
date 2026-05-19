from typing import Optional
from pathlib import Path
import logging

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import SceneFilter
from navsim.common.dataloader import SceneLoader
from navsim.planning.training.dataset import CacheOnlyDataset, Dataset
from navsim.planning.training.agent_lightning_module import AgentLightningModule

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/training"
CONFIG_NAME = "default_training"


def build_val_dataset(cfg: DictConfig, agent: AbstractAgent) -> Dataset:
    """Build a validation dataset from the omega config."""
    val_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    if val_scene_filter.log_names is not None:
        val_scene_filter.log_names = [
            log_name for log_name in val_scene_filter.log_names if log_name in cfg.val_logs
        ]
    else:
        val_scene_filter.log_names = cfg.val_logs

    data_path = Path(cfg.navsim_log_path)
    sensor_blobs_path = Path(cfg.sensor_blobs_path)

    val_scene_loader = SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=data_path,
        scene_filter=val_scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    return Dataset(
        scene_loader=val_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
    )


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """Validation-only entrypoint for an agent."""
    pl.seed_everything(cfg.seed, workers=True)
    logger.info(f"Global Seed set to {cfg.seed}")
    logger.info(f"Path where all results are stored: {cfg.output_dir}")

    logger.info("Building Agent")
    agent: AbstractAgent = instantiate(cfg.agent)

    logger.info("Building Lightning Module")
    lightning_module = AgentLightningModule(agent=agent)

    if cfg.use_cache_without_dataset:
        logger.info("Using cached data without building SceneLoader")
        assert (
            not cfg.force_cache_computation
        ), "force_cache_computation must be False when using cached data without building SceneLoader"
        assert (
            cfg.cache_path is not None
        ), "cache_path must be provided when using cached data without building SceneLoader"
        scorer_cache_path = getattr(cfg, "scorer_cache_path", None)
        scorer_cache_filename = getattr(cfg, "scorer_cache_filename", "scorer_cache.gz")
        scorer_cache_require = bool(getattr(cfg, "scorer_cache_require", False))
        val_data = CacheOnlyDataset(
            cache_path=cfg.cache_path,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            log_names=cfg.val_logs,
            scorer_cache_path=scorer_cache_path,
            scorer_cache_filename=scorer_cache_filename,
            require_scorer_cache=scorer_cache_require,
        )
    else:
        logger.info("Building SceneLoader")
        val_data = build_val_dataset(cfg, agent)

    logger.info("Building Datasets")
    val_dataloader = DataLoader(val_data, **cfg.dataloader.params, shuffle=False)
    logger.info("Num validation samples: %d", len(val_data))

    logger.info("Building Trainer")
    trainer_params = dict(cfg.trainer.params)
    ckpt_path: Optional[str] = trainer_params.pop("resume_from_checkpoint", None)
    if ckpt_path is None:
        ckpt_path = trainer_params.pop("ckpt_path", None)
    trainer = pl.Trainer(**trainer_params)

    logger.info("Starting Validation")
    validate_kwargs = dict(
        model=lightning_module,
        dataloaders=val_dataloader,
    )
    if ckpt_path:
        validate_kwargs["ckpt_path"] = ckpt_path
    trainer.validate(**validate_kwargs)


if __name__ == "__main__":
    main()
