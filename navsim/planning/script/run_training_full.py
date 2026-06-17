from typing import List, Optional, Tuple
from pathlib import Path
import logging
import json

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torch.utils.data import Dataset as TorchDataset

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import SceneFilter
from navsim.common.dataloader import SceneLoader
from navsim.planning.training.dataset import CacheOnlyDataset, Dataset
from navsim.planning.training.agent_lightning_module import AgentLightningModule

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/training"
CONFIG_NAME = "default_training"


class _EmptyDataset(TorchDataset):
    """Dataset placeholder used to keep dataloader indices stable."""

    def __len__(self) -> int:
        return 0

    def __getitem__(self, index: int):
        raise IndexError("Empty dataset")


class _AdaptiveValFrequencyCallback(pl.Callback):
    """Adjust validation frequency during training based on epoch."""

    def __init__(
        self,
        switch_epoch: int,
        every_n_epoch_before: int,
        every_n_epoch_after: int,
    ) -> None:
        super().__init__()
        self.switch_epoch = max(1, int(switch_epoch))
        self.every_n_epoch_before = max(1, int(every_n_epoch_before))
        self.every_n_epoch_after = max(1, int(every_n_epoch_after))

    def _target_interval(self, epoch_idx_0based: int) -> int:
        epoch_1based = int(epoch_idx_0based) + 1
        if epoch_1based < self.switch_epoch:
            return self.every_n_epoch_before
        return self.every_n_epoch_after

    def _apply(self, trainer: pl.Trainer, stage: str) -> None:
        target = self._target_interval(int(trainer.current_epoch))
        current = max(1, int(getattr(trainer, "check_val_every_n_epoch", 1)))
        if current != target:
            trainer.check_val_every_n_epoch = target
            logger.info(
                "Adaptive validation frequency (%s): epoch=%d, check_val_every_n_epoch %d -> %d",
                stage,
                int(trainer.current_epoch) + 1,
                current,
                target,
            )

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._apply(trainer, stage="fit_start")

    def on_train_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        self._apply(trainer, stage="train_epoch_start")


def load_token_subset(cfg: DictConfig) -> Optional[List[str]]:
    subset_json = getattr(cfg, "train_subset_json", None)
    if not subset_json:
        return None

    subset_path = Path(str(subset_json)).expanduser()
    if not subset_path.exists():
        raise FileNotFoundError(f"train_subset_json not found: {subset_path}")

    subset_key = str(getattr(cfg, "train_subset_key", "mixed_tokens"))
    data = json.loads(subset_path.read_text(encoding="utf-8"))
    tokens = data.get(subset_key)
    if not isinstance(tokens, list):
        raise ValueError(
            f"Subset key '{subset_key}' missing or not a list in {subset_path}"
        )
    tokens = [str(token) for token in tokens]
    logger.info(
        "Loaded training subset tokens: key=%s count=%d file=%s",
        subset_key,
        len(tokens),
        subset_path,
    )
    return tokens


def _merge_train_logs(cfg: DictConfig) -> List[str]:
    """Merge train/val logs for full training while preserving order and uniqueness."""
    merged: List[str] = []
    seen = set()
    for name in list(cfg.train_logs) + list(cfg.val_logs):
        if name in seen:
            continue
        seen.add(name)
        merged.append(name)
    return merged


def build_datasets(
    cfg: DictConfig,
    agent: AbstractAgent,
    build_val: bool = True,
    train_subset_tokens: Optional[List[str]] = None,
) -> Tuple[Dataset, Optional[Dataset]]:
    """
    Builds training and validation datasets from omega config.
    Full-training behavior: train split uses train_logs + val_logs.
    """
    full_train_logs = _merge_train_logs(cfg)

    train_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    if train_scene_filter.log_names is not None:
        train_scene_filter.log_names = [
            log_name for log_name in train_scene_filter.log_names if log_name in full_train_logs
        ]
    else:
        train_scene_filter.log_names = full_train_logs
    if train_subset_tokens is not None:
        train_scene_filter.tokens = train_subset_tokens

    data_path = Path(cfg.navsim_log_path)
    sensor_blobs_path = Path(cfg.sensor_blobs_path)

    train_scene_loader = SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=data_path,
        scene_filter=train_scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    train_data = Dataset(
        scene_loader=train_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
    )

    if not build_val:
        return train_data, None

    val_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    if val_scene_filter.log_names is not None:
        val_scene_filter.log_names = [log_name for log_name in val_scene_filter.log_names if log_name in cfg.val_logs]
    else:
        val_scene_filter.log_names = cfg.val_logs

    val_scene_loader = SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=data_path,
        scene_filter=val_scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    val_data = Dataset(
        scene_loader=val_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
    )

    return train_data, val_data


def build_dataset_from_logs(
    cfg: DictConfig, agent: AbstractAgent, log_names: list[str]
) -> Dataset:
    """
    Builds a dataset from explicit log names using the same scene filter template.
    """
    scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    if scene_filter.log_names is not None:
        scene_filter.log_names = [
            log_name for log_name in scene_filter.log_names if log_name in log_names
        ]
    else:
        scene_filter.log_names = log_names

    data_path = Path(cfg.navsim_log_path)
    sensor_blobs_path = Path(cfg.sensor_blobs_path)
    scene_loader = SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=data_path,
        scene_filter=scene_filter,
        sensor_config=agent.get_sensor_config(),
    )
    return Dataset(
        scene_loader=scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
    )


def resolve_monitor_logs(cfg: DictConfig) -> list[str]:
    """
    Resolve log names for monitor dataloader.
    Priority:
      1) monitor_logs (explicit)
      2) test_logs (when monitor_use_test_logs=True)
      3) empty list (disabled)
    """
    if not bool(getattr(cfg, "monitor_enable", False)):
        return []

    explicit_logs = getattr(cfg, "monitor_logs", None)
    if explicit_logs is not None and len(explicit_logs) > 0:
        return list(explicit_logs)

    if bool(getattr(cfg, "monitor_use_test_logs", True)):
        test_logs = getattr(cfg, "test_logs", None)
        if test_logs is not None and len(test_logs) > 0:
            return list(test_logs)

    return []


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for training an agent.
    :param cfg: omegaconf dictionary
    """

    pl.seed_everything(cfg.seed, workers=True)
    logger.info(f"Global Seed set to {cfg.seed}")
    logger.info(f"Path where all results are stored: {cfg.output_dir}")

    logger.info("Building Agent")
    agent: AbstractAgent = instantiate(cfg.agent)
    train_subset_tokens = load_token_subset(cfg)
    skip_primary_val_loader = bool(
        getattr(getattr(agent, "_config", None), "pdm_val_skip_primary_loader", False)
    )

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
        cache_validation_kwargs = dict(
            validate_cache=bool(getattr(cfg, "cache_validate", False)),
            validation_max_attempts=int(getattr(cfg, "cache_validation_max_attempts", 16)),
            validation_max_trajectory_translation=float(
                getattr(cfg, "cache_validation_max_trajectory_translation", 120.0)
            ),
            validation_max_trajectory_step=float(
                getattr(cfg, "cache_validation_max_trajectory_step", 60.0)
            ),
            validation_max_status_abs=float(getattr(cfg, "cache_validation_max_status_abs", 100.0)),
        )
        if cache_validation_kwargs["validate_cache"]:
            logger.info("Enabled runtime cache validation: %s", cache_validation_kwargs)
        train_data = CacheOnlyDataset(
            cache_path=cfg.cache_path,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            log_names=_merge_train_logs(cfg),
            tokens=train_subset_tokens,
            scorer_cache_path=scorer_cache_path,
            scorer_cache_filename=scorer_cache_filename,
            require_scorer_cache=scorer_cache_require,
            **cache_validation_kwargs,
        )
        if skip_primary_val_loader:
            val_data = None
        else:
            val_data = CacheOnlyDataset(
                cache_path=cfg.cache_path,
                feature_builders=agent.get_feature_builders(),
                target_builders=agent.get_target_builders(),
                log_names=cfg.val_logs,
                scorer_cache_path=scorer_cache_path,
                scorer_cache_filename=scorer_cache_filename,
                require_scorer_cache=scorer_cache_require,
                **cache_validation_kwargs,
            )
    else:
        logger.info("Building SceneLoader")
        train_data, val_data = build_datasets(
            cfg,
            agent,
            build_val=not skip_primary_val_loader,
            train_subset_tokens=train_subset_tokens,
        )

    monitor_data = None
    monitor_logs = resolve_monitor_logs(cfg)
    if monitor_logs:
        logger.info("Building monitor dataset from %d logs", len(monitor_logs))
        if cfg.use_cache_without_dataset:
            scorer_cache_path = getattr(cfg, "scorer_cache_path", None)
            scorer_cache_filename = getattr(cfg, "scorer_cache_filename", "scorer_cache.gz")
            scorer_cache_require = bool(getattr(cfg, "scorer_cache_require", False))
            monitor_data = CacheOnlyDataset(
                cache_path=cfg.cache_path,
                feature_builders=agent.get_feature_builders(),
                target_builders=agent.get_target_builders(),
                log_names=monitor_logs,
                scorer_cache_path=scorer_cache_path,
                scorer_cache_filename=scorer_cache_filename,
                require_scorer_cache=scorer_cache_require,
                **cache_validation_kwargs,
            )
        else:
            monitor_data = build_dataset_from_logs(cfg, agent, monitor_logs)
    elif bool(getattr(cfg, "monitor_enable", False)):
        logger.warning(
            "monitor_enable=True but no monitor logs found. "
            "Set monitor_logs or monitor_use_test_logs with available test_logs."
        )

    logger.info("Building Datasets")
    train_dataloader = DataLoader(train_data, **cfg.dataloader.params, shuffle=True)
    logger.info("Num training samples: %d", len(train_data))
    val_dataloader = None
    if val_data is not None:
        val_dataloader = DataLoader(val_data, **cfg.dataloader.params, shuffle=False)
        logger.info("Num validation samples: %d", len(val_data))
    elif skip_primary_val_loader:
        logger.info(
            "Primary validation dataloader (dataloader_idx=0) is disabled by "
            "agent.config.pdm_val_skip_primary_loader=true"
        )

    monitor_dataloader = None
    if monitor_data is not None:
        monitor_dataloader = DataLoader(
            monitor_data, **cfg.dataloader.params, shuffle=False
        )
        logger.info("Num monitor samples: %d", len(monitor_data))
        if skip_primary_val_loader and val_dataloader is None:
            # Keep monitor as dataloader_idx=1 for metric naming/checkpoint monitor compatibility.
            val_dataloader = DataLoader(
                _EmptyDataset(), **cfg.dataloader.params, shuffle=False
            )
            logger.info(
                "Inserted empty primary val dataloader to keep monitor as dataloader_idx=1."
            )

    logger.info("Building Trainer")
    trainer_params = dict(cfg.trainer.params)
    ckpt_path = trainer_params.pop("resume_from_checkpoint", None)
    if ckpt_path is None:
        ckpt_path = trainer_params.pop("ckpt_path", None)
    callbacks = list(agent.get_training_callbacks())
    monitor_schedule_cfg = getattr(cfg, "monitor_schedule", None)
    if monitor_schedule_cfg is not None and bool(
        getattr(monitor_schedule_cfg, "enable", False)
    ):
        adaptive_cb = _AdaptiveValFrequencyCallback(
            switch_epoch=int(getattr(monitor_schedule_cfg, "switch_epoch", 50)),
            every_n_epoch_before=int(
                getattr(monitor_schedule_cfg, "every_n_epoch_before", 1)
            ),
            every_n_epoch_after=int(
                getattr(monitor_schedule_cfg, "every_n_epoch_after", 1)
            ),
        )
        callbacks.append(adaptive_cb)
        logger.info(
            "Enabled adaptive validation/monitor frequency: before epoch %d every %d epoch(s), "
            "from epoch %d every %d epoch(s).",
            int(getattr(monitor_schedule_cfg, "switch_epoch", 50)),
            int(getattr(monitor_schedule_cfg, "every_n_epoch_before", 1)),
            int(getattr(monitor_schedule_cfg, "switch_epoch", 50)),
            int(getattr(monitor_schedule_cfg, "every_n_epoch_after", 1)),
        )
    trainer = pl.Trainer(**trainer_params, callbacks=callbacks)

    logger.info("Starting Training")
    if monitor_dataloader is not None and val_dataloader is not None:
        val_loaders = [val_dataloader, monitor_dataloader]
    elif monitor_dataloader is not None:
        val_loaders = monitor_dataloader
    else:
        val_loaders = val_dataloader
    fit_kwargs = dict(
        model=lightning_module,
        train_dataloaders=train_dataloader,
    )
    if val_loaders is not None:
        fit_kwargs["val_dataloaders"] = val_loaders
    else:
        logger.warning(
            "No validation dataloaders configured. Validation/monitor loops will be skipped."
        )
    if ckpt_path:
        fit_kwargs["ckpt_path"] = ckpt_path
    trainer.fit(**fit_kwargs)


if __name__ == "__main__":
    main()
