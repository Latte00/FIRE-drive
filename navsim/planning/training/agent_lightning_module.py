import logging

import pytorch_lightning as pl
import torch

from torch import Tensor
from typing import Dict, Tuple

from navsim.agents.abstract_agent import AbstractAgent

logger = logging.getLogger(__name__)


class AgentLightningModule(pl.LightningModule):
    """Pytorch lightning wrapper for learnable agent."""

    def __init__(self, agent: AbstractAgent):
        """
        Initialise the lightning module wrapper.
        :param agent: agent interface in NAVSIM
        """
        super().__init__()
        self.agent = agent
        self._nonfinite_loss_warning_count = 0

    def _config_value(self, name: str, default):
        return getattr(getattr(self.agent, "_config", None), name, default)

    @staticmethod
    def _finite_or_none(value: Tensor) -> bool:
        return torch.is_tensor(value) and torch.isfinite(value.detach()).all().item()

    def _should_log_loss_debug(self) -> bool:
        if not bool(self._config_value("loss_debug_log_enable", False)):
            return False
        every_n = max(1, int(self._config_value("loss_debug_log_every_n_steps", 50)))
        step = int(getattr(self, "global_step", 0) or 0)
        return step < 10 or step % every_n == 0

    def _log_tensor_debug_stats(
        self,
        logging_prefix: str,
        name: str,
        value: Tensor,
        batch_size: int,
    ) -> None:
        if value is None or not torch.is_tensor(value):
            return
        tensor = value.detach()
        numel = int(tensor.numel())
        ref = tensor.reshape(-1)[0] if numel > 0 else next(self.parameters()).detach()
        if numel == 0:
            finite_ratio = ref.new_tensor(1.0, dtype=torch.float32)
            nonfinite_count = ref.new_tensor(0.0, dtype=torch.float32)
            absmax = ref.new_tensor(0.0, dtype=torch.float32)
            zero_ratio = ref.new_tensor(1.0, dtype=torch.float32)
        else:
            tensor_float = tensor.float()
            finite_mask = torch.isfinite(tensor_float)
            finite_ratio = finite_mask.float().mean()
            nonfinite_count = (~finite_mask).float().sum()
            clean = torch.nan_to_num(tensor_float, nan=0.0, posinf=0.0, neginf=0.0)
            absmax = clean.abs().amax()
            zero_ratio = ((clean == 0.0) & finite_mask).float().mean()

        safe_name = name.replace(".", "_")
        log_kwargs = dict(
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(f"{logging_prefix}/debug/{safe_name}_finite_ratio", finite_ratio, **log_kwargs)
        self.log(f"{logging_prefix}/debug/{safe_name}_nonfinite_count", nonfinite_count, **log_kwargs)
        self.log(f"{logging_prefix}/debug/{safe_name}_absmax", absmax, **log_kwargs)
        self.log(f"{logging_prefix}/debug/{safe_name}_zero_ratio", zero_ratio, **log_kwargs)

    def _log_debug_tensor_dict(
        self,
        logging_prefix: str,
        scope: str,
        values: Dict[str, Tensor],
        batch_size: int,
    ) -> None:
        max_tensors = max(1, int(self._config_value("loss_debug_max_tensors", 24)))
        logged = 0
        for key, value in values.items():
            if not torch.is_tensor(value):
                continue
            self._log_tensor_debug_stats(
                logging_prefix,
                f"{scope}/{key}",
                value,
                batch_size,
            )
            logged += 1
            if logged >= max_tensors:
                break

    def _log_loss_debug_stats(
        self,
        logging_prefix: str,
        features: Dict[str, Tensor],
        targets: Dict[str, Tensor],
        prediction: Dict[str, Tensor],
        batch_size: int,
    ) -> None:
        if not self._should_log_loss_debug():
            return
        self._log_debug_tensor_dict(logging_prefix, "feature", features, batch_size)
        self._log_debug_tensor_dict(logging_prefix, "prediction", prediction, batch_size)
        self._log_debug_tensor_dict(logging_prefix, "target", targets, batch_size)

    def _sanitize_loss_dict(
        self,
        loss_dict: Dict[str, Tensor],
        logging_prefix: str,
        batch_size: int,
    ) -> Tensor:
        """Log finite losses and skip optimizer updates for non-finite batches."""
        ref_tensor = None
        for value in loss_dict.values():
            if torch.is_tensor(value):
                ref_tensor = value
                break
        if ref_tensor is None:
            ref_tensor = next(self.parameters())

        loss = loss_dict.get("loss")
        loss_is_finite = self._finite_or_none(loss)
        nonfinite_flag = ref_tensor.detach().new_tensor(0.0 if loss_is_finite else 1.0)
        guarded_zero_flag = ref_tensor.detach().new_tensor(0.0)
        zero_scalar_loss_count = ref_tensor.detach().new_tensor(0.0)
        nonfinite_keys = []
        nonfinite_details = []
        guard_enable = bool(self._config_value("loss_nonfinite_guard_enable", False))

        for key, value in loss_dict.items():
            if value is None or not torch.is_tensor(value):
                continue
            log_value = value.detach()
            if not torch.isfinite(log_value).all().item():
                nonfinite_keys.append(key)
                log_value_float = log_value.float()
                bad_count = int((~torch.isfinite(log_value_float)).sum().item())
                clean = torch.nan_to_num(log_value_float, nan=0.0, posinf=0.0, neginf=0.0)
                absmax = float(clean.abs().amax().item()) if clean.numel() > 0 else 0.0
                nonfinite_details.append(
                    f"{key}:shape={tuple(value.shape)},bad={bad_count},absmax={absmax:.4g}"
                )
                if guard_enable:
                    log_value = torch.nan_to_num(log_value, nan=0.0, posinf=0.0, neginf=0.0)
            elif log_value.numel() == 1 and float(log_value.float().abs().item()) == 0.0:
                zero_scalar_loss_count = zero_scalar_loss_count + 1.0
            self.log(
                f"{logging_prefix}/{key}",
                log_value,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
                batch_size=batch_size,
            )

        if nonfinite_keys:
            self._nonfinite_loss_warning_count += 1
            if self._nonfinite_loss_warning_count <= 20 or self._nonfinite_loss_warning_count % 100 == 0:
                logger.warning(
                    "Non-finite %s loss at global_step=%s keys=%s",
                    logging_prefix,
                    getattr(self, "global_step", "unknown"),
                    "; ".join(nonfinite_details) if nonfinite_details else ",".join(nonfinite_keys),
                )

        self.log(
            f"{logging_prefix}/nonfinite_loss",
            nonfinite_flag,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
            batch_size=batch_size,
        )
        if not loss_is_finite and guard_enable:
            guarded_zero_flag = ref_tensor.detach().new_tensor(1.0)
        self.log(
            f"{logging_prefix}/guarded_zero_loss",
            guarded_zero_flag,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            f"{logging_prefix}/zero_scalar_loss_count",
            zero_scalar_loss_count,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
            batch_size=batch_size,
        )

        if loss_is_finite:
            return loss
        if not guard_enable:
            return loss
        # Keep the graph connected while producing zero gradients for this batch.
        return torch.nan_to_num(ref_tensor, nan=0.0, posinf=0.0, neginf=0.0).sum() * 0.0

    def _step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], logging_prefix: str) -> Tensor:
        """
        Propagates the model forward and backwards and computes/logs losses and metrics.
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param logging_prefix: prefix where to log step
        :return: scalar loss
        """
        features, targets = batch
        prediction = self.agent.forward(features, targets)
        # loss = self.agent.compute_loss(features, targets, prediction)
        # self.log(f"{logging_prefix}/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        # return loss
        loss_dict = self.agent.compute_loss(features, targets, prediction)
        self._log_loss_debug_stats(logging_prefix, features, targets, prediction, len(batch[0]))
        return self._sanitize_loss_dict(loss_dict, logging_prefix, len(batch[0]))

    def training_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int) -> Tensor:
        """
        Step called on training samples
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param batch_idx: index of batch (ignored)
        :return: scalar loss
        """
        return self._step(batch, "train")

    def validation_step(
        self,
        batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]],
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        """
        Step called on validation samples
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param batch_idx: index of batch (ignored)
        :return: scalar loss
        """
        if dataloader_idx == 0:
            logging_prefix = "val"
        elif dataloader_idx == 1:
            logging_prefix = "monitor"
        else:
            logging_prefix = f"monitor{dataloader_idx}"

        monitor_cache_override = None
        if dataloader_idx == 1:
            monitor_cache_override = getattr(
                getattr(self.agent, "_config", None),
                "pdm_metric_cache_path_monitor",
                None,
            )

        features, targets = batch
        prediction = self.agent.forward(features, targets)
        if monitor_cache_override:
            try:
                loss_dict = self.agent.compute_loss(
                    features,
                    targets,
                    prediction,
                    metric_cache_path_override=monitor_cache_override,
                )
            except TypeError:
                loss_dict = self.agent.compute_loss(features, targets, prediction)
        else:
            loss_dict = self.agent.compute_loss(features, targets, prediction)
        self._log_loss_debug_stats(logging_prefix, features, targets, prediction, len(batch[0]))
        loss = self._sanitize_loss_dict(loss_dict, logging_prefix, len(batch[0]))
        skip_primary_pdm = bool(
            getattr(getattr(self.agent, "_config", None), "pdm_val_skip_primary_loader", False)
        )
        should_log_pdm = not (dataloader_idx == 0 and skip_primary_pdm)
        metrics_prediction = prediction
        if should_log_pdm and bool(
            getattr(
                getattr(self.agent, "_config", None),
                "pdm_val_metrics_use_inference_forward",
                False,
            )
        ):
            with torch.no_grad():
                try:
                    metrics_prediction = self.agent.forward(features)
                except TypeError:
                    metrics_prediction = self.agent.forward(features, None)
        if should_log_pdm and hasattr(self.agent, "compute_pdm_val_metrics"):
            if monitor_cache_override:
                try:
                    metrics = self.agent.compute_pdm_val_metrics(
                        targets,
                        metrics_prediction,
                        metric_cache_path_override=monitor_cache_override,
                    )
                except TypeError:
                    metrics = self.agent.compute_pdm_val_metrics(
                        targets, metrics_prediction
                    )
            else:
                metrics = self.agent.compute_pdm_val_metrics(
                    targets, metrics_prediction
                )
            if metrics:
                for key, value in metrics.items():
                    self.log(
                        f"{logging_prefix}/{key}",
                        value,
                        on_step=False,
                        on_epoch=True,
                        prog_bar=True,
                        sync_dist=True,
                        batch_size=len(batch[0]),
                    )
        return loss

    def configure_optimizers(self):
        """Inherited, see superclass."""
        return self.agent.get_optimizers()
