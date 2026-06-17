from typing import Any, List, Dict, Optional, Tuple, Union
import logging
import os
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig

from navsim.agents.diffusiondrive.transfuser_model_v2 import V2TransfuserModel as TransfuserModel

from navsim.agents.diffusiondrive.transfuser_callback import TransfuserCallback 
from navsim.agents.diffusiondrive.transfuser_loss import transfuser_loss
from navsim.agents.diffusiondrive.transfuser_features import TransfuserFeatureBuilder, TransfuserTargetBuilder
from navsim.agents.diffusiondrive.modules.pdm_supervision import (
    PDMSupervision,
    PDMScoreConfig,
)
from navsim.evaluate.pdm_score import pdm_score as evaluate_pdm_score
from navsim.common.dataclasses import SensorConfig
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder
from navsim.agents.diffusiondrive.modules.scheduler import WarmupCosLR
from omegaconf import DictConfig, OmegaConf, open_dict
import torch.optim as optim
from navsim.common.dataclasses import AgentInput, Trajectory, SensorConfig      

logger = logging.getLogger(__name__)

_PDM_COMPONENT_KEYS = (
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "ego_progress",
    "time_to_collision_within_bound",
    "comfort",
    "driving_direction_compliance",
)
def build_from_configs(obj, cfg: DictConfig, **kwargs):
    if cfg is None:
        return None
    cfg = cfg.copy()
    if isinstance(cfg, DictConfig):
        OmegaConf.set_struct(cfg, False)
    type = cfg.pop('type')
    return getattr(obj, type)(**cfg, **kwargs)

class TransfuserAgent(AbstractAgent):
    """Agent interface for TransFuser baseline."""

    def __init__(
        self,
        config: TransfuserConfig,
        lr: float,
        checkpoint_path: Optional[str] = None,
    ):
        """
        Initializes TransFuser agent.
        :param config: global config of TransFuser agent
        :param lr: learning rate during training
        :param checkpoint_path: optional path string to checkpoint, defaults to None
        """
        super().__init__()

        self._config = config
        self._lr = lr

        self._checkpoint_path = checkpoint_path
        if bool(getattr(self._config, "hardcase_gate_enable", False)) or bool(
            getattr(self._config, "hardcase_gate_supervision_enable", False)
        ) or bool(getattr(self._config, "hardcase_gate_train_only", False)):
            logger.warning(
                "Learned hardcase gate is deprecated in this branch and will be disabled. "
                "Use hardcase_score_residual_* with hardcase_r2se_enable instead."
            )
        self._config.hardcase_gate_enable = False
        self._config.hardcase_gate_supervision_enable = False
        self._config.hardcase_gate_supervision_weight = 0.0
        self._config.hardcase_gate_prior_weight = 0.0
        self._config.hardcase_gate_residual_reg_weight = 0.0
        self._config.hardcase_gate_train_only = False
        if bool(getattr(self._config, "hardcase_score_residual_enable", False)):
            if bool(getattr(self._config, "hardcase_specialist_enable", False)):
                logger.warning(
                    "hardcase_score_residual_enable=True: disabling trajectory specialist branch "
                    "(hardcase_specialist_enable=False) to keep score-only refinement."
                )
            self._config.hardcase_specialist_enable = False
        if getattr(self._config, "pdm_score_head_only", False):
            if not getattr(self._config, "pdm_score_use_head", False):
                self._config.pdm_score_use_head = True
        self._transfuser_model = self._build_model(config)
        self.init_from_pretrained()
        self._maybe_load_bev_semantic_pretrained()
        self._maybe_freeze_bev_semantic()
        self._maybe_freeze_pdm_score_head_only()
        self._maybe_freeze_score_residual_only()
        self._feasible_target_builder: Optional[TransfuserTargetBuilder] = None
        self._pdm_supervision_by_path: Dict[str, PDMSupervision] = {}
        self._pdm_val_invalid_batches = 0
        self._last_inference_debug: Dict[str, float] = {}
        self._r2se_gpd_params: Optional[Dict[str, float]] = None
        self._r2se_gpd_loaded: bool = False

    def _build_model(self, config: TransfuserConfig) -> torch.nn.Module:
        return TransfuserModel(config)

    def _log_pdm_val_invalid(
        self,
        targets: Dict[str, Any],
        predictions: Dict[str, torch.Tensor],
        reason: str,
    ) -> None:
        if self.training:
            return
        self._pdm_val_invalid_batches += 1
        if self._pdm_val_invalid_batches != 1 and self._pdm_val_invalid_batches % 100 != 0:
            return
        tokens = targets.get("token")
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.detach().cpu().tolist()
        elif isinstance(tokens, (str, bytes)):
            tokens = [tokens]
        token_count = len(tokens) if isinstance(tokens, (list, tuple)) else 0
        token_sample = tokens[:3] if isinstance(tokens, (list, tuple)) else None
        poses = predictions.get("poses_reg")
        if poses is not None:
            finite_mask = torch.isfinite(poses)
            non_finite = int((~finite_mask).sum().item())
            max_abs = float(poses.abs().max().item()) if poses.numel() else 0.0
        else:
            non_finite = -1
            max_abs = 0.0
        logger.warning(
            "PDM val invalid (%s): batches=%d tokens=%s count=%d non_finite=%d max_abs=%.3f metric_cache=%s",
            reason,
            self._pdm_val_invalid_batches,
            token_sample,
            token_count,
            non_finite,
            max_abs,
            self._config.pdm_metric_cache_path,
        )

    def init_from_pretrained(self):
        # import ipdb; ipdb.set_trace()
        if self._checkpoint_path:
            if torch.cuda.is_available():
                checkpoint = torch.load(self._checkpoint_path)
            else:
                checkpoint = torch.load(self._checkpoint_path, map_location=torch.device('cpu'))

            state_dict = checkpoint['state_dict']

            # Remove 'agent.' prefix from keys if present
            state_dict = {k.replace('agent.', ''): v for k, v in state_dict.items()}

            self._load_state_dict_flexible(state_dict)
        else:
            print("No checkpoint path provided. Initializing from scratch.")

    def _maybe_load_bev_semantic_pretrained(self) -> None:
        path = self._config.bev_semantic_pretrained_path
        if not path:
            return
        if torch.cuda.is_available():
            checkpoint = torch.load(path)
        else:
            checkpoint = torch.load(path, map_location=torch.device("cpu"))
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        filtered_state = {}
        skipped = []
        model_state = self._transfuser_model.state_dict()
        for key, value in state_dict.items():
            key = key.replace("agent.", "")
            if key.startswith("_transfuser_model."):
                key = key[len("_transfuser_model.") :]
            if not (key.startswith("_backbone.") or key.startswith("_bev_semantic_head.")):
                continue
            if key not in model_state or model_state[key].shape != value.shape:
                skipped.append(key)
                continue
            filtered_state[key] = value
        missing_keys, unexpected_keys = self._transfuser_model.load_state_dict(filtered_state, strict=False)
        if skipped:
            print(f"Skipped BEV semantic keys with shape mismatch: {skipped}")
        if missing_keys:
            print(f"Missing BEV semantic keys when loading pretrained weights: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected BEV semantic keys when loading pretrained weights: {unexpected_keys}")

    def _maybe_freeze_bev_semantic(self) -> None:
        if self._config.freeze_bev_semantic_backbone:
            for param in self._transfuser_model._backbone.parameters():
                param.requires_grad = False
        if self._config.freeze_bev_semantic_head:
            for param in self._transfuser_model._bev_semantic_head.parameters():
                param.requires_grad = False

    def _maybe_freeze_pdm_score_head_only(self) -> None:
        if not getattr(self._config, "pdm_score_head_only", False):
            return
        if float(getattr(self._config, "pdm_score_weight", 0.0)) <= 0.0:
            raise ValueError(
                "pdm_score_head_only=True requires pdm_score_weight > 0 to "
                "train the scorer head."
            )
        head = getattr(self._transfuser_model._trajectory_head, "_pdm_score_head", None)
        if head is None:
            raise ValueError(
                "pdm_score_head_only=True but pdm_score_head is not initialized. "
                "Enable pdm_score_use_head or set pdm_score_weight > 0."
            )
        for param in self._transfuser_model.parameters():
            param.requires_grad = False
        for param in head.parameters():
            param.requires_grad = True

    def _get_residual_train_modules(self) -> List[nn.Module]:
        traj_head = getattr(self._transfuser_model, "_trajectory_head", None)
        pdm_head = getattr(traj_head, "_pdm_score_head", None) if traj_head is not None else None
        if pdm_head is None:
            return []
        residual_module_names = (
            "_score_res_base_proj",
            "_score_res_scene_proj",
            "_score_res_proposal_proj",
            "_score_res_mode_context_proj",
            "_score_residual_head",
            "_score_residual_lora_adapters",
        )
        modules: List[nn.Module] = []
        for name in residual_module_names:
            module = getattr(pdm_head, name, None)
            if isinstance(module, nn.Module):
                modules.append(module)
        return modules

    def _enforce_frozen_module_modes(self) -> None:
        if not self.training:
            return
        if bool(getattr(self._config, "hardcase_score_residual_train_only", False)):
            # Keep the frozen generalist path in eval mode so BN/dropout states stay aligned
            # with the base checkpoint while training only the residual branch.
            self._transfuser_model.eval()
            for module in self._get_residual_train_modules():
                module.train()
            return
        if bool(getattr(self._config, "pdm_score_head_only", False)):
            self._transfuser_model.eval()
            head = getattr(self._transfuser_model._trajectory_head, "_pdm_score_head", None)
            if isinstance(head, nn.Module):
                head.train()

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            self._enforce_frozen_module_modes()
        return self

    def _load_state_dict_flexible(self, state_dict: Dict[str, Any]) -> None:
        model_state = self.state_dict()
        filtered_state = {}
        mismatched_keys = []
        for k, v in state_dict.items():
            if k not in model_state:
                continue
            if model_state[k].shape != v.shape:
                mismatched_keys.append(k)
                continue
            filtered_state[k] = v
        missing_keys, unexpected_keys = self.load_state_dict(filtered_state, strict=False)
        if mismatched_keys:
            print(f"Skipped keys with shape mismatch: {mismatched_keys}")
        if missing_keys:
            print(f"Missing keys when loading pretrained weights: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys when loading pretrained weights: {unexpected_keys}")
    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def initialize(self) -> None:
        """Inherited, see superclass."""
        if torch.cuda.is_available():
            state_dict: Dict[str, Any] = torch.load(self._checkpoint_path)["state_dict"]
        else:
            state_dict: Dict[str, Any] = torch.load(self._checkpoint_path, map_location=torch.device("cpu"))[
                "state_dict"
            ]
        state_dict = {k.replace("agent.", ""): v for k, v in state_dict.items()}
        self._load_state_dict_flexible(state_dict)
        self._maybe_load_bev_semantic_pretrained()
        self._maybe_freeze_bev_semantic()
        self._maybe_freeze_pdm_score_head_only()
        self._maybe_freeze_score_residual_only()

    def _maybe_freeze_score_residual_only(self) -> None:
        if not bool(getattr(self._config, "hardcase_score_residual_train_only", False)):
            return
        if not bool(getattr(self._config, "hardcase_score_residual_enable", False)):
            raise ValueError(
                "hardcase_score_residual_train_only=True requires hardcase_score_residual_enable=True."
            )

        traj_head = getattr(self._transfuser_model, "_trajectory_head", None)
        pdm_head = getattr(traj_head, "_pdm_score_head", None) if traj_head is not None else None
        if pdm_head is None:
            raise ValueError(
                "hardcase_score_residual_train_only=True requires pdm_score_head to be initialized. "
                "Set pdm_score_use_head=True or pdm_score_weight>0."
            )

        residual_module_names = (
            "_score_res_base_proj",
            "_score_res_scene_proj",
            "_score_res_proposal_proj",
            "_score_res_mode_context_proj",
            "_score_residual_head",
            "_score_residual_lora_adapters",
        )
        residual_params: List[torch.nn.Parameter] = []
        for name in residual_module_names:
            module = getattr(pdm_head, name, None)
            if isinstance(module, nn.Module):
                residual_params.extend(list(module.parameters()))
        if len(residual_params) == 0:
            raise ValueError(
                "hardcase_score_residual_train_only=True but no residual-score parameters were found. "
                "Check hardcase_score_residual_enable and model initialization."
            )

        for param in self._transfuser_model.parameters():
            param.requires_grad = False
        for param in residual_params:
            param.requires_grad = True

        trainable_count = int(sum(p.numel() for p in residual_params))
        logger.info(
            "Enabled hardcase_score_residual_train_only: frozen generalist, "
            "trainable residual-score params=%d",
            trainable_count,
        )
        self._enforce_frozen_module_modes()


    def get_sensor_config(self) -> SensorConfig:
        """Inherited, see superclass."""
        return SensorConfig.build_all_sensors(include=[3])

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        """Inherited, see superclass."""
        return [TransfuserTargetBuilder(config=self._config)]

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        """Inherited, see superclass."""
        return [TransfuserFeatureBuilder(config=self._config)]

    def forward(self, features: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]=None) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""
        if targets is not None:
            self._maybe_add_feasible_targets(targets)
        return self._transfuser_model(features,targets=targets)

    def _maybe_add_feasible_targets(self, targets: Dict[str, torch.Tensor]) -> None:
        if not self._config.extract_feasible_lane:
            return
        if (
            "feasible_area_mask" in targets
            and "feasible_lane_mask" in targets
        ):
            return
        if "bev_semantic_map" not in targets:
            return
        if self._feasible_target_builder is None:
            self._feasible_target_builder = TransfuserTargetBuilder(config=self._config)
        with torch.no_grad():
            feasible_area_mask, feasible_lane_mask = (
                self._feasible_target_builder._extract_feasible_lane_from_map(
                    targets["bev_semantic_map"]
                )
            )
        targets.setdefault("feasible_area_mask", feasible_area_mask)
        targets.setdefault("feasible_lane_mask", feasible_lane_mask)

    def _get_pdm_supervision(
        self, cache_path_override: Optional[str] = None
    ) -> Optional[PDMSupervision]:
        grpo_active = bool(getattr(self._config, "grpo_enable", False)) and (
            float(getattr(self._config, "grpo_weight", 0.0)) > 0.0
        )
        if (
            self._config.pdm_score_weight <= 0
            and cache_path_override is None
            and not grpo_active
        ):
            return None
        cache_path = cache_path_override
        if not cache_path:
            cache_path = self._config.pdm_metric_cache_path
        if cache_path is None:
            navsim_root = os.getenv("NAVSIM_EXP_ROOT")
            if navsim_root:
                cache_path = os.path.join(navsim_root, "metric_cache")
        if not cache_path:
            return None
        cache_key = str(cache_path)
        cached = self._pdm_supervision_by_path.get(cache_key)
        if cached is not None:
            return cached
        supervisor = PDMSupervision(
            PDMScoreConfig(
                cache_path=cache_path,
                use_ray=getattr(self._config, "pdm_score_use_ray", False),
                ray_threads=getattr(self._config, "pdm_score_ray_threads", 0),
                ray_debug=getattr(self._config, "pdm_score_ray_debug", False),
                cache_lru_size=getattr(self._config, "pdm_score_cache_lru_size", 0),
                progress_use_reference_baseline=getattr(
                    self._config,
                    "pdm_score_progress_use_reference_baseline",
                    False,
                ),
            )
        )
        self._pdm_supervision_by_path[cache_key] = supervisor
        return supervisor

    def _get_pdm_component_targets(
        self,
        targets: Dict[str, Any],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        components = targets.get("pdm_score_components")
        if components is not None:
            if not torch.is_tensor(components):
                components = torch.tensor(components)
            if components.ndim == 2:
                components = components.unsqueeze(0)
            return components.to(device=device, dtype=dtype)
        component_dict = targets.get("pdm_components")
        if isinstance(component_dict, dict):
            comp_list = []
            for key in _PDM_COMPONENT_KEYS:
                value = component_dict.get(key)
                if value is None:
                    return None
                if not torch.is_tensor(value):
                    value = torch.tensor(value)
                comp_list.append(value)
            components = torch.stack(comp_list, dim=-1)
            if components.ndim == 2:
                components = components.unsqueeze(0)
            return components.to(device=device, dtype=dtype)
        return None

    def _should_output_pdm_infraction_details(self) -> bool:
        if not bool(
            getattr(self._config, "pdm_score_output_infraction_details", False)
        ):
            return False
        if self.training:
            return bool(
                getattr(
                    self._config,
                    "pdm_score_output_infraction_in_train",
                    True,
                )
            )
        return bool(
            getattr(
                self._config,
                "pdm_score_output_infraction_in_val",
                True,
            )
        )

    def _attach_pdm_infraction_details_to_predictions(
        self,
        predictions: Dict[str, torch.Tensor],
        infraction_data: Optional[Dict[str, Any]],
        mode_indices: Optional[torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        if not isinstance(infraction_data, dict):
            return
        key_map = {
            "collision_position_xy": "pdm_collision_position_xy",
            "collision_actor_position_xy": "pdm_collision_actor_position_xy",
            "collision_agent_position_xy": "pdm_collision_agent_position_xy",
            "collision_time_s": "pdm_collision_time_s",
            "collision_valid": "pdm_collision_valid",
            "collision_is_agent": "pdm_collision_is_agent",
            "offroad_position_xy": "pdm_offroad_position_xy",
            "offroad_time_s": "pdm_offroad_time_s",
            "offroad_valid": "pdm_offroad_valid",
        }
        for src_key, dst_key in key_map.items():
            value = infraction_data.get(src_key)
            if value is None:
                continue
            tensor = torch.as_tensor(value, device=device)
            if tensor.dtype.is_floating_point:
                tensor = tensor.to(dtype=dtype)
            predictions[dst_key] = tensor
        if mode_indices is not None:
            predictions["pdm_infraction_mode_indices"] = mode_indices.to(device=device)

    def _compute_mode_bev_attention_aux_loss(
        self, predictions: Dict[str, torch.Tensor]
    ) -> Optional[torch.Tensor]:
        if not bool(getattr(self._config, "mode_bev_attention_aux_enable", False)):
            return None
        if not self._should_output_pdm_infraction_details():
            return None
        weight = float(getattr(self._config, "mode_bev_attention_aux_weight", 0.0))
        if weight <= 0.0:
            return None

        attn_map = predictions.get("mode_bev_attention_map")
        if not torch.is_tensor(attn_map) or attn_map.dim() != 4:
            return None
        bs, num_modes, height, width = attn_map.shape
        if bs <= 0 or num_modes <= 0 or height <= 0 or width <= 0:
            return None

        mode_indices = predictions.get("pdm_infraction_mode_indices")
        if torch.is_tensor(mode_indices):
            if mode_indices.dim() == 1:
                mode_indices = mode_indices.unsqueeze(0)
            if mode_indices.dim() != 2 or mode_indices.shape[0] != bs:
                return None
            mode_indices = mode_indices.to(device=attn_map.device, dtype=torch.long)
        else:
            mode_indices = torch.arange(
                num_modes, device=attn_map.device, dtype=torch.long
            )[None, :].expand(bs, -1)

        pixel_size = float(getattr(self._config, "bev_pixel_size", 0.25))
        if pixel_size <= 0:
            pixel_size = 0.25
        sigma_px = max(
            float(getattr(self._config, "mode_bev_attention_aux_sigma_px", 2.0)),
            1e-3,
        )
        time_decay_enable = bool(
            getattr(self._config, "mode_bev_attention_aux_time_decay_enable", False)
        )
        time_decay_tau_s = max(
            float(getattr(self._config, "mode_bev_attention_aux_time_decay_tau_s", 2.0)),
            1e-3,
        )
        time_decay_min = float(
            getattr(self._config, "mode_bev_attention_aux_time_decay_min", 0.0)
        )
        time_decay_min = max(0.0, min(1.0, time_decay_min))

        row_grid = torch.arange(
            height, device=attn_map.device, dtype=attn_map.dtype
        )[:, None]
        col_grid = torch.arange(
            width, device=attn_map.device, dtype=attn_map.dtype
        )[None, :]
        ego_col = (width - 1) * 0.5

        target = torch.zeros_like(attn_map)
        valid_mode = torch.zeros(
            (bs, num_modes), device=attn_map.device, dtype=torch.bool
        )

        def _add_events(
            pos_key: str,
            valid_key: str,
            time_key: Optional[str] = None,
            gate_key: Optional[str] = None,
        ) -> None:
            positions = predictions.get(pos_key)
            valid = predictions.get(valid_key)
            if not torch.is_tensor(positions) or not torch.is_tensor(valid):
                return
            if positions.dim() != 3 or positions.shape[-1] != 2:
                return
            if valid.dim() != 2:
                return
            if positions.shape[0] != bs or valid.shape[0] != bs:
                return
            if positions.shape[1] != mode_indices.shape[1]:
                return

            positions = positions.to(device=attn_map.device, dtype=attn_map.dtype)
            valid = valid.to(device=attn_map.device, dtype=torch.bool)
            event_times: Optional[torch.Tensor] = None
            if time_key is not None:
                time_values = predictions.get(time_key)
                if (
                    torch.is_tensor(time_values)
                    and time_values.dim() == 2
                    and time_values.shape[0] == bs
                    and time_values.shape[1] == positions.shape[1]
                ):
                    event_times = time_values.to(
                        device=attn_map.device,
                        dtype=attn_map.dtype,
                    )
            gate_values: Optional[torch.Tensor] = None
            if gate_key is not None:
                gate = predictions.get(gate_key)
                if (
                    not torch.is_tensor(gate)
                    or gate.dim() != 2
                    or gate.shape[0] != bs
                    or gate.shape[1] != positions.shape[1]
                ):
                    return
                gate_values = gate.to(device=attn_map.device, dtype=torch.bool)

            for batch_idx in range(bs):
                for local_mode_idx in range(positions.shape[1]):
                    is_valid = bool(valid[batch_idx, local_mode_idx])
                    if gate_values is not None:
                        is_valid = is_valid and bool(
                            gate_values[batch_idx, local_mode_idx]
                        )
                    if not is_valid:
                        continue
                    global_mode_idx = int(mode_indices[batch_idx, local_mode_idx].item())
                    if global_mode_idx < 0 or global_mode_idx >= num_modes:
                        continue
                    xy = positions[batch_idx, local_mode_idx]
                    if not torch.isfinite(xy).all():
                        continue
                    row = xy[0] / pixel_size
                    col = xy[1] / pixel_size + ego_col
                    row_val = float(row.item())
                    col_val = float(col.item())
                    if (
                        row_val < 0.0
                        or row_val > float(height - 1)
                        or col_val < 0.0
                        or col_val > float(width - 1)
                    ):
                        continue
                    event_weight = 1.0
                    event_sigma = sigma_px
                    if time_decay_enable and event_times is not None:
                        event_time = event_times[batch_idx, local_mode_idx]
                        if torch.isfinite(event_time):
                            event_time = torch.clamp(event_time, min=0.0)
                            event_weight = float(
                                torch.exp(-event_time / time_decay_tau_s).item()
                            )
                            event_weight = max(time_decay_min, event_weight)
                            event_sigma = sigma_px / max(event_weight, 1e-3)
                    d2 = (row_grid - row).pow(2) + (col_grid - col).pow(2)
                    blob = torch.exp(-0.5 * d2 / (event_sigma * event_sigma))
                    if event_weight != 1.0:
                        blob = blob * event_weight
                    target[batch_idx, global_mode_idx] = torch.maximum(
                        target[batch_idx, global_mode_idx], blob
                    )
                    valid_mode[batch_idx, global_mode_idx] = True

        _add_events(
            "pdm_collision_position_xy",
            "pdm_collision_valid",
            time_key="pdm_collision_time_s",
        )
        _add_events(
            "pdm_collision_agent_position_xy",
            "pdm_collision_valid",
            time_key="pdm_collision_time_s",
            gate_key="pdm_collision_is_agent",
        )
        _add_events(
            "pdm_offroad_position_xy",
            "pdm_offroad_valid",
            time_key="pdm_offroad_time_s",
        )

        target_flat = target.view(bs, num_modes, -1)
        target_sum = target_flat.sum(dim=-1, keepdim=True)
        valid_mode = valid_mode & (target_sum.squeeze(-1) > 0)
        if not valid_mode.any():
            return None
        target_prob = target_flat / target_sum.clamp_min(1e-8)

        pred_flat = attn_map.view(bs, num_modes, -1).clamp_min(1e-8)
        pred_prob = pred_flat / pred_flat.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        kl = target_prob * (
            torch.log(target_prob.clamp_min(1e-8)) - torch.log(pred_prob)
        )
        per_mode_loss = kl.sum(dim=-1)
        return per_mode_loss[valid_mode].mean()

    def _compute_cached_pose_align_loss(
        self,
        targets: Dict[str, Any],
        predictions: Dict[str, torch.Tensor],
    ) -> Optional[torch.Tensor]:
        weight = float(getattr(self._config, "pdm_score_cached_pose_align_weight", 0.0))
        if weight <= 0.0:
            return None
        pred = predictions.get("poses_reg")
        cached = targets.get("poses_reg")
        if pred is None or cached is None:
            return None
        if cached.dim() == 3:
            cached = cached.unsqueeze(0)
        cached = cached.to(device=pred.device, dtype=pred.dtype)
        min_modes = min(pred.shape[1], cached.shape[1])
        if min_modes <= 0:
            return None
        pred = pred[:, :min_modes]
        cached = cached[:, :min_modes]
        min_steps = min(pred.shape[2], cached.shape[2])
        if min_steps <= 0:
            return None
        pred = pred[:, :, :min_steps, :2]
        cached = cached[:, :, :min_steps, :2]
        return (pred - cached).pow(2).mean()

    def _bev_to_class_map(
        self, bev_semantic_map: Optional[torch.Tensor]
    ) -> Optional[torch.Tensor]:
        if not isinstance(bev_semantic_map, torch.Tensor):
            return None
        if bev_semantic_map.dim() == 4:
            return bev_semantic_map.argmax(dim=1).long()
        if bev_semantic_map.dim() == 3:
            if bev_semantic_map.shape[0] == self._config.num_bev_classes:
                return bev_semantic_map.argmax(dim=0).long().unsqueeze(0)
            if bev_semantic_map.shape[-1] == self._config.num_bev_classes:
                return bev_semantic_map.argmax(dim=-1).long().unsqueeze(0)
            return bev_semantic_map.long()
        if bev_semantic_map.dim() == 2:
            return bev_semantic_map.long().unsqueeze(0)
        return None

    def _trajectory_xy_to_grid(self, traj_xy: torch.Tensor) -> torch.Tensor:
        norm = traj_xy.clone()
        denom_y = max(abs(float(self._config.lidar_max_y)), 1e-6)
        denom_x = max(abs(float(self._config.lidar_max_x)), 1e-6)
        norm[..., 0] = norm[..., 0] / denom_y
        norm[..., 1] = norm[..., 1] / denom_x
        norm = norm[..., [1, 0]]
        return norm

    def _sample_bev_map_along_trajectory(
        self, bev_map: torch.Tensor, traj_xy: torch.Tensor
    ) -> Optional[torch.Tensor]:
        if not torch.is_tensor(bev_map) or not torch.is_tensor(traj_xy):
            return None
        if bev_map.dim() != 4 or traj_xy.dim() != 4 or traj_xy.shape[-1] < 2:
            return None
        bs, num_modes, num_steps, _ = traj_xy.shape
        if bev_map.shape[0] != bs:
            if bev_map.shape[0] == 1:
                bev_map = bev_map.expand(bs, -1, -1, -1)
            else:
                return None
        grid = self._trajectory_xy_to_grid(traj_xy[..., :2]).view(
            bs * num_modes, num_steps, 1, 2
        )
        value = bev_map.unsqueeze(1).expand(-1, num_modes, -1, -1, -1)
        value = value.reshape(
            bs * num_modes, bev_map.shape[1], bev_map.shape[2], bev_map.shape[3]
        )
        sampled = F.grid_sample(
            value,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        return sampled.view(bs, num_modes, bev_map.shape[1], num_steps)

    def _build_pdm_risk_area_targets(
        self,
        targets: Dict[str, Any],
        poses_reg: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        source_bev = targets.get("bev_semantic_map")
        class_map = self._bev_to_class_map(source_bev)
        if class_map is None:
            return None
        class_map = class_map.to(device=poses_reg.device)

        road_label = int(self._config.bev_road_label)
        centerline_label = int(self._config.bev_centerline_label)

        drivable_mask = (class_map == road_label) | (class_map == centerline_label)
        offroad_map = (~drivable_mask).float()

        area_maps = offroad_map.unsqueeze(1).to(
            device=poses_reg.device,
            dtype=poses_reg.dtype,
        )
        sampled = self._sample_bev_map_along_trajectory(
            area_maps,
            poses_reg.detach()[..., :2].to(dtype=poses_reg.dtype),
        )
        if sampled is None:
            return None
        return sampled.amax(dim=-1).clamp(0.0, 1.0)

    def _compute_pdm_risk_area_aux_loss(
        self,
        targets: Dict[str, Any],
        predictions: Dict[str, torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        logs: Dict[str, torch.Tensor] = {}
        weight = float(getattr(self._config, "pdm_score_risk_area_weight", 0.0))
        if weight <= 0.0 and not bool(
            getattr(self._config, "pdm_score_risk_area_enable", False)
        ):
            return None, logs

        logits = predictions.get("pdm_score_risk_area_logits")
        poses_reg = predictions.get("poses_reg")
        if (
            not torch.is_tensor(logits)
            or logits.dim() != 3
            or logits.shape[-1] != 1
            or not torch.is_tensor(poses_reg)
        ):
            return None, logs

        targets_risk = self._build_pdm_risk_area_targets(targets, poses_reg)
        if targets_risk is None or targets_risk.shape != logits.shape:
            return None, logs

        logits = logits.float()
        targets_risk = targets_risk.to(device=logits.device, dtype=logits.dtype)
        raw = F.binary_cross_entropy_with_logits(
            logits,
            targets_risk,
            reduction="none",
        )
        loss = raw.mean()

        pred_prob = torch.sigmoid(logits)
        logs["pdm_score_risk_area_offroad_loss_raw"] = loss.detach()
        logs["pdm_score_risk_area_offroad_target_mean"] = (
            targets_risk[..., 0].mean().detach()
        )
        logs["pdm_score_risk_area_offroad_pred_mean"] = (
            pred_prob[..., 0].mean().detach()
        )
        return weight * loss, logs

    def _compute_hardcase_gate_aux_loss(
        self,
        predictions: Dict[str, torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Optional regularization for the scene+proposal hardcase gate branch.
        Returns weighted total loss and logging terms.
        """
        logs: Dict[str, torch.Tensor] = {}
        if not bool(getattr(self._config, "hardcase_gate_enable", False)):
            return None, logs

        gate_prob = predictions.get("hardcase_gate_prob")
        if not torch.is_tensor(gate_prob):
            return None, logs
        gate_prob = torch.nan_to_num(
            gate_prob.float(), nan=0.0, posinf=1.0, neginf=0.0
        ).clamp(0.0, 1.0)

        logs["hardcase_gate_rate"] = gate_prob.mean()
        gate_weight = predictions.get("hardcase_gate_weight")
        if torch.is_tensor(gate_weight):
            logs["hardcase_gate_trigger_rate"] = (
                torch.nan_to_num(gate_weight.float(), nan=0.0).clamp(0.0, 1.0).mean()
            )

        total_loss: Optional[torch.Tensor] = None
        prior_weight = float(getattr(self._config, "hardcase_gate_prior_weight", 0.0))
        if prior_weight > 0.0:
            prior = float(getattr(self._config, "hardcase_gate_prior", 0.1))
            prior_loss = (gate_prob.mean() - prior).pow(2)
            weighted_prior = prior_weight * prior_loss
            logs["hardcase_gate_prior_loss"] = weighted_prior
            total_loss = weighted_prior if total_loss is None else total_loss + weighted_prior

        residual_reg_weight = float(
            getattr(self._config, "hardcase_gate_residual_reg_weight", 0.0)
        )
        residual_l1 = predictions.get("hardcase_gate_residual_l1")
        if residual_reg_weight > 0.0 and torch.is_tensor(residual_l1):
            residual_l1 = torch.nan_to_num(
                residual_l1.float(), nan=0.0, posinf=0.0, neginf=0.0
            )
            while residual_l1.dim() > gate_prob.dim():
                residual_l1 = residual_l1.mean(dim=-1)
            if residual_l1.shape != gate_prob.shape and residual_l1.numel() == gate_prob.numel():
                residual_l1 = residual_l1.reshape_as(gate_prob)
            residual_reg = ((1.0 - gate_prob.detach()) * residual_l1).mean()
            weighted_residual_reg = residual_reg_weight * residual_reg
            logs["hardcase_gate_residual_reg_loss"] = weighted_residual_reg
            total_loss = (
                weighted_residual_reg
                if total_loss is None
                else total_loss + weighted_residual_reg
            )

        return total_loss, logs

    @staticmethod
    def _prepare_bev_target_labels_for_gate(
        pred_logits: torch.Tensor,
        target_labels: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if not torch.is_tensor(pred_logits) or not torch.is_tensor(target_labels):
            return None
        labels = target_labels
        if labels.dim() == 4 and labels.shape[1] == pred_logits.shape[1]:
            labels = labels.argmax(dim=1)
        if labels.dim() == 3 and labels.shape[0] == pred_logits.shape[1]:
            labels = labels.argmax(dim=0, keepdim=True)
        if labels.dim() == 2:
            labels = labels.unsqueeze(0)
        if labels.dim() != 3:
            return None
        if labels.shape[0] != pred_logits.shape[0]:
            if labels.shape[0] == 1:
                labels = labels.repeat(pred_logits.shape[0], 1, 1)
            else:
                return None
        return labels.to(pred_logits.device).long()

    @staticmethod
    def _align_mode_tensor_for_gate(
        tensor: Optional[torch.Tensor],
        batch_size: int,
        max_modes: int,
    ) -> Optional[torch.Tensor]:
        if not torch.is_tensor(tensor):
            return None
        value = tensor
        if value.dim() == 3:
            value = value.mean(dim=-1)
        if value.dim() != 2:
            return None
        if value.shape[0] != batch_size:
            if value.shape[0] == 1:
                value = value.expand(batch_size, -1)
            else:
                return None
        if value.shape[1] <= 0:
            return None
        return value[:, : min(max_modes, value.shape[1])]

    def _select_mode_indices_for_gate(
        self,
        predictions: Dict[str, torch.Tensor],
        batch_size: int,
        num_modes: int,
        device: torch.device,
    ) -> torch.Tensor:
        poses_cls = self._align_mode_tensor_for_gate(
            predictions.get("poses_cls"),
            batch_size=batch_size,
            max_modes=num_modes,
        )
        score_logits = self._align_mode_tensor_for_gate(
            predictions.get("pdm_score"),
            batch_size=batch_size,
            max_modes=num_modes,
        )

        mode_idx: Optional[torch.Tensor] = None
        use_pdm_select = bool(getattr(self._config, "pdm_score_use_for_selection", False))
        if use_pdm_select and score_logits is not None:
            topk_select = int(getattr(self._config, "pdm_score_select_topk", 0) or 0)
            if poses_cls is not None and 0 < topk_select < poses_cls.shape[1]:
                topk_idx = torch.topk(poses_cls, topk_select, dim=-1).indices
                topk_scores = torch.gather(score_logits, 1, topk_idx)
                best_in_topk = topk_scores.argmax(dim=-1, keepdim=True)
                mode_idx = topk_idx.gather(1, best_in_topk).squeeze(1)
            else:
                mode_idx = score_logits.argmax(dim=-1)

        if mode_idx is None:
            if poses_cls is not None:
                mode_idx = poses_cls.argmax(dim=-1)
            elif score_logits is not None:
                mode_idx = score_logits.argmax(dim=-1)
            else:
                mode_idx = torch.zeros(batch_size, device=device, dtype=torch.long)
        return mode_idx.long()

    def _compute_hardcase_gate_supervision_loss(
        self,
        targets: Dict[str, Any],
        predictions: Dict[str, torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        logs: Dict[str, torch.Tensor] = {}
        if not bool(getattr(self._config, "hardcase_gate_enable", False)):
            return None, logs
        if not bool(getattr(self._config, "hardcase_gate_supervision_enable", False)):
            return None, logs
        sup_weight = float(getattr(self._config, "hardcase_gate_supervision_weight", 0.0))
        if sup_weight <= 0.0:
            return None, logs

        gate_logit = predictions.get("hardcase_gate_logit")
        if not (torch.is_tensor(gate_logit) and gate_logit.dim() == 2):
            return None, logs
        gate_logit = torch.nan_to_num(gate_logit.float(), nan=0.0, posinf=0.0, neginf=0.0)

        bs, num_modes = gate_logit.shape
        batch_idx = torch.arange(bs, device=gate_logit.device)
        selected_idx = self._select_mode_indices_for_gate(
            predictions=predictions,
            batch_size=bs,
            num_modes=num_modes,
            device=gate_logit.device,
        ).clamp(0, max(num_modes - 1, 0))
        selected_gate_logit = gate_logit[batch_idx, selected_idx]

        score_risk: Optional[torch.Tensor] = None
        selected_score: Optional[torch.Tensor] = None
        score_logits = self._align_mode_tensor_for_gate(
            predictions.get("pdm_score"),
            batch_size=bs,
            max_modes=num_modes,
        )
        if score_logits is not None:
            score_logits = torch.nan_to_num(
                score_logits.float(), nan=0.0, posinf=0.0, neginf=0.0
            )
            selected_score = torch.sigmoid(score_logits[batch_idx, selected_idx]).clamp(0.0, 1.0)
        else:
            poses_cls = self._align_mode_tensor_for_gate(
                predictions.get("poses_cls"),
                batch_size=bs,
                max_modes=num_modes,
            )
            if poses_cls is not None:
                selected_score = torch.softmax(poses_cls.float(), dim=-1)[
                    batch_idx, selected_idx
                ].clamp(0.0, 1.0)
        if selected_score is not None:
            score_thresh = float(
                getattr(self._config, "hardcase_gate_supervision_score_thresh", 0.55)
            )
            score_temp = max(
                float(getattr(self._config, "hardcase_gate_supervision_score_temp", 0.10)),
                1e-6,
            )
            score_risk = torch.sigmoid((score_thresh - selected_score.detach()) / score_temp)
            logs["hardcase_gate_supervision_selected_score"] = selected_score.detach().mean()
            logs["hardcase_gate_supervision_score_risk"] = score_risk.detach().mean()

        bev_risk: Optional[torch.Tensor] = None
        pred_bev = predictions.get("bev_semantic_map")
        target_bev = targets.get("bev_semantic_map")
        if torch.is_tensor(pred_bev) and torch.is_tensor(target_bev) and pred_bev.dim() == 4:
            bev_labels = self._prepare_bev_target_labels_for_gate(
                pred_logits=pred_bev,
                target_labels=target_bev,
            )
            if bev_labels is not None:
                if bool(getattr(self._config, "bev_loss_logit_guard_enable", False)):
                    safe_pred_bev = torch.nan_to_num(
                        pred_bev.detach().float(), nan=0.0, posinf=30.0, neginf=-30.0
                    ).clamp(-30.0, 30.0)
                else:
                    safe_pred_bev = pred_bev.detach()
                bev_ce = F.cross_entropy(
                    safe_pred_bev,
                    bev_labels,
                    reduction="none",
                )
                bev_per_sample = bev_ce.view(bev_ce.shape[0], -1).mean(dim=-1)
                bev_mean = bev_per_sample.mean()
                bev_std = bev_per_sample.std(unbiased=False).clamp_min(1e-6)
                bev_z = (bev_per_sample - bev_mean) / bev_std
                bev_z_thresh = float(
                    getattr(self._config, "hardcase_gate_supervision_bev_z_thresh", 0.5)
                )
                bev_temp = max(
                    float(getattr(self._config, "hardcase_gate_supervision_bev_temp", 0.5)),
                    1e-6,
                )
                bev_risk = torch.sigmoid((bev_z - bev_z_thresh) / bev_temp)
                logs["hardcase_gate_supervision_bev_ce"] = bev_per_sample.detach().mean()
                logs["hardcase_gate_supervision_bev_risk"] = bev_risk.detach().mean()

        target_risk: Optional[torch.Tensor] = None
        if score_risk is not None and bev_risk is not None:
            combine_mode = str(
                getattr(self._config, "hardcase_gate_supervision_combine", "max")
            ).lower()
            if combine_mode == "mean":
                target_risk = 0.5 * (score_risk + bev_risk)
            else:
                target_risk = torch.maximum(score_risk, bev_risk)
        elif score_risk is not None:
            target_risk = score_risk
        elif bev_risk is not None:
            target_risk = bev_risk

        if target_risk is None:
            return None, logs
        target_risk = target_risk.clamp(0.0, 1.0)

        sup_loss_raw = F.binary_cross_entropy_with_logits(
            selected_gate_logit,
            target_risk.detach(),
        )
        sup_loss = sup_weight * sup_loss_raw
        logs["hardcase_gate_supervision_target_rate"] = target_risk.detach().mean()
        logs["hardcase_gate_supervision_loss_raw"] = sup_loss_raw.detach()
        return sup_loss, logs

    def _compute_hardcase_specialist_loss(
        self,
        targets: Dict[str, Any],
        predictions: Dict[str, torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        logs: Dict[str, torch.Tensor] = {}
        if not bool(getattr(self._config, "hardcase_specialist_enable", False)):
            return None, logs
        loss_weight = float(getattr(self._config, "hardcase_specialist_loss_weight", 0.0))
        if loss_weight <= 0.0:
            return None, logs

        traj_spec = predictions.get("trajectory_specialist")
        traj_gen = predictions.get("trajectory_generalist")
        if not torch.is_tensor(traj_gen):
            traj_gen = predictions.get("trajectory")
        target_traj = targets.get("trajectory")
        if not (torch.is_tensor(traj_spec) and torch.is_tensor(traj_gen) and torch.is_tensor(target_traj)):
            return None, logs

        if traj_spec.dim() == 4 and traj_spec.shape[1] == 1:
            traj_spec = traj_spec[:, 0]
        if traj_gen.dim() == 4 and traj_gen.shape[1] == 1:
            traj_gen = traj_gen[:, 0]
        if target_traj.dim() == 2:
            target_traj = target_traj.unsqueeze(0)
        if traj_spec.dim() != 3 or traj_gen.dim() != 3 or target_traj.dim() != 3:
            return None, logs
        if traj_spec.shape[0] != target_traj.shape[0] or traj_gen.shape[0] != target_traj.shape[0]:
            return None, logs

        bs = traj_spec.shape[0]
        num_modes = 1
        poses_reg = predictions.get("poses_reg")
        if torch.is_tensor(poses_reg) and poses_reg.dim() == 4 and poses_reg.shape[0] == bs:
            num_modes = max(int(poses_reg.shape[1]), 1)
        elif torch.is_tensor(predictions.get("pdm_score")) and predictions["pdm_score"].dim() == 2:
            num_modes = max(int(predictions["pdm_score"].shape[1]), 1)
        elif torch.is_tensor(predictions.get("poses_cls")) and predictions["poses_cls"].dim() == 2:
            num_modes = max(int(predictions["poses_cls"].shape[1]), 1)

        selected_idx = self._select_mode_indices_for_gate(
            predictions=predictions,
            batch_size=bs,
            num_modes=num_modes,
            device=traj_spec.device,
        ).clamp(0, max(num_modes - 1, 0))
        logs["hardcase_specialist_selected_mode_mean"] = selected_idx.float().mean()

        score_risk: Optional[torch.Tensor] = None
        bev_risk: Optional[torch.Tensor] = None
        use_score_risk = bool(
            getattr(self._config, "hardcase_specialist_use_score_risk", True)
        )
        use_bev_risk = bool(
            getattr(self._config, "hardcase_specialist_use_bev_risk", True)
        )

        if use_score_risk:
            batch_idx = torch.arange(bs, device=traj_spec.device)
            score_logits = self._align_mode_tensor_for_gate(
                predictions.get("pdm_score"),
                batch_size=bs,
                max_modes=num_modes,
            )
            selected_score: Optional[torch.Tensor] = None
            if score_logits is not None:
                score_logits = torch.nan_to_num(
                    score_logits.float(), nan=0.0, posinf=0.0, neginf=0.0
                )
                selected_score = torch.sigmoid(score_logits[batch_idx, selected_idx]).clamp(
                    0.0, 1.0
                )
            else:
                poses_cls = self._align_mode_tensor_for_gate(
                    predictions.get("poses_cls"),
                    batch_size=bs,
                    max_modes=num_modes,
                )
                if poses_cls is not None:
                    selected_score = torch.softmax(poses_cls.float(), dim=-1)[
                        batch_idx, selected_idx
                    ].clamp(0.0, 1.0)
            if selected_score is not None:
                score_thresh = float(
                    getattr(self._config, "hardcase_gate_supervision_score_thresh", 0.55)
                )
                score_temp = max(
                    float(
                        getattr(
                            self._config, "hardcase_gate_supervision_score_temp", 0.10
                        )
                    ),
                    1e-6,
                )
                score_risk = torch.sigmoid(
                    (score_thresh - selected_score.detach()) / score_temp
                )
                logs["hardcase_specialist_selected_score"] = selected_score.detach().mean()
                logs["hardcase_specialist_score_risk"] = score_risk.detach().mean()

        if use_bev_risk:
            pred_bev = predictions.get("bev_semantic_map")
            target_bev = targets.get("bev_semantic_map")
            if torch.is_tensor(pred_bev) and torch.is_tensor(target_bev) and pred_bev.dim() == 4:
                bev_labels = self._prepare_bev_target_labels_for_gate(
                    pred_logits=pred_bev,
                    target_labels=target_bev,
                )
                if bev_labels is not None:
                    if bool(getattr(self._config, "bev_loss_logit_guard_enable", False)):
                        safe_pred_bev = torch.nan_to_num(
                            pred_bev.detach().float(), nan=0.0, posinf=30.0, neginf=-30.0
                        ).clamp(-30.0, 30.0)
                    else:
                        safe_pred_bev = pred_bev.detach()
                    bev_ce = F.cross_entropy(
                        safe_pred_bev,
                        bev_labels,
                        reduction="none",
                    )
                    bev_per_sample = bev_ce.view(bev_ce.shape[0], -1).mean(dim=-1)
                    bev_mean = bev_per_sample.mean()
                    bev_std = bev_per_sample.std(unbiased=False).clamp_min(1e-6)
                    bev_z = (bev_per_sample - bev_mean) / bev_std
                    bev_z_thresh = float(
                        getattr(self._config, "hardcase_gate_supervision_bev_z_thresh", 0.5)
                    )
                    bev_temp = max(
                        float(
                            getattr(self._config, "hardcase_gate_supervision_bev_temp", 0.5)
                        ),
                        1e-6,
                    )
                    bev_risk = torch.sigmoid((bev_z - bev_z_thresh) / bev_temp)
                    logs["hardcase_specialist_bev_ce"] = bev_per_sample.detach().mean()
                    logs["hardcase_specialist_bev_risk"] = bev_risk.detach().mean()

        target_risk: Optional[torch.Tensor] = None
        if score_risk is not None and bev_risk is not None:
            combine_mode = str(
                getattr(self._config, "hardcase_gate_supervision_combine", "max")
            ).lower()
            if combine_mode == "mean":
                target_risk = 0.5 * (score_risk + bev_risk)
            else:
                target_risk = torch.maximum(score_risk, bev_risk)
        elif score_risk is not None:
            target_risk = score_risk
        elif bev_risk is not None:
            target_risk = bev_risk

        if target_risk is None:
            return None, logs
        target_risk = target_risk.clamp(0.0, 1.0)
        hard_thresh = float(
            getattr(self._config, "hardcase_specialist_hard_target_thresh", 0.6)
        )
        hard_mask = target_risk >= hard_thresh
        logs["hardcase_specialist_target_rate"] = target_risk.detach().mean()
        logs["hardcase_specialist_hard_rate"] = hard_mask.float().mean()

        overlap_steps = min(traj_spec.shape[1], traj_gen.shape[1], target_traj.shape[1])
        if overlap_steps <= 0:
            return None, logs
        spec_xy = traj_spec[:, :overlap_steps, :2]
        gen_xy = traj_gen[:, :overlap_steps, :2]
        gt_xy = target_traj[:, :overlap_steps, :2].to(
            device=traj_spec.device,
            dtype=traj_spec.dtype,
        )
        valid = torch.isfinite(gt_xy).all(dim=(-1, -2))
        if not valid.any():
            return None, logs

        spec_err = torch.linalg.norm(spec_xy - gt_xy, dim=-1).mean(dim=-1)
        gen_err = torch.linalg.norm(gen_xy - gt_xy, dim=-1).mean(dim=-1)
        improvement = gen_err - spec_err
        logs["hardcase_specialist_error_generalist"] = gen_err[valid].mean().detach()
        logs["hardcase_specialist_error_specialist"] = spec_err[valid].mean().detach()
        logs["hardcase_specialist_improvement"] = improvement[valid].mean().detach()

        total_raw: Optional[torch.Tensor] = None
        hard_valid = valid & hard_mask
        if hard_valid.any():
            hard_loss = F.smooth_l1_loss(spec_xy[hard_valid], gt_xy[hard_valid])
            logs["hardcase_specialist_hard_loss_raw"] = hard_loss.detach()
            total_raw = hard_loss

        nonhard_weight = float(
            getattr(self._config, "hardcase_specialist_nonhard_consistency_weight", 0.05)
        )
        nonhard_valid = valid & (~hard_mask)
        if nonhard_weight > 0.0 and nonhard_valid.any():
            consistency_raw = F.smooth_l1_loss(
                spec_xy[nonhard_valid], gen_xy[nonhard_valid].detach()
            )
            weighted_consistency = nonhard_weight * consistency_raw
            logs["hardcase_specialist_consistency_raw"] = consistency_raw.detach()
            total_raw = (
                weighted_consistency
                if total_raw is None
                else total_raw + weighted_consistency
            )

        delta_reg_weight = float(
            getattr(self._config, "hardcase_specialist_delta_reg_weight", 0.0)
        )
        delta = predictions.get("hardcase_specialist_delta")
        if delta_reg_weight > 0.0 and torch.is_tensor(delta):
            delta_l1 = torch.nan_to_num(
                delta.float(), nan=0.0, posinf=0.0, neginf=0.0
            ).abs().mean()
            weighted_delta_reg = delta_reg_weight * delta_l1
            logs["hardcase_specialist_delta_reg_raw"] = delta_l1.detach()
            total_raw = (
                weighted_delta_reg
                if total_raw is None
                else total_raw + weighted_delta_reg
            )

        if total_raw is None:
            return None, logs
        logs["hardcase_specialist_loss_raw"] = total_raw.detach()
        return loss_weight * total_raw, logs

    def _match_cached_component_targets(
        self,
        poses_reg: torch.Tensor,
        cached_poses: torch.Tensor,
        comp_targets: torch.Tensor,
        targets: Dict[str, Any],
    ) -> torch.Tensor:
        if cached_poses.dim() == 3:
            cached_poses = cached_poses.unsqueeze(0)
        if comp_targets.dim() == 2:
            comp_targets = comp_targets.unsqueeze(0)
        if cached_poses.shape[0] != poses_reg.shape[0]:
            if cached_poses.shape[0] == 1:
                cached_poses = cached_poses.expand(
                    poses_reg.shape[0], -1, -1, -1
                )
            else:
                return comp_targets
        if comp_targets.shape[0] != poses_reg.shape[0]:
            if comp_targets.shape[0] == 1:
                comp_targets = comp_targets.expand(
                    poses_reg.shape[0], -1, -1
                )
            else:
                return comp_targets
        if cached_poses.shape[1] != comp_targets.shape[1]:
            min_modes = min(cached_poses.shape[1], comp_targets.shape[1])
            if min_modes <= 0:
                return comp_targets
            cached_poses = cached_poses[:, :min_modes]
            comp_targets = comp_targets[:, :min_modes]
        cached_poses = cached_poses.to(device=poses_reg.device, dtype=poses_reg.dtype)
        comp_targets = comp_targets.to(device=poses_reg.device, dtype=poses_reg.dtype)

        cache_valid = torch.isfinite(comp_targets).all(dim=-1)
        cache_valid_flag = targets.get("pdm_score_valid")
        if cache_valid_flag is not None:
            if not torch.is_tensor(cache_valid_flag):
                cache_valid_flag = torch.tensor(cache_valid_flag)
            if cache_valid_flag.dim() == 1:
                cache_valid_flag = cache_valid_flag.unsqueeze(0)
            if cache_valid_flag.shape[0] != cache_valid.shape[0]:
                if cache_valid_flag.shape[0] == 1:
                    cache_valid_flag = cache_valid_flag.expand(
                        cache_valid.shape[0], -1
                    )
                else:
                    cache_valid_flag = None
            if cache_valid_flag is not None:
                cache_valid = cache_valid & cache_valid_flag.to(
                    device=cache_valid.device, dtype=torch.bool
                )

        min_steps = min(poses_reg.shape[2], cached_poses.shape[2])
        if min_steps <= 0:
            return comp_targets
        pred_xy = poses_reg.detach()[:, :, :min_steps, :2]
        cached_xy = cached_poses.detach()[:, :, :min_steps, :2]
        dist = torch.linalg.norm(
            pred_xy[:, :, None, :, :] - cached_xy[:, None, :, :, :], dim=-1
        ).mean(dim=-1)
        max_dist = float(
            getattr(self._config, "pdm_score_offline_assign_max_dist_m", 0.0)
        )
        if max_dist > 0.0:
            dist = dist.masked_fill(dist > max_dist, float("inf"))
        dist = dist.masked_fill(~cache_valid[:, None, :], float("inf"))
        temp = max(
            float(getattr(self._config, "pdm_score_offline_assign_temp", 1.0)),
            1e-6,
        )
        weights = torch.softmax(-dist / temp, dim=-1)
        weights = torch.where(torch.isfinite(weights), weights, torch.zeros_like(weights))
        weight_sum = weights.sum(dim=-1, keepdim=True)
        weights = torch.where(
            weight_sum > 0,
            weights / weight_sum.clamp_min(1e-6),
            weights,
        )
        matched = torch.einsum("bmk,bkc->bmc", weights, comp_targets)
        mode_valid = weight_sum.squeeze(-1) > 0
        matched = matched.masked_fill(~mode_valid[..., None], float("-inf"))
        return matched

    def _compute_pdm_component_bce(
        self,
        score_components: torch.Tensor,
        comp_targets: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if score_components is None or comp_targets is None:
            return None
        if score_components.shape != comp_targets.shape:
            return None
        finite_mask = torch.isfinite(comp_targets)
        if not finite_mask.any():
            return None
        targets = comp_targets.clamp(0.0, 1.0)
        pos_thresh = float(
            getattr(self._config, "pdm_score_component_pos_thresh", 0.5)
        )
        pos_mask = (targets >= pos_thresh) & finite_mask
        neg_mask = (targets < pos_thresh) & finite_mask
        hard_neg_topk = int(
            getattr(self._config, "pdm_score_component_hard_neg_topk", 0) or 0
        )
        if hard_neg_topk > 0:
            neg_scores = score_components.detach().masked_fill(
                ~neg_mask, float("-inf")
            )
            topk = min(hard_neg_topk, neg_scores.shape[1])
            if topk > 0:
                topk_vals, topk_idx = torch.topk(neg_scores, k=topk, dim=1)
                topk_mask = torch.isfinite(topk_vals)
                neg_keep = torch.zeros_like(neg_mask)
                neg_keep.scatter_(1, topk_idx, topk_mask)
            else:
                neg_keep = torch.zeros_like(neg_mask)
            sample_mask = pos_mask | neg_keep
        else:
            sample_mask = pos_mask | neg_mask
        if not sample_mask.any():
            return None
        pos_weight = torch.tensor(
            getattr(
                self._config,
                "pdm_score_component_pos_weight",
                (1.0,) * score_components.shape[-1],
            ),
            device=score_components.device,
            dtype=score_components.dtype,
        )
        neg_weight = torch.tensor(
            getattr(
                self._config,
                "pdm_score_component_neg_weight",
                (1.0,) * score_components.shape[-1],
            ),
            device=score_components.device,
            dtype=score_components.dtype,
        )
        if pos_weight.numel() != score_components.shape[-1]:
            pos_weight = torch.ones(
                score_components.shape[-1],
                device=score_components.device,
                dtype=score_components.dtype,
            )
        if neg_weight.numel() != score_components.shape[-1]:
            neg_weight = torch.ones(
                score_components.shape[-1],
                device=score_components.device,
                dtype=score_components.dtype,
            )
        pos_weight = pos_weight.view(1, 1, -1)
        neg_weight = neg_weight.view(1, 1, -1)
        weights = torch.where(pos_mask, pos_weight, neg_weight)
        weights = weights * sample_mask.float()
        weight_sum = weights.sum().clamp_min(1.0)
        loss = F.binary_cross_entropy_with_logits(
            score_components, targets, weight=weights, reduction="sum"
        )
        return loss / weight_sum

    def _aggregate_component_targets_for_pairwise(
        self,
        comp_targets: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if comp_targets is None or comp_targets.dim() != 3:
            return None, None
        finite = torch.isfinite(comp_targets)
        valid_mode = finite.any(dim=-1)
        if not valid_mode.any():
            return None, None

        targets = comp_targets.clamp(0.0, 1.0).masked_fill(~finite, 0.0)
        weights = torch.tensor(
            getattr(
                self._config,
                "pdm_score_component_weights",
                (1.0,) * comp_targets.shape[-1],
            ),
            device=comp_targets.device,
            dtype=comp_targets.dtype,
        )
        if weights.numel() != comp_targets.shape[-1]:
            weights = torch.ones(
                comp_targets.shape[-1],
                device=comp_targets.device,
                dtype=comp_targets.dtype,
            )
        weights = weights.view(1, 1, -1)
        weighted_mask = finite.float() * weights
        denom = weighted_mask.sum(dim=-1).clamp_min(1e-6)
        scores = (targets * weights).sum(dim=-1) / denom
        scores = scores.masked_fill(~valid_mode, float("nan"))
        return scores, valid_mode

    def _compute_pdm_pairwise_ranking_loss(
        self,
        pred_scores: Optional[torch.Tensor],
        target_scores: Optional[torch.Tensor],
        valid_mode: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        if pred_scores is None or target_scores is None:
            return None
        if pred_scores.dim() != 2 or target_scores.dim() != 2:
            return None
        if pred_scores.shape != target_scores.shape:
            return None

        pairwise_enable = bool(getattr(self._config, "pdm_score_pairwise_enable", False))
        pairwise_weight = float(getattr(self._config, "pdm_score_pairwise_weight", 1.0))
        if not pairwise_enable or pairwise_weight <= 0.0:
            return None

        min_gap = max(
            float(getattr(self._config, "pdm_score_pairwise_min_target_gap", 0.0)),
            0.0,
        )
        margin = float(getattr(self._config, "pdm_score_pairwise_margin", 0.0))
        max_pairs = int(getattr(self._config, "pdm_score_pairwise_max_pairs", 0) or 0)

        valid = torch.isfinite(pred_scores) & torch.isfinite(target_scores)
        if valid_mode is not None:
            if valid_mode.shape != valid.shape:
                return None
            valid = valid & valid_mode
        if not valid.any():
            return None

        total_loss = pred_scores.new_zeros(())
        total_pairs = 0
        for b in range(pred_scores.shape[0]):
            idx = torch.nonzero(valid[b], as_tuple=False).view(-1)
            if idx.numel() < 2:
                continue
            p = pred_scores[b].index_select(0, idx)
            t = target_scores[b].index_select(0, idx)
            target_diff = t[:, None] - t[None, :]
            pair_mask = target_diff > min_gap
            if not pair_mask.any():
                continue
            pred_diff = p[:, None] - p[None, :]
            if margin > 0.0:
                pair_loss = F.relu(margin - pred_diff)
            else:
                pair_loss = F.softplus(-pred_diff)
            pair_loss = pair_loss[pair_mask]
            if pair_loss.numel() == 0:
                continue
            if max_pairs > 0 and pair_loss.numel() > max_pairs:
                k = min(max_pairs, pair_loss.numel())
                pair_loss = torch.topk(pair_loss, k=k, largest=True).values
            total_loss = total_loss + pair_loss.sum()
            total_pairs += int(pair_loss.numel())

        if total_pairs <= 0:
            return None
        return total_loss / float(total_pairs)

    def _extract_hardcase_signals(
        self,
        predictions: Dict[str, torch.Tensor],
    ) -> Optional[Dict[str, torch.Tensor]]:
        score_generalist = predictions.get("pdm_score_generalist")
        if not (
            torch.is_tensor(score_generalist) and score_generalist.dim() == 2
        ):
            score_generalist = predictions.get("pdm_score")
        score_specialist = predictions.get("pdm_score_specialist")

        score_logits = score_generalist
        poses_cls = predictions.get("poses_cls")
        mode_logits: Optional[torch.Tensor] = None
        if torch.is_tensor(score_logits) and score_logits.dim() == 2:
            mode_logits = score_logits
        elif torch.is_tensor(poses_cls) and poses_cls.dim() == 2:
            mode_logits = poses_cls
        if mode_logits is None or mode_logits.numel() == 0:
            return None

        mode_logits = mode_logits.float()
        bs, num_modes = mode_logits.shape
        if bs <= 0 or num_modes <= 0:
            return None

        mode_prob = F.softmax(mode_logits, dim=-1)
        selected_idx = mode_prob.argmax(dim=-1)
        batch_idx = torch.arange(bs, device=mode_prob.device)
        selected_prob = mode_prob[batch_idx, selected_idx]
        selected_logit = mode_logits[batch_idx, selected_idx]

        generalist_selected_idx = selected_idx
        generalist_selected_score = selected_logit
        specialist_selected_idx = selected_idx
        specialist_selected_score = selected_logit
        score_mode_var_generalist = mode_prob.new_zeros((bs,))
        score_mode_var_specialist = mode_prob.new_zeros((bs,))
        specialist_score_on_generalist_mode = selected_logit
        generalist_score_on_specialist_mode = selected_logit
        specialist_residual_on_generalist_mode = mode_prob.new_zeros((bs,))
        specialist_residual_on_specialist_mode = mode_prob.new_zeros((bs,))
        if (
            torch.is_tensor(score_generalist)
            and score_generalist.dim() == 2
            and score_generalist.shape[0] == bs
            and score_generalist.shape[1] == num_modes
        ):
            score_generalist = score_generalist.float().to(mode_logits.device)
            generalist_selected_score = score_generalist[batch_idx, generalist_selected_idx]
            score_mode_var_generalist = score_generalist.var(dim=-1, unbiased=False)
        if (
            torch.is_tensor(score_specialist)
            and score_specialist.dim() == 2
            and score_specialist.shape[0] == bs
            and score_specialist.shape[1] == num_modes
        ):
            score_specialist = score_specialist.float().to(mode_logits.device)
            specialist_selected_idx = score_specialist.argmax(dim=-1)
            specialist_selected_score = score_specialist[batch_idx, specialist_selected_idx]
            specialist_score_on_generalist_mode = score_specialist[
                batch_idx, generalist_selected_idx
            ]
            generalist_score_on_specialist_mode = score_generalist[
                batch_idx, specialist_selected_idx
            ]
            specialist_residual_on_generalist_mode = (
                specialist_score_on_generalist_mode - generalist_selected_score
            )
            specialist_residual_on_specialist_mode = (
                specialist_selected_score - generalist_score_on_specialist_mode
            )
            score_mode_var_specialist = score_specialist.var(dim=-1, unbiased=False)
        selected_gate_prob = mode_prob.new_zeros((bs,))
        gate_prob = predictions.get("hardcase_gate_prob")
        if (
            torch.is_tensor(gate_prob)
            and gate_prob.dim() == 2
            and gate_prob.shape[0] == bs
            and gate_prob.shape[1] == num_modes
        ):
            gate_prob = gate_prob.to(device=mode_prob.device, dtype=mode_prob.dtype)
            selected_gate_prob = gate_prob[batch_idx, selected_idx].clamp(0.0, 1.0)

        adapter_variance = mode_prob.new_zeros((bs,))
        adapter_variance_raw = mode_prob.new_zeros((bs,))
        adapter_variance_mean = mode_prob.new_zeros((bs,))
        adapter_variance_mean_raw = mode_prob.new_zeros((bs,))
        adapter_variance_max = mode_prob.new_zeros((bs,))
        adapter_variance_max_raw = mode_prob.new_zeros((bs,))
        adapter_variance_mode_var_raw = mode_prob.new_zeros((bs,))
        adapter_variance_specialist_selected = mode_prob.new_zeros((bs,))
        adapter_variance_specialist_selected_raw = mode_prob.new_zeros((bs,))
        specialist_uncertainty = predictions.get("pdm_score_specialist_uncertainty")
        if torch.is_tensor(specialist_uncertainty):
            unc_tensor = torch.nan_to_num(
                specialist_uncertainty.float(), nan=0.0, posinf=0.0, neginf=0.0
            )
            if unc_tensor.dim() == 3:
                unc_tensor = unc_tensor.mean(dim=-1)
            if (
                unc_tensor.dim() == 2
                and unc_tensor.shape[0] == bs
                and unc_tensor.shape[1] == num_modes
            ):
                selected_unc = unc_tensor[batch_idx, selected_idx].clamp_min(0.0)
                adapter_variance_raw = selected_unc
                adapter_variance_mean_raw = unc_tensor.mean(dim=-1).clamp_min(0.0)
                adapter_variance_max_raw = unc_tensor.max(dim=-1).values.clamp_min(0.0)
                adapter_variance_mode_var_raw = unc_tensor.var(
                    dim=-1, unbiased=False
                ).clamp_min(0.0)
                unc_ref = max(
                    float(
                        getattr(
                            self._config,
                            "hardcase_adapter_uncertainty_ref",
                            0.05,
                        )
                    ),
                    1e-6,
                )
                adapter_variance = (selected_unc / unc_ref).clamp(0.0, 1.0)
                adapter_variance_mean = (
                    adapter_variance_mean_raw / unc_ref
                ).clamp(0.0, 1.0)
                adapter_variance_max = (
                    adapter_variance_max_raw / unc_ref
                ).clamp(0.0, 1.0)
                if (
                    torch.is_tensor(score_specialist)
                    and score_specialist.dim() == 2
                    and score_specialist.shape[0] == bs
                    and score_specialist.shape[1] == num_modes
                ):
                    specialist_selected_unc = unc_tensor[
                        batch_idx, specialist_selected_idx
                    ].clamp_min(0.0)
                    adapter_variance_specialist_selected_raw = specialist_selected_unc
                    adapter_variance_specialist_selected = (
                        specialist_selected_unc / unc_ref
                    ).clamp(0.0, 1.0)

        hard_threshold = float(getattr(self._config, "hardcase_threshold", 0.55))
        is_hardcase = (adapter_variance >= hard_threshold).to(dtype=mode_prob.dtype)

        return {
            "u_adapter_variance": adapter_variance.clamp(0.0, 1.0),
            "u_adapter_variance_raw": adapter_variance_raw,
            "u_adapter_variance_mean": adapter_variance_mean.clamp(0.0, 1.0),
            "u_adapter_variance_mean_raw": adapter_variance_mean_raw,
            "u_adapter_variance_max": adapter_variance_max.clamp(0.0, 1.0),
            "u_adapter_variance_max_raw": adapter_variance_max_raw,
            "u_adapter_variance_mode_var_raw": adapter_variance_mode_var_raw,
            "u_adapter_variance_specialist_selected": adapter_variance_specialist_selected.clamp(
                0.0, 1.0
            ),
            "u_adapter_variance_specialist_selected_raw": adapter_variance_specialist_selected_raw,
            "is_hardcase": is_hardcase,
            "selected_mode_idx": selected_idx.to(dtype=mode_prob.dtype),
            "selected_mode_prob": selected_prob.clamp(0.0, 1.0),
            "selected_mode_logit": selected_logit,
            "generalist_selected_idx": generalist_selected_idx.to(dtype=mode_prob.dtype),
            "generalist_selected_score": generalist_selected_score,
            "score_mode_var_generalist": score_mode_var_generalist,
            "specialist_selected_idx": specialist_selected_idx.to(dtype=mode_prob.dtype),
            "specialist_selected_score": specialist_selected_score,
            "score_mode_var_specialist": score_mode_var_specialist,
            "specialist_score_on_generalist_mode": specialist_score_on_generalist_mode,
            "specialist_residual_on_generalist_mode": specialist_residual_on_generalist_mode,
            "generalist_score_on_specialist_mode": generalist_score_on_specialist_mode,
            "specialist_residual_on_specialist_mode": specialist_residual_on_specialist_mode,
            "u_gate_prob": selected_gate_prob,
            "num_modes": mode_prob.new_full((bs,), float(num_modes)),
        }

    @staticmethod
    def _summarize_hardcase_signals(
        signals: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        summary: Dict[str, float] = {}
        first_keys = {
            "selected_mode_idx",
            "selected_mode_prob",
            "selected_mode_logit",
            "generalist_selected_idx",
            "generalist_selected_score",
            "specialist_selected_idx",
            "specialist_selected_score",
            "specialist_score_on_generalist_mode",
            "specialist_residual_on_generalist_mode",
            "generalist_score_on_specialist_mode",
            "specialist_residual_on_specialist_mode",
            "u_gate_prob",
            "num_modes",
            "is_hardcase",
        }
        for key, value in signals.items():
            if not torch.is_tensor(value):
                continue
            flat = value.detach().float().cpu().view(-1)
            if flat.numel() <= 0:
                continue
            if key in first_keys:
                summary[key] = float(flat[0].item())
            else:
                summary[key] = float(flat.mean().item())
        return summary

    def _append_hardcase_val_metrics(
        self,
        metrics: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
    ) -> None:
        if not bool(getattr(self._config, "hardcase_record_enable", False)):
            return
        signals = self._extract_hardcase_signals(predictions)
        if signals is None:
            return
        key_map = {
            "u_adapter_variance": "hard_u_adapter_variance",
            "u_adapter_variance_raw": "hard_u_adapter_variance_raw",
            "selected_mode_prob": "hard_selected_mode_prob",
            "u_gate_prob": "hard_u_gate_prob",
            "num_modes": "hard_num_modes",
            "is_hardcase": "hard_case_rate",
        }
        for src_key, dst_key in key_map.items():
            tensor = signals.get(src_key)
            if torch.is_tensor(tensor):
                metrics[dst_key] = tensor.mean()

    def _load_r2se_gpd_params(self) -> Optional[Dict[str, float]]:
        if self._r2se_gpd_loaded:
            return self._r2se_gpd_params
        self._r2se_gpd_loaded = True
        gpd_path = getattr(self._config, "hardcase_r2se_gpd_param_path", None)
        if not gpd_path:
            return None
        try:
            path = Path(str(gpd_path)).expanduser()
            if not path.exists():
                logger.warning("R2SE GPD parameter file not found: %s", path)
                return None
            data = json.loads(path.read_text(encoding="utf-8"))
            u0 = float(data.get("u0", data.get("threshold", 0.0)))
            shape = float(data.get("shape", data.get("xi", data.get("c", 0.0))))
            scale = float(data.get("scale", data.get("beta", 0.0)))
            if not math.isfinite(scale) or scale <= 0.0:
                logger.warning("R2SE GPD parameter invalid scale=%s in %s", scale, path)
                return None
            if not math.isfinite(u0):
                u0 = 0.0
            if not math.isfinite(shape):
                shape = 0.0
            self._r2se_gpd_params = {
                "u0": u0,
                "shape": shape,
                "scale": scale,
            }
            return self._r2se_gpd_params
        except Exception as exc:
            logger.warning("Failed to load R2SE GPD parameters: %s", exc)
            return None

    @staticmethod
    def _gpd_cdf_from_u(
        u: torch.Tensor,
        u0: float,
        shape: float,
        scale: float,
    ) -> torch.Tensor:
        x = (u - float(u0)).clamp_min(0.0)
        if abs(float(shape)) < 1e-8:
            cdf = 1.0 - torch.exp(-x / max(float(scale), 1e-6))
            return cdf.clamp(0.0, 1.0)
        xi = float(shape)
        base = 1.0 + xi * x / max(float(scale), 1e-6)
        if xi < 0.0:
            # Finite upper support for xi<0: beyond boundary => CDF=1.
            finite_mask = base > 0.0
            cdf = torch.ones_like(x)
            if finite_mask.any():
                cdf[finite_mask] = 1.0 - torch.pow(base[finite_mask], -1.0 / xi)
            return cdf.clamp(0.0, 1.0)
        base = torch.clamp(base, min=1e-8)
        cdf = 1.0 - torch.pow(base, -1.0 / xi)
        return cdf.clamp(0.0, 1.0)

    def _select_generalist_trajectory(
        self,
        predictions: Dict[str, torch.Tensor],
    ) -> Optional[torch.Tensor]:
        direct = predictions.get("trajectory_generalist")
        if torch.is_tensor(direct):
            if direct.dim() == 4 and direct.shape[1] == 1:
                direct = direct[:, 0]
            if direct.dim() == 3:
                return direct
        poses_reg = predictions.get("poses_reg")
        if not torch.is_tensor(poses_reg) or poses_reg.dim() != 4:
            return None
        bs, num_modes, num_steps, state_dim = poses_reg.shape
        if bs <= 0 or num_modes <= 0:
            return None
        policy = str(getattr(self._config, "hardcase_r2se_fallback_policy", "poses_cls")).lower()
        mode_idx = torch.zeros(bs, device=poses_reg.device, dtype=torch.long)
        if policy == "poses_cls":
            poses_cls = predictions.get("poses_cls")
            if (
                torch.is_tensor(poses_cls)
                and poses_cls.dim() == 2
                and poses_cls.shape[0] == bs
                and poses_cls.shape[1] >= num_modes
            ):
                mode_idx = poses_cls[:, :num_modes].argmax(dim=-1).long()
        elif policy == "first_mode":
            mode_idx = torch.zeros(bs, device=poses_reg.device, dtype=torch.long)
        else:
            # Unknown policy: fallback to poses_cls if possible, otherwise mode 0.
            poses_cls = predictions.get("poses_cls")
            if (
                torch.is_tensor(poses_cls)
                and poses_cls.dim() == 2
                and poses_cls.shape[0] == bs
                and poses_cls.shape[1] >= num_modes
            ):
                mode_idx = poses_cls[:, :num_modes].argmax(dim=-1).long()
        gather_idx = mode_idx[:, None, None, None].expand(-1, 1, num_steps, state_dim)
        return torch.gather(poses_reg, 1, gather_idx).squeeze(1)

    def _apply_r2se_inference_switch(
        self,
        predictions: Dict[str, torch.Tensor],
        signals: Optional[Dict[str, torch.Tensor]],
    ) -> Dict[str, float]:
        debug: Dict[str, float] = {"enabled": 1.0}
        if signals is None:
            debug["signals_valid"] = 0.0
            return debug
        u_key = str(getattr(self._config, "hardcase_r2se_u_key", "u_adapter_variance"))
        u_tensor = signals.get(u_key)
        if not torch.is_tensor(u_tensor):
            u_tensor = signals.get("u_adapter_variance")
        if not torch.is_tensor(u_tensor):
            debug["signals_valid"] = 0.0
            return debug
        u_tensor = torch.nan_to_num(
            u_tensor.detach().float(),
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        ).clamp(0.0, 1.0)
        if u_tensor.dim() == 0:
            u_tensor = u_tensor.view(1)
        debug["signals_valid"] = 1.0
        debug["u_mean"] = float(u_tensor.mean().item())

        gpd = self._load_r2se_gpd_params()
        if gpd is None:
            pgpd = u_tensor
            debug["gpd_available"] = 0.0
            debug["u0"] = 0.0
            debug["shape"] = 0.0
            debug["scale"] = 0.0
        else:
            pgpd = self._gpd_cdf_from_u(
                u=u_tensor,
                u0=float(gpd["u0"]),
                shape=float(gpd["shape"]),
                scale=float(gpd["scale"]),
            )
            debug["gpd_available"] = 1.0
            debug["u0"] = float(gpd["u0"])
            debug["shape"] = float(gpd["shape"])
            debug["scale"] = float(gpd["scale"])

        sigma = float(getattr(self._config, "hardcase_r2se_sigma", 0.75))
        sigma = max(0.0, min(1.0, sigma))
        force_specialist = bool(
            getattr(self._config, "hardcase_r2se_force_specialist", False)
        )
        if force_specialist:
            switch_specialist = torch.ones_like(pgpd, dtype=torch.float32)
        else:
            switch_specialist = (pgpd > sigma).float()
        debug["sigma"] = sigma
        debug["force_specialist"] = 1.0 if force_specialist else 0.0
        debug["pgpd_mean"] = float(pgpd.mean().item())
        debug["switch_specialist_rate"] = float(switch_specialist.mean().item())

        predictions["r2se_u"] = u_tensor
        predictions["r2se_pgpd"] = pgpd
        predictions["r2se_switch_specialist"] = switch_specialist

        apply_fallback = bool(getattr(self._config, "hardcase_r2se_apply_fallback", True))
        if not apply_fallback:
            debug["fallback_applied"] = 0.0
            return debug

        score_generalist = predictions.get("pdm_score_generalist")
        if not torch.is_tensor(score_generalist):
            score_generalist = predictions.get("pdm_score")
        score_specialist = predictions.get("pdm_score_specialist")
        score_switch_applied = 0.0
        if (
            torch.is_tensor(score_generalist)
            and torch.is_tensor(score_specialist)
            and score_generalist.shape == score_specialist.shape
            and score_generalist.shape[0] == switch_specialist.shape[0]
        ):
            score_mask = switch_specialist[:, None].to(dtype=torch.bool)
            predictions["pdm_score"] = torch.where(
                score_mask, score_specialist, score_generalist
            )
            score_switch_applied = 1.0
        debug["score_switch_applied"] = score_switch_applied

        specialist_traj = predictions.get("trajectory_specialist")
        if not torch.is_tensor(specialist_traj):
            specialist_traj = predictions.get("trajectory")
        generalist_traj = predictions.get("trajectory_generalist")
        if not torch.is_tensor(generalist_traj):
            generalist_traj = self._select_generalist_trajectory(predictions)
        if (
            not torch.is_tensor(specialist_traj)
            or not torch.is_tensor(generalist_traj)
            or specialist_traj.shape != generalist_traj.shape
            or specialist_traj.shape[0] != switch_specialist.shape[0]
        ):
            debug["fallback_applied"] = 0.0
            return debug

        traj_mask = switch_specialist[:, None, None].to(dtype=torch.bool)
        predictions["trajectory"] = torch.where(traj_mask, specialist_traj, generalist_traj)
        debug["fallback_applied"] = 1.0
        debug["fallback_generalist_rate"] = float((1.0 - switch_specialist).mean().item())
        return debug

    def _on_inference_predictions(self, predictions: Dict[str, torch.Tensor]) -> None:
        hardcase_record_enable = bool(getattr(self._config, "hardcase_record_enable", False))
        r2se_enable = bool(getattr(self._config, "hardcase_r2se_enable", False))
        if not hardcase_record_enable and not r2se_enable:
            self._last_inference_debug = {}
            return
        signals = self._extract_hardcase_signals(predictions)
        debug: Dict[str, float] = {}
        if hardcase_record_enable and signals is not None:
            summary = self._summarize_hardcase_signals(signals)
            debug.update({f"hc_{k}": v for k, v in summary.items()})
        if r2se_enable:
            r2se_debug = self._apply_r2se_inference_switch(predictions, signals)
            if bool(getattr(self._config, "hardcase_r2se_debug_record", True)):
                debug.update({f"r2se_{k}": v for k, v in r2se_debug.items()})
        self._last_inference_debug = debug

    def get_last_inference_debug(self) -> Dict[str, float]:
        return dict(self._last_inference_debug)

    def compute_pdm_val_metrics(
        self,
        targets: Dict[str, Any],
        predictions: Dict[str, torch.Tensor],
        metric_cache_path_override: Optional[str] = None,
    ) -> Optional[Dict[str, torch.Tensor]]:
        if not bool(getattr(self._config, "pdm_val_use_online_score", False)):
            return None
        supervisor = self._get_pdm_supervision(
            cache_path_override=metric_cache_path_override
        )
        if supervisor is None:
            self._log_pdm_val_invalid(
                targets, predictions, reason="pdm_supervision_missing"
            )
            return None
        use_selected_traj = bool(
            getattr(self._config, "pdm_val_score_use_selected_trajectory", False)
        )
        poses_reg = predictions.get("poses_reg")
        if poses_reg is None and not use_selected_traj:
            return None
        tokens = targets.get("token")
        if tokens is None:
            self._log_pdm_val_invalid(targets, predictions, reason="pdm_token_missing")
            return None
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.detach().cpu().tolist()
        elif isinstance(tokens, (str, bytes)):
            tokens = [tokens]
        if not isinstance(tokens, (list, tuple)) or len(tokens) == 0:
            return None

        if use_selected_traj:
            selected_traj = predictions.get("trajectory")
            if not torch.is_tensor(selected_traj):
                return None
            if selected_traj.dim() == 4 and selected_traj.shape[1] == 1:
                selected_traj = selected_traj[:, 0]
            if (
                selected_traj.dim() != 3
                or selected_traj.shape[0] != len(tokens)
                or selected_traj.shape[-1] < 3
            ):
                return None
            worker_state = getattr(supervisor, "_worker_state", None)
            if worker_state is None:
                return None
            future_sampling = getattr(worker_state, "_proposal_sampling", None)
            simulator = getattr(worker_state, "_simulator", None)
            scorer = getattr(worker_state, "_scorer", None)
            load_metric_cache = getattr(worker_state, "_load_metric_cache", None)
            if (
                future_sampling is None
                or simulator is None
                or scorer is None
                or load_metric_cache is None
            ):
                return None

            with torch.no_grad():
                traj_np = selected_traj.detach().cpu().numpy().astype("float32", copy=False)
            score_list = []
            comp_list = []
            for token, traj_row in zip(list(tokens), traj_np):
                metric_cache = load_metric_cache(token)
                if metric_cache is None:
                    continue
                try:
                    pdm_result = evaluate_pdm_score(
                        metric_cache=metric_cache,
                        model_trajectory=Trajectory(traj_row),
                        future_sampling=future_sampling,
                        simulator=simulator,
                        scorer=scorer,
                    )
                except Exception:
                    continue
                score_list.append(float(pdm_result.score))
                comp_list.append(
                    [
                        float(pdm_result.no_at_fault_collisions),
                        float(pdm_result.drivable_area_compliance),
                        float(pdm_result.ego_progress),
                        float(pdm_result.time_to_collision_within_bound),
                        float(pdm_result.comfort),
                        float(pdm_result.driving_direction_compliance),
                    ]
                )

            if len(score_list) == 0:
                self._log_pdm_val_invalid(
                    targets, predictions, reason="pdm_selected_traj_scores_invalid"
                )
                return None

            scores = torch.tensor(
                score_list, device=selected_traj.device, dtype=selected_traj.dtype
            )
            comps = torch.tensor(
                comp_list, device=selected_traj.device, dtype=selected_traj.dtype
            )

            one = scores.new_ones(())
            zero = scores.new_zeros(())
            metrics = {
                "score": scores.mean(),
                "best_score": scores.mean(),
                "mean_score": scores.mean(),
                "score_hit_rate": one,
                "lost_score": zero,
                "top_5_score_hit_rate": one,
                "selected_mode_oracle_rank": one,
                "selected_mode_oracle_percentile": one,
                "selected_mode_oracle_score": scores.mean(),
                "best_mode_oracle_score": scores.mean(),
                "collision": comps[:, 0].mean(),
                "dac": comps[:, 1].mean(),
                "progress": comps[:, 2].mean(),
                "ttc": comps[:, 3].mean(),
                "comfort": comps[:, 4].mean(),
                "ddc": comps[:, 5].mean(),
            }
            self._append_hardcase_val_metrics(metrics, predictions)
            return metrics

        poses_cls = predictions.get("poses_cls")
        score_logits = predictions.get("pdm_score")
        num_modes = poses_reg.shape[1]
        mode_indices = torch.arange(
            num_modes, device=poses_reg.device, dtype=torch.long
        )[None, :].expand(poses_reg.shape[0], -1)
        topk = int(getattr(self._config, "pdm_val_score_topk", 0) or 0)
        if topk > 0 and topk < num_modes:
            topk_source = poses_cls if poses_cls is not None else score_logits
            if topk_source is None:
                topk = 0
        if topk > 0 and topk < num_modes:
            topk_idx = torch.topk(topk_source.detach(), topk, dim=-1).indices
            mode_indices = topk_idx
            gather_idx = topk_idx[..., None, None].expand(
                -1, -1, poses_reg.shape[2], poses_reg.shape[3]
            )
            poses_reg = torch.gather(poses_reg, 1, gather_idx)
            if poses_cls is not None:
                poses_cls = torch.gather(poses_cls, 1, topk_idx)
            if score_logits is not None:
                score_logits = torch.gather(score_logits, 1, topk_idx)

        traj_np = poses_reg.detach().cpu().numpy()
        output_infraction = self._should_output_pdm_infraction_details()
        with torch.no_grad():
            if output_infraction:
                scores_np, infraction_data = supervisor.score_batch_with_infraction_data(
                    list(tokens), traj_np
                )
                comps_np, _ = supervisor.score_batch_components_with_infraction_data(
                    list(tokens), traj_np
                )
                self._attach_pdm_infraction_details_to_predictions(
                    predictions=predictions,
                    infraction_data=infraction_data,
                    mode_indices=mode_indices,
                    device=poses_reg.device,
                    dtype=poses_reg.dtype,
                )
            else:
                scores_np = supervisor.score_batch(list(tokens), traj_np)
                comps_np = supervisor.score_batch_components(list(tokens), traj_np)

        scores = torch.from_numpy(scores_np).to(
            device=poses_reg.device, dtype=poses_reg.dtype
        )
        scores = torch.nan_to_num(scores, nan=float("-inf"))
        comps = torch.from_numpy(comps_np).to(
            device=poses_reg.device, dtype=poses_reg.dtype
        )
        comps = torch.nan_to_num(comps, nan=float("nan"))

        pred_idx = None
        use_pdm_select = bool(getattr(self._config, "pdm_score_use_for_selection", False))
        if use_pdm_select and score_logits is not None:
            topk_select = int(getattr(self._config, "pdm_score_select_topk", 0) or 0)
            if poses_cls is not None and 0 < topk_select < score_logits.shape[1]:
                topk_idx = torch.topk(poses_cls, topk_select, dim=-1).indices
                topk_scores = torch.gather(score_logits, 1, topk_idx)
                best_in_topk = topk_scores.argmax(dim=-1, keepdim=True)
                pred_idx = topk_idx.gather(1, best_in_topk).squeeze(1)
            else:
                pred_idx = score_logits.argmax(dim=-1)

        if pred_idx is None:
            if poses_cls is not None:
                pred_idx = poses_cls.argmax(dim=-1)
            elif score_logits is not None:
                pred_idx = score_logits.argmax(dim=-1)
            else:
                pred_idx = torch.zeros(scores.shape[0], device=scores.device, dtype=torch.long)

        batch_idx = torch.arange(scores.shape[0], device=scores.device)
        best_idx = scores.argmax(dim=-1)
        best_scores = scores[batch_idx, best_idx]
        pred_scores = scores[batch_idx, pred_idx]
        mean_scores = scores.mean(dim=-1)
        lost_score = best_scores - pred_scores
        selected_rank = (
            (scores > pred_scores[:, None]).sum(dim=-1).to(dtype=poses_reg.dtype) + 1.0
        )
        if scores.shape[1] > 1:
            selected_percentile = 1.0 - (
                (selected_rank - 1.0) / float(scores.shape[1] - 1)
            )
        else:
            selected_percentile = torch.ones_like(selected_rank)

        topk_val = min(5, scores.shape[1])
        top5_idx = torch.topk(scores, k=topk_val, dim=-1).indices
        hit_top5 = (top5_idx == pred_idx[:, None]).any(dim=-1).float()

        selected_components = comps[batch_idx, pred_idx]

        metrics = {
            "score": pred_scores.mean(),
            "best_score": best_scores.mean(),
            "mean_score": mean_scores.mean(),
            "score_hit_rate": (pred_idx == best_idx).float().mean(),
            "lost_score": lost_score.mean(),
            "top_5_score_hit_rate": hit_top5.mean(),
            "selected_mode_oracle_rank": selected_rank.mean(),
            "selected_mode_oracle_percentile": selected_percentile.mean(),
            "selected_mode_oracle_score": pred_scores.mean(),
            "best_mode_oracle_score": best_scores.mean(),
            "collision": selected_components[:, 0].mean(),
            "dac": selected_components[:, 1].mean(),
            "progress": selected_components[:, 2].mean(),
            "ttc": selected_components[:, 3].mean(),
            "comfort": selected_components[:, 4].mean(),
            "ddc": selected_components[:, 5].mean(),
        }
        self._append_hardcase_val_metrics(metrics, predictions)
        return metrics

    def _select_b2d_pseudo_score_trajectories(
        self,
        predictions: Dict[str, torch.Tensor],
    ) -> Optional[torch.Tensor]:
        selected_traj = predictions.get("trajectory")
        if torch.is_tensor(selected_traj):
            if selected_traj.dim() == 4 and selected_traj.shape[1] == 1:
                selected_traj = selected_traj[:, 0]
            if selected_traj.dim() == 3 and selected_traj.shape[-1] >= 2:
                return selected_traj.unsqueeze(1)

        poses_reg = predictions.get("poses_reg")
        if not torch.is_tensor(poses_reg) or poses_reg.dim() != 4:
            return None
        batch_size, num_modes = poses_reg.shape[:2]
        if batch_size <= 0 or num_modes <= 0:
            return None
        mode_idx = self._select_mode_indices_for_gate(
            predictions=predictions,
            batch_size=batch_size,
            num_modes=num_modes,
            device=poses_reg.device,
        ).clamp(0, num_modes - 1)
        batch_idx = torch.arange(batch_size, device=poses_reg.device)
        return poses_reg[batch_idx, mode_idx].unsqueeze(1)

    @staticmethod
    def _compute_b2d_pseudo_score_from_components(
        components: torch.Tensor,
    ) -> torch.Tensor:
        safe = torch.nan_to_num(components.float(), nan=0.0, posinf=0.0, neginf=0.0)
        safe = safe.clamp(0.0, 1.0)
        no_collision = safe[..., 0]
        drivable = safe[..., 1]
        progress = safe[..., 2]
        ttc = safe[..., 3]
        comfort = safe[..., 4]
        wrong_lane = safe[..., 5] if safe.shape[-1] > 5 else torch.ones_like(no_collision)
        route_quality = (5.0 * progress + 5.0 * ttc + 2.0 * comfort) / 12.0
        return (no_collision * drivable * wrong_lane * route_quality).clamp(0.0, 1.0)

    def _compute_b2d_pseudo_selected_score_metrics(
        self,
        targets: Dict[str, Any],
        predictions: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        if not bool(getattr(self._config, "b2d_pdm_score_enable", False)):
            return {}
        trajectories = self._select_b2d_pseudo_score_trajectories(predictions)
        if not torch.is_tensor(trajectories):
            return {}
        components = self._compute_b2d_pdm_components(targets, trajectories)
        if not torch.is_tensor(components) or components.numel() == 0:
            return {}
        selected_components = components[:, 0]
        selected_scores = self._compute_b2d_pseudo_score_from_components(
            selected_components
        )
        finite = torch.isfinite(selected_scores)
        if not finite.any():
            return {}

        selected_components = selected_components[finite]
        comp_finite = torch.isfinite(selected_components)
        comp_safe = selected_components.masked_fill(~comp_finite, 0.0)
        comp_denom = comp_finite.float().sum(dim=0).clamp_min(1.0)
        comp_mean = comp_safe.sum(dim=0) / comp_denom

        metrics: Dict[str, torch.Tensor] = {
            "b2d_pseudo_selected_score": selected_scores[finite].mean().detach(),
        }
        component_names = (
            "no_collision",
            "drivable",
            "progress",
            "ttc",
            "comfort",
            "wrong_lane",
        )
        for idx, name in enumerate(component_names):
            if idx < comp_mean.numel():
                metrics[f"b2d_pseudo_selected_{name}"] = comp_mean[idx].detach()
        return metrics

    @staticmethod
    def _align_batch_tensor(tensor: torch.Tensor, batch_size: int) -> Optional[torch.Tensor]:
        if tensor.shape[0] == batch_size:
            return tensor
        if tensor.shape[0] == 1:
            return tensor.expand(batch_size, *tensor.shape[1:])
        if batch_size == 1:
            return tensor[:1]
        return None

    def _b2d_gather_bev_values(
        self,
        bev: torch.Tensor,
        xy: torch.Tensor,
    ) -> torch.Tensor:
        if bev.dim() == 2:
            bev = bev.unsqueeze(0)
        if bev.dim() != 3:
            return torch.zeros(xy.shape[:-1], device=xy.device, dtype=xy.dtype)
        bev = bev.to(device=xy.device, dtype=xy.dtype)
        batch_size = xy.shape[0]
        bev = self._align_batch_tensor(bev, batch_size)
        if bev is None:
            return torch.zeros(xy.shape[:-1], device=xy.device, dtype=xy.dtype)

        h, w = bev.shape[-2:]
        pixel_size = max(float(getattr(self._config, "b2d_pdm_bev_pixel_size", 0.25)), 1e-6)
        lateral_origin = float(getattr(self._config, "b2d_pdm_bev_left_m", 32.0))
        row = torch.round(xy[..., 0] / pixel_size).long()
        col = torch.round(xy[..., 1] / pixel_size + lateral_origin / pixel_size).long()
        valid = (row >= 0) & (row < h) & (col >= 0) & (col < w)
        flat_idx = (row.clamp(0, h - 1) * w + col.clamp(0, w - 1)).reshape(batch_size, -1)
        flat_bev = bev.reshape(batch_size, -1)
        values = torch.gather(flat_bev, dim=1, index=flat_idx).reshape(xy.shape[:-1])
        return values * valid.to(dtype=xy.dtype)

    def _b2d_gather_lane_direction(
        self,
        targets: Dict[str, Any],
        xy: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        direction = targets.get("lane_direction_bev")
        if not torch.is_tensor(direction):
            return None, None
        if direction.dim() == 3:
            direction = direction.unsqueeze(0)
        if direction.dim() != 4 or direction.shape[1] != 2:
            return None, None
        batch_size = xy.shape[0]
        direction = direction.to(device=xy.device, dtype=xy.dtype)
        direction = self._align_batch_tensor(direction, batch_size)
        if direction is None:
            return None, None

        values = []
        for channel in range(2):
            values.append(self._b2d_gather_bev_values(direction[:, channel], xy))
        vec = torch.stack(values, dim=-1)
        mask = targets.get("lane_direction_mask")
        if torch.is_tensor(mask):
            lane_valid = self._b2d_gather_bev_values(mask, xy) > 0.5
        else:
            lane_valid = torch.linalg.norm(vec, dim=-1) > 1e-3
        return vec, lane_valid

    def _b2d_ego_corners_from_traj(self, traj: torch.Tensor) -> torch.Tensor:
        """Build PAD-style ego boxes for trajectories shaped [B, M, T, 3]."""
        xy = traj[..., :2]
        if traj.shape[-1] >= 3:
            heading = traj[..., 2]
        else:
            delta = xy[..., 1:, :] - xy[..., :-1, :]
            heading = torch.atan2(delta[..., 1], delta[..., 0])
            heading = torch.cat([heading, heading[..., -1:]], dim=-1)

        half_l = float(getattr(self._config, "b2d_pdm_ego_half_length", 2.042))
        half_w = float(getattr(self._config, "b2d_pdm_ego_half_width", 0.925))
        rear_to_center = float(getattr(self._config, "b2d_pdm_ego_rear_axle_to_center", 0.39))
        local = torch.tensor(
            [[half_l, half_w], [-half_l, half_w], [-half_l, -half_w], [half_l, -half_w]],
            device=traj.device,
            dtype=traj.dtype,
        )
        cos_yaw = torch.cos(heading)
        sin_yaw = torch.sin(heading)
        center = xy + rear_to_center * torch.stack([cos_yaw, sin_yaw], dim=-1)
        rot_x = cos_yaw[..., None] * local[:, 0] - sin_yaw[..., None] * local[:, 1]
        rot_y = sin_yaw[..., None] * local[:, 0] + cos_yaw[..., None] * local[:, 1]
        return torch.stack([rot_x + center[..., 0:1], rot_y + center[..., 1:2]], dim=-1)

    @staticmethod
    def _b2d_rect_axes(corners: torch.Tensor) -> torch.Tensor:
        edge_long = corners[..., 1, :] - corners[..., 0, :]
        edge_lat = corners[..., 3, :] - corners[..., 0, :]
        axes = torch.stack([edge_long, edge_lat], dim=-2)
        return axes / torch.linalg.norm(axes, dim=-1, keepdim=True).clamp_min(1e-6)

    def _b2d_rect_intersects(self, ego_corners: torch.Tensor, agent_corners: torch.Tensor) -> torch.Tensor:
        ego_axes = self._b2d_rect_axes(ego_corners)
        agent_axes = self._b2d_rect_axes(agent_corners)
        axes = torch.cat([ego_axes, agent_axes], dim=-2)
        ego_proj = (ego_corners[..., None, :, :] * axes[..., :, None, :]).sum(dim=-1)
        agent_proj = (agent_corners[..., None, :, :] * axes[..., :, None, :]).sum(dim=-1)
        ego_min = ego_proj.amin(dim=-1)
        ego_max = ego_proj.amax(dim=-1)
        agent_min = agent_proj.amin(dim=-1)
        agent_max = agent_proj.amax(dim=-1)
        return ((ego_max >= agent_min) & (agent_max >= ego_min)).all(dim=-1)

    def _b2d_pad_collision_and_ttc(
        self,
        traj: torch.Tensor,
        target_traj: Optional[torch.Tensor],
        boxes: torch.Tensor,
        boxes_mask: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """PAD-style polygon collision and TTC, with GT collisions exempted."""
        batch_size, num_modes, num_steps = traj.shape[:3]
        if boxes.dim() == 4:
            boxes = boxes.unsqueeze(0)
        if boxes_mask.dim() == 2:
            boxes_mask = boxes_mask.unsqueeze(0)
        boxes = self._align_batch_tensor(boxes, batch_size)
        boxes_mask = self._align_batch_tensor(boxes_mask, batch_size)
        if boxes is None or boxes_mask is None or boxes.numel() == 0:
            return None, None

        steps = min(num_steps, boxes.shape[2], boxes_mask.shape[2])
        if steps <= 0:
            return None, None
        traj = traj[:, :, :steps]
        boxes = boxes[:, :, :steps].to(device=traj.device, dtype=traj.dtype)
        boxes_mask = boxes_mask[:, :, :steps].to(device=traj.device).bool()

        dt = max(float(self._config.trajectory_sampling.interval_length), 1e-3)
        xy = traj[..., :2]
        vel = torch.cat([xy[..., :1, :], xy[..., 1:, :] - xy[..., :-1, :]], dim=-2) / dt
        expanded = [traj]
        for horizon in (dt, 2.0 * dt):
            shifted = traj.clone()
            shifted[..., :2] = xy + vel * horizon
            expanded.append(shifted)
        ttc_traj = torch.stack(expanded, dim=3)
        ego_corners = self._b2d_ego_corners_from_traj(ttc_traj.reshape(batch_size, num_modes, steps * 3, -1))
        ego_corners = ego_corners.reshape(batch_size, num_modes, steps, 3, 4, 2)

        num_agents = boxes.shape[1]
        ego_exp = ego_corners[:, :, None].expand(-1, -1, num_agents, -1, -1, -1, -1)
        agent_exp = boxes[:, None, :, :, None].expand(-1, num_modes, -1, -1, 3, -1, -1)
        intersects = self._b2d_rect_intersects(ego_exp, agent_exp)
        valid = boxes_mask[:, None, :, :, None]
        intersects = intersects & valid
        candidate_collision_by_step = intersects[..., 0].any(dim=2)
        candidate_ttc_by_step = intersects[..., 1:].any(dim=(2, 4))

        gt_collision_by_step = None
        gt_ttc_by_step = None
        if torch.is_tensor(target_traj):
            gt = target_traj.to(device=traj.device, dtype=traj.dtype)
            if gt.dim() == 2:
                gt = gt.unsqueeze(0)
            gt = self._align_batch_tensor(gt, batch_size)
            if gt is not None and gt.shape[-1] >= 2 and gt.shape[1] >= steps:
                gt = gt[:, :steps].unsqueeze(1)
                gt_xy = gt[..., :2]
                gt_vel = torch.cat([gt_xy[..., :1, :], gt_xy[..., 1:, :] - gt_xy[..., :-1, :]], dim=-2) / dt
                gt_expanded = [gt]
                for horizon in (dt, 2.0 * dt):
                    shifted = gt.clone()
                    shifted[..., :2] = gt_xy + gt_vel * horizon
                    gt_expanded.append(shifted)
                gt_ttc_traj = torch.stack(gt_expanded, dim=3)
                gt_corners = self._b2d_ego_corners_from_traj(gt_ttc_traj.reshape(batch_size, 1, steps * 3, -1))
                gt_corners = gt_corners.reshape(batch_size, 1, steps, 3, 4, 2)
                gt_ego_exp = gt_corners[:, :, None].expand(-1, -1, num_agents, -1, -1, -1, -1)
                gt_agent_exp = boxes[:, None, :, :, None].expand(-1, 1, -1, -1, 3, -1, -1)
                gt_intersects = self._b2d_rect_intersects(gt_ego_exp, gt_agent_exp) & valid[:, :1]
                gt_collision_by_step = gt_intersects[..., 0].any(dim=2).squeeze(1)
                gt_ttc_by_step = gt_intersects[..., 1:].any(dim=(2, 4)).squeeze(1)

        if gt_collision_by_step is not None:
            candidate_collision_by_step = candidate_collision_by_step & (~gt_collision_by_step[:, None])
        if gt_ttc_by_step is not None:
            candidate_ttc_by_step = candidate_ttc_by_step & (~gt_ttc_by_step[:, None])
        candidate_collision = candidate_collision_by_step.any(dim=-1)
        candidate_ttc = candidate_ttc_by_step.any(dim=-1)
        return (~candidate_collision).to(dtype=traj.dtype), (~candidate_ttc).to(dtype=traj.dtype)

    def _b2d_wrong_direction_enabled(
        self,
        targets: Dict[str, Any],
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        value = targets.get("wrong_direction_check_enabled", True)
        if torch.is_tensor(value):
            enabled = value.to(device=device).bool().view(-1)
        elif isinstance(value, (list, tuple)):
            enabled = torch.as_tensor(value, device=device, dtype=torch.bool).view(-1)
        else:
            enabled = torch.full((batch_size,), bool(value), device=device, dtype=torch.bool)
        if enabled.numel() == 1 and batch_size > 1:
            enabled = enabled.expand(batch_size)
        if enabled.numel() != batch_size:
            enabled = torch.ones((batch_size,), device=device, dtype=torch.bool)
        return enabled

    def _compute_b2d_pdm_components(
        self,
        targets: Dict[str, Any],
        trajectories: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if trajectories is None or trajectories.dim() != 4 or trajectories.shape[-1] < 2:
            return None
        traj = trajectories.detach()
        device = traj.device
        dtype = traj.dtype
        batch_size, num_modes, num_steps = traj.shape[:3]
        xy = traj[..., :2]

        feasible_area = targets.get("feasible_area_mask")
        if torch.is_tensor(feasible_area):
            drivable = self._b2d_gather_bev_values(feasible_area, xy).mean(dim=-1).clamp(0.0, 1.0)
        else:
            drivable = torch.ones((batch_size, num_modes), device=device, dtype=dtype)

        target_traj = targets.get("trajectory")
        if torch.is_tensor(target_traj):
            target_traj = target_traj.to(device=device, dtype=dtype)
            if target_traj.dim() == 2:
                target_traj = target_traj.unsqueeze(0)
            target_traj = self._align_batch_tensor(target_traj, batch_size)
        if torch.is_tensor(target_traj):
            ref_progress = target_traj[:, -1, 0].abs().clamp_min(1.0)
        else:
            ref_progress = torch.full((batch_size,), 10.0, device=device, dtype=dtype)
        progress = (xy[:, :, -1, 0].clamp_min(0.0) / ref_progress[:, None]).clamp(0.0, 1.0)

        boxes = targets.get("future_agent_boxes")
        boxes_mask = targets.get("future_agent_boxes_mask")
        no_collision = torch.ones((batch_size, num_modes), device=device, dtype=dtype)
        ttc = torch.ones_like(no_collision)
        if torch.is_tensor(boxes) and torch.is_tensor(boxes_mask):
            boxes = boxes.to(device=device, dtype=dtype)
            boxes_mask = boxes_mask.to(device=device).bool()
            pad_no_collision: Optional[torch.Tensor] = None
            pad_ttc: Optional[torch.Tensor] = None
            if str(getattr(self._config, "b2d_pdm_ttc_mode", "pad")).lower() == "pad":
                pad_no_collision, pad_ttc = self._b2d_pad_collision_and_ttc(
                    traj=traj,
                    target_traj=target_traj if torch.is_tensor(target_traj) else None,
                    boxes=boxes,
                    boxes_mask=boxes_mask,
                )
            if pad_ttc is not None:
                ttc = pad_ttc.to(device=device, dtype=dtype)
            if boxes.dim() == 4:
                boxes = boxes.unsqueeze(0)
            if boxes_mask.dim() == 2:
                boxes_mask = boxes_mask.unsqueeze(0)
            aligned_boxes = self._align_batch_tensor(boxes, batch_size)
            aligned_boxes_mask = self._align_batch_tensor(boxes_mask, batch_size)
            if aligned_boxes is not None and aligned_boxes_mask is not None and aligned_boxes.numel() > 0:
                steps = min(num_steps, boxes.shape[2], boxes_mask.shape[2])
                centers = aligned_boxes[:, :, :steps].mean(dim=-2)
                dist = torch.linalg.norm(
                    xy[:, :, None, :steps, :] - centers[:, None, :, :, :],
                    dim=-1,
                )
                valid = aligned_boxes_mask[:, None, :, :steps]
                inf = torch.full_like(dist, float("inf"))
                min_dist = torch.where(valid, dist, inf).amin(dim=(2, 3))
                collision_dist = float(getattr(self._config, "b2d_pdm_collision_distance", 2.5))
                safe_dist = max(float(getattr(self._config, "b2d_pdm_ttc_safe_distance", 8.0)), collision_dist + 1e-3)
                no_collision = (min_dist > collision_dist).to(dtype=dtype)
                if pad_no_collision is not None:
                    no_collision = pad_no_collision.to(device=device, dtype=dtype)
                if pad_ttc is None:
                    ttc = ((min_dist - collision_dist) / (safe_dist - collision_dist)).clamp(0.0, 1.0)
                    ttc = torch.where(torch.isfinite(ttc), ttc, torch.ones_like(ttc))

        if traj.shape[-1] >= 3:
            heading = traj[..., 2]
        else:
            delta = xy[..., 1:, :] - xy[..., :-1, :]
            heading = torch.atan2(delta[..., 1], delta[..., 0])
            heading = torch.cat([heading, heading[..., -1:]], dim=-1)

        lane_vec, lane_valid = self._b2d_gather_lane_direction(targets, xy)
        if lane_vec is not None and lane_valid is not None:
            heading_vec = torch.stack([torch.cos(heading), torch.sin(heading)], dim=-1)
            lane_norm = lane_vec / torch.linalg.norm(lane_vec, dim=-1, keepdim=True).clamp_min(1e-6)
            cos_sim = (heading_vec * lane_norm).sum(dim=-1)
            threshold = float(getattr(self._config, "b2d_pdm_wrong_lane_cos_threshold", 0.0))
            ok = (cos_sim > threshold) | (~lane_valid)
            denom = lane_valid.float().sum(dim=-1).clamp_min(1.0)
            raw_wrong_lane = (ok.float() * lane_valid.float()).sum(dim=-1) / denom
            has_lane = lane_valid.any(dim=-1)
            raw_wrong_lane = torch.where(has_lane, raw_wrong_lane, torch.ones_like(raw_wrong_lane))
        else:
            raw_wrong_lane = torch.ones((batch_size, num_modes), device=device, dtype=dtype)
        enabled = self._b2d_wrong_direction_enabled(targets, batch_size, device)
        allow_wrong_lane = ~enabled
        if (
            bool(getattr(self._config, "b2d_pdm_wrong_lane_allow_from_gt", True))
            and torch.is_tensor(target_traj)
        ):
            gt_xy = target_traj[:, :num_steps, :2].to(device=device, dtype=dtype)
            gt_heading = (
                target_traj[:, :num_steps, 2].to(device=device, dtype=dtype)
                if target_traj.shape[-1] >= 3
                else None
            )
            if gt_heading is None:
                gt_delta = gt_xy[:, 1:] - gt_xy[:, :-1]
                gt_heading = torch.atan2(gt_delta[..., 1], gt_delta[..., 0])
                gt_heading = torch.cat([gt_heading, gt_heading[:, -1:]], dim=-1)
            gt_lane_vec, gt_lane_valid = self._b2d_gather_lane_direction(targets, gt_xy[:, None])
            if gt_lane_vec is not None and gt_lane_valid is not None:
                gt_heading_vec = torch.stack([torch.cos(gt_heading), torch.sin(gt_heading)], dim=-1)[:, None]
                gt_lane_norm = gt_lane_vec / torch.linalg.norm(gt_lane_vec, dim=-1, keepdim=True).clamp_min(1e-6)
                gt_cos_sim = (gt_heading_vec * gt_lane_norm).sum(dim=-1)
                threshold = float(getattr(self._config, "b2d_pdm_wrong_lane_cos_threshold", 0.0))
                gt_ok = (gt_cos_sim > threshold) | (~gt_lane_valid)
                gt_denom = gt_lane_valid.float().sum(dim=-1).clamp_min(1.0)
                gt_wrong_lane = (gt_ok.float() * gt_lane_valid.float()).sum(dim=-1) / gt_denom
                gt_has_lane = gt_lane_valid.any(dim=-1)
                gt_wrong_lane = torch.where(gt_has_lane, gt_wrong_lane, torch.ones_like(gt_wrong_lane)).squeeze(1)
                allow_threshold = float(getattr(self._config, "b2d_pdm_gt_wrong_lane_allow_threshold", 0.95))
                allow_wrong_lane = allow_wrong_lane | (gt_wrong_lane < allow_threshold)

        allowed_floor = float(getattr(self._config, "b2d_pdm_wrong_lane_allowed_floor", 0.70))
        allowed_floor = min(max(allowed_floor, 0.0), 1.0)
        strict_power = max(float(getattr(self._config, "b2d_pdm_wrong_lane_strict_power", 2.0)), 1e-3)
        allowed_wrong_lane = allowed_floor + (1.0 - allowed_floor) * raw_wrong_lane
        strict_wrong_lane = raw_wrong_lane.clamp(0.0, 1.0).pow(strict_power)
        wrong_lane = torch.where(allow_wrong_lane[:, None], allowed_wrong_lane, strict_wrong_lane)

        if num_steps >= 3:
            dt = max(float(self._config.trajectory_sampling.interval_length), 1e-3)
            vel = (xy[..., 1:, :] - xy[..., :-1, :]) / dt
            accel = (vel[..., 1:, :] - vel[..., :-1, :]) / dt
            max_accel = torch.linalg.norm(accel, dim=-1).amax(dim=-1)
            yaw_delta = torch.atan2(torch.sin(heading[..., 1:] - heading[..., :-1]), torch.cos(heading[..., 1:] - heading[..., :-1]))
            yaw_rate = (yaw_delta / dt).abs().amax(dim=-1)
            comfort_mode = str(getattr(self._config, "b2d_pdm_comfort_mode", "relative")).lower()
            if comfort_mode in ("relative", "pad_relative", "gt_relative"):
                if torch.is_tensor(target_traj):
                    gt_xy = target_traj[:, :num_steps, :2].to(device=device, dtype=dtype)
                    gt_heading = (
                        target_traj[:, :num_steps, 2].to(device=device, dtype=dtype)
                        if target_traj.shape[-1] >= 3
                        else None
                    )
                    gt_vel = (gt_xy[:, 1:] - gt_xy[:, :-1]) / dt
                    gt_accel = (gt_vel[:, 1:] - gt_vel[:, :-1]) / dt
                    gt_max_accel = torch.linalg.norm(gt_accel, dim=-1).amax(dim=-1)
                    if gt_heading is None:
                        gt_delta = gt_xy[:, 1:] - gt_xy[:, :-1]
                        gt_heading = torch.atan2(gt_delta[..., 1], gt_delta[..., 0])
                        gt_heading = torch.cat([gt_heading, gt_heading[:, -1:]], dim=-1)
                    gt_yaw_delta = torch.atan2(
                        torch.sin(gt_heading[:, 1:] - gt_heading[:, :-1]),
                        torch.cos(gt_heading[:, 1:] - gt_heading[:, :-1]),
                    )
                    gt_yaw_rate = (gt_yaw_delta / dt).abs().amax(dim=-1)
                    accel_floor = float(getattr(self._config, "b2d_pdm_comfort_accel_floor", 0.5))
                    yaw_floor = float(getattr(self._config, "b2d_pdm_comfort_yaw_rate_floor", 0.1))
                    accel_ref = gt_max_accel.clamp_min(max(accel_floor, 1e-3))[:, None]
                    yaw_ref = gt_yaw_rate.clamp_min(max(yaw_floor, 1e-3))[:, None]
                    comfort = torch.minimum(
                        accel_ref / torch.maximum(max_accel, accel_ref),
                        yaw_ref / torch.maximum(yaw_rate, yaw_ref),
                    ).clamp(0.0, 1.0)
                else:
                    comfort = torch.ones((batch_size, num_modes), device=device, dtype=dtype)
            elif comfort_mode in ("pad", "pad_binary", "binary"):
                if torch.is_tensor(target_traj):
                    gt_xy = target_traj[:, :num_steps, :2].to(device=device, dtype=dtype)
                    gt_heading = (
                        target_traj[:, :num_steps, 2].to(device=device, dtype=dtype)
                        if target_traj.shape[-1] >= 3
                        else None
                    )
                    gt_vel = (gt_xy[:, 1:] - gt_xy[:, :-1]) / dt
                    gt_accel = (gt_vel[:, 1:] - gt_vel[:, :-1]) / dt
                    gt_max_accel = torch.linalg.norm(gt_accel, dim=-1).amax(dim=-1)[:, None]
                    if gt_heading is None:
                        gt_delta = gt_xy[:, 1:] - gt_xy[:, :-1]
                        gt_heading = torch.atan2(gt_delta[..., 1], gt_delta[..., 0])
                        gt_heading = torch.cat([gt_heading, gt_heading[:, -1:]], dim=-1)
                    gt_yaw_delta = torch.atan2(
                        torch.sin(gt_heading[:, 1:] - gt_heading[:, :-1]),
                        torch.cos(gt_heading[:, 1:] - gt_heading[:, :-1]),
                    )
                    gt_yaw_rate = (gt_yaw_delta / dt).abs().amax(dim=-1)[:, None]
                    comfort = ((max_accel <= gt_max_accel) & (yaw_rate <= gt_yaw_rate)).to(dtype=dtype)
                else:
                    comfort = torch.ones((batch_size, num_modes), device=device, dtype=dtype)
            else:
                accel_th = float(getattr(self._config, "b2d_pdm_comfort_accel_threshold", 4.5))
                yaw_th = float(getattr(self._config, "b2d_pdm_comfort_yaw_rate_threshold", 1.2))
                comfort = torch.minimum(
                    (1.0 - (max_accel / max(accel_th, 1e-3))).clamp(0.0, 1.0),
                    (1.0 - (yaw_rate / max(yaw_th, 1e-3))).clamp(0.0, 1.0),
                )
        else:
            comfort = torch.ones((batch_size, num_modes), device=device, dtype=dtype)

        return torch.stack(
            [
                no_collision,
                drivable,
                progress,
                ttc,
                comfort,
                wrong_lane,
            ],
            dim=-1,
        ).clamp(0.0, 1.0)

    def _navsim_vehicle_dims(self) -> Tuple[float, float, float]:
        try:
            from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters

            vehicle = get_pacifica_parameters()
            return (
                float(vehicle.half_length) * 2.0,
                float(vehicle.half_width) * 2.0,
                float(vehicle.rear_axle_to_center),
            )
        except Exception:
            half_l = float(getattr(self._config, "b2d_pdm_ego_half_length", 2.042))
            half_w = float(getattr(self._config, "b2d_pdm_ego_half_width", 0.925))
            rear_to_center = float(
                getattr(self._config, "b2d_pdm_ego_rear_axle_to_center", 0.39)
            )
            return 2.0 * half_l, 2.0 * half_w, rear_to_center

    def _navsim_gather_bev_values(
        self,
        bev: torch.Tensor,
        xy: torch.Tensor,
    ) -> torch.Tensor:
        if bev.dim() == 2:
            bev = bev.unsqueeze(0)
        if bev.dim() != 3:
            return torch.zeros(xy.shape[:-1], device=xy.device, dtype=xy.dtype)
        batch_size = xy.shape[0]
        bev = bev.to(device=xy.device, dtype=xy.dtype)
        bev = self._align_batch_tensor(bev, batch_size)
        if bev is None:
            return torch.zeros(xy.shape[:-1], device=xy.device, dtype=xy.dtype)

        height, width = bev.shape[-2:]
        pixel_size = max(float(getattr(self._config, "bev_pixel_size", 0.25)), 1e-6)
        ego_col = (width - 1) * 0.5
        row = torch.round(xy[..., 0] / pixel_size).long()
        col = torch.round(xy[..., 1] / pixel_size + ego_col).long()
        valid = (row >= 0) & (row < height) & (col >= 0) & (col < width)
        flat_idx = (row.clamp(0, height - 1) * width + col.clamp(0, width - 1)).reshape(
            batch_size, -1
        )
        flat_bev = bev.reshape(batch_size, -1)
        values = torch.gather(flat_bev, dim=1, index=flat_idx).reshape(xy.shape[:-1])
        return values * valid.to(dtype=xy.dtype)

    def _navsim_ego_corners_from_traj(self, traj: torch.Tensor) -> torch.Tensor:
        xy = traj[..., :2]
        if traj.shape[-1] >= 3:
            heading = traj[..., 2]
        else:
            delta = xy[..., 1:, :] - xy[..., :-1, :]
            heading = torch.atan2(delta[..., 1], delta[..., 0])
            heading = torch.cat([heading, heading[..., -1:]], dim=-1)
        ego_length, ego_width, rear_to_center = self._navsim_vehicle_dims()
        half_l = 0.5 * ego_length
        half_w = 0.5 * ego_width
        local = torch.tensor(
            [[half_l, half_w], [-half_l, half_w], [-half_l, -half_w], [half_l, -half_w]],
            device=traj.device,
            dtype=traj.dtype,
        )
        cos_yaw = torch.cos(heading)
        sin_yaw = torch.sin(heading)
        center = xy + rear_to_center * torch.stack([cos_yaw, sin_yaw], dim=-1)
        rot_x = cos_yaw[..., None] * local[:, 0] - sin_yaw[..., None] * local[:, 1]
        rot_y = sin_yaw[..., None] * local[:, 0] + cos_yaw[..., None] * local[:, 1]
        return torch.stack([rot_x + center[..., 0:1], rot_y + center[..., 1:2]], dim=-1)

    def _navsim_obb_corners(
        self,
        centers: torch.Tensor,
        heading: torch.Tensor,
        length: torch.Tensor,
        width: torch.Tensor,
    ) -> torch.Tensor:
        ux = torch.stack([torch.cos(heading), torch.sin(heading)], dim=-1)
        uy = torch.stack([-torch.sin(heading), torch.cos(heading)], dim=-1)
        half_l = 0.5 * length
        half_w = 0.5 * width
        return torch.stack(
            [
                centers + half_l[..., None] * ux + half_w[..., None] * uy,
                centers - half_l[..., None] * ux + half_w[..., None] * uy,
                centers - half_l[..., None] * ux - half_w[..., None] * uy,
                centers + half_l[..., None] * ux - half_w[..., None] * uy,
            ],
            dim=-2,
        )

    def _compute_navsim_fast_nc_dac_risk_targets(
        self,
        targets: Dict[str, Any],
        trajectories: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if trajectories is None or trajectories.dim() != 4 or trajectories.shape[-1] < 2:
            return None
        traj = trajectories.detach()
        device = traj.device
        dtype = traj.dtype
        batch_size, num_modes, num_steps = traj.shape[:3]
        risk_targets = torch.full(
            (batch_size, num_modes, 2),
            float("nan"),
            device=device,
            dtype=dtype,
        )

        feasible_area = targets.get("feasible_area_mask")
        if torch.is_tensor(feasible_area):
            feasible_area = feasible_area.to(device=device)
            ego_corners = self._navsim_ego_corners_from_traj(traj)
            corners_flat = ego_corners.reshape(batch_size, -1, 2)
            in_mask = self._navsim_gather_bev_values(feasible_area, corners_flat).reshape(
                batch_size, num_modes, num_steps, 4
            )
            drivable = (in_mask > 0.5).all(dim=-1).all(dim=-1).to(dtype=dtype)
            risk_targets[..., 1] = 1.0 - drivable

        obb = targets.get("future_agent_obb")
        obb_mask = targets.get("future_agent_mask")
        if torch.is_tensor(obb) and torch.is_tensor(obb_mask):
            if obb.dim() == 3:
                obb = obb.unsqueeze(0)
            if obb_mask.dim() == 2:
                obb_mask = obb_mask.unsqueeze(0)
            obb = self._align_batch_tensor(obb.to(device=device, dtype=dtype), batch_size)
            obb_mask = self._align_batch_tensor(obb_mask.to(device=device).bool(), batch_size)
            if obb is not None and obb_mask is not None and obb.numel() > 0:
                steps = min(num_steps, int(obb.shape[2]), int(obb_mask.shape[2]))
                if steps > 0:
                    traj_local = traj[:, :, :steps]
                    ego_corners = self._navsim_ego_corners_from_traj(traj_local)
                    other = obb[:, :, :steps]
                    other_corners = self._navsim_obb_corners(
                        centers=other[..., :2],
                        heading=other[..., 2],
                        length=other[..., 3],
                        width=other[..., 4],
                    )
                    intersects = self._b2d_rect_intersects(
                        ego_corners[:, :, None],
                        other_corners[:, None],
                    )
                    valid = obb_mask[:, None, :, :steps]

                    ignore = targets.get("future_agent_ignore")
                    if torch.is_tensor(ignore):
                        if ignore.dim() == 1:
                            ignore = ignore.unsqueeze(0)
                        ignore = self._align_batch_tensor(
                            ignore.to(device=device).bool(), batch_size
                        )
                        if ignore is not None:
                            valid = valid & (~ignore[:, None, :, None])

                    collision_mode = str(
                        getattr(self._config, "proformer_risk_collision_mode", "any")
                    ).lower()
                    if collision_mode == "front":
                        if traj_local.shape[-1] >= 3:
                            ego_heading = traj_local[..., 2]
                        else:
                            delta = traj_local[..., 1:, :2] - traj_local[..., :-1, :2]
                            ego_heading = torch.atan2(delta[..., 1], delta[..., 0])
                            ego_heading = torch.cat([ego_heading, ego_heading[..., -1:]], dim=-1)
                        ego_length, _, rear_to_center = self._navsim_vehicle_dims()
                        ego_center = traj_local[..., :2] + rear_to_center * torch.stack(
                            [torch.cos(ego_heading), torch.sin(ego_heading)], dim=-1
                        )
                        rel = other[:, None, :, :, :2] - ego_center[:, :, None, :, :]
                        lon = rel[..., 0] * torch.cos(ego_heading)[:, :, None, :] + rel[
                            ..., 1
                        ] * torch.sin(ego_heading)[:, :, None, :]
                        valid = valid & (lon >= (-0.25 * ego_length))

                    intersects = intersects & valid
                    is_agent = targets.get("future_agent_is_agent")
                    if torch.is_tensor(is_agent):
                        if is_agent.dim() == 1:
                            is_agent = is_agent.unsqueeze(0)
                        is_agent = self._align_batch_tensor(
                            is_agent.to(device=device).bool(), batch_size
                        )
                    if is_agent is None:
                        is_agent = torch.ones(
                            (batch_size, other.shape[1]), device=device, dtype=torch.bool
                        )
                    agent_mask = is_agent[:, None, :, None]
                    agent_hit = (intersects & agent_mask).any(dim=(2, 3))
                    static_hit = (intersects & (~agent_mask)).any(dim=(2, 3))
                    no_collision = torch.ones(
                        (batch_size, num_modes), device=device, dtype=dtype
                    )
                    no_collision = torch.where(
                        static_hit, torch.full_like(no_collision, 0.5), no_collision
                    )
                    no_collision = torch.where(
                        agent_hit, torch.zeros_like(no_collision), no_collision
                    )
                    risk_targets[..., 0] = 1.0 - no_collision

        if torch.isfinite(risk_targets).any():
            return risk_targets.clamp(0.0, 1.0)
        return None

    def _compute_fast_proposal_risk_targets(
        self,
        targets: Dict[str, Any],
        trajectories: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if torch.is_tensor(targets.get("future_agent_boxes")) and torch.is_tensor(
            targets.get("future_agent_boxes_mask")
        ):
            components = self._compute_b2d_pdm_components(targets, trajectories)
            if not torch.is_tensor(components) or components.numel() == 0:
                return None
            risk_targets = 1.0 - components[..., :2]
            return torch.where(
                torch.isfinite(risk_targets),
                risk_targets.clamp(0.0, 1.0),
                torch.full_like(risk_targets, float("nan")),
            )
        return self._compute_navsim_fast_nc_dac_risk_targets(targets, trajectories)

    def _compute_proformer_risk_aux_loss(
        self,
        targets: Dict[str, Any],
        predictions: Dict[str, torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        logs: Dict[str, torch.Tensor] = {}
        if not self.training:
            return None, logs
        if not bool(getattr(self._config, "proformer_risk_aux_enable", False)):
            return None, logs
        if float(getattr(self._config, "proformer_risk_aux_weight", 0.0)) <= 0.0:
            return None, logs

        risk_logits = predictions.get("proposal_risk_logits")
        proposal_trajs = predictions.get("proposal_risk_trajectories")
        if not torch.is_tensor(risk_logits) or not torch.is_tensor(proposal_trajs):
            return None, logs
        if risk_logits.dim() == 3:
            risk_logits = risk_logits.unsqueeze(0)
        if proposal_trajs.dim() == 4:
            proposal_trajs = proposal_trajs.unsqueeze(0)
        if risk_logits.dim() != 4 or proposal_trajs.dim() != 5:
            return None, logs

        num_rounds = min(int(risk_logits.shape[0]), int(proposal_trajs.shape[0]))
        if num_rounds <= 0:
            return None, logs

        nc_weight = max(float(getattr(self._config, "proformer_risk_aux_nc_weight", 1.0)), 0.0)
        dac_weight = max(float(getattr(self._config, "proformer_risk_aux_dac_weight", 1.0)), 0.0)
        total_losses: List[torch.Tensor] = []
        nc_losses: List[torch.Tensor] = []
        dac_losses: List[torch.Tensor] = []
        pred_means: List[torch.Tensor] = []
        target_means: List[torch.Tensor] = []

        for round_idx in range(num_rounds):
            round_targets = self._compute_fast_proposal_risk_targets(
                targets, proposal_trajs[round_idx]
            )
            if not torch.is_tensor(round_targets) or round_targets.numel() == 0:
                continue
            logits_i = risk_logits[round_idx]
            if round_targets.shape[0] != logits_i.shape[0]:
                continue
            modes = min(int(round_targets.shape[1]), int(logits_i.shape[1]))
            if modes <= 0:
                continue
            round_targets = round_targets[:, :modes]
            logits_i = logits_i[:, :modes]

            component_terms: List[torch.Tensor] = []
            component_weights: List[float] = []
            for comp_idx, comp_weight in ((0, nc_weight), (1, dac_weight)):
                if comp_weight <= 0.0:
                    continue
                target_i = round_targets[..., comp_idx]
                pred_i = logits_i[..., comp_idx]
                valid = torch.isfinite(target_i)
                if not valid.any():
                    continue
                target_valid = target_i[valid].clamp(0.0, 1.0)
                pred_valid = pred_i[valid]
                comp_loss = F.binary_cross_entropy_with_logits(pred_valid, target_valid)
                component_terms.append(comp_loss * comp_weight)
                component_weights.append(comp_weight)
                if comp_idx == 0:
                    nc_losses.append(comp_loss.detach())
                else:
                    dac_losses.append(comp_loss.detach())
                pred_means.append(torch.sigmoid(pred_valid.detach()).mean())
                target_means.append(target_valid.detach().mean())
            if component_weights:
                total_losses.append(
                    sum(component_terms) / max(float(sum(component_weights)), 1e-6)
                )

        if not total_losses:
            return None, logs

        raw_loss = torch.stack(total_losses).mean()
        logs["proformer_risk_aux_loss_raw"] = raw_loss.detach()
        if nc_losses:
            logs["proformer_risk_aux_nc_loss_raw"] = torch.stack(nc_losses).mean()
        if dac_losses:
            logs["proformer_risk_aux_dac_loss_raw"] = torch.stack(dac_losses).mean()
        if pred_means:
            logs["proformer_risk_aux_pred_mean"] = torch.stack(pred_means).mean()
        if target_means:
            logs["proformer_risk_aux_target_mean"] = torch.stack(target_means).mean()
        return raw_loss, logs

    def _compute_b2d_pdm_score_loss(
        self,
        targets: Dict[str, Any],
        trajectories: torch.Tensor,
        score_logits: Optional[torch.Tensor],
        score_components: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        comp_targets = self._compute_b2d_pdm_components(targets, trajectories)
        if comp_targets is None:
            return None
        if score_components is not None:
            if comp_targets.shape[:2] != score_components.shape[:2]:
                min_modes = min(comp_targets.shape[1], score_components.shape[1])
                if min_modes <= 0:
                    return None
                comp_targets = comp_targets[:, :min_modes]
                score_components = score_components[:, :min_modes]
                if score_logits is not None:
                    score_logits = score_logits[:, :min_modes]
            temp = max(float(getattr(self._config, "pdm_score_temp", 1.0)), 1e-6)
            target_probs = torch.softmax(comp_targets / temp, dim=1)
            pred_log_probs = F.log_softmax(score_components, dim=1)
            kl = F.kl_div(pred_log_probs, target_probs, reduction="none").sum(dim=1)
            weights = torch.tensor(
                getattr(self._config, "pdm_score_component_weights", (1.0,) * score_components.shape[-1]),
                device=score_components.device,
                dtype=score_components.dtype,
            )
            if weights.numel() != score_components.shape[-1]:
                weights = torch.ones(score_components.shape[-1], device=score_components.device, dtype=score_components.dtype)
            loss = (kl * weights.view(1, -1)).sum(dim=-1) / weights.sum().clamp_min(1e-6)
            base_loss = loss.mean() * float(getattr(self._config, "pdm_score_component_kl_weight", 1.0))
            bce_weight = float(getattr(self._config, "pdm_score_component_bce_weight", 0.0))
            if bce_weight > 0.0:
                bce_loss = self._compute_pdm_component_bce(score_components, comp_targets)
                if bce_loss is not None:
                    base_loss = base_loss + bce_weight * bce_loss
            pairwise_weight = float(getattr(self._config, "pdm_score_pairwise_weight", 1.0))
            if pairwise_weight > 0.0:
                pair_targets, pair_valid = self._aggregate_component_targets_for_pairwise(comp_targets)
                pairwise_loss = self._compute_pdm_pairwise_ranking_loss(
                    pred_scores=score_logits,
                    target_scores=pair_targets,
                    valid_mode=pair_valid,
                )
                if pairwise_loss is not None:
                    base_loss = base_loss + pairwise_weight * pairwise_loss
            return base_loss

        if score_logits is None:
            return None
        target_scores, valid_mode = self._aggregate_component_targets_for_pairwise(comp_targets)
        if target_scores is None:
            return None
        if target_scores.shape != score_logits.shape:
            min_modes = min(target_scores.shape[1], score_logits.shape[1])
            if min_modes <= 0:
                return None
            target_scores = target_scores[:, :min_modes]
            score_logits = score_logits[:, :min_modes]
            valid_mode = valid_mode[:, :min_modes] if valid_mode is not None else None
        scores = target_scores.masked_fill(~torch.isfinite(target_scores), float("-inf"))
        if valid_mode is not None:
            scores = scores.masked_fill(~valid_mode, float("-inf"))
        valid_samples = torch.isfinite(scores).any(dim=-1)
        if not valid_samples.any():
            return None
        temp = max(float(getattr(self._config, "pdm_score_temp", 1.0)), 1e-6)
        target_probs = torch.softmax(scores[valid_samples] / temp, dim=-1)
        pred_log_probs = F.log_softmax(score_logits[valid_samples], dim=-1)
        base_loss = F.kl_div(pred_log_probs, target_probs, reduction="batchmean")
        pairwise_weight = float(getattr(self._config, "pdm_score_pairwise_weight", 1.0))
        if pairwise_weight > 0.0:
            pairwise_loss = self._compute_pdm_pairwise_ranking_loss(
                pred_scores=score_logits[valid_samples],
                target_scores=scores[valid_samples],
                valid_mode=None,
            )
            if pairwise_loss is not None:
                base_loss = base_loss + pairwise_weight * pairwise_loss
        return base_loss

    def _compute_pdm_score_loss(
        self,
        targets: Dict[str, Any],
        predictions: Dict[str, torch.Tensor],
        metric_cache_path_override: Optional[str] = None,
    ) -> Optional[torch.Tensor]:
        if self._config.pdm_score_weight <= 0:
            return None
        if self._config.pdm_score_train_only and not self.training:
            return None
        poses_reg = predictions.get("poses_reg")
        score_logits = predictions.get("pdm_score")
        score_components = predictions.get("pdm_score_components")
        use_cached_pred = False
        if getattr(self._config, "pdm_score_use_cached_poses", False):
            cached_logits = predictions.get("pdm_score_cached")
            cached_components = predictions.get("pdm_score_components_cached")
            if cached_components is not None:
                score_components = cached_components
                if cached_logits is not None:
                    score_logits = cached_logits
                use_cached_pred = True
            elif cached_logits is not None:
                score_logits = cached_logits
        poses_cls = predictions.get("poses_cls")
        if score_components is None and score_logits is not None and score_logits.dim() == 3:
            score_components = score_logits
            score_logits = None
        if score_logits is None:
            score_logits = poses_cls
        use_components = bool(getattr(self._config, "pdm_score_use_components", False))
        if use_components and score_components is not None:
            comp_targets = self._get_pdm_component_targets(
                targets, device=score_components.device, dtype=score_components.dtype
            )
            if comp_targets is not None:
                use_cached_targets = bool(
                    getattr(
                        self._config,
                        "pdm_score_use_cached_targets_in_train",
                        True,
                    )
                    if self.training
                    else getattr(
                        self._config,
                        "pdm_score_use_cached_targets_in_val",
                        True,
                    )
                )
                if not use_cached_targets:
                    comp_targets = None
            if comp_targets is not None:
                matched_targets = False
                if comp_targets.ndim == 2:
                    comp_targets = comp_targets.unsqueeze(0)
                if comp_targets.shape[0] != score_components.shape[0]:
                    if comp_targets.shape[0] == 1:
                        comp_targets = comp_targets.expand(score_components.shape[0], -1, -1)
                    else:
                        return None
                cached_targets = "pdm_score_components" in targets
                cached_poses = targets.get("poses_reg")
                if (
                    cached_targets
                    and poses_reg is not None
                    and cached_poses is not None
                    and torch.is_tensor(cached_poses)
                    and not use_cached_pred
                ):
                    comp_targets = self._match_cached_component_targets(
                        poses_reg=poses_reg,
                        cached_poses=cached_poses,
                        comp_targets=comp_targets,
                        targets=targets,
                    )
                    matched_targets = True
                # Align mode dimension between cached targets and current predictions.
                if not matched_targets and comp_targets.shape[1] != score_components.shape[1]:
                    min_modes = min(comp_targets.shape[1], score_components.shape[1])
                    if min_modes == 0:
                        return None
                    comp_targets = comp_targets[:, :min_modes]
                    score_components = score_components[:, :min_modes]
                    if score_logits is not None:
                        score_logits = score_logits[:, :min_modes]
                    if poses_cls is not None:
                        poses_cls = poses_cls[:, :min_modes]
                num_modes = score_components.shape[1]
                topk = int(getattr(self._config, "pdm_score_topk", 0) or 0)
                if topk <= 0 or topk >= num_modes:
                    topk = 0
                topk_source = score_logits if score_logits is not None else poses_cls
                if topk_source is not None and topk > 0 and topk < num_modes:
                    topk_idx = torch.topk(topk_source.detach(), topk, dim=-1).indices
                    gather_idx = topk_idx.unsqueeze(-1).expand(
                        -1, -1, score_components.shape[-1]
                    )
                    score_components = torch.gather(score_components, 1, gather_idx)
                    comp_targets = torch.gather(comp_targets.to(score_components.device), 1, gather_idx)
                finite_mask = torch.isfinite(comp_targets)
                valid_comp = finite_mask.any(dim=1)
                if not valid_comp.any():
                    self._log_pdm_val_invalid(
                        targets, predictions, reason="pdm_cached_components_invalid"
                    )
                    return None
                comp_targets = comp_targets.masked_fill(~finite_mask, float("-inf"))
                temp = max(float(getattr(self._config, "pdm_score_temp", 1.0)), 1e-6)
                target_probs = torch.softmax(comp_targets / temp, dim=1)
                pred_log_probs = F.log_softmax(score_components, dim=1)
                kl = F.kl_div(pred_log_probs, target_probs, reduction="none").sum(dim=1)
                weights = torch.tensor(
                    getattr(self._config, "pdm_score_component_weights", (1.0,) * score_components.shape[-1]),
                    device=score_components.device,
                    dtype=score_components.dtype,
                )
                if weights.numel() != score_components.shape[-1]:
                    weights = torch.ones(
                        score_components.shape[-1],
                        device=score_components.device,
                        dtype=score_components.dtype,
                    )
                weights = weights.view(1, -1)
                valid_mask = valid_comp.float()
                weight_sum = (weights * valid_mask).sum(dim=-1).clamp_min(1e-6)
                loss = (kl * weights * valid_mask).sum(dim=-1) / weight_sum
                valid_samples = valid_comp.any(dim=-1)
                if not valid_samples.any():
                    self._log_pdm_val_invalid(
                        targets, predictions, reason="pdm_cached_components_no_samples"
                    )
                    return None
                kl_weight = float(
                    getattr(self._config, "pdm_score_component_kl_weight", 1.0)
                )
                base_loss = loss[valid_samples].mean() * kl_weight
                bce_weight = float(
                    getattr(self._config, "pdm_score_component_bce_weight", 0.0)
                )
                if bce_weight > 0.0:
                    bce_loss = self._compute_pdm_component_bce(
                        score_components, comp_targets
                    )
                    if bce_loss is not None:
                        base_loss = base_loss + bce_weight * bce_loss
                pairwise_weight = float(
                    getattr(self._config, "pdm_score_pairwise_weight", 1.0)
                )
                if pairwise_weight > 0.0:
                    pair_targets, pair_valid = self._aggregate_component_targets_for_pairwise(
                        comp_targets
                    )
                    pairwise_loss = self._compute_pdm_pairwise_ranking_loss(
                        pred_scores=score_logits,
                        target_scores=pair_targets,
                        valid_mode=pair_valid,
                    )
                    if pairwise_loss is not None:
                        base_loss = base_loss + pairwise_weight * pairwise_loss
                return base_loss
        if poses_reg is None or score_logits is None:
            return None
        if getattr(self._config, "pdm_score_use_offline_targets", False):
            cand_scores = targets.get("pdm_score_targets")
            if cand_scores is None:
                pdm_components = targets.get("pdm_components")
                if isinstance(pdm_components, dict):
                    cand_scores = pdm_components.get("score")
            candidates = targets.get("trajectory_candidates")
            if cand_scores is not None and candidates is not None and candidates.numel() > 0:
                cand_scores = cand_scores.to(
                    device=score_logits.device, dtype=score_logits.dtype
                )
                candidates = candidates.to(
                    device=poses_reg.device, dtype=poses_reg.dtype
                )
                if cand_scores.ndim == 1:
                    cand_scores = cand_scores.unsqueeze(0)
                if candidates.ndim == 3:
                    candidates = candidates.unsqueeze(0)
                cand_mask = targets.get("trajectory_candidates_mask")
                if cand_mask is not None:
                    cand_mask = cand_mask.to(device=poses_reg.device).bool()
                    if cand_mask.ndim == 1:
                        cand_mask = cand_mask.unsqueeze(0)
                    cand_scores = cand_scores.masked_fill(~cand_mask, 0.0)
                    valid_samples = cand_mask.any(dim=1)
                else:
                    cand_scores = torch.nan_to_num(cand_scores, nan=0.0)
                    valid_samples = torch.isfinite(cand_scores).any(dim=1)

                cand_xy = candidates[..., :2]
                pred_xy = poses_reg[..., :2]
                dist = torch.linalg.norm(
                    pred_xy[:, :, None, :, :] - cand_xy[:, None, :, :, :], dim=-1
                ).mean(dim=-1)
                if cand_mask is not None:
                    dist = dist.masked_fill(~cand_mask[:, None, :], float("inf"))
                assign_temp = max(
                    float(getattr(self._config, "pdm_score_offline_assign_temp", 1.0)),
                    1e-6,
                )
                weights = torch.softmax(-dist / assign_temp, dim=-1)
                if cand_mask is not None:
                    weights = weights * cand_mask[:, None, :].float()
                    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
                mode_scores = (weights * cand_scores[:, None, :]).sum(dim=-1)
                valid_mask = valid_samples & torch.isfinite(mode_scores).any(dim=-1)
                if valid_mask.any():
                    scores = mode_scores[valid_mask]
                    logits = score_logits[valid_mask]
                    temp = max(float(getattr(self._config, "pdm_score_temp", 1.0)), 1e-6)
                    target_probs = torch.softmax(scores / temp, dim=-1)
                    pred_log_probs = F.log_softmax(logits, dim=-1)
                    base_loss = F.kl_div(pred_log_probs, target_probs, reduction="batchmean")
                    pairwise_weight = float(
                        getattr(self._config, "pdm_score_pairwise_weight", 1.0)
                    )
                    if pairwise_weight > 0.0:
                        pairwise_loss = self._compute_pdm_pairwise_ranking_loss(
                            pred_scores=logits,
                            target_scores=scores,
                            valid_mode=None,
                        )
                        if pairwise_loss is not None:
                            base_loss = base_loss + pairwise_weight * pairwise_loss
                    return base_loss
        tokens = targets.get("token")
        if tokens is None:
            return None
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.detach().cpu().tolist()
        elif isinstance(tokens, (str, bytes)):
            tokens = [tokens]
        if not isinstance(tokens, (list, tuple)):
            self._log_pdm_val_invalid(targets, predictions, reason="pdm_token_invalid")
            return None

        num_modes = poses_reg.shape[1]
        topk = int(getattr(self._config, "pdm_score_topk", 0) or 0)
        infraction_mode_indices = torch.arange(
            num_modes, device=poses_reg.device, dtype=torch.long
        )[None, :].expand(poses_reg.shape[0], -1)
        score_components_for_loss = score_components
        if topk > 0 and topk < num_modes:
            topk_source = poses_cls if poses_cls is not None else score_logits  
            if topk_source is None:
                traj_for_score = poses_reg
                logits_for_loss = score_logits
            else:
                topk_idx = torch.topk(topk_source.detach(), topk, dim=-1).indices
                infraction_mode_indices = topk_idx
                gather_idx = topk_idx[..., None, None].expand(
                    -1, -1, poses_reg.shape[2], poses_reg.shape[3]
                )
                traj_for_score = torch.gather(poses_reg, 1, gather_idx)
                logits_for_loss = torch.gather(score_logits, 1, topk_idx)
                if score_components_for_loss is not None:
                    comp_gather_idx = topk_idx.unsqueeze(-1).expand(
                        -1, -1, score_components_for_loss.shape[-1]
                    )
                    score_components_for_loss = torch.gather(
                        score_components_for_loss, 1, comp_gather_idx
                    )
        else:
            traj_for_score = poses_reg
            logits_for_loss = score_logits

        if bool(getattr(self._config, "b2d_pdm_score_enable", False)):
            return self._compute_b2d_pdm_score_loss(
                targets=targets,
                trajectories=traj_for_score,
                score_logits=logits_for_loss,
                score_components=score_components_for_loss if use_components else None,
            )

        supervisor = self._get_pdm_supervision(
            cache_path_override=metric_cache_path_override
        )
        if supervisor is None:
            return None
        traj_np = traj_for_score.detach().cpu().numpy()
        token_list = list(tokens)
        output_infraction = self._should_output_pdm_infraction_details()
        if use_components and score_components_for_loss is not None:
            if output_infraction:
                scores_np, components_np, infraction_data = supervisor._score_batch_internal(
                    tokens=token_list,
                    trajectories=traj_np,
                    return_components=True,
                    return_infraction_data=True,
                )
                self._attach_pdm_infraction_details_to_predictions(
                    predictions=predictions,
                    infraction_data=infraction_data,
                    mode_indices=infraction_mode_indices,
                    device=score_components_for_loss.device,
                    dtype=score_components_for_loss.dtype,
                )
            else:
                scores_np, components_np, _ = supervisor._score_batch_internal(
                    tokens=token_list,
                    trajectories=traj_np,
                    return_components=True,
                    return_infraction_data=False,
                )
            self._store_online_pdm_score_cache(
                predictions=predictions,
                tokens=token_list,
                mode_indices=infraction_mode_indices,
                scores_np=scores_np,
                components_np=components_np,
                device=score_components_for_loss.device,
                dtype=score_components_for_loss.dtype,
            )
            comp_targets = torch.from_numpy(components_np).to(
                device=score_components_for_loss.device,
                dtype=score_components_for_loss.dtype,
            )
            finite_mask = torch.isfinite(comp_targets)
            valid_comp = finite_mask.any(dim=1)
            if not valid_comp.any():
                self._log_pdm_val_invalid(
                    targets, predictions, reason="pdm_online_components_invalid"
                )
                return None
            comp_targets = comp_targets.masked_fill(~finite_mask, float("-inf"))
            temp = max(float(getattr(self._config, "pdm_score_temp", 1.0)), 1e-6)
            target_probs = torch.softmax(comp_targets / temp, dim=1)
            pred_log_probs = F.log_softmax(score_components_for_loss, dim=1)
            kl = F.kl_div(pred_log_probs, target_probs, reduction="none").sum(dim=1)
            weights = torch.tensor(
                getattr(self._config, "pdm_score_component_weights", (1.0,) * score_components_for_loss.shape[-1]),
                device=score_components_for_loss.device,
                dtype=score_components_for_loss.dtype,
            )
            if weights.numel() != score_components_for_loss.shape[-1]:
                weights = torch.ones(
                    score_components_for_loss.shape[-1],
                    device=score_components_for_loss.device,
                    dtype=score_components_for_loss.dtype,
                )
            weights = weights.view(1, -1)
            valid_mask = valid_comp.float()
            weight_sum = (weights * valid_mask).sum(dim=-1).clamp_min(1e-6)
            loss = (kl * weights * valid_mask).sum(dim=-1) / weight_sum
            valid_samples = valid_comp.any(dim=-1)
            if not valid_samples.any():
                self._log_pdm_val_invalid(
                    targets, predictions, reason="pdm_online_components_no_samples"
                )
                return None
            kl_weight = float(
                getattr(self._config, "pdm_score_component_kl_weight", 1.0)
            )
            base_loss = loss[valid_samples].mean() * kl_weight
            bce_weight = float(
                getattr(self._config, "pdm_score_component_bce_weight", 0.0)
            )
            if bce_weight > 0.0:
                bce_loss = self._compute_pdm_component_bce(
                    score_components_for_loss, comp_targets
                )
                if bce_loss is not None:
                    base_loss = base_loss + bce_weight * bce_loss
            pairwise_weight = float(
                getattr(self._config, "pdm_score_pairwise_weight", 1.0)
            )
            if pairwise_weight > 0.0:
                pair_targets, pair_valid = self._aggregate_component_targets_for_pairwise(
                    comp_targets
                )
                pairwise_loss = self._compute_pdm_pairwise_ranking_loss(
                    pred_scores=logits_for_loss,
                    target_scores=pair_targets,
                    valid_mode=pair_valid,
                )
                if pairwise_loss is not None:
                    base_loss = base_loss + pairwise_weight * pairwise_loss
            return base_loss

        if output_infraction:
            scores_np, infraction_data = supervisor.score_batch_with_infraction_data(
                token_list, traj_np
            )
            self._attach_pdm_infraction_details_to_predictions(
                predictions=predictions,
                infraction_data=infraction_data,
                mode_indices=infraction_mode_indices,
                device=logits_for_loss.device,
                dtype=logits_for_loss.dtype,
            )
        else:
            scores_np = supervisor.score_batch(token_list, traj_np)
        self._store_online_pdm_score_cache(
            predictions=predictions,
            tokens=token_list,
            mode_indices=infraction_mode_indices,
            scores_np=scores_np,
            device=logits_for_loss.device,
            dtype=logits_for_loss.dtype,
        )
        scores = torch.from_numpy(scores_np).to(
            device=logits_for_loss.device, dtype=logits_for_loss.dtype
        )
        scores = scores.masked_fill(~torch.isfinite(scores), float("-inf"))     
        valid_mask = torch.isfinite(scores).any(dim=-1)
        if not valid_mask.any():
            self._log_pdm_val_invalid(
                targets, predictions, reason="pdm_online_scores_invalid"
            )
            return None
        scores = scores[valid_mask]
        logits = logits_for_loss[valid_mask]
        temp = max(float(getattr(self._config, "pdm_score_temp", 1.0)), 1e-6)
        target_probs = torch.softmax(scores / temp, dim=-1)
        pred_log_probs = F.log_softmax(logits, dim=-1)
        base_loss = F.kl_div(pred_log_probs, target_probs, reduction="batchmean")
        pairwise_weight = float(
            getattr(self._config, "pdm_score_pairwise_weight", 1.0)
        )
        if pairwise_weight > 0.0:
            pairwise_loss = self._compute_pdm_pairwise_ranking_loss(
                pred_scores=logits,
                target_scores=scores,
                valid_mode=None,
            )
            if pairwise_loss is not None:
                base_loss = base_loss + pairwise_weight * pairwise_loss
            return base_loss

    def _store_online_pdm_score_cache(
        self,
        predictions: Dict[str, Any],
        tokens: List[Any],
        mode_indices: torch.Tensor,
        scores_np: Optional[Any],
        device: torch.device,
        dtype: torch.dtype,
        components_np: Optional[Any] = None,
    ) -> None:
        if scores_np is None or not torch.is_tensor(mode_indices):
            return
        scores = torch.as_tensor(scores_np, device=device, dtype=dtype)
        if scores.dim() != 2 or scores.shape != mode_indices.shape:
            return
        cache: Dict[str, Any] = {
            "tokens": tuple(str(token) for token in tokens),
            "mode_indices": mode_indices.detach().to(device=device, dtype=torch.long),
            "scores": scores.detach(),
        }
        if components_np is not None:
            components = torch.as_tensor(components_np, device=device, dtype=dtype)
            if components.dim() == 3 and components.shape[:2] == scores.shape:
                cache["components"] = components.detach()
        predictions["_pdm_online_score_cache"] = cache

    def _get_cached_online_pdm_scores(
        self,
        predictions: Dict[str, Any],
        tokens: List[Any],
        requested_mode_indices: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        cache = predictions.get("_pdm_online_score_cache")
        if not isinstance(cache, dict):
            return None
        if tuple(str(token) for token in tokens) != cache.get("tokens"):
            return None
        scores = cache.get("scores")
        cached_indices = cache.get("mode_indices")
        if not torch.is_tensor(scores) or not torch.is_tensor(cached_indices):
            return None
        if not torch.is_tensor(requested_mode_indices):
            return None
        if scores.dim() != 2 or cached_indices.dim() != 2 or requested_mode_indices.dim() != 2:
            return None
        if scores.shape != cached_indices.shape:
            return None
        if cached_indices.shape[0] != requested_mode_indices.shape[0]:
            return None

        requested = requested_mode_indices.detach().to(
            device=cached_indices.device, dtype=torch.long
        )
        cached_indices = cached_indices.to(dtype=torch.long)
        scores = scores.to(device=device, dtype=dtype)

        if cached_indices.shape == requested.shape and torch.equal(cached_indices, requested):
            return scores

        match = requested[..., None].eq(cached_indices[:, None, :])
        found = match.any(dim=-1)
        if not bool(found.all().item()):
            return None
        gather_pos = match.to(dtype=torch.long).argmax(dim=-1).to(device=scores.device)
        return torch.gather(scores, 1, gather_pos)

    def _aggregate_component_logits_for_policy(self, component_logits: torch.Tensor) -> torch.Tensor:
        if component_logits.dim() != 3:
            return component_logits
        weights = torch.tensor(
            getattr(
                self._config,
                "pdm_score_component_weights",
                (1.0,) * component_logits.shape[-1],
            ),
            device=component_logits.device,
            dtype=component_logits.dtype,
        )
        if weights.numel() != component_logits.shape[-1]:
            weights = torch.ones(
                component_logits.shape[-1],
                device=component_logits.device,
                dtype=component_logits.dtype,
            )
        weights = weights.view(1, 1, -1)
        denom = weights.sum(dim=-1).clamp_min(1e-6)
        if bool(getattr(self._config, "pdm_score_use_logsigmoid_aggregate", True)):
            return (F.logsigmoid(component_logits) * weights).sum(dim=-1) / denom
        return (component_logits * weights).sum(dim=-1) / denom

    def _get_grpo_policy_logits(
        self, predictions: Dict[str, torch.Tensor]
    ) -> Optional[torch.Tensor]:
        source = str(getattr(self._config, "grpo_policy_source", "pdm_score")).lower()
        source_map = {
            "score": "pdm_score",
            "pdm": "pdm_score",
            "pdm_score": "pdm_score",
            "cls": "poses_cls",
            "poses_cls": "poses_cls",
            "generalist": "pdm_score_generalist",
            "pdm_score_generalist": "pdm_score_generalist",
            "specialist": "pdm_score_specialist",
            "pdm_score_specialist": "pdm_score_specialist",
        }
        primary = source_map.get(source, source)
        candidate_keys = [primary]
        for fallback in ("pdm_score", "poses_cls"):
            if fallback not in candidate_keys:
                candidate_keys.append(fallback)

        for key in candidate_keys:
            logits = predictions.get(key)
            if not torch.is_tensor(logits):
                continue
            if logits.dim() == 3:
                logits = self._aggregate_component_logits_for_policy(logits)
            if logits.dim() == 2:
                return logits
        return None

    def _compute_mode_grpo_loss(
        self,
        targets: Dict[str, Any],
        predictions: Dict[str, torch.Tensor],
        metric_cache_path_override: Optional[str] = None,
    ) -> Tuple[Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        logs: Dict[str, torch.Tensor] = {}
        if not self.training:
            return None, logs
        if not bool(getattr(self._config, "grpo_enable", False)):
            return None, logs
        if float(getattr(self._config, "grpo_weight", 0.0)) <= 0.0:
            return None, logs

        poses_reg = predictions.get("poses_reg")
        if not torch.is_tensor(poses_reg) or poses_reg.dim() != 4:
            return None, logs
        policy_logits = self._get_grpo_policy_logits(predictions)
        if not torch.is_tensor(policy_logits) or policy_logits.dim() != 2:
            return None, logs

        tokens = targets.get("token")
        if tokens is None:
            return None, logs
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.detach().cpu().tolist()
        elif isinstance(tokens, (str, bytes)):
            tokens = [tokens]
        if not isinstance(tokens, (list, tuple)):
            return None, logs

        batch_size = poses_reg.shape[0]
        if len(tokens) != batch_size or policy_logits.shape[0] != batch_size:
            return None, logs

        num_modes = min(int(poses_reg.shape[1]), int(policy_logits.shape[1]))
        if num_modes < 2:
            return None, logs
        poses_reg = poses_reg[:, :num_modes]
        policy_logits = policy_logits[:, :num_modes]

        topk = int(getattr(self._config, "grpo_reward_topk", 0) or 0)
        if topk > 0:
            topk = min(topk, num_modes)
        if topk == 1:
            return None, logs
        mode_indices = torch.arange(
            num_modes, device=poses_reg.device, dtype=torch.long
        )[None, :].expand(batch_size, -1)
        if topk > 1 and topk < num_modes:
            topk_idx = torch.topk(policy_logits.detach(), topk, dim=-1).indices
            mode_indices = topk_idx
            gather_traj_idx = topk_idx[..., None, None].expand(
                -1, -1, poses_reg.shape[2], poses_reg.shape[3]
            )
            poses_reg = torch.gather(poses_reg, 1, gather_traj_idx)
            policy_logits = torch.gather(policy_logits, 1, topk_idx)

        token_list = list(tokens)
        cached_rewards = self._get_cached_online_pdm_scores(
            predictions=predictions,
            tokens=token_list,
            requested_mode_indices=mode_indices,
            device=policy_logits.device,
            dtype=policy_logits.dtype,
        )
        if cached_rewards is not None:
            rewards = cached_rewards
            logs["grpo_reward_cache_hit"] = torch.ones(
                (), device=policy_logits.device, dtype=policy_logits.dtype
            )
        else:
            logs["grpo_reward_cache_hit"] = torch.zeros(
                (), device=policy_logits.device, dtype=policy_logits.dtype
            )
            supervisor = self._get_pdm_supervision(
                cache_path_override=metric_cache_path_override
            )
            if supervisor is None:
                return None, logs

            traj_np = poses_reg.detach().cpu().numpy()
            with torch.no_grad():
                rewards_np = supervisor.score_batch(token_list, traj_np)
            rewards = torch.from_numpy(rewards_np).to(
                device=policy_logits.device, dtype=policy_logits.dtype
            )
        if rewards.dim() == 1:
            rewards = rewards.unsqueeze(0)
        if rewards.shape != policy_logits.shape:
            min_batch = min(int(rewards.shape[0]), int(policy_logits.shape[0]))
            min_modes = min(int(rewards.shape[1]), int(policy_logits.shape[1]))
            if min_batch <= 0 or min_modes < 2:
                return None, logs
            rewards = rewards[:min_batch, :min_modes]
            policy_logits = policy_logits[:min_batch, :min_modes]
            mode_indices = mode_indices[:min_batch, :min_modes]

        finite = torch.isfinite(rewards)
        valid_counts = finite.sum(dim=-1)
        safe_rewards = rewards.masked_fill(~finite, 0.0)
        denom = valid_counts.to(dtype=policy_logits.dtype).clamp_min(1.0)
        reward_mean = safe_rewards.sum(dim=-1) / denom
        centered = (safe_rewards - reward_mean[:, None]).masked_fill(~finite, 0.0)
        reward_std = torch.sqrt((centered.pow(2).sum(dim=-1) / denom).clamp_min(0.0))
        min_std = max(float(getattr(self._config, "grpo_min_reward_std", 1e-3)), 1e-8)
        valid_samples = (valid_counts >= 2) & (reward_std >= min_std)
        logs["grpo_valid_sample_rate"] = valid_samples.float().mean().detach()
        if not valid_samples.any():
            return None, logs

        finite = finite[valid_samples]
        rewards_valid = safe_rewards[valid_samples]
        logits_valid = policy_logits[valid_samples]
        mean_valid = reward_mean[valid_samples]
        std_valid = reward_std[valid_samples].clamp_min(min_std)
        advantages = (rewards_valid - mean_valid[:, None]) / std_valid[:, None]
        advantages = advantages.masked_fill(~finite, 0.0)
        if bool(getattr(self._config, "grpo_detach_reward", True)):
            advantages = advantages.detach()

        temp = max(float(getattr(self._config, "grpo_temp", 1.0)), 1e-6)
        masked_logits = logits_valid.masked_fill(~finite, -1.0e4)
        log_probs = F.log_softmax(masked_logits / temp, dim=-1)
        old_log_probs = log_probs.detach()
        ratio = torch.exp(log_probs - old_log_probs)
        clip_eps = max(float(getattr(self._config, "grpo_clip_eps", 0.2)), 0.0)
        clipped_ratio = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
        objective = torch.minimum(ratio * advantages, clipped_ratio * advantages)
        objective = objective.masked_select(finite)
        if objective.numel() == 0:
            return None, logs

        policy_loss = -objective.mean()
        probs = torch.softmax(masked_logits / temp, dim=-1)
        entropy = -(probs * log_probs).masked_fill(~finite, 0.0).sum(dim=-1).mean()
        entropy_weight = max(float(getattr(self._config, "grpo_entropy_weight", 0.0)), 0.0)
        loss = policy_loss - entropy_weight * entropy

        with torch.no_grad():
            reward_for_best = rewards_valid.masked_fill(~finite, float("-inf"))
            best_reward = reward_for_best.max(dim=-1).values
            pred_idx = masked_logits.argmax(dim=-1)
            selected_reward = rewards_valid.gather(1, pred_idx[:, None]).squeeze(1)
            finite_rewards = rewards_valid.masked_select(finite)
            finite_adv = advantages.masked_select(finite)
            logs["grpo_reward_mean"] = finite_rewards.mean().detach()
            logs["grpo_reward_std"] = std_valid.mean().detach()
            logs["grpo_best_reward"] = best_reward.mean().detach()
            logs["grpo_selected_reward"] = selected_reward.mean().detach()
            logs["grpo_reward_regret"] = (best_reward - selected_reward).mean().detach()
            logs["grpo_adv_abs_mean"] = finite_adv.abs().mean().detach()
            logs["grpo_entropy"] = entropy.detach()
            logs["grpo_policy_loss_raw"] = policy_loss.detach()
            logs["grpo_scored_modes"] = torch.as_tensor(
                float(policy_logits.shape[1]), device=policy_logits.device
            )
            logs["grpo_original_mode_mean"] = mode_indices.float().mean().detach()
        return loss, logs

    def _maybe_add_grpo_loss(
        self,
        loss_dict: Dict[str, torch.Tensor],
        targets: Dict[str, Any],
        predictions: Dict[str, torch.Tensor],
        metric_cache_path_override: Optional[str] = None,
    ) -> None:
        grpo_loss, grpo_logs = self._compute_mode_grpo_loss(
            targets=targets,
            predictions=predictions,
            metric_cache_path_override=metric_cache_path_override,
        )
        for key, value in grpo_logs.items():
            loss_dict[key] = value
        if grpo_loss is None:
            return
        weight = float(getattr(self._config, "grpo_weight", 0.0))
        weighted_grpo = weight * grpo_loss
        loss_dict["grpo_loss"] = weighted_grpo
        loss_dict["grpo_loss_raw"] = grpo_loss.detach()
        loss_dict["loss"] = loss_dict["loss"] + weighted_grpo

    @staticmethod
    def _filter_batch_dict_by_mask(
        data: Dict[str, Any],
        batch_mask: torch.Tensor,
    ) -> Dict[str, Any]:
        if not torch.is_tensor(batch_mask):
            return dict(data)
        mask = batch_mask.detach().bool().view(-1)
        bs = int(mask.shape[0])
        keep_idx_cpu = torch.nonzero(mask, as_tuple=False).view(-1).cpu().tolist()
        if len(keep_idx_cpu) == 0:
            return {}

        out: Dict[str, Any] = {}
        for key, value in data.items():
            if torch.is_tensor(value):
                if value.dim() > 0 and value.shape[0] == bs:
                    keep_idx = torch.tensor(keep_idx_cpu, device=value.device, dtype=torch.long)
                    out[key] = value.index_select(0, keep_idx)
                elif value.dim() > 0 and value.shape[0] == 1 and bs > 1:
                    expanded = value.expand(bs, *value.shape[1:])
                    keep_idx = torch.tensor(
                        keep_idx_cpu, device=expanded.device, dtype=torch.long
                    )
                    out[key] = expanded.index_select(0, keep_idx)
                else:
                    out[key] = value
            elif isinstance(value, (list, tuple)) and len(value) == bs:
                out[key] = [value[i] for i in keep_idx_cpu]
            else:
                out[key] = value
        return out

    def _build_score_bev_hardcase_target_risk(
        self,
        targets: Dict[str, Any],
        predictions: Dict[str, torch.Tensor],
        metric_cache_path_override: Optional[str] = None,
    ) -> Tuple[Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        logs: Dict[str, torch.Tensor] = {}
        use_score_risk = bool(
            getattr(self._config, "hardcase_score_residual_use_score_risk", True)
        )
        use_bev_risk = bool(
            getattr(self._config, "hardcase_score_residual_use_bev_risk", True)
        )
        if not use_score_risk and not use_bev_risk:
            return None, logs

        score_generalist = predictions.get("pdm_score_generalist")
        if not torch.is_tensor(score_generalist):
            score_generalist = predictions.get("pdm_score")
        poses_cls = predictions.get("poses_cls")
        if not torch.is_tensor(score_generalist) and not torch.is_tensor(poses_cls):
            return None, logs

        ref = score_generalist if torch.is_tensor(score_generalist) else poses_cls
        assert ref is not None
        if ref.dim() != 2:
            return None, logs
        bs, num_modes = ref.shape
        if bs <= 0 or num_modes <= 0:
            return None, logs

        select_predictions = dict(predictions)
        if torch.is_tensor(score_generalist):
            select_predictions["pdm_score"] = score_generalist
        selected_idx = self._select_mode_indices_for_gate(
            predictions=select_predictions,
            batch_size=bs,
            num_modes=num_modes,
            device=ref.device,
        ).clamp(0, max(num_modes - 1, 0))
        batch_idx = torch.arange(bs, device=ref.device)

        score_risk: Optional[torch.Tensor] = None
        if use_score_risk:
            selected_score: Optional[torch.Tensor] = None
            use_oracle_score = bool(
                getattr(self._config, "hardcase_score_residual_use_oracle_score_risk", False)
            )
            if use_oracle_score:
                poses_reg = predictions.get("poses_reg")
                tokens = targets.get("token")
                if (
                    torch.is_tensor(poses_reg)
                    and poses_reg.dim() == 4
                    and poses_reg.shape[0] == bs
                    and isinstance(tokens, (list, tuple, torch.Tensor, str, bytes))
                ):
                    token_list: Optional[List[Any]] = None
                    if isinstance(tokens, torch.Tensor):
                        token_list = tokens.detach().cpu().tolist()
                    elif isinstance(tokens, (str, bytes)):
                        token_list = [tokens]
                    elif isinstance(tokens, (list, tuple)):
                        token_list = list(tokens)
                    if token_list is not None and len(token_list) == bs:
                        gather_idx = selected_idx[..., None, None, None].expand(
                            -1, 1, poses_reg.shape[2], poses_reg.shape[3]
                        )
                        selected_traj = torch.gather(poses_reg, 1, gather_idx).squeeze(1)
                        supervisor = self._get_pdm_supervision(
                            cache_path_override=metric_cache_path_override
                        )
                        if supervisor is not None:
                            traj_np = (
                                selected_traj[:, None, :, :]
                                .detach()
                                .cpu()
                                .numpy()
                                .astype("float32", copy=False)
                            )
                            try:
                                scores_np = supervisor.score_batch(token_list, traj_np)
                                scores_oracle = torch.from_numpy(scores_np).to(
                                    device=ref.device,
                                    dtype=torch.float32,
                                )
                                if scores_oracle.dim() == 2 and scores_oracle.shape[1] >= 1:
                                    selected_score = scores_oracle[:, 0].clamp(0.0, 1.0)
                                    logs["hardcase_score_residual_oracle_score"] = (
                                        selected_score.detach().mean()
                                    )
                            except Exception:
                                selected_score = None

            if selected_score is None:
                if torch.is_tensor(score_generalist) and score_generalist.dim() == 2:
                    logits = torch.nan_to_num(
                        score_generalist.float(), nan=0.0, posinf=0.0, neginf=0.0
                    )
                    selected_score = torch.sigmoid(logits[batch_idx, selected_idx]).clamp(
                        0.0, 1.0
                    )
                elif torch.is_tensor(poses_cls) and poses_cls.dim() == 2:
                    selected_score = torch.softmax(poses_cls.float(), dim=-1)[
                        batch_idx, selected_idx
                    ].clamp(0.0, 1.0)
            if selected_score is not None:
                score_thresh = float(
                    getattr(self._config, "hardcase_score_residual_score_thresh", 0.55)
                )
                score_temp = max(
                    float(getattr(self._config, "hardcase_score_residual_score_temp", 0.10)),
                    1e-6,
                )
                score_risk = torch.sigmoid((score_thresh - selected_score.detach()) / score_temp)
                logs["hardcase_score_residual_selected_score"] = selected_score.detach().mean()
                logs["hardcase_score_residual_score_risk"] = score_risk.detach().mean()

        bev_risk: Optional[torch.Tensor] = None
        if use_bev_risk:
            pred_bev = predictions.get("bev_semantic_map")
            target_bev = targets.get("bev_semantic_map")
            if torch.is_tensor(pred_bev) and torch.is_tensor(target_bev) and pred_bev.dim() == 4:
                bev_labels = self._prepare_bev_target_labels_for_gate(
                    pred_logits=pred_bev,
                    target_labels=target_bev,
                )
                if bev_labels is not None:
                    if bool(getattr(self._config, "bev_loss_logit_guard_enable", False)):
                        safe_pred_bev = torch.nan_to_num(
                            pred_bev.detach().float(), nan=0.0, posinf=30.0, neginf=-30.0
                        ).clamp(-30.0, 30.0)
                    else:
                        safe_pred_bev = pred_bev.detach()
                    bev_ce = F.cross_entropy(safe_pred_bev, bev_labels, reduction="none")
                    bev_per_sample = bev_ce.view(bev_ce.shape[0], -1).mean(dim=-1)
                    bev_mean = bev_per_sample.mean()
                    bev_std = bev_per_sample.std(unbiased=False).clamp_min(1e-6)
                    bev_z = (bev_per_sample - bev_mean) / bev_std
                    bev_z_thresh = float(
                        getattr(self._config, "hardcase_score_residual_bev_z_thresh", 0.5)
                    )
                    bev_temp = max(
                        float(getattr(self._config, "hardcase_score_residual_bev_temp", 0.5)),
                        1e-6,
                    )
                    bev_risk = torch.sigmoid((bev_z - bev_z_thresh) / bev_temp)
                    logs["hardcase_score_residual_bev_ce"] = bev_per_sample.detach().mean()
                    logs["hardcase_score_residual_bev_risk"] = bev_risk.detach().mean()

        target_risk: Optional[torch.Tensor] = None
        if score_risk is not None and bev_risk is not None:
            combine_mode = str(
                getattr(self._config, "hardcase_score_residual_combine", "max")
            ).lower()
            if combine_mode == "mean":
                target_risk = 0.5 * (score_risk + bev_risk)
            else:
                target_risk = torch.maximum(score_risk, bev_risk)
        elif score_risk is not None:
            target_risk = score_risk
        elif bev_risk is not None:
            target_risk = bev_risk

        if target_risk is None:
            return None, logs
        return target_risk.clamp(0.0, 1.0), logs

    def _compute_adapter_independent_specialist_loss(
        self,
        targets: Dict[str, Any],
        predictions: Dict[str, torch.Tensor],
        adapter_scores: Optional[torch.Tensor],
        metric_cache_path_override: Optional[str] = None,
    ) -> Optional[torch.Tensor]:
        if not torch.is_tensor(adapter_scores):
            return None
        if adapter_scores.dim() not in (3, 4):
            return None

        adapter_losses: List[torch.Tensor] = []
        num_adapters = int(adapter_scores.shape[2])
        for adapter_idx in range(num_adapters):
            adapter_predictions = dict(predictions)
            if adapter_scores.dim() == 4:
                adapter_component_scores = adapter_scores[:, :, adapter_idx, :]
                adapter_predictions["pdm_score_components"] = adapter_component_scores
                adapter_predictions["pdm_score"] = adapter_component_scores
            else:
                adapter_predictions["pdm_score_components"] = None
                adapter_predictions["pdm_score"] = adapter_scores[:, :, adapter_idx]
            adapter_loss = self._compute_pdm_score_loss(
                targets,
                adapter_predictions,
                metric_cache_path_override=metric_cache_path_override,
            )
            if adapter_loss is not None:
                adapter_losses.append(adapter_loss)

        if len(adapter_losses) == 0:
            return None
        return torch.stack(adapter_losses).mean()

    def _compute_pdm_specialist_score_loss(
        self,
        targets: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
        metric_cache_path_override: Optional[str] = None,
    ) -> Tuple[Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        logs: Dict[str, torch.Tensor] = {}
        weight = float(getattr(self._config, "hardcase_score_residual_loss_weight", 0.0))
        if weight <= 0.0:
            return None, logs
        specialist_score = predictions.get("pdm_score_specialist")
        if not torch.is_tensor(specialist_score):
            return None, logs
        spec_predictions = dict(predictions)
        spec_predictions["pdm_score"] = specialist_score
        specialist_components = predictions.get("pdm_score_components_specialist")
        if torch.is_tensor(specialist_components):
            spec_predictions["pdm_score_components"] = specialist_components

        selected_idx_for_unc: Optional[torch.Tensor] = None
        if specialist_score.dim() == 2:
            bs_unc = specialist_score.shape[0]
            num_modes_unc = specialist_score.shape[1]
            select_predictions = dict(predictions)
            score_for_select = predictions.get("pdm_score_generalist")
            if not torch.is_tensor(score_for_select):
                score_for_select = predictions.get("pdm_score")
            if (
                torch.is_tensor(score_for_select)
                and score_for_select.dim() == 2
                and score_for_select.shape[0] == bs_unc
                and score_for_select.shape[1] == num_modes_unc
            ):
                select_predictions["pdm_score"] = score_for_select
            selected_idx_for_unc = self._select_mode_indices_for_gate(
                predictions=select_predictions,
                batch_size=bs_unc,
                num_modes=num_modes_unc,
                device=specialist_score.device,
            ).clamp(0, max(num_modes_unc - 1, 0))

        unc_tensor: Optional[torch.Tensor] = None
        selected_unc_tensor: Optional[torch.Tensor] = None
        specialist_uncertainty = predictions.get("pdm_score_specialist_uncertainty")
        if torch.is_tensor(specialist_uncertainty):
            unc_tensor_tmp = torch.nan_to_num(
                specialist_uncertainty.float(), nan=0.0, posinf=0.0, neginf=0.0
            )
            if unc_tensor_tmp.dim() == 3:
                unc_tensor_tmp = unc_tensor_tmp.mean(dim=-1)
            if (
                unc_tensor_tmp.dim() == 2
                and specialist_score.dim() == 2
                and unc_tensor_tmp.shape == specialist_score.shape
            ):
                unc_tensor = unc_tensor_tmp
                logs["hardcase_score_residual_adapter_uncertainty_all"] = (
                    unc_tensor.mean().detach()
                )
                if selected_idx_for_unc is not None:
                    batch_idx = torch.arange(
                        unc_tensor.shape[0], device=unc_tensor.device
                    )
                    selected_unc_tensor = unc_tensor[batch_idx, selected_idx_for_unc]
                    logs["hardcase_score_residual_adapter_uncertainty"] = (
                        selected_unc_tensor.mean().detach()
                    )

        total: Optional[torch.Tensor] = None
        hard_mask: Optional[torch.Tensor] = None
        target_risk: Optional[torch.Tensor] = None
        hard_filter_enable = bool(
            getattr(self._config, "hardcase_score_residual_hard_filter_enable", True)
        )
        if hard_filter_enable:
            target_risk, risk_logs = self._build_score_bev_hardcase_target_risk(
                targets=targets,
                predictions=predictions,
                metric_cache_path_override=metric_cache_path_override,
            )
            logs.update(risk_logs)
            if target_risk is None:
                return None, logs
            hard_thresh = float(
                getattr(self._config, "hardcase_score_residual_hard_target_thresh", 0.6)
            )
            hard_mask = target_risk >= hard_thresh
            logs["hardcase_score_residual_target_risk"] = target_risk.detach().mean()
            logs["hardcase_score_residual_hard_rate"] = hard_mask.float().mean()

        targets_for_loss: Dict[str, Any] = targets
        preds_for_loss: Dict[str, Any] = spec_predictions
        if hard_filter_enable and hard_mask is not None:
            if hard_mask.any():
                targets_for_loss = self._filter_batch_dict_by_mask(targets, hard_mask)
                preds_for_loss = self._filter_batch_dict_by_mask(spec_predictions, hard_mask)
            else:
                targets_for_loss = {}
                preds_for_loss = {}

        spec_loss: Optional[torch.Tensor] = None
        if isinstance(targets_for_loss, dict) and len(targets_for_loss) > 0:
            spec_loss = self._compute_pdm_score_loss(
                targets_for_loss,
                preds_for_loss,
                metric_cache_path_override=metric_cache_path_override,
            )
        if spec_loss is not None:
            total = weight * spec_loss
            logs["pdm_score_specialist_loss_raw"] = spec_loss.detach()

        adapter_loss_enable = bool(
            getattr(
                self._config,
                "hardcase_score_residual_adapter_independent_loss_enable",
                True,
            )
        )
        adapter_loss_weight = float(
            getattr(
                self._config,
                "hardcase_score_residual_adapter_independent_loss_weight",
                0.5,
            )
        )
        if adapter_loss_enable and adapter_loss_weight > 0.0:
            adapter_scores = predictions.get("pdm_score_specialist_adapter_scores")
            adapter_scores_for_loss = adapter_scores
            if hard_filter_enable and hard_mask is not None:
                if hard_mask.any():
                    adapter_scores_for_loss = self._filter_batch_dict_by_mask(
                        {"adapter_scores": adapter_scores},
                        hard_mask,
                    ).get("adapter_scores")
                else:
                    adapter_scores_for_loss = None
            adapter_loss = None
            if isinstance(targets_for_loss, dict) and len(targets_for_loss) > 0:
                adapter_loss = self._compute_adapter_independent_specialist_loss(
                    targets=targets_for_loss,
                    predictions=preds_for_loss,
                    adapter_scores=adapter_scores_for_loss,
                    metric_cache_path_override=metric_cache_path_override,
                )
            if adapter_loss is not None:
                weighted_adapter_loss = weight * adapter_loss_weight * adapter_loss
                total = weighted_adapter_loss if total is None else (total + weighted_adapter_loss)
                logs["pdm_score_specialist_adapter_loss_raw"] = adapter_loss.detach()

        score_generalist = predictions.get("pdm_score_generalist")
        if not torch.is_tensor(score_generalist):
            score_generalist = predictions.get("pdm_score")
        selected_idx_for_easy = selected_idx_for_unc
        selected_score_for_easy: Optional[torch.Tensor] = None
        if (
            torch.is_tensor(score_generalist)
            and score_generalist.dim() == 2
            and specialist_score.dim() == 2
            and score_generalist.shape == specialist_score.shape
        ):
            bs_easy, num_modes_easy = specialist_score.shape
            if selected_idx_for_easy is None:
                select_predictions = dict(predictions)
                select_predictions["pdm_score"] = score_generalist
                selected_idx_for_easy = self._select_mode_indices_for_gate(
                    predictions=select_predictions,
                    batch_size=bs_easy,
                    num_modes=num_modes_easy,
                    device=specialist_score.device,
                ).clamp(0, max(num_modes_easy - 1, 0))
            if selected_idx_for_easy is not None:
                batch_idx_easy = torch.arange(bs_easy, device=specialist_score.device)
                selected_score_for_easy = torch.sigmoid(
                    torch.nan_to_num(
                        score_generalist.float(), nan=0.0, posinf=0.0, neginf=0.0
                    )[batch_idx_easy, selected_idx_for_easy]
                ).clamp(0.0, 1.0)

        easy_cons_weight = float(
            getattr(
                self._config, "hardcase_score_residual_easy_consistency_weight", 0.0
            )
        )
        easy_dist_weight = float(
            getattr(
                self._config, "hardcase_score_residual_easy_distribution_weight", 0.0
            )
        )
        easy_unc_weight = float(
            getattr(
                self._config, "hardcase_score_residual_easy_uncertainty_weight", 0.0
            )
        )
        if (
            (easy_cons_weight > 0.0 or easy_dist_weight > 0.0 or easy_unc_weight > 0.0)
            and hard_filter_enable
            and target_risk is not None
            and specialist_score.dim() == 2
        ):
            easy_thresh = float(
                getattr(
                    self._config, "hardcase_score_residual_easy_target_risk_thresh", 0.30
                )
            )
            easy_mask = (target_risk <= easy_thresh).bool()
            if hard_mask is not None and hard_mask.shape == easy_mask.shape:
                easy_mask = easy_mask & (~hard_mask.bool())
            if bool(
                getattr(
                    self._config,
                    "hardcase_score_residual_easy_require_high_score",
                    False,
                )
            ):
                if selected_score_for_easy is not None:
                    easy_score_thresh = float(
                        getattr(
                            self._config,
                            "hardcase_score_residual_easy_selected_score_thresh",
                            0.90,
                        )
                    )
                    easy_mask = easy_mask & (selected_score_for_easy >= easy_score_thresh)
                else:
                    easy_mask = easy_mask & False
            logs["hardcase_score_residual_easy_rate"] = easy_mask.float().mean()

            if easy_mask.any():
                if (
                    easy_cons_weight > 0.0
                    and torch.is_tensor(score_generalist)
                    and score_generalist.dim() == 2
                    and score_generalist.shape == specialist_score.shape
                    and selected_idx_for_easy is not None
                ):
                    batch_idx_easy = torch.arange(
                        specialist_score.shape[0], device=specialist_score.device
                    )
                    spec_sel = specialist_score[batch_idx_easy, selected_idx_for_easy]
                    gen_sel = score_generalist[batch_idx_easy, selected_idx_for_easy].detach()
                    easy_cons_raw = F.smooth_l1_loss(spec_sel[easy_mask], gen_sel[easy_mask])
                    easy_cons_loss = easy_cons_weight * easy_cons_raw
                    total = easy_cons_loss if total is None else (total + easy_cons_loss)
                    logs["hardcase_score_residual_easy_consistency_raw"] = (
                        easy_cons_raw.detach()
                    )

                if (
                    easy_dist_weight > 0.0
                    and torch.is_tensor(score_generalist)
                    and score_generalist.dim() == 2
                    and score_generalist.shape == specialist_score.shape
                ):
                    spec_easy = specialist_score[easy_mask]
                    gen_easy = score_generalist[easy_mask].detach()
                    spec_log_probs = F.log_softmax(spec_easy, dim=-1)
                    gen_probs = F.softmax(gen_easy, dim=-1)
                    easy_dist_raw = F.kl_div(
                        spec_log_probs,
                        gen_probs,
                        reduction="batchmean",
                    )
                    easy_dist_loss = easy_dist_weight * easy_dist_raw
                    total = easy_dist_loss if total is None else (total + easy_dist_loss)
                    logs["hardcase_score_residual_easy_distribution_raw"] = (
                        easy_dist_raw.detach()
                    )

                if (
                    easy_unc_weight > 0.0
                    and selected_unc_tensor is not None
                    and selected_unc_tensor.shape[0] == easy_mask.shape[0]
                ):
                    easy_unc_raw = selected_unc_tensor[easy_mask].mean()
                    easy_unc_loss = easy_unc_weight * easy_unc_raw
                    total = easy_unc_loss if total is None else (total + easy_unc_loss)
                    logs["hardcase_score_residual_easy_uncertainty_raw"] = (
                        easy_unc_raw.detach()
                    )

        decor_weight = float(
            getattr(
                self._config,
                "hardcase_score_residual_adapter_decor_weight",
                0.0,
            )
        )
        if (
            decor_weight > 0.0
            and hard_mask is not None
            and torch.is_tensor(hard_mask)
            and hard_mask.any()
        ):
            adapter_outputs = predictions.get("hardcase_score_residual_adapter_outputs")
            if (
                torch.is_tensor(adapter_outputs)
                and adapter_outputs.dim() == 4
                and specialist_score.dim() == 2
            ):
                bs_a, num_modes_a, num_adapters, out_dim = adapter_outputs.shape
                if (
                    bs_a == specialist_score.shape[0]
                    and num_modes_a == specialist_score.shape[1]
                    and num_adapters > 1
                    and out_dim > 0
                ):
                    selected_idx_for_decor = selected_idx_for_unc
                    if selected_idx_for_decor is None:
                        selected_idx_for_decor = self._select_mode_indices_for_gate(
                            predictions=dict(predictions),
                            batch_size=bs_a,
                            num_modes=num_modes_a,
                            device=adapter_outputs.device,
                        ).clamp(0, max(num_modes_a - 1, 0))
                    batch_idx = torch.arange(bs_a, device=adapter_outputs.device)
                    selected_adapters = adapter_outputs[batch_idx, selected_idx_for_decor]
                    selected_adapters = selected_adapters[hard_mask]
                    if selected_adapters.numel() > 0:
                        eps = max(
                            float(
                                getattr(
                                    self._config,
                                    "hardcase_score_residual_adapter_decor_eps",
                                    1e-6,
                                )
                            ),
                            1e-8,
                        )
                        vec = torch.nan_to_num(
                            selected_adapters.float(),
                            nan=0.0,
                            posinf=0.0,
                            neginf=0.0,
                        )
                        vec = vec - vec.mean(dim=-1, keepdim=True)
                        norm = torch.linalg.norm(vec, dim=-1, keepdim=True).clamp_min(eps)
                        vec = vec / norm
                        sim = torch.einsum("bkd,bjd->bkj", vec, vec)
                        eye = torch.eye(
                            num_adapters, device=sim.device, dtype=sim.dtype
                        ).unsqueeze(0)
                        offdiag_sq = (sim * (1.0 - eye)).pow(2)
                        denom = float(max(num_adapters * (num_adapters - 1), 1))
                        decor_raw = (
                            offdiag_sq.sum(dim=(1, 2)) / denom
                        ).mean()
                        decor_loss = decor_weight * decor_raw
                        total = decor_loss if total is None else (total + decor_loss)
                        logs["hardcase_score_residual_adapter_decor_raw"] = decor_raw.detach()
                        logs["hardcase_score_residual_adapter_decor_hard_count"] = (
                            hard_mask.float().sum().detach()
                        )

        diverge_weight = float(
            getattr(self._config, "hardcase_score_residual_diverge_weight", 0.0)
        )
        if diverge_weight > 0.0:
            score_generalist = predictions.get("pdm_score_generalist")
            if not torch.is_tensor(score_generalist):
                score_generalist = predictions.get("pdm_score")
            if (
                torch.is_tensor(score_generalist)
                and score_generalist.dim() == 2
                and specialist_score.dim() == 2
                and score_generalist.shape == specialist_score.shape
            ):
                bs = specialist_score.shape[0]
                num_modes = specialist_score.shape[1]
                select_predictions = dict(predictions)
                select_predictions["pdm_score"] = score_generalist
                selected_idx = self._select_mode_indices_for_gate(
                    predictions=select_predictions,
                    batch_size=bs,
                    num_modes=num_modes,
                    device=specialist_score.device,
                ).clamp(0, max(num_modes - 1, 0))
                batch_idx = torch.arange(bs, device=specialist_score.device)
                gap = (
                    specialist_score[batch_idx, selected_idx]
                    - score_generalist[batch_idx, selected_idx]
                ).abs()
                if hard_mask is not None and torch.is_tensor(hard_mask) and hard_mask.shape[0] == bs:
                    gap = gap[hard_mask]
                if gap.numel() > 0:
                    margin = float(
                        getattr(self._config, "hardcase_score_residual_diverge_margin", 0.15)
                    )
                    diverge_loss = F.relu(margin - gap).mean()
                    weighted_diverge = diverge_weight * diverge_loss
                    total = weighted_diverge if total is None else (total + weighted_diverge)
                    logs["hardcase_score_residual_diverge_raw"] = diverge_loss.detach()
                    logs["hardcase_score_residual_selected_gap"] = gap.mean().detach()
        if total is None:
            return None, logs
        return total, logs

    def _is_score_residual_train_only_active(self) -> bool:
        return bool(getattr(self._config, "hardcase_score_residual_train_only", False)) and bool(
            getattr(self._config, "hardcase_score_residual_enable", False)
        )

    def _build_trainable_zero_loss(self, predictions: Dict[str, torch.Tensor]) -> torch.Tensor:
        zero_loss: Optional[torch.Tensor] = None
        for param in self._transfuser_model.parameters():
            if not param.requires_grad:
                continue
            term = param.sum() * 0.0
            zero_loss = term if zero_loss is None else (zero_loss + term)
        if zero_loss is not None:
            return zero_loss
        for value in predictions.values():
            if torch.is_tensor(value):
                return value.sum() * 0.0
        return next(self._transfuser_model.parameters()).new_zeros(())

    def _maybe_override_loss_for_residual_train_only(
        self,
        loss_dict: Dict[str, torch.Tensor],
        specialist_score_loss: Optional[torch.Tensor],
        predictions: Dict[str, torch.Tensor],
    ) -> None:
        if not self.training:
            return
        if not self._is_score_residual_train_only_active():
            return
        if specialist_score_loss is not None:
            loss_dict["loss"] = specialist_score_loss
        else:
            loss_dict["loss"] = self._build_trainable_zero_loss(predictions)

    def compute_loss(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
        metric_cache_path_override: Optional[str] = None,
    ) -> torch.Tensor:
        """Inherited, see superclass."""
        if getattr(self._config, "pdm_score_head_only", False):
            pdm_loss = self._compute_pdm_score_loss(
                targets,
                predictions,
                metric_cache_path_override=metric_cache_path_override,
            )
            if pdm_loss is None:
                self._log_pdm_val_invalid(
                    targets, predictions, reason="pdm_score_loss_none"
                )
                if not self.training:
                    ref_tensor: Optional[torch.Tensor] = None
                    for key in ("pdm_score", "poses_reg"):
                        value = predictions.get(key)
                        if torch.is_tensor(value):
                            ref_tensor = value
                            break
                    if ref_tensor is None:
                        for value in predictions.values():
                            if torch.is_tensor(value):
                                ref_tensor = value
                                break
                    if ref_tensor is None:
                        ref_tensor = next(self.parameters()).detach()
                    zero_loss = ref_tensor.new_zeros(())
                    return {
                        "loss": zero_loss,
                        "pdm_score_loss": None,
                    }
                raise ValueError(
                    "pdm_score_head_only=True requires a valid PDM score loss. "
                    "Check that targets contain 'token' and metric cache is available."
                )
            weight = float(self._config.pdm_score_weight)
            loss = weight * pdm_loss
            loss_dict: Dict[str, torch.Tensor] = {
                "loss": loss,
                "pdm_score_loss": loss,
            }
            self._maybe_add_grpo_loss(
                loss_dict=loss_dict,
                targets=targets,
                predictions=predictions,
                metric_cache_path_override=metric_cache_path_override,
            )
            align_loss = self._compute_cached_pose_align_loss(targets, predictions)
            if align_loss is not None:
                align_weight = float(
                    getattr(self._config, "pdm_score_cached_pose_align_weight", 0.0)
                )
                loss_dict["pdm_cached_pose_align_loss"] = align_weight * align_loss
                loss_dict["loss"] = loss_dict["loss"] + align_weight * align_loss
            attn_aux_loss = self._compute_mode_bev_attention_aux_loss(predictions)
            if attn_aux_loss is not None:
                attn_aux_weight = float(
                    getattr(self._config, "mode_bev_attention_aux_weight", 0.0)
                )
                weighted_attn_aux = attn_aux_weight * attn_aux_loss
                loss_dict["mode_bev_attention_aux_loss"] = weighted_attn_aux
                loss_dict["loss"] = loss_dict["loss"] + weighted_attn_aux
            risk_area_loss, risk_area_logs = self._compute_pdm_risk_area_aux_loss(
                targets,
                predictions,
            )
            if risk_area_loss is not None:
                loss_dict["pdm_score_risk_area_loss"] = risk_area_loss
                loss_dict["loss"] = loss_dict["loss"] + risk_area_loss
            for key, value in risk_area_logs.items():
                loss_dict[key] = value
            proformer_risk_loss, proformer_risk_logs = self._compute_proformer_risk_aux_loss(
                targets,
                predictions,
            )
            if proformer_risk_loss is not None:
                proformer_risk_weight = float(
                    getattr(self._config, "proformer_risk_aux_weight", 0.0)
                )
                weighted_proformer_risk = proformer_risk_weight * proformer_risk_loss
                loss_dict["proformer_risk_aux_loss"] = weighted_proformer_risk
                loss_dict["loss"] = loss_dict["loss"] + weighted_proformer_risk
            for key, value in proformer_risk_logs.items():
                loss_dict[key] = value
            specialist_score_loss, specialist_score_logs = self._compute_pdm_specialist_score_loss(
                targets,
                predictions,
                metric_cache_path_override=metric_cache_path_override,
            )
            if specialist_score_loss is not None:
                loss_dict["pdm_score_specialist_loss"] = specialist_score_loss
                loss_dict["loss"] = loss_dict["loss"] + specialist_score_loss
            for key, value in specialist_score_logs.items():
                loss_dict[key] = value
            self._maybe_override_loss_for_residual_train_only(
                loss_dict=loss_dict,
                specialist_score_loss=specialist_score_loss,
                predictions=predictions,
            )
            return loss_dict
        loss_dict = transfuser_loss(targets, predictions, self._config)
        pdm_loss = self._compute_pdm_score_loss(
            targets,
            predictions,
            metric_cache_path_override=metric_cache_path_override,
        )
        if pdm_loss is not None:
            weight = float(self._config.pdm_score_weight)
            loss_dict["pdm_score_loss"] = weight * pdm_loss
            loss_dict["loss"] = loss_dict["loss"] + weight * pdm_loss
        self._maybe_add_grpo_loss(
            loss_dict=loss_dict,
            targets=targets,
            predictions=predictions,
            metric_cache_path_override=metric_cache_path_override,
        )
        attn_aux_loss = self._compute_mode_bev_attention_aux_loss(predictions)
        if attn_aux_loss is not None:
            attn_aux_weight = float(
                getattr(self._config, "mode_bev_attention_aux_weight", 0.0)
            )
            weighted_attn_aux = attn_aux_weight * attn_aux_loss
            loss_dict["mode_bev_attention_aux_loss"] = weighted_attn_aux
            loss_dict["loss"] = loss_dict["loss"] + weighted_attn_aux
        risk_area_loss, risk_area_logs = self._compute_pdm_risk_area_aux_loss(
            targets,
            predictions,
        )
        if risk_area_loss is not None:
            loss_dict["pdm_score_risk_area_loss"] = risk_area_loss
            loss_dict["loss"] = loss_dict["loss"] + risk_area_loss
        for key, value in risk_area_logs.items():
            loss_dict[key] = value
        proformer_risk_loss, proformer_risk_logs = self._compute_proformer_risk_aux_loss(
            targets,
            predictions,
        )
        if proformer_risk_loss is not None:
            proformer_risk_weight = float(
                getattr(self._config, "proformer_risk_aux_weight", 0.0)
            )
            weighted_proformer_risk = proformer_risk_weight * proformer_risk_loss
            loss_dict["proformer_risk_aux_loss"] = weighted_proformer_risk
            loss_dict["loss"] = loss_dict["loss"] + weighted_proformer_risk
        for key, value in proformer_risk_logs.items():
            loss_dict[key] = value
        align_loss = self._compute_cached_pose_align_loss(targets, predictions)
        if align_loss is not None:
            align_weight = float(
                getattr(self._config, "pdm_score_cached_pose_align_weight", 0.0)
            )
            loss_dict["pdm_cached_pose_align_loss"] = align_weight * align_loss
            loss_dict["loss"] = loss_dict["loss"] + align_weight * align_loss
        specialist_score_loss, specialist_score_logs = self._compute_pdm_specialist_score_loss(
            targets,
            predictions,
            metric_cache_path_override=metric_cache_path_override,
        )
        if specialist_score_loss is not None:
            loss_dict["pdm_score_specialist_loss"] = specialist_score_loss
            loss_dict["loss"] = loss_dict["loss"] + specialist_score_loss
        for key, value in specialist_score_logs.items():
            loss_dict[key] = value
        self._maybe_override_loss_for_residual_train_only(
            loss_dict=loss_dict,
            specialist_score_loss=specialist_score_loss,
            predictions=predictions,
        )
        if not self.training:
            loss_dict.update(
                self._compute_b2d_pseudo_selected_score_metrics(targets, predictions)
            )
        return loss_dict

    def get_optimizers(self) -> Union[Optimizer, Dict[str, Union[Optimizer, LRScheduler]]]:
        """Inherited, see superclass."""
        return self.get_coslr_optimizers()

    def get_step_lr_optimizers(self):
        trainable_params = [p for p in self._transfuser_model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable_params, lr=self._lr, weight_decay=self._config.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self._config.lr_steps, gamma=0.1)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def get_coslr_optimizers(self):
        # import ipdb; ipdb.set_trace()
        optimizer_cfg = dict(type=self._config.optimizer_type, 
                            lr=self._lr, 
                            weight_decay=self._config.weight_decay,
                            paramwise_cfg=self._config.opt_paramwise_cfg
                            )
        scheduler_cfg = dict(type=self._config.scheduler_type,
                            milestones=self._config.lr_steps,
                            gamma=0.1,
        )

        optimizer_cfg = DictConfig(optimizer_cfg)
        scheduler_cfg = DictConfig(scheduler_cfg)
        
        with open_dict(optimizer_cfg):
            paramwise_cfg = optimizer_cfg.pop('paramwise_cfg', None)
        
        if paramwise_cfg:
            params = []
            pgs = [[] for _ in paramwise_cfg['name']]

            for k, v in self._transfuser_model.named_parameters():
                if not v.requires_grad:
                    continue
                in_param_group = True
                for i, (pattern, pg_cfg) in enumerate(paramwise_cfg['name'].items()):
                    if pattern in k:
                        pgs[i].append(v)
                        in_param_group = False
                if in_param_group:
                    params.append(v)
        else:
            params = [p for p in self._transfuser_model.parameters() if p.requires_grad]
        if len(params) == 0 and (not paramwise_cfg or all(len(pg) == 0 for pg in pgs)):
            raise ValueError("No trainable parameters found when building optimizer.")
        
        optimizer = build_from_configs(optim, optimizer_cfg, params=params)
        # import ipdb; ipdb.set_trace()
        if paramwise_cfg:
            for pg, (_, pg_cfg) in zip(pgs, paramwise_cfg['name'].items()):
                if len(pg) == 0:
                    continue
                cfg = {}
                if 'lr_mult' in pg_cfg:
                    cfg['lr'] = optimizer_cfg['lr'] * pg_cfg['lr_mult']
                optimizer.add_param_group({'params': pg, **cfg})
        
        # scheduler = build_from_configs(optim.lr_scheduler, scheduler_cfg, optimizer=optimizer)
        scheduler = WarmupCosLR(
            optimizer=optimizer,
            lr=self._lr,
            min_lr=1e-6,
            epochs=100,
            warmup_epochs=3,
        )
        
        if 'interval' in scheduler_cfg:
            scheduler = {'scheduler': scheduler, 'interval': scheduler_cfg['interval']}
        
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def get_training_callbacks(self) -> List[pl.Callback]:
        """Inherited, see superclass."""
        callbacks: List[pl.Callback] = [TransfuserCallback(self._config)]
        save_last = bool(getattr(self._config, "checkpoint_save_last", False))
        save_last_on_train_epoch_end = bool(
            getattr(self._config, "checkpoint_save_last_on_train_epoch_end", True)
        )
        callbacks.append(
            ModelCheckpoint(
                monitor=self._config.checkpoint_monitor,
                mode=self._config.checkpoint_mode,
                save_top_k=self._config.checkpoint_save_top_k,
                save_last=save_last and not save_last_on_train_epoch_end,
                filename=self._config.checkpoint_filename,
            )
        )
        if save_last and save_last_on_train_epoch_end:
            every_n_train_steps = int(
                getattr(self._config, "checkpoint_last_every_n_train_steps", 0) or 0
            )
            last_kwargs = {
                "monitor": None,
                "save_top_k": 0,
                "save_last": True,
                "filename": "last",
                "save_on_train_epoch_end": True,
            }
            if every_n_train_steps > 0:
                last_kwargs["every_n_train_steps"] = every_n_train_steps
                last_kwargs["every_n_epochs"] = 0
            else:
                last_kwargs["every_n_epochs"] = max(
                    1,
                    int(getattr(self._config, "checkpoint_last_every_n_epochs", 1) or 1),
                )
            callbacks.append(ModelCheckpoint(**last_kwargs))
        if getattr(self._config, "save_best_bev_semantic_ckpt", True):
            callbacks.append(
                ModelCheckpoint(
                    monitor="val/bev_semantic_loss_epoch",
                    mode="min",
                    save_top_k=1,
                    save_last=False,
                    filename="best_bev",
                )
            )
        return callbacks
