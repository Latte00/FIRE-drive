from typing import Dict
import numpy as np
import torch
import torch.nn as nn
import copy
import math
from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig
from navsim.agents.diffusiondrive.transfuser_backbone import TransfuserBackbone
from navsim.agents.diffusiondrive.transfuser_backbone_vit_late import (
    TransfuserViTLateFusionBackbone,
)
try:
    from navsim.agents.diffusiondrive.transfuser_features import BoundingBox2DIndex
except ImportError:
    from enum import IntEnum

    class BoundingBox2DIndex(IntEnum):
        _X = 0
        _Y = 1
        _HEADING = 2
        _LENGTH = 3
        _WIDTH = 4

        @classmethod
        def size(cls):
            return 5

        @classmethod
        @property
        def X(cls):
            return cls._X

        @classmethod
        @property
        def Y(cls):
            return cls._Y

        @classmethod
        @property
        def HEADING(cls):
            return cls._HEADING

        @classmethod
        @property
        def LENGTH(cls):
            return cls._LENGTH

        @classmethod
        @property
        def WIDTH(cls):
            return cls._WIDTH

        @classmethod
        @property
        def POINT(cls):
            return slice(cls._X, cls._Y + 1)

        @classmethod
        @property
        def STATE_SE2(cls):
            return slice(cls._X, cls._HEADING + 1)
from navsim.common.enums import StateSE2Index
from diffusers.schedulers import DDIMScheduler
from navsim.agents.diffusiondrive.modules.conditional_unet1d import ConditionalUnet1D,SinusoidalPosEmb
import torch.nn.functional as F
from navsim.agents.diffusiondrive.modules.blocks import linear_relu_ln,bias_init_with_prob, gen_sineembed_for_position, GridSampleCrossBEVAttention
from navsim.agents.diffusiondrive.modules.multimodal_loss import LossComputer
from navsim.agents.diffusiondrive.modules.kinematic_residual import (
    KinematicResidualHead,
    constant_accel_next_xy,
)
from torch.nn import TransformerDecoder,TransformerDecoderLayer
from typing import Any, List, Dict, Optional, Union, Tuple
import matplotlib.pyplot as plt
import os


class ParallelLoRAResidualAdapters(nn.Module):
    """
    K parallel low-rank residual adapters.
    Input:  (..., in_dim)
    Output: (..., K, out_dim)
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_adapters: int,
        rank: int,
        alpha: float = 1.0,
        dropout: float = 0.0,
        init_scale: float = 1e-3,
        use_bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.num_adapters = max(1, int(num_adapters))
        self.rank = max(1, int(rank))
        self.scaling = float(alpha) / float(self.rank)
        self._dropout = nn.Dropout(float(dropout)) if float(dropout) > 0.0 else nn.Identity()

        self.lora_a = nn.Parameter(
            torch.empty(self.num_adapters, self.in_dim, self.rank)
        )
        self.lora_b = nn.Parameter(
            torch.empty(self.num_adapters, self.rank, self.out_dim)
        )
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(self.num_adapters, self.out_dim))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters(init_scale=float(init_scale))

    def reset_parameters(self, init_scale: float = 1e-3) -> None:
        nn.init.normal_(self.lora_a, mean=0.0, std=max(float(init_scale), 1e-6))
        nn.init.zeros_(self.lora_b)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() < 2:
            raise ValueError(f"LoRA adapter expects tensor with dim>=2, got {x.shape}")
        orig_shape = x.shape[:-1]
        x_flat = self._dropout(x.reshape(-1, x.shape[-1]))
        inter = torch.einsum("ni,kir->nkr", x_flat, self.lora_a)
        out = torch.einsum("nkr,kro->nko", inter, self.lora_b)
        out = out * self.scaling
        if self.bias is not None:
            out = out + self.bias.unsqueeze(0)
        return out.reshape(*orig_shape, self.num_adapters, self.out_dim)


class V2TransfuserModel(nn.Module):
    """Torch module for Transfuser."""

    def __init__(self, config: TransfuserConfig):
        """
        Initializes TransFuser torch module.
        :param config: global config dataclass of TransFuser.
        """

        super().__init__()

        self._query_splits = [
            1,
            config.num_bounding_boxes,
        ]

        self._config = config
        backbone_mode = getattr(config, "image_backbone_mode", "transfuser")
        if backbone_mode == "transfuser":
            self._backbone = TransfuserBackbone(config)
        elif backbone_mode == "davit_vitl_late":
            self._backbone = TransfuserViTLateFusionBackbone(config)
        else:
            raise ValueError(f"Unsupported image_backbone_mode: {backbone_mode}")

        self._keyval_embedding = nn.Embedding(8**2 + 1, config.tf_d_model)  # 8x8 feature grid + trajectory -> (64+1, d_model)一组可学习的位置编码参数
        self._query_embedding = nn.Embedding(sum(self._query_splits), config.tf_d_model)

        # usually, the BEV features are variable in size.
        self._bev_downscale = nn.Conv2d(
            self._backbone.num_lidar_features, config.tf_d_model, kernel_size=1
        )
        self._status_encoding = nn.Linear(4 + 2 + 2, config.tf_d_model)   
        self._ego_history_encoder = None
        if config.include_ego_history:
            self._ego_history_encoder = nn.GRU(
                input_size=3, hidden_size=config.tf_d_model, batch_first=True
            )
        self._kinematic_residual_head = None
        if (
            config.kinematic_residual_enable
            or config.kinematic_residual_weight > 0
            or config.kinematic_residual_as_condition
        ):
            residual_in_dim = 4 + 2 + 2
            if config.kinematic_residual_use_history and config.include_ego_history:
                residual_in_dim += config.tf_d_model
            self._kinematic_residual_head = KinematicResidualHead(
                residual_in_dim,
                hidden_dim=config.kinematic_residual_hidden_dim,
                out_dim=2,
            )
        self._kinematic_residual_proj = None
        if config.kinematic_residual_as_condition:
            self._kinematic_residual_proj = nn.Linear(2, config.tf_d_model)
        self._img_token_pool = None
        self._lidar_token_pool = None
        self._img_token_proj = None
        self._lidar_token_proj = None
        if config.denoise_use_image_tokens or config.pdm_score_use_image_tokens:
            self._img_token_pool = nn.AdaptiveAvgPool2d(
                (config.img_vert_anchors, config.img_horz_anchors)
            )
            self._img_token_proj = nn.Conv2d(
                self._backbone.num_image_features, config.tf_d_model, kernel_size=1
            )
        if config.denoise_use_lidar_tokens or config.pdm_score_use_lidar_tokens:
            self._lidar_token_pool = nn.AdaptiveAvgPool2d(
                (config.lidar_vert_anchors, config.lidar_horz_anchors)
            )
            self._lidar_token_proj = nn.Conv2d(
                self._backbone.num_lidar_features, config.tf_d_model, kernel_size=1
            )

        self._bev_semantic_head = nn.Sequential(    # bev语义分割头
            nn.Conv2d(
                config.bev_features_channels,
                config.bev_features_channels,
                kernel_size=(3, 3),
                stride=1,
                padding=(1, 1),
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                config.bev_features_channels,
                config.num_bev_classes,
                kernel_size=(1, 1),
                stride=1,
                padding=0,
                bias=True,
            ),
            nn.Upsample(
                size=(config.lidar_resolution_height // 2, config.lidar_resolution_width),
                mode="bilinear",
                align_corners=False,
            ),
        )

        tf_decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.tf_d_model,
            nhead=config.tf_num_head,
            dim_feedforward=config.tf_d_ffn,
            dropout=config.tf_dropout,
            batch_first=True,
        )

        self._tf_decoder = nn.TransformerDecoder(tf_decoder_layer, config.tf_num_layers)
        self._agent_head = AgentHead(
            num_agents=config.num_bounding_boxes,
            d_ffn=config.tf_d_ffn,
            d_model=config.tf_d_model,
        )

        self._trajectory_head = TrajectoryHead(     # 轨迹预测扩散模型
            num_poses=config.trajectory_sampling.num_poses,
            d_ffn=config.tf_d_ffn,
            d_model=config.tf_d_model,
            plan_anchor_path=config.plan_anchor_path,
            config=config,
        ) 
        self.bev_proj = nn.Sequential(
            *linear_relu_ln(256, 1, 1,321),
        )


    def forward(self, features: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]=None) -> Dict[str, torch.Tensor]:
        """Torch module forward pass."""
        # 1. 提取多模态特征
        camera_feature: torch.Tensor = features["camera_feature"]   # 相机特征（batch_size, 3, 256, 1024）左中右三个方向的三通道宽图
        lidar_feature: torch.Tensor = features["lidar_feature"]   # 雷达特征（batch_size, 1, 256, 1024）左中右三个方向的单通道宽图
        status_feature: torch.Tensor = features["status_feature"]   # 状态特征（batch_size, 4+2+2）

        batch_size = status_feature.shape[0]
        proformer_pure_visual_mode = (
            getattr(self._config, "trajectory_decoder_type", "diffusion") == "proformer"
            and bool(getattr(self._config, "proformer_pure_visual_mode", False))
        )
        if proformer_pure_visual_mode:
            lidar_feature = torch.zeros_like(lidar_feature)
        # import pdb; pdb.set_trace()
        # 2. 用backboen提取BEV特征（不做修改），相机＋雷达
        if bool(getattr(self._config, "proformer_four_camera_input", False)):
            if hasattr(self, "_prepare_camera_feature"):
                camera_feature = self._prepare_camera_feature(camera_feature)
            bev_feature_upscale, bev_feature, image_feature, lidar_feature = self._backbone(
                camera_feature,
                lidar_feature,
                lidar2img=features.get("lidar2img"),
                img_shape=features.get("img_shape"),
            )
        else:
            bev_feature_upscale, bev_feature, image_feature, lidar_feature = self._backbone(
                camera_feature, lidar_feature
            )

        bev_feature_upscale_sem = bev_feature_upscale
        bev_feature_upscale_traj = bev_feature_upscale
        bev_feature_traj = bev_feature
        image_feature_traj = image_feature.detach() if image_feature is not None else None
        lidar_feature_traj = None if proformer_pure_visual_mode else (
            lidar_feature.detach() if lidar_feature is not None else None
        )
        cross_bev_feature = bev_feature_upscale_sem
        bev_spatial_shape = bev_feature_upscale_sem.shape[2:]   # 上采样BEV特征的空 间形状（高度，宽度）
        concat_cross_bev_shape = bev_feature_traj.shape[2:]
        bev_feature = self._bev_downscale(bev_feature_traj).flatten(-2, -1)
        bev_semantic_map = self._bev_semantic_head(bev_feature_upscale_sem)

        # 3. 对BEV特征进行编码，构建key-value和query
        bev_feature = bev_feature.permute(0, 2, 1)
        status_encoding = self._status_encoding(status_feature)
        history_tokens = None
        history_state_last = None
        ego_history = None
        if self._ego_history_encoder is not None:
            ego_history = features.get("ego_history")
            if ego_history is None and targets is not None:
                ego_history = targets.get("ego_history")
            if ego_history is not None:
                if ego_history.dim() == 2:
                    ego_history = ego_history.unsqueeze(0)
                ego_history = ego_history.to(
                    device=status_feature.device, dtype=status_feature.dtype    
                )
                history_output, history_state = self._ego_history_encoder(ego_history)
                history_state_last = history_state[-1]
                if self._config.ego_history_to_status:
                    status_encoding = status_encoding + history_state[-1]
                if self._config.denoise_use_history_tokens:
                    history_tokens = history_output

        kinematic_next_base = None
        kinematic_next_residual = None
        kinematic_next_pred = None
        if self._kinematic_residual_head is not None:
            dt = float(self._config.trajectory_sampling.interval_length)
            kinematic_next_base = constant_accel_next_xy(status_feature, dt)
            residual_input = status_feature
            if (
                self._config.kinematic_residual_use_history
                and history_state_last is not None
            ):
                residual_input = torch.cat(
                    [residual_input, history_state_last], dim=-1
                )
            kinematic_next_residual = self._kinematic_residual_head(residual_input)
            kinematic_next_pred = kinematic_next_base + kinematic_next_residual
            if self._kinematic_residual_proj is not None:
                kinematic_cond = kinematic_next_pred.detach()
                status_encoding = status_encoding + self._kinematic_residual_proj(
                    kinematic_cond
                )

        keyval = torch.concatenate([bev_feature, status_encoding[:, None]], dim=1)
        keyval += self._keyval_embedding.weight[None, ...]  # 对BEV特征和状态编码添加位置编码，增加第一维为batch

        concat_cross_bev = keyval[:,:-1].permute(0,2,1).contiguous().view(batch_size, -1, concat_cross_bev_shape[0], concat_cross_bev_shape[1])
        # upsample to the same shape as bev_feature_upscale

        concat_cross_bev = F.interpolate(concat_cross_bev, size=bev_spatial_shape, mode='bilinear', align_corners=False)
        feasible_masks: Optional[Dict[str, torch.Tensor]] = None
        if targets is not None:
            feasible_area = targets.get("feasible_area_mask")
            feasible_lane = targets.get("feasible_lane_mask")
            if feasible_area is not None and feasible_lane is not None:
                feasible_masks = {
                    "feasible_area_mask": feasible_area,
                    "feasible_lane_mask": feasible_lane,
                }
        if not self.training and self._config.eval_use_predicted_bev_masks:
            feasible_masks = None
        if feasible_masks is None and self._config.extract_feasible_lane:       
            with torch.no_grad():
                feasible_masks = self._extract_feasible_lane_masks(
                    bev_semantic_map
                )
            feasible_masks = {k: v.detach() for k, v in feasible_masks.items()}

        reachability_mask = None
        reachability_feature = None
        physical_area_mask = None
        physical_lane_mask = None
        if self._config.reachability_use_bicycle:
            with torch.no_grad():
                mask_shape = (
                    feasible_masks["feasible_area_mask"].shape[-2:]
                    if feasible_masks is not None
                    else bev_semantic_map.shape[-2:]
                )
                reachability_mask, reachability_feature = self._compute_reachability_mask(
                    status_feature,
                    mask_shape,
                    ego_history=ego_history
                    if getattr(self._config, "reachability_use_history", False)
                    else None,
                )
            if feasible_masks is not None:
                physical_area_mask = (
                    feasible_masks["feasible_area_mask"] & reachability_mask
                )
                lane_src = feasible_masks.get("feasible_lane_mask")
                if lane_src is not None:
                    physical_lane_mask = lane_src & physical_area_mask
            else:
                physical_area_mask = reachability_mask

        lane_mask = None
        if feasible_masks is not None:
            lane_mask = feasible_masks.get("feasible_lane_mask")
        elif targets is not None:
            lane_mask = targets.get("feasible_lane_mask")
        if lane_mask is not None:
            if lane_mask.dim() == 2:
                lane_mask = lane_mask.unsqueeze(0).unsqueeze(1)
            elif lane_mask.dim() == 3:
                lane_mask = lane_mask.unsqueeze(1)
            lane_mask = lane_mask.detach().float()
            if lane_mask.shape[-2:] != bev_spatial_shape:
                lane_mask = F.interpolate(
                    lane_mask,
                    size=bev_spatial_shape,
                    mode="nearest",
                )
        else:
            lane_mask = torch.zeros(
                (batch_size, 1, bev_spatial_shape[0], bev_spatial_shape[1]),
                device=cross_bev_feature.device,
                dtype=cross_bev_feature.dtype,
            )
        # concat concat_cross_bev, cross_bev_feature, and feasible lane mask    
        cross_bev_feature = torch.cat([concat_cross_bev, cross_bev_feature, lane_mask], dim=1)

        cross_bev_feature = self.bev_proj(cross_bev_feature.flatten(-2,-1).permute(0,2,1))
        cross_bev_feature = cross_bev_feature.permute(0,2,1).contiguous().view(batch_size, -1, bev_spatial_shape[0], bev_spatial_shape[1])
        # 4. Transformer 解码
        query = self._query_embedding.weight[None, ...].repeat(batch_size, 1, 1)
        query_out = self._tf_decoder(query, keyval)

        # 5. 分离轨迹和智能体的query
        trajectory_query, agents_query = query_out.split(self._query_splits, dim=1)

        output: Dict[str, torch.Tensor] = {"bev_semantic_map": bev_semantic_map}
        if feasible_masks is not None:
            output.update(feasible_masks)
        if reachability_mask is not None:
            output["reachability_mask"] = reachability_mask
        if physical_area_mask is not None:
            output["physical_feasible_area_mask"] = physical_area_mask
        if physical_lane_mask is not None:
            output["physical_feasible_lane_mask"] = physical_lane_mask
        if reachability_feature is not None:
            output["reachability_feature"] = reachability_feature
        cross_bev_decorrelation_loss = torch.tensor(0.0, device=cross_bev_feature.device)
        if self.training and self._config.cross_bev_decorrelation_weight > 0:
            cross_bev_decorrelation_loss = self._cross_bev_decorrelation_loss(cross_bev_feature)
        output["cross_bev_decorrelation_loss"] = cross_bev_decorrelation_loss
        if getattr(self._config, "output_bev_feature", False):
            output["cross_bev_feature"] = cross_bev_feature.detach()
        if kinematic_next_pred is not None:
            output["kinematic_next_base"] = kinematic_next_base
            output["kinematic_next_residual"] = kinematic_next_residual
            output["kinematic_next_pred"] = kinematic_next_pred
        # import ipdb; ipdb.set_trace()
        # 可视化 feasible_area_mask 和 feasible_lane_mask
        
        image_tokens = None
        lidar_tokens = None
        score_image_tokens = None
        score_lidar_tokens = None
        if (
            image_feature_traj is not None
            and self._img_token_pool is not None
            and self._img_token_proj is not None
        ):
            if self._config.denoise_use_image_tokens:
                img_feat = self._img_token_pool(image_feature_traj)
                img_feat = self._img_token_proj(img_feat)
                image_tokens = img_feat.flatten(-2, -1).permute(0, 2, 1).contiguous()
            if self._config.pdm_score_use_image_tokens:
                if image_tokens is not None:
                    score_image_tokens = image_tokens
                else:
                    img_feat = self._img_token_pool(image_feature_traj)
                    img_feat = self._img_token_proj(img_feat)
                    score_image_tokens = img_feat.flatten(-2, -1).permute(0, 2, 1).contiguous()
        if (
            lidar_feature_traj is not None
            and self._lidar_token_pool is not None
            and self._lidar_token_proj is not None
        ):
            if self._config.denoise_use_lidar_tokens:
                lidar_feat = self._lidar_token_pool(lidar_feature_traj)
                lidar_feat = self._lidar_token_proj(lidar_feat)
                lidar_tokens = lidar_feat.flatten(-2, -1).permute(0, 2, 1).contiguous()
            if self._config.pdm_score_use_lidar_tokens:
                if lidar_tokens is not None:
                    score_lidar_tokens = lidar_tokens
                else:
                    lidar_feat = self._lidar_token_pool(lidar_feature_traj)     
                    lidar_feat = self._lidar_token_proj(lidar_feat)
                    score_lidar_tokens = lidar_feat.flatten(-2, -1).permute(0, 2, 1).contiguous()

        use_feasible_anchor_mask = bool(
            getattr(self._config, "anchor_use_feasible_area_mask", True)
        )
        anchor_mask = (
            feasible_masks["feasible_area_mask"]
            if feasible_masks is not None and use_feasible_anchor_mask
            else None
        )
        anchor_mask_relaxed = None
        anchor_mask_pad = True
        anchor_mask_relaxed_pad = True
        anchor_strict_steps = int(
            getattr(self._config, "reachability_anchor_strict_steps", 0)
        )
        if self._config.reachability_use_bicycle and self._config.reachability_use_for_anchor:
            if use_feasible_anchor_mask and physical_area_mask is not None:
                anchor_mask = physical_area_mask
                anchor_mask_pad = False
                if feasible_masks is not None:
                    anchor_mask_relaxed = feasible_masks.get("feasible_area_mask")
            elif reachability_mask is not None and anchor_mask is None:
                anchor_mask = reachability_mask
                anchor_mask_pad = False
        if (
            anchor_strict_steps > 0
            and bool(getattr(self._config, "anchor_relaxed_allow_vehicle_overlap", False))
        ):
            relaxed_base = anchor_mask_relaxed
            if relaxed_base is None and use_feasible_anchor_mask and feasible_masks is not None:
                relaxed_base = feasible_masks.get("feasible_area_mask")
            if relaxed_base is not None:
                vehicle_source = bev_semantic_map
                if bool(
                    getattr(self._config, "anchor_relaxed_use_target_vehicle_mask", True)
                ) and targets is not None:
                    target_bev = targets.get("bev_semantic_map")
                    if isinstance(target_bev, torch.Tensor):
                        vehicle_source = target_bev
                vehicle_mask = self._extract_relax_obstacle_mask(
                    bev_semantic_map=vehicle_source,
                    ref_shape=tuple(relaxed_base.shape[-2:]),
                )
                if vehicle_mask is not None:
                    relaxed_base = relaxed_base.to(device=vehicle_mask.device, dtype=torch.bool)
                    if vehicle_mask.shape[0] != relaxed_base.shape[0]:
                        if vehicle_mask.shape[0] == 1:
                            vehicle_mask = vehicle_mask.repeat(relaxed_base.shape[0], 1, 1)
                        elif relaxed_base.shape[0] == 1:
                            relaxed_base = relaxed_base.repeat(vehicle_mask.shape[0], 1, 1)
                        else:
                            vehicle_mask = None
                    if vehicle_mask is not None:
                        anchor_mask_relaxed = relaxed_base | vehicle_mask

        trajectory = self._trajectory_head(
            trajectory_query,
            agents_query,
            cross_bev_feature,
            bev_spatial_shape,
            status_encoding[:, None],
            targets=targets,
            risk_bev_map=self._build_mode_bev_risk_map(
                bev_semantic_map=bev_semantic_map,
                targets=targets,
                ref_shape=bev_spatial_shape,
            ),
            global_img=None,
            image_tokens=image_tokens,
            lidar_tokens=lidar_tokens,
            score_image_tokens=score_image_tokens,
            score_lidar_tokens=score_lidar_tokens,
            history_tokens=history_tokens,
            anchor_mask=anchor_mask,
            anchor_mask_relaxed=anchor_mask_relaxed,
            anchor_mask_pad=anchor_mask_pad,
            anchor_mask_relaxed_pad=anchor_mask_relaxed_pad,
            anchor_strict_steps=anchor_strict_steps,
            status_feature=status_feature,
        )
        output.update(trajectory)

        # vis_area = None
        # vis_lane = None
        # vis_bev = None
        # if targets is not None:
        #     vis_area = targets.get("feasible_area_mask")
        #     vis_lane = targets.get("feasible_lane_mask")
        #     vis_bev = targets.get("bev_semantic_map")
        # if vis_area is None or vis_lane is None:
        #     vis_area = output.get("feasible_area_mask")
        #     vis_lane = output.get("feasible_lane_mask")
        # if vis_bev is None:
        #     vis_bev = output.get("bev_semantic_map")
        # if vis_area is not None and vis_lane is not None and vis_bev is not None:
        #     self._visualize_masks(
        #         vis_area,
        #         vis_lane,
        #         vis_bev,
        #         plan_anchor=targets.get("trajectory_candidates")[0,:,:,0:2],
        #     )
        # import ipdb; ipdb.set_trace()
        
        agents = self._agent_head(agents_query)
        output.update(agents)

        return output

    def _extract_feasible_lane_masks(
        self, bev_semantic_map: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Extract feasible drivable area and lane centerlines from BEV logits.
        """
        class_map = bev_semantic_map.argmax(dim=1)
        road_label = self._config.bev_road_label
        centerline_label = self._config.bev_centerline_label

        drivable_mask = (class_map == road_label) | (class_map == centerline_label)
        centerline_mask = class_map == centerline_label

        batch_size, height, width = class_map.shape
        ego_row = 0
        ego_col = (width - 1) // 2

        seed_rows = max(1, min(self._config.feasible_lane_seed_rows, height))   
        row_start = 0
        row_end = min(height, seed_rows)
        seed_cols = max(0, min(self._config.feasible_lane_seed_cols, width // 2))
        col_start = max(0, ego_col - seed_cols)
        col_end = min(width, ego_col + seed_cols + 1)
        if col_end <= col_start:
            col_start = min(ego_col, width - 1)
            col_end = min(width, col_start + 1)

        drivable_scores = torch.maximum(
            bev_semantic_map[:, road_label],
            bev_semantic_map[:, centerline_label],
        )

        seed_mask = torch.zeros_like(drivable_mask)
        for batch_idx in range(batch_size):
            region_mask = drivable_mask[batch_idx, row_start:row_end, col_start:col_end]
            if region_mask.any().item():
                region_scores = drivable_scores[
                    batch_idx, row_start:row_end, col_start:col_end
                ].masked_fill(~region_mask, float("-inf"))
                flat_idx = region_scores.view(-1).argmax().item()
                row_offset = flat_idx // (col_end - col_start)
                col_offset = flat_idx % (col_end - col_start)
                seed_mask[batch_idx, row_start + row_offset, col_start + col_offset] = True
            else:
                full_mask = drivable_mask[batch_idx]
                if full_mask.any().item():
                    coords = full_mask.nonzero(as_tuple=False)
                    ego_index = torch.tensor(
                        [ego_row, ego_col], device=coords.device, dtype=coords.dtype
                    )
                    deltas = coords - ego_index
                    dist2 = deltas[:, 0] * deltas[:, 0] + deltas[:, 1] * deltas[:, 1]
                    best_idx = dist2.argmin().item()
                    seed_mask[batch_idx, coords[best_idx, 0], coords[best_idx, 1]] = True
                else:
                    seed_mask[batch_idx, min(ego_row, height - 1), min(ego_col, width - 1)] = True

        feasible_area_mask = self._flood_fill_mask(drivable_mask, seed_mask)
        feasible_lane_mask = feasible_area_mask & centerline_mask

        return {
            "feasible_area_mask": feasible_area_mask,
            "feasible_lane_mask": feasible_lane_mask,
        }

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

    def _extract_relax_obstacle_mask(
        self,
        bev_semantic_map: Optional[torch.Tensor],
        ref_shape: Optional[Tuple[int, int]] = None,
    ) -> Optional[torch.Tensor]:
        class_map = self._bev_to_class_map(bev_semantic_map)
        if class_map is None:
            return None
        vehicle_label = int(getattr(self._config, "bev_vehicle_label", 5))
        obstacle_mask = class_map == vehicle_label
        if bool(getattr(self._config, "anchor_relaxed_include_pedestrians", False)):
            ped_label = int(getattr(self._config, "bev_pedestrian_label", 6))
            obstacle_mask = obstacle_mask | (class_map == ped_label)
        if ref_shape is not None and tuple(obstacle_mask.shape[-2:]) != tuple(ref_shape):
            obstacle_mask = (
                F.interpolate(
                    obstacle_mask.float().unsqueeze(1),
                    size=ref_shape,
                    mode="nearest",
                ).squeeze(1)
                > 0.5
            )
        return obstacle_mask.bool()

    def _build_mode_bev_risk_map(
        self,
        bev_semantic_map: Optional[torch.Tensor],
        targets: Optional[Dict[str, torch.Tensor]] = None,
        ref_shape: Optional[Tuple[int, int]] = None,
    ) -> Optional[torch.Tensor]:
        if not bool(getattr(self._config, "mode_bev_attention_risk_enable", False)):
            return None

        source_bev = bev_semantic_map
        if (
            self.training
            and bool(
                getattr(
                    self._config,
                    "mode_bev_attention_risk_use_target_in_train",
                    True,
                )
            )
            and isinstance(targets, dict)
            and isinstance(targets.get("bev_semantic_map"), torch.Tensor)
        ):
            source_bev = targets.get("bev_semantic_map")

        if not isinstance(source_bev, torch.Tensor):
            return None
        if source_bev.dim() == 3:
            source_bev = source_bev.unsqueeze(0)
        if source_bev.dim() != 4 or source_bev.shape[1] != self._config.num_bev_classes:
            return None

        source_bev = source_bev.to(dtype=torch.float32)
        bev_prob = torch.softmax(source_bev, dim=1)
        road_label = int(self._config.bev_road_label)
        centerline_label = int(self._config.bev_centerline_label)
        static_label = int(getattr(self._config, "bev_static_label", 4))
        vehicle_label = int(getattr(self._config, "bev_vehicle_label", 5))
        ped_label = int(getattr(self._config, "bev_pedestrian_label", 6))

        drivable_prob = torch.maximum(
            bev_prob[:, road_label], bev_prob[:, centerline_label]
        )
        obstacle_prob = bev_prob[:, vehicle_label]
        static_prob = None
        if bool(
            getattr(
                self._config,
                "mode_bev_attention_risk_include_static",
                True,
            )
        ) and 0 <= static_label < bev_prob.shape[1]:
            static_prob = bev_prob[:, static_label]
        if bool(
            getattr(
                self._config,
                "mode_bev_attention_risk_include_pedestrian",
                True,
            )
        ) and 0 <= ped_label < bev_prob.shape[1]:
            obstacle_prob = torch.maximum(obstacle_prob, bev_prob[:, ped_label])

        offroad_weight = float(
            getattr(
                self._config,
                "mode_bev_attention_risk_offroad_weight",
                1.0,
            )
        )
        obstacle_weight = float(
            getattr(
                self._config,
                "mode_bev_attention_risk_obstacle_weight",
                0.5,
            )
        )
        static_weight = float(
            getattr(
                self._config,
                "mode_bev_attention_risk_static_weight",
                obstacle_weight,
            )
        )
        risk_map = offroad_weight * (1.0 - drivable_prob.clamp(0.0, 1.0))
        risk_map = risk_map + obstacle_weight * obstacle_prob.clamp(0.0, 1.0)
        if static_prob is not None:
            risk_map = risk_map + static_weight * static_prob.clamp(0.0, 1.0)
        risk_map = risk_map.clamp_min(0.0)

        if ref_shape is not None and tuple(risk_map.shape[-2:]) != tuple(ref_shape):
            risk_map = F.interpolate(
                risk_map.unsqueeze(1),
                size=ref_shape,
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)
        denom = risk_map.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        return (risk_map / denom).clamp(0.0, 1.0)

    def _estimate_history_heading_speed(
        self,
        ego_history: torch.Tensor,
        dt: float,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if ego_history is None:
            return None, None
        if ego_history.dim() == 2:
            ego_history = ego_history.unsqueeze(0)
        if ego_history.shape[1] < 2:
            return None, None
        history_steps = int(getattr(self._config, "reachability_history_steps", 2))
        history_steps = max(2, min(history_steps, ego_history.shape[1]))
        min_dist = float(getattr(self._config, "reachability_history_min_dist_m", 0.5))
        last = ego_history[:, -1, :2]
        prev = ego_history[:, -history_steps, :2]
        delta = last - prev
        dist = torch.linalg.norm(delta, dim=-1)
        heading = torch.atan2(delta[:, 1], delta[:, 0])
        use_heading = dist >= min_dist
        if ego_history.shape[-1] >= 3:
            hist_heading = ego_history[:, -1, 2]
            heading = torch.where(use_heading, heading, hist_heading)
        else:
            heading = torch.where(use_heading, heading, torch.zeros_like(heading))
        dt = float(dt) if dt > 0 else 1.0
        speed = dist / (dt * max(1, history_steps - 1))
        speed = speed * use_heading.float()
        return heading, speed

    def _compute_reachability_mask(
        self,
        status_feature: torch.Tensor,
        spatial_shape: Tuple[int, int],
        ego_history: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute a bicycle-model reachability mask in BEV coordinates.
        """
        if status_feature.dim() == 1:
            status_feature = status_feature.unsqueeze(0)
        batch_size = status_feature.shape[0]
        height, width = spatial_shape
        device = status_feature.device
        dtype = status_feature.dtype

        dt = float(self._config.trajectory_sampling.interval_length)
        if dt <= 0.0:
            dt = 1.0
        steps = int(getattr(self._config, "reachability_horizon_steps", 0))
        if steps <= 0:
            steps = int(self._config.trajectory_sampling.num_poses)
        steps = max(1, steps)

        wheel_base = max(
            float(getattr(self._config, "reachability_wheel_base_m", 3.089)), 1e-3
        )
        max_steer_deg = float(
            getattr(self._config, "reachability_max_steer_deg", 35.0)
        )
        max_steer_rad = math.radians(max_steer_deg)
        steer_samples = max(
            1, int(getattr(self._config, "reachability_steer_samples", 5))
        )
        if steer_samples == 1 or max_steer_rad <= 1e-6:
            steer_vals = torch.zeros((1,), device=device, dtype=dtype)
        else:
            steer_vals = torch.linspace(
                -max_steer_rad, max_steer_rad, steer_samples, device=device, dtype=dtype
            )

        accel_min = float(
            getattr(self._config, "reachability_min_accel_mps2", -4.0)
        )
        accel_max = float(
            getattr(self._config, "reachability_max_accel_mps2", 2.0)
        )
        if accel_min > accel_max:
            accel_min, accel_max = accel_max, accel_min
        accel_samples = max(
            1, int(getattr(self._config, "reachability_accel_samples", 3))
        )
        accel_mid = 0.5 * (accel_min + accel_max)
        if accel_samples == 1 or abs(accel_max - accel_min) <= 1e-6:
            accel_vals = torch.tensor([accel_mid], device=device, dtype=dtype)
        else:
            accel_vals = torch.linspace(
                accel_min, accel_max, accel_samples, device=device, dtype=dtype
            )

        v0 = torch.zeros((batch_size,), device=device, dtype=dtype)
        if status_feature.shape[-1] >= 6:
            velocity = status_feature[:, 4:6]
            v0 = torch.linalg.norm(velocity, dim=-1)
        if not getattr(self._config, "reachability_use_speed", True):
            step_m = float(getattr(self._config, "anchor_free_step_meters", 0.0))
            if step_m > 0:
                v0 = torch.full_like(v0, step_m / max(dt, 1e-3))
        v0 = v0.clamp(min=0.0)

        steer_grid, accel_grid = torch.meshgrid(steer_vals, accel_vals)
        steer = steer_grid.reshape(1, -1)
        accel = accel_grid.reshape(1, -1)
        num_samples = steer.shape[1]

        x = torch.zeros((batch_size, num_samples), device=device, dtype=dtype)
        y = torch.zeros_like(x)
        heading = torch.zeros_like(x)
        v = v0[:, None].expand(batch_size, num_samples).clone()
        if getattr(self._config, "reachability_use_history", False) and ego_history is not None:
            hist_heading, hist_speed = self._estimate_history_heading_speed(ego_history, dt)
            if hist_heading is not None:
                heading = heading + hist_heading[:, None]
            if (
                hist_speed is not None
                and getattr(self._config, "reachability_use_history_speed", False)
            ):
                v = hist_speed[:, None].expand_as(v).clone()

        mask = torch.zeros(
            (batch_size, height, width), device=device, dtype=torch.bool
        )
        ego_row = 0
        ego_col = (width - 1) // 2
        mask[:, ego_row, ego_col] = True

        tan_steer = torch.tan(steer)
        batch_index = torch.arange(batch_size, device=device)[:, None].expand(
            batch_size, num_samples
        )
        pixel_size = float(self._config.bev_pixel_size)
        if pixel_size <= 0.0:
            pixel_size = 0.25

        for _ in range(steps):
            v = (v + accel * dt).clamp(min=0.0)
            heading = heading + (v * tan_steer / wheel_base) * dt
            x = x + v * torch.cos(heading) * dt
            y = y + v * torch.sin(heading) * dt

            row = (x / pixel_size + ego_row).round().to(torch.long)
            col = (y / pixel_size + ego_col).round().to(torch.long)
            valid = (row >= 0) & (row < height) & (col >= 0) & (col < width)
            if valid.any():
                flat_idx = row * width + col
                mask_flat = mask.view(batch_size, -1)
                mask_flat[batch_index[valid], flat_idx[valid]] = True

        dilate_m = float(getattr(self._config, "reachability_dilate_m", 0.0))
        if dilate_m > 0.0:
            radius_px = int(round(dilate_m / pixel_size))
            if radius_px > 0:
                kernel = 2 * radius_px + 1
                mask = (
                    F.max_pool2d(
                        mask.float().unsqueeze(1),
                        kernel_size=kernel,
                        stride=1,
                        padding=radius_px,
                    )
                    .squeeze(1)
                    .bool()
                )

        reachability_feature = None
        if getattr(self._config, "reachability_output_feature", False):
            steer_tensor = torch.tensor(max_steer_rad, device=device, dtype=dtype)
            max_yaw_rate = v0 * torch.tan(steer_tensor) / wheel_base
            reachability_feature = torch.stack(
                [
                    v0,
                    max_yaw_rate,
                    torch.full_like(v0, accel_min),
                    torch.full_like(v0, accel_max),
                ],
                dim=-1,
            )

        return mask, reachability_feature

    def _cross_bev_decorrelation_loss(self, cross_bev_feature: torch.Tensor) -> torch.Tensor:
        if cross_bev_feature.dim() != 4:
            return torch.tensor(0.0, device=cross_bev_feature.device)
        stride = max(1, int(self._config.cross_bev_decorrelation_stride))
        feat = cross_bev_feature
        if stride > 1:
            feat = F.avg_pool2d(feat, kernel_size=stride, stride=stride)
        batch_size, channels, height, width = feat.shape
        if channels <= 1 or height * width <= 1:
            return torch.tensor(0.0, device=feat.device)
        feat = feat.view(batch_size, channels, height * width)
        feat = feat - feat.mean(dim=-1, keepdim=True)
        feat = F.normalize(feat, dim=-1, eps=1e-6)
        gram = torch.matmul(feat, feat.transpose(1, 2))
        eye = torch.eye(channels, device=feat.device, dtype=feat.dtype)
        off_diag = gram * (1 - eye[None, ...])
        loss = (off_diag ** 2).sum(dim=(1, 2)) / (channels * (channels - 1))
        return loss.mean()

    def _flood_fill_mask(
        self, mask: torch.Tensor, seed: torch.Tensor
    ) -> torch.Tensor:
        """
        Flood fill on a boolean mask using 8-connected neighborhood.
        """
        max_iters = self._config.feasible_lane_max_iters
        if max_iters <= 0:
            max_iters = mask.shape[-2] + mask.shape[-1]

        current = seed
        for _ in range(max_iters):
            expanded = F.max_pool2d(
                current.float().unsqueeze(1), kernel_size=3, stride=1, padding=1
            ).squeeze(1) > 0
            next_mask = expanded & mask
            if torch.equal(next_mask, current):
                break
            current = next_mask

        return current

    def _visualize_masks(
        self,
        feasible_area_mask: torch.Tensor,
        feasible_lane_mask: torch.Tensor,
        bev_semantic_map: torch.Tensor,
        plan_anchor: Optional[torch.Tensor] = None,
    ):
        """
        可视化 feasible_area_mask 和 feasible_lane_mask
        """
        # 转换为 numpy 数组
        feasible_area_np = feasible_area_mask[0].cpu().numpy()
        feasible_lane_np = feasible_lane_mask[0].cpu().numpy()
        
        # 处理 bev_semantic_map - 检查是否已经是类别索引还是 logits
        bev_semantic_tensor = bev_semantic_map[0]
        if bev_semantic_tensor.dim() == 3:
            # 如果是 3D [C, H, W]，应用 argmax 获取类别索引
            bev_semantic_np = bev_semantic_tensor.argmax(dim=0).cpu().numpy()
        elif bev_semantic_tensor.dim() == 2:
            # 如果已经是 2D [H, W] 的类别索引，直接使用
            bev_semantic_np = bev_semantic_tensor.cpu().numpy()
        else:
            raise ValueError(f"Unexpected bev_semantic_map shape: {bev_semantic_tensor.shape}")

        
        # 打印形状用于调试
        anchor_rows = None
        anchor_cols = None
        ego_row = None
        ego_col = None
        if plan_anchor is not None:
            if plan_anchor.dim() == 4:
                anchor_tensor = plan_anchor[0]
            else:
                anchor_tensor = plan_anchor
            if anchor_tensor.dim() == 3 and anchor_tensor.shape[-1] == 2:
                anchor_points = anchor_tensor.reshape(-1, 2).detach().cpu()
            elif anchor_tensor.dim() == 2 and anchor_tensor.shape[-1] == 2:
                anchor_points = anchor_tensor.detach().cpu()
            else:
                anchor_points = None
            if anchor_points is not None:
                height, width = bev_semantic_np.shape
                ego_row = 0
                ego_col = (width - 1) // 2
                anchor_rows = ego_row + (anchor_points[:, 0] / self._config.bev_pixel_size)
                anchor_cols = ego_col + (anchor_points[:, 1] / self._config.bev_pixel_size)
                valid = (
                    (anchor_rows >= 0) & (anchor_rows < height) & (anchor_cols >= 0) & (anchor_cols < width)
                )
                anchor_rows = anchor_rows[valid].numpy()
                anchor_cols = anchor_cols[valid].numpy()
        print(f"feasible_area_np shape: {feasible_area_np.shape}")
        print(f"feasible_lane_np shape: {feasible_lane_np.shape}")
        print(f"bev_semantic_np shape: {bev_semantic_np.shape}")
        
        # 创建输出目录
        output_dir = "/home/xqf/DiffusionDrive-main/mask_visualizations"
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建 3x2 的子图
        fig, axes = plt.subplots(3, 2, figsize=(12, 18))
        
        # 1. BEV 语义图
        im1 = axes[0, 0].imshow(bev_semantic_np, cmap='tab20', origin='lower')
        axes[0, 0].set_title('BEV Semantic Map')
        axes[0, 0].axis('off')
        plt.colorbar(im1, ax=axes[0, 0])
        if anchor_rows is not None and anchor_cols is not None:
            axes[0, 0].scatter(anchor_cols, anchor_rows, s=8, c='yellow', alpha=0.8)
            if ego_row is not None and ego_col is not None:
                axes[0, 0].scatter([ego_col], [ego_row], s=30, c='white', marker='x')
            axes[0, 0].set_title('BEV Semantic Map + Anchors')
        
        # 2. Feasible Area Mask
        im2 = axes[0, 1].imshow(feasible_area_np, cmap='Blues', origin='lower')
        axes[0, 1].set_title('Feasible Area Mask')
        axes[0, 1].axis('off')
        plt.colorbar(im2, ax=axes[0, 1])
        
        # 3. Feasible Lane Mask
        im3 = axes[1, 0].imshow(feasible_lane_np, cmap='Reds', origin='lower')
        axes[1, 0].set_title('Feasible Lane Mask')
        axes[1, 0].axis('off')
        plt.colorbar(im3, ax=axes[1, 0])
        
        # 4. BEV + Feasible Area (叠加)
        axes[1, 1].imshow(bev_semantic_np, cmap='tab20', origin='lower', alpha=0.7)
        axes[1, 1].imshow(feasible_area_np, cmap='Blues', origin='lower', alpha=0.3)
        axes[1, 1].set_title('BEV + Feasible Area Overlay')
        axes[1, 1].axis('off')
        
        # 5. BEV + Feasible Lane (叠加)
        axes[2, 0].imshow(bev_semantic_np, cmap='tab20', origin='lower', alpha=0.7)
        axes[2, 0].imshow(feasible_lane_np, cmap='Reds', origin='lower', alpha=0.5)
        axes[2, 0].set_title('BEV + Feasible Lane Overlay')
        axes[2, 0].axis('off')
        
        # 6. Feasible Area + Feasible Lane (叠加)
        axes[2, 1].imshow(feasible_area_np, cmap='Blues', origin='lower', alpha=0.5)
        axes[2, 1].imshow(feasible_lane_np, cmap='Reds', origin='lower', alpha=0.7)
        axes[2, 1].set_title('Feasible Area + Feasible Lane Overlay')
        axes[2, 1].axis('off')
        
        plt.tight_layout()
        
        # 保存图片
        import time
        timestamp = int(time.time() * 1000)
        save_path = os.path.join(output_dir, f'mask_visualization_{timestamp}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Mask visualization saved to: {save_path}")
        
        # 显示图片
        plt.show()
        plt.close()

class AgentHead(nn.Module):
    """Bounding box prediction head."""

    def __init__(
        self,
        num_agents: int,
        d_ffn: int,
        d_model: int,
    ):
        """
        Initializes prediction head.
        :param num_agents: maximum number of agents to predict
        :param d_ffn: dimensionality of feed-forward network
        :param d_model: input dimensionality
        """
        super(AgentHead, self).__init__()

        self._num_objects = num_agents
        self._d_model = d_model
        self._d_ffn = d_ffn

        self._mlp_states = nn.Sequential(
            nn.Linear(self._d_model, self._d_ffn),
            nn.ReLU(),
            nn.Linear(self._d_ffn, BoundingBox2DIndex.size()),
        )

        self._mlp_label = nn.Sequential(
            nn.Linear(self._d_model, 1),
        )

    def forward(self, agent_queries) -> Dict[str, torch.Tensor]:
        """Torch module forward pass."""

        agent_states = self._mlp_states(agent_queries)
        agent_states[..., BoundingBox2DIndex.POINT] = agent_states[..., BoundingBox2DIndex.POINT].tanh() * 32
        agent_states[..., BoundingBox2DIndex.HEADING] = agent_states[..., BoundingBox2DIndex.HEADING].tanh() * np.pi

        agent_labels = self._mlp_label(agent_queries).squeeze(dim=-1)

        return {"agent_states": agent_states, "agent_labels": agent_labels}

class DiffMotionPlanningRefinementModule(nn.Module):
    def __init__(
        self,
        embed_dims=256,
        ego_fut_ts=8,
        ego_fut_mode=20,
        if_zeroinit_reg=True,
    ):
        super(DiffMotionPlanningRefinementModule, self).__init__()
        self.embed_dims = embed_dims
        self.ego_fut_ts = ego_fut_ts
        self.ego_fut_mode = ego_fut_mode
        self.plan_cls_branch = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 2),
            nn.Linear(embed_dims, 1),
        )
        self.plan_reg_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, ego_fut_ts * 3),
        )
        self.if_zeroinit_reg = False

        self.init_weight()

    def init_weight(self):
        if self.if_zeroinit_reg:
            nn.init.constant_(self.plan_reg_branch[-1].weight, 0)
            nn.init.constant_(self.plan_reg_branch[-1].bias, 0)

        bias_init = bias_init_with_prob(0.01)
        nn.init.constant_(self.plan_cls_branch[-1].bias, bias_init)
    def forward(
        self,
        traj_feature,
    ):
        bs, ego_fut_mode, _ = traj_feature.shape

        # 6. get final prediction
        traj_feature = traj_feature.view(bs, ego_fut_mode,-1)
        plan_cls = self.plan_cls_branch(traj_feature).squeeze(-1)
        traj_delta = self.plan_reg_branch(traj_feature)
        plan_reg = traj_delta.reshape(bs,ego_fut_mode, self.ego_fut_ts, 3)

        return plan_reg, plan_cls
class ModulationLayer(nn.Module):

    def __init__(self, embed_dims: int, condition_dims: int):
        super(ModulationLayer, self).__init__()
        self.if_zeroinit_scale=False
        self.embed_dims = embed_dims
        self.scale_shift_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(condition_dims, embed_dims*2),
        )
        self.init_weight()

    def init_weight(self):
        if self.if_zeroinit_scale:
            nn.init.constant_(self.scale_shift_mlp[-1].weight, 0)
            nn.init.constant_(self.scale_shift_mlp[-1].bias, 0)

    def forward(
        self,
        traj_feature,
        time_embed,
        global_cond=None,
        global_img=None,
    ):
        if global_cond is not None:
            global_feature = torch.cat([
                    global_cond, time_embed
                ], axis=-1)
        else:
            global_feature = time_embed
        if global_img is not None:
            global_img = global_img.flatten(2,3).permute(0,2,1).contiguous()
            global_feature = torch.cat([
                    global_img, global_feature
                ], axis=-1)
        
        scale_shift = self.scale_shift_mlp(global_feature)
        scale,shift = scale_shift.chunk(2,dim=-1)
        traj_feature = traj_feature * (1 + scale) + shift
        return traj_feature

class CustomTransformerDecoderLayer(nn.Module):
    def __init__(self, 
                 num_poses,
                 d_model,
                 d_ffn,
                 config,
                 ):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        self.dropout1 = nn.Dropout(0.1)
        self.cross_bev_attention = GridSampleCrossBEVAttention(
            config.tf_d_model,
            config.tf_num_head,
            num_points=num_poses,
            config=config,
            in_bev_dims=256,
        )
        self.cross_agent_attention = nn.MultiheadAttention(
            config.tf_d_model,
            config.tf_num_head,
            dropout=config.tf_dropout,
            batch_first=True,
        )
        self.cross_img_attention = None
        self.cross_lidar_attention = None
        self.cross_history_attention = None
        if config.denoise_use_image_tokens:
            self.cross_img_attention = nn.MultiheadAttention(
                config.tf_d_model,
                config.tf_num_head,
                dropout=config.tf_dropout,
                batch_first=True,
            )
        if config.denoise_use_lidar_tokens:
            self.cross_lidar_attention = nn.MultiheadAttention(
                config.tf_d_model,
                config.tf_num_head,
                dropout=config.tf_dropout,
                batch_first=True,
            )
        if config.denoise_use_history_tokens:
            self.cross_history_attention = nn.MultiheadAttention(
                config.tf_d_model,
                config.tf_num_head,
                dropout=config.tf_dropout,
                batch_first=True,
            )
        self.cross_ego_attention = None
        if config.denoise_use_ego_query:
            self.cross_ego_attention = nn.MultiheadAttention(
                config.tf_d_model,
                config.tf_num_head,
                dropout=config.tf_dropout,
                batch_first=True,
            )
        self.ffn = nn.Sequential(
            nn.Linear(config.tf_d_model, config.tf_d_ffn),
            nn.ReLU(),
            nn.Linear(config.tf_d_ffn, config.tf_d_model),
        )
        self.norm1 = nn.LayerNorm(config.tf_d_model)
        self.norm_bev = nn.LayerNorm(config.tf_d_model) if config.denoise_norm_bev else None
        self.norm_img = nn.LayerNorm(config.tf_d_model) if config.denoise_use_image_tokens else None
        self.norm_lidar = nn.LayerNorm(config.tf_d_model) if config.denoise_use_lidar_tokens else None
        self.norm_history = nn.LayerNorm(config.tf_d_model) if config.denoise_use_history_tokens else None
        self.norm2 = nn.LayerNorm(config.tf_d_model) if config.denoise_use_ego_query else None
        self.norm3 = nn.LayerNorm(config.tf_d_model)
        self._decorrelation_weight = config.diff_output_decorrelation_weight
        self.time_modulation = None
        if config.denoise_use_time_embed:
            self.time_modulation = ModulationLayer(config.tf_d_model, 256)
        self.task_decoder = DiffMotionPlanningRefinementModule(
            embed_dims=config.tf_d_model,
            ego_fut_ts=num_poses,
            ego_fut_mode=config.ego_fut_mode,
        )

    def _decorrelation_loss(self, traj_feature: torch.Tensor) -> torch.Tensor:
        if traj_feature.dim() != 3:
            return torch.tensor(0.0, device=traj_feature.device)
        batch_size, num_modes, feat_dim = traj_feature.shape
        if batch_size * num_modes <= 1 or feat_dim <= 1:
            return torch.tensor(0.0, device=traj_feature.device)
        rep = traj_feature.reshape(batch_size * num_modes, feat_dim)
        mu = rep.mean(dim=0, keepdim=True)
        var = rep.var(dim=0, keepdim=True, unbiased=False)
        rep = (rep - mu) / torch.sqrt(var + 1e-8)
        corr = rep.t().matmul(rep)
        eye = torch.eye(feat_dim, device=traj_feature.device, dtype=traj_feature.dtype)
        off_diag = corr[~eye.bool()]
        denom = max(1, batch_size * num_modes)
        return (off_diag ** 2).mean() / denom

    def forward(self,
                traj_feature,
                noisy_traj_points,
                bev_feature,
                bev_spatial_shape,
                agents_query,
                ego_query,
                time_embed,
                status_encoding,
                global_img=None,
                image_tokens=None,
                lidar_tokens=None,
                history_tokens=None):
        # 交叉注意力：加噪轨迹特征（q）x 加噪轨迹点 x bev特征 /x bev空间形状（长、宽形状，未启用该形状参数）
        traj_feature = self.cross_bev_attention(traj_feature,noisy_traj_points,bev_feature,bev_spatial_shape)
        if self.norm_bev is not None:
            traj_feature = self.norm_bev(traj_feature)
        if image_tokens is not None and self.cross_img_attention is not None and self.norm_img is not None:
            traj_feature = traj_feature + self.dropout(
                self.cross_img_attention(traj_feature, image_tokens, image_tokens)[0]
            )
            traj_feature = self.norm_img(traj_feature)
        if lidar_tokens is not None and self.cross_lidar_attention is not None and self.norm_lidar is not None:
            traj_feature = traj_feature + self.dropout(
                self.cross_lidar_attention(traj_feature, lidar_tokens, lidar_tokens)[0]
            )
            traj_feature = self.norm_lidar(traj_feature)
        if history_tokens is not None and self.cross_history_attention is not None and self.norm_history is not None:
            traj_feature = traj_feature + self.dropout(
                self.cross_history_attention(traj_feature, history_tokens, history_tokens)[0]
            )
            traj_feature = self.norm_history(traj_feature)
        # 加噪轨迹注意力 + 交叉注意力: 加噪轨迹注意力（q）x 所有车q（k,v）
        traj_feature = traj_feature + self.dropout(self.cross_agent_attention(traj_feature, agents_query,agents_query)[0])
        traj_feature = self.norm1(traj_feature)
        
        # traj_feature = traj_feature + self.dropout(self.self_attn(traj_feature, traj_feature, traj_feature)[0])

        # 4.5 cross attention with  ego query
        # 加噪轨迹注意力 + 交叉注意力: 加噪轨迹注意力（q）x 自车q（k,v）
        if (
            ego_query is not None
            and self.cross_ego_attention is not None
            and self.norm2 is not None
        ):
            traj_feature = traj_feature + self.dropout1(
                self.cross_ego_attention(traj_feature, ego_query, ego_query)[0]
            )
            traj_feature = self.norm2(traj_feature)
        
        # 4.6 feedforward network
        traj_feature = self.norm3(self.ffn(traj_feature))
        # 4.8 modulate with time steps
        if self.time_modulation is not None and time_embed is not None:
            traj_feature = self.time_modulation(
                traj_feature,
                time_embed,
                global_cond=None,
                global_img=global_img,
            )

        decorrelation_loss = torch.tensor(0.0, device=traj_feature.device)
        if self.training and self._decorrelation_weight > 0:
            decorrelation_loss = self._decorrelation_loss(traj_feature)

        # 4.9 predict the offset & heading
        poses_reg, poses_cls = self.task_decoder(traj_feature) # bs,num_modes,num_poses,3; bs,num_modes
        poses_reg[...,:2] = poses_reg[...,:2] + noisy_traj_points   # 残差
        poses_reg[..., StateSE2Index.HEADING] = poses_reg[..., StateSE2Index.HEADING].tanh() * np.pi

        return poses_reg, poses_cls, decorrelation_loss
def _get_clones(module, N):
    # FIXME: copy.deepcopy() is not defined on nn.module
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class CustomTransformerDecoder(nn.Module):
    """Custom transformer decoder. 自定义Transformer解码器, 用于扩散模型"""
    def __init__(
        self, 
        decoder_layer, 
        num_layers,
        norm=None,
    ):
        super().__init__()
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
    
    def forward(self,
                traj_feature,
                noisy_traj_points,
                bev_feature,
                bev_spatial_shape,
                agents_query,
                ego_query,
                time_embed,
                status_encoding,
                global_img=None,
                image_tokens=None,
                lidar_tokens=None,
                history_tokens=None):
        poses_reg_list = []
        poses_cls_list = []
        decorrelation_losses = []
        traj_points = noisy_traj_points
        for mod in self.layers:
            poses_reg, poses_cls, decorrelation_loss = mod(
                traj_feature,
                traj_points,
                bev_feature,
                bev_spatial_shape,
                agents_query,
                ego_query,
                time_embed,
                status_encoding,
                global_img,
                image_tokens=image_tokens,
                lidar_tokens=lidar_tokens,
                history_tokens=history_tokens,
            )
            poses_reg_list.append(poses_reg)
            poses_cls_list.append(poses_cls)
            decorrelation_losses.append(decorrelation_loss)
            traj_points = poses_reg[...,:2].clone().detach()
        return poses_reg_list, poses_cls_list, decorrelation_losses

class TrajectoryScorer(nn.Module):
    """Score trajectory proposals using BEV features."""

    def __init__(self, config: TransfuserConfig, in_bev_dims: Optional[int] = None) -> None:
        super().__init__()
        self._config = config
        in_bev_dims = in_bev_dims or config.tf_d_model
        self._use_decoder = bool(getattr(config, "pdm_score_use_decoder", True))
        self._traj_point_dim = int(getattr(config, "pdm_score_traj_dim", 3))
        self._use_components = bool(getattr(config, "pdm_score_use_components", False))
        self._num_components = 6
        self._value_proj = nn.Sequential(
            nn.Conv2d(in_bev_dims, config.tf_d_model, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self._scene_bev_proj = nn.Conv2d(
            in_bev_dims, config.tf_d_model, kernel_size=1, bias=True
        )
        pool_hw = getattr(config, "pdm_score_bev_pool_hw", (8, 16))
        self._scene_bev_pool = nn.AdaptiveAvgPool2d(pool_hw) if pool_hw else None
        self._traj_point_embed = nn.Linear(self._traj_point_dim, config.tf_d_model)
        self._traj_time_embed = nn.Embedding(
            config.trajectory_sampling.num_poses, config.tf_d_model
        )
        self._traj_token_norm = nn.LayerNorm(config.tf_d_model)
        self._scene_token_norm = nn.LayerNorm(config.tf_d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.tf_d_model,
            nhead=int(getattr(config, "pdm_score_decoder_heads", config.tf_num_head)),
            dim_feedforward=config.tf_d_ffn,
            dropout=float(getattr(config, "pdm_score_decoder_dropout", config.tf_dropout)),
            batch_first=True,
        )
        self._decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=int(getattr(config, "pdm_score_decoder_layers", 2)),
        )
        self._score_trunk = nn.Sequential(
            nn.Linear(config.tf_d_model, config.tf_d_ffn),
            nn.ReLU(inplace=True),
        )
        self._mode_context_fuse_pre = nn.Sequential(
            nn.Linear(config.tf_d_model * 2, config.tf_d_model),
            nn.ReLU(inplace=True),
            nn.LayerNorm(config.tf_d_model),
        )
        self._mode_context_fuse_post = nn.Sequential(
            nn.Linear(config.tf_d_model * 2, config.tf_d_model),
            nn.ReLU(inplace=True),
            nn.LayerNorm(config.tf_d_model),
        )
        self._risk_area_head: Optional[nn.Module] = None
        self._risk_area_enable = bool(
            getattr(config, "pdm_score_risk_area_enable", False)
        ) or float(getattr(config, "pdm_score_risk_area_weight", 0.0)) > 0.0
        if self._risk_area_enable:
            self._risk_area_head = nn.Sequential(
                nn.Linear(config.tf_d_ffn, config.tf_d_ffn),
                nn.ReLU(inplace=True),
                nn.Linear(config.tf_d_ffn, 1),
            )
        if self._use_components:
            self._score_heads = nn.ModuleList(
                [nn.Linear(config.tf_d_ffn, 1) for _ in range(self._num_components)]
            )
            self._score_head = None
        else:
            self._score_head = nn.Linear(config.tf_d_ffn, 1)
            self._score_heads = None

        # Specialist residual score branch (no learned trigger).
        self._score_residual_enable = bool(
            getattr(config, "hardcase_score_residual_enable", False)
        )
        self._score_residual_use_scene_feature = bool(
            getattr(config, "hardcase_score_residual_use_scene_feature", True)
        )
        self._score_residual_use_proposal_feature = bool(
            getattr(config, "hardcase_score_residual_use_proposal_feature", True)
        )
        self._score_residual_use_mode_attention_feature = bool(
            getattr(
                config,
                "hardcase_score_residual_use_mode_attention_feature",
                False,
            )
        )
        self._score_residual_scale = float(
            getattr(config, "hardcase_score_residual_scale", 0.5)
        )
        self._score_residual_lora_enable = bool(
            getattr(config, "hardcase_score_residual_lora_enable", False)
        )
        self._score_residual_lora_num_adapters = max(
            1, int(getattr(config, "hardcase_score_residual_lora_num_adapters", 4))
        )
        self._score_residual_lora_rank = max(
            1, int(getattr(config, "hardcase_score_residual_lora_rank", 8))
        )
        self._score_residual_lora_alpha = float(
            getattr(config, "hardcase_score_residual_lora_alpha", 8.0)
        )
        self._score_residual_lora_dropout = float(
            getattr(config, "hardcase_score_residual_lora_dropout", 0.0)
        )
        self._score_residual_lora_init_scale = float(
            getattr(config, "hardcase_score_residual_lora_init_scale", 1e-3)
        )
        self._score_res_base_proj: Optional[nn.Module] = None
        self._score_res_scene_proj: Optional[nn.Module] = None
        self._score_res_proposal_proj: Optional[nn.Module] = None
        self._score_res_mode_context_proj: Optional[nn.Module] = None
        self._score_residual_head: Optional[nn.Module] = None
        self._score_residual_lora_adapters: Optional[nn.Module] = None
        self._score_res_last_aux: Optional[Dict[str, torch.Tensor]] = None
        if self._score_residual_enable:
            res_hidden = max(
                32, int(getattr(config, "hardcase_score_residual_hidden_dim", 256))
            )
            self._score_res_base_proj = nn.Sequential(
                nn.Linear(config.tf_d_ffn, config.tf_d_model),
                nn.ReLU(inplace=True),
                nn.LayerNorm(config.tf_d_model),
            )
            self._score_res_scene_proj = nn.Sequential(
                nn.Linear(config.tf_d_model, config.tf_d_model),
                nn.ReLU(inplace=True),
                nn.LayerNorm(config.tf_d_model),
            )
            self._score_res_proposal_proj = nn.Sequential(
                nn.Linear(8, config.tf_d_model),
                nn.ReLU(inplace=True),
                nn.LayerNorm(config.tf_d_model),
            )
            self._score_res_mode_context_proj = nn.Sequential(
                nn.Linear(config.tf_d_model, config.tf_d_model),
                nn.ReLU(inplace=True),
                nn.LayerNorm(config.tf_d_model),
            )
            res_in_dim = config.tf_d_model
            if self._score_residual_use_scene_feature:
                res_in_dim += config.tf_d_model
            if self._score_residual_use_proposal_feature:
                res_in_dim += config.tf_d_model
            if self._score_residual_use_mode_attention_feature:
                res_in_dim += config.tf_d_model
            res_out_dim = self._num_components if self._use_components else 1
            if self._score_residual_lora_enable:
                self._score_residual_lora_adapters = ParallelLoRAResidualAdapters(
                    in_dim=res_in_dim,
                    out_dim=res_out_dim,
                    num_adapters=self._score_residual_lora_num_adapters,
                    rank=self._score_residual_lora_rank,
                    alpha=self._score_residual_lora_alpha,
                    dropout=self._score_residual_lora_dropout,
                    init_scale=self._score_residual_lora_init_scale,
                    use_bias=True,
                )
            else:
                self._score_residual_head = nn.Sequential(
                    nn.Linear(res_in_dim, res_hidden),
                    nn.ReLU(inplace=True),
                    nn.Linear(res_hidden, res_out_dim),
                )

    @staticmethod
    def _proposal_geometry_features(coords: torch.Tensor) -> torch.Tensor:
        """
        Build per-proposal geometry features from trajectory coordinates.
        Output shape: (bs, num_modes, 8).
        """
        bs, num_modes, num_poses, dim = coords.shape
        xy = coords[..., :2]
        endpoint = xy[:, :, -1, :]
        endpoint_radius = torch.linalg.norm(endpoint, dim=-1)
        zero = endpoint_radius.new_zeros((bs, num_modes))

        if num_poses > 1:
            dxy = xy[:, :, 1:, :] - xy[:, :, :-1, :]
            step_len = torch.linalg.norm(dxy, dim=-1)
            path_len = step_len.sum(dim=-1)
            mean_step = step_len.mean(dim=-1)
            max_step = step_len.max(dim=-1).values
            seg_heading = torch.atan2(dxy[..., 1], dxy[..., 0])
            if seg_heading.shape[-1] > 1:
                dhead = torch.atan2(
                    torch.sin(seg_heading[..., 1:] - seg_heading[..., :-1]),
                    torch.cos(seg_heading[..., 1:] - seg_heading[..., :-1]),
                )
                abs_heading_change = dhead.abs().sum(dim=-1)
                heading_std = dhead.std(dim=-1, unbiased=False)
            else:
                abs_heading_change = zero
                heading_std = zero
        else:
            path_len = zero
            mean_step = zero
            max_step = zero
            abs_heading_change = zero
            heading_std = zero

        if dim > 2:
            heading_span = torch.atan2(
                torch.sin(coords[:, :, -1, 2] - coords[:, :, 0, 2]),
                torch.cos(coords[:, :, -1, 2] - coords[:, :, 0, 2]),
            ).abs()
        else:
            heading_span = zero

        feat = torch.stack(
            [
                endpoint[..., 0],
                endpoint[..., 1],
                path_len,
                mean_step,
                max_step,
                endpoint_radius,
                abs_heading_change,
                heading_span + heading_std,
            ],
            dim=-1,
        )
        return torch.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)

    def pop_last_hardcase_aux(self) -> Optional[Dict[str, torch.Tensor]]:
        aux = self._score_res_last_aux
        self._score_res_last_aux = None
        return aux

    def forward(
        self,
        traj_points: torch.Tensor,
        bev_feature: torch.Tensor,
        image_tokens: Optional[torch.Tensor] = None,
        lidar_tokens: Optional[torch.Tensor] = None,
        agent_tokens: Optional[torch.Tensor] = None,
        mode_context: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Args:
            traj_points: (bs, num_modes, num_poses, 3) or (bs, num_modes, num_poses, 2)
            bev_feature: (bs, C, H, W)
        Returns:
            scores: (bs, num_modes) or (bs, num_modes, 6)
        """
        coords = traj_points
        if coords.shape[-1] < self._traj_point_dim:
            pad = torch.zeros(
                (*coords.shape[:-1], self._traj_point_dim - coords.shape[-1]),
                device=coords.device,
                dtype=coords.dtype,
            )
            coords = torch.cat([coords, pad], dim=-1)
        elif coords.shape[-1] > self._traj_point_dim:
            coords = coords[..., : self._traj_point_dim]
        bs, num_modes, num_poses, _ = coords.shape
        inject_pre = bool(
            getattr(self._config, "mode_bev_attention_use_scorer_pre", False)
        )
        inject_post = bool(
            getattr(self._config, "mode_bev_attention_use_scorer_post", True)
        )
        context = None
        if (
            mode_context is not None
            and mode_context.dim() == 3
            and mode_context.shape[0] == bs
            and mode_context.shape[1] == num_modes
        ):
            context = mode_context.to(
                device=coords.device,
                dtype=coords.dtype,
            )

        if not self._use_decoder:
            grid_coords = coords[..., :2]
            norm = grid_coords.clone()
            norm[..., 0] = norm[..., 0] / self._config.lidar_max_y
            norm[..., 1] = norm[..., 1] / self._config.lidar_max_x
            norm = norm[..., [1, 0]]  # swap to (x, y) for grid_sample
            grid = norm.view(bs * num_modes, num_poses, 1, 2)
            value = self._value_proj(bev_feature)
            value = value.unsqueeze(1).expand(-1, num_modes, -1, -1, -1)
            value = value.reshape(
                bs * num_modes, value.shape[2], value.shape[3], value.shape[4]
            )
            sampled = torch.nn.functional.grid_sample(
                value,
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
            pooled = sampled.squeeze(-1).mean(dim=-1)
            traj_tokens = pooled.view(bs, num_modes, -1)
            if inject_pre and context is not None:
                traj_tokens = self._mode_context_fuse_pre(
                    torch.cat(
                        [
                            traj_tokens,
                            context.to(
                                device=traj_tokens.device,
                                dtype=traj_tokens.dtype,
                            ),
                        ],
                        dim=-1,
                    )
                )
        else:
            point_embed = self._traj_point_embed(coords)
            time_idx = torch.arange(num_poses, device=coords.device)
            time_embed = self._traj_time_embed(time_idx)[None, None, :, :]
            traj_tokens = self._traj_token_norm((point_embed + time_embed).mean(dim=2))
            if inject_pre and context is not None:
                traj_tokens = self._mode_context_fuse_pre(
                    torch.cat(
                        [
                            traj_tokens,
                            context.to(
                                device=traj_tokens.device,
                                dtype=traj_tokens.dtype,
                            ),
                        ],
                        dim=-1,
                    )
                )

            scene_tokens = []
            if image_tokens is not None:
                scene_tokens.append(image_tokens)
            if lidar_tokens is not None:
                scene_tokens.append(lidar_tokens)
            if agent_tokens is not None:
                scene_tokens.append(agent_tokens)
            if scene_tokens:
                scene_tokens = torch.cat(scene_tokens, dim=1)
            else:
                bev_tokens = self._scene_bev_proj(bev_feature)
                if self._scene_bev_pool is not None:
                    bev_tokens = self._scene_bev_pool(bev_tokens)
                scene_tokens = bev_tokens.flatten(2).transpose(1, 2).contiguous()
            scene_tokens = self._scene_token_norm(scene_tokens)
            traj_tokens = self._decoder(traj_tokens, scene_tokens)

        if context is not None and inject_post:
            traj_tokens = self._mode_context_fuse_post(
                torch.cat(
                    [
                        traj_tokens,
                        context.to(
                            device=traj_tokens.device,
                            dtype=traj_tokens.dtype,
                        ),
                    ],
                    dim=-1,
                )
            )

        features = self._score_trunk(traj_tokens)
        risk_area_logits = None
        if self._risk_area_head is not None:
            risk_area_logits = self._risk_area_head(features).view(bs, num_modes, 1)
        if self._use_components and self._score_heads is not None:
            scores = torch.cat([head(features) for head in self._score_heads], dim=-1)
            scores = scores.view(bs, num_modes, self._num_components)
        else:
            if self._score_head is None:
                raise RuntimeError("TrajectoryScorer missing score head")
            scores = self._score_head(features).view(bs, num_modes)

        hard_aux: Optional[Dict[str, torch.Tensor]] = None
        if (
            self._score_residual_enable
            and self._score_res_base_proj is not None
            and (
                self._score_residual_head is not None
                or self._score_residual_lora_adapters is not None
            )
        ):
            res_inputs = [self._score_res_base_proj(features)]
            if self._score_residual_use_scene_feature and self._score_res_scene_proj is not None:
                scene_feat = self._scene_bev_proj(bev_feature).mean(dim=(2, 3))
                scene_feat = self._score_res_scene_proj(scene_feat)
                scene_feat = scene_feat[:, None, :].expand(-1, num_modes, -1)
                res_inputs.append(scene_feat)
            if (
                self._score_residual_use_proposal_feature
                and self._score_res_proposal_proj is not None
            ):
                proposal_feat = self._score_res_proposal_proj(
                    self._proposal_geometry_features(coords)
                )
                res_inputs.append(proposal_feat)
            if (
                self._score_residual_use_mode_attention_feature
                and self._score_res_mode_context_proj is not None
            ):
                if context is not None:
                    mode_context_feat = context.to(
                        device=features.device,
                        dtype=features.dtype,
                    )
                else:
                    mode_context_feat = torch.zeros(
                        (bs, num_modes, self._config.tf_d_model),
                        device=features.device,
                        dtype=features.dtype,
                    )
                mode_context_feat = self._score_res_mode_context_proj(mode_context_feat)
                res_inputs.append(mode_context_feat)
            res_feat = torch.cat(res_inputs, dim=-1)
            residual_var: Optional[torch.Tensor] = None
            adapter_specialist_scores: Optional[torch.Tensor] = None
            if self._score_residual_lora_adapters is not None:
                adapter_residual = (
                    self._score_residual_lora_adapters(res_feat) * self._score_residual_scale
                )
                residual = adapter_residual.mean(dim=-2)
                residual_var = adapter_residual.var(dim=-2, unbiased=False)
                if self._use_components:
                    adapter_specialist_scores = scores.unsqueeze(-2) + adapter_residual
                else:
                    adapter_specialist_scores = (
                        scores.unsqueeze(-1) + adapter_residual.squeeze(-1)
                    )
            else:
                residual = self._score_residual_head(res_feat) * self._score_residual_scale

            specialist_uncertainty: Optional[torch.Tensor] = None
            if self._use_components:
                scores_specialist = scores + residual
                residual_l1 = residual.abs().mean(dim=-1)
                if residual_var is not None:
                    specialist_uncertainty = residual_var.mean(dim=-1)
            else:
                residual = residual.squeeze(-1)
                scores_specialist = scores + residual
                residual_l1 = residual.abs()
                if residual_var is not None:
                    specialist_uncertainty = residual_var.squeeze(-1)
            hard_aux = {
                "pdm_score_generalist": scores,
                "pdm_score_specialist": scores_specialist,
                "hardcase_score_residual": residual,
                "hardcase_score_residual_l1": residual_l1,
                "hardcase_score_residual_adapter_var": residual_var,
                "hardcase_score_residual_adapter_outputs": (
                    adapter_residual if self._score_residual_lora_adapters is not None else None
                ),
                "pdm_score_specialist_adapter_scores": adapter_specialist_scores,
                "pdm_score_specialist_uncertainty": specialist_uncertainty,
            }
        if risk_area_logits is not None:
            if hard_aux is None:
                hard_aux = {}
            hard_aux["pdm_score_risk_area_logits"] = risk_area_logits

        self._score_res_last_aux = hard_aux
        return scores, hard_aux

class TrajectoryHead(nn.Module):
    """Trajectory prediction head. 扩散模型轨迹预测头"""

    def __init__(self, num_poses: int, d_ffn: int, d_model: int, plan_anchor_path: str,config: TransfuserConfig):
        """
        Initializes trajectory head.
        :param num_poses: number of (x,y,θ) poses to predict
        :param d_ffn: dimensionality of feed-forward network
        :param d_model: input dimensionality
        """
        super(TrajectoryHead, self).__init__()

        self._num_poses = num_poses
        self._d_model = d_model
        self._d_ffn = d_ffn
        self._config = config
        self.diff_loss_weight = 2.0
        self.ego_fut_mode = config.ego_fut_mode

        self.diffusion_scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_schedule="scaled_linear",
            prediction_type="sample",
        )

        self._pdm_score_head = None
        use_pdm_head = (
            getattr(config, "pdm_score_use_head", False)
            or getattr(config, "pdm_score_use_for_selection", False)
            or float(getattr(config, "pdm_score_weight", 0.0)) > 0.0
            or getattr(config, "pdm_score_use_components", False)
        )
        if use_pdm_head:
            self._pdm_score_head = TrajectoryScorer(config, in_bev_dims=config.tf_d_model)
        if getattr(config, "pdm_score_use_components", False):
            self.register_buffer(
                "_pdm_component_weights",
                torch.tensor(config.pdm_score_component_weights, dtype=torch.float32),
                persistent=False,
            )
        else:
            self._pdm_component_weights = None


        plan_anchor = np.load(plan_anchor_path)
        plan_anchor = self._resize_plan_anchor(plan_anchor, self.ego_fut_mode)
        plan_anchor = self._resize_plan_anchor_steps(plan_anchor, self._num_poses)

        self.plan_anchor = nn.Parameter(
            torch.tensor(plan_anchor, dtype=torch.float32),
            requires_grad=False,
        ) # [modes, poses, 2] 可学习的计划锚点
        self.plan_anchor_encoder = nn.Sequential(
            *linear_relu_ln(d_model, 1, 1, self._num_poses * 64), # Linear(num_poses * 64, d_model) -> ReLU -> LayerNorm(d_model)
            nn.Linear(d_model, d_model),
        )
        # Learned proposal tokens (DrivoR-style): anchor + learned_mode + ego.
        self._drivor_mode_token = nn.Embedding(self.ego_fut_mode, d_model)
        nn.init.normal_(self._drivor_mode_token.weight, mean=0.0, std=1e-2)
        self._drivor_token_norm = nn.LayerNorm(d_model)
        self._anchor_contrastive_anchor_proj = nn.Sequential(
            nn.Linear(self._num_poses * 2, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
        )
        self._anchor_contrastive_gt_proj = nn.Sequential(
            nn.Linear(self._num_poses * 2, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
        )
        self._anchor_gaussian_context = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )
        self._anchor_gaussian_mode_embed = nn.Embedding(self.ego_fut_mode, d_model)
        nn.init.normal_(self._anchor_gaussian_mode_embed.weight, mean=0.0, std=1e-2)
        self._anchor_gaussian_mu_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, self._num_poses * 2),
        )
        self._anchor_gaussian_logstd_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, self._num_poses * 2),
        )
        self.time_mlp = None
        if config.denoise_use_time_embed:
            self.time_mlp = nn.Sequential(
                SinusoidalPosEmb(d_model),
                nn.Linear(d_model, d_model * 4),
                nn.Mish(),
                nn.Linear(d_model * 4, d_model),
            )

        diff_decoder_layer = CustomTransformerDecoderLayer(     # 定义一个注意力解码层
            num_poses=num_poses,
            d_model=d_model,
            d_ffn=d_ffn,
            config=config,
        )
        diff_decoder_layers = max(int(getattr(config, "diff_decoder_layers", 2)), 1)
        self.diff_decoder = CustomTransformerDecoder(diff_decoder_layer, diff_decoder_layers)     # 复制N个注意力解码层
        refine_hidden_dim = max(1, int(getattr(config, "refine_hidden_dim", d_model)))
        self._refine_token_fuse = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(inplace=True),
            nn.LayerNorm(d_model),
        )
        self._refine_token_fuse_with_mode_attn: Optional[nn.Module] = None
        self._mode_bev_attn_bev_proj: Optional[nn.Module] = None
        self._mode_bev_attn_key_norm: Optional[nn.Module] = None
        self._mode_bev_attn_query_fuse: Optional[nn.Module] = None
        if bool(getattr(config, "mode_bev_attention_enable", False)):
            self._mode_bev_attn_bev_proj = nn.Conv2d(
                config.tf_d_model, d_model, kernel_size=1, bias=True
            )
            self._mode_bev_attn_key_norm = nn.LayerNorm(d_model)
            self._mode_bev_attn_query_fuse = nn.Sequential(
                nn.Linear(d_model * 3, d_model),
                nn.ReLU(inplace=True),
                nn.LayerNorm(d_model),
            )
            self._refine_token_fuse_with_mode_attn = nn.Sequential(
                nn.Linear(d_model * 3, d_model),
                nn.ReLU(inplace=True),
                nn.LayerNorm(d_model),
            )
        self._refine_delta_head = nn.Sequential(
            nn.Linear(d_model, refine_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(refine_hidden_dim, self._num_poses * 3),
        )
        self._hardcase_specialist_enable = bool(
            getattr(config, "hardcase_specialist_enable", False)
        )
        self._hardcase_specialist_num_heads = max(
            1, int(getattr(config, "hardcase_specialist_num_heads", 2))
        )
        self._hardcase_specialist_hidden_dim = max(
            32, int(getattr(config, "hardcase_specialist_hidden_dim", 256))
        )
        self._hardcase_specialist_use_status = bool(
            getattr(config, "hardcase_specialist_use_status", True)
        )
        self._hardcase_specialist_delta_scale = float(
            getattr(config, "hardcase_specialist_delta_scale", 0.5)
        )
        self._hardcase_specialist_delta_clip_xy = float(
            getattr(config, "hardcase_specialist_delta_clip_xy", 1.0)
        )
        self._hardcase_specialist_delta_clip_heading = float(
            getattr(config, "hardcase_specialist_delta_clip_heading", 0.1)
        )
        self._hardcase_specialist_trunk: Optional[nn.Module] = None
        self._hardcase_specialist_status_proj: Optional[nn.Module] = None
        self._hardcase_specialist_mix_head: Optional[nn.Module] = None
        self._hardcase_specialist_heads: Optional[nn.ModuleList] = None
        if self._hardcase_specialist_enable:
            specialist_in_dim = self._num_poses * 3
            self._hardcase_specialist_trunk = nn.Sequential(
                nn.Linear(specialist_in_dim, self._hardcase_specialist_hidden_dim),
                nn.ReLU(inplace=True),
                nn.LayerNorm(self._hardcase_specialist_hidden_dim),
            )
            specialist_feat_dim = self._hardcase_specialist_hidden_dim
            if self._hardcase_specialist_use_status:
                self._hardcase_specialist_status_proj = nn.Sequential(
                    nn.Linear(d_model, self._hardcase_specialist_hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.LayerNorm(self._hardcase_specialist_hidden_dim),
                )
                specialist_feat_dim += self._hardcase_specialist_hidden_dim
            self._hardcase_specialist_mix_head = nn.Linear(
                specialist_feat_dim, self._hardcase_specialist_num_heads
            )
            self._hardcase_specialist_heads = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(specialist_feat_dim, self._hardcase_specialist_hidden_dim),
                        nn.ReLU(inplace=True),
                        nn.Linear(self._hardcase_specialist_hidden_dim, self._num_poses * 3),
                    )
                    for _ in range(self._hardcase_specialist_num_heads)
                ]
            )

        self.loss_computer = LossComputer(config)
        self._anchor_vis_forward_count = 0
        self._anchor_vis_saved_count = 0

    def _build_pdm_component_vector(
        self,
        values: Union[torch.Tensor, Tuple[float, ...], List[float]],
        like: torch.Tensor,
        default_fill: float = 1.0,
    ) -> torch.Tensor:
        if isinstance(values, torch.Tensor):
            vec = values.to(device=like.device, dtype=like.dtype)
        else:
            vec = torch.tensor(values, device=like.device, dtype=like.dtype)
        if vec.numel() != like.shape[-1]:
            vec = torch.full(
                (like.shape[-1],),
                float(default_fill),
                device=like.device,
                dtype=like.dtype,
            )
        return vec

    def _aggregate_pdm_components_impl(
        self,
        pdm_score_components: torch.Tensor,
        weights: torch.Tensor,
        use_logsigmoid: bool,
        logit_temp: float = 1.0,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        logits = pdm_score_components
        temp = max(float(logit_temp), 1e-6)
        if abs(temp - 1.0) > 1e-6:
            logits = logits / temp
        if bias is not None:
            logits = logits + bias.view(1, 1, -1)

        if not use_logsigmoid:
            return (logits * weights.view(1, 1, -1)).sum(dim=-1)

        # DrivoR-style multiplicative aggregation in log space.
        # Component order: [noc, dac, progress, ttc, comfort, ddc]
        w = weights.clamp_min(0.0)
        noc = logits[..., 0]
        dac = logits[..., 1]
        progress = logits[..., 2]
        ttc = logits[..., 3]
        comfort = logits[..., 4]
        ddc = logits[..., 5]

        log_score = (
            w[0] * F.logsigmoid(noc)
            + w[1] * F.logsigmoid(dac)
            + w[5] * F.logsigmoid(ddc)
        )
        linear = (
            w[3] * torch.sigmoid(ttc)
            + w[2] * torch.sigmoid(progress)
            + w[4] * torch.sigmoid(comfort)
        )
        linear = linear.clamp_min(1e-6)
        return log_score + torch.log(linear)

    def _aggregate_pdm_components(self, pdm_score_components: torch.Tensor) -> torch.Tensor:
        weights = getattr(self, "_pdm_component_weights", None)
        if weights is None:
            weights = torch.ones(
                pdm_score_components.shape[-1],
                device=pdm_score_components.device,
                dtype=pdm_score_components.dtype,
            )
        weights = self._build_pdm_component_vector(
            values=weights,
            like=pdm_score_components,
            default_fill=1.0,
        )
        use_logsigmoid = bool(
            getattr(self._config, "pdm_score_use_logsigmoid_aggregate", False)
        )
        return self._aggregate_pdm_components_impl(
            pdm_score_components=pdm_score_components,
            weights=weights,
            use_logsigmoid=use_logsigmoid,
            logit_temp=1.0,
            bias=None,
        )

    def _aggregate_pdm_components_inference(
        self, pdm_score_components: torch.Tensor
    ) -> torch.Tensor:
        if not bool(getattr(self._config, "inference_pdm_postprocess_enable", False)):
            return self._aggregate_pdm_components(pdm_score_components)
        weights = self._build_pdm_component_vector(
            values=getattr(
                self._config,
                "inference_pdm_component_weights",
                (1.0,) * pdm_score_components.shape[-1],
            ),
            like=pdm_score_components,
            default_fill=1.0,
        )
        bias = self._build_pdm_component_vector(
            values=getattr(
                self._config,
                "inference_pdm_component_bias",
                (0.0,) * pdm_score_components.shape[-1],
            ),
            like=pdm_score_components,
            default_fill=0.0,
        )
        use_logsigmoid = bool(
            getattr(
                self._config,
                "inference_pdm_use_logsigmoid_aggregate",
                getattr(self._config, "pdm_score_use_logsigmoid_aggregate", False),
            )
        )
        logit_temp = float(
            getattr(self._config, "inference_pdm_logit_temperature", 1.0)
        )
        return self._aggregate_pdm_components_impl(
            pdm_score_components=pdm_score_components,
            weights=weights,
            use_logsigmoid=use_logsigmoid,
            logit_temp=logit_temp,
            bias=bias,
        )

    @staticmethod
    def _resize_plan_anchor(plan_anchor: np.ndarray, target_modes: int) -> np.ndarray:
        if plan_anchor.ndim != 3:
            return plan_anchor
        current_modes = plan_anchor.shape[0]
        if current_modes == target_modes:
            return plan_anchor
        if current_modes > target_modes:
            return plan_anchor[:target_modes]
        reps = int(np.ceil(target_modes / current_modes))
        return np.tile(plan_anchor, (reps, 1, 1))[:target_modes]

    @staticmethod
    def _resize_plan_anchor_steps(plan_anchor: np.ndarray, target_steps: int) -> np.ndarray:
        if plan_anchor.ndim != 3:
            return plan_anchor
        current_steps = plan_anchor.shape[1]
        if current_steps == target_steps:
            return plan_anchor
        if current_steps > target_steps:
            return plan_anchor[:, :target_steps]
        if current_steps <= 0:
            return plan_anchor
        pad = np.repeat(plan_anchor[:, -1:, :], target_steps - current_steps, axis=1)
        return np.concatenate([plan_anchor, pad], axis=1)

    def norm_odo(self, odo_info_fut):
        odo_info_fut_x = odo_info_fut[..., 0:1]
        odo_info_fut_y = odo_info_fut[..., 1:2]
        odo_info_fut_head = odo_info_fut[..., 2:3]

        odo_info_fut_x = 2*(odo_info_fut_x + 1.2)/56.9 -1
        odo_info_fut_y = 2*(odo_info_fut_y + 20)/46 -1
        odo_info_fut_head = 2*(odo_info_fut_head + 2)/3.9 -1
        return torch.cat([odo_info_fut_x, odo_info_fut_y, odo_info_fut_head], dim=-1)
    def denorm_odo(self, odo_info_fut):
        odo_info_fut_x = odo_info_fut[..., 0:1]
        odo_info_fut_y = odo_info_fut[..., 1:2]
        odo_info_fut_head = odo_info_fut[..., 2:3]

        odo_info_fut_x = (odo_info_fut_x + 1)/2 * 56.9 - 1.2
        odo_info_fut_y = (odo_info_fut_y + 1)/2 * 46 - 20
        odo_info_fut_head = (odo_info_fut_head + 1)/2 * 3.9 - 2
        return torch.cat([odo_info_fut_x, odo_info_fut_y, odo_info_fut_head], dim=-1)

    def _build_traj_tokens(
        self,
        noisy_traj_points: torch.Tensor,
        status_encoding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Build trajectory tokens from geometry + learned mode + optional ego token."""
        bs, ego_fut_mode = noisy_traj_points.shape[:2]
        traj_pos_embed = gen_sineembed_for_position(noisy_traj_points, hidden_dim=64)
        traj_pos_embed = traj_pos_embed.flatten(-2)
        anchor_tokens = self.plan_anchor_encoder(traj_pos_embed).view(bs, ego_fut_mode, -1)

        if not bool(getattr(self._config, "drivor_token_init_enable", True)):
            return anchor_tokens

        disable_anchor = bool(
            getattr(self._config, "drivor_token_init_disable_anchor", False)
        )
        use_anchor = bool(getattr(self._config, "drivor_token_init_use_anchor", True))
        if disable_anchor:
            use_anchor = False
        use_mode = bool(
            getattr(self._config, "drivor_token_init_use_learned_mode", True)
        )
        use_ego = bool(getattr(self._config, "drivor_token_init_use_ego", True))
        anchor_w = float(getattr(self._config, "drivor_token_anchor_weight", 1.0))
        mode_w = float(getattr(self._config, "drivor_token_mode_weight", 1.0))
        ego_w = float(getattr(self._config, "drivor_token_ego_weight", 1.0))

        traj_tokens = torch.zeros_like(anchor_tokens)
        if use_anchor and anchor_w != 0.0:
            traj_tokens = traj_tokens + anchor_w * anchor_tokens

        if use_mode and mode_w != 0.0:
            mode_tokens = self._drivor_mode_token.weight
            if mode_tokens.shape[0] < ego_fut_mode:
                reps = int(np.ceil(ego_fut_mode / max(mode_tokens.shape[0], 1)))
                mode_tokens = mode_tokens.repeat(reps, 1)
            mode_tokens = mode_tokens[:ego_fut_mode].to(
                device=anchor_tokens.device, dtype=anchor_tokens.dtype
            )
            traj_tokens = traj_tokens + mode_w * mode_tokens[None, ...]

        if use_ego and ego_w != 0.0 and status_encoding is not None:
            if status_encoding.dim() == 3:
                ego_token = status_encoding[:, :1, :]
            elif status_encoding.dim() == 2:
                ego_token = status_encoding[:, None, :]
            else:
                ego_token = None
            if ego_token is not None:
                ego_token = ego_token.to(
                    device=anchor_tokens.device, dtype=anchor_tokens.dtype
                )
                traj_tokens = traj_tokens + ego_w * ego_token.expand(
                    -1, ego_fut_mode, -1
                )

        if not (use_anchor or use_mode or use_ego):
            if disable_anchor:
                traj_tokens = torch.zeros_like(anchor_tokens)
            else:
                traj_tokens = anchor_tokens

        return self._drivor_token_norm(traj_tokens)

    def _trajectory_xy_to_grid(self, traj_xy: torch.Tensor) -> torch.Tensor:
        norm = traj_xy.clone()
        denom_y = max(abs(float(self._config.lidar_max_y)), 1e-6)
        denom_x = max(abs(float(self._config.lidar_max_x)), 1e-6)
        norm[..., 0] = norm[..., 0] / denom_y
        norm[..., 1] = norm[..., 1] / denom_x
        norm = norm[..., [1, 0]]
        return norm

    def _sample_bev_along_trajectory(
        self, bev_feature: torch.Tensor, traj_xy: torch.Tensor
    ) -> Optional[torch.Tensor]:
        if bev_feature is None or traj_xy is None:
            return None
        if bev_feature.dim() != 4 or traj_xy.dim() != 4 or traj_xy.shape[-1] < 2:
            return None
        bs, num_modes, num_steps, _ = traj_xy.shape
        grid = self._trajectory_xy_to_grid(traj_xy[..., :2]).view(
            bs * num_modes, num_steps, 1, 2
        )
        value = bev_feature.unsqueeze(1).expand(-1, num_modes, -1, -1, -1)
        value = value.reshape(
            bs * num_modes, bev_feature.shape[1], bev_feature.shape[2], bev_feature.shape[3]
        )
        sampled = F.grid_sample(
            value,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        return sampled.view(bs, num_modes, bev_feature.shape[1], num_steps)

    def _compute_mode_bev_attention(
        self,
        poses_reg: Optional[torch.Tensor],
        bev_feature: Optional[torch.Tensor],
        status_encoding: Optional[torch.Tensor] = None,
        risk_bev_map: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not bool(getattr(self._config, "mode_bev_attention_enable", False)):
            return None, None
        if (
            poses_reg is None
            or bev_feature is None
            or self._mode_bev_attn_bev_proj is None
            or self._mode_bev_attn_key_norm is None
            or self._mode_bev_attn_query_fuse is None
        ):
            return None, None
        if poses_reg.dim() != 4 or bev_feature.dim() != 4 or poses_reg.shape[-1] < 2:
            return None, None

        bs, num_modes = poses_reg.shape[:2]
        traj_tokens = self._build_traj_tokens(
            noisy_traj_points=poses_reg,
            status_encoding=None,
        )

        ego_token = None
        if status_encoding is not None:
            if status_encoding.dim() == 3:
                ego_token = status_encoding[:, :1, :]
            elif status_encoding.dim() == 2:
                ego_token = status_encoding[:, None, :]
        if ego_token is None:
            ego_token = torch.zeros_like(traj_tokens)
        else:
            ego_token = ego_token.to(
                device=traj_tokens.device,
                dtype=traj_tokens.dtype,
            ).expand(-1, num_modes, -1)

        bev_proj = self._mode_bev_attn_bev_proj(
            bev_feature.to(device=traj_tokens.device, dtype=traj_tokens.dtype)
        )
        sampled = self._sample_bev_along_trajectory(
            bev_feature=bev_proj,
            traj_xy=poses_reg[..., :2].to(device=traj_tokens.device, dtype=traj_tokens.dtype),
        )
        if sampled is None:
            return None, None
        sampled_pool = sampled.mean(dim=-1)

        query = self._mode_bev_attn_query_fuse(
            torch.cat([traj_tokens, sampled_pool, ego_token], dim=-1)
        )
        key_tokens = bev_proj.flatten(2).transpose(1, 2).contiguous()
        key_tokens = self._mode_bev_attn_key_norm(key_tokens)
        scale = math.sqrt(max(float(query.shape[-1]), 1.0))
        logits = torch.einsum("bmd,bnd->bmn", query, key_tokens) / scale
        if (
            bool(getattr(self._config, "mode_bev_attention_risk_enable", False))
            and isinstance(risk_bev_map, torch.Tensor)
        ):
            if risk_bev_map.dim() == 4:
                risk_bev_map = risk_bev_map.squeeze(1)
            if risk_bev_map.dim() == 3:
                risk_bev_map = risk_bev_map.to(
                    device=logits.device, dtype=logits.dtype
                )
                if tuple(risk_bev_map.shape[-2:]) != tuple(bev_proj.shape[-2:]):
                    risk_bev_map = F.interpolate(
                        risk_bev_map.unsqueeze(1),
                        size=tuple(bev_proj.shape[-2:]),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(1)
                risk_flat = risk_bev_map.flatten(1)
                risk_flat = risk_flat - risk_flat.mean(dim=-1, keepdim=True)
                risk_bias_weight = float(
                    getattr(
                        self._config,
                        "mode_bev_attention_risk_bias_weight",
                        2.0,
                    )
                )
                logits = logits + risk_bias_weight * risk_flat[:, None, :]
        temp = max(
            float(getattr(self._config, "mode_bev_attention_temperature", 1.0)),
            1e-6,
        )
        attn_flat = torch.softmax(logits / temp, dim=-1)
        mode_context = torch.einsum("bmn,bnd->bmd", attn_flat, key_tokens)
        attn_map = attn_flat.view(bs, num_modes, bev_proj.shape[2], bev_proj.shape[3])
        return attn_map, mode_context

    def _compute_mode_to_gt_distance(
        self, poses_reg: torch.Tensor, target_traj: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, bool]:
        zero = torch.zeros((), device=poses_reg.device, dtype=poses_reg.dtype)
        if target_traj is None:
            return zero, False
        if target_traj.dim() == 2:
            target_traj = target_traj.unsqueeze(0)
        if target_traj.dim() != 3 or target_traj.shape[-1] < 2:
            return zero, False
        target_xy = target_traj[..., :2].to(
            device=poses_reg.device,
            dtype=poses_reg.dtype,
        )
        overlap_steps = min(poses_reg.shape[2], target_xy.shape[1])
        if overlap_steps <= 0:
            return zero, False
        valid_target = torch.isfinite(target_xy[:, :overlap_steps]).all(dim=(-1, -2))
        if not valid_target.any():
            return zero, False
        dist = torch.linalg.norm(
            poses_reg[:, :, :overlap_steps, :2] - target_xy[:, None, :overlap_steps, :],
            dim=-1,
        ).mean(dim=-1)
        min_dist = dist.min(dim=-1).values
        min_dist = min_dist[valid_target]
        finite = torch.isfinite(min_dist)
        if not finite.any():
            return zero, False
        return min_dist[finite].mean(), True

    def _compute_mode_to_candidate_distance(
        self,
        poses_reg: torch.Tensor,
        targets: Optional[Dict[str, torch.Tensor]],
    ) -> Tuple[torch.Tensor, bool]:
        zero = torch.zeros((), device=poses_reg.device, dtype=poses_reg.dtype)
        if targets is None:
            return zero, False

        batch_size = poses_reg.shape[0]
        cand_xy, cand_valid = self._collect_anchor_candidate_targets(
            targets=targets,
            batch_size=batch_size,
            device=poses_reg.device,
            dtype=poses_reg.dtype,
            use_score_filter=bool(
                getattr(self._config, "refine_aux_candidate_use_score_filter", True)
            ),
        )
        if cand_xy is None or cand_valid is None or not cand_valid.any():
            return zero, False

        pred_xy = poses_reg[..., :2]
        dist = torch.linalg.norm(
            pred_xy[:, :, None, :, :] - cand_xy[:, None, :, :, :],
            dim=-1,
        ).mean(dim=-1)
        dist = dist.masked_fill(~cand_valid[:, None, :], float("inf"))
        mode_min = dist.min(dim=-1).values

        # Candidate supervision should not act on GT-matched mode.
        valid_mode = torch.isfinite(mode_min)
        num_modes = poses_reg.shape[1]
        mode_mask = torch.ones(
            (batch_size, num_modes), device=poses_reg.device, dtype=torch.bool
        )
        target_traj = targets.get("trajectory")
        if target_traj is not None and torch.is_tensor(target_traj):
            if target_traj.dim() == 2:
                target_traj = target_traj.unsqueeze(0)
            if target_traj.dim() == 3 and target_traj.shape[-1] >= 2:
                gt_xy = target_traj[..., :2].to(
                    device=poses_reg.device,
                    dtype=poses_reg.dtype,
                )
                overlap_steps = min(pred_xy.shape[2], gt_xy.shape[1])
                if overlap_steps > 0:
                    valid_target = torch.isfinite(gt_xy[:, :overlap_steps]).all(
                        dim=(-1, -2)
                    )
                    if valid_target.any():
                        gt_dist = torch.linalg.norm(
                            pred_xy[:, :, :overlap_steps, :]
                            - gt_xy[:, None, :overlap_steps, :],
                            dim=-1,
                        ).mean(dim=-1)
                        gt_mode_idx = gt_dist.argmin(dim=-1)
                        valid_rows = torch.nonzero(
                            valid_target, as_tuple=False
                        ).view(-1)
                        mode_mask[valid_rows, gt_mode_idx[valid_rows]] = False
        valid_mode = valid_mode & mode_mask
        if not valid_mode.any():
            return zero, False

        sample_loss = (
            mode_min.masked_fill(~valid_mode, 0.0).sum(dim=-1)
            / valid_mode.float().sum(dim=-1).clamp_min(1.0)
        )
        finite_sample = torch.isfinite(sample_loss)
        if not finite_sample.any():
            return zero, False
        return sample_loss[finite_sample].mean(), True

    def _apply_refinement(
        self,
        poses_reg: Optional[torch.Tensor],
        status_encoding: Optional[torch.Tensor] = None,
        bev_feature: Optional[torch.Tensor] = None,
        risk_bev_map: Optional[torch.Tensor] = None,
    ) -> Tuple[
        Optional[torch.Tensor],
        List[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        if poses_reg is None or poses_reg.dim() != 4:
            return poses_reg, [], None, None
        if not bool(getattr(self._config, "refine_enable", False)):
            return poses_reg, [], None, None

        refine_steps = max(1, int(getattr(self._config, "refine_steps", 1)))
        use_ego_context = bool(getattr(self._config, "refine_use_ego_context", True))
        clip_xy = float(getattr(self._config, "refine_delta_clip_xy", 0.0))
        clip_heading = float(getattr(self._config, "refine_delta_clip_heading", 0.0))
        use_mode_bev_attn = (
            bool(getattr(self._config, "mode_bev_attention_enable", False))
            and bool(getattr(self._config, "mode_bev_attention_use_refine", True))
            and bev_feature is not None
            and self._refine_token_fuse_with_mode_attn is not None
        )
        recompute_mode_attn_each_step = bool(
            getattr(self._config, "mode_bev_attention_refine_recompute_each_step", False)
        )

        refined = poses_reg
        refined_steps: List[torch.Tensor] = []
        mode_attn_map: Optional[torch.Tensor] = None
        mode_attn_context: Optional[torch.Tensor] = None
        for _ in range(refine_steps):
            traj_tokens = self._build_traj_tokens(
                noisy_traj_points=refined,
                status_encoding=None,
            )

            ego_token = None
            if use_ego_context and status_encoding is not None:
                if status_encoding.dim() == 3:
                    ego_token = status_encoding[:, :1, :]
                elif status_encoding.dim() == 2:
                    ego_token = status_encoding[:, None, :]
            if ego_token is None:
                ego_token = torch.zeros_like(traj_tokens)
            else:
                ego_token = ego_token.to(
                    device=traj_tokens.device,
                    dtype=traj_tokens.dtype,
                ).expand(-1, refined.shape[1], -1)
            if use_mode_bev_attn and (
                mode_attn_context is None or recompute_mode_attn_each_step
            ):
                mode_attn_map, mode_attn_context = self._compute_mode_bev_attention(
                    poses_reg=refined,
                    bev_feature=bev_feature,
                    status_encoding=status_encoding,
                    risk_bev_map=risk_bev_map,
                )
            if (
                use_mode_bev_attn
                and mode_attn_context is not None
                and mode_attn_context.shape[:2] == traj_tokens.shape[:2]
            ):
                refined_token = self._refine_token_fuse_with_mode_attn(
                    torch.cat([traj_tokens, ego_token, mode_attn_context], dim=-1)
                )
            else:
                refined_token = self._refine_token_fuse(
                    torch.cat([traj_tokens, ego_token], dim=-1)
                )
            delta = self._refine_delta_head(refined_token).view(
                refined.shape[0], refined.shape[1], self._num_poses, 3
            )
            delta_xy = delta[..., :2]
            delta_heading = delta[..., 2:3]
            if clip_xy > 0.0:
                delta_xy = torch.clamp(delta_xy, min=-clip_xy, max=clip_xy)
            if clip_heading > 0.0:
                delta_heading = torch.clamp(
                    delta_heading, min=-clip_heading, max=clip_heading
                )

            next_xy = refined[..., :2] + delta_xy
            next_heading = refined[..., 2:3] + delta_heading
            next_heading = torch.atan2(torch.sin(next_heading), torch.cos(next_heading))
            refined = torch.cat([next_xy, next_heading], dim=-1)
            refined_steps.append(refined)

        return refined, refined_steps, mode_attn_map, mode_attn_context

    def _apply_hardcase_specialist(
        self,
        poses_reg: Optional[torch.Tensor],
        status_encoding: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
        if poses_reg is None or poses_reg.dim() != 4:
            return poses_reg, None
        if not self._hardcase_specialist_enable:
            return poses_reg, None
        if (
            self._hardcase_specialist_trunk is None
            or self._hardcase_specialist_mix_head is None
            or self._hardcase_specialist_heads is None
        ):
            return poses_reg, None

        bs, num_modes, num_steps, state_dim = poses_reg.shape
        if num_steps <= 0 or state_dim <= 0:
            return poses_reg, None

        base = poses_reg[..., : min(state_dim, 3)]
        if base.shape[-1] < 3:
            pad = torch.zeros(
                (*base.shape[:-1], 3 - base.shape[-1]),
                device=base.device,
                dtype=base.dtype,
            )
            base = torch.cat([base, pad], dim=-1)
        base_flat = base.reshape(bs * num_modes, -1)
        base_feat = self._hardcase_specialist_trunk(base_flat).view(
            bs, num_modes, -1
        )

        feature_parts = [base_feat]
        if self._hardcase_specialist_use_status and self._hardcase_specialist_status_proj is not None:
            ego_token = None
            if status_encoding is not None:
                if status_encoding.dim() == 3:
                    ego_token = status_encoding[:, :1, :]
                elif status_encoding.dim() == 2:
                    ego_token = status_encoding[:, None, :]
            if ego_token is None:
                status_feat = torch.zeros_like(base_feat)
            else:
                status_feat = self._hardcase_specialist_status_proj(
                    ego_token.to(device=base_feat.device, dtype=base_feat.dtype)
                )
                status_feat = status_feat.expand(-1, num_modes, -1)
            feature_parts.append(status_feat)

        fused_feat = torch.cat(feature_parts, dim=-1)
        expert_logits = self._hardcase_specialist_mix_head(fused_feat)
        expert_weights = torch.softmax(expert_logits, dim=-1)

        fused_flat = fused_feat.reshape(bs * num_modes, -1)
        expert_delta_list: List[torch.Tensor] = []
        for head in self._hardcase_specialist_heads:
            delta = head(fused_flat).view(bs, num_modes, self._num_poses, 3)
            expert_delta_list.append(delta)
        delta_stack = torch.stack(expert_delta_list, dim=2)
        mixed_delta = (
            expert_weights[..., None, None] * delta_stack
        ).sum(dim=2)
        mixed_delta = mixed_delta * self._hardcase_specialist_delta_scale

        delta_xy = mixed_delta[..., :2]
        delta_heading = mixed_delta[..., 2:3]
        if self._hardcase_specialist_delta_clip_xy > 0.0:
            delta_xy = torch.clamp(
                delta_xy,
                min=-self._hardcase_specialist_delta_clip_xy,
                max=self._hardcase_specialist_delta_clip_xy,
            )
        if self._hardcase_specialist_delta_clip_heading > 0.0:
            delta_heading = torch.clamp(
                delta_heading,
                min=-self._hardcase_specialist_delta_clip_heading,
                max=self._hardcase_specialist_delta_clip_heading,
            )
        mixed_delta = torch.cat([delta_xy, delta_heading], dim=-1)

        corrected_xy = poses_reg[..., :2] + mixed_delta[..., :2]
        if state_dim >= 3:
            corrected_heading = poses_reg[..., 2:3] + mixed_delta[..., 2:3]
            corrected_heading = torch.atan2(
                torch.sin(corrected_heading), torch.cos(corrected_heading)
            )
            corrected = torch.cat([corrected_xy, corrected_heading], dim=-1)
            if state_dim > 3:
                corrected = torch.cat([corrected, poses_reg[..., 3:]], dim=-1)
        else:
            corrected = corrected_xy

        expert_entropy = (
            -(expert_weights.clamp_min(1e-8) * torch.log(expert_weights.clamp_min(1e-8)))
            .sum(dim=-1)
            .mean()
        )
        aux = {
            "hardcase_specialist_delta": mixed_delta,
            "hardcase_specialist_expert_weight": expert_weights,
            "hardcase_specialist_expert_logit": expert_logits,
            "hardcase_specialist_expert_entropy": expert_entropy,
            "hardcase_specialist_delta_l1": mixed_delta.abs().mean(),
        }
        return corrected, aux

    def _diff_input_decorrelation_loss(self, traj_feature: torch.Tensor) -> torch.Tensor:
        if traj_feature.dim() != 3:
            return torch.tensor(0.0, device=traj_feature.device)
        num_modes = traj_feature.shape[1]
        if num_modes <= 1:
            return torch.tensor(0.0, device=traj_feature.device)
        centered = traj_feature - traj_feature.mean(dim=-1, keepdim=True)
        normed = F.normalize(centered, dim=-1, eps=1e-6)
        gram = torch.matmul(normed, normed.transpose(1, 2))
        eye = torch.eye(num_modes, device=traj_feature.device, dtype=traj_feature.dtype)
        off_diag = gram * (1 - eye[None, ...])
        loss = (off_diag ** 2).sum(dim=(1, 2)) / (num_modes * (num_modes - 1))
        return loss.mean()

    @staticmethod
    def _pad_or_trim_trajectory_steps(traj_xy: torch.Tensor, target_steps: int) -> torch.Tensor:
        current_steps = traj_xy.shape[-2]
        if current_steps == target_steps:
            return traj_xy
        if current_steps > target_steps:
            return traj_xy[..., :target_steps, :]
        if current_steps == 0:
            pad_shape = list(traj_xy.shape)
            pad_shape[-2] = target_steps
            return traj_xy.new_zeros(pad_shape)
        pad_shape = list(traj_xy.shape)
        pad_shape[-2] = target_steps - current_steps
        pad = traj_xy[..., -1:, :].expand(*pad_shape)
        return torch.cat([traj_xy, pad], dim=-2)

    def _compute_anchor_contrastive_loss(
        self,
        plan_anchor: torch.Tensor,
        target_traj: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        zero = torch.zeros(
            (),
            device=plan_anchor.device,
            dtype=plan_anchor.dtype,
        )
        if target_traj is None:
            return zero, zero
        if target_traj.dim() == 2:
            target_traj = target_traj.unsqueeze(0)
        if target_traj.dim() != 3 or target_traj.shape[-1] < 2:
            return zero, zero

        gt_xy = target_traj[..., :2].to(
            device=plan_anchor.device,
            dtype=plan_anchor.dtype,
        )
        anchor_xy = plan_anchor[..., :2]
        valid_mask = torch.isfinite(gt_xy).all(dim=(-1, -2))
        if not valid_mask.any():
            return zero, zero
        anchor_xy = anchor_xy[valid_mask]
        gt_xy = gt_xy[valid_mask]

        overlap_steps = min(anchor_xy.shape[2], gt_xy.shape[1], self._num_poses)
        if overlap_steps <= 0:
            return zero, zero
        overlap_anchor = anchor_xy[:, :, :overlap_steps, :]
        overlap_gt = gt_xy[:, :overlap_steps, :]
        dist = torch.linalg.norm(
            overlap_anchor - overlap_gt[:, None, :, :],
            dim=-1,
        ).mean(dim=-1)
        positive_idx = dist.argmin(dim=-1)

        anchor_feat_xy = self._pad_or_trim_trajectory_steps(anchor_xy, self._num_poses)
        gt_feat_xy = self._pad_or_trim_trajectory_steps(gt_xy, self._num_poses)
        anchor_feat = self._anchor_contrastive_anchor_proj(
            anchor_feat_xy.reshape(anchor_feat_xy.shape[0], anchor_feat_xy.shape[1], -1)
        )
        gt_feat = self._anchor_contrastive_gt_proj(
            gt_feat_xy.reshape(gt_feat_xy.shape[0], -1)
        )
        anchor_feat = F.normalize(anchor_feat, dim=-1, eps=1e-6)
        gt_feat = F.normalize(gt_feat, dim=-1, eps=1e-6)

        logits = torch.einsum("bmd,bd->bm", anchor_feat, gt_feat)
        temperature = max(
            float(getattr(self._config, "anchor_contrastive_temperature", 0.07)),
            1e-4,
        )
        loss = F.cross_entropy(logits / temperature, positive_idx)
        positive_distance = dist.gather(1, positive_idx[:, None]).mean()
        return loss, positive_distance

    def _compute_learned_gaussian_supervision(
        self,
        anchor_mu: Optional[torch.Tensor],
        anchor_std: Optional[torch.Tensor],
        target_traj: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        zero = torch.zeros((), device=self.plan_anchor.device, dtype=torch.float32)
        if anchor_mu is None or anchor_std is None or target_traj is None:
            return zero, zero, zero
        if target_traj.dim() == 2:
            target_traj = target_traj.unsqueeze(0)
        if target_traj.dim() != 3 or target_traj.shape[-1] < 2:
            return zero, zero, zero

        gt_xy = target_traj[..., :2].to(
            device=anchor_mu.device,
            dtype=anchor_mu.dtype,
        )
        valid_mask = torch.isfinite(gt_xy).all(dim=(-1, -2))
        if not valid_mask.any():
            return zero.to(anchor_mu.device, anchor_mu.dtype), zero.to(anchor_mu.device, anchor_mu.dtype), zero.to(anchor_mu.device, anchor_mu.dtype)

        mu_xy = anchor_mu[valid_mask]
        std_xy = anchor_std[valid_mask].clamp_min(1e-3)
        gt_xy = gt_xy[valid_mask]

        mu_xy = self._pad_or_trim_trajectory_steps(mu_xy, self._num_poses)
        std_xy = self._pad_or_trim_trajectory_steps(std_xy, self._num_poses).clamp_min(1e-3)
        gt_xy = self._pad_or_trim_trajectory_steps(gt_xy, self._num_poses)

        dist = torch.linalg.norm(
            mu_xy - gt_xy[:, None, :, :],
            dim=-1,
        ).mean(dim=-1)
        pos_idx = dist.argmin(dim=-1)
        batch_idx = torch.arange(mu_xy.shape[0], device=mu_xy.device)
        mu_pos = mu_xy[batch_idx, pos_idx]
        std_pos = std_xy[batch_idx, pos_idx].clamp_min(1e-3)

        sq_term = ((gt_xy - mu_pos) / std_pos) ** 2
        log_term = 2.0 * torch.log(std_pos) + math.log(2.0 * math.pi)
        nll = 0.5 * (sq_term + log_term)
        nll = nll.mean()
        reg = F.smooth_l1_loss(mu_pos, gt_xy)
        pos_dist = dist[batch_idx, pos_idx].mean()
        return nll, reg, pos_dist

    @staticmethod
    def _align_batch_dim(
        tensor: torch.Tensor,
        batch_size: int,
    ) -> Optional[torch.Tensor]:
        if tensor.shape[0] == batch_size:
            return tensor
        if tensor.shape[0] == 1:
            return tensor.expand(batch_size, *tensor.shape[1:])
        if batch_size == 1:
            return tensor[:1]
        return None

    def _collect_anchor_candidate_targets(
        self,
        targets: Optional[Dict[str, torch.Tensor]],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        use_score_filter: Optional[bool] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if targets is None:
            return None, None
        candidates = targets.get("trajectory_candidates")
        if candidates is None or not torch.is_tensor(candidates) or candidates.numel() == 0:
            return None, None
        if candidates.dim() == 3:
            candidates = candidates.unsqueeze(0)
        if candidates.dim() != 4 or candidates.shape[-1] < 2:
            return None, None
        candidates = candidates.to(device=device, dtype=dtype)
        candidates = self._align_batch_dim(candidates, batch_size)
        if candidates is None:
            return None, None

        cand_xy = self._pad_or_trim_trajectory_steps(
            candidates[..., :2], self._num_poses
        )
        valid = torch.isfinite(cand_xy).all(dim=(-1, -2))

        cand_mask = targets.get("trajectory_candidates_mask")
        if cand_mask is not None and torch.is_tensor(cand_mask):
            if cand_mask.dim() == 1:
                cand_mask = cand_mask.unsqueeze(0)
            cand_mask = cand_mask.to(device=device).bool()
            cand_mask = self._align_batch_dim(cand_mask, batch_size)
            if cand_mask is not None and cand_mask.shape == valid.shape:
                valid = valid & cand_mask

        if use_score_filter is None:
            use_score_filter = bool(
                getattr(
                    self._config,
                    "anchor_learned_gaussian_candidate_use_score_filter",
                    True,
                )
            )
        if use_score_filter:
            cand_scores = targets.get("pdm_score_targets")
            gt_score = targets.get("gt_pdm_score")
            if (
                cand_scores is not None
                and gt_score is not None
                and torch.is_tensor(cand_scores)
                and torch.is_tensor(gt_score)
            ):
                if cand_scores.dim() == 1:
                    cand_scores = cand_scores.unsqueeze(0)
                cand_scores = cand_scores.to(device=device, dtype=dtype)
                cand_scores = self._align_batch_dim(cand_scores, batch_size)
                gt_score = gt_score.to(device=device, dtype=dtype).view(-1)
                if gt_score.numel() == 1 and batch_size > 1:
                    gt_score = gt_score.expand(batch_size)
                elif gt_score.numel() != batch_size:
                    gt_score = None
                if cand_scores is not None and gt_score is not None:
                    score_valid = torch.isfinite(cand_scores) & (
                        cand_scores > gt_score[:, None]
                    )
                    if score_valid.shape == valid.shape:
                        valid = valid & score_valid

        return cand_xy, valid

    def _compute_learned_gaussian_candidate_supervision(
        self,
        anchor_mu: Optional[torch.Tensor],
        anchor_std: Optional[torch.Tensor],
        targets: Optional[Dict[str, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if anchor_mu is None or anchor_std is None:
            zero = torch.zeros((), device=self.plan_anchor.device, dtype=torch.float32)
            return zero, zero

        batch_size = anchor_mu.shape[0]
        device = anchor_mu.device
        dtype = anchor_mu.dtype
        zero = torch.zeros((), device=device, dtype=dtype)

        cand_xy, cand_valid = self._collect_anchor_candidate_targets(
            targets=targets,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )
        if cand_xy is None or cand_valid is None or not cand_valid.any():
            return zero, zero

        mu_xy = self._pad_or_trim_trajectory_steps(anchor_mu, self._num_poses)
        std_xy = self._pad_or_trim_trajectory_steps(anchor_std, self._num_poses).clamp_min(
            1e-3
        )

        dist = torch.linalg.norm(
            mu_xy[:, :, None, :, :] - cand_xy[:, None, :, :, :],
            dim=-1,
        ).mean(dim=-1)
        dist = dist.masked_fill(~cand_valid[:, None, :], float("inf"))
        best_mode = dist.argmin(dim=1)

        batch_idx = (
            torch.arange(batch_size, device=device)[:, None].expand_as(best_mode)
        )
        mu_sel = mu_xy[batch_idx, best_mode]
        std_sel = std_xy[batch_idx, best_mode].clamp_min(1e-3)

        sq_term = ((cand_xy - mu_sel) / std_sel) ** 2
        log_term = 2.0 * torch.log(std_sel) + math.log(2.0 * math.pi)
        nll = 0.5 * (sq_term + log_term).mean(dim=(-1, -2))
        valid_nll = cand_valid & torch.isfinite(nll)
        if not valid_nll.any():
            return zero, zero
        candidate_nll = nll[valid_nll].mean()

        dist_best = dist.min(dim=1).values
        valid_dist = cand_valid & torch.isfinite(dist_best)
        if valid_dist.any():
            candidate_dist = dist_best[valid_dist].mean()
        else:
            candidate_dist = zero
        return candidate_nll, candidate_dist

    def _compute_anchor_step_meters(
        self,
        batch_size: int,
        device: torch.device,
        status_feature: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        step_m = torch.full(
            (batch_size,),
            float(self._config.anchor_free_step_meters),
            device=device,
        )
        if self._config.anchor_free_step_use_speed and status_feature is not None:
            if status_feature.dim() == 1:
                status_feature = status_feature.unsqueeze(0)
            if status_feature.shape[-1] >= 6:
                velocity = status_feature[:, 4:6]
                speed = torch.linalg.norm(velocity, dim=-1)
                dt = float(self._config.trajectory_sampling.interval_length)
                step_m = speed * dt
        min_m = float(getattr(self._config, "anchor_free_step_min_m", 0.0))
        max_m = float(getattr(self._config, "anchor_free_step_max_m", 0.0))
        if min_m > 0:
            step_m = step_m.clamp(min=min_m)
        if max_m > 0:
            step_m = step_m.clamp(max=max_m)
        step_m = torch.clamp(step_m, min=float(self._config.bev_pixel_size))
        return step_m

    @staticmethod
    def _take_first_token(token: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if token is None:
            return None
        if token.dim() == 3:
            return token[:, 0, :]
        if token.dim() == 2:
            return token
        return None

    def _build_physics_prior_anchor_mean(
        self,
        mask: torch.Tensor,
        status_feature: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, _, width = mask.shape
        device = mask.device
        num_modes = self.ego_fut_mode
        num_poses = self._num_poses
        pixel_size = float(self._config.bev_pixel_size)

        start_rows, start_cols = self._select_anchor_seeds(mask)
        ego_row = 0
        ego_col = (width - 1) // 2
        start_x = (start_rows.to(torch.float32) - float(ego_row)) * pixel_size
        start_y = (start_cols.to(torch.float32) - float(ego_col)) * pixel_size

        dt = float(self._config.trajectory_sampling.interval_length)
        if dt <= 0.0:
            dt = 0.5
        step_meters = self._compute_anchor_step_meters(
            batch_size=batch_size,
            device=device,
            status_feature=status_feature,
        )

        status = status_feature
        if status is not None:
            if status.dim() == 1:
                status = status.unsqueeze(0)
            if status.shape[0] != batch_size:
                if status.shape[0] == 1:
                    status = status.repeat(batch_size, 1)
                else:
                    status = None

        if status is not None and status.shape[-1] >= 6:
            velocity = status[:, 4:6].to(device=device, dtype=torch.float32)
        else:
            velocity = torch.stack(
                [step_meters / dt, torch.zeros_like(step_meters)],
                dim=-1,
            )
        if status is not None and status.shape[-1] >= 8:
            acceleration = status[:, 6:8].to(device=device, dtype=torch.float32)
        else:
            acceleration = torch.zeros_like(velocity)

        time_idx = torch.arange(1, num_poses + 1, device=device, dtype=torch.float32)
        time_sec = time_idx[None, :, None] * dt
        v = velocity[:, None, :]
        a = acceleration[:, None, :]
        delta = v * time_sec + 0.5 * a * (time_sec**2)

        mu_x = start_x[:, None] + delta[..., 0]
        mu_y = start_y[:, None] + delta[..., 1]
        if self._config.anchor_free_forward_only:
            floor_x = start_x[:, None]
            mu_x = torch.maximum(mu_x, floor_x)
            mu_x = torch.cummax(mu_x, dim=1).values
        mu = torch.stack([mu_x, mu_y], dim=-1)
        return mu[:, None, :, :].repeat(1, num_modes, 1, 1)

    def _build_learned_anchor_context(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        status_encoding: Optional[torch.Tensor] = None,
        ego_query: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        status_token = self._take_first_token(status_encoding)
        ego_token = self._take_first_token(ego_query)

        if status_token is None:
            status_token = torch.zeros(
                (batch_size, self._d_model), device=device, dtype=dtype
            )
        else:
            status_token = status_token.to(device=device, dtype=dtype)
        if ego_token is None:
            ego_token = torch.zeros(
                (batch_size, self._d_model), device=device, dtype=dtype
            )
        else:
            ego_token = ego_token.to(device=device, dtype=dtype)

        if status_token.shape[0] != batch_size:
            if status_token.shape[0] == 1:
                status_token = status_token.repeat(batch_size, 1)
            else:
                status_token = status_token[:batch_size]
        if ego_token.shape[0] != batch_size:
            if ego_token.shape[0] == 1:
                ego_token = ego_token.repeat(batch_size, 1)
            else:
                ego_token = ego_token[:batch_size]

        context = torch.cat([ego_token, status_token], dim=-1)
        return self._anchor_gaussian_context(context)

    def _project_anchor_to_feasible_mask(
        self,
        candidate_xy: torch.Tensor,
        mu_xy: torch.Tensor,
        mask: torch.Tensor,
        relaxed_mask: Optional[torch.Tensor] = None,
        strict_steps: int = 0,
    ) -> torch.Tensor:
        batch_size, num_modes, num_poses, _ = candidate_xy.shape
        _, height, width = mask.shape
        device = mask.device
        pixel_size = float(self._config.bev_pixel_size)
        ego_row = 0
        ego_col = (width - 1) // 2
        pad_rows = int(round(self._config.anchor_free_forward_pad_m / pixel_size))
        if pad_rows < 0:
            pad_rows = 0
        max_row = height - 1 + pad_rows

        start_rows, start_cols = self._select_anchor_seeds(mask)
        current = torch.stack([start_rows, start_cols], dim=-1).to(torch.float32)
        current = current[:, None, :].repeat(1, num_modes, 1)

        traj_pix = torch.empty(
            (batch_size, num_modes, num_poses, 2),
            device=device,
            dtype=torch.float32,
        )
        batch_idx = (
            torch.arange(batch_size, device=device)[:, None].expand(batch_size, num_modes)
        )
        correction_iters = int(
            max(0, getattr(self._config, "anchor_free_gaussian_correction_iters", 2))
        )
        correction_blend = float(
            getattr(self._config, "anchor_free_gaussian_correction_blend", 0.5)
        )
        correction_blend = max(0.0, min(1.0, correction_blend))
        strict_steps = int(strict_steps) if strict_steps else 0

        for step in range(num_poses):
            step_mask = mask
            if relaxed_mask is not None and strict_steps > 0 and step >= strict_steps:
                step_mask = relaxed_mask

            step_xy = candidate_xy[:, :, step, :]
            mu_step = mu_xy[:, :, step, :]
            cand_rows = (
                step_xy[..., 0] / pixel_size + float(ego_row)
            ).round().clamp(0, max_row).long()
            cand_cols = (
                step_xy[..., 1] / pixel_size + float(ego_col)
            ).round().clamp(0, width - 1).long()
            in_bounds = cand_rows < height
            cand_rows_idx = cand_rows.clamp(max=height - 1)
            feasible = step_mask[batch_idx, cand_rows_idx, cand_cols] & in_bounds
            if pad_rows > 0:
                feasible = feasible | (~in_bounds)
            candidate_pix = torch.stack(
                [cand_rows.to(torch.float32), cand_cols.to(torch.float32)],
                dim=-1,
            )

            if correction_iters > 0 and (~feasible).any():
                target_rows = (
                    mu_step[..., 0] / pixel_size + float(ego_row)
                ).clamp(0, max_row)
                target_cols = (
                    mu_step[..., 1] / pixel_size + float(ego_col)
                ).clamp(0, width - 1)
                target = torch.stack([target_rows, target_cols], dim=-1)
                target = 0.5 * (target + current)
                corrected = candidate_pix.clone()
                infeasible = ~feasible

                for _ in range(correction_iters):
                    if not infeasible.any():
                        break
                    corrected = torch.where(
                        infeasible.unsqueeze(-1),
                        correction_blend * corrected + (1.0 - correction_blend) * target,
                        corrected,
                    )
                    corr_rows = corrected[..., 0].round().clamp(0, max_row).long()
                    corr_cols = corrected[..., 1].round().clamp(0, width - 1).long()
                    corr_in_bounds = corr_rows < height
                    corr_rows_idx = corr_rows.clamp(max=height - 1)
                    corr_feasible = (
                        step_mask[batch_idx, corr_rows_idx, corr_cols] & corr_in_bounds
                    )
                    if pad_rows > 0:
                        corr_feasible = corr_feasible | (~corr_in_bounds)
                    corrected_pix = torch.stack(
                        [corr_rows.to(torch.float32), corr_cols.to(torch.float32)],
                        dim=-1,
                    )
                    candidate_pix = torch.where(
                        corr_feasible.unsqueeze(-1),
                        corrected_pix,
                        candidate_pix,
                    )
                    feasible = feasible | corr_feasible
                    infeasible = ~feasible

            current = torch.where(feasible.unsqueeze(-1), candidate_pix, current)
            traj_pix[:, :, step, :] = current

        x = (traj_pix[..., 0] - float(ego_row)) * pixel_size
        y = (traj_pix[..., 1] - float(ego_col)) * pixel_size
        return torch.stack([x, y], dim=-1)

    def _generate_learned_gaussian_anchor_from_mask(
        self,
        mask: torch.Tensor,
        status_feature: Optional[torch.Tensor] = None,
        status_encoding: Optional[torch.Tensor] = None,
        ego_query: Optional[torch.Tensor] = None,
        relaxed_mask: Optional[torch.Tensor] = None,
        strict_steps: int = 0,
        force_sample: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        batch_size, height, width = mask.shape
        device = mask.device
        num_modes = self.ego_fut_mode
        num_poses = self._num_poses
        pixel_size = float(self._config.bev_pixel_size)

        context = self._build_learned_anchor_context(
            batch_size=batch_size,
            device=device,
            dtype=torch.float32,
            status_encoding=status_encoding,
            ego_query=ego_query,
        )
        mode_embed = self._anchor_gaussian_mode_embed.weight
        if mode_embed.shape[0] < num_modes:
            reps = int(np.ceil(num_modes / max(mode_embed.shape[0], 1)))
            mode_embed = mode_embed.repeat(reps, 1)
        mode_embed = mode_embed[:num_modes].to(device=device, dtype=context.dtype)
        mode_context = context[:, None, :] + mode_embed[None, :, :]

        mu_residual = self._anchor_gaussian_mu_head(mode_context).view(
            batch_size, num_modes, num_poses, 2
        )
        log_std = self._anchor_gaussian_logstd_head(mode_context).view(
            batch_size, num_modes, num_poses, 2
        )

        std_xy = F.softplus(log_std)
        min_std = float(getattr(self._config, "anchor_learned_gaussian_min_std_m", 0.2))
        max_std = float(getattr(self._config, "anchor_learned_gaussian_max_std_m", 8.0))
        step_growth = float(
            getattr(self._config, "anchor_learned_gaussian_step_growth", 0.05)
        )
        step_growth = max(0.0, step_growth)
        if step_growth > 0.0:
            growth = 1.0 + step_growth * torch.arange(
                num_poses, device=device, dtype=std_xy.dtype
            )
            std_xy = std_xy * growth[None, None, :, None]
        std_xy = std_xy.clamp(min=min_std, max=max_std)

        if bool(getattr(self._config, "anchor_learned_gaussian_use_physics_prior", True)):
            mu_prior = self._build_physics_prior_anchor_mean(
                mask=mask,
                status_feature=status_feature,
            ).to(device=device, dtype=mu_residual.dtype)
            residual_scale = float(
                getattr(self._config, "anchor_learned_gaussian_residual_scale", 0.5)
            )
            mu_xy = mu_prior + residual_scale * mu_residual
        else:
            mu_xy = mu_residual

        pad_rows = int(round(self._config.anchor_free_forward_pad_m / pixel_size))
        if pad_rows < 0:
            pad_rows = 0
        max_x = (height - 1 + pad_rows) * pixel_size
        ego_col = (width - 1) // 2
        min_y = (0.0 - float(ego_col)) * pixel_size
        max_y = ((width - 1) - float(ego_col)) * pixel_size
        mu_x = mu_xy[..., 0]
        mu_y = mu_xy[..., 1]
        if self._config.anchor_free_forward_only:
            mu_x = mu_x.clamp(min=0.0, max=max_x)
            mu_x = torch.cummax(mu_x, dim=2).values
        else:
            mu_x = mu_x.clamp(min=-max_x, max=max_x)
        mu_y = mu_y.clamp(min=min_y, max=max_y)
        mu_xy = torch.stack([mu_x, mu_y], dim=-1)

        do_sample = bool(force_sample) or self.training or bool(
            getattr(self._config, "anchor_learned_gaussian_sample_in_eval", False)
        )
        if do_sample:
            sample_xy = mu_xy + std_xy * torch.randn_like(mu_xy)
        else:
            sample_xy = mu_xy

        plan_anchor = self._project_anchor_to_feasible_mask(
            candidate_xy=sample_xy,
            mu_xy=mu_xy,
            mask=mask,
            relaxed_mask=relaxed_mask,
            strict_steps=strict_steps,
        )
        return plan_anchor, {"anchor_mu": mu_xy, "anchor_std": std_xy}

    def _build_plan_anchor(
        self,
        batch_size: int,
        device: torch.device,
        targets: Optional[Dict[str, torch.Tensor]] = None,
        anchor_mask: Optional[torch.Tensor] = None,
        anchor_mask_relaxed: Optional[torch.Tensor] = None,
        anchor_mask_pad: bool = True,
        anchor_mask_relaxed_pad: bool = True,
        anchor_strict_steps: int = 0,
        status_feature: Optional[torch.Tensor] = None,
        status_encoding: Optional[torch.Tensor] = None,
        ego_query: Optional[torch.Tensor] = None,
        force_sample: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        anchor_aux: Dict[str, torch.Tensor] = {}
        if not self._config.anchor_free:
            return self.plan_anchor.unsqueeze(0).repeat(batch_size, 1, 1, 1).to(device), anchor_aux

        pad_m_override = None if anchor_mask_pad else 0.0
        mask = self._select_anchor_mask(
            batch_size=batch_size,
            device=device,
            targets=targets,
            anchor_mask=anchor_mask,
            pad_m_override=pad_m_override,
        )
        relaxed_mask = None
        if anchor_mask_relaxed is not None:
            relaxed_pad_override = None if anchor_mask_relaxed_pad else 0.0
            relaxed_mask = self._select_anchor_mask(
                batch_size=batch_size,
                device=device,
                targets=None,
                anchor_mask=anchor_mask_relaxed,
                pad_m_override=relaxed_pad_override,
            )
            if relaxed_mask is not None and mask is not None and relaxed_mask.shape != mask.shape:
                relaxed_mask = None
        if mask is None:
            if relaxed_mask is None:
                return self.plan_anchor.unsqueeze(0).repeat(batch_size, 1, 1, 1).to(device), anchor_aux
            mask = relaxed_mask
            relaxed_mask = None

        if bool(getattr(self._config, "anchor_learned_gaussian_enable", False)):
            plan_anchor, anchor_aux = self._generate_learned_gaussian_anchor_from_mask(
                mask=mask,
                status_feature=status_feature,
                status_encoding=status_encoding,
                ego_query=ego_query,
                relaxed_mask=relaxed_mask,
                strict_steps=anchor_strict_steps,
                force_sample=force_sample,
            )
        elif self._config.anchor_free_gaussian:
            plan_anchor = self._generate_gaussian_anchor_from_mask(
                mask,
                status_feature=status_feature,
                relaxed_mask=relaxed_mask,
                strict_steps=anchor_strict_steps,
            )
        else:
            plan_anchor = self._generate_anchor_from_mask(
                mask,
                status_feature=status_feature,
                relaxed_mask=relaxed_mask,
                strict_steps=anchor_strict_steps,
            )
        has_any = mask.reshape(batch_size, -1).any(dim=1)
        if not has_any.all():
            plan_anchor = plan_anchor.clone()
            anchor_aux = {}
            if relaxed_mask is not None:
                if bool(getattr(self._config, "anchor_learned_gaussian_enable", False)):
                    fallback_anchor, _ = self._generate_learned_gaussian_anchor_from_mask(
                        mask=relaxed_mask,
                        status_feature=status_feature,
                        status_encoding=status_encoding,
                        ego_query=ego_query,
                        force_sample=force_sample,
                    )
                elif self._config.anchor_free_gaussian:
                    fallback_anchor = self._generate_gaussian_anchor_from_mask(
                        relaxed_mask,
                        status_feature=status_feature,
                    )
                else:
                    fallback_anchor = self._generate_anchor_from_mask(
                        relaxed_mask,
                        status_feature=status_feature,
                    )
                plan_anchor[~has_any] = fallback_anchor[~has_any]
            else:
                fallback = self.plan_anchor.unsqueeze(0).repeat(batch_size, 1, 1, 1).to(device)
                plan_anchor[~has_any] = fallback[~has_any]
        return plan_anchor, anchor_aux

    def _maybe_visualize_anchor_bundle(
        self,
        plan_anchor: torch.Tensor,
        anchor_aux: Optional[Dict[str, torch.Tensor]] = None,
        targets: Optional[Dict[str, torch.Tensor]] = None,
        stage: str = "train",
    ) -> None:
        if not bool(getattr(self._config, "anchor_visualize_enable", False)):
            return
        self._anchor_vis_forward_count += 1
        every_n = max(1, int(getattr(self._config, "anchor_visualize_every_n_forward", 500)))
        if (self._anchor_vis_forward_count % every_n) != 0:
            return
        max_files = int(getattr(self._config, "anchor_visualize_max_files", 200))
        if max_files > 0 and self._anchor_vis_saved_count >= max_files:
            return
        if not isinstance(plan_anchor, torch.Tensor) or plan_anchor.dim() != 4:
            return

        out_dir = str(getattr(self._config, "anchor_visualize_dir", "./anchor_vis"))
        os.makedirs(out_dir, exist_ok=True)

        batch_size = int(plan_anchor.shape[0])
        if batch_size <= 0:
            return
        batch_idx = int(getattr(self._config, "anchor_visualize_batch_index", 0))
        batch_idx = max(0, min(batch_idx, batch_size - 1))

        fig = None
        try:
            anchor_np = plan_anchor.detach().cpu().numpy()
            anchor_xy = anchor_np[batch_idx, :, :, :2]
            num_modes = anchor_xy.shape[0]

            mu_xy = None
            std_xy = None
            if anchor_aux is not None:
                mu_tensor = anchor_aux.get("anchor_mu")
                if isinstance(mu_tensor, torch.Tensor) and mu_tensor.dim() == 4 and mu_tensor.shape[0] > batch_idx:
                    mu_xy = mu_tensor.detach().cpu().numpy()[batch_idx, :, :, :2]
                std_tensor = anchor_aux.get("anchor_std")
                if isinstance(std_tensor, torch.Tensor) and std_tensor.dim() == 4 and std_tensor.shape[0] > batch_idx:
                    std_xy = std_tensor.detach().cpu().numpy()[batch_idx, :, :, :2]

            gt_xy = None
            cand_xy = None
            cand_mask = None
            if targets is not None:
                gt = targets.get("trajectory")
                if isinstance(gt, torch.Tensor) and gt.dim() >= 3 and gt.shape[0] > batch_idx:
                    gt_xy = gt.detach().cpu().numpy()[batch_idx, :, :2]
                cands = targets.get("trajectory_candidates")
                if isinstance(cands, torch.Tensor):
                    cands_cpu = cands.detach().cpu()
                    if cands_cpu.dim() == 3:
                        cands_cpu = cands_cpu.unsqueeze(0)
                    if cands_cpu.dim() >= 4 and cands_cpu.shape[0] > batch_idx:
                        cand_xy = cands_cpu[batch_idx, :, :, :2].numpy()
                cands_mask = targets.get("trajectory_candidates_mask")
                if isinstance(cands_mask, torch.Tensor):
                    cands_mask_cpu = cands_mask.detach().cpu()
                    if cands_mask_cpu.dim() == 1:
                        cands_mask_cpu = cands_mask_cpu.unsqueeze(0)
                    if cands_mask_cpu.dim() >= 2 and cands_mask_cpu.shape[0] > batch_idx:
                        cand_mask = cands_mask_cpu[batch_idx].bool().numpy()

            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            cmap = plt.get_cmap("tab20")
            for mode_idx in range(num_modes):
                color = cmap(mode_idx % 20)
                traj_xy = anchor_xy[mode_idx]
                ax.plot(
                    traj_xy[:, 0],
                    traj_xy[:, 1],
                    color=color,
                    linewidth=1.2,
                    alpha=0.9,
                    label="anchor" if mode_idx == 0 else None,
                )
                ax.scatter(
                    traj_xy[0, 0],
                    traj_xy[0, 1],
                    color=color,
                    s=10,
                    alpha=0.6,
                )
                if (
                    bool(getattr(self._config, "anchor_visualize_plot_mu", True))
                    and mu_xy is not None
                    and mode_idx < mu_xy.shape[0]
                ):
                    mean_xy = mu_xy[mode_idx]
                    ax.plot(
                        mean_xy[:, 0],
                        mean_xy[:, 1],
                        color=color,
                        linestyle="--",
                        linewidth=1.0,
                        alpha=0.8,
                        label="mu" if mode_idx == 0 else None,
                    )
                if (
                    bool(getattr(self._config, "anchor_visualize_plot_std", True))
                    and std_xy is not None
                    and mode_idx < std_xy.shape[0]
                ):
                    center_xy = traj_xy[-1]
                    if mu_xy is not None and mode_idx < mu_xy.shape[0]:
                        center_xy = mu_xy[mode_idx, -1]
                    radius = float(np.mean(std_xy[mode_idx, -1]))
                    radius = max(0.05, min(radius, 15.0))
                    circle = plt.Circle(
                        (float(center_xy[0]), float(center_xy[1])),
                        radius=radius,
                        fill=False,
                        color=color,
                        linestyle=":",
                        linewidth=0.8,
                        alpha=0.6,
                    )
                    ax.add_patch(circle)

            if (
                bool(getattr(self._config, "anchor_visualize_plot_candidates", True))
                and cand_xy is not None
            ):
                max_show = min(int(cand_xy.shape[0]), 60)
                for cand_idx in range(max_show):
                    if cand_mask is not None and cand_idx < cand_mask.shape[0] and not bool(cand_mask[cand_idx]):
                        continue
                    traj_xy = cand_xy[cand_idx]
                    ax.plot(
                        traj_xy[:, 0],
                        traj_xy[:, 1],
                        color="gray",
                        linewidth=0.7,
                        alpha=0.25,
                        label="candidate" if cand_idx == 0 else None,
                    )

            if (
                bool(getattr(self._config, "anchor_visualize_plot_gt", True))
                and gt_xy is not None
            ):
                ax.plot(
                    gt_xy[:, 0],
                    gt_xy[:, 1],
                    color="black",
                    linewidth=2.0,
                    alpha=0.95,
                    label="gt",
                )

            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")
            ax.grid(True, alpha=0.2)
            ax.legend(loc="best", fontsize=8)
            ax.set_title(
                f"Anchor Vis [{stage}] fwd={self._anchor_vis_forward_count} batch={batch_idx}"
            )

            file_name = (
                f"{stage}_anchor_fwd_{self._anchor_vis_forward_count:08d}"
                f"_save_{self._anchor_vis_saved_count:05d}.png"
            )
            save_path = os.path.join(out_dir, file_name)
            fig.savefig(save_path, dpi=180, bbox_inches="tight")
            self._anchor_vis_saved_count += 1
        except Exception as e:
            print(f"[anchor_visualize] failed at forward={self._anchor_vis_forward_count}: {e}")
        finally:
            if fig is not None:
                plt.close(fig)

    def _build_inference_dedup_keep_mask(
        self,
        poses_reg: Optional[torch.Tensor],
        mode_scores: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Build a keep-mask for proposal deduplication in inference.
        If mode_scores are provided, keep order follows score descending.
        A candidate is treated as duplicate when:
        - mean trajectory L2 distance is small, and
        - endpoint distance is small, and
        - (optional) final heading difference is small.
        """
        if poses_reg is None or poses_reg.dim() != 4 or poses_reg.shape[1] <= 1:
            return None, None
        if not bool(getattr(self._config, "inference_dedup_enable", False)):
            return None, None

        bs, num_modes, _, _ = poses_reg.shape
        device = poses_reg.device
        keep_mask = torch.zeros((bs, num_modes), dtype=torch.bool, device=device)
        keep_counts = torch.zeros((bs,), dtype=torch.long, device=device)

        mean_thresh = float(
            getattr(self._config, "inference_dedup_mean_l2_thresh_m", 0.6)
        )
        end_thresh = float(
            getattr(self._config, "inference_dedup_endpoint_thresh_m", 1.2)
        )
        heading_thresh_deg = float(
            getattr(self._config, "inference_dedup_heading_thresh_deg", 15.0)
        )
        heading_thresh_rad = math.radians(max(0.0, heading_thresh_deg))
        use_heading = poses_reg.shape[-1] > 2 and heading_thresh_rad > 0.0
        max_modes_cfg = int(getattr(self._config, "inference_dedup_max_modes", 0) or 0)
        max_modes = num_modes
        if max_modes_cfg > 0:
            max_modes = max(1, min(num_modes, max_modes_cfg))

        for batch_idx in range(bs):
            traj_xy = poses_reg[batch_idx, :, :, :2]
            end_xy = traj_xy[:, -1, :]
            # Pairwise trajectory and endpoint distances.
            mean_dist = torch.linalg.norm(
                traj_xy[:, None, :, :] - traj_xy[None, :, :, :], dim=-1
            ).mean(dim=-1)
            end_dist = torch.linalg.norm(
                end_xy[:, None, :] - end_xy[None, :, :], dim=-1
            )
            heading_dist = None
            if use_heading:
                heading = poses_reg[batch_idx, :, -1, 2]
                delta = heading[:, None] - heading[None, :]
                heading_dist = torch.abs(torch.atan2(torch.sin(delta), torch.cos(delta)))

            order = torch.arange(num_modes, device=device)
            if (
                mode_scores is not None
                and mode_scores.dim() == 2
                and mode_scores.shape[0] == bs
                and mode_scores.shape[1] == num_modes
            ):
                score_row = torch.nan_to_num(
                    mode_scores[batch_idx], nan=-1e9, posinf=1e9, neginf=-1e9
                )
                order = torch.argsort(score_row, descending=True)

            selected: List[int] = []
            for mode_idx_tensor in order:
                if len(selected) >= max_modes:
                    break
                mode_idx = int(mode_idx_tensor.item())
                is_duplicate = False
                for sel_idx in selected:
                    close_xy = bool(
                        mean_dist[mode_idx, sel_idx] <= mean_thresh
                        and end_dist[mode_idx, sel_idx] <= end_thresh
                    )
                    if not close_xy:
                        continue
                    if heading_dist is not None and bool(
                        heading_dist[mode_idx, sel_idx] > heading_thresh_rad
                    ):
                        continue
                    is_duplicate = True
                    break
                if not is_duplicate:
                    selected.append(mode_idx)
                    keep_mask[batch_idx, mode_idx] = True

            if not selected:
                fallback_idx = int(order[0].item()) if order.numel() > 0 else 0
                keep_mask[batch_idx, fallback_idx] = True
                keep_counts[batch_idx] = 1
            else:
                keep_counts[batch_idx] = len(selected)

        return keep_mask, keep_counts

    def _select_anchor_mask(
        self,
        batch_size: int,
        device: torch.device,
        targets: Optional[Dict[str, torch.Tensor]] = None,
        anchor_mask: Optional[torch.Tensor] = None,
        pad_m_override: Optional[float] = None,
    ) -> Optional[torch.Tensor]:
        mask = anchor_mask
        use_target_mask = (
            self._config.anchor_free_use_target
            and bool(getattr(self._config, "anchor_use_feasible_area_mask", True))
            and targets is not None
        )
        if not self.training and self._config.eval_use_predicted_bev_masks:
            use_target_mask = False
        if mask is None and use_target_mask:
            mask = targets.get("feasible_area_mask")
        if mask is None:
            return None

        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        elif mask.dim() == 4 and mask.shape[1] == 1:
            mask = mask[:, 0]
        if mask.dim() != 3:
            return None
        if mask.shape[0] != batch_size:
            if mask.shape[0] == 1:
                mask = mask.repeat(batch_size, 1, 1)
            else:
                return None
        mask = mask.detach().to(device=device, dtype=torch.bool)
        pad_m = self._config.anchor_free_forward_pad_m
        if pad_m_override is not None:
            pad_m = pad_m_override
        if pad_m and pad_m > 0:
            mask = self._pad_anchor_mask(mask, pad_m)
        return mask

    def _pad_anchor_mask(self, mask: torch.Tensor, pad_m: float) -> torch.Tensor:
        pad_rows = int(round(pad_m / self._config.bev_pixel_size))
        if pad_rows <= 0:
            return mask
        mask = mask.clone()
        batch_size, height, _ = mask.shape
        for batch_idx in range(batch_size):
            row_has_any = mask[batch_idx].any(dim=1)
            if not row_has_any.any().item():
                continue
            last_row = row_has_any.nonzero(as_tuple=False)[-1, 0].item()
            start_row = last_row + 1
            if start_row >= height:
                continue
            end_row = min(height, start_row + pad_rows)
            tail_start = max(0, last_row - 2)
            tail_mask = mask[batch_idx, tail_start : last_row + 1].any(dim=0)
            mask[batch_idx, start_row:end_row] = tail_mask
        return mask

    def _select_anchor_seeds(self, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, height, width = mask.shape
        ego_row = 0
        ego_col = (width - 1) // 2
        max_offset_m = self._config.anchor_free_max_seed_offset_m
        max_dist2 = None
        if max_offset_m and max_offset_m > 0:
            max_px = max_offset_m / self._config.bev_pixel_size
            max_dist2 = max_px * max_px

        seed_rows = max(1, min(self._config.feasible_lane_seed_rows, height))
        row_start = 0
        row_end = min(height, seed_rows)
        seed_cols = max(0, min(self._config.feasible_lane_seed_cols, width // 2))
        col_start = max(0, ego_col - seed_cols)
        col_end = min(width, ego_col + seed_cols + 1)
        if col_end <= col_start:
            col_start = min(ego_col, width - 1)
            col_end = min(width, col_start + 1)

        start_rows = torch.full((batch_size,), ego_row, device=mask.device, dtype=torch.long)
        start_cols = torch.full((batch_size,), ego_col, device=mask.device, dtype=torch.long)
        ego_index = torch.tensor([ego_row, ego_col], device=mask.device, dtype=torch.long)

        for batch_idx in range(batch_size):
            full_coords = mask[batch_idx].nonzero(as_tuple=False)
            if full_coords.numel() == 0:
                continue

            region = mask[batch_idx, row_start:row_end, col_start:col_end]
            coords = region.nonzero(as_tuple=False)
            if coords.numel() > 0:
                coords = coords + torch.tensor(
                    [row_start, col_start], device=coords.device, dtype=coords.dtype
                )
            else:
                coords = full_coords

            deltas = coords - ego_index
            dist2 = deltas[:, 0] * deltas[:, 0] + deltas[:, 1] * deltas[:, 1]
            best_idx = dist2.argmin()
            start_rows[batch_idx] = coords[best_idx, 0]
            start_cols[batch_idx] = coords[best_idx, 1]

            if max_dist2 is not None:
                full_deltas = full_coords - ego_index
                full_dist2 = (
                    full_deltas[:, 0] * full_deltas[:, 0]
                    + full_deltas[:, 1] * full_deltas[:, 1]
                )
                within = full_dist2 <= max_dist2
                if within.any():
                    within_coords = full_coords[within]
                    within_dist2 = full_dist2[within]
                    best_within = within_dist2.argmin()
                    start_rows[batch_idx] = within_coords[best_within, 0]
                    start_cols[batch_idx] = within_coords[best_within, 1]

        return start_rows, start_cols

    def _generate_anchor_from_mask(
        self,
        mask: torch.Tensor,
        status_feature: Optional[torch.Tensor] = None,
        relaxed_mask: Optional[torch.Tensor] = None,
        strict_steps: int = 0,
    ) -> torch.Tensor:
        batch_size, height, width = mask.shape
        device = mask.device
        num_modes = self.ego_fut_mode
        num_poses = self._num_poses

        step_meters = self._compute_anchor_step_meters(
            batch_size=batch_size,
            device=device,
            status_feature=status_feature,
        )
        step_px = (
            (step_meters / self._config.bev_pixel_size)
            .round()
            .clamp(min=1)
            .to(torch.long)
        )
        step_px = step_px[:, None].expand(batch_size, num_modes)
        pad_rows = int(round(self._config.anchor_free_forward_pad_m / self._config.bev_pixel_size))
        if pad_rows < 0:
            pad_rows = 0
        max_row = height - 1 + pad_rows

        if self._config.anchor_free_forward_only:
            directions = torch.tensor(
                [[1, 0], [1, -1], [1, 1]], device=device, dtype=torch.long
            )
        else:
            directions = torch.tensor(
                [
                    [-1, 0],
                    [1, 0],
                    [0, -1],
                    [0, 1],
                    [-1, -1],
                    [-1, 1],
                    [1, -1],
                    [1, 1],
                ],
                device=device,
                dtype=torch.long,
            )
        start_rows, start_cols = self._select_anchor_seeds(mask)
        current = torch.stack([start_rows, start_cols], dim=-1)
        current = current[:, None, :].repeat(1, num_modes, 1)

        traj_pix = torch.empty(
            (batch_size, num_modes, num_poses, 2), device=device, dtype=torch.float32
        )
        batch_idx = torch.arange(batch_size, device=device)[:, None].expand(batch_size, num_modes)

        strict_steps = int(strict_steps) if strict_steps else 0
        for step in range(num_poses):
            dir_idx = torch.randint(0, directions.shape[0], (batch_size, num_modes), device=device)
            delta = directions[dir_idx] * step_px[..., None]
            candidate = current + delta
            candidate[..., 0] = candidate[..., 0].clamp(0, max_row)
            candidate[..., 1] = candidate[..., 1].clamp(0, width - 1)

            cand_rows = candidate[..., 0]
            cand_cols = candidate[..., 1]
            in_bounds = cand_rows < height
            cand_rows_idx = cand_rows.clamp(max=height - 1)
            step_mask = mask
            if relaxed_mask is not None and strict_steps > 0 and step >= strict_steps:
                step_mask = relaxed_mask
            feasible = step_mask[batch_idx, cand_rows_idx, cand_cols] & in_bounds
            if pad_rows > 0:
                feasible = feasible | (~in_bounds)
            current = torch.where(feasible.unsqueeze(-1), candidate, current)
            traj_pix[..., step, :] = current.to(torch.float32)

        ego_row = 0
        ego_col = (width - 1) // 2
        x = (traj_pix[..., 0] - ego_row) * self._config.bev_pixel_size      
        y = (traj_pix[..., 1] - ego_col) * self._config.bev_pixel_size      
        return torch.stack([x, y], dim=-1)

    def _generate_gaussian_anchor_from_mask(
        self,
        mask: torch.Tensor,
        status_feature: Optional[torch.Tensor] = None,
        relaxed_mask: Optional[torch.Tensor] = None,
        strict_steps: int = 0,
    ) -> torch.Tensor:
        if bool(getattr(self._config, "anchor_free_gaussian_temporal_enable", False)):
            return self._generate_temporal_gaussian_anchor_from_mask(
                mask=mask,
                status_feature=status_feature,
                relaxed_mask=relaxed_mask,
                strict_steps=strict_steps,
            )

        batch_size, height, width = mask.shape
        device = mask.device
        num_modes = self.ego_fut_mode
        num_poses = self._num_poses

        step_meters = self._compute_anchor_step_meters(
            batch_size=batch_size,
            device=device,
            status_feature=status_feature,
        )
        sigma_px = step_meters / self._config.bev_pixel_size
        sigma_px = sigma_px.clamp(min=1.0)
        sigma_px = sigma_px[:, None].expand(batch_size, num_modes)
        pad_rows = int(round(self._config.anchor_free_forward_pad_m / self._config.bev_pixel_size))
        if pad_rows < 0:
            pad_rows = 0
        max_row = height - 1 + pad_rows

        start_rows, start_cols = self._select_anchor_seeds(mask)
        current = torch.stack([start_rows, start_cols], dim=-1).to(torch.float32)
        current = current[:, None, :].repeat(1, num_modes, 1)

        traj_pix = torch.empty(
            (batch_size, num_modes, num_poses, 2), device=device, dtype=torch.float32
        )
        batch_idx = torch.arange(batch_size, device=device)[:, None].expand(batch_size, num_modes)

        strict_steps = int(strict_steps) if strict_steps else 0
        for step in range(num_poses):
            delta = torch.randn((batch_size, num_modes, 2), device=device) * sigma_px[..., None]
            if self._config.anchor_free_forward_only:
                delta[..., 0] = delta[..., 0].abs()
            candidate = current + delta
            cand_rows = candidate[..., 0].round().clamp(0, max_row).long()
            cand_cols = candidate[..., 1].round().clamp(0, width - 1).long()
            in_bounds = cand_rows < height
            cand_rows_idx = cand_rows.clamp(max=height - 1)
            step_mask = mask
            if relaxed_mask is not None and strict_steps > 0 and step >= strict_steps:
                step_mask = relaxed_mask
            feasible = step_mask[batch_idx, cand_rows_idx, cand_cols] & in_bounds
            if pad_rows > 0:
                feasible = feasible | (~in_bounds)
            candidate_pix = torch.stack([cand_rows.to(torch.float32), cand_cols.to(torch.float32)], dim=-1)
            current = torch.where(feasible.unsqueeze(-1), candidate_pix, current)
            traj_pix[..., step, :] = current

        ego_row = 0
        ego_col = (width - 1) // 2
        x = (traj_pix[..., 0] - ego_row) * self._config.bev_pixel_size
        y = (traj_pix[..., 1] - ego_col) * self._config.bev_pixel_size
        return torch.stack([x, y], dim=-1)

    def _generate_temporal_gaussian_anchor_from_mask(
        self,
        mask: torch.Tensor,
        status_feature: Optional[torch.Tensor] = None,
        relaxed_mask: Optional[torch.Tensor] = None,
        strict_steps: int = 0,
    ) -> torch.Tensor:
        """
        Generate anchors with temporally correlated constrained Gaussian noise.
        Mean trajectory is physics-guided from ego velocity/acceleration.
        """
        batch_size, height, width = mask.shape
        device = mask.device
        num_modes = self.ego_fut_mode
        num_poses = self._num_poses

        step_meters = self._compute_anchor_step_meters(
            batch_size=batch_size,
            device=device,
            status_feature=status_feature,
        )
        sigma_base_px = step_meters / self._config.bev_pixel_size
        sigma_base_px = sigma_base_px.clamp(min=1.0)[:, None].expand(batch_size, num_modes)

        pad_rows = int(round(self._config.anchor_free_forward_pad_m / self._config.bev_pixel_size))
        if pad_rows < 0:
            pad_rows = 0
        max_row = height - 1 + pad_rows

        start_rows, start_cols = self._select_anchor_seeds(mask)
        start_pix = torch.stack([start_rows, start_cols], dim=-1).to(torch.float32)
        current = start_pix[:, None, :].repeat(1, num_modes, 1)

        # Build physics-guided mean trajectory mu(k) in pixel coordinates.
        dt = float(self._config.trajectory_sampling.interval_length)
        if dt <= 0:
            dt = 0.5
        status = status_feature
        if status is not None:
            if status.dim() == 1:
                status = status.unsqueeze(0)
            if status.shape[0] != batch_size:
                if status.shape[0] == 1:
                    status = status.repeat(batch_size, 1)
                else:
                    status = None
        if status is not None and status.shape[-1] >= 6:
            velocity = status[:, 4:6]
        else:
            velocity = torch.stack(
                [step_meters / dt, torch.zeros_like(step_meters)],
                dim=-1,
            )
        if status is not None and status.shape[-1] >= 8:
            acceleration = status[:, 6:8]
        else:
            acceleration = torch.zeros_like(velocity)

        time_idx = torch.arange(1, num_poses + 1, device=device, dtype=torch.float32)
        time_sec = time_idx[None, None, :, None] * dt
        velocity = velocity[:, None, None, :]
        acceleration = acceleration[:, None, None, :]
        mu_xy_m = velocity * time_sec + 0.5 * acceleration * (time_sec**2)
        mu_pix = torch.empty((batch_size, num_modes, num_poses, 2), device=device, dtype=torch.float32)
        mu_pix[..., 0] = current[..., 0][:, :, None] + mu_xy_m[..., 0] / self._config.bev_pixel_size
        mu_pix[..., 1] = current[..., 1][:, :, None] + mu_xy_m[..., 1] / self._config.bev_pixel_size
        if self._config.anchor_free_forward_only:
            mu_pix[..., 0] = torch.maximum(mu_pix[..., 0], current[..., 0][:, :, None])

        rho = float(getattr(self._config, "anchor_free_gaussian_temporal_rho", 0.85))
        rho = max(0.0, min(0.999, rho))
        rho_noise = math.sqrt(max(1.0 - rho * rho, 0.0))
        lat_ratio = float(getattr(self._config, "anchor_free_gaussian_lateral_ratio", 0.35))
        lat_ratio = max(0.0, lat_ratio)
        step_growth = float(getattr(self._config, "anchor_free_gaussian_step_growth", 0.08))
        step_growth = max(0.0, step_growth)
        correction_iters = int(max(0, getattr(self._config, "anchor_free_gaussian_correction_iters", 2)))
        correction_blend = float(getattr(self._config, "anchor_free_gaussian_correction_blend", 0.5))
        correction_blend = max(0.0, min(1.0, correction_blend))

        eps_parallel = torch.zeros((batch_size, num_modes), device=device, dtype=torch.float32)
        eps_lateral = torch.zeros((batch_size, num_modes), device=device, dtype=torch.float32)
        traj_pix = torch.empty((batch_size, num_modes, num_poses, 2), device=device, dtype=torch.float32)
        batch_idx = torch.arange(batch_size, device=device)[:, None].expand(batch_size, num_modes)

        strict_steps = int(strict_steps) if strict_steps else 0
        for step in range(num_poses):
            step_mask = mask
            if relaxed_mask is not None and strict_steps > 0 and step >= strict_steps:
                step_mask = relaxed_mask

            mu_step = mu_pix[:, :, step, :]
            if step == 0:
                tangent = mu_step - current
            else:
                tangent = mu_pix[:, :, step, :] - mu_pix[:, :, step - 1, :]
            tangent_norm = torch.linalg.norm(tangent, dim=-1, keepdim=True)
            default_dir = torch.zeros_like(tangent)
            default_dir[..., 0] = 1.0
            tangent_unit = torch.where(
                tangent_norm > 1e-3,
                tangent / tangent_norm.clamp_min(1e-3),
                default_dir,
            )
            normal_unit = torch.stack(
                [-tangent_unit[..., 1], tangent_unit[..., 0]],
                dim=-1,
            )

            sigma_scale = 1.0 + step_growth * float(step)
            sigma_parallel = sigma_base_px * sigma_scale
            sigma_lateral = sigma_parallel * lat_ratio
            eps_parallel = (
                rho * eps_parallel
                + rho_noise * torch.randn_like(eps_parallel) * sigma_parallel
            )
            eps_lateral = (
                rho * eps_lateral
                + rho_noise * torch.randn_like(eps_lateral) * sigma_lateral
            )
            noise_delta = (
                tangent_unit * eps_parallel[..., None]
                + normal_unit * eps_lateral[..., None]
            )

            candidate = mu_step + noise_delta
            if self._config.anchor_free_forward_only:
                candidate[..., 0] = torch.maximum(candidate[..., 0], current[..., 0])
            cand_rows = candidate[..., 0].round().clamp(0, max_row).long()
            cand_cols = candidate[..., 1].round().clamp(0, width - 1).long()
            in_bounds = cand_rows < height
            cand_rows_idx = cand_rows.clamp(max=height - 1)
            feasible = step_mask[batch_idx, cand_rows_idx, cand_cols] & in_bounds
            if pad_rows > 0:
                feasible = feasible | (~in_bounds)
            candidate_pix = torch.stack(
                [cand_rows.to(torch.float32), cand_cols.to(torch.float32)],
                dim=-1,
            )

            if correction_iters > 0 and (~feasible).any():
                target = 0.5 * (current + mu_step)
                corrected = candidate_pix.clone()
                infeasible = ~feasible
                for _ in range(correction_iters):
                    if not infeasible.any():
                        break
                    corrected = torch.where(
                        infeasible.unsqueeze(-1),
                        correction_blend * corrected + (1.0 - correction_blend) * target,
                        corrected,
                    )
                    corr_rows = corrected[..., 0].round().clamp(0, max_row).long()
                    corr_cols = corrected[..., 1].round().clamp(0, width - 1).long()
                    corr_in_bounds = corr_rows < height
                    corr_rows_idx = corr_rows.clamp(max=height - 1)
                    corr_feasible = step_mask[batch_idx, corr_rows_idx, corr_cols] & corr_in_bounds
                    if pad_rows > 0:
                        corr_feasible = corr_feasible | (~corr_in_bounds)
                    corrected_pix = torch.stack(
                        [corr_rows.to(torch.float32), corr_cols.to(torch.float32)],
                        dim=-1,
                    )
                    candidate_pix = torch.where(
                        corr_feasible.unsqueeze(-1),
                        corrected_pix,
                        candidate_pix,
                    )
                    feasible = feasible | corr_feasible
                    infeasible = ~feasible

            current = torch.where(feasible.unsqueeze(-1), candidate_pix, current)
            traj_pix[:, :, step, :] = current

        ego_row = 0
        ego_col = (width - 1) // 2
        x = (traj_pix[..., 0] - ego_row) * self._config.bev_pixel_size
        y = (traj_pix[..., 1] - ego_col) * self._config.bev_pixel_size
        return torch.stack([x, y], dim=-1)

    def forward(
        self,
        ego_query,
        agents_query,
        bev_feature,
        bev_spatial_shape,
        status_encoding,
        targets=None,
        global_img=None,
        image_tokens=None,
        lidar_tokens=None,
        score_image_tokens=None,
        score_lidar_tokens=None,
        history_tokens=None,
        risk_bev_map=None,
        anchor_mask=None,
        anchor_mask_relaxed=None,
        anchor_mask_pad=True,
        anchor_mask_relaxed_pad=True,
        anchor_strict_steps=0,
        status_feature=None,
    ) -> Dict[str, torch.Tensor]:
        """Torch module forward pass."""
        use_train_forward = self.training or self._config.force_train_forward
        if use_train_forward:
            return self.forward_train(
                ego_query,
                agents_query,
                bev_feature,
                bev_spatial_shape,
                status_encoding,
                targets,
                global_img,
                image_tokens=image_tokens,
                lidar_tokens=lidar_tokens,
                score_image_tokens=score_image_tokens,
                score_lidar_tokens=score_lidar_tokens,
                history_tokens=history_tokens,
                risk_bev_map=risk_bev_map,
                anchor_mask=anchor_mask,
                anchor_mask_relaxed=anchor_mask_relaxed,
                anchor_mask_pad=anchor_mask_pad,
                anchor_mask_relaxed_pad=anchor_mask_relaxed_pad,
                anchor_strict_steps=anchor_strict_steps,
                status_feature=status_feature,
            )
        return self.forward_test(
            ego_query,
            agents_query,
            bev_feature,
            bev_spatial_shape,
            status_encoding,
            global_img,
            image_tokens=image_tokens,
            lidar_tokens=lidar_tokens,
            score_image_tokens=score_image_tokens,
            score_lidar_tokens=score_lidar_tokens,
            history_tokens=history_tokens,
            risk_bev_map=risk_bev_map,
            targets=targets,
            anchor_mask=anchor_mask,
            anchor_mask_relaxed=anchor_mask_relaxed,
            anchor_mask_pad=anchor_mask_pad,
            anchor_mask_relaxed_pad=anchor_mask_relaxed_pad,
            anchor_strict_steps=anchor_strict_steps,
            status_feature=status_feature,
        )

    def _decode_proposals(
        self,
        traj_feature,
        traj_points,
        bev_feature,
        bev_spatial_shape,
        agents_query,
        ego_query,
        time_embed,
        status_encoding,
        global_img=None,
        image_tokens=None,
        lidar_tokens=None,
        history_tokens=None,
        risk_bev_map=None,
    ):
        return self.diff_decoder(
            traj_feature,
            traj_points,
            bev_feature,
            bev_spatial_shape,
            agents_query,
            ego_query,
            time_embed,
            status_encoding,
            global_img,
            image_tokens=image_tokens,
            lidar_tokens=lidar_tokens,
            history_tokens=history_tokens,
        )


    def forward_train(
        self,
        ego_query,
        agents_query,
        bev_feature,
        bev_spatial_shape,
        status_encoding,
        targets=None,
        global_img=None,
        image_tokens=None,
        lidar_tokens=None,
        score_image_tokens=None,
        score_lidar_tokens=None,
        history_tokens=None,
        risk_bev_map=None,
        anchor_mask=None,
        anchor_mask_relaxed=None,
        anchor_mask_pad=True,
        anchor_mask_relaxed_pad=True,
        anchor_strict_steps=0,
        status_feature=None,
    ) -> Dict[str, torch.Tensor]:
        bs = ego_query.shape[0]
        device = ego_query.device
        if score_image_tokens is None:
            score_image_tokens = image_tokens
        if score_lidar_tokens is None:
            score_lidar_tokens = lidar_tokens
        score_agent_tokens = None
        if getattr(self._config, "pdm_score_use_agent_tokens", False):
            score_agent_tokens = agents_query.detach()
        direct_proposal_decoder = (
            str(getattr(self._config, "trajectory_decoder_type", "diffusion")).lower()
            == "proformer"
        )
        # 1. add truncated noise to the plan anchor
        plan_anchor, anchor_aux = self._build_plan_anchor(
            batch_size=bs,
            device=device,
            targets=targets,
            anchor_mask=anchor_mask,
            anchor_mask_relaxed=anchor_mask_relaxed,
            anchor_mask_pad=anchor_mask_pad,
            anchor_mask_relaxed_pad=anchor_mask_relaxed_pad,
            anchor_strict_steps=anchor_strict_steps,
            status_feature=status_feature,
            status_encoding=status_encoding,
            ego_query=ego_query,
        )    # [bs, modes, poses, 2] anchor from feasible area
        self._maybe_visualize_anchor_bundle(
            plan_anchor=plan_anchor,
            anchor_aux=anchor_aux,
            targets=targets,
            stage="train",
        )
        # plot_trajectory_anchors(plan_anchor,file_name="plan_anchor.png")
        if direct_proposal_decoder or (self._config.anchor_free and self._config.anchor_free_skip_diffusion_noise):
            timesteps = torch.ones((bs,), device=device, dtype=torch.long) * 8
            noisy_traj_points = plan_anchor
        else:
            odo_info_fut = self.norm_odo(plan_anchor)
            timesteps = torch.randint(
                0, 50,
                (bs,), device=device
            )
            noise = torch.randn(odo_info_fut.shape, device=device)
            noisy_traj_points = self.diffusion_scheduler.add_noise(
                original_samples=odo_info_fut,
                noise=noise,
                timesteps=timesteps,
            ).float()
            noisy_traj_points = torch.clamp(noisy_traj_points, min=-1, max=1)
            noisy_traj_points = self.denorm_odo(noisy_traj_points)
            # plot_trajectory_anchors(noisy_traj_points,file_name="noisy_traj_points.png")
        ego_fut_mode = noisy_traj_points.shape[1]   # 自车未来轨迹模式数
        # 2. initialize trajectory tokens
        traj_feature = self._build_traj_tokens(
            noisy_traj_points=noisy_traj_points,
            status_encoding=status_encoding,
        )
        # 3. embed the timesteps, 时间步的嵌入向量
        time_embed = None
        if self.time_mlp is not None:
            time_embed = self.time_mlp(timesteps)     # [bs,d_model]
            time_embed = time_embed.view(bs,1,-1)     # [bs,1,d_model] 时间步的嵌入向量


        # 4. begin the stacked decoder, 多层解码器去噪（加噪轨迹特征，加噪轨迹点，bev特征，bev空间特征（x），周车q，自车q，加噪时间嵌入，状态嵌入（x））
        decorrelation_loss = torch.tensor(0.0, device=device)
        if self._config.diff_input_decorrelation_weight > 0:
            decorrelation_loss = self._diff_input_decorrelation_loss(traj_feature)
        decode_outputs = self._decode_proposals(
            traj_feature,
            noisy_traj_points,
            bev_feature,
            bev_spatial_shape,
            agents_query,
            ego_query,
            time_embed,
            status_encoding,
            global_img,
            image_tokens=image_tokens,
            lidar_tokens=lidar_tokens,
            history_tokens=history_tokens,
            risk_bev_map=risk_bev_map,
        )
        proposal_risk_logits_list = None
        if isinstance(decode_outputs, (tuple, list)) and len(decode_outputs) == 4:
            (
                poses_reg_list,
                poses_cls_list,
                decorrelation_losses,
                proposal_risk_logits_list,
            ) = decode_outputs
        else:
            poses_reg_list, poses_cls_list, decorrelation_losses = decode_outputs
        proposal_risk_trajectories = torch.stack(
            [poses_reg.clone() for poses_reg in poses_reg_list], dim=0
        )
        refine_intermediate: List[torch.Tensor] = []
        refine_mode_attn_map: Optional[torch.Tensor] = None
        refine_mode_context: Optional[torch.Tensor] = None
        refined_last, refine_intermediate, refine_mode_attn_map, refine_mode_context = self._apply_refinement(
            poses_reg=poses_reg_list[-1],
            status_encoding=status_encoding,
            bev_feature=bev_feature,
            risk_bev_map=risk_bev_map,
        )
        if refined_last is not None:
            poses_reg_list[-1] = refined_last
        scorer_mode_attn_map: Optional[torch.Tensor] = None
        scorer_mode_context: Optional[torch.Tensor] = None
        if bool(getattr(self._config, "mode_bev_attention_enable", False)) and bool(
            getattr(self._config, "mode_bev_attention_use_scorer", True)
        ):
            recompute_after_refine = bool(
                getattr(self._config, "mode_bev_attention_recompute_after_refine", True)
            )
            if (
                (not recompute_after_refine)
                and refine_mode_context is not None
                and refine_mode_context.shape[:2] == poses_reg_list[-1].shape[:2]
            ):
                scorer_mode_attn_map = refine_mode_attn_map
                scorer_mode_context = refine_mode_context
            else:
                scorer_mode_attn_map, scorer_mode_context = self._compute_mode_bev_attention(
                    poses_reg=poses_reg_list[-1],
                    bev_feature=bev_feature,
                    status_encoding=status_encoding,
                    risk_bev_map=risk_bev_map,
                )
        pdm_score = None
        pdm_score_specialist = None
        pdm_score_components = None
        pdm_score_components_specialist = None
        pdm_score_cached = None
        pdm_score_components_cached = None
        hardcase_score_aux: Optional[Dict[str, torch.Tensor]] = None
        if self._pdm_score_head is not None:
            pdm_score_raw, hardcase_score_aux = self._pdm_score_head(
                poses_reg_list[-1].detach(),
                bev_feature,
                image_tokens=score_image_tokens,
                lidar_tokens=score_lidar_tokens,
                agent_tokens=score_agent_tokens,
                mode_context=scorer_mode_context,
            )
            pdm_score_specialist_raw = None
            if hardcase_score_aux is not None:
                pdm_score_specialist_raw = hardcase_score_aux.get("pdm_score_specialist")
            pdm_score = pdm_score_raw
            if pdm_score is not None and pdm_score.dim() == 3:
                pdm_score_components = pdm_score
                pdm_score = self._aggregate_pdm_components_inference(
                    pdm_score_components
                )
            if torch.is_tensor(pdm_score_specialist_raw):
                pdm_score_specialist = pdm_score_specialist_raw
                if pdm_score_specialist.dim() == 3:
                    pdm_score_components_specialist = pdm_score_specialist
                    pdm_score_specialist = self._aggregate_pdm_components_inference(
                        pdm_score_components_specialist
                    )
            else:
                pdm_score_specialist = pdm_score
            if (
                self._config.pdm_score_use_cached_poses
                and targets is not None
                and "poses_reg" in targets
            ):
                cached_poses = targets["poses_reg"]
                if cached_poses.dim() == 3:
                    cached_poses = cached_poses.unsqueeze(0)
                cached_poses = cached_poses.to(
                    device=poses_reg_list[-1].device,
                    dtype=poses_reg_list[-1].dtype,
                )
                cached_mode_context: Optional[torch.Tensor] = None
                if bool(getattr(self._config, "mode_bev_attention_enable", False)) and bool(
                    getattr(self._config, "mode_bev_attention_use_scorer", True)
                ):
                    _, cached_mode_context = self._compute_mode_bev_attention(
                        poses_reg=cached_poses,
                        bev_feature=bev_feature,
                        status_encoding=status_encoding,
                        risk_bev_map=risk_bev_map,
                    )
                pdm_score_cached, _ = self._pdm_score_head(
                    cached_poses.detach(),
                    bev_feature,
                    image_tokens=score_image_tokens,
                    lidar_tokens=score_lidar_tokens,
                    agent_tokens=score_agent_tokens,
                    mode_context=cached_mode_context,
                )
                if pdm_score_cached is not None and pdm_score_cached.dim() == 3:
                    pdm_score_components_cached = pdm_score_cached
                    pdm_score_cached = self._aggregate_pdm_components_inference(
                        pdm_score_components_cached
                    )
        """
        pred_traj: (bs, num_modes, num_poses, 3)
        pred_cls: (bs, 20)
        plan_anchor: (bs, num_modes, num_poses, 2)
        targets['trajectory']: (bs, 8, 3)
        """
        trajectory_loss_dict = {}
        ret_traj_loss = 0
        for idx, (poses_reg, poses_cls) in enumerate(zip(poses_reg_list, poses_cls_list)):
            trajectory_loss = self.loss_computer(poses_reg, poses_cls, targets, plan_anchor)
            trajectory_loss_dict[f"trajectory_loss_{idx}"] = trajectory_loss
            ret_traj_loss += trajectory_loss
        refine_aux_weight = float(getattr(self._config, "refine_aux_loss_weight", 0.0))
        if refine_aux_weight > 0.0 and targets is not None and refine_intermediate:
            target_traj = targets.get("trajectory")
            gt_w = max(float(getattr(self._config, "refine_aux_gt_weight", 1.0)), 0.0)
            cand_w = max(
                float(getattr(self._config, "refine_aux_candidate_weight", 1.0)), 0.0
            )
            use_cands = bool(getattr(self._config, "refine_aux_use_candidates", True))
            per_step_losses: List[torch.Tensor] = []
            for step_reg in refine_intermediate:
                step_terms: List[torch.Tensor] = []
                step_weights: List[float] = []
                if gt_w > 0.0:
                    gt_loss, gt_valid = self._compute_mode_to_gt_distance(
                        step_reg, target_traj
                    )
                    if gt_valid:
                        step_terms.append(gt_loss * gt_w)
                        step_weights.append(gt_w)
                if use_cands and cand_w > 0.0:
                    cand_loss, cand_valid = self._compute_mode_to_candidate_distance(
                        step_reg, targets
                    )
                    if cand_valid:
                        step_terms.append(cand_loss * cand_w)
                        step_weights.append(cand_w)
                if step_weights:
                    per_step_losses.append(
                        sum(step_terms) / max(float(sum(step_weights)), 1e-6)
                    )
            if per_step_losses:
                refine_aux_loss = torch.stack(per_step_losses).mean()
                weighted_refine_aux = refine_aux_weight * refine_aux_loss
                ret_traj_loss += weighted_refine_aux
                trajectory_loss_dict["refine_aux_loss"] = weighted_refine_aux

        if bool(getattr(self._config, "anchor_learned_gaussian_enable", False)):
            nll_w = float(
                getattr(self._config, "anchor_learned_gaussian_nll_weight", 0.0)
            )
            reg_w = float(
                getattr(self._config, "anchor_learned_gaussian_reg_weight", 0.0)
            )
            cand_w = float(
                getattr(self._config, "anchor_learned_gaussian_candidate_weight", 0.0)
            )
            if (nll_w > 0.0 or reg_w > 0.0 or cand_w > 0.0) and targets is not None:
                anchor_mu = anchor_aux.get("anchor_mu")
                anchor_std = anchor_aux.get("anchor_std")
                target_traj = targets.get("trajectory")
                gauss_nll, gauss_reg, gauss_pos_dist = self._compute_learned_gaussian_supervision(
                    anchor_mu=anchor_mu,
                    anchor_std=anchor_std,
                    target_traj=target_traj,
                )
                if nll_w > 0.0:
                    weighted_nll = nll_w * gauss_nll
                    ret_traj_loss += weighted_nll
                    trajectory_loss_dict["anchor_gaussian_nll_loss"] = weighted_nll
                if reg_w > 0.0:
                    weighted_reg = reg_w * gauss_reg
                    ret_traj_loss += weighted_reg
                    trajectory_loss_dict["anchor_gaussian_reg_loss"] = weighted_reg
                trajectory_loss_dict["anchor_gaussian_positive_distance"] = gauss_pos_dist
                if cand_w > 0.0:
                    cand_nll, cand_dist = self._compute_learned_gaussian_candidate_supervision(
                        anchor_mu=anchor_mu,
                        anchor_std=anchor_std,
                        targets=targets,
                    )
                    weighted_cand_nll = cand_w * cand_nll
                    ret_traj_loss += weighted_cand_nll
                    trajectory_loss_dict[
                        "anchor_gaussian_candidate_nll_loss"
                    ] = weighted_cand_nll
                    trajectory_loss_dict[
                        "anchor_gaussian_candidate_distance"
                    ] = cand_dist

        if bool(getattr(self._config, "anchor_contrastive_enable", False)):
            target_traj = targets.get("trajectory") if targets is not None else None
            anchor_contrastive_raw, anchor_positive_distance = self._compute_anchor_contrastive_loss(
                plan_anchor=plan_anchor,
                target_traj=target_traj,
            )
            anchor_contrastive_weight = float(
                getattr(self._config, "anchor_contrastive_weight", 1.0)
            )
            anchor_contrastive_loss = anchor_contrastive_weight * anchor_contrastive_raw
            ret_traj_loss += anchor_contrastive_loss
            trajectory_loss_dict["anchor_contrastive_loss"] = anchor_contrastive_loss
            trajectory_loss_dict["anchor_positive_distance"] = anchor_positive_distance

        gt_mode: Optional[torch.Tensor] = None
        with torch.no_grad():
            end_xy = poses_reg_list[-1][..., -1, :2]
            end_std = end_xy.std(dim=1, unbiased=False)
            trajectory_loss_dict["trajectory_endpoint_dispersion"] = end_std.mean(dim=-1).mean()
            target_traj = targets.get("trajectory") if targets is not None else None
            if target_traj is not None:
                dist = torch.linalg.norm(
                    target_traj[:, None, :, :2] - poses_reg_list[-1][..., :2],
                    dim=-1,
                ).mean(dim=-1)
                gt_mode = dist.argmin(dim=-1)
                if torch.is_tensor(poses_cls_list[-1]):
                    pred_mode = poses_cls_list[-1].argmax(dim=-1)
                elif torch.is_tensor(pdm_score):
                    pred_mode = pdm_score.argmax(dim=-1)
                else:
                    pred_mode = gt_mode
                match = (pred_mode == gt_mode).float().mean()
                batch_idx = torch.arange(dist.shape[0], device=dist.device)
                gap = (dist[batch_idx, pred_mode] - dist[batch_idx, gt_mode]).mean()
                trajectory_loss_dict["trajectory_mode_match_rate"] = match
                trajectory_loss_dict["trajectory_mode_gap"] = gap
        if hardcase_score_aux is not None:
            residual_l1 = hardcase_score_aux.get("hardcase_score_residual_l1")
            if torch.is_tensor(residual_l1):
                trajectory_loss_dict["hardcase_score_residual_l1"] = residual_l1.mean()
            residual_var = hardcase_score_aux.get("hardcase_score_residual_adapter_var")
            if torch.is_tensor(residual_var):
                trajectory_loss_dict["hardcase_score_residual_adapter_var"] = residual_var.mean()

        if torch.is_tensor(poses_cls_list[-1]):
            mode_idx = poses_cls_list[-1].argmax(dim=-1)
        elif torch.is_tensor(pdm_score):
            mode_idx = pdm_score.argmax(dim=-1)
        elif gt_mode is not None:
            mode_idx = gt_mode
        else:
            mode_idx = torch.zeros(bs, device=device, dtype=torch.long)
        mode_idx = mode_idx[...,None,None,None].repeat(1,1,self._num_poses,3)
        best_reg_generalist = torch.gather(poses_reg_list[-1], 1, mode_idx).squeeze(1)
        decoder_decorrelation_loss = torch.tensor(0.0, device=device)
        if decorrelation_losses:
            decoder_decorrelation_loss = decorrelation_losses[-1]
        return {
            "trajectory": best_reg_generalist,
            "trajectory_generalist": best_reg_generalist,
            "trajectory_specialist": best_reg_generalist,
            "trajectory_loss": ret_traj_loss,
            "trajectory_loss_dict": trajectory_loss_dict,
            "plan_anchor": plan_anchor,
            "diffusion_input_decorrelation_loss": decorrelation_loss,
            "diffusion_output_decorrelation_loss": decoder_decorrelation_loss,  
            "poses_reg": poses_reg_list[-1],
            "poses_reg_specialist": poses_reg_list[-1],
            "poses_cls": poses_cls_list[-1],
            "proposal_risk_logits": (
                torch.stack(proposal_risk_logits_list, dim=0)
                if proposal_risk_logits_list
                else None
            ),
            "proposal_risk_trajectories": proposal_risk_trajectories,
            "pdm_score": pdm_score,
            "pdm_score_generalist": pdm_score,
            "pdm_score_specialist": pdm_score_specialist,
            "pdm_score_components": pdm_score_components,
            "pdm_score_components_specialist": pdm_score_components_specialist,
            "pdm_score_cached": pdm_score_cached,
            "pdm_score_components_cached": pdm_score_components_cached,
            "anchor_mu": anchor_aux.get("anchor_mu"),
            "anchor_std": anchor_aux.get("anchor_std"),
            "mode_bev_attention_map": scorer_mode_attn_map,
            "pdm_score_risk_area_logits": (
                hardcase_score_aux.get("pdm_score_risk_area_logits")
                if hardcase_score_aux is not None
                else None
            ),
            "hardcase_score_residual": (
                hardcase_score_aux.get("hardcase_score_residual")
                if hardcase_score_aux is not None
                else None
            ),
            "hardcase_score_residual_l1": (
                hardcase_score_aux.get("hardcase_score_residual_l1")
                if hardcase_score_aux is not None
                else None
            ),
            "hardcase_score_residual_adapter_var": (
                hardcase_score_aux.get("hardcase_score_residual_adapter_var")
                if hardcase_score_aux is not None
                else None
            ),
            "hardcase_score_residual_adapter_outputs": (
                hardcase_score_aux.get("hardcase_score_residual_adapter_outputs")
                if hardcase_score_aux is not None
                else None
            ),
            "pdm_score_specialist_adapter_scores": (
                hardcase_score_aux.get("pdm_score_specialist_adapter_scores")
                if hardcase_score_aux is not None
                else None
            ),
            "pdm_score_specialist_uncertainty": (
                hardcase_score_aux.get("pdm_score_specialist_uncertainty")
                if hardcase_score_aux is not None
                else None
            ),
        }

    def forward_test(
        self,
        ego_query,
        agents_query,
        bev_feature,
        bev_spatial_shape,
        status_encoding,
        global_img,
        targets=None,
        image_tokens=None,
        lidar_tokens=None,
        score_image_tokens=None,
        score_lidar_tokens=None,
        history_tokens=None,
        risk_bev_map=None,
        anchor_mask=None,
        anchor_mask_relaxed=None,
        anchor_mask_pad=True,
        anchor_mask_relaxed_pad=True,
        anchor_strict_steps=0,
        status_feature=None,
    ) -> Dict[str, torch.Tensor]:
        step_num = 2
        bs = ego_query.shape[0]
        device = ego_query.device
        if score_image_tokens is None:
            score_image_tokens = image_tokens
        if score_lidar_tokens is None:
            score_lidar_tokens = lidar_tokens
        score_agent_tokens = None
        if getattr(self._config, "pdm_score_use_agent_tokens", False):
            score_agent_tokens = agents_query.detach()
        direct_proposal_decoder = (
            str(getattr(self._config, "trajectory_decoder_type", "diffusion")).lower()
            == "proformer"
        )
        self.diffusion_scheduler.set_timesteps(1000, device)
        step_ratio = 20 / step_num
        roll_timesteps = (np.arange(0, step_num) * step_ratio).round()[::-1].copy().astype(np.int64)
        roll_timesteps = torch.from_numpy(roll_timesteps).to(device)
        if direct_proposal_decoder or (self._config.anchor_free and self._config.anchor_free_skip_diffusion_noise):
            roll_timesteps = torch.tensor([8], device=device, dtype=torch.long)
        multisample_enable = bool(
            getattr(self._config, "inference_multisample_enable", False)
        )
        multisample_count = int(
            getattr(self._config, "inference_multisample_count", 1)
        )
        if not multisample_enable:
            multisample_count = 1
        multisample_count = max(1, multisample_count)
        force_anchor_sample = bool(
            getattr(self._config, "inference_multisample_force_anchor_sample", True)
        ) and multisample_count > 1

        sampled_plan_anchors: List[torch.Tensor] = []
        sampled_poses_reg: List[torch.Tensor] = []
        sampled_poses_cls: List[torch.Tensor] = []
        sampled_anchor_mu: List[torch.Tensor] = []
        sampled_anchor_std: List[torch.Tensor] = []
        first_anchor_aux: Dict[str, torch.Tensor] = {}
        proposal_risk_trajectories = None
        proposal_risk_logits_list = None

        for sample_idx in range(multisample_count):
            plan_anchor_i, anchor_aux_i = self._build_plan_anchor(
                batch_size=bs,
                device=device,
                targets=targets,
                anchor_mask=anchor_mask,
                anchor_mask_relaxed=anchor_mask_relaxed,
                anchor_mask_pad=anchor_mask_pad,
                anchor_mask_relaxed_pad=anchor_mask_relaxed_pad,
                anchor_strict_steps=anchor_strict_steps,
                status_feature=status_feature,
                status_encoding=status_encoding,
                ego_query=ego_query,
                force_sample=force_anchor_sample,
            )  # [bs, modes, poses, 2]
            if sample_idx == 0:
                first_anchor_aux = anchor_aux_i
                self._maybe_visualize_anchor_bundle(
                    plan_anchor=plan_anchor_i,
                    anchor_aux=anchor_aux_i,
                    targets=targets,
                    stage="test",
                )

            img = self.norm_odo(plan_anchor_i)
            if not direct_proposal_decoder and not (
                self._config.anchor_free and self._config.anchor_free_skip_diffusion_noise
            ):
                noise = torch.randn(img.shape, device=device)
                trunc_timesteps = torch.ones((bs,), device=device, dtype=torch.long) * 8
                img = self.diffusion_scheduler.add_noise(
                    original_samples=img,
                    noise=noise,
                    timesteps=trunc_timesteps,
                )
            poses_reg_i = None
            poses_cls_i = None
            for k in roll_timesteps[:]:
                x_boxes = torch.clamp(img, min=-1, max=1)
                noisy_traj_points = self.denorm_odo(x_boxes)

                # 2. initialize trajectory tokens
                traj_feature = self._build_traj_tokens(
                    noisy_traj_points=noisy_traj_points,
                    status_encoding=status_encoding,
                )

                timesteps = k
                if not torch.is_tensor(timesteps):
                    timesteps = torch.tensor(
                        [timesteps], dtype=torch.long, device=img.device
                    )
                elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
                    timesteps = timesteps[None].to(img.device)

                # 3. embed the timesteps
                time_embed = None
                if self.time_mlp is not None:
                    timesteps = timesteps.expand(img.shape[0])
                    time_embed = self.time_mlp(timesteps)
                    time_embed = time_embed.view(bs, 1, -1)

                # 4. begin the stacked decoder
                decode_outputs = self._decode_proposals(
                    traj_feature,
                    noisy_traj_points,
                    bev_feature,
                    bev_spatial_shape,
                    agents_query,
                    ego_query,
                    time_embed,
                    status_encoding,
                    global_img,
                    image_tokens=image_tokens,
                    lidar_tokens=lidar_tokens,
                    history_tokens=history_tokens,
                    risk_bev_map=risk_bev_map,
                )
                proposal_risk_logits_list = None
                if isinstance(decode_outputs, (tuple, list)) and len(decode_outputs) == 4:
                    poses_reg_list, poses_cls_list, _, proposal_risk_logits_list = decode_outputs
                else:
                    poses_reg_list, poses_cls_list, _ = decode_outputs
                proposal_risk_trajectories = torch.stack(
                    [poses_reg.clone() for poses_reg in poses_reg_list], dim=0
                )
                poses_reg_i = poses_reg_list[-1]
                poses_cls_i = poses_cls_list[-1]
                x_start = poses_reg_i[..., :2]
                if not direct_proposal_decoder:
                    x_start = self.norm_odo(x_start)
                    img = self.diffusion_scheduler.step(
                        model_output=x_start,
                        timestep=k,
                        sample=img,
                    ).prev_sample

            if poses_reg_i is None:
                raise RuntimeError("Diffusion rollout produced no trajectory prediction.")
            poses_reg_i, _, _, _ = self._apply_refinement(
                poses_reg=poses_reg_i,
                status_encoding=status_encoding,
                bev_feature=bev_feature,
                risk_bev_map=risk_bev_map,
            )
            if poses_reg_i is None:
                raise RuntimeError("Refinement failed to produce trajectory prediction.")
            sampled_plan_anchors.append(plan_anchor_i)
            sampled_poses_reg.append(poses_reg_i)
            if poses_cls_i is not None:
                sampled_poses_cls.append(poses_cls_i)
            anchor_mu_i = anchor_aux_i.get("anchor_mu")
            if isinstance(anchor_mu_i, torch.Tensor):
                sampled_anchor_mu.append(anchor_mu_i)
            anchor_std_i = anchor_aux_i.get("anchor_std")
            if isinstance(anchor_std_i, torch.Tensor):
                sampled_anchor_std.append(anchor_std_i)

        if len(sampled_poses_reg) == 1:
            plan_anchor = sampled_plan_anchors[0]
            poses_reg = sampled_poses_reg[0]
            poses_cls = sampled_poses_cls[0] if sampled_poses_cls else None
            anchor_mu = first_anchor_aux.get("anchor_mu")
            anchor_std = first_anchor_aux.get("anchor_std")
        else:
            plan_anchor = torch.cat(sampled_plan_anchors, dim=1)
            poses_reg = torch.cat(sampled_poses_reg, dim=1)
            poses_cls = (
                torch.cat(sampled_poses_cls, dim=1) if sampled_poses_cls else None
            )
            anchor_mu = (
                torch.cat(sampled_anchor_mu, dim=1)
                if len(sampled_anchor_mu) == len(sampled_plan_anchors)
                else first_anchor_aux.get("anchor_mu")
            )
            anchor_std = (
                torch.cat(sampled_anchor_std, dim=1)
                if len(sampled_anchor_std) == len(sampled_plan_anchors)
                else first_anchor_aux.get("anchor_std")
            )

        pdm_score = None
        pdm_score_specialist = None
        pdm_score_components = None
        pdm_score_components_specialist = None
        pdm_score_cached = None
        pdm_score_components_cached = None
        hardcase_score_aux: Optional[Dict[str, torch.Tensor]] = None
        scorer_mode_attn_map: Optional[torch.Tensor] = None
        scorer_mode_context: Optional[torch.Tensor] = None
        if bool(getattr(self._config, "mode_bev_attention_enable", False)) and bool(
            getattr(self._config, "mode_bev_attention_use_scorer", True)
        ):
            scorer_mode_attn_map, scorer_mode_context = self._compute_mode_bev_attention(
                poses_reg=poses_reg,
                bev_feature=bev_feature,
                status_encoding=status_encoding,
                risk_bev_map=risk_bev_map,
            )
        if self._pdm_score_head is not None:
            pdm_score_raw, hardcase_score_aux = self._pdm_score_head(
                poses_reg.detach(),
                bev_feature,
                image_tokens=score_image_tokens,
                lidar_tokens=score_lidar_tokens,
                agent_tokens=score_agent_tokens,
                mode_context=scorer_mode_context,
            )
            pdm_score_specialist_raw = None
            if hardcase_score_aux is not None:
                pdm_score_specialist_raw = hardcase_score_aux.get("pdm_score_specialist")
            pdm_score = pdm_score_raw
            if pdm_score is not None and pdm_score.dim() == 3:
                pdm_score_components = pdm_score
                pdm_score = self._aggregate_pdm_components_inference(
                    pdm_score_components
                )
            if torch.is_tensor(pdm_score_specialist_raw):
                pdm_score_specialist = pdm_score_specialist_raw
                if pdm_score_specialist.dim() == 3:
                    pdm_score_components_specialist = pdm_score_specialist
                    pdm_score_specialist = self._aggregate_pdm_components_inference(
                        pdm_score_components_specialist
                    )
            else:
                pdm_score_specialist = pdm_score
            if (
                self._config.pdm_score_use_cached_poses
                and targets is not None
                and "poses_reg" in targets
            ):
                cached_poses = targets["poses_reg"]
                if cached_poses.dim() == 3:
                    cached_poses = cached_poses.unsqueeze(0)
                cached_poses = cached_poses.to(
                    device=poses_reg.device,
                    dtype=poses_reg.dtype,
                )
                cached_mode_context: Optional[torch.Tensor] = None
                if bool(getattr(self._config, "mode_bev_attention_enable", False)) and bool(
                    getattr(self._config, "mode_bev_attention_use_scorer", True)
                ):
                    _, cached_mode_context = self._compute_mode_bev_attention(
                        poses_reg=cached_poses,
                        bev_feature=bev_feature,
                        status_encoding=status_encoding,
                        risk_bev_map=risk_bev_map,
                    )
                pdm_score_cached, _ = self._pdm_score_head(
                    cached_poses.detach(),
                    bev_feature,
                    image_tokens=score_image_tokens,
                    lidar_tokens=score_lidar_tokens,
                    agent_tokens=score_agent_tokens,
                    mode_context=cached_mode_context,
                )
                if pdm_score_cached is not None and pdm_score_cached.dim() == 3:
                    pdm_score_components_cached = pdm_score_cached
                    pdm_score_cached = self._aggregate_pdm_components_inference(
                        pdm_score_components_cached
                    )
        dedup_scores = pdm_score if pdm_score is not None else poses_cls
        dedup_keep_mask, dedup_keep_counts = self._build_inference_dedup_keep_mask(
            poses_reg=poses_reg,
            mode_scores=dedup_scores,
        )
        if dedup_keep_mask is not None:
            if poses_cls is not None:
                poses_cls = poses_cls.masked_fill(~dedup_keep_mask, float("-inf"))
            if pdm_score is not None:
                pdm_score = pdm_score.masked_fill(~dedup_keep_mask, float("-inf"))
            if pdm_score_specialist is not None:
                pdm_score_specialist = pdm_score_specialist.masked_fill(
                    ~dedup_keep_mask, float("-inf")
                )
        mode_idx = None
        use_pdm_select = (
            pdm_score is not None and self._config.pdm_score_use_for_selection
        )
        if use_pdm_select:
            topk_select = int(getattr(self._config, "pdm_score_select_topk", 0) or 0)
            if poses_cls is not None and 0 < topk_select < poses_cls.shape[1]:
                topk_idx = torch.topk(poses_cls, topk_select, dim=-1).indices
                topk_scores = torch.gather(pdm_score, 1, topk_idx)
                best_in_topk = topk_scores.argmax(dim=-1, keepdim=True)
                mode_idx = topk_idx.gather(1, best_in_topk).squeeze(1)
            else:
                mode_idx = pdm_score.argmax(dim=-1)
        if mode_idx is None:
            if poses_cls is not None:
                mode_idx = poses_cls.argmax(dim=-1)
            elif pdm_score is not None:
                mode_idx = pdm_score.argmax(dim=-1)
            else:
                mode_idx = torch.zeros(bs, device=device, dtype=torch.long)
        selected_mode_idx = mode_idx
        mode_idx_generalist = mode_idx

        mode_idx_specialist = mode_idx_generalist
        use_pdm_select_specialist = (
            pdm_score_specialist is not None and self._config.pdm_score_use_for_selection
        )
        if use_pdm_select_specialist:
            topk_select = int(getattr(self._config, "pdm_score_select_topk", 0) or 0)
            if poses_cls is not None and 0 < topk_select < poses_cls.shape[1]:
                topk_idx = torch.topk(poses_cls, topk_select, dim=-1).indices
                topk_scores = torch.gather(pdm_score_specialist, 1, topk_idx)
                best_in_topk = topk_scores.argmax(dim=-1, keepdim=True)
                mode_idx_specialist = topk_idx.gather(1, best_in_topk).squeeze(1)
            else:
                mode_idx_specialist = pdm_score_specialist.argmax(dim=-1)
        elif pdm_score_specialist is not None:
            mode_idx_specialist = pdm_score_specialist.argmax(dim=-1)

        gather_generalist = mode_idx_generalist[..., None, None, None].repeat(
            1, 1, self._num_poses, 3
        )
        gather_specialist = mode_idx_specialist[..., None, None, None].repeat(
            1, 1, self._num_poses, 3
        )
        best_reg_generalist = torch.gather(poses_reg, 1, gather_generalist).squeeze(1)
        best_reg_specialist = torch.gather(poses_reg, 1, gather_specialist).squeeze(1)
        trajectory_loss_dict = {}
        with torch.no_grad():
            if dedup_keep_mask is not None:
                trajectory_loss_dict["inference_dedup_keep_ratio"] = (
                    dedup_keep_mask.float().mean()
                )
                if dedup_keep_counts is not None:
                    trajectory_loss_dict["inference_dedup_keep_count"] = (
                        dedup_keep_counts.float().mean()
                    )
            end_xy = poses_reg[..., -1, :2]
            end_std = end_xy.std(dim=1, unbiased=False)
            trajectory_loss_dict["trajectory_endpoint_dispersion"] = end_std.mean(dim=-1).mean()
            if targets is not None and "trajectory" in targets:
                target_traj = targets["trajectory"]
                dist = torch.linalg.norm(
                    target_traj[:, None, :, :2] - poses_reg[..., :2],
                    dim=-1,
                ).mean(dim=-1)
                gt_mode = dist.argmin(dim=-1)
                pred_mode = (
                    poses_cls.argmax(dim=-1) if poses_cls is not None else selected_mode_idx
                )
                match = (pred_mode == gt_mode).float().mean()
                batch_idx = torch.arange(dist.shape[0], device=dist.device)
                gap = (dist[batch_idx, pred_mode] - dist[batch_idx, gt_mode]).mean()
                trajectory_loss_dict["trajectory_mode_match_rate"] = match
                trajectory_loss_dict["trajectory_mode_gap"] = gap
        if hardcase_score_aux is not None:
            residual_l1 = hardcase_score_aux.get("hardcase_score_residual_l1")
            if torch.is_tensor(residual_l1):
                trajectory_loss_dict["hardcase_score_residual_l1"] = residual_l1.mean()
            residual_var = hardcase_score_aux.get("hardcase_score_residual_adapter_var")
            if torch.is_tensor(residual_var):
                trajectory_loss_dict["hardcase_score_residual_adapter_var"] = residual_var.mean()
        return {
            "trajectory": best_reg_generalist,
            "trajectory_generalist": best_reg_generalist,
            "trajectory_specialist": best_reg_specialist,
            "plan_anchor": plan_anchor,
            "poses_reg": poses_reg,
            "poses_reg_specialist": poses_reg,
            "poses_cls": poses_cls,
            "proposal_risk_logits": (
                torch.stack(proposal_risk_logits_list, dim=0)
                if proposal_risk_logits_list
                else None
            ),
            "proposal_risk_trajectories": proposal_risk_trajectories,
            "pdm_score": pdm_score,
            "pdm_score_generalist": pdm_score,
            "pdm_score_specialist": pdm_score_specialist,
            "pdm_score_components": pdm_score_components,
            "pdm_score_components_specialist": pdm_score_components_specialist,
            "pdm_score_cached": pdm_score_cached,
            "pdm_score_components_cached": pdm_score_components_cached,
            "trajectory_loss_dict": trajectory_loss_dict,
            "mode_bev_attention_map": scorer_mode_attn_map,
            "pdm_score_risk_area_logits": (
                hardcase_score_aux.get("pdm_score_risk_area_logits")
                if hardcase_score_aux is not None
                else None
            ),
            "anchor_mu": anchor_mu,
            "anchor_std": anchor_std,
            "hardcase_score_residual": (
                hardcase_score_aux.get("hardcase_score_residual")
                if hardcase_score_aux is not None
                else None
            ),
            "hardcase_score_residual_l1": (
                hardcase_score_aux.get("hardcase_score_residual_l1")
                if hardcase_score_aux is not None
                else None
            ),
            "hardcase_score_residual_adapter_var": (
                hardcase_score_aux.get("hardcase_score_residual_adapter_var")
                if hardcase_score_aux is not None
                else None
            ),
            "hardcase_score_residual_adapter_outputs": (
                hardcase_score_aux.get("hardcase_score_residual_adapter_outputs")
                if hardcase_score_aux is not None
                else None
            ),
            "pdm_score_specialist_adapter_scores": (
                hardcase_score_aux.get("pdm_score_specialist_adapter_scores")
                if hardcase_score_aux is not None
                else None
            ),
            "pdm_score_specialist_uncertainty": (
                hardcase_score_aux.get("pdm_score_specialist_uncertainty")
                if hardcase_score_aux is not None
                else None
            ),
        }

def plot_trajectory_anchors(trajectory_anchors, pi=None, file_name="trajectory_anchors.png"):
    """
    可视化轨迹锚点tensor并保存为PNG图像
    
    Args:
        trajectory_anchors: 轨迹锚点tensor，形状为 (bs, modes, poses, 2)
        file_name: 保存的文件名
    """
    # 确保输出目录存在
    output_dir = "/home/xqf/DiffusionDrive-main/free_anchors"
    os.makedirs(output_dir, exist_ok=True)
    
    # 将tensor转换为numpy数组
    if isinstance(trajectory_anchors, torch.Tensor):
        trajectory_anchors = trajectory_anchors.detach().cpu().numpy()
    
    # 获取维度信息
    bs, modes, poses, _ = trajectory_anchors.shape
    # import pdb; pdb.set_trace()
    if pi is not None:
        max_prob_indices = pi.argmax(dim=1)
        max_prob = pi[torch.arange(bs), max_prob_indices]
    
    # 为每个batch创建一个图像
    for batch_idx in range(5):
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        
        # 为每个mode绘制轨迹
        for mode_idx in range(modes):
            # 获取当前batch和mode的轨迹数据
            trajectory = trajectory_anchors[batch_idx, mode_idx, :, :]  # [poses, 2]
            
            # 绘制轨迹点
            ax.plot(trajectory[:, 0], trajectory[:, 1], marker='o', markersize=4, linewidth=2, 
                    label=f'Mode {mode_idx}')
            
            # 标记起始点和结束点
            ax.scatter(trajectory[0, 0], trajectory[0, 1], color='green', s=50, marker='s', 
                      label='Start' if mode_idx == 0 else "")
            ax.scatter(trajectory[-1, 0], trajectory[-1, 1], color='red', s=50, marker='s',
                      label='End' if mode_idx == 0 else "")
        
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        if pi is not None:
            ax.set_title(f'Trajectory Anchors - Batch {batch_idx} - best_Mode {max_prob_indices[batch_idx]} - prob {max_prob[batch_idx]:.3f}')
        else:
            ax.set_title(f'Trajectory Anchors - Batch {batch_idx}')
        ax.legend()
        ax.grid(True)
        
        # 保存图像
        save_path = os.path.join(output_dir, f"batch_{batch_idx}_{file_name}")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
    print(f"Saved trajectory anchors visualization to {output_dir}")
