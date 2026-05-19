from typing import Dict
import numpy as np
import torch
import torch.nn as nn
import copy
import math
from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig
from navsim.agents.diffusiondrive.transfuser_backbone import TransfuserBackbone
from navsim.agents.diffusiondrive.transfuser_features import BoundingBox2DIndex
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
class V3TransfuserModel(nn.Module):
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
        self._backbone = TransfuserBackbone(config)

        self._keyval_embedding = nn.Embedding(8**2 + 1, config.tf_d_model)  # 8x8 feature grid + trajectory -> (64+1, d_model)一组可学习的位置编码参数
        self._query_embedding = nn.Embedding(sum(self._query_splits), config.tf_d_model)

        # usually, the BEV features are variable in size.
        self._bev_downscale = nn.Conv2d(512, config.tf_d_model, kernel_size=1)  
        self._status_encoding = nn.Linear(4 + 2 + 2, config.tf_d_model)
        self._ego_history_encoder = None
        self._ego_history_pos_emb = None
        self._ego_history_pos_proj = None
        if config.include_ego_history:
            self._ego_history_encoder = nn.GRU(
                input_size=3, hidden_size=config.tf_d_model, batch_first=True
            )
            self._ego_history_pos_emb = SinusoidalPosEmb(config.tf_d_model)
            self._ego_history_pos_proj = nn.Sequential(
                nn.Linear(config.tf_d_model, config.tf_d_model),
                nn.ReLU(inplace=True),
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
        # import pdb; pdb.set_trace()
        # 2. 用backboen提取BEV特征（不做修改），相机＋雷达
        bev_feature_upscale, bev_feature, image_feature, lidar_feature = self._backbone(
            camera_feature, lidar_feature
        )

        bev_feature_upscale_sem = bev_feature_upscale
        bev_feature_upscale_traj = bev_feature_upscale
        bev_feature_traj = bev_feature
        image_feature_traj = image_feature.detach() if image_feature is not None else None
        lidar_feature_traj = lidar_feature.detach() if lidar_feature is not None else None
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
                if self._ego_history_pos_emb is not None and self._ego_history_pos_proj is not None:
                    hist_len = history_output.shape[1]
                    pos = torch.arange(hist_len, device=history_output.device, dtype=history_output.dtype)
                    pos = pos.unsqueeze(0).expand(history_output.shape[0], -1).reshape(-1)
                    pos_emb = self._ego_history_pos_emb(pos)
                    pos_emb = pos_emb.view(history_output.shape[0], hist_len, -1)
                    pos_emb = self._ego_history_pos_proj(pos_emb)
                    history_output = history_output + pos_emb
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

        anchor_mask = feasible_masks["feasible_area_mask"] if feasible_masks else None
        anchor_mask_relaxed = None
        anchor_mask_pad = True
        anchor_mask_relaxed_pad = True
        anchor_strict_steps = int(
            getattr(self._config, "reachability_anchor_strict_steps", 0)
        )
        if self._config.reachability_use_bicycle and self._config.reachability_use_for_anchor:
            if physical_area_mask is not None:
                anchor_mask = physical_area_mask
                anchor_mask_pad = False
                if feasible_masks is not None:
                    anchor_mask_relaxed = feasible_masks.get("feasible_area_mask")
            elif reachability_mask is not None and anchor_mask is None:
                anchor_mask = reachability_mask
                anchor_mask_pad = False

        trajectory = self._trajectory_head(
            trajectory_query,
            agents_query,
            cross_bev_feature,
            bev_spatial_shape,
            status_encoding[:, None],
            targets=targets,
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
            if global_cond.dim() == 2:
                global_cond = global_cond.unsqueeze(1)
            global_feature = time_embed + global_cond
        else:
            global_feature = time_embed
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
        self.dropout = nn.Dropout(config.tf_dropout)
        self.self_attn = nn.MultiheadAttention(
            config.tf_d_model,
            config.tf_num_head,
            dropout=config.tf_dropout,
            batch_first=True,
        )
        self.cross_bev_attention = GridSampleCrossBEVAttention(
            config.tf_d_model,
            config.tf_num_head,
            num_points=num_poses,
            config=config,
            in_bev_dims=256,
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
        self._decorrelation_weight = config.diff_output_decorrelation_weight
        self.norm_self = nn.LayerNorm(config.tf_d_model)
        self.norm_bev = nn.LayerNorm(config.tf_d_model) if config.denoise_norm_bev else None
        self.norm_img = nn.LayerNorm(config.tf_d_model) if config.denoise_use_image_tokens else None
        self.norm_lidar = nn.LayerNorm(config.tf_d_model) if config.denoise_use_lidar_tokens else None
        self.norm_history = nn.LayerNorm(config.tf_d_model) if config.denoise_use_history_tokens else None
        self.time_modulation = None
        if config.denoise_use_time_embed:
            self.time_modulation = ModulationLayer(config.tf_d_model, config.tf_d_model)
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
        history_cond = None
        if history_tokens is not None:
            history_cond = history_tokens.mean(dim=1, keepdim=True)
        cond = status_encoding
        if cond is not None and cond.dim() == 2:
            cond = cond.unsqueeze(1)
        if history_cond is not None:
            cond = history_cond if cond is None else cond + history_cond

        if self.time_modulation is not None and time_embed is not None:
            traj_feature = self.time_modulation(
                traj_feature,
                time_embed,
                global_cond=cond,
            )
        if history_tokens is not None:
            all_tokens = torch.cat([traj_feature, history_tokens], dim=1)
            all_tokens = all_tokens + self.dropout(
                self.self_attn(all_tokens, all_tokens, all_tokens)[0]
            )
            all_tokens = self.norm_self(all_tokens)
            traj_feature = all_tokens[:, : traj_feature.shape[1], :]
        else:
            traj_feature = traj_feature + self.dropout(
                self.self_attn(traj_feature, traj_feature, traj_feature)[0]
            )
            traj_feature = self.norm_self(traj_feature)

        if history_tokens is not None and self.cross_history_attention is not None and self.norm_history is not None:
            traj_feature = traj_feature + self.dropout(
                self.cross_history_attention(traj_feature, history_tokens, history_tokens)[0]
            )
            traj_feature = self.norm_history(traj_feature)

        if lidar_tokens is not None and self.cross_lidar_attention is not None and self.norm_lidar is not None:
            traj_feature = traj_feature + self.dropout(
                self.cross_lidar_attention(traj_feature, lidar_tokens, lidar_tokens)[0]
            )
            traj_feature = self.norm_lidar(traj_feature)

        traj_feature = self.cross_bev_attention(
            traj_feature,
            noisy_traj_points,
            bev_feature,
            bev_spatial_shape,
        )
        if self.norm_bev is not None:
            traj_feature = self.norm_bev(traj_feature)

        if image_tokens is not None and self.cross_img_attention is not None and self.norm_img is not None:
            traj_feature = traj_feature + self.dropout(
                self.cross_img_attention(traj_feature, image_tokens, image_tokens)[0]
            )
            traj_feature = self.norm_img(traj_feature)

        decorrelation_loss = torch.tensor(0.0, device=traj_feature.device)
        if self.training and self._decorrelation_weight > 0:
            decorrelation_loss = self._decorrelation_loss(traj_feature)

        poses_reg, poses_cls = self.task_decoder(traj_feature)
        poses_reg[..., :2] = poses_reg[..., :2] + noisy_traj_points
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
        if self._use_components:
            self._score_heads = nn.ModuleList(
                [nn.Linear(config.tf_d_ffn, 1) for _ in range(self._num_components)]
            )
            self._score_head = None
        else:
            self._score_head = nn.Linear(config.tf_d_ffn, 1)
            self._score_heads = None

    def forward(
        self,
        traj_points: torch.Tensor,
        bev_feature: torch.Tensor,
        image_tokens: Optional[torch.Tensor] = None,
        lidar_tokens: Optional[torch.Tensor] = None,
        agent_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
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
        else:
            point_embed = self._traj_point_embed(coords)
            time_idx = torch.arange(num_poses, device=coords.device)
            time_embed = self._traj_time_embed(time_idx)[None, None, :, :]
            traj_tokens = self._traj_token_norm((point_embed + time_embed).mean(dim=2))

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

        features = self._score_trunk(traj_tokens)
        if self._use_components and self._score_heads is not None:
            scores = torch.cat([head(features) for head in self._score_heads], dim=-1)
            return scores.view(bs, num_modes, self._num_components)
        if self._score_head is None:
            raise RuntimeError("TrajectoryScorer missing score head")
        scores = self._score_head(features).view(bs, num_modes)
        return scores

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

        self.plan_anchor = nn.Parameter(
            torch.tensor(plan_anchor, dtype=torch.float32),
            requires_grad=False,
        ) # [modes, poses, 2] 可学习的计划锚点
        self.plan_anchor_encoder = nn.Sequential(
            *linear_relu_ln(d_model, 1, 1,512), # Linear(512, d_model) → ReLU → LayerNorm(d_model)的组合
            nn.Linear(d_model, d_model),
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
        self.diff_decoder = CustomTransformerDecoder(diff_decoder_layer, 2)     # 复制2个注意力解码层

        self.loss_computer = LossComputer(config)

    def _aggregate_pdm_components(self, pdm_score_components: torch.Tensor) -> torch.Tensor:
        weights = getattr(self, "_pdm_component_weights", None)
        if weights is None or weights.numel() != pdm_score_components.shape[-1]:
            weights = torch.ones(
                pdm_score_components.shape[-1],
                device=pdm_score_components.device,
                dtype=pdm_score_components.dtype,
            )
        else:
            weights = weights.to(
                device=pdm_score_components.device, dtype=pdm_score_components.dtype
            )

        use_logsigmoid = bool(
            getattr(self._config, "pdm_score_use_logsigmoid_aggregate", False)
        )
        if not use_logsigmoid:
            return (pdm_score_components * weights.view(1, 1, -1)).sum(dim=-1)

        # DrivoR-style multiplicative aggregation in log space.
        # Component order: [noc, dac, progress, ttc, comfort, ddc]
        w = weights.clamp_min(0.0)
        noc = pdm_score_components[..., 0]
        dac = pdm_score_components[..., 1]
        progress = pdm_score_components[..., 2]
        ttc = pdm_score_components[..., 3]
        comfort = pdm_score_components[..., 4]
        ddc = pdm_score_components[..., 5]

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
    ) -> torch.Tensor:
        if not self._config.anchor_free:
            return self.plan_anchor.unsqueeze(0).repeat(batch_size, 1, 1, 1).to(device)

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
                return self.plan_anchor.unsqueeze(0).repeat(batch_size, 1, 1, 1).to(device)
            mask = relaxed_mask
            relaxed_mask = None

        if self._config.anchor_free_gaussian:
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
            if relaxed_mask is not None:
                if self._config.anchor_free_gaussian:
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
        return plan_anchor

    def _select_anchor_mask(
        self,
        batch_size: int,
        device: torch.device,
        targets: Optional[Dict[str, torch.Tensor]] = None,
        anchor_mask: Optional[torch.Tensor] = None,
        pad_m_override: Optional[float] = None,
    ) -> Optional[torch.Tensor]:
        mask = anchor_mask
        use_target_mask = self._config.anchor_free_use_target and targets is not None
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
            targets=targets,
            anchor_mask=anchor_mask,
            anchor_mask_relaxed=anchor_mask_relaxed,
            anchor_mask_pad=anchor_mask_pad,
            anchor_mask_relaxed_pad=anchor_mask_relaxed_pad,
            anchor_strict_steps=anchor_strict_steps,
            status_feature=status_feature,
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
        # 1. add truncated noise to the plan anchor
        plan_anchor = self._build_plan_anchor(
            batch_size=bs,
            device=device,
            targets=targets,
            anchor_mask=anchor_mask,
            anchor_mask_relaxed=anchor_mask_relaxed,
            anchor_mask_pad=anchor_mask_pad,
            anchor_mask_relaxed_pad=anchor_mask_relaxed_pad,
            anchor_strict_steps=anchor_strict_steps,
            status_feature=status_feature,
        )    # [bs, modes, poses, 2] anchor from feasible area
        # plot_trajectory_anchors(plan_anchor,file_name="plan_anchor.png")
        if self._config.anchor_free and self._config.anchor_free_skip_diffusion_noise:
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
        # 2. proj noisy_traj_points to the query
        traj_pos_embed = gen_sineembed_for_position(noisy_traj_points,hidden_dim=64)    # [bs, modes, poses, 64] 带噪（锚点）轨迹的位置编码
        traj_pos_embed = traj_pos_embed.flatten(-2)     # 后两维展平
        traj_feature = self.plan_anchor_encoder(traj_pos_embed)     # [bs,20,d_model] 带噪（锚点）轨迹的特征编码
        traj_feature = traj_feature.view(bs,ego_fut_mode,-1)     # [bs,20,d_model] 带噪（锚点）轨迹的特征编码
        # 3. embed the timesteps, 时间步的嵌入向量
        time_embed = None
        if self.time_mlp is not None:
            time_embed = self.time_mlp(timesteps)     # [bs,d_model]
            time_embed = time_embed.view(bs,1,-1)     # [bs,1,d_model] 时间步的嵌入向量


        # 4. begin the stacked decoder, 多层解码器去噪（加噪轨迹特征，加噪轨迹点，bev特征，bev空间特征（x），周车q，自车q，加噪时间嵌入，状态嵌入（x））
        decorrelation_loss = torch.tensor(0.0, device=device)
        if self._config.diff_input_decorrelation_weight > 0:
            decorrelation_loss = self._diff_input_decorrelation_loss(traj_feature)
        poses_reg_list, poses_cls_list, decorrelation_losses = self.diff_decoder(
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
        )
        decoder_decorrelation_loss = torch.tensor(0.0, device=device)
        if decorrelation_losses:
            decoder_decorrelation_loss = decorrelation_losses[-1]
        pdm_score = None
        pdm_score_components = None
        pdm_score_cached = None
        pdm_score_components_cached = None
        if self._pdm_score_head is not None:
            pdm_score = self._pdm_score_head(
                poses_reg_list[-1].detach(),
                bev_feature,
                image_tokens=score_image_tokens,
                lidar_tokens=score_lidar_tokens,
                agent_tokens=score_agent_tokens,
            )
            if pdm_score is not None and pdm_score.dim() == 3:
                pdm_score_components = pdm_score
                pdm_score = self._aggregate_pdm_components(pdm_score_components)
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
                pdm_score_cached = self._pdm_score_head(
                    cached_poses.detach(),
                    bev_feature,
                    image_tokens=score_image_tokens,
                    lidar_tokens=score_lidar_tokens,
                    agent_tokens=score_agent_tokens,
                )
                if pdm_score_cached is not None and pdm_score_cached.dim() == 3:
                    pdm_score_components_cached = pdm_score_cached
                    pdm_score_cached = self._aggregate_pdm_components(
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
                pred_mode = poses_cls_list[-1].argmax(dim=-1)
                match = (pred_mode == gt_mode).float().mean()
                batch_idx = torch.arange(dist.shape[0], device=dist.device)
                gap = (dist[batch_idx, pred_mode] - dist[batch_idx, gt_mode]).mean()
                trajectory_loss_dict["trajectory_mode_match_rate"] = match
                trajectory_loss_dict["trajectory_mode_gap"] = gap

        mode_idx = poses_cls_list[-1].argmax(dim=-1)
        mode_idx = mode_idx[...,None,None,None].repeat(1,1,self._num_poses,3)
        best_reg = torch.gather(poses_reg_list[-1], 1, mode_idx).squeeze(1)
        return {
            "trajectory": best_reg,
            "trajectory_loss": ret_traj_loss,
            "trajectory_loss_dict": trajectory_loss_dict,
            "plan_anchor": plan_anchor,
            "diffusion_input_decorrelation_loss": decorrelation_loss,
            "diffusion_output_decorrelation_loss": decoder_decorrelation_loss,
            "poses_reg": poses_reg_list[-1],
            "poses_cls": poses_cls_list[-1],
            "pdm_score": pdm_score,
            "pdm_score_components": pdm_score_components,
            "pdm_score_cached": pdm_score_cached,
            "pdm_score_components_cached": pdm_score_components_cached,
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
        self.diffusion_scheduler.set_timesteps(1000, device)
        step_ratio = 20 / step_num
        roll_timesteps = (np.arange(0, step_num) * step_ratio).round()[::-1].copy().astype(np.int64)
        roll_timesteps = torch.from_numpy(roll_timesteps).to(device)
        if self._config.anchor_free and self._config.anchor_free_skip_diffusion_noise:
            roll_timesteps = torch.tensor([8], device=device, dtype=torch.long)

        # 1. add truncated noise to the plan anchor
        plan_anchor = self._build_plan_anchor(
            batch_size=bs,
            device=device,
            targets=targets,
            anchor_mask=anchor_mask,
            anchor_mask_relaxed=anchor_mask_relaxed,
            anchor_mask_pad=anchor_mask_pad,
            anchor_mask_relaxed_pad=anchor_mask_relaxed_pad,
            anchor_strict_steps=anchor_strict_steps,
            status_feature=status_feature,
        )    # [bs, modes, poses, 2] anchor from feasible area
        img = self.norm_odo(plan_anchor)
        if not (self._config.anchor_free and self._config.anchor_free_skip_diffusion_noise):
            noise = torch.randn(img.shape, device=device)
            trunc_timesteps = torch.ones((bs,), device=device, dtype=torch.long) * 8
            img = self.diffusion_scheduler.add_noise(
                original_samples=img,
                noise=noise,
                timesteps=trunc_timesteps
            )
        noisy_trajs = self.denorm_odo(img)
        # plot_trajectory_anchors(noisy_trajs,file_name="noisy_traj_points.png")
        ego_fut_mode = img.shape[1]
        for k in roll_timesteps[:]:
            x_boxes = torch.clamp(img, min=-1, max=1)
            noisy_traj_points = self.denorm_odo(x_boxes)

            # 2. proj noisy_traj_points to the query
            traj_pos_embed = gen_sineembed_for_position(noisy_traj_points,hidden_dim=64)
            traj_pos_embed = traj_pos_embed.flatten(-2)
            traj_feature = self.plan_anchor_encoder(traj_pos_embed)
            traj_feature = traj_feature.view(bs,ego_fut_mode,-1)

            timesteps = k
            if not torch.is_tensor(timesteps):
                timesteps = torch.tensor([timesteps], dtype=torch.long, device=img.device)
            elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
                timesteps = timesteps[None].to(img.device)

            # 3. embed the timesteps
            time_embed = None
            if self.time_mlp is not None:
                timesteps = timesteps.expand(img.shape[0])
                time_embed = self.time_mlp(timesteps)
                time_embed = time_embed.view(bs,1,-1)

            # 4. begin the stacked decoder ?????
            poses_reg_list, poses_cls_list, _ = self.diff_decoder(
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
            )
            poses_reg = poses_reg_list[-1]
            poses_cls = poses_cls_list[-1]
            x_start = poses_reg[...,:2]
            x_start = self.norm_odo(x_start)
            img = self.diffusion_scheduler.step(
                model_output=x_start,
                timestep=k,
                sample=img
            ).prev_sample
        pdm_score = None
        pdm_score_components = None
        pdm_score_cached = None
        pdm_score_components_cached = None
        if self._pdm_score_head is not None:
            pdm_score = self._pdm_score_head(
                poses_reg.detach(),
                bev_feature,
                image_tokens=score_image_tokens,
                lidar_tokens=score_lidar_tokens,
                agent_tokens=score_agent_tokens,
            )
            if pdm_score is not None and pdm_score.dim() == 3:
                pdm_score_components = pdm_score
                pdm_score = self._aggregate_pdm_components(pdm_score_components)
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
                pdm_score_cached = self._pdm_score_head(
                    cached_poses.detach(),
                    bev_feature,
                    image_tokens=score_image_tokens,
                    lidar_tokens=score_lidar_tokens,
                    agent_tokens=score_agent_tokens,
                )
                if pdm_score_cached is not None and pdm_score_cached.dim() == 3:
                    pdm_score_components_cached = pdm_score_cached
                    pdm_score_cached = self._aggregate_pdm_components(
                        pdm_score_components_cached
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
            mode_idx = poses_cls.argmax(dim=-1)
        mode_idx = mode_idx[...,None,None,None].repeat(1,1,self._num_poses,3)
        best_reg = torch.gather(poses_reg, 1, mode_idx).squeeze(1)
        trajectory_loss_dict = {}
        with torch.no_grad():
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
                pred_mode = poses_cls.argmax(dim=-1)
                match = (pred_mode == gt_mode).float().mean()
                batch_idx = torch.arange(dist.shape[0], device=dist.device)
                gap = (dist[batch_idx, pred_mode] - dist[batch_idx, gt_mode]).mean()
                trajectory_loss_dict["trajectory_mode_match_rate"] = match
                trajectory_loss_dict["trajectory_mode_gap"] = gap
        return {
            "trajectory": best_reg,
            "plan_anchor": plan_anchor,
            "pdm_score": pdm_score,
            "pdm_score_components": pdm_score_components,
            "pdm_score_cached": pdm_score_cached,
            "pdm_score_components_cached": pdm_score_components_cached,
            "trajectory_loss_dict": trajectory_loss_dict,
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
