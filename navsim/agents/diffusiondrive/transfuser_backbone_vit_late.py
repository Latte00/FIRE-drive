"""
Optional ViT image backbone with late fusion into the LiDAR BEV branch.
"""

import timm
import torch
import torch.nn.functional as F
from torch import nn

from navsim.agents.diffusiondrive.modules.davit import DAViT
from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig


class TransfuserViTLateFusionBackbone(nn.Module):
    def __init__(self, config: TransfuserConfig):
        super().__init__()
        self.config = config

        vit_ckpt = getattr(config, "image_vit_ckpt", "") or None
        vit_encoder = getattr(config, "vit_encoder", "vitl")
        self.image_encoder = DAViT(encoder=vit_encoder, ckpt=vit_ckpt)

        if config.use_ground_plane:
            in_channels = 2 * config.lidar_seq_len
        else:
            in_channels = config.lidar_seq_len

        if config.latent:
            self.lidar_latent = nn.Parameter(
                torch.randn(
                    (
                        1,
                        in_channels,
                        config.lidar_resolution_width,
                        config.lidar_resolution_height,
                    ),
                    requires_grad=True,
                )
            )
        else:
            self.lidar_latent = None

        self.lidar_encoder = timm.create_model(
            config.lidar_architecture,
            pretrained=False,
            in_chans=in_channels,
            features_only=True,
        )

        self.avgpool_img = nn.AdaptiveAvgPool2d(
            (self.config.img_vert_anchors, self.config.img_horz_anchors)
        )
        self.global_pool_img = nn.AdaptiveAvgPool2d(output_size=1)
        self.global_pool_lidar = nn.AdaptiveAvgPool2d(output_size=1)

        start_index = 0
        if len(self.lidar_encoder.return_layers) > 4:
            start_index += 1

        self.num_image_features = int(getattr(config, "vit_image_channels", 1024))
        self.num_lidar_features = self.lidar_encoder.feature_info.info[start_index + 3]["num_chs"]
        self.num_features = self.num_lidar_features
        self.perspective_upsample_factor = 1

        self.image_to_lidar_proj = nn.Conv2d(
            self.num_image_features, self.num_lidar_features, kernel_size=1
        )
        self.vit_late_fuse_weight = float(getattr(config, "vit_late_fuse_weight", 1.0))
        self.vit_late_fuse_use_gate = bool(
            getattr(config, "vit_late_fuse_use_gate", True)
        )
        if self.vit_late_fuse_use_gate:
            self.image_lidar_gate = nn.Sequential(
                nn.Conv2d(self.num_lidar_features * 2, self.num_lidar_features, kernel_size=1),
                nn.Sigmoid(),
            )
        else:
            self.image_lidar_gate = None

        channel = self.config.bev_features_channels
        self.relu = nn.ReLU(inplace=True)
        if self.config.detect_boxes or self.config.use_bev_semantic:
            self.upsample = nn.Upsample(
                scale_factor=self.config.bev_upsample_factor,
                mode="bilinear",
                align_corners=False,
            )
            self.upsample2 = nn.Upsample(
                size=(
                    self.config.lidar_resolution_height // self.config.bev_down_sample_factor,
                    self.config.lidar_resolution_width // self.config.bev_down_sample_factor,
                ),
                mode="bilinear",
                align_corners=False,
            )
            self.up_conv5 = nn.Conv2d(channel, channel, (3, 3), padding=1)
            self.up_conv4 = nn.Conv2d(channel, channel, (3, 3), padding=1)
            self.c5_conv = nn.Conv2d(self.num_lidar_features, channel, (1, 1))

    def top_down(self, x):
        p5 = self.relu(self.c5_conv(x))
        p4 = self.relu(self.up_conv5(self.upsample(p5)))
        p3 = self.relu(self.up_conv4(self.upsample2(p4)))
        return p3

    def forward_layer_block(self, layers, return_layers, features):
        for name, module in layers:
            features = module(features)
            if name in return_layers:
                break
        return features

    def _forward_lidar_encoder(self, lidar):
        lidar_features = lidar
        if self.lidar_latent is not None:
            batch_size = lidar.shape[0]
            lidar_features = self.lidar_latent.repeat(batch_size, 1, 1, 1)

        lidar_layers = iter(self.lidar_encoder.items())
        if len(self.lidar_encoder.return_layers) > 4:
            lidar_features = self.forward_layer_block(
                lidar_layers, self.lidar_encoder.return_layers, lidar_features
            )
        for _ in range(4):
            lidar_features = self.forward_layer_block(
                lidar_layers, self.lidar_encoder.return_layers, lidar_features
            )
        return lidar_features

    def _late_fuse(self, image_features, lidar_features):
        image_late = F.adaptive_avg_pool2d(image_features, lidar_features.shape[-2:])
        image_late = self.image_to_lidar_proj(image_late)
        if self.image_lidar_gate is not None:
            gate = self.image_lidar_gate(torch.cat([lidar_features, image_late], dim=1))
            image_late = image_late * gate
        return lidar_features + self.vit_late_fuse_weight * image_late

    def forward(self, image, lidar):
        image_features = self.image_encoder(image)[-1]
        lidar_features = self._forward_lidar_encoder(lidar)
        fused_lidar = self._late_fuse(image_features, lidar_features)

        if self.config.detect_boxes or self.config.use_bev_semantic:
            features = self.top_down(fused_lidar)
        else:
            features = None

        return features, fused_lidar, image_features, fused_lidar
