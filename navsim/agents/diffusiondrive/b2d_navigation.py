"""Opt-in near/far status conditioning for Bench2Drive."""
import torch
from torch import nn


def initialize_navigation(model, config):
    model._b2d_navigation_encoder = None
    model._b2d_navigation_gate = None
    model._b2d_navigation_fuse_norm = None
    if not getattr(config, "b2d_navigation_enable", False):
        return
    dim = config.tf_d_model
    model._b2d_navigation_encoder = nn.Sequential(
        nn.Linear(6, dim), nn.ReLU(inplace=True), nn.Linear(dim, dim)
    )
    model._b2d_navigation_gate = nn.Linear(6, dim)
    nn.init.constant_(model._b2d_navigation_gate.bias, config.b2d_navigation_gate_bias)
    model._b2d_navigation_fuse_norm = nn.LayerNorm(dim)


def encode_navigation(model, features, reference):
    if model._b2d_navigation_encoder is None:
        return None
    config = model._config
    points, validity = [], []
    for name, scale, tolerance in (
        ("near", config.b2d_navigation_near_scale_m, config.b2d_navigation_near_behind_tolerance_m),
        ("far", config.b2d_navigation_far_scale_m, config.b2d_navigation_far_behind_tolerance_m),
    ):
        key = "tp_" + name
        point = features.get(key)
        if point is None:
            if config.b2d_navigation_require_features:
                raise KeyError(f"Navigation requires cached feature {key}; patch the B2D cache first")
            return torch.zeros_like(reference)
        point = point.to(device=reference.device, dtype=reference.dtype).reshape(-1, 2)
        if point.shape[0] != reference.shape[0]:
            raise ValueError(f"Navigation batch mismatch: {key}")
        valid = features.get(key + "_valid")
        if valid is None:
            valid = torch.isfinite(point).all(dim=-1)
        valid = valid.to(reference.device).reshape(-1, 1).bool()
        if valid.shape[0] != reference.shape[0]:
            raise ValueError(f"Navigation validity batch mismatch: {key}")
        if config.b2d_navigation_ignore_behind_enable:
            valid = valid & torch.isfinite(point).all(dim=-1, keepdim=True)
            valid = valid & (point[:, :1] >= -max(float(tolerance), 0.0))
        point = torch.nan_to_num(point).masked_fill(~valid, 0.0)
        clip = config.b2d_navigation_clip
        points.append((point / max(float(scale), 1e-6)).clamp(-clip, clip))
        validity.append(valid)
    inputs = torch.cat(points + [v.to(reference.dtype) for v in validity], dim=-1)
    context = model._b2d_navigation_fuse_norm(model._b2d_navigation_encoder(inputs))
    gate = torch.sigmoid(model._b2d_navigation_gate(inputs))
    return context * gate * (validity[0] | validity[1]).to(reference.dtype)
