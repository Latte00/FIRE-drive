import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from typing import Callable, Optional, Tuple
import numpy as np
from scipy import ndimage
from torch import Tensor
from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig
# from mmcv.ops import sigmoid_focal_loss as _sigmoid_focal_loss
# from mmdet.models.losses import FocalLoss

def reduce_loss(loss: Tensor, reduction: str) -> Tensor:
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".

    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()

def weight_reduce_loss(loss: Tensor,
                       weight: Optional[Tensor] = None,
                       reduction: str = 'mean',
                       avg_factor: Optional[float] = None) -> Tensor:
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): Element-wise loss.
        weight (Optional[Tensor], optional): Element-wise weights.
            Defaults to None.
        reduction (str, optional): Same as built-in losses of PyTorch.
            Defaults to 'mean'.
        avg_factor (Optional[float], optional): Average factor when
            computing the mean of losses. Defaults to None.

    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            # Avoid causing ZeroDivisionError when avg_factor is 0.0,
            # i.e., all labels of an image belong to ignore index.
            eps = torch.finfo(torch.float32).eps
            loss = loss.sum() / (avg_factor + eps)
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss

def py_sigmoid_focal_loss(pred,
                          target,
                          weight=None,
                          gamma=2.0,
                          alpha=0.25,
                          reduction='mean',
                          avg_factor=None):
    """PyTorch version of `Focal Loss <https://arxiv.org/abs/1708.02002>`_.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the
            number of classes
        target (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 0.25.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    """
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    # Actually, pt here denotes (1 - pt) in the Focal Loss paper
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    # Thus it's pt.pow(gamma) rather than (1 - pt).pow(gamma)
    focal_weight = (alpha * target + (1 - alpha) *
                    (1 - target)) * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(
        pred, target, reduction='none') * focal_weight
    if weight is not None:
        if weight.shape != loss.shape:
            if weight.size(0) == loss.size(0):
                # For most cases, weight is of shape (num_priors, ),
                #  which means it does not have the second axis num_class
                weight = weight.view(-1, 1)
            else:
                # Sometimes, weight per anchor per class is also needed. e.g.
                #  in FSAF. But it may be flattened of shape
                #  (num_priors x num_class, ), while loss is still of shape
                #  (num_priors, num_class).
                assert weight.numel() == loss.numel()
                weight = weight.view(loss.size(0), -1)
        assert weight.ndim == loss.ndim
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


class LossComputer(nn.Module):
    def __init__(self,config: TransfuserConfig):
        self._config = config
        super(LossComputer, self).__init__()
        # self.focal_loss = FocalLoss(use_sigmoid=True, gamma=2.0, alpha=0.25, reduction='mean', loss_weight=1.0, activated=False)
        self.cls_loss_weight = config.trajectory_cls_weight
        self.reg_loss_weight = config.trajectory_reg_weight

    def _flood_fill_mask(self, mask: Tensor, seed: Tensor) -> Tensor:
        max_iters = int(self._config.feasible_lane_max_iters)
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

    def _build_seed_mask(self, drivable_mask: Tensor) -> Tensor:
        batch_size, height, width = drivable_mask.shape
        ego_row = 0
        ego_col = (width - 1) // 2
        seed_mask = torch.zeros_like(drivable_mask, dtype=torch.bool)
        row_end = min(height, max(1, int(self._config.feasible_lane_seed_rows)))
        seed_cols = max(
            0, min(int(self._config.feasible_lane_seed_cols), width // 2)
        )
        col_start = max(0, ego_col - seed_cols)
        col_end = min(width, ego_col + seed_cols + 1)
        ego_index = torch.tensor(
            [ego_row, ego_col], device=drivable_mask.device, dtype=torch.long
        )
        for batch_idx in range(batch_size):
            region = drivable_mask[batch_idx, :row_end, col_start:col_end]
            coords = region.nonzero(as_tuple=False)
            if coords.numel() > 0:
                abs_coords = coords + torch.tensor(
                    [0, col_start], device=coords.device, dtype=coords.dtype
                )
            else:
                abs_coords = drivable_mask[batch_idx].nonzero(as_tuple=False)
            if abs_coords.numel() > 0:
                deltas = abs_coords - ego_index
                dist2 = deltas[:, 0] * deltas[:, 0] + deltas[:, 1] * deltas[:, 1]
                best_idx = dist2.argmin().item()
                seed_mask[
                    batch_idx, abs_coords[best_idx, 0], abs_coords[best_idx, 1]
                ] = True
            else:
                seed_mask[
                    batch_idx, min(ego_row, height - 1), min(ego_col, width - 1)
                ] = True
        return seed_mask

    def _compute_branch_progress_targets(
        self, future_bev: Tensor, end_xy: Tensor
    ) -> Tuple[Tensor, Tensor]:
        if future_bev.dim() == 4:
            class_map = future_bev.argmax(dim=1)
        elif future_bev.dim() == 3:
            class_map = future_bev
        elif future_bev.dim() == 2:
            class_map = future_bev.unsqueeze(0)
        else:
            return (
                torch.zeros((0, 0), device=future_bev.device),
                torch.zeros((0, 0), dtype=torch.bool, device=future_bev.device),
            )
        class_map = class_map.long()
        road_label = int(self._config.bev_road_label)
        centerline_label = int(self._config.bev_centerline_label)
        drivable_mask = (class_map == road_label) | (class_map == centerline_label)
        centerline_mask = class_map == centerline_label
        seed_mask = self._build_seed_mask(drivable_mask)
        feasible_area = self._flood_fill_mask(drivable_mask, seed_mask)
        feasible_lane = feasible_area & centerline_mask

        batch_size, height, width = class_map.shape
        num_modes = end_xy.shape[1]
        target_progress = np.zeros((batch_size, num_modes), dtype=np.float32)
        valid = np.zeros((batch_size, num_modes), dtype=bool)
        pixel_size = float(self._config.bev_pixel_size)
        max_branches = max(1, int(self._config.trajectory_progress_max_branches))
        structure = np.ones((3, 3), dtype=np.int32)

        lane_np = feasible_lane.detach().cpu().numpy()
        area_np = feasible_area.detach().cpu().numpy()
        end_xy_np = end_xy.detach().cpu().numpy()

        for batch_idx in range(batch_size):
            lane_mask = lane_np[batch_idx].astype(np.uint8)
            labels, num = ndimage.label(lane_mask, structure=structure)
            if num == 0:
                area_mask = area_np[batch_idx].astype(np.uint8)
                if area_mask.any():
                    labels = area_mask
                    num = 1
                else:
                    continue

            branches = []
            for comp_id in range(1, num + 1):
                coords = np.column_stack(np.nonzero(labels == comp_id))
                if coords.size == 0:
                    continue
                max_row = int(coords[:, 0].max())
                branches.append((coords, max_row))
            if not branches:
                continue
            branches.sort(key=lambda item: item[1], reverse=True)
            if len(branches) > max_branches:
                branches = branches[:max_branches]

            for mode_idx in range(num_modes):
                end_x = float(end_xy_np[batch_idx, mode_idx, 0])
                end_y = float(end_xy_np[batch_idx, mode_idx, 1])
                end_row = end_x / pixel_size
                end_col = end_y / pixel_size + (width / 2.0)
                best_idx = None
                best_dist = None
                for branch_idx, (coords, max_row) in enumerate(branches):
                    dr = coords[:, 0] - end_row
                    dc = coords[:, 1] - end_col
                    dist2 = (dr * dr + dc * dc).min()
                    if best_dist is None or dist2 < best_dist:
                        best_dist = dist2
                        best_idx = branch_idx
                if best_idx is None:
                    continue
                target_progress[batch_idx, mode_idx] = (
                    branches[best_idx][1] * pixel_size
                )
                valid[batch_idx, mode_idx] = True

        target_progress_t = torch.tensor(
            target_progress, device=end_xy.device, dtype=end_xy.dtype
        )
        valid_t = torch.tensor(valid, device=end_xy.device, dtype=torch.bool)   
        return target_progress_t, valid_t

    def _compute_future_bev_mode_scores(
        self,
        future_bev: Tensor,
        pred_xy: Tensor,
        current_bev: Optional[Tensor] = None,
    ) -> Optional[Tensor]:
        def _to_class_map(bev: Optional[Tensor]) -> Optional[Tensor]:
            if bev is None:
                return None
            if bev.dim() == 4:
                return bev.argmax(dim=1)
            if bev.dim() == 3:
                return bev
            if bev.dim() == 2:
                return bev.unsqueeze(0)
            return None

        class_map = _to_class_map(future_bev)
        if class_map is None:
            return None
        class_map = class_map.long()
        current_map = _to_class_map(current_bev)
        if current_map is not None and current_map.shape != class_map.shape:
            current_map = None

        road_label = int(self._config.bev_road_label)
        centerline_label = int(self._config.bev_centerline_label)
        drivable_future = (class_map == road_label) | (
            class_map == centerline_label
        )
        drivable_current = None
        if current_map is not None:
            current_map = current_map.long()
            drivable_current = (current_map == road_label) | (
                current_map == centerline_label
            )

        batch_size, height, width = class_map.shape
        pixel_size = float(self._config.bev_pixel_size)
        row = torch.round(pred_xy[..., 0] / pixel_size).long()
        col = torch.round(pred_xy[..., 1] / pixel_size + (width / 2.0)).long()
        valid = (row >= 0) & (row < height) & (col >= 0) & (col < width)
        row = row.clamp(0, height - 1)
        col = col.clamp(0, width - 1)

        batch_idx = torch.arange(
            batch_size, device=pred_xy.device, dtype=torch.long
        ).view(batch_size, 1, 1)
        mask_vals = drivable_future[batch_idx, row, col]
        if drivable_current is not None:
            current_steps = int(
                getattr(self._config, "trajectory_cls_current_bev_steps", 0)
            )
            if current_steps > 0:
                max_steps = pred_xy.shape[2]
                current_steps = min(current_steps, max_steps)
                step_ids = torch.arange(
                    max_steps, device=pred_xy.device, dtype=torch.long
                ).view(1, 1, max_steps)
                use_current = step_ids < current_steps
                mask_current = drivable_current[batch_idx, row, col]
                mask_vals = torch.where(use_current, mask_current, mask_vals)

        mask_vals = mask_vals & valid
        feasible_ratio = mask_vals.float().mean(dim=-1)

        masked_progress = pred_xy[..., 0].masked_fill(~mask_vals, float("-inf"))
        progress = masked_progress.max(dim=-1).values
        progress = torch.where(
            torch.isfinite(progress), progress, torch.zeros_like(progress)
        )
        max_forward = max((height - 1) * pixel_size, 1e-6)
        progress_norm = torch.clamp(progress, 0.0, max_forward) / max_forward

        w_progress = float(
            getattr(self._config, "trajectory_cls_future_progress_weight", 1.0)
        )
        w_feasible = float(
            getattr(self._config, "trajectory_cls_future_feasible_weight", 1.0)
        )
        return w_progress * progress_norm + w_feasible * feasible_ratio

    def _count_sign_flips(self, values: Tensor, eps: float) -> Tensor:
        sign = torch.sign(values)
        sign = torch.where(sign.abs() < eps, torch.zeros_like(sign), sign)
        for i in range(1, sign.shape[-1]):
            prev = sign[..., i - 1]
            sign[..., i] = torch.where(sign[..., i] == 0, prev, sign[..., i])
        for i in range(sign.shape[-1] - 2, -1, -1):
            nxt = sign[..., i + 1]
            sign[..., i] = torch.where(sign[..., i] == 0, nxt, sign[..., i])
        return (sign[..., 1:] * sign[..., :-1] < 0).sum(dim=-1)

    def _candidate_geometry_mask(self, candidates: Tensor) -> Optional[Tensor]:
        max_dyaw = float(getattr(self._config, "trajectory_candidate_max_dyaw", 0.0))
        max_yaw_flips = int(
            getattr(self._config, "trajectory_candidate_max_yaw_flips", 0)
        )
        max_lat_flips = int(
            getattr(self._config, "trajectory_candidate_max_lat_flips", 0)
        )
        lat_eps = float(getattr(self._config, "trajectory_candidate_lat_eps", 1e-3))
        if max_dyaw <= 0.0 and max_yaw_flips <= 0 and max_lat_flips <= 0:
            return None
        cand = candidates.detach()
        if cand.shape[-1] >= 3:
            heading = cand[..., 2]
        else:
            delta = cand[..., 1:, :2] - cand[..., :-1, :2]
            heading = torch.atan2(delta[..., 1], delta[..., 0])
            heading = torch.cat([heading[..., :1], heading], dim=-1)
        dyaw = heading[..., 1:] - heading[..., :-1]
        dyaw = torch.atan2(torch.sin(dyaw), torch.cos(dyaw))
        mask = torch.ones(cand.shape[:2], device=cand.device, dtype=torch.bool)
        if max_dyaw > 0.0:
            mask = mask & (dyaw.abs().max(dim=-1).values <= max_dyaw)
        if max_yaw_flips > 0:
            yaw_flips = self._count_sign_flips(dyaw, eps=1e-3)
            mask = mask & (yaw_flips <= max_yaw_flips)
        if max_lat_flips > 0:
            dy = cand[..., 1:, 1] - cand[..., :-1, 1]
            lat_flips = self._count_sign_flips(dy, eps=lat_eps)
            mask = mask & (lat_flips <= max_lat_flips)
        return mask

    def forward(self, poses_reg, poses_cls, targets, plan_anchor):
        """
        pred_traj: (bs, num_modes, num_poses, 3)
        pred_cls: (bs, num_modes)
        plan_anchor: (bs, num_modes, num_poses, 2)
        targets['trajectory']: (bs, num_poses, 3)
        """
        bs, num_mode, ts, d = poses_reg.shape
        target_traj = targets["trajectory"]
        dist = torch.linalg.norm(
            target_traj.unsqueeze(1)[..., :2] - poses_reg[..., :2], dim=-1
        )
        dist = dist.mean(dim=-1)
        mode_idx = torch.argmin(dist, dim=-1)
        cls_target = mode_idx
        gt_mode_idx = mode_idx
        mode_idx = mode_idx[...,None,None,None].repeat(1,1,ts,d)
        best_reg = torch.gather(poses_reg, 1, mode_idx).squeeze(1)
        # import ipdb; ipdb.set_trace()
        # Calculate cls loss using future BEV mode scores (soft) or fallback to focal loss
        loss_cls = poses_reg.new_zeros(())
        gt_soft_weight = float(getattr(self._config, "trajectory_cls_gt_soft_weight", 0.0))
        gt_probs = None
        if poses_cls is not None and gt_soft_weight > 0.0:
            gt_temp = max(
                float(getattr(self._config, "trajectory_cls_gt_soft_temp", 1.0)),
                1e-6,
            )
            gt_probs = torch.softmax(-dist.detach() / gt_temp, dim=-1)

        bev_probs = None
        if poses_cls is not None and getattr(self._config, "trajectory_cls_use_future_bev", False):
            future_bev = targets.get("future_bev_semantic_map")
            if future_bev is not None:
                future_bev = future_bev.detach().to(poses_reg.device)
                current_bev = targets.get("bev_semantic_map")
                if current_bev is not None:
                    current_bev = current_bev.detach().to(poses_reg.device)
                scores = self._compute_future_bev_mode_scores(
                    future_bev,
                    poses_reg[..., :2].detach(),
                    current_bev=current_bev,
                )
                if scores is not None:
                    temp = max(
                        float(
                            getattr(self._config, "trajectory_cls_future_temp", 1.0)
                        ),
                        1e-6,
                    )
                    bev_probs = torch.softmax(scores / temp, dim=-1)

        target_probs = None
        if gt_probs is not None and bev_probs is not None:
            alpha = min(max(gt_soft_weight, 0.0), 1.0)
            target_probs = alpha * gt_probs + (1.0 - alpha) * bev_probs
        elif gt_probs is not None:
            target_probs = gt_probs
        elif bev_probs is not None:
            target_probs = bev_probs

        if poses_cls is not None and target_probs is not None:
            pred_log_probs = F.log_softmax(poses_cls, dim=-1)
            loss_cls = self.cls_loss_weight * F.kl_div(
                pred_log_probs, target_probs, reduction="batchmean"
            )
        if poses_cls is not None and target_probs is None:
            target_classes_onehot = torch.zeros(
                [bs, num_mode],
                dtype=poses_cls.dtype,
                layout=poses_cls.layout,
                device=poses_cls.device,
            )
            target_classes_onehot.scatter_(1, cls_target.unsqueeze(1), 1)
            # Use py_sigmoid_focal_loss function for focal loss calculation
            loss_cls = self.cls_loss_weight * py_sigmoid_focal_loss(
                poses_cls,
                target_classes_onehot,
                weight=None,
                gamma=2.0,
                alpha=0.25,
                reduction="mean",
                avg_factor=None,
            )

        # Calculate regression loss
        reg_loss = self.reg_loss_weight * F.l1_loss(best_reg, target_traj)
        # import ipdb; ipdb.set_trace()
        # Combine classification and regression losses
        ret_loss = loss_cls + reg_loss

        progress_weight = float(getattr(self._config, "trajectory_progress_weight", 0.0))
        if progress_weight > 0.0:
            future_bev = targets.get("future_bev_semantic_map")
            if future_bev is not None:
                future_bev = future_bev.detach().to(poses_reg.device)
                end_xy = poses_reg[..., -1, :2]
                target_progress, valid = self._compute_branch_progress_targets(
                    future_bev, end_xy
                )
                if valid.any():
                    pred_progress = poses_reg[..., -1, 0]
                    per_mode_loss = F.smooth_l1_loss(
                        pred_progress, target_progress, reduction="none"
                    )
                    per_mode_loss = per_mode_loss * valid.float()
                    if self._config.trajectory_progress_mode_weighted and poses_cls is not None:
                        weights = torch.softmax(poses_cls.detach(), dim=-1)
                        weights = weights * valid.float()
                        denom = weights.sum(dim=1).clamp_min(1e-6)
                        progress_loss = (per_mode_loss * weights).sum(dim=1) / denom
                    else:
                        denom = valid.float().sum(dim=1).clamp_min(1.0)
                        progress_loss = per_mode_loss.sum(dim=1) / denom
                    ret_loss = ret_loss + progress_weight * progress_loss.mean()

        cand_weight = float(getattr(self._config, "trajectory_candidate_weight", 0.0))
        cand_cls_weight = float(
            getattr(self._config, "trajectory_candidate_cls_weight", 0.0)
        )
        if cand_weight > 0.0 or cand_cls_weight > 0.0:
            candidates = targets.get("trajectory_candidates")
            if candidates is not None and candidates.numel() > 0:
                if candidates.dim() == 3:
                    candidates = candidates.unsqueeze(0)
                cand_mask = targets.get("trajectory_candidates_mask")
                cand_scores = targets.get("pdm_score_targets")
                gt_score = targets.get("gt_pdm_score")
                cand_xy = candidates[..., :2]
                pred_xy = poses_reg[..., :2]
                dist = torch.linalg.norm(
                    pred_xy[:, :, None, :, :] - cand_xy[:, None, :, :, :], dim=-1
                ).mean(dim=-1)
                score_mask = None
                if cand_scores is not None and gt_score is not None:
                    cand_scores = cand_scores.detach().to(
                        device=dist.device, dtype=dist.dtype
                    )
                    gt_score = gt_score.detach().to(device=dist.device, dtype=dist.dtype)
                    if cand_scores.ndim == 1:
                        cand_scores = cand_scores.unsqueeze(0)
                    gt_score = gt_score.view(-1)
                    if gt_score.numel() == 1 and cand_scores.shape[0] > 1:
                        gt_score = gt_score.expand(cand_scores.shape[0])
                    if gt_score.numel() == cand_scores.shape[0]:
                        score_mask = torch.isfinite(cand_scores) & (
                            cand_scores > gt_score[:, None]
                        )
                    else:
                        cand_scores = None
                geom_mask = self._candidate_geometry_mask(candidates)
                mask = None
                if cand_mask is not None:
                    if cand_mask.dim() == 1:
                        cand_mask = cand_mask.unsqueeze(0)
                    mask = cand_mask.bool()
                if score_mask is not None:
                    mask = score_mask if mask is None else (mask & score_mask)
                if geom_mask is not None:
                    mask = geom_mask if mask is None else (mask & geom_mask)

                if mask is not None:
                    dist = dist.masked_fill(~mask[:, None, :], float("inf"))
                    valid_samples = mask.any(dim=1)
                else:
                    valid_samples = torch.ones(
                        dist.shape[0], dtype=torch.bool, device=dist.device
                    )
                mode_mask = None
                if getattr(self._config, "trajectory_candidate_exclude_gt_mode", False):
                    mode_mask = torch.ones(
                        (bs, num_mode), device=dist.device, dtype=torch.bool
                    )
                    mode_mask.scatter_(1, gt_mode_idx.unsqueeze(1), False)
                    valid_samples = valid_samples & mode_mask.any(dim=1)
                if valid_samples.any():
                    pair_losses = []
                    cls_losses = []
                    cand_cls_temp = max(
                        float(
                            getattr(self._config, "trajectory_candidate_cls_temp", 1.0)
                        ),
                        1e-6,
                    )
                    for b in torch.nonzero(valid_samples, as_tuple=False).view(-1):
                        dist_b = dist[b]
                        if mode_mask is not None:
                            mode_idx = torch.nonzero(
                                mode_mask[b], as_tuple=False
                            ).view(-1)
                        else:
                            mode_idx = torch.arange(num_mode, device=dist.device)
                        if mask is not None:
                            cand_idx = torch.nonzero(
                                mask[b], as_tuple=False
                            ).view(-1)
                        else:
                            cand_idx = torch.arange(dist_b.shape[1], device=dist.device)
                        if mode_idx.numel() == 0 or cand_idx.numel() == 0:
                            continue
                        dist_sub = dist_b.index_select(0, mode_idx).index_select(1, cand_idx)
                        flat = dist_sub.reshape(-1)
                        order = torch.argsort(flat)
                        used_modes = set()
                        used_cands = set()
                        selected = []
                        num_cands = dist_sub.shape[1]
                        for idx in order:
                            idx_int = int(idx.item())
                            mode_i = idx_int // num_cands
                            cand_i = idx_int % num_cands
                            if mode_i in used_modes or cand_i in used_cands:
                                continue
                            val = dist_sub[mode_i, cand_i]
                            if not torch.isfinite(val):
                                continue
                            used_modes.add(mode_i)
                            used_cands.add(cand_i)
                            selected.append((mode_i, cand_i, val))
                            if len(used_modes) >= mode_idx.numel() or len(used_cands) >= cand_idx.numel():
                                break
                        if selected:
                            pair_losses.append(torch.stack([item[2] for item in selected]).mean())
                            if (
                                cand_cls_weight > 0.0
                                and cand_scores is not None
                                and poses_cls is not None
                            ):
                                score_b = cand_scores[b]
                                mode_ids = []
                                score_vals = []
                                for mode_i, cand_i, _ in selected:
                                    mode_id = mode_idx[mode_i]
                                    cand_id = cand_idx[cand_i]
                                    score_val = score_b[cand_id]
                                    if torch.isfinite(score_val):
                                        mode_ids.append(mode_id)
                                        score_vals.append(score_val)
                                if score_vals:
                                    mode_ids_t = torch.stack(mode_ids)
                                    score_vals_t = torch.stack(score_vals)
                                    cand_logits = torch.full(
                                        (num_mode,),
                                        float("-inf"),
                                        device=dist.device,
                                        dtype=dist.dtype,
                                    )
                                    cand_logits[mode_ids_t] = score_vals_t
                                    cand_probs = torch.softmax(
                                        cand_logits / cand_cls_temp, dim=-1
                                    )
                                    pred_log_probs = F.log_softmax(
                                        poses_cls[b], dim=-1
                                    )
                                    cls_losses.append(
                                        F.kl_div(
                                            pred_log_probs,
                                            cand_probs,
                                            reduction="batchmean",
                                        )
                                    )
                    if pair_losses and cand_weight > 0.0:
                        ret_loss = ret_loss + cand_weight * torch.stack(pair_losses).mean()
                    if cls_losses and cand_cls_weight > 0.0:
                        ret_loss = ret_loss + cand_cls_weight * torch.stack(cls_losses).mean()
        return ret_loss
