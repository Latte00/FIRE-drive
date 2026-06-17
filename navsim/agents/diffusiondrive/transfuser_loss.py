from typing import Dict, Tuple
from scipy.optimize import linear_sum_assignment

import torch
import torch.nn.functional as F

from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig
from navsim.agents.diffusiondrive.transfuser_features import BoundingBox2DIndex


def _safe_bev_logits(pred_logits: torch.Tensor, config: TransfuserConfig) -> torch.Tensor:
    if not bool(getattr(config, "bev_loss_logit_guard_enable", False)):
        return pred_logits
    return torch.nan_to_num(
        pred_logits.float(), nan=0.0, posinf=30.0, neginf=-30.0
    ).clamp(-30.0, 30.0)


def _binary_dice_loss(prob: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    prob = prob.reshape(prob.shape[0], -1)
    target = target.reshape(target.shape[0], -1)
    intersection = (prob * target).sum(dim=1)
    denom = prob.sum(dim=1) + target.sum(dim=1)
    dice = (2 * intersection + eps) / (denom + eps)
    return (1 - dice).mean()


def _bev_aux_losses(
    pred_logits: torch.Tensor,
    target_labels: torch.Tensor,
    config: TransfuserConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    pred_logits = _safe_bev_logits(pred_logits, config)
    if target_labels.dim() == 4 and target_labels.shape[1] == pred_logits.shape[1]:
        target_labels = target_labels.argmax(dim=1)
    if target_labels.dim() == 3 and target_labels.shape[0] == pred_logits.shape[1]:
        target_labels = target_labels.argmax(dim=0, keepdim=True)
    if target_labels.dim() == 2:
        target_labels = target_labels.unsqueeze(0)
    if target_labels.dim() == 3 and pred_logits.dim() == 4 and target_labels.shape[0] != pred_logits.shape[0]:
        if target_labels.shape[0] == 1:
            target_labels = target_labels.repeat(pred_logits.shape[0], 1, 1)
    target_labels = target_labels.to(pred_logits.device).long()

    road_label = config.bev_road_label
    centerline_label = config.bev_centerline_label

    num_classes = pred_logits.shape[1]
    center_logits = pred_logits[:, centerline_label]
    if num_classes > 1:
        rest_center = torch.logsumexp(
            torch.cat(
                [pred_logits[:, :centerline_label], pred_logits[:, centerline_label + 1:]],
                dim=1,
            ),
            dim=1,
        )
    else:
        rest_center = torch.zeros_like(center_logits)
    center_logit = center_logits - rest_center
    center_prob = torch.sigmoid(center_logit)

    keep_mask = torch.ones(num_classes, device=pred_logits.device, dtype=torch.bool)
    keep_mask[road_label] = False
    keep_mask[centerline_label] = False
    drive_logits = pred_logits[:, [road_label, centerline_label]]
    drive_logsum = torch.logsumexp(drive_logits, dim=1)
    if keep_mask.any():
        rest_drive = torch.logsumexp(pred_logits[:, keep_mask], dim=1)
    else:
        rest_drive = torch.zeros_like(drive_logsum)
    drivable_logit = drive_logsum - rest_drive
    drivable_prob = torch.sigmoid(drivable_logit)

    center_target = (target_labels == centerline_label).float()
    drivable_target = (
        (target_labels == road_label) | (target_labels == centerline_label)
    ).float()

    center_bce = F.binary_cross_entropy_with_logits(center_logit, center_target)
    drivable_bce = F.binary_cross_entropy_with_logits(drivable_logit, drivable_target)
    center_dice = _binary_dice_loss(center_prob, center_target)
    drivable_dice = _binary_dice_loss(drivable_prob, drivable_target)

    center_loss = config.bev_aux_bce_weight * center_bce + config.bev_aux_dice_weight * center_dice
    drivable_loss = config.bev_aux_bce_weight * drivable_bce + config.bev_aux_dice_weight * drivable_dice
    return center_loss, drivable_loss


def transfuser_loss(
    targets: Dict[str, torch.Tensor], predictions: Dict[str, torch.Tensor], config: TransfuserConfig
):
    """
    Helper function calculating complete loss of Transfuser
    :param targets: dictionary of name tensor pairings
    :param predictions: dictionary of name tensor pairings
    :param config: global Transfuser config
    :return: combined loss value
    """
    # import ipdb; ipdb.set_trace()
    if "trajectory_loss" in predictions:
        trajectory_loss = predictions["trajectory_loss"]
    else:
        trajectory_loss = F.l1_loss(predictions["trajectory"], targets["trajectory"])
    selected_loss = None
    selected_weight = float(getattr(config, "trajectory_selected_weight", 0.0))
    if "trajectory" in predictions:
        selected_loss = F.l1_loss(
            predictions["trajectory"], targets["trajectory"]
        )
    agent_class_loss, agent_box_loss = _agent_loss(targets, predictions, config)
    bev_logits = _safe_bev_logits(predictions["bev_semantic_map"], config)
    bev_semantic_loss = F.cross_entropy(bev_logits, targets["bev_semantic_map"].long())
    centerline_aux_loss, drivable_aux_loss = _bev_aux_losses(
        bev_logits, targets["bev_semantic_map"], config
    )
    if 'diffusion_loss' in predictions:
        diffusion_loss = predictions['diffusion_loss']
    else:
        diffusion_loss = 0
    diff_input_loss = predictions.get("diffusion_input_decorrelation_loss", 0)
    diff_output_loss = predictions.get("diffusion_output_decorrelation_loss", 0)
    cross_bev_loss = predictions.get("cross_bev_decorrelation_loss", 0)
    kinematic_next_loss = None
    kinematic_next_error = None
    kinematic_next_base_error = None
    if (
        "kinematic_next_pred" in predictions
        and targets is not None
        and "trajectory" in targets
    ):
        gt_next = targets["trajectory"][:, 0, :2]
        kinematic_next_pred = predictions["kinematic_next_pred"]
        kinematic_next_loss = F.smooth_l1_loss(kinematic_next_pred, gt_next)
        kinematic_next_error = F.l1_loss(kinematic_next_pred, gt_next)
        kinematic_next_base = predictions.get("kinematic_next_base")
        if kinematic_next_base is not None:
            kinematic_next_base_error = F.l1_loss(kinematic_next_base, gt_next)
    loss = (
        config.trajectory_weight * trajectory_loss
        + (selected_weight * selected_loss if selected_loss is not None else 0.0)
        + config.diff_loss_weight * diffusion_loss
        + config.diff_input_decorrelation_weight * diff_input_loss        
        + config.diff_output_decorrelation_weight * diff_output_loss      
        + config.cross_bev_decorrelation_weight * cross_bev_loss
        + (
            config.kinematic_residual_weight * kinematic_next_loss
            if kinematic_next_loss is not None
            else 0.0
        )
        + config.agent_class_weight * agent_class_loss
        + config.agent_box_weight * agent_box_loss
        + config.bev_semantic_weight * bev_semantic_loss
        + config.bev_centerline_aux_weight * centerline_aux_loss
        + config.bev_drivable_aux_weight * drivable_aux_loss
    )
    loss_dict = {
        'loss': loss,
        'trajectory_loss': config.trajectory_weight*trajectory_loss,
        'trajectory_selected_loss': (
            selected_loss if selected_loss is not None else 0.0
        ),
        'diffusion_loss': config.diff_loss_weight*diffusion_loss,
        'diffusion_input_decorrelation_loss': config.diff_input_decorrelation_weight*diff_input_loss,
        'diffusion_output_decorrelation_loss': config.diff_output_decorrelation_weight*diff_output_loss,
        'cross_bev_decorrelation_loss': config.cross_bev_decorrelation_weight*cross_bev_loss,
        'kinematic_next_loss': (
            config.kinematic_residual_weight * kinematic_next_loss
            if kinematic_next_loss is not None
            else 0.0
        ),
        'kinematic_next_error': kinematic_next_error,
        'kinematic_next_base_error': kinematic_next_base_error,
        'agent_class_loss': config.agent_class_weight*agent_class_loss,
        'agent_box_loss': config.agent_box_weight*agent_box_loss,
        'bev_semantic_loss': config.bev_semantic_weight*bev_semantic_loss,
        'bev_centerline_aux_loss': config.bev_centerline_aux_weight*centerline_aux_loss,
        'bev_drivable_aux_loss': config.bev_drivable_aux_weight*drivable_aux_loss,
    }
    if "trajectory_loss_dict" in predictions:
        trajectory_loss_dict = predictions["trajectory_loss_dict"]
        loss_dict.update(trajectory_loss_dict)
    # import ipdb; ipdb.set_trace()
    return loss_dict


def _agent_loss(
    targets: Dict[str, torch.Tensor], predictions: Dict[str, torch.Tensor], config: TransfuserConfig
):
    """
    Hungarian matching loss for agent detection
    :param targets: dictionary of name tensor pairings
    :param predictions: dictionary of name tensor pairings
    :param config: global Transfuser config
    :return: detection loss
    """

    gt_states, gt_valid = targets["agent_states"], targets["agent_labels"]
    pred_states, pred_logits = predictions["agent_states"], predictions["agent_labels"]
    gt_states = torch.nan_to_num(gt_states, nan=0.0, posinf=1e3, neginf=-1e3)
    pred_states = torch.nan_to_num(pred_states, nan=0.0, posinf=1e3, neginf=-1e3)
    pred_logits = torch.nan_to_num(pred_logits, nan=0.0, posinf=20.0, neginf=-20.0)

    if config.latent:
        rad_to_ego = torch.arctan2(
            gt_states[..., BoundingBox2DIndex.Y],
            gt_states[..., BoundingBox2DIndex.X],
        )

        in_latent_rad_thresh = torch.logical_and(
            -config.latent_rad_thresh <= rad_to_ego,
            rad_to_ego <= config.latent_rad_thresh,
        )
        gt_valid = torch.logical_and(in_latent_rad_thresh, gt_valid)

    # save constants
    batch_dim, num_instances = pred_states.shape[:2]
    num_gt_instances = gt_valid.sum()
    num_gt_instances = num_gt_instances if num_gt_instances > 0 else num_gt_instances + 1

    ce_cost = _get_ce_cost(gt_valid, pred_logits)
    l1_cost = _get_l1_cost(gt_states, pred_states, gt_valid)

    cost = config.agent_class_weight * ce_cost + config.agent_box_weight * l1_cost
    cost = cost.cpu()
    if not torch.isfinite(cost).all():
        cost = torch.nan_to_num(cost, nan=1e6, posinf=1e6, neginf=1e6)

    indices = [linear_sum_assignment(c) for i, c in enumerate(cost)]
    matching = [
        (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
        for i, j in indices
    ]
    idx = _get_src_permutation_idx(matching)

    pred_states_idx = pred_states[idx]
    gt_states_idx = torch.cat([t[i] for t, (_, i) in zip(gt_states, indices)], dim=0)

    pred_valid_idx = pred_logits[idx]
    gt_valid_idx = torch.cat([t[i] for t, (_, i) in zip(gt_valid, indices)], dim=0).float()

    l1_loss = F.l1_loss(pred_states_idx, gt_states_idx, reduction="none")
    l1_loss = l1_loss.sum(-1) * gt_valid_idx
    l1_loss = l1_loss.view(batch_dim, -1).sum() / num_gt_instances

    ce_loss = F.binary_cross_entropy_with_logits(pred_valid_idx, gt_valid_idx, reduction="none")
    ce_loss = ce_loss.view(batch_dim, -1).mean()

    return ce_loss, l1_loss


@torch.no_grad()
def _get_ce_cost(gt_valid: torch.Tensor, pred_logits: torch.Tensor) -> torch.Tensor:
    """
    Function to calculate cross-entropy cost for cost matrix.
    :param gt_valid: tensor of binary ground-truth labels
    :param pred_logits: tensor of predicted logits of neural net
    :return: bce cost matrix as tensor
    """

    # NOTE: numerically stable BCE with logits
    # https://github.com/pytorch/pytorch/blob/c64e006fc399d528bb812ae589789d0365f3daf4/aten/src/ATen/native/Loss.cpp#L214
    gt_valid_expanded = gt_valid[:, :, None].detach().float()  # (b, n, 1)
    pred_logits_expanded = pred_logits[:, None, :].detach()  # (b, 1, n)

    max_val = torch.relu(-pred_logits_expanded)
    helper_term = max_val + torch.log(
        torch.exp(-max_val) + torch.exp(-pred_logits_expanded - max_val)
    )
    ce_cost = (1 - gt_valid_expanded) * pred_logits_expanded + helper_term  # (b, n, n)
    ce_cost = ce_cost.permute(0, 2, 1)

    return ce_cost


@torch.no_grad()
def _get_l1_cost(
    gt_states: torch.Tensor, pred_states: torch.Tensor, gt_valid: torch.Tensor
) -> torch.Tensor:
    """
    Function to calculate L1 cost for cost matrix.
    :param gt_states: tensor of ground-truth bounding boxes
    :param pred_states: tensor of predicted bounding boxes
    :param gt_valid: mask of binary ground-truth labels
    :return: l1 cost matrix as tensor
    """

    gt_states_expanded = gt_states[:, :, None, :2].detach()  # (b, n, 1, 2)
    pred_states_expanded = pred_states[:, None, :, :2].detach()  # (b, 1, n, 2)
    l1_cost = gt_valid[..., None].float() * (gt_states_expanded - pred_states_expanded).abs().sum(
        dim=-1
    )
    l1_cost = l1_cost.permute(0, 2, 1)
    return l1_cost


def _get_src_permutation_idx(indices):
    """
    Helper function to align indices after matching
    :param indices: matched indices
    :return: permuted indices
    """
    # permute predictions following indices
    batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
    src_idx = torch.cat([src for (src, _) in indices])
    return batch_idx, src_idx
