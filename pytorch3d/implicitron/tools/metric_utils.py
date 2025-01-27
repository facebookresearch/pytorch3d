# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import logging
import math
from typing import Optional, Tuple

import torch
from torch.nn import functional as F

logger = logging.getLogger(__name__)


def eval_depth(
    pred: torch.Tensor,
    gt: torch.Tensor,
    crop: int = 1,
    mask: Optional[torch.Tensor] = None,
    get_best_scale: bool = True,
    mask_thr: float = 0.5,
    best_scale_clamp_thr: float = 1e-4,
    use_disparity: bool = False,
    disparity_eps: float = 1e-4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Evaluate the depth error between the prediction `pred` and the ground
    truth `gt`.

    Args:
        pred: A tensor of shape (N, 1, H, W) denoting the predicted depth maps.
        gt: A tensor of shape (N, 1, H, W) denoting the ground truth depth maps.
        crop: The number of pixels to crop from the border.
        mask: A mask denoting the valid regions of the gt depth.
        get_best_scale: If `True`, estimates a scaling factor of the predicted depth
            that yields the best mean squared error between `pred` and `gt`.
            This is typically enabled for cases where predicted reconstructions
            are inherently defined up to an arbitrary scaling factor.
        mask_thr: A constant used to threshold the `mask` to specify the valid
            regions.
        best_scale_clamp_thr: The threshold for clamping the divisor in best
            scale estimation.

    Returns:
        mse_depth: Mean squared error between `pred` and `gt`.
        abs_depth: Mean absolute difference between `pred` and `gt`.
    """

    # chuck out the border
    if crop > 0:
        gt = gt[:, :, crop:-crop, crop:-crop]
        pred = pred[:, :, crop:-crop, crop:-crop]

    if mask is not None:
        # mult gt by mask
        if crop > 0:
            mask = mask[:, :, crop:-crop, crop:-crop]
        gt = gt * (mask > mask_thr).float()

    dmask = (gt > 0.0).float()
    dmask_mass = torch.clamp(dmask.sum((1, 2, 3)), 1e-4)

    if get_best_scale:
        # mult preds by a scalar "scale_best"
        # 	s.t. we get best possible mse error
        scale_best = estimate_depth_scale_factor(pred, gt, dmask, best_scale_clamp_thr)
        pred = pred * scale_best[:, None, None, None]
    if use_disparity:
        gt = torch.div(1.0, (gt + disparity_eps))
        pred = torch.div(1.0, (pred + disparity_eps))
        scale_best = estimate_depth_scale_factor(
            pred, gt, dmask, best_scale_clamp_thr
        ).detach()
        pred = pred * scale_best[:, None, None, None]

    df = gt - pred

    # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and `int`.
    mse_depth = (dmask * (df**2)).sum((1, 2, 3)) / dmask_mass
    abs_depth = (dmask * df.abs()).sum((1, 2, 3)) / dmask_mass

    return mse_depth, abs_depth


def estimate_depth_scale_factor(pred, gt, mask, clamp_thr):
    xy = pred * gt * mask
    xx = pred * pred * mask
    scale_best = xy.mean((1, 2, 3)) / torch.clamp(xx.mean((1, 2, 3)), clamp_thr)
    return scale_best


def calc_psnr(
    x: torch.Tensor,
    y: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Calculates the Peak-signal-to-noise ratio between tensors `x` and `y`.
    """
    mse = calc_mse(x, y, mask=mask)
    psnr = torch.log10(mse.clamp(1e-10)) * (-10.0)
    return psnr


def calc_mse(
    x: torch.Tensor,
    y: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Calculates the mean square error between tensors `x` and `y`.
    """
    if mask is None:
        # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and `int`.
        return torch.mean((x - y) ** 2)
    else:
        # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and `int`.
        return (((x - y) ** 2) * mask).sum() / mask.expand_as(x).sum().clamp(1e-5)


def calc_bce(
    pred: torch.Tensor,
    gt: torch.Tensor,
    equal_w: bool = True,
    pred_eps: float = 0.01,
    mask: Optional[torch.Tensor] = None,
    lerp_bound: Optional[float] = None,
    pred_logits: bool = False,
) -> torch.Tensor:
    """
    Calculates the binary cross entropy.
    """
    if pred_eps > 0.0:
        # up/low bound the predictions
        pred = torch.clamp(pred, pred_eps, 1.0 - pred_eps)

    if mask is None:
        mask = torch.ones_like(gt)

    if equal_w:
        mask_fg = (gt > 0.5).float() * mask
        mask_bg = (1 - mask_fg) * mask
        weight = mask_fg / mask_fg.sum().clamp(1.0) + mask_bg / mask_bg.sum().clamp(1.0)
        # weight sum should be at this point ~2
        # pyre-fixme[58]: `/` is not supported for operand types `int` and `Tensor`.
        weight = weight * (weight.numel() / weight.sum().clamp(1.0))
    else:
        weight = torch.ones_like(gt) * mask

    if lerp_bound is not None:
        # binary_cross_entropy_lerp requires pred to be in [0, 1]
        if pred_logits:
            pred = F.sigmoid(pred)

        return binary_cross_entropy_lerp(pred, gt, weight, lerp_bound)
    else:
        if pred_logits:
            loss = F.binary_cross_entropy_with_logits(
                pred,
                gt,
                reduction="none",
                weight=weight,
            )
        else:
            loss = F.binary_cross_entropy(pred, gt, reduction="none", weight=weight)

        return loss.mean()


def binary_cross_entropy_lerp(
    pred: torch.Tensor,
    gt: torch.Tensor,
    weight: torch.Tensor,
    lerp_bound: float,
):
    """
    Binary cross entropy which avoids exploding gradients by linearly
    extrapolating the log function for log(1-pred) mad log(pred) whenever
    pred or 1-pred is smaller than lerp_bound.
    """
    loss = log_lerp(1 - pred, lerp_bound) * (1 - gt) + log_lerp(pred, lerp_bound) * gt
    loss_reduced = -(loss * weight).sum() / weight.sum().clamp(1e-4)
    return loss_reduced


def log_lerp(x: torch.Tensor, b: float):
    """
    Linearly extrapolated log for x < b.
    """
    assert b > 0
    return torch.where(x >= b, x.log(), math.log(b) + (x - b) / b)


def rgb_l1(
    pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Calculates the mean absolute error between the predicted colors `pred`
    and ground truth colors `target`.
    """
    if mask is None:
        mask = torch.ones_like(pred[:, :1])
    return ((pred - target).abs() * mask).sum(dim=(1, 2, 3)) / mask.sum(
        dim=(1, 2, 3)
    ).clamp(1)


def huber(dfsq: torch.Tensor, scaling: float = 0.03) -> torch.Tensor:
    """
    Calculates the huber function of the input squared error `dfsq`.
    The function smoothly transitions from a region with unit gradient
    to a hyperbolic function at `dfsq=scaling`.
    """
    loss = (safe_sqrt(1 + dfsq / (scaling * scaling), eps=1e-4) - 1) * scaling
    return loss


def neg_iou_loss(
    predict: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    This is a great loss because it emphasizes on the active
    regions of the predict and targets
    """
    return 1.0 - iou(predict, target, mask=mask)


def safe_sqrt(A: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """
    performs safe differentiable sqrt
    """
    return (torch.clamp(A, float(0)) + eps).sqrt()


def iou(
    predict: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    This is a great loss because it emphasizes on the active
    regions of the predict and targets
    """
    dims = tuple(range(predict.dim())[1:])
    if mask is not None:
        predict = predict * mask
        target = target * mask
    intersect = (predict * target).sum(dims)
    union = (predict + target - predict * target).sum(dims) + 1e-4
    return (intersect / union).sum() / intersect.numel()


def beta_prior(pred: torch.Tensor, cap: float = 0.1) -> torch.Tensor:
    if cap <= 0.0:
        raise ValueError("capping should be positive to avoid unbound loss")

    min_value = math.log(cap) + math.log(cap + 1.0)
    return (torch.log(pred + cap) + torch.log(1.0 - pred + cap)).mean() - min_value
