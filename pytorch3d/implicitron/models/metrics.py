# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import warnings
from typing import Dict, Optional

import torch
from pytorch3d.implicitron.tools import metric_utils as utils
from pytorch3d.renderer import utils as rend_utils


class ViewMetrics(torch.nn.Module):
    def forward(
        self,
        image_sampling_grid: torch.Tensor,
        images: Optional[torch.Tensor] = None,
        images_pred: Optional[torch.Tensor] = None,
        depths: Optional[torch.Tensor] = None,
        depths_pred: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None,
        masks_pred: Optional[torch.Tensor] = None,
        masks_crop: Optional[torch.Tensor] = None,
        grad_theta: Optional[torch.Tensor] = None,
        density_grid: Optional[torch.Tensor] = None,
        keys_prefix: str = "loss_",
        mask_renders_by_pred: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Calculates various differentiable metrics useful for supervising
        differentiable rendering pipelines.

        Args:
            image_sampling_grid: A tensor of shape `(B, ..., 2)` containing 2D
                    image locations at which the predictions are defined.
                    All ground truth inputs are sampled at these
                    locations in order to extract values that correspond
                    to the predictions.
            images: A tensor of shape `(B, H, W, 3)` containing ground truth
                rgb values.
            images_pred: A tensor of shape `(B, ..., 3)` containing predicted
                rgb values.
            depths: A tensor of shape `(B, Hd, Wd, 1)` containing ground truth
                depth values.
            depths_pred: A tensor of shape `(B, ..., 1)` containing predicted
                depth values.
            masks: A tensor of shape `(B, Hm, Wm, 1)` containing ground truth
                foreground masks.
            masks_pred: A tensor of shape `(B, ..., 1)` containing predicted
                foreground masks.
            grad_theta: A tensor of shape `(B, ..., 3)` containing an evaluation
                of a gradient of a signed distance function w.r.t.
                input 3D coordinates used to compute the eikonal loss.
            density_grid: A tensor of shape `(B, Hg, Wg, Dg, 1)` containing a
                `Hg x Wg x Dg` voxel grid of density values.
            keys_prefix: A common prefix for all keys in the output dictionary
                containing all metrics.
            mask_renders_by_pred: If `True`, masks rendered images by the predicted
                `masks_pred` prior to computing all rgb metrics.

        Returns:
            metrics: A dictionary `{metric_name_i: metric_value_i}` keyed by the
                names of the output metrics `metric_name_i` with their corresponding
                values `metric_value_i` represented as 0-dimensional float tensors.

                The calculated metrics are:
                    rgb_huber: A robust huber loss between `image_pred` and `image`.
                    rgb_mse: Mean squared error between `image_pred` and `image`.
                    rgb_psnr: Peak signal-to-noise ratio between `image_pred` and `image`.
                    rgb_psnr_fg: Peak signal-to-noise ratio between the foreground
                        region of `image_pred` and `image` as defined by `mask`.
                    rgb_mse_fg: Mean squared error between the foreground
                        region of `image_pred` and `image` as defined by `mask`.
                    mask_neg_iou: (1 - intersection-over-union) between `mask_pred`
                        and `mask`.
                    mask_bce: Binary cross entropy between `mask_pred` and `mask`.
                    mask_beta_prior: A loss enforcing strictly binary values
                        of `mask_pred`: `log(mask_pred) + log(1-mask_pred)`
                    depth_abs: Mean per-pixel L1 distance between
                        `depth_pred` and `depth`.
                    depth_abs_fg: Mean per-pixel L1 distance between the foreground
                        region of `depth_pred` and `depth` as defined by `mask`.
                    eikonal: Eikonal regularizer `(||grad_theta|| - 1)**2`.
                    density_tv: The Total Variation regularizer of density
                        values in `density_grid` (sum of L1 distances of values
                        of all 4-neighbouring cells).
                    depth_neg_penalty: `min(depth_pred, 0)**2` penalizing negative
                        predicted depth values.
        """

        # TODO: extract functions

        # reshape from B x ... x DIM to B x DIM x -1 x 1
        images_pred, masks_pred, depths_pred = [
            _reshape_nongrid_var(x) for x in [images_pred, masks_pred, depths_pred]
        ]
        # reshape the sampling grid as well
        # TODO: we can get rid of the singular dimension here and in _reshape_nongrid_var
        # now that we use rend_utils.ndc_grid_sample
        image_sampling_grid = image_sampling_grid.reshape(
            image_sampling_grid.shape[0], -1, 1, 2
        )

        # closure with the given image_sampling_grid
        def sample(tensor, mode):
            if tensor is None:
                return tensor
            return rend_utils.ndc_grid_sample(tensor, image_sampling_grid, mode=mode)

        # eval all results in this size
        images = sample(images, mode="bilinear")
        depths = sample(depths, mode="nearest")
        masks = sample(masks, mode="nearest")
        masks_crop = sample(masks_crop, mode="nearest")
        if masks_crop is None and images_pred is not None:
            masks_crop = torch.ones_like(images_pred[:, :1])
        if masks_crop is None and depths_pred is not None:
            masks_crop = torch.ones_like(depths_pred[:, :1])

        preds = {}
        if images is not None and images_pred is not None:
            # TODO: mask_renders_by_pred is always false; simplify
            preds.update(
                _rgb_metrics(
                    images,
                    images_pred,
                    masks,
                    masks_pred,
                    masks_crop,
                    mask_renders_by_pred,
                )
            )

        if masks_pred is not None:
            preds["mask_beta_prior"] = utils.beta_prior(masks_pred)
        if masks is not None and masks_pred is not None:
            preds["mask_neg_iou"] = utils.neg_iou_loss(
                masks_pred, masks, mask=masks_crop
            )
            preds["mask_bce"] = utils.calc_bce(masks_pred, masks, mask=masks_crop)

        if depths is not None and depths_pred is not None:
            assert masks_crop is not None
            _, abs_ = utils.eval_depth(
                depths_pred, depths, get_best_scale=True, mask=masks_crop, crop=0
            )
            preds["depth_abs"] = abs_.mean()

            if masks is not None:
                mask = masks * masks_crop
                _, abs_ = utils.eval_depth(
                    depths_pred, depths, get_best_scale=True, mask=mask, crop=0
                )
                preds["depth_abs_fg"] = abs_.mean()

        # regularizers
        if grad_theta is not None:
            preds["eikonal"] = _get_eikonal_loss(grad_theta)

        if density_grid is not None:
            preds["density_tv"] = _get_grid_tv_loss(density_grid)

        if depths_pred is not None:
            preds["depth_neg_penalty"] = _get_depth_neg_penalty_loss(depths_pred)

        if keys_prefix is not None:
            preds = {(keys_prefix + k): v for k, v in preds.items()}

        return preds


def _rgb_metrics(
    images, images_pred, masks, masks_pred, masks_crop, mask_renders_by_pred
):
    assert masks_crop is not None
    if mask_renders_by_pred:
        images = images[..., masks_pred.reshape(-1), :]
        masks_crop = masks_crop[..., masks_pred.reshape(-1), :]
        masks = masks is not None and masks[..., masks_pred.reshape(-1), :]
    rgb_squared = ((images_pred - images) ** 2).mean(dim=1, keepdim=True)
    rgb_loss = utils.huber(rgb_squared, scaling=0.03)
    crop_mass = masks_crop.sum().clamp(1.0)
    preds = {
        "rgb_huber": (rgb_loss * masks_crop).sum() / crop_mass,
        "rgb_mse": (rgb_squared * masks_crop).sum() / crop_mass,
        "rgb_psnr": utils.calc_psnr(images_pred, images, mask=masks_crop),
    }
    if masks is not None:
        masks = masks_crop * masks
        preds["rgb_psnr_fg"] = utils.calc_psnr(images_pred, images, mask=masks)
        preds["rgb_mse_fg"] = (rgb_squared * masks).sum() / masks.sum().clamp(1.0)
    return preds


def _get_eikonal_loss(grad_theta):
    return ((grad_theta.norm(2, dim=1) - 1) ** 2).mean()


def _get_grid_tv_loss(grid, log_domain: bool = True, eps: float = 1e-5):
    if log_domain:
        if (grid <= -eps).any():
            warnings.warn("Grid has negative values; this will produce NaN loss")
        grid = torch.log(grid + eps)

    # this is an isotropic version, note that it ignores last rows/cols
    return torch.mean(
        utils.safe_sqrt(
            (grid[..., :-1, :-1, 1:] - grid[..., :-1, :-1, :-1]) ** 2
            + (grid[..., :-1, 1:, :-1] - grid[..., :-1, :-1, :-1]) ** 2
            + (grid[..., 1:, :-1, :-1] - grid[..., :-1, :-1, :-1]) ** 2,
            eps=1e-5,
        )
    )


def _get_depth_neg_penalty_loss(depth):
    neg_penalty = depth.clamp(min=None, max=0.0) ** 2
    return torch.mean(neg_penalty)


def _reshape_nongrid_var(x):
    if x is None:
        return None

    ba, *_, dim = x.shape
    return x.reshape(ba, -1, 1, dim).permute(0, 3, 1, 2).contiguous()
