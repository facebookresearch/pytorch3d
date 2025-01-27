# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import warnings
from typing import Any, Dict, Optional

import torch
from pytorch3d.implicitron.models.renderer.ray_sampler import ImplicitronRayBundle
from pytorch3d.implicitron.tools import metric_utils as utils
from pytorch3d.implicitron.tools.config import registry, ReplaceableBase
from pytorch3d.ops import padded_to_packed
from pytorch3d.renderer import utils as rend_utils

from .renderer.base import RendererOutput


class RegularizationMetricsBase(ReplaceableBase, torch.nn.Module):
    """
    Replaceable abstract base for regularization metrics.
    `forward()` method produces regularization metrics and (unlike ViewMetrics) can
    depend on the model's parameters.
    """

    def forward(
        self, model: Any, keys_prefix: str = "loss_", **kwargs
    ) -> Dict[str, Any]:
        """
        Calculates various regularization terms useful for supervising differentiable
        rendering pipelines.

        Args:
            model: A model instance. Useful, for example, to implement
                weights-based regularization.
            keys_prefix: A common prefix for all keys in the output dictionary
                containing all regularization metrics.

        Returns:
            A dictionary with the resulting regularization metrics. The items
                will have form `{metric_name_i: metric_value_i}` keyed by the
                names of the output metrics `metric_name_i` with their corresponding
                values `metric_value_i` represented as 0-dimensional float tensors.
        """
        raise NotImplementedError


class ViewMetricsBase(ReplaceableBase, torch.nn.Module):
    """
    Replaceable abstract base for model metrics.
    `forward()` method produces losses and other metrics.
    """

    def forward(
        self,
        raymarched: RendererOutput,
        ray_bundle: ImplicitronRayBundle,
        image_rgb: Optional[torch.Tensor] = None,
        depth_map: Optional[torch.Tensor] = None,
        fg_probability: Optional[torch.Tensor] = None,
        mask_crop: Optional[torch.Tensor] = None,
        keys_prefix: str = "loss_",
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Calculates various metrics and loss functions useful for supervising
        differentiable rendering pipelines. Any additional parameters can be passed
        in the `raymarched.aux` dictionary.

        Args:
            results: A dictionary with the resulting view metrics. The items
                will have form `{metric_name_i: metric_value_i}` keyed by the
                names of the output metrics `metric_name_i` with their corresponding
                values `metric_value_i` represented as 0-dimensional float tensors.
            raymarched: Output of the renderer.
            ray_bundle: ImplicitronRayBundle object which was used to produce the raymarched
                object
            image_rgb: A tensor of shape `(B, H, W, 3)` containing ground truth rgb
                values.
            depth_map: A tensor of shape `(B, Hd, Wd, 1)` containing ground truth depth
                values.
            fg_probability: A tensor of shape `(B, Hm, Wm, 1)` containing ground truth
                foreground masks.
            keys_prefix: A common prefix for all keys in the output dictionary
                containing all view metrics.

        Returns:
            A dictionary with the resulting view metrics. The items
                will have form `{metric_name_i: metric_value_i}` keyed by the
                names of the output metrics `metric_name_i` with their corresponding
                values `metric_value_i` represented as 0-dimensional float tensors.
        """
        raise NotImplementedError()


@registry.register
class RegularizationMetrics(RegularizationMetricsBase):
    def forward(
        self, model: Any, keys_prefix: str = "loss_", **kwargs
    ) -> Dict[str, Any]:
        """
        Calculates the AD penalty, or returns an empty dict if the model's autoencoder
        is inactive.

        Args:
            model: A model instance.
            keys_prefix: A common prefix for all keys in the output dictionary
                containing all regularization metrics.

        Returns:
            A dictionary with the resulting regularization metrics. The items
                will have form `{metric_name_i: metric_value_i}` keyed by the
                names of the output metrics `metric_name_i` with their corresponding
                values `metric_value_i` represented as 0-dimensional float tensors.

            The calculated metric is:
                autoencoder_norm: Autoencoder weight norm regularization term.
        """
        metrics = {}
        if getattr(model, "sequence_autodecoder", None) is not None:
            ad_penalty = model.sequence_autodecoder.calculate_squared_encoding_norm()
            if ad_penalty is not None:
                metrics["autodecoder_norm"] = ad_penalty

        if keys_prefix is not None:
            metrics = {(keys_prefix + k): v for k, v in metrics.items()}

        return metrics


@registry.register
class ViewMetrics(ViewMetricsBase):
    def forward(
        self,
        raymarched: RendererOutput,
        ray_bundle: ImplicitronRayBundle,
        image_rgb: Optional[torch.Tensor] = None,
        depth_map: Optional[torch.Tensor] = None,
        fg_probability: Optional[torch.Tensor] = None,
        mask_crop: Optional[torch.Tensor] = None,
        keys_prefix: str = "loss_",
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Calculates various differentiable metrics useful for supervising
        differentiable rendering pipelines.

        Args:
            results: A dict to store the results in.
            raymarched.features: Predicted rgb or feature values.
            raymarched.depths: A tensor of shape `(B, ..., 1)` containing
                predicted depth values.
            raymarched.masks: A tensor of shape `(B, ..., 1)` containing
                predicted foreground masks.
            raymarched.aux["grad_theta"]: A tensor of shape `(B, ..., 3)` containing an
                evaluation of a gradient of a signed distance function w.r.t.
                input 3D coordinates used to compute the eikonal loss.
            raymarched.aux["density_grid"]: A tensor of shape `(B, Hg, Wg, Dg, 1)`
                containing a `Hg x Wg x Dg` voxel grid of density values.
            ray_bundle: ImplicitronRayBundle object which was used to produce the raymarched
                object
            image_rgb: A tensor of shape `(B, H, W, 3)` containing ground truth rgb
                values.
            depth_map: A tensor of shape `(B, Hd, Wd, 1)` containing ground truth depth
                values.
            fg_probability: A tensor of shape `(B, Hm, Wm, 1)` containing ground truth
                foreground masks.
            keys_prefix: A common prefix for all keys in the output dictionary
                containing all view metrics.

        Returns:
            A dictionary `{metric_name_i: metric_value_i}` keyed by the
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
        metrics = self._calculate_stage(
            raymarched,
            ray_bundle,
            image_rgb,
            depth_map,
            fg_probability,
            mask_crop,
            keys_prefix,
        )

        if raymarched.prev_stage:
            metrics.update(
                self(
                    raymarched.prev_stage,
                    ray_bundle,
                    image_rgb,
                    depth_map,
                    fg_probability,
                    mask_crop,
                    keys_prefix=(keys_prefix + "prev_stage_"),
                )
            )

        return metrics

    def _calculate_stage(
        self,
        raymarched: RendererOutput,
        ray_bundle: ImplicitronRayBundle,
        image_rgb: Optional[torch.Tensor] = None,
        depth_map: Optional[torch.Tensor] = None,
        fg_probability: Optional[torch.Tensor] = None,
        mask_crop: Optional[torch.Tensor] = None,
        keys_prefix: str = "loss_",
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Calculate metrics for the current stage.
        """
        # TODO: extract functions

        # reshape from B x ... x DIM to B x DIM x -1 x 1
        image_rgb_pred, fg_probability_pred, depth_map_pred = [
            _reshape_nongrid_var(x)
            for x in [raymarched.features, raymarched.masks, raymarched.depths]
        ]
        xys = ray_bundle.xys

        # If ray_bundle is packed than we can sample images in padded state to lower
        # memory requirements. Instead of having one image for every element in
        # ray_bundle we can than have one image per unique sampled camera.
        if ray_bundle.is_packed():
            xys, first_idxs, num_inputs = ray_bundle.get_padded_xys()

        # reshape the sampling grid as well
        # TODO: we can get rid of the singular dimension here and in _reshape_nongrid_var
        # now that we use rend_utils.ndc_grid_sample
        xys = xys.reshape(xys.shape[0], -1, 1, 2)

        # closure with the given xys
        def sample_full(tensor, mode):
            if tensor is None:
                return tensor
            return rend_utils.ndc_grid_sample(tensor, xys, mode=mode)

        def sample_packed(tensor, mode):
            if tensor is None:
                return tensor

            # select images that corespond to sampled cameras if raybundle is packed
            tensor = tensor[ray_bundle.camera_ids]
            if ray_bundle.is_packed():
                # select images that corespond to sampled cameras if raybundle is packed
                tensor = tensor[ray_bundle.camera_ids]
            result = rend_utils.ndc_grid_sample(tensor, xys, mode=mode)
            return padded_to_packed(result, first_idxs, num_inputs, max_size_dim=2)[
                :, :, None
            ]  # the result is [n_rays_total_training, 3, 1, 1]

        sample = sample_packed if ray_bundle.is_packed() else sample_full

        # eval all results in this size
        image_rgb = sample(image_rgb, mode="bilinear")
        depth_map = sample(depth_map, mode="nearest")
        fg_probability = sample(fg_probability, mode="nearest")
        mask_crop = sample(mask_crop, mode="nearest")
        if mask_crop is None and image_rgb_pred is not None:
            mask_crop = torch.ones_like(image_rgb_pred[:, :1])
        if mask_crop is None and depth_map_pred is not None:
            mask_crop = torch.ones_like(depth_map_pred[:, :1])

        metrics = {}
        if image_rgb is not None and image_rgb_pred is not None:
            metrics.update(
                _rgb_metrics(
                    image_rgb,
                    image_rgb_pred,
                    masks=fg_probability,
                    masks_crop=mask_crop,
                )
            )

        if fg_probability_pred is not None:
            metrics["mask_beta_prior"] = utils.beta_prior(fg_probability_pred)
        if fg_probability is not None and fg_probability_pred is not None:
            metrics["mask_neg_iou"] = utils.neg_iou_loss(
                fg_probability_pred, fg_probability, mask=mask_crop
            )
            if torch.is_autocast_enabled():
                # To avoid issues with mixed precision
                metrics["mask_bce"] = utils.calc_bce(
                    fg_probability_pred.logit(),
                    fg_probability,
                    mask=mask_crop,
                    pred_logits=True,
                )
            else:
                metrics["mask_bce"] = utils.calc_bce(
                    fg_probability_pred,
                    fg_probability,
                    mask=mask_crop,
                    pred_logits=False,
                )

        if depth_map is not None and depth_map_pred is not None:
            assert mask_crop is not None
            _, abs_ = utils.eval_depth(
                depth_map_pred, depth_map, get_best_scale=True, mask=mask_crop, crop=0
            )
            metrics["depth_abs"] = abs_.mean()

            if fg_probability is not None:
                mask = fg_probability * mask_crop
                _, abs_ = utils.eval_depth(
                    depth_map_pred,
                    depth_map,
                    get_best_scale=True,
                    mask=mask,
                    crop=0,
                )
                metrics["depth_abs_fg"] = abs_.mean()

        # regularizers
        grad_theta = raymarched.aux.get("grad_theta")
        if grad_theta is not None:
            metrics["eikonal"] = _get_eikonal_loss(grad_theta)

        density_grid = raymarched.aux.get("density_grid")
        if density_grid is not None:
            metrics["density_tv"] = _get_grid_tv_loss(density_grid)

        if depth_map_pred is not None:
            metrics["depth_neg_penalty"] = _get_depth_neg_penalty_loss(depth_map_pred)

        if keys_prefix is not None:
            metrics = {(keys_prefix + k): v for k, v in metrics.items()}

        return metrics


def _rgb_metrics(
    images,
    images_pred,
    masks=None,
    masks_crop=None,
    huber_scaling: float = 0.03,
):
    assert masks_crop is not None
    if images.shape[1] != images_pred.shape[1]:
        raise ValueError(
            f"Network output's RGB images had {images_pred.shape[1]} "
            f"channels. {images.shape[1]} expected."
        )
    rgb_abs = ((images_pred - images).abs()).mean(dim=1, keepdim=True)
    rgb_squared = ((images_pred - images) ** 2).mean(dim=1, keepdim=True)
    rgb_loss = utils.huber(rgb_squared, scaling=huber_scaling)
    crop_mass = masks_crop.sum().clamp(1.0)
    results = {
        "rgb_huber": (rgb_loss * masks_crop).sum() / crop_mass,
        "rgb_l1": (rgb_abs * masks_crop).sum() / crop_mass,
        "rgb_mse": (rgb_squared * masks_crop).sum() / crop_mass,
        "rgb_psnr": utils.calc_psnr(images_pred, images, mask=masks_crop),
    }
    if masks is not None:
        masks = masks_crop * masks
        results["rgb_psnr_fg"] = utils.calc_psnr(images_pred, images, mask=masks)
        results["rgb_mse_fg"] = (rgb_squared * masks).sum() / masks.sum().clamp(1.0)
    return results


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
