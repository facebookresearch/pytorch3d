# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe


# Note: The #noqa comments below are for unused imports of pluggable implementations
# which are part of implicitron. They ensure that the registry is prepopulated.

import warnings
from logging import Logger
from typing import Any, Dict, Optional, Tuple

import torch
import tqdm
from pytorch3d.common.compat import prod

from pytorch3d.implicitron.models.renderer.base import ImplicitronRayBundle

from pytorch3d.implicitron.tools import image_utils

from pytorch3d.implicitron.tools.utils import cat_dataclass


def preprocess_input(
    image_rgb: Optional[torch.Tensor],
    fg_probability: Optional[torch.Tensor],
    depth_map: Optional[torch.Tensor],
    mask_images: bool,
    mask_depths: bool,
    mask_threshold: float,
    bg_color: Tuple[float, float, float],
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Helper function to preprocess the input images and optional depth maps
    to apply masking if required.

    Args:
        image_rgb: A tensor of shape `(B, 3, H, W)` containing a batch of rgb images
            corresponding to the source viewpoints from which features will be extracted
        fg_probability: A tensor of shape `(B, 1, H, W)` containing a batch
            of foreground masks with values in [0, 1].
        depth_map: A tensor of shape `(B, 1, H, W)` containing a batch of depth maps.
        mask_images: Whether or not to mask the RGB image background given the
            foreground mask (the `fg_probability` argument of `GenericModel.forward`)
        mask_depths: Whether or not to mask the depth image background given the
            foreground mask (the `fg_probability` argument of `GenericModel.forward`)
        mask_threshold: If greater than 0.0, the foreground mask is
            thresholded by this value before being applied to the RGB/Depth images
        bg_color: RGB values for setting the background color of input image
            if mask_images=True. Defaults to (0.0, 0.0, 0.0). Each renderer has its own
            way to determine the background color of its output, unrelated to this.

    Returns:
        Modified image_rgb, fg_mask, depth_map
    """
    if image_rgb is not None and image_rgb.ndim == 3:
        # The FrameData object is used for both frames and batches of frames,
        # and a user might get this error if those were confused.
        # Perhaps a user has a FrameData `fd` representing a single frame and
        # wrote something like `model(**fd)` instead of
        # `model(**fd.collate([fd]))`.
        raise ValueError(
            "Model received unbatched inputs. "
            + "Perhaps they came from a FrameData which had not been collated."
        )

    fg_mask = fg_probability
    if fg_mask is not None and mask_threshold > 0.0:
        # threshold masks
        warnings.warn("Thresholding masks!")
        fg_mask = (fg_mask >= mask_threshold).type_as(fg_mask)

    if mask_images and fg_mask is not None and image_rgb is not None:
        # mask the image
        warnings.warn("Masking images!")
        image_rgb = image_utils.mask_background(
            image_rgb, fg_mask, dim_color=1, bg_color=torch.tensor(bg_color)
        )

    if mask_depths and fg_mask is not None and depth_map is not None:
        # mask the depths
        assert (
            mask_threshold > 0.0
        ), "Depths should be masked only with thresholded masks"
        warnings.warn("Masking depths!")
        depth_map = depth_map * fg_mask

    return image_rgb, fg_mask, depth_map


def log_loss_weights(loss_weights: Dict[str, float], logger: Logger) -> None:
    """
    Print a table of the loss weights.
    """
    loss_weights_message = (
        "-------\nloss_weights:\n"
        + "\n".join(f"{k:40s}: {w:1.2e}" for k, w in loss_weights.items())
        + "-------"
    )
    logger.info(loss_weights_message)


def weighted_sum_losses(
    preds: Dict[str, torch.Tensor], loss_weights: Dict[str, float]
) -> Optional[torch.Tensor]:
    """
    A helper function to compute the overall loss as the dot product
    of individual loss functions with the corresponding weights.
    """
    losses_weighted = [
        preds[k] * float(w)
        for k, w in loss_weights.items()
        if (k in preds and w != 0.0)
    ]
    if len(losses_weighted) == 0:
        warnings.warn("No main objective found.")
        return None
    loss = sum(losses_weighted)
    assert torch.is_tensor(loss)
    return loss


def apply_chunked(func, chunk_generator, tensor_collator):
    """
    Helper function to apply a function on a sequence of
    chunked inputs yielded by a generator and collate
    the result.
    """
    processed_chunks = [
        func(*chunk_args, **chunk_kwargs)
        for chunk_args, chunk_kwargs in chunk_generator
    ]

    return cat_dataclass(processed_chunks, tensor_collator)


def chunk_generator(
    chunk_size: int,
    ray_bundle: ImplicitronRayBundle,
    chunked_inputs: Dict[str, torch.Tensor],
    tqdm_trigger_threshold: int,
    *args,
    **kwargs,
):
    """
    Helper function which yields chunks of rays from the
    input ray_bundle, to be used when the number of rays is
    large and will not fit in memory for rendering.
    """
    (
        batch_size,
        *spatial_dim,
        n_pts_per_ray,
    ) = ray_bundle.lengths.shape  # B x ... x n_pts_per_ray
    if n_pts_per_ray > 0 and chunk_size % n_pts_per_ray != 0:
        raise ValueError(
            f"chunk_size_grid ({chunk_size}) should be divisible "
            f"by n_pts_per_ray ({n_pts_per_ray})"
        )

    n_rays = prod(spatial_dim)
    # special handling for raytracing-based methods
    n_chunks = -(-n_rays * max(n_pts_per_ray, 1) // chunk_size)
    chunk_size_in_rays = -(-n_rays // n_chunks)

    iter = range(0, n_rays, chunk_size_in_rays)
    if len(iter) >= tqdm_trigger_threshold:
        iter = tqdm.tqdm(iter)

    def _safe_slice(
        tensor: Optional[torch.Tensor], start_idx: int, end_idx: int
    ) -> Any:
        return tensor[start_idx:end_idx] if tensor is not None else None

    for start_idx in iter:
        end_idx = min(start_idx + chunk_size_in_rays, n_rays)
        bins = (
            None
            if ray_bundle.bins is None
            else ray_bundle.bins.reshape(batch_size, n_rays, n_pts_per_ray + 1)[
                :, start_idx:end_idx
            ]
        )
        pixel_radii_2d = (
            None
            if ray_bundle.pixel_radii_2d is None
            else ray_bundle.pixel_radii_2d.reshape(batch_size, -1, 1)[
                :, start_idx:end_idx
            ]
        )
        ray_bundle_chunk = ImplicitronRayBundle(
            origins=ray_bundle.origins.reshape(batch_size, -1, 3)[:, start_idx:end_idx],
            directions=ray_bundle.directions.reshape(batch_size, -1, 3)[
                :, start_idx:end_idx
            ],
            lengths=ray_bundle.lengths.reshape(batch_size, n_rays, n_pts_per_ray)[
                :, start_idx:end_idx
            ],
            xys=ray_bundle.xys.reshape(batch_size, -1, 2)[:, start_idx:end_idx],
            bins=bins,
            pixel_radii_2d=pixel_radii_2d,
            camera_ids=_safe_slice(ray_bundle.camera_ids, start_idx, end_idx),
            camera_counts=_safe_slice(ray_bundle.camera_counts, start_idx, end_idx),
        )
        extra_args = kwargs.copy()
        for k, v in chunked_inputs.items():
            extra_args[k] = v.flatten(2)[:, :, start_idx:end_idx]
        yield [ray_bundle_chunk, *args], extra_args
