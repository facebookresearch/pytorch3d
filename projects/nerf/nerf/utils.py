# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch


def sample_pdf(
    bins: torch.Tensor,
    weights: torch.Tensor,
    N_samples: int,
    det: bool = False,
    eps: float = 1e-5,
):
    """
    Samples a probability density functions defined by bin edges `bins` and
    the non-negative per-bin probabilities `weights`.

    Note: This is a direct conversion of the TensorFlow function from the original
    release [1] to PyTorch.

    Args:
        bins: Tensor of shape `(..., n_bins+1)` denoting the edges of the sampling bins.
        weights: Tensor of shape `(..., n_bins)` containing non-negative numbers
            representing the probability of sampling the corresponding bin.
        N_samples: The number of samples to draw from each set of bins.
        det: If `False`, the sampling is random. `True` yields deterministic
            uniformly-spaced sampling from the inverse cumulative density function.
        eps: A constant preventing division by zero in case empty bins are present.

    Returns:
        samples: Tensor of shape `(..., N_samples)` containing `N_samples` samples
            drawn from each set probability distribution.

    Refs:
        [1] https://github.com/bmild/nerf/blob/55d8b00244d7b5178f4d003526ab6667683c9da9/run_nerf_helpers.py#L183  # noqa E501
    """

    # Get pdf
    weights = weights + eps  # prevent nans
    pdf = weights / weights.sum(dim=-1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)

    # Take uniform samples
    if det:
        u = torch.linspace(0.0, 1.0, N_samples, device=cdf.device, dtype=cdf.dtype)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples]).contiguous()
    else:
        u = torch.rand(
            list(cdf.shape[:-1]) + [N_samples], device=cdf.device, dtype=cdf.dtype
        )

    # Invert CDF
    inds = torch.searchsorted(cdf, u, right=True)
    below = (inds - 1).clamp(0)
    above = inds.clamp(max=cdf.shape[-1] - 1)
    inds_g = torch.stack([below, above], -1).view(
        *below.shape[:-1], below.shape[-1] * 2
    )

    cdf_g = torch.gather(cdf, -1, inds_g).view(*below.shape, 2)
    bins_g = torch.gather(bins, -1, inds_g).view(*below.shape, 2)

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom = torch.where(denom < eps, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


def calc_mse(x: torch.Tensor, y: torch.Tensor):
    """
    Calculates the mean square error between tensors `x` and `y`.
    """
    return torch.mean((x - y) ** 2)


def calc_psnr(x: torch.Tensor, y: torch.Tensor):
    """
    Calculates the Peak-signal-to-noise ratio between tensors `x` and `y`.
    """
    mse = calc_mse(x, y)
    psnr = -10.0 * torch.log10(mse)
    return psnr


def sample_images_at_mc_locs(
    target_images: torch.Tensor,
    sampled_rays_xy: torch.Tensor,
):
    """
    Given a set of pixel locations `sampled_rays_xy` this method samples the tensor
    `target_images` at the respective 2D locations.

    This function is used in order to extract the colors from ground truth images
    that correspond to the colors rendered using a Monte Carlo rendering.

    Args:
        target_images: A tensor of shape `(batch_size, ..., 3)`.
        sampled_rays_xy: A tensor of shape `(batch_size, S_1, ..., S_N, 2)`.

    Returns:
        images_sampled: A tensor of shape `(batch_size, S_1, ..., S_N, 3)`
            containing `target_images` sampled at `sampled_rays_xy`.
    """
    ba = target_images.shape[0]
    dim = target_images.shape[-1]
    spatial_size = sampled_rays_xy.shape[1:-1]

    # The coordinate grid convention for grid_sample has both x and y
    # directions inverted.
    xy_sample = -sampled_rays_xy.view(ba, -1, 1, 2).clone()

    images_sampled = torch.nn.functional.grid_sample(
        target_images.permute(0, 3, 1, 2),
        xy_sample,
        align_corners=True,
        mode="bilinear",
    )
    return images_sampled.permute(0, 2, 3, 1).view(ba, *spatial_size, dim)
