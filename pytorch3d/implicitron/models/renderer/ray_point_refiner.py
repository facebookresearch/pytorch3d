# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy

import torch
from pytorch3d.implicitron.models.renderer.base import ImplicitronRayBundle
from pytorch3d.implicitron.tools.config import Configurable, expand_args_fields

from pytorch3d.renderer.implicit.sample_pdf import sample_pdf


@expand_args_fields
# pyre-fixme[13]: Attribute `n_pts_per_ray` is never initialized.
# pyre-fixme[13]: Attribute `random_sampling` is never initialized.
class RayPointRefiner(Configurable, torch.nn.Module):
    """
    Implements the importance sampling of points along rays.
    The input is a `RayBundle` object with a `ray_weights` tensor
    which specifies the probabilities of sampling a point along each ray.

    This raysampler is used for the fine rendering pass of NeRF.
    As such, the forward pass accepts the RayBundle output by the
    raysampling of the coarse rendering pass. Hence, it does not
    take cameras as input.

    Args:
        n_pts_per_ray: The number of points to sample along each ray.
        random_sampling: If `False`, returns equispaced percentiles of the
            distribution defined by the input weights, otherwise performs
            sampling from that distribution.
        add_input_samples: Concatenates and returns the sampled values
            together with the input samples.
        blurpool_weights: Use blurpool defined in [1], on the input weights.
        sample_pdf_eps: A constant preventing division by zero in case empty bins
            are present.

    References:
        [1] Jonathan T. Barron, et al. "Mip-NeRF: A Multiscale Representation
            for Anti-Aliasing Neural Radiance Fields." ICCV 2021.
    """

    n_pts_per_ray: int
    random_sampling: bool
    add_input_samples: bool = True
    blurpool_weights: bool = False
    sample_pdf_eps: float = 1e-5

    def forward(
        self,
        input_ray_bundle: ImplicitronRayBundle,
        ray_weights: torch.Tensor,
        blurpool_weights: bool = False,
        sample_pdf_padding: float = 1e-5,
        **kwargs,
    ) -> ImplicitronRayBundle:
        """
        Args:
            input_ray_bundle: An instance of `ImplicitronRayBundle` specifying the
                source rays for sampling of the probability distribution.
            ray_weights: A tensor of shape
                `(..., input_ray_bundle.lengths.shape[-1])` with non-negative
                elements defining the probability distribution to sample
                ray points from.
            blurpool_weights: Use blurpool defined in [1], on the input weights.
            sample_pdf_padding: A constant preventing division by zero in case empty bins
                are present.

        Returns:
            ray_bundle: A new `ImplicitronRayBundle` instance containing the input ray
                points together with `n_pts_per_ray` additionally sampled
                points per ray. For each ray, the lengths are sorted.

        References:
            [1] Jonathan T. Barron, et al. "Mip-NeRF: A Multiscale Representation
                for Anti-Aliasing Neural Radiance Fields." ICCV 2021.

        """

        with torch.no_grad():
            if self.blurpool_weights:
                ray_weights = apply_blurpool_on_weights(ray_weights)

            n_pts_per_ray = self.n_pts_per_ray
            ray_weights = ray_weights.view(-1, ray_weights.shape[-1])
            if input_ray_bundle.bins is None:
                z_vals: torch.Tensor = input_ray_bundle.lengths
                ray_weights = ray_weights[..., 1:-1]
                bins = torch.lerp(z_vals[..., 1:], z_vals[..., :-1], 0.5)
            else:
                z_vals = input_ray_bundle.bins
                n_pts_per_ray += 1
                bins = z_vals
            z_samples = sample_pdf(
                bins.view(-1, bins.shape[-1]),
                ray_weights,
                n_pts_per_ray,
                det=not self.random_sampling,
                eps=self.sample_pdf_eps,
            ).view(*z_vals.shape[:-1], n_pts_per_ray)

        if self.add_input_samples:
            z_vals = torch.cat((z_vals, z_samples), dim=-1)
        else:
            z_vals = z_samples
        # Resort by depth.
        z_vals, _ = torch.sort(z_vals, dim=-1)
        ray_bundle = copy.copy(input_ray_bundle)
        if input_ray_bundle.bins is None:
            ray_bundle.lengths = z_vals
        else:
            ray_bundle.bins = z_vals

        return ray_bundle


def apply_blurpool_on_weights(weights) -> torch.Tensor:
    """
    Filter weights with a 2-tap max filters followed by a 2-tap blur filter,
    which produces a wide and smooth upper envelope on the weights.

    Args:
        weights: Tensor of shape `(..., dim)`

    Returns:
        blured_weights: Tensor of shape `(..., dim)`
    """
    weights_pad = torch.concatenate(
        [
            weights[..., :1],
            weights,
            weights[..., -1:],
        ],
        dim=-1,
    )

    weights_max = torch.nn.functional.max_pool1d(
        weights_pad.flatten(end_dim=-2), 2, stride=1
    )
    return torch.lerp(weights_max[..., :-1], weights_max[..., 1:], 0.5).reshape_as(
        weights
    )
