# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from pytorch3d.implicitron.tools.config import Configurable, expand_args_fields
from pytorch3d.renderer import RayBundle
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
    """

    n_pts_per_ray: int
    random_sampling: bool
    add_input_samples: bool = True

    def __post_init__(self) -> None:
        super().__init__()

    def forward(
        self,
        input_ray_bundle: RayBundle,
        ray_weights: torch.Tensor,
        **kwargs,
    ) -> RayBundle:
        """
        Args:
            input_ray_bundle: An instance of `RayBundle` specifying the
                source rays for sampling of the probability distribution.
            ray_weights: A tensor of shape
                `(..., input_ray_bundle.legths.shape[-1])` with non-negative
                elements defining the probability distribution to sample
                ray points from.

        Returns:
            ray_bundle: A new `RayBundle` instance containing the input ray
                points together with `n_pts_per_ray` additionally sampled
                points per ray. For each ray, the lengths are sorted.
        """

        z_vals = input_ray_bundle.lengths
        with torch.no_grad():
            z_vals_mid = torch.lerp(z_vals[..., 1:], z_vals[..., :-1], 0.5)
            z_samples = sample_pdf(
                z_vals_mid.view(-1, z_vals_mid.shape[-1]),
                ray_weights.view(-1, ray_weights.shape[-1])[..., 1:-1],
                self.n_pts_per_ray,
                det=not self.random_sampling,
            ).view(*z_vals.shape[:-1], self.n_pts_per_ray)

        if self.add_input_samples:
            # Add the new samples to the input ones.
            z_vals = torch.cat((z_vals, z_samples), dim=-1)
        else:
            z_vals = z_samples
        # Resort by depth.
        z_vals, _ = torch.sort(z_vals, dim=-1)

        return RayBundle(
            origins=input_ray_bundle.origins,
            directions=input_ray_bundle.directions,
            lengths=z_vals,
            xys=input_ray_bundle.xys,
        )
