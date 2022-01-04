# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from pytorch3d.renderer import EmissionAbsorptionRaymarcher
from pytorch3d.renderer.implicit.raymarching import (
    _check_density_bounds,
    _check_raymarcher_inputs,
    _shifted_cumprod,
)


class EmissionAbsorptionNeRFRaymarcher(EmissionAbsorptionRaymarcher):
    """
    This is essentially the `pytorch3d.renderer.EmissionAbsorptionRaymarcher`
    which additionally returns the rendering weights. It also skips returning
    the computation of the alpha-mask which is, in case of NeRF, equal to 1
    everywhere.

    The weights are later used in the NeRF pipeline to carry out the importance
    ray-sampling for the fine rendering pass.

    For more details about the EmissionAbsorptionRaymarcher please refer to
    the documentation of `pytorch3d.renderer.EmissionAbsorptionRaymarcher`.
    """

    def forward(
        self,
        rays_densities: torch.Tensor,
        rays_features: torch.Tensor,
        eps: float = 1e-10,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            rays_densities: Per-ray density values represented with a tensor
                of shape `(..., n_points_per_ray, 1)` whose values range in [0, 1].
            rays_features: Per-ray feature values represented with a tensor
                of shape `(..., n_points_per_ray, feature_dim)`.
            eps: A lower bound added to `rays_densities` before computing
                the absorption function (cumprod of `1-rays_densities` along
                each ray). This prevents the cumprod to yield exact 0
                which would inhibit any gradient-based learning.

        Returns:
            features: A tensor of shape `(..., feature_dim)` containing
                the rendered features for each ray.
            weights: A tensor of shape `(..., n_points_per_ray)` containing
                the ray-specific emission-absorption distribution.
                Each ray distribution `(..., :)` is a valid probability
                distribution, i.e. it contains non-negative values that integrate
                to 1, such that `weights.sum(dim=-1)==1).all()` yields `True`.
        """
        _check_raymarcher_inputs(
            rays_densities,
            rays_features,
            None,
            z_can_be_none=True,
            features_can_be_none=False,
            density_1d=True,
        )
        _check_density_bounds(rays_densities)
        rays_densities = rays_densities[..., 0]
        absorption = _shifted_cumprod(
            (1.0 + eps) - rays_densities, shift=self.surface_thickness
        )
        weights = rays_densities * absorption
        features = (weights[..., None] * rays_features).sum(dim=-2)

        return features, weights
