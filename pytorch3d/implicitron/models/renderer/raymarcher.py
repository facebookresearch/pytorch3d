# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, Dict, Tuple, Union

import torch
from pytorch3d.renderer.implicit.raymarching import _check_raymarcher_inputs


_TTensor = torch.Tensor


class GenericRaymarcher(torch.nn.Module):
    """
    This generalizes the `pytorch3d.renderer.EmissionAbsorptionRaymarcher`
    and NeuralVolumes' Accumulative ray marcher. It additionally returns
    the rendering weights that can be used in the NVS pipeline to carry out
    the importance ray-sampling in the refining pass.
    Different from `EmissionAbsorptionRaymarcher`, it takes raw
    (non-exponentiated) densities.

    Args:
        bg_color: background_color. Must be of shape (1,) or (feature_dim,)
    """

    def __init__(
        self,
        surface_thickness: int = 1,
        bg_color: Union[Tuple[float, ...], _TTensor] = (0.0,),
        capping_function: str = "exponential",  # exponential | cap1
        weight_function: str = "product",  # product | minimum
        background_opacity: float = 0.0,
        density_relu: bool = True,
        blend_output: bool = True,
    ):
        """
        Args:
            surface_thickness: Denotes the overlap between the absorption
                function and the density function.
        """
        super().__init__()
        self.surface_thickness = surface_thickness
        self.density_relu = density_relu
        self.background_opacity = background_opacity
        self.blend_output = blend_output
        if not isinstance(bg_color, torch.Tensor):
            bg_color = torch.tensor(bg_color)

        if bg_color.ndim != 1:
            raise ValueError(f"bg_color (shape {bg_color.shape}) should be a 1D tensor")

        self.register_buffer("_bg_color", bg_color, persistent=False)

        self._capping_function: Callable[[_TTensor], _TTensor] = {
            "exponential": lambda x: 1.0 - torch.exp(-x),
            "cap1": lambda x: x.clamp(max=1.0),
        }[capping_function]

        self._weight_function: Callable[[_TTensor, _TTensor], _TTensor] = {
            "product": lambda curr, acc: curr * acc,
            "minimum": lambda curr, acc: torch.minimum(curr, acc),
        }[weight_function]

    def forward(
        self,
        rays_densities: torch.Tensor,
        rays_features: torch.Tensor,
        aux: Dict[str, Any],
        ray_lengths: torch.Tensor,
        density_noise_std: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Args:
            rays_densities: Per-ray density values represented with a tensor
                of shape `(..., n_points_per_ray, 1)`.
            rays_features: Per-ray feature values represented with a tensor
                of shape `(..., n_points_per_ray, feature_dim)`.
            aux: a dictionary with extra information.
            ray_lengths: Per-ray depth values represented with a tensor
                of shape `(..., n_points_per_ray, feature_dim)`.
            density_noise_std: the magnitude of the noise added to densities.

        Returns:
            features: A tensor of shape `(..., feature_dim)` containing
                the rendered features for each ray.
            depth: A tensor of shape `(..., 1)` containing estimated depth.
            opacities: A tensor of shape `(..., 1)` containing rendered opacsities.
            weights: A tensor of shape `(..., n_points_per_ray)` containing
                the ray-specific non-negative opacity weights. In general, they
                don't sum to 1 but do not overcome it, i.e.
                `(weights.sum(dim=-1) <= 1.0).all()` holds.
        """
        _check_raymarcher_inputs(
            rays_densities,
            rays_features,
            ray_lengths,
            z_can_be_none=True,
            features_can_be_none=False,
            density_1d=True,
        )

        deltas = torch.cat(
            (
                ray_lengths[..., 1:] - ray_lengths[..., :-1],
                self.background_opacity * torch.ones_like(ray_lengths[..., :1]),
            ),
            dim=-1,
        )

        rays_densities = rays_densities[..., 0]

        if density_noise_std > 0.0:
            rays_densities = (
                rays_densities + torch.randn_like(rays_densities) * density_noise_std
            )
        if self.density_relu:
            rays_densities = torch.relu(rays_densities)

        weighted_densities = deltas * rays_densities
        capped_densities = self._capping_function(weighted_densities)

        rays_opacities = self._capping_function(
            torch.cumsum(weighted_densities, dim=-1)
        )
        opacities = rays_opacities[..., -1:]
        absorption_shifted = (-rays_opacities + 1.0).roll(
            self.surface_thickness, dims=-1
        )
        absorption_shifted[..., : self.surface_thickness] = 1.0

        weights = self._weight_function(capped_densities, absorption_shifted)
        features = (weights[..., None] * rays_features).sum(dim=-2)
        depth = (weights * ray_lengths)[..., None].sum(dim=-2)

        alpha = opacities if self.blend_output else 1
        if self._bg_color.shape[-1] not in [1, features.shape[-1]]:
            raise ValueError("Wrong number of background color channels.")
        features = alpha * features + (1 - opacities) * self._bg_color

        return features, depth, opacities, weights, aux
