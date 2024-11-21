# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import Any, Callable, Dict, Optional, Tuple

import torch
from pytorch3d.implicitron.models.renderer.base import RendererOutput
from pytorch3d.implicitron.tools.config import registry, ReplaceableBase
from pytorch3d.renderer.implicit.raymarching import _check_raymarcher_inputs


_TTensor = torch.Tensor


class RaymarcherBase(ReplaceableBase):
    """
    Defines a base class for raymarchers. Specifically, a raymarcher is responsible
    for taking a set of features and density descriptors along rendering rays
    and marching along them in order to generate a feature render.
    """

    def forward(
        self,
        rays_densities: torch.Tensor,
        rays_features: torch.Tensor,
        aux: Dict[str, Any],
    ) -> RendererOutput:
        """
        Args:
            rays_densities: Per-ray density values represented with a tensor
                of shape `(..., n_points_per_ray, 1)`.
            rays_features: Per-ray feature values represented with a tensor
                of shape `(..., n_points_per_ray, feature_dim)`.
            aux: a dictionary with extra information.
        """
        raise NotImplementedError()


class AccumulativeRaymarcherBase(RaymarcherBase, torch.nn.Module):
    """
    This generalizes the `pytorch3d.renderer.EmissionAbsorptionRaymarcher`
    and NeuralVolumes' cumsum ray marcher. It additionally returns
    the rendering weights that can be used in the NVS pipeline to carry out
    the importance ray-sampling in the refining pass.
    Different from `pytorch3d.renderer.EmissionAbsorptionRaymarcher`, it takes raw
    (non-exponentiated) densities.

    Args:
        surface_thickness: The thickness of the raymarched surface.
        bg_color: The background color. A tuple of either 1 element or of D elements,
            where D matches the feature dimensionality; it is broadcast when necessary.
        replicate_last_interval: If True, the ray length assigned to the last interval
            for the opacity delta calculation is copied from the penultimate interval.
        background_opacity: The length over which the last raw opacity value
            (i.e. before exponentiation) is considered to apply, for the delta
            calculation. Ignored if replicate_last_interval=True.
        density_relu: If `True`, passes the input density through ReLU before
            raymarching.
        blend_output: If `True`, alpha-blends the output renders with the
            background color using the rendered opacity mask.

        capping_function: The capping function of the raymarcher.
            Options:
                - "exponential" (`cap_fn(x) = 1 - exp(-x)`)
                - "cap1" (`cap_fn(x) = min(x, 1)`)
            Set to "exponential" for the standard Emission Absorption raymarching.
        weight_function: The weighting function of the raymarcher.
            Options:
                - "product" (`weight_fn(w, x) = w * x`)
                - "minimum" (`weight_fn(w, x) = min(w, x)`)
            Set to "product" for the standard Emission Absorption raymarching.
    """

    surface_thickness: int = 1
    bg_color: Tuple[float, ...] = (0.0,)
    replicate_last_interval: bool = False
    background_opacity: float = 0.0
    density_relu: bool = True
    blend_output: bool = False

    @property
    def capping_function_type(self) -> str:
        raise NotImplementedError()

    @property
    def weight_function_type(self) -> str:
        raise NotImplementedError()

    def __post_init__(self):
        """
        Args:
            surface_thickness: Denotes the overlap between the absorption
                function and the density function.
        """
        bg_color = torch.tensor(self.bg_color)
        if bg_color.ndim != 1:
            raise ValueError(f"bg_color (shape {bg_color.shape}) should be a 1D tensor")

        self.register_buffer("_bg_color", bg_color, persistent=False)

        self._capping_function: Callable[[_TTensor], _TTensor] = {
            "exponential": lambda x: 1.0 - torch.exp(-x),
            "cap1": lambda x: x.clamp(max=1.0),
        }[self.capping_function_type]

        self._weight_function: Callable[[_TTensor, _TTensor], _TTensor] = {
            "product": lambda curr, acc: curr * acc,
            "minimum": lambda curr, acc: torch.minimum(curr, acc),
        }[self.weight_function_type]

    # pyre-fixme[14]: `forward` overrides method defined in `RaymarcherBase`
    #  inconsistently.
    def forward(
        self,
        rays_densities: torch.Tensor,
        rays_features: torch.Tensor,
        aux: Dict[str, Any],
        ray_lengths: torch.Tensor,
        ray_deltas: Optional[torch.Tensor] = None,
        density_noise_std: float = 0.0,
        **kwargs,
    ) -> RendererOutput:
        """
        Args:
            rays_densities: Per-ray density values represented with a tensor
                of shape `(..., n_points_per_ray, 1)`.
            rays_features: Per-ray feature values represented with a tensor
                of shape `(..., n_points_per_ray, feature_dim)`.
            aux: a dictionary with extra information.
            ray_lengths: Per-ray depth values represented with a tensor
                of shape `(..., n_points_per_ray, feature_dim)`.
            ray_deltas: Optional differences between consecutive elements along the ray bundle
                represented with a tensor of shape `(..., n_points_per_ray)`. If None,
                these differences are computed from ray_lengths.
            density_noise_std: the magnitude of the noise added to densities.

        Returns:
            features: A tensor of shape `(..., feature_dim)` containing
                the rendered features for each ray.
            depth: A tensor of shape `(..., 1)` containing estimated depth.
            opacities: A tensor of shape `(..., 1)` containing rendered opacities.
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

        if ray_deltas is None:
            ray_lengths_diffs = torch.diff(ray_lengths, dim=-1)
            if self.replicate_last_interval:
                last_interval = ray_lengths_diffs[..., -1:]
            else:
                last_interval = torch.full_like(
                    ray_lengths[..., :1], self.background_opacity
                )
            deltas = torch.cat((ray_lengths_diffs, last_interval), dim=-1)
        else:
            deltas = ray_deltas

        rays_densities = rays_densities[..., 0]

        if density_noise_std > 0.0:
            noise: _TTensor = torch.randn_like(rays_densities).mul(density_noise_std)
            rays_densities = rays_densities + noise
        if self.density_relu:
            rays_densities = torch.relu(rays_densities)

        weighted_densities = deltas * rays_densities
        # pyre-fixme[29]: `Union[Tensor, Module]` is not a function.
        capped_densities = self._capping_function(weighted_densities)

        # pyre-fixme[29]: `Union[Tensor, Module]` is not a function.
        rays_opacities = self._capping_function(
            torch.cumsum(weighted_densities, dim=-1)
        )
        opacities = rays_opacities[..., -1:]
        absorption_shifted = (-rays_opacities + 1.0).roll(
            self.surface_thickness, dims=-1
        )
        absorption_shifted[..., : self.surface_thickness] = 1.0

        # pyre-fixme[29]: `Union[Tensor, Module]` is not a function.
        weights = self._weight_function(capped_densities, absorption_shifted)
        features = (weights[..., None] * rays_features).sum(dim=-2)
        depth = (weights * ray_lengths)[..., None].sum(dim=-2)

        alpha = opacities if self.blend_output else 1
        if self._bg_color.shape[-1] not in [1, features.shape[-1]]:
            raise ValueError("Wrong number of background color channels.")
        # pyre-fixme[58]: `*` is not supported for operand types `int` and
        #  `Union[Tensor, Module]`.
        features = alpha * features + (1 - opacities) * self._bg_color

        return RendererOutput(
            features=features,
            depths=depth,
            masks=opacities,
            weights=weights,
            aux=aux,
        )


@registry.register
class EmissionAbsorptionRaymarcher(AccumulativeRaymarcherBase):
    """
    Implements the EmissionAbsorption raymarcher.
    """

    background_opacity: float = 1e10

    @property
    def capping_function_type(self) -> str:
        return "exponential"

    @property
    def weight_function_type(self) -> str:
        return "product"


@registry.register
class CumsumRaymarcher(AccumulativeRaymarcherBase):
    """
    Implements the NeuralVolumes' cumulative-sum raymarcher.
    """

    @property
    def capping_function_type(self) -> str:
        return "cap1"

    @property
    def weight_function_type(self) -> str:
        return "minimum"
