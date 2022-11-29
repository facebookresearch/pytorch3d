# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from typing import Optional, Tuple, Union

import torch


class EmissionAbsorptionRaymarcher(torch.nn.Module):
    """
    Raymarch using the Emission-Absorption (EA) algorithm.

    The algorithm independently renders each ray by analyzing density and
    feature values sampled at (typically uniformly) spaced 3D locations along
    each ray. The density values `rays_densities` are of shape
    `(..., n_points_per_ray)`, their values should range between [0, 1], and
    represent the opaqueness of each point (the higher the less transparent).
    The feature values `rays_features` of shape
    `(..., n_points_per_ray, feature_dim)` represent the content of the
    point that is supposed to be rendered in case the given point is opaque
    (i.e. its density -> 1.0).

    EA first utilizes `rays_densities` to compute the absorption function
    along each ray as follows::

        absorption = cumprod(1 - rays_densities, dim=-1)

    The value of absorption at position `absorption[..., k]` specifies
    how much light has reached `k`-th point along a ray since starting
    its trajectory at `k=0`-th point.

    Each ray is then rendered into a tensor `features` of shape `(..., feature_dim)`
    by taking a weighed combination of per-ray features `rays_features` as follows::

        weights = absorption * rays_densities
        features = (rays_features * weights).sum(dim=-2)

    Where `weights` denote a function that has a strong peak around the location
    of the first surface point that a given ray passes through.

    Note that for a perfectly bounded volume (with a strictly binary density),
    the `weights = cumprod(1 - rays_densities, dim=-1) * rays_densities`
    function would yield 0 everywhere. In order to prevent this,
    the result of the cumulative product is shifted `self.surface_thickness`
    elements along the ray direction.
    """

    def __init__(self, surface_thickness: int = 1) -> None:
        """
        Args:
            surface_thickness: Denotes the overlap between the absorption
                function and the density function.
        """
        super().__init__()
        self.surface_thickness = surface_thickness

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
            features_opacities: A tensor of shape `(..., feature_dim+1)`
                that concatenates two tensors along the last dimension:
                    1) features: A tensor of per-ray renders
                        of shape `(..., feature_dim)`.
                    2) opacities: A tensor of per-ray opacity values
                        of shape `(..., 1)`. Its values range between [0, 1] and
                        denote the total amount of light that has been absorbed
                        for each ray. E.g. a value of 0 corresponds to the ray
                        completely passing through a volume. Please refer to the
                        `AbsorptionOnlyRaymarcher` documentation for the
                        explanation of the algorithm that computes `opacities`.
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
        opacities = 1.0 - torch.prod(1.0 - rays_densities, dim=-1, keepdim=True)

        return torch.cat((features, opacities), dim=-1)


class AbsorptionOnlyRaymarcher(torch.nn.Module):
    """
    Raymarch using the Absorption-Only (AO) algorithm.

    The algorithm independently renders each ray by analyzing density and
    feature values sampled at (typically uniformly) spaced 3D locations along
    each ray. The density values `rays_densities` are of shape
    `(..., n_points_per_ray, 1)`, their values should range between [0, 1], and
    represent the opaqueness of each point (the higher the less transparent).
    The algorithm only measures the total amount of light absorbed along each ray
    and, besides outputting per-ray `opacity` values of shape `(...,)`,
    does not produce any feature renderings.

    The algorithm simply computes `total_transmission = prod(1 - rays_densities)`
    of shape `(..., 1)` which, for each ray, measures the total amount of light
    that passed through the volume.
    It then returns `opacities = 1 - total_transmission`.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self, rays_densities: torch.Tensor, **kwargs
    ) -> Union[None, torch.Tensor]:
        """
        Args:
            rays_densities: Per-ray density values represented with a tensor
                of shape `(..., n_points_per_ray)` whose values range in [0, 1].

        Returns:
            opacities: A tensor of per-ray opacity values of shape `(..., 1)`.
                Its values range between [0, 1] and denote the total amount
                of light that has been absorbed for each ray. E.g. a value
                of 0 corresponds to the ray completely passing through a volume.
        """

        _check_raymarcher_inputs(
            rays_densities,
            None,
            None,
            features_can_be_none=True,
            z_can_be_none=True,
            density_1d=True,
        )
        rays_densities = rays_densities[..., 0]
        _check_density_bounds(rays_densities)
        total_transmission = torch.prod(1 - rays_densities, dim=-1, keepdim=True)
        opacities = 1.0 - total_transmission
        return opacities


def _shifted_cumprod(x, shift: int = 1):
    """
    Computes `torch.cumprod(x, dim=-1)` and prepends `shift` number of
    ones and removes `shift` trailing elements to/from the last dimension
    of the result.
    """
    x_cumprod = torch.cumprod(x, dim=-1)
    x_cumprod_shift = torch.cat(
        [torch.ones_like(x_cumprod[..., :shift]), x_cumprod[..., :-shift]], dim=-1
    )
    return x_cumprod_shift


def _check_density_bounds(
    rays_densities: torch.Tensor, bounds: Tuple[float, float] = (0.0, 1.0)
) -> None:
    """
    Checks whether the elements of `rays_densities` range within `bounds`.
    If not issues a warning.
    """
    with torch.no_grad():
        if (rays_densities.max() > bounds[1]) or (rays_densities.min() < bounds[0]):
            warnings.warn(
                "One or more elements of rays_densities are outside of valid"
                + f"range {str(bounds)}"
            )


def _check_raymarcher_inputs(
    rays_densities: torch.Tensor,
    rays_features: Optional[torch.Tensor],
    rays_z: Optional[torch.Tensor],
    features_can_be_none: bool = False,
    z_can_be_none: bool = False,
    density_1d: bool = True,
) -> None:
    """
    Checks the validity of the inputs to raymarching algorithms.
    """
    if not torch.is_tensor(rays_densities):
        raise ValueError("rays_densities has to be an instance of torch.Tensor.")

    if not z_can_be_none and not torch.is_tensor(rays_z):
        raise ValueError("rays_z has to be an instance of torch.Tensor.")

    if not features_can_be_none and not torch.is_tensor(rays_features):
        raise ValueError("rays_features has to be an instance of torch.Tensor.")

    if rays_densities.ndim < 1:
        raise ValueError("rays_densities have to have at least one dimension.")

    if density_1d and rays_densities.shape[-1] != 1:
        raise ValueError(
            "The size of the last dimension of rays_densities has to be one."
            + f" Got shape {rays_densities.shape}."
        )

    rays_shape = rays_densities.shape[:-1]

    # pyre-fixme[16]: `Optional` has no attribute `shape`.
    if not z_can_be_none and rays_z.shape != rays_shape:
        raise ValueError("rays_z have to be of the same shape as rays_densities.")

    if not features_can_be_none and rays_features.shape[:-1] != rays_shape:
        raise ValueError(
            "The first to previous to last dimensions of rays_features"
            " have to be the same as all dimensions of rays_densities."
        )
