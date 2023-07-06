# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import torch
from pytorch3d.implicitron.tools.config import ReplaceableBase
from pytorch3d.ops import packed_to_padded
from pytorch3d.renderer.implicit.utils import ray_bundle_variables_to_ray_points


class EvaluationMode(Enum):
    TRAINING = "training"
    EVALUATION = "evaluation"


class RenderSamplingMode(Enum):
    MASK_SAMPLE = "mask_sample"
    FULL_GRID = "full_grid"


class ImplicitronRayBundle:
    """
    Parametrizes points along projection rays by storing ray `origins`,
    `directions` vectors and `lengths` at which the ray-points are sampled.
    Furthermore, the xy-locations (`xys`) of the ray pixels are stored as well.
    Note that `directions` don't have to be normalized; they define unit vectors
    in the respective 1D coordinate systems; see documentation for
    :func:`ray_bundle_to_ray_points` for the conversion formula.

    Ray bundle may represent rays from multiple cameras. In that case, cameras
    are stored in the packed form (i.e. rays from the same camera are stored in
    the consecutive elements). The following indices will be set:
        camera_ids: A tensor of shape (N, ) which indicates which camera
            was used to sample the rays. `N` is the number of different
            sampled cameras.
        camera_counts: A tensor of shape (N, ) which how many times the
            coresponding camera in `camera_ids` was sampled.
            `sum(camera_counts) == minibatch`, where `minibatch = origins.shape[0]`.

    Attributes:
        origins: A tensor of shape `(..., 3)` denoting the
            origins of the sampling rays in world coords.
        directions: A tensor of shape `(..., 3)` containing the direction
            vectors of sampling rays in world coords. They don't have to be normalized;
            they define unit vectors in the respective 1D coordinate systems; see
            documentation for :func:`ray_bundle_to_ray_points` for the conversion formula.
        lengths: A tensor of shape `(..., num_points_per_ray)`
            containing the lengths at which the rays are sampled.
        xys: A tensor of shape `(..., 2)`, the xy-locations (`xys`) of the ray pixels
        camera_ids: An optional tensor of shape (N, ) which indicates which camera
            was used to sample the rays. `N` is the number of unique sampled cameras.
        camera_counts: An optional tensor of shape (N, ) indicates how many times the
            coresponding camera in `camera_ids` was sampled.
            `sum(camera_counts)==total_number_of_rays`.
        bins: An optional tensor of shape `(..., num_points_per_ray + 1)`
            containing the bins at which the rays are sampled. In this case
            lengths should be equal to the midpoints of bins `(..., num_points_per_ray)`.
        pixel_radii_2d: An optional tensor of shape `(..., 1)`
            base radii of the conical frustums.

    Raises:
        ValueError: If either bins or lengths are not provided.
        ValueError: If bins is provided and the last dim is inferior or equal to 1.
    """

    def __init__(
        self,
        origins: torch.Tensor,
        directions: torch.Tensor,
        lengths: Optional[torch.Tensor],
        xys: torch.Tensor,
        camera_ids: Optional[torch.LongTensor] = None,
        camera_counts: Optional[torch.LongTensor] = None,
        bins: Optional[torch.Tensor] = None,
        pixel_radii_2d: Optional[torch.Tensor] = None,
    ):
        if bins is not None and bins.shape[-1] <= 1:
            raise ValueError(
                "The last dim of bins must be at least superior or equal to 2."
            )

        if bins is None and lengths is None:
            raise ValueError(
                "Please set either bins or lengths to initialize an ImplicitronRayBundle."
            )

        self.origins = origins
        self.directions = directions
        self._lengths = lengths if bins is None else None
        self.xys = xys
        self.bins = bins
        self.pixel_radii_2d = pixel_radii_2d
        self.camera_ids = camera_ids
        self.camera_counts = camera_counts

    @property
    def lengths(self) -> torch.Tensor:
        if self.bins is not None:
            # equivalent to: 0.5 * (bins[..., 1:] + bins[..., :-1]) but more efficient
            # pyre-ignore
            return torch.lerp(self.bins[..., :-1], self.bins[..., 1:], 0.5)
        return self._lengths

    @lengths.setter
    def lengths(self, value):
        if self.bins is not None:
            raise ValueError(
                "If the bins attribute is not None you cannot set the lengths attribute."
            )
        else:
            self._lengths = value

    def is_packed(self) -> bool:
        """
        Returns whether the ImplicitronRayBundle carries data in packed state
        """
        return self.camera_ids is not None and self.camera_counts is not None

    def get_padded_xys(self) -> Tuple[torch.Tensor, torch.LongTensor, int]:
        """
        For a packed ray bundle, returns padded rays. Assumes the input bundle is packed
        (i.e. `camera_ids` and `camera_counts` are set).

        Returns:
            - xys: Tensor of shape (N, max_size, ...) containing the padded
                representation of the pixel coordinated;
                where max_size is max of `camera_counts`. The values for camera id `i`
                will be copied to `xys[i, :]`, with zeros padding out the extra inputs.
            - first_idxs: cumulative sum of `camera_counts` defininf the boundaries
                between cameras in the packed representation
            - num_inputs: the number of cameras in the bundle.
        """
        if not self.is_packed():
            raise ValueError("get_padded_xys can be called only on a packed bundle")

        camera_counts = self.camera_counts
        assert camera_counts is not None

        cumsum = torch.cumsum(camera_counts, dim=0, dtype=torch.long)
        first_idxs = torch.cat(
            (camera_counts.new_zeros((1,), dtype=torch.long), cumsum[:-1])
        )
        num_inputs = camera_counts.sum().item()
        max_size = torch.max(camera_counts).item()
        xys = packed_to_padded(self.xys, first_idxs, max_size)
        # pyre-ignore [7] pytorch typeshed inaccuracy
        return xys, first_idxs, num_inputs


@dataclass
class RendererOutput:
    """
    A structure for storing the output of a renderer.

    Args:
        features: rendered features (usually RGB colors), (B, ..., C) tensor.
        depth: rendered ray-termination depth map, in NDC coordinates, (B, ..., 1) tensor.
        mask: rendered object mask, values in [0, 1], (B, ..., 1) tensor.
        prev_stage: for multi-pass renderers (e.g. in NeRF),
            a reference to the output of the previous stage.
        normals: surface normals, for renderers that estimate them; (B, ..., 3) tensor.
        points: ray-termination points in the world coordinates, (B, ..., 3) tensor.
        aux: dict for implementation-specific renderer outputs.
    """

    features: torch.Tensor
    depths: torch.Tensor
    masks: torch.Tensor
    prev_stage: Optional[RendererOutput] = None
    normals: Optional[torch.Tensor] = None
    points: Optional[torch.Tensor] = None  # TODO: redundant with depths
    weights: Optional[torch.Tensor] = None
    aux: Dict[str, Any] = field(default_factory=lambda: {})


class ImplicitFunctionWrapper(torch.nn.Module):
    def __init__(self, fn: torch.nn.Module):
        super().__init__()
        self._fn = fn
        self.bound_args = {}

    def bind_args(self, **bound_args):
        self.bound_args = bound_args
        self._fn.on_bind_args()

    def unbind_args(self):
        self.bound_args = {}

    def forward(self, *args, **kwargs):
        return self._fn(*args, **{**kwargs, **self.bound_args})


class BaseRenderer(ABC, ReplaceableBase):
    """
    Base class for all Renderer implementations.
    """

    def requires_object_mask(self) -> bool:
        """
        Whether `forward` needs the object_mask.
        """
        return False

    @abstractmethod
    def forward(
        self,
        ray_bundle: ImplicitronRayBundle,
        implicit_functions: List[ImplicitFunctionWrapper],
        evaluation_mode: EvaluationMode = EvaluationMode.EVALUATION,
        **kwargs,
    ) -> RendererOutput:
        """
        Each Renderer should implement its own forward function
        that returns an instance of RendererOutput.

        Args:
            ray_bundle: An ImplicitronRayBundle object containing the following variables:
                origins: A tensor of shape (minibatch, ..., 3) denoting
                    the origins of the rendering rays.
                directions: A tensor of shape (minibatch, ..., 3)
                    containing the direction vectors of rendering rays.
                lengths: A tensor of shape
                    (minibatch, ..., num_points_per_ray)containing the
                    lengths at which the ray points are sampled.
                    The coordinates of the points on the rays are thus computed
                    as `origins + lengths * directions`.
                xys: A tensor of shape
                    (minibatch, ..., 2) containing the
                    xy locations of each ray's pixel in the NDC screen space.
                camera_ids: A tensor of shape (N, ) which indicates which camera
                    was used to sample the rays. `N` is the number of different
                    sampled cameras.
                camera_counts: A tensor of shape (N, ) which how many times the
                    coresponding camera in `camera_ids` was sampled.
                    `sum(camera_counts)==minibatch`
            implicit_functions: List of ImplicitFunctionWrappers which define the
                implicit function methods to be used. Most Renderers only allow
                a single implicit function. Currently, only the
                MultiPassEmissionAbsorptionRenderer allows specifying mulitple
                values in the list.
            evaluation_mode: one of EvaluationMode.TRAINING or
                EvaluationMode.EVALUATION which determines the settings used for
                rendering.
            **kwargs: In addition to the name args, custom keyword args can be specified.
                For example in the SignedDistanceFunctionRenderer, an object_mask is
                required which needs to be passed via the kwargs.

        Returns:
            instance of RendererOutput
        """
        pass


def compute_3d_diagonal_covariance_gaussian(
    rays_directions: torch.Tensor,
    rays_dir_variance: torch.Tensor,
    radii_variance: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Transform the variances (rays_dir_variance, radii_variance) of the gaussians from
    the coordinate frame of the conical frustum to 3D world coordinates.

    It follows the equation 16 of `MIP-NeRF <https://arxiv.org/abs/2103.13415>`_

    Args:
        rays_directions: A tensor of shape `(..., 3)`
        rays_dir_variance: A tensor of shape `(..., num_intervals)` representing
            the variance of the conical frustum  with respect to the rays direction.
        radii_variance: A tensor of shape `(..., num_intervals)` representing
            the variance of the conical frustum with respect to its radius.
        eps: a small number to prevent division by zero.

    Returns:
        A tensor of shape `(..., num_intervals, 3)` containing the diagonal
            of the covariance matrix.
    """
    d_outer_diag = torch.pow(rays_directions, 2)
    dir_mag_sq = torch.clamp(torch.sum(d_outer_diag, dim=-1, keepdim=True), min=eps)

    null_outer_diag = 1 - d_outer_diag / dir_mag_sq
    ray_dir_cov_diag = rays_dir_variance[..., None] * d_outer_diag[..., None, :]
    xy_cov_diag = radii_variance[..., None] * null_outer_diag[..., None, :]
    return ray_dir_cov_diag + xy_cov_diag


def approximate_conical_frustum_as_gaussians(
    bins: torch.Tensor, radii: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Approximates a conical frustum as two Gaussian distributions.

    The Gaussian distributions are characterized by
    three values:

    - rays_dir_mean: mean along the rays direction
        (defined as t in the parametric representation of a cone).
    - rays_dir_variance: the variance of the conical frustum  along the rays direction.
    - radii_variance: variance of the conical frustum with respect to its radius.


    The computation is stable and follows equation 7
    of `MIP-NeRF <https://arxiv.org/abs/2103.13415>`_.

    For more information on how the mean and variances are computed
    refers to the appendix of the paper.

    Args:
        bins: A tensor of shape `(..., num_points_per_ray + 1)`
            containing the bins at which the rays are sampled.
            `bin[..., t]` and `bin[..., t+1]` represent respectively
            the left and right coordinates of the interval.
        t0: A tensor of shape `(..., num_points_per_ray)`
            containing the left coordinates of the intervals
            on which the rays are sampled.
        t1: A tensor of shape `(..., num_points_per_ray)`
            containing the rights coordinates of the intervals
            on which the rays are sampled.
        radii: A tensor of shape `(..., 1)`
            base radii of the conical frustums.

    Returns:
        rays_dir_mean: A tensor of shape `(..., num_intervals)` representing
            the mean along the rays direction
            (t in the parametric represention of the cone)
        rays_dir_variance: A tensor of shape `(...,  num_intervals)` representing
            the variance of the conical frustum along the rays
            (t in the parametric represention of the cone).
        radii_variance: A tensor of shape `(..., num_intervals)` representing
            the variance of the conical frustum with respect to its radius.
    """
    t_mu = torch.lerp(bins[..., 1:], bins[..., :-1], 0.5)
    t_delta = torch.diff(bins, dim=-1) / 2

    t_mu_pow2 = torch.pow(t_mu, 2)
    t_delta_pow2 = torch.pow(t_delta, 2)
    t_delta_pow4 = torch.pow(t_delta, 4)

    den = 3 * t_mu_pow2 + t_delta_pow2

    # mean along the rays direction
    rays_dir_mean = t_mu + 2 * t_mu * t_delta_pow2 / den

    # Variance of the conical frustum with along the rays directions
    rays_dir_variance = t_delta_pow2 / 3 - (4 / 15) * (
        t_delta_pow4 * (12 * t_mu_pow2 - t_delta_pow2) / torch.pow(den, 2)
    )

    # Variance of the conical frustum with respect to its radius
    radii_variance = torch.pow(radii, 2) * (
        t_mu_pow2 / 4 + (5 / 12) * t_delta_pow2 - 4 / 15 * (t_delta_pow4) / den
    )
    return rays_dir_mean, rays_dir_variance, radii_variance


def conical_frustum_to_gaussian(
    ray_bundle: ImplicitronRayBundle,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Approximate a conical frustum following a ray bundle as a Gaussian.

    Args:
        ray_bundle: A `RayBundle` or `HeterogeneousRayBundle` object with fields:
            origins: A tensor of shape `(..., 3)`
            directions: A tensor of shape `(..., 3)`
            lengths: A tensor of shape `(..., num_points_per_ray)`
            bins: A tensor of shape `(..., num_points_per_ray + 1)`
                containing the bins at which the rays are sampled. .
            pixel_radii_2d: A tensor of shape `(..., 1)`
                base radii of the conical frustums.

    Returns:
        means: A tensor of shape `(..., num_points_per_ray - 1, 3)`
            representing the means of the Gaussians
            approximating the conical frustums.
        diag_covariances: A tensor of shape `(...,num_points_per_ray -1, 3)`
            representing the diagonal covariance matrices of our Gaussians.
    """

    if ray_bundle.pixel_radii_2d is None or ray_bundle.bins is None:
        raise ValueError(
            "RayBundle pixel_radii_2d or bins have not been provided."
            " Look at pytorch3d.renderer.implicit.renderer.ray_sampler::"
            "AbstractMaskRaySampler to see how to compute them. Have you forgot to set"
            "`cast_ray_bundle_as_cone` to True?"
        )

    (
        rays_dir_mean,
        rays_dir_variance,
        radii_variance,
    ) = approximate_conical_frustum_as_gaussians(
        ray_bundle.bins,
        ray_bundle.pixel_radii_2d,
    )
    means = ray_bundle_variables_to_ray_points(
        ray_bundle.origins, ray_bundle.directions, rays_dir_mean
    )
    diag_covariances = compute_3d_diagonal_covariance_gaussian(
        ray_bundle.directions, rays_dir_variance, radii_variance
    )
    return means, diag_covariances
