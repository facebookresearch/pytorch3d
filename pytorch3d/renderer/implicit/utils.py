# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
from typing import NamedTuple, Optional, Union

import torch


class RayBundle(NamedTuple):
    """
    Parametrizes points along projection rays by storing:

        origins: A tensor of shape `(..., 3)` denoting the
            origins of the sampling rays in world coords.
        directions: A tensor of shape `(..., 3)` containing the direction
            vectors of sampling rays in world coords. They don't have to be normalized;
            they define unit vectors in the respective 1D coordinate systems; see
            documentation for :func:`ray_bundle_to_ray_points` for the conversion formula.
        lengths: A tensor of shape `(..., num_points_per_ray)`
            containing the lengths at which the rays are sampled.
        xys: A tensor of shape `(..., 2)`, the xy-locations (`xys`) of the ray pixels
    """

    origins: torch.Tensor
    directions: torch.Tensor
    lengths: torch.Tensor
    xys: torch.Tensor


@dataclasses.dataclass
class HeterogeneousRayBundle:
    """
    Members:
        origins: A tensor of shape `(..., 3)` denoting the
            origins of the sampling rays in world coords.
        directions: A tensor of shape `(..., 3)` containing the direction
            vectors of sampling rays in world coords. They don't have to be normalized;
            they define unit vectors in the respective 1D coordinate systems; see
            documentation for :func:`ray_bundle_to_ray_points` for the conversion formula.
        lengths: A tensor of shape `(..., num_points_per_ray)`
            containing the lengths at which the rays are sampled.
        xys: A tensor of shape `(..., 2)`, the xy-locations (`xys`) of the ray pixels
        camera_ids: A tensor of shape (N, ) which indicates which camera
            was used to sample the rays. `N` is the number of unique sampled cameras.
        camera_counts: A tensor of shape (N, ) which how many times the
            coresponding camera in `camera_ids` was sampled.
            `sum(camera_counts)==total_number_of_rays`

    If we sample cameras of ids [0, 3, 5, 3, 1, 0, 0] that would be
    stored as camera_ids=[1, 3, 5, 0] and camera_counts=[1, 2, 1, 3]. `camera_ids` is a
    set like object with no particular ordering of elements. ith element of
    `camera_ids` coresponds to the ith element of `camera_counts`.
    """

    origins: torch.Tensor
    directions: torch.Tensor
    lengths: torch.Tensor
    xys: torch.Tensor
    camera_ids: Optional[torch.LongTensor] = None
    camera_counts: Optional[torch.LongTensor] = None


def ray_bundle_to_ray_points(
    ray_bundle: Union[RayBundle, HeterogeneousRayBundle]
) -> torch.Tensor:
    """
    Converts rays parametrized with a `ray_bundle` (an instance of the `RayBundle`
    named tuple or HeterogeneousRayBundle dataclass) to 3D points by
    extending each ray according to the corresponding length.

    E.g. for 2 dimensional tensors `ray_bundle.origins`, `ray_bundle.directions`
        and `ray_bundle.lengths`, the ray point at position `[i, j]` is::

            ray_bundle.points[i, j, :] = (
                ray_bundle.origins[i, :]
                + ray_bundle.directions[i, :] * ray_bundle.lengths[i, j]
            )

    Note that both the directions and magnitudes of the vectors in
    `ray_bundle.directions` matter.

    Args:
        ray_bundle: A `RayBundle` or `HeterogeneousRayBundle` object with fields:
            origins: A tensor of shape `(..., 3)`
            directions: A tensor of shape `(..., 3)`
            lengths: A tensor of shape `(..., num_points_per_ray)`

    Returns:
        rays_points: A tensor of shape `(..., num_points_per_ray, 3)`
            containing the points sampled along each ray.
    """
    return ray_bundle_variables_to_ray_points(
        ray_bundle.origins, ray_bundle.directions, ray_bundle.lengths
    )


def ray_bundle_variables_to_ray_points(
    rays_origins: torch.Tensor,
    rays_directions: torch.Tensor,
    rays_lengths: torch.Tensor,
) -> torch.Tensor:
    """
    Converts rays parametrized with origins and directions
    to 3D points by extending each ray according to the corresponding
    ray length:

    E.g. for 2 dimensional input tensors `rays_origins`, `rays_directions`
    and `rays_lengths`, the ray point at position `[i, j]` is::

            rays_points[i, j, :] = (
                rays_origins[i, :]
                + rays_directions[i, :] * rays_lengths[i, j]
            )

    Note that both the directions and magnitudes of the vectors in
    `rays_directions` matter.

    Args:
        rays_origins: A tensor of shape `(..., 3)`
        rays_directions: A tensor of shape `(..., 3)`
        rays_lengths: A tensor of shape `(..., num_points_per_ray)`

    Returns:
        rays_points: A tensor of shape `(..., num_points_per_ray, 3)`
            containing the points sampled along each ray.
    """
    rays_points = (
        rays_origins[..., None, :]
        + rays_lengths[..., :, None] * rays_directions[..., None, :]
    )
    return rays_points


def _validate_ray_bundle_variables(
    rays_origins: torch.Tensor,
    rays_directions: torch.Tensor,
    rays_lengths: torch.Tensor,
) -> None:
    """
    Validate the shapes of RayBundle variables
    `rays_origins`, `rays_directions`, and `rays_lengths`.
    """
    ndim = rays_origins.ndim
    if any(r.ndim != ndim for r in (rays_directions, rays_lengths)):
        raise ValueError(
            "rays_origins, rays_directions and rays_lengths"
            + " have to have the same number of dimensions."
        )

    if ndim <= 2:
        raise ValueError(
            "rays_origins, rays_directions and rays_lengths"
            + " have to have at least 3 dimensions."
        )

    spatial_size = rays_origins.shape[:-1]
    if any(spatial_size != r.shape[:-1] for r in (rays_directions, rays_lengths)):
        raise ValueError(
            "The shapes of rays_origins, rays_directions and rays_lengths"
            + " may differ only in the last dimension."
        )

    if any(r.shape[-1] != 3 for r in (rays_origins, rays_directions)):
        raise ValueError(
            "The size of the last dimension of rays_origins/rays_directions"
            + "has to be 3."
        )
