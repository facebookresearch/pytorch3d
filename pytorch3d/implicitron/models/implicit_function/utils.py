# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Optional

import torch

import torch.nn.functional as F
from pytorch3d.common.compat import prod
from pytorch3d.implicitron.models.renderer.base import ImplicitronRayBundle
from pytorch3d.renderer import ray_bundle_to_ray_points
from pytorch3d.renderer.cameras import CamerasBase


def broadcast_global_code(embeds: torch.Tensor, global_code: torch.Tensor):
    """
    Expands the `global_code` of shape (minibatch, dim)
    so that it can be appended to `embeds` of shape (minibatch, ..., dim2),
    and appends to the last dimension of `embeds`.
    """
    bs = embeds.shape[0]
    global_code_broadcast = global_code.view(bs, *([1] * (embeds.ndim - 2)), -1).expand(
        *embeds.shape[:-1],
        global_code.shape[-1],
    )
    return torch.cat([embeds, global_code_broadcast], dim=-1)


def create_embeddings_for_implicit_function(
    xyz_world: torch.Tensor,
    xyz_in_camera_coords: bool,
    global_code: Optional[torch.Tensor],
    camera: Optional[CamerasBase],
    fun_viewpool: Optional[Callable],
    xyz_embedding_function: Optional[Callable],
    diag_cov: Optional[torch.Tensor] = None,
) -> torch.Tensor:

    bs, *spatial_size, pts_per_ray, _ = xyz_world.shape

    if xyz_in_camera_coords:
        if camera is None:
            raise ValueError("Camera must be given if xyz_in_camera_coords")

        ray_points_for_embed = (
            camera.get_world_to_view_transform()
            .transform_points(xyz_world.view(bs, -1, 3))
            .view(xyz_world.shape)
        )
    else:
        ray_points_for_embed = xyz_world

    if xyz_embedding_function is None:
        embeds = torch.empty(
            bs,
            1,
            prod(spatial_size),
            pts_per_ray,
            0,
        )
    else:

        embeds = xyz_embedding_function(ray_points_for_embed, diag_cov=diag_cov)
        embeds = embeds.reshape(
            bs,
            1,
            prod(spatial_size),
            pts_per_ray,
            -1,
        )  # flatten spatial, add n_src dim

    if fun_viewpool is not None:
        # viewpooling
        embeds_viewpooled = fun_viewpool(xyz_world.reshape(bs, -1, 3))
        embed_shape = (
            bs,
            embeds_viewpooled.shape[1],
            prod(spatial_size),
            pts_per_ray,
            -1,
        )
        embeds_viewpooled = embeds_viewpooled.reshape(*embed_shape)
        if embeds is not None:
            embeds = torch.cat([embeds.expand(*embed_shape), embeds_viewpooled], dim=-1)
        else:
            embeds = embeds_viewpooled

    if global_code is not None:
        # append the broadcasted global code to embeds
        embeds = broadcast_global_code(embeds, global_code)

    return embeds


def interpolate_line(
    points: torch.Tensor,
    source: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    """
    Linearly interpolates values of source grids. The first dimension of points represents
    number of points and the second coordinate, for example ([[x0], [x1], ...]). The first
    dimension of argument source represents feature and ones after that the spatial
    dimension.

    Arguments:
        points: shape (n_grids, n_points, 1),
        source: tensor of shape (n_grids, features, width),
    Returns:
        interpolated tensor of shape (n_grids, n_points, features)
    """
    # To enable sampling of the source using the torch.functional.grid_sample
    # points need to have 2 coordinates.
    expansion = points.new_zeros(points.shape)
    points = torch.cat((points, expansion), dim=-1)

    source = source[:, :, None, :]
    points = points[:, :, None, :]

    out = F.grid_sample(
        grid=points,
        input=source,
        **kwargs,
    )
    return out[:, :, :, 0].permute(0, 2, 1)


def interpolate_plane(
    points: torch.Tensor,
    source: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    """
    Bilinearly interpolates values of source grids. The first dimension of points represents
    number of points and the second coordinates, for example ([[x0, y0], [x1, y1], ...]).
    The first dimension of argument source represents feature and ones after that the
    spatial dimension.

    Arguments:
        points: shape (n_grids, n_points, 2),
        source: tensor of shape (n_grids, features, width, height),
    Returns:
        interpolated tensor of shape (n_grids, n_points, features)
    """
    # permuting because torch.nn.functional.grid_sample works with
    # (features, height, width) and not
    # (features, width, height)
    source = source.permute(0, 1, 3, 2)
    points = points[:, :, None, :]

    out = F.grid_sample(
        grid=points,
        input=source,
        **kwargs,
    )
    return out[:, :, :, 0].permute(0, 2, 1)


def interpolate_volume(
    points: torch.Tensor, source: torch.Tensor, **kwargs
) -> torch.Tensor:
    """
    Interpolates values of source grids. The first dimension of points represents
    number of points and the second coordinates, for example
    [[x0, y0, z0], [x1, y1, z1], ...]. The first dimension of a source represents features
    and ones after that the spatial dimension.

    Arguments:
        points: shape (n_grids, n_points, 3),
        source: tensor of shape (n_grids, features, width, height, depth),
    Returns:
        interpolated tensor of shape (n_grids, n_points, features)
    """
    if "mode" in kwargs and kwargs["mode"] == "trilinear":
        kwargs = kwargs.copy()
        kwargs["mode"] = "bilinear"
    # permuting because torch.nn.functional.grid_sample works with
    # (features, depth, height, width) and not (features, width, height, depth)
    source = source.permute(0, 1, 4, 3, 2)
    grid = points[:, :, None, None, :]

    out = F.grid_sample(
        grid=grid,
        input=source,
        **kwargs,
    )
    return out[:, :, :, 0, 0].permute(0, 2, 1)


def get_rays_points_world(
    ray_bundle: Optional[ImplicitronRayBundle] = None,
    rays_points_world: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Converts the ray_bundle to rays_points_world if rays_points_world is not defined
    and raises error if both are defined.

    Args:
        ray_bundle: An ImplicitronRayBundle object or None
        rays_points_world: A torch.Tensor representing ray points converted to
            world coordinates
    Returns:
        A torch.Tensor representing ray points converted to world coordinates
            of shape [minibatch x ... x pts_per_ray x 3].
    """
    if rays_points_world is not None and ray_bundle is not None:
        raise ValueError(
            "Cannot define both rays_points_world and ray_bundle,"
            + " one has to be None."
        )
    if rays_points_world is not None:
        return rays_points_world
    if ray_bundle is not None:
        # pyre-ignore[6]
        return ray_bundle_to_ray_points(ray_bundle)
    raise ValueError("ray_bundle and rays_points_world cannot both be None")
