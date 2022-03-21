# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Callable, Optional

import torch
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
            math.prod(spatial_size),
            pts_per_ray,
            0,
            dtype=xyz_world.dtype,
            device=xyz_world.device,
        )
    else:
        embeds = xyz_embedding_function(ray_points_for_embed).reshape(
            bs,
            1,
            math.prod(spatial_size),
            pts_per_ray,
            -1,
        )  # flatten spatial, add n_src dim

    if fun_viewpool is not None:
        # viewpooling
        embeds_viewpooled = fun_viewpool(xyz_world.reshape(bs, -1, 3))
        embed_shape = (
            bs,
            embeds_viewpooled.shape[1],
            math.prod(spatial_size),
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
