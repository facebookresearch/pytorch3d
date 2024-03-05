# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe


from typing import Optional, Tuple

import torch
from pytorch3d.common.compat import meshgrid_ij
from pytorch3d.renderer.mesh.textures import TexturesAtlas
from pytorch3d.structures.meshes import Meshes


def checkerboard(
    radius: int = 4,
    color1: Tuple[float, ...] = (0.0, 0.0, 0.0),
    color2: Tuple[float, ...] = (1.0, 1.0, 1.0),
    device: Optional[torch.types._device] = None,
) -> Meshes:
    """
    Returns a mesh of squares in the xy-plane where each unit is one of the two given
    colors and adjacent squares have opposite colors.
    Args:
        radius: how many squares in each direction from the origin
        color1: background color
        color2: foreground color (must have the same number of channels as color1)
    Returns:
        new Meshes object containing one mesh.
    """

    if device is None:
        device = torch.device("cpu")
    if radius < 1:
        raise ValueError("radius must be > 0")

    num_verts_per_row = 2 * radius + 1

    # construct 2D grid of 3D vertices
    x = torch.arange(-radius, radius + 1, device=device)
    grid_y, grid_x = meshgrid_ij(x, x)
    verts = torch.stack(
        [grid_x, grid_y, torch.zeros((2 * radius + 1, 2 * radius + 1))], dim=-1
    )
    verts = verts.view(1, -1, 3)

    top_triangle_idx = torch.arange(0, num_verts_per_row * (num_verts_per_row - 1))
    top_triangle_idx = torch.stack(
        [
            top_triangle_idx,
            top_triangle_idx + 1,
            top_triangle_idx + num_verts_per_row + 1,
        ],
        dim=-1,
    )

    bottom_triangle_idx = top_triangle_idx[:, [0, 2, 1]] + torch.tensor(
        [0, 0, num_verts_per_row - 1]
    )

    faces = torch.zeros(
        (1, len(top_triangle_idx) + len(bottom_triangle_idx), 3),
        dtype=torch.long,
        device=device,
    )
    faces[0, ::2] = top_triangle_idx
    faces[0, 1::2] = bottom_triangle_idx

    # construct range of indices that excludes the boundary to avoid wrong triangles
    indexing_range = torch.arange(0, 2 * num_verts_per_row * num_verts_per_row).view(
        num_verts_per_row, num_verts_per_row, 2
    )
    indexing_range = indexing_range[:-1, :-1]  # removes boundaries from list of indices
    indexing_range = indexing_range.reshape(
        2 * (num_verts_per_row - 1) * (num_verts_per_row - 1)
    )

    faces = faces[:, indexing_range]

    # adding color
    colors = torch.tensor(color1).repeat(2 * num_verts_per_row * num_verts_per_row, 1)
    colors[2::4] = torch.tensor(color2)
    colors[3::4] = torch.tensor(color2)
    colors = colors[None, indexing_range, None, None]

    texture_atlas = TexturesAtlas(colors)

    return Meshes(verts=verts, faces=faces, textures=texture_atlas)
