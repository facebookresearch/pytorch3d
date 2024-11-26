# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe


from typing import Optional

import torch
import torch.nn.functional as F

from pytorch3d.common.compat import meshgrid_ij

from pytorch3d.structures import Meshes


def unravel_index(idx, dims) -> torch.Tensor:
    r"""
    Equivalent to np.unravel_index
    Args:
      idx: A LongTensor whose elements are indices into the
          flattened version of an array of dimensions dims.
      dims: The shape of the array to be indexed.
    Implemented only for dims=(N, H, W, D)
    """
    if len(dims) != 4:
        raise ValueError("Expects a 4-element list.")
    N, H, W, D = dims
    n = idx // (H * W * D)
    h = (idx - n * H * W * D) // (W * D)
    w = (idx - n * H * W * D - h * W * D) // D
    d = idx - n * H * W * D - h * W * D - w * D
    return torch.stack((n, h, w, d), dim=1)


def ravel_index(idx, dims) -> torch.Tensor:
    """
    Computes the linear index in an array of shape dims.
    It performs the reverse functionality of unravel_index
    Args:
      idx: A LongTensor of shape (N, 3). Each row corresponds to indices into an
          array of dimensions dims.
      dims: The shape of the array to be indexed.
    Implemented only for dims=(H, W, D)
    """
    if len(dims) != 3:
        raise ValueError("Expects a 3-element list")
    if idx.shape[1] != 3:
        raise ValueError("Expects an index tensor of shape Nx3")
    H, W, D = dims
    linind = idx[:, 0] * W * D + idx[:, 1] * D + idx[:, 2]
    return linind


@torch.no_grad()
def cubify(
    voxels: torch.Tensor,
    thresh: float,
    *,
    feats: Optional[torch.Tensor] = None,
    device=None,
    align: str = "topleft",
) -> Meshes:
    r"""
    Converts a voxel to a mesh by replacing each occupied voxel with a cube
    consisting of 12 faces and 8 vertices. Shared vertices are merged, and
    internal faces are removed.
    Args:
      voxels: A FloatTensor of shape (N, D, H, W) containing occupancy probabilities.
      thresh: A scalar threshold. If a voxel occupancy is larger than
          thresh, the voxel is considered occupied.
      feats: A FloatTensor of shape (N, K, D, H, W) containing the color information
          of each voxel. K is the number of channels. This is supported only when
          align == "center"
      device: The device of the output meshes
      align: Defines the alignment of the mesh vertices and the grid locations.
          Has to be one of {"topleft", "corner", "center"}. See below for explanation.
          Default is "topleft".
    Returns:
      meshes: A Meshes object of the corresponding meshes.


    The alignment between the vertices of the cubified mesh and the voxel locations (or pixels)
    is defined by the choice of `align`. We support three modes, as shown below for a 2x2 grid:

                X---X----         X-------X        ---------
                |   |   |         |   |   |        | X | X |
                X---X----         ---------        ---------
                |   |   |         |   |   |        | X | X |
                ---------         X-------X        ---------

                 topleft           corner            center

    In the figure, X denote the grid locations and the squares represent the added cuboids.
    When `align="topleft"`, then the top left corner of each cuboid corresponds to the
    pixel coordinate of the input grid.
    When `align="corner"`, then the corners of the output mesh span the whole grid.
    When `align="center"`, then the grid locations form the center of the cuboids.
    """

    if device is None:
        device = voxels.device

    if align not in ["topleft", "corner", "center"]:
        raise ValueError("Align mode must be one of (topleft, corner, center).")

    if len(voxels) == 0:
        return Meshes(verts=[], faces=[])

    N, D, H, W = voxels.size()
    # vertices corresponding to a unit cube: 8x3
    cube_verts = torch.tensor(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
        ],
        dtype=torch.int64,
        device=device,
    )

    # faces corresponding to a unit cube: 12x3
    cube_faces = torch.tensor(
        [
            [0, 1, 2],
            [1, 3, 2],  # left face: 0, 1
            [2, 3, 6],
            [3, 7, 6],  # bottom face: 2, 3
            [0, 2, 6],
            [0, 6, 4],  # front face: 4, 5
            [0, 5, 1],
            [0, 4, 5],  # up face: 6, 7
            [6, 7, 5],
            [6, 5, 4],  # right face: 8, 9
            [1, 7, 3],
            [1, 5, 7],  # back face: 10, 11
        ],
        dtype=torch.int64,
        device=device,
    )

    wx = torch.tensor([0.5, 0.5], device=device).view(1, 1, 1, 1, 2)
    wy = torch.tensor([0.5, 0.5], device=device).view(1, 1, 1, 2, 1)
    wz = torch.tensor([0.5, 0.5], device=device).view(1, 1, 2, 1, 1)

    voxelt = voxels.ge(thresh).float()
    # N x 1 x D x H x W
    voxelt = voxelt.view(N, 1, D, H, W)

    # N x 1 x (D-1) x (H-1) x (W-1)
    voxelt_x = F.conv3d(voxelt, wx).gt(0.5).float()
    voxelt_y = F.conv3d(voxelt, wy).gt(0.5).float()
    voxelt_z = F.conv3d(voxelt, wz).gt(0.5).float()

    # 12 x N x 1 x D x H x W
    faces_idx = torch.ones((cube_faces.size(0), N, 1, D, H, W), device=device)

    # add left face
    faces_idx[0, :, :, :, :, 1:] = 1 - voxelt_x
    faces_idx[1, :, :, :, :, 1:] = 1 - voxelt_x
    # add bottom face
    faces_idx[2, :, :, :, :-1, :] = 1 - voxelt_y
    faces_idx[3, :, :, :, :-1, :] = 1 - voxelt_y
    # add front face
    faces_idx[4, :, :, 1:, :, :] = 1 - voxelt_z
    faces_idx[5, :, :, 1:, :, :] = 1 - voxelt_z
    # add up face
    faces_idx[6, :, :, :, 1:, :] = 1 - voxelt_y
    faces_idx[7, :, :, :, 1:, :] = 1 - voxelt_y
    # add right face
    faces_idx[8, :, :, :, :, :-1] = 1 - voxelt_x
    faces_idx[9, :, :, :, :, :-1] = 1 - voxelt_x
    # add back face
    faces_idx[10, :, :, :-1, :, :] = 1 - voxelt_z
    faces_idx[11, :, :, :-1, :, :] = 1 - voxelt_z

    faces_idx *= voxelt

    # N x H x W x D x 12
    faces_idx = faces_idx.permute(1, 2, 4, 5, 3, 0).squeeze(1)
    # (NHWD) x 12
    faces_idx = faces_idx.contiguous()
    faces_idx = faces_idx.view(-1, cube_faces.size(0))

    # boolean to linear index
    # NF x 2
    linind = torch.nonzero(faces_idx, as_tuple=False)

    # NF x 4
    nyxz = unravel_index(linind[:, 0], (N, H, W, D))

    # NF x 3: faces
    faces = torch.index_select(cube_faces, 0, linind[:, 1])

    grid_faces = []
    for d in range(cube_faces.size(1)):
        # NF x 3
        xyz = torch.index_select(cube_verts, 0, faces[:, d])
        permute_idx = torch.tensor([1, 0, 2], device=device)
        yxz = torch.index_select(xyz, 1, permute_idx)
        yxz += nyxz[:, 1:]
        # NF x 1
        temp = ravel_index(yxz, (H + 1, W + 1, D + 1))
        grid_faces.append(temp)
    # NF x 3
    grid_faces = torch.stack(grid_faces, dim=1)

    y, x, z = meshgrid_ij(torch.arange(H + 1), torch.arange(W + 1), torch.arange(D + 1))
    y = y.to(device=device, dtype=torch.float32)
    x = x.to(device=device, dtype=torch.float32)
    z = z.to(device=device, dtype=torch.float32)

    if align == "center":
        x = x - 0.5
        y = y - 0.5
        z = z - 0.5

    margin = 0.0 if align == "corner" else 1.0
    y = y * 2.0 / (H - margin) - 1.0
    x = x * 2.0 / (W - margin) - 1.0
    z = z * 2.0 / (D - margin) - 1.0

    # ((H+1)(W+1)(D+1)) x 3
    grid_verts = torch.stack((x, y, z), dim=3).view(-1, 3)

    if len(nyxz) == 0:
        verts_list = [torch.tensor([], dtype=torch.float32, device=device)] * N
        faces_list = [torch.tensor([], dtype=torch.int64, device=device)] * N
        return Meshes(verts=verts_list, faces=faces_list)

    num_verts = grid_verts.size(0)
    grid_faces += nyxz[:, 0].view(-1, 1) * num_verts
    idleverts = torch.ones(num_verts * N, dtype=torch.uint8, device=device)

    indices = grid_faces.flatten()
    if device.type == "cpu":
        indices = torch.unique(indices)
    idleverts.scatter_(0, indices, 0)
    grid_faces -= nyxz[:, 0].view(-1, 1) * num_verts
    split_size = torch.bincount(nyxz[:, 0], minlength=N)
    faces_list = list(torch.split(grid_faces, split_size.tolist(), 0))

    idleverts = idleverts.view(N, num_verts)
    idlenum = idleverts.cumsum(1)

    verts_list = [
        grid_verts.index_select(0, (idleverts[n] == 0).nonzero(as_tuple=False)[:, 0])
        for n in range(N)
    ]

    textures_list = None
    if feats is not None and align == "center":
        # We return a TexturesAtlas containing one color for each face
        # N x K x D x H x W  -> N x H x W x D x K
        feats = feats.permute(0, 3, 4, 2, 1)

        # (NHWD) x K
        feats = feats.reshape(-1, feats.size(4))
        feats = torch.index_select(feats, 0, linind[:, 0])
        feats = feats.reshape(-1, 1, 1, feats.size(1))
        feats_list = list(torch.split(feats, split_size.tolist(), 0))
        from pytorch3d.renderer.mesh.textures import TexturesAtlas

        textures_list = TexturesAtlas(feats_list)

    faces_list = [nface - idlenum[n][nface] for n, nface in enumerate(faces_list)]
    return Meshes(verts=verts_list, faces=faces_list, textures=textures_list)
