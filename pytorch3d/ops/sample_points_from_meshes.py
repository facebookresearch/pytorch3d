# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


"""
This module implements utility functions for sampling points from
batches of meshes.
"""
import sys
from typing import Tuple, Union

import torch
from pytorch3d.ops.mesh_face_areas_normals import mesh_face_areas_normals
from pytorch3d.ops.packed_to_padded import packed_to_padded
from pytorch3d.renderer.mesh.rasterizer import Fragments as MeshFragments


def sample_points_from_meshes(
    meshes,
    num_samples: int = 10000,
    return_normals: bool = False,
    return_textures: bool = False,
) -> Union[
    torch.Tensor,
    Tuple[torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]:
    """
    Convert a batch of meshes to a batch of pointclouds by uniformly sampling
    points on the surface of the mesh with probability proportional to the
    face area.

    Args:
        meshes: A Meshes object with a batch of N meshes.
        num_samples: Integer giving the number of point samples per mesh.
        return_normals: If True, return normals for the sampled points.
        return_textures: If True, return textures for the sampled points.

    Returns:
        3-element tuple containing

        - **samples**: FloatTensor of shape (N, num_samples, 3) giving the
          coordinates of sampled points for each mesh in the batch. For empty
          meshes the corresponding row in the samples array will be filled with 0.
        - **normals**: FloatTensor of shape (N, num_samples, 3) giving a normal vector
          to each sampled point. Only returned if return_normals is True.
          For empty meshes the corresponding row in the normals array will
          be filled with 0.
        - **textures**: FloatTensor of shape (N, num_samples, C) giving a C-dimensional
          texture vector to each sampled point. Only returned if return_textures is True.
          For empty meshes the corresponding row in the textures array will
          be filled with 0.

        Note that in a future releases, we will replace the 3-element tuple output
        with a `Pointclouds` datastructure, as follows

        .. code-block:: python

            Pointclouds(samples, normals=normals, features=textures)
    """
    if meshes.isempty():
        raise ValueError("Meshes are empty.")

    verts = meshes.verts_packed()
    if not torch.isfinite(verts).all():
        raise ValueError("Meshes contain nan or inf.")

    if return_textures and meshes.textures is None:
        raise ValueError("Meshes do not contain textures.")

    faces = meshes.faces_packed()
    mesh_to_face = meshes.mesh_to_faces_packed_first_idx()
    num_meshes = len(meshes)
    num_valid_meshes = torch.sum(meshes.valid)  # Non empty meshes.

    # Initialize samples tensor with fill value 0 for empty meshes.
    samples = torch.zeros((num_meshes, num_samples, 3), device=meshes.device)

    # Only compute samples for non empty meshes
    with torch.no_grad():
        areas, _ = mesh_face_areas_normals(verts, faces)  # Face areas can be zero.
        max_faces = meshes.num_faces_per_mesh().max().item()
        areas_padded = packed_to_padded(
            areas, mesh_to_face[meshes.valid], max_faces
        )  # (N, F)

        # TODO (gkioxari) Confirm multinomial bug is not present with real data.
        sample_face_idxs = areas_padded.multinomial(
            num_samples, replacement=True
        )  # (N, num_samples)
        sample_face_idxs += mesh_to_face[meshes.valid].view(num_valid_meshes, 1)

    # Get the vertex coordinates of the sampled faces.
    face_verts = verts[faces]
    v0, v1, v2 = face_verts[:, 0], face_verts[:, 1], face_verts[:, 2]

    # Randomly generate barycentric coords.
    w0, w1, w2 = _rand_barycentric_coords(
        num_valid_meshes, num_samples, verts.dtype, verts.device
    )

    # Use the barycentric coords to get a point on each sampled face.
    a = v0[sample_face_idxs]  # (N, num_samples, 3)
    b = v1[sample_face_idxs]
    c = v2[sample_face_idxs]
    samples[meshes.valid] = w0[:, :, None] * a + w1[:, :, None] * b + w2[:, :, None] * c

    if return_normals:
        # Initialize normals tensor with fill value 0 for empty meshes.
        # Normals for the sampled points are face normals computed from
        # the vertices of the face in which the sampled point lies.
        normals = torch.zeros((num_meshes, num_samples, 3), device=meshes.device)
        vert_normals = (v1 - v0).cross(v2 - v1, dim=1)
        vert_normals = vert_normals / vert_normals.norm(dim=1, p=2, keepdim=True).clamp(
            min=sys.float_info.epsilon
        )
        vert_normals = vert_normals[sample_face_idxs]
        normals[meshes.valid] = vert_normals

    if return_textures:
        # fragment data are of shape NxHxWxK. Here H=S, W=1 & K=1.
        pix_to_face = sample_face_idxs.view(len(meshes), num_samples, 1, 1)  # NxSx1x1
        bary = torch.stack((w0, w1, w2), dim=2).unsqueeze(2).unsqueeze(2)  # NxSx1x1x3
        # zbuf and dists are not used in `sample_textures` so we initialize them with dummy
        dummy = torch.zeros(
            (len(meshes), num_samples, 1, 1), device=meshes.device, dtype=torch.float32
        )  # NxSx1x1
        fragments = MeshFragments(
            pix_to_face=pix_to_face, zbuf=dummy, bary_coords=bary, dists=dummy
        )
        textures = meshes.sample_textures(fragments)  # NxSx1x1xC
        textures = textures[:, :, 0, 0, :]  # NxSxC

    # return
    # TODO(gkioxari) consider returning a Pointclouds instance [breaking]
    if return_normals and return_textures:
        # pyre-fixme[61]: `normals` may not be initialized here.
        # pyre-fixme[61]: `textures` may not be initialized here.
        return samples, normals, textures
    if return_normals:  # return_textures is False
        # pyre-fixme[61]: `normals` may not be initialized here.
        return samples, normals
    if return_textures:  # return_normals is False
        # pyre-fixme[61]: `textures` may not be initialized here.
        return samples, textures
    return samples


def _rand_barycentric_coords(
    size1, size2, dtype: torch.dtype, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Helper function to generate random barycentric coordinates which are uniformly
    distributed over a triangle.

    Args:
        size1, size2: The number of coordinates generated will be size1*size2.
                      Output tensors will each be of shape (size1, size2).
        dtype: Datatype to generate.
        device: A torch.device object on which the outputs will be allocated.

    Returns:
        w0, w1, w2: Tensors of shape (size1, size2) giving random barycentric
            coordinates
    """
    uv = torch.rand(2, size1, size2, dtype=dtype, device=device)
    u, v = uv[0], uv[1]
    u_sqrt = u.sqrt()
    w0 = 1.0 - u_sqrt
    w1 = u_sqrt * (1.0 - v)
    w2 = u_sqrt * v
    return w0, w1, w2
