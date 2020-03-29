# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.


"""
This module implements utility functions for sampling points from
batches of meshes.
"""
import sys
from typing import Tuple, Union

import torch
from pytorch3d.ops.mesh_face_areas_normals import mesh_face_areas_normals
from pytorch3d.ops.packed_to_padded import packed_to_padded


def sample_points_from_meshes(
    meshes, num_samples: int = 10000, return_normals: bool = False
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Convert a batch of meshes to a pointcloud by uniformly sampling points on
    the surface of the mesh with probability proportional to the face area.

    Args:
        meshes: A Meshes object with a batch of N meshes.
        num_samples: Integer giving the number of point samples per mesh.
        return_normals: If True, return normals for the sampled points.
        eps: (float) used to clamp the norm of the normals to avoid dividing by 0.

    Returns:
        2-element tuple containing

        - **samples**: FloatTensor of shape (N, num_samples, 3) giving the
          coordinates of sampled points for each mesh in the batch. For empty
          meshes the corresponding row in the samples array will be filled with 0.
        - **normals**: FloatTensor of shape (N, num_samples, 3) giving a normal vector
          to each sampled point. Only returned if return_normals is True.
          For empty meshes the corresponding row in the normals array will
          be filled with 0.
    """
    if meshes.isempty():
        raise ValueError("Meshes are empty.")

    verts = meshes.verts_packed()
    if not torch.isfinite(verts).all():
        raise ValueError("Meshes contain nan or inf.")
    faces = meshes.faces_packed()
    mesh_to_face = meshes.mesh_to_faces_packed_first_idx()
    num_meshes = len(meshes)
    num_valid_meshes = torch.sum(meshes.valid)  # Non empty meshes.

    # Intialize samples tensor with fill value 0 for empty meshes.
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
    face_verts = verts[faces.long()]
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
        # Intialize normals tensor with fill value 0 for empty meshes.
        # Normals for the sampled points are face normals computed from
        # the vertices of the face in which the sampled point lies.
        normals = torch.zeros((num_meshes, num_samples, 3), device=meshes.device)
        vert_normals = (v1 - v0).cross(v2 - v1, dim=1)
        vert_normals = vert_normals / vert_normals.norm(dim=1, p=2, keepdim=True).clamp(
            min=sys.float_info.epsilon
        )
        vert_normals = vert_normals[sample_face_idxs]
        normals[meshes.valid] = vert_normals

        return samples, normals
    else:
        return samples


def _rand_barycentric_coords(
    size1, size2, dtype, device
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
