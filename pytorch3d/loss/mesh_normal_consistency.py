# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import torch
from pytorch3d import _C


def mesh_normal_consistency(meshes):
    r"""
    Computes the normal consistency of each mesh in meshes.
    We compute the normal consistency for each pair of neighboring faces.
    If e = (v0, v1) is the connecting edge of two neighboring faces f0 and f1,
    then the normal consistency between f0 and f1

    .. code-block:: python

                    a
                    /\
                   /  \
                  / f0 \
                 /      \
            v0  /____e___\ v1
                \        /
                 \      /
                  \ f1 /
                   \  /
                    \/
                    b

    The normal consistency is

    .. code-block:: python

        nc(f0, f1) = 1 - cos(n0, n1)

        where cos(n0, n1) = n0^n1 / ||n0|| / ||n1|| is the cosine of the angle
        between the normals n0 and n1, and

        n0 = (v1 - v0) x (a - v0)
        n1 = - (v1 - v0) x (b - v0) = (b - v0) x (v1 - v0)

    This means that if nc(f0, f1) = 0 then n0 and n1 point to the same
    direction, while if nc(f0, f1) = 2 then n0 and n1 point opposite direction.

    .. note::
        For well-constructed meshes the assumption that only two faces share an
        edge is true. This assumption could make the implementation easier and faster.
        This implementation does not follow this assumption. All the faces sharing e,
        which can be any in number, are discovered.

    Args:
        meshes: Meshes object with a batch of meshes.

    Returns:
        loss: Average normal consistency across the batch.
        Returns 0 if meshes contains no meshes or all empty meshes.
    """
    if meshes.isempty():
        return torch.tensor(
            [0.0], dtype=torch.float32, device=meshes.device, requires_grad=True
        )

    N = len(meshes)
    verts_packed = meshes.verts_packed()  # (sum(V_n), 3)
    faces_packed = meshes.faces_packed()  # (sum(F_n), 3)
    edges_packed = meshes.edges_packed()  # (sum(E_n), 2)
    verts_packed_to_mesh_idx = meshes.verts_packed_to_mesh_idx()  # (sum(V_n),)
    face_to_edge = meshes.faces_packed_to_edges_packed()  # (sum(F_n), 3)
    E = edges_packed.shape[0]  # sum(E_n)
    F = faces_packed.shape[0]  # sum(F_n)

    # We don't want gradients for the following operation. The goal is to
    # find for each edge e all the vertices associated with e. In the example
    # above, the vertices associated with e are (a, b), i.e. the points connected
    # on faces to e.
    with torch.no_grad():
        edge_idx = face_to_edge.reshape(F * 3)  # (3 * F,) indexes into edges
        vert_idx = (
            faces_packed.view(1, F, 3).expand(3, F, 3).transpose(0, 1).reshape(3 * F, 3)
        )
        edge_idx, edge_sort_idx = edge_idx.sort()
        vert_idx = vert_idx[edge_sort_idx]

        # In well constructed meshes each edge is shared by precisely 2 faces
        # However, in many meshes, this assumption is not always satisfied.
        # We want to find all faces that share an edge, a number which can
        # vary and which depends on the topology.
        # In particular, we find the vertices not on the edge on the shared faces.
        # In the example above, we want to associate edge e with vertices a and b.
        # This operation is done more efficiently in cpu with lists.
        # TODO(gkioxari) find a better way to do this.

        # edge_idx represents the index of the edge for each vertex. We can count
        # the number of vertices which are associated with each edge.
        # There can be a different number for each edge.
        edge_num = edge_idx.bincount(minlength=E)

        # This calculates all pairs of vertices which are opposite to the same edge.
        vert_edge_pair_idx = _C.mesh_normal_consistency_find_verts(edge_num.cpu()).to(
            edge_num.device
        )

    if vert_edge_pair_idx.shape[0] == 0:
        return torch.tensor(
            [0.0], dtype=torch.float32, device=meshes.device, requires_grad=True
        )

    v0_idx = edges_packed[edge_idx, 0]
    v0 = verts_packed[v0_idx]
    v1_idx = edges_packed[edge_idx, 1]
    v1 = verts_packed[v1_idx]

    # two of the following cross products are zeros as they are cross product
    # with either (v1-v0)x(v1-v0) or (v1-v0)x(v0-v0)
    n_temp0 = (v1 - v0).cross(verts_packed[vert_idx[:, 0]] - v0, dim=1)
    n_temp1 = (v1 - v0).cross(verts_packed[vert_idx[:, 1]] - v0, dim=1)
    n_temp2 = (v1 - v0).cross(verts_packed[vert_idx[:, 2]] - v0, dim=1)
    n = n_temp0 + n_temp1 + n_temp2
    n0 = n[vert_edge_pair_idx[:, 0]]
    n1 = -n[vert_edge_pair_idx[:, 1]]
    loss = 1 - torch.cosine_similarity(n0, n1, dim=1)

    verts_packed_to_mesh_idx = verts_packed_to_mesh_idx[vert_idx[:, 0]]
    verts_packed_to_mesh_idx = verts_packed_to_mesh_idx[vert_edge_pair_idx[:, 0]]
    num_normals = verts_packed_to_mesh_idx.bincount(minlength=N)
    weights = 1.0 / num_normals[verts_packed_to_mesh_idx].float()

    loss = loss * weights
    return loss.sum() / N
