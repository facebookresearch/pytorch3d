# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from pytorch3d.ops import norm_laplacian
from pytorch3d.structures import Meshes, utils as struct_utils


# ------------------------ Mesh Smoothing ------------------------ #
# This file contains differentiable operators to filter meshes
# The ops include
# 1) Taubin Smoothing
# TODO(gkioxari) add more! :)
# ---------------------------------------------------------------- #


# ----------------------- Taubin Smoothing ----------------------- #


def taubin_smoothing(
    meshes: Meshes, lambd: float = 0.53, mu: float = -0.53, num_iter: int = 10
) -> Meshes:
    """
    Taubin smoothing [1] is an iterative smoothing operator for meshes.
    At each iteration
        verts := (1 - λ) * verts + λ * L * verts
        verts := (1 - μ) * verts + μ * L * verts

    This function returns a new mesh with smoothed vertices.
    Args:
        meshes: Meshes input to be smoothed
        lambd, mu: float parameters for Taubin smoothing,
            lambd > 0, mu < 0
        num_iter: number of iterations to execute smoothing
    Returns:
        mesh: Smoothed input Meshes

    [1] Curve and Surface Smoothing without Shrinkage,
        Gabriel Taubin, ICCV 1997
    """
    verts = meshes.verts_packed()  # V x 3
    edges = meshes.edges_packed()  # E x 3

    for _ in range(num_iter):
        L = norm_laplacian(verts, edges)
        total_weight = torch.sparse.sum(L, dim=1).to_dense().view(-1, 1)
        verts = (1 - lambd) * verts + lambd * torch.mm(L, verts) / total_weight

        L = norm_laplacian(verts, edges)
        total_weight = torch.sparse.sum(L, dim=1).to_dense().view(-1, 1)
        verts = (1 - mu) * verts + mu * torch.mm(L, verts) / total_weight

    verts_list = struct_utils.packed_to_list(
        verts, meshes.num_verts_per_mesh().tolist()
    )
    mesh = Meshes(verts=list(verts_list), faces=meshes.faces_list())
    return mesh
