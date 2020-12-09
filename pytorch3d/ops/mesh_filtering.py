# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
import torch
from pytorch3d.structures import Meshes, utils as struct_utils


# ------------------------ Mesh Smoothing ------------------------ #
# This file contains differentiable operators to filter meshes
# The ops include
# 1) Taubin Smoothing
# TODO(gkioxari) add more! :)
# ---------------------------------------------------------------- #


# ----------------------- Taubin Smoothing ----------------------- #


def norm_laplacian(verts: torch.Tensor, edges: torch.Tensor, eps: float = 1e-12):
    """
    Norm laplacian computes a variant of the laplacian matrix which weights each
    affinity with the normalized distance of the neighboring nodes.
    More concretely,
    L[i, j] = 1. / wij where wij = ||vi - vj|| if (vi, vj) are neighboring nodes

    Args:
        verts: tensor of shape (V, 3) containing the vertices of the graph
        edges: tensor of shape (E, 2) containing the vertex indices of each edge
    """
    edge_verts = verts[edges]  # (E, 2, 3)
    v0, v1 = edge_verts[:, 0], edge_verts[:, 1]

    # Side lengths of each edge, of shape (E,)
    w01 = 1.0 / ((v0 - v1).norm(dim=1) + eps)

    # Construct a sparse matrix by basically doing:
    # L[v0, v1] = w01
    # L[v1, v0] = w01
    e01 = edges.t()  # (2, E)

    V = verts.shape[0]
    L = torch.sparse.FloatTensor(e01, w01, (V, V))
    L = L + L.t()

    return L


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

        # pyre-ignore
        L = norm_laplacian(verts, edges)
        total_weight = torch.sparse.sum(L, dim=1).to_dense().view(-1, 1)
        verts = (1 - mu) * verts + mu * torch.mm(L, verts) / total_weight

    verts_list = struct_utils.packed_to_list(
        verts, meshes.num_verts_per_mesh().tolist()
    )
    mesh = Meshes(verts=list(verts_list), faces=meshes.faces_list())
    return mesh
