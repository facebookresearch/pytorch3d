# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import torch
from pytorch3d import _C


def knn_points_idx(
    p1,
    p2,
    K: int,
    lengths1=None,
    lengths2=None,
    sorted: bool = False,
    version: int = -1,
):
    """
    K-Nearest neighbors on point clouds.

    Args:
        p1: Tensor of shape (N, P1, D) giving a batch of point clouds, each
            containing up to P1 points of dimension D.
        p2: Tensor of shape (N, P2, D) giving a batch of point clouds, each
            containing up to P2 points of dimension D.
        K: Integer giving the number of nearest neighbors to return.
        lengths1: LongTensor of shape (N,) of values in the range [0, P1], giving the
            length of each pointcloud in p1. Or None to indicate that every cloud has
            length P1.
        lengths2: LongTensor of shape (N,) of values in the range [0, P2], giving the
            length of each pointcloud in p2. Or None to indicate that every cloud has
            length P2.
        sorted: Whether to sort the resulting points.
        version: Which KNN implementation to use in the backend. If version=-1,
            the correct implementation is selected based on the shapes of the inputs.

    Returns:
        p1_neighbor_idx: LongTensor of shape (N, P1, K) giving the indices of the
            K nearest neighbors from points in p1 to points in p2.
            Concretely, if idx[n, i, k] = j then p2[n, j] is one of the K nearest
            neighbors to p1[n, i] in p2[n]. If sorted=True, then p2[n, j] is the kth
            nearest neighbor to p1[n, i]. This is padded with zeros both where a cloud
            in p2 has fewer than K points and where a cloud in p1 has fewer than P1
            points.
            If you want an (N, P1, K, D) tensor of the actual points, you can get it
            using
                p2[:, :, None].expand(-1, -1, K, -1).gather(1,
                    x_idx[:, :, :, None].expand(-1, -1, -1, D)
                )
            If K=1 and you want an (N, P1, D) tensor of the actual points, use
                p2.gather(1, x_idx.expand(-1, -1, D))

        p1_neighbor_dists: Tensor of shape (N, P1, K) giving the squared distances to
            the nearest neighbors. This is padded with zeros both where a cloud in p2
            has fewer than K points and where a cloud in p1 has fewer than P1 points.
            Warning: this is calculated outside of the autograd framework.
    """
    P1 = p1.shape[1]
    P2 = p2.shape[1]
    if lengths1 is None:
        lengths1 = torch.full((p1.shape[0],), P1, dtype=torch.int64, device=p1.device)
    if lengths2 is None:
        lengths2 = torch.full((p1.shape[0],), P2, dtype=torch.int64, device=p1.device)
    idx, dists = _C.knn_points_idx(p1, p2, lengths1, lengths2, K, version)
    if sorted:
        if lengths2.min() < K:
            device = dists.device
            mask1 = lengths2[:, None] <= torch.arange(K, device=device)[None]
            # mask1 has shape [N, K], true where dists irrelevant
            mask2 = mask1[:, None].expand(-1, P1, -1)
            # mask2 has shape [N, P1, K], true where dists irrelevant
            dists[mask2] = float("inf")
            dists, sort_idx = dists.sort(dim=2)
            dists[mask2] = 0
        else:
            dists, sort_idx = dists.sort(dim=2)
        idx = idx.gather(2, sort_idx)
    return idx, dists


@torch.no_grad()
def _knn_points_idx_naive(p1, p2, K: int, lengths1, lengths2) -> torch.Tensor:
    """
    Naive PyTorch implementation of K-Nearest Neighbors.

    This is much less efficient than _C.knn_points_idx, but we include this
    naive implementation for testing and benchmarking.

    Args:
        p1: Tensor of shape (N, P1, D) giving a batch of point clouds, each
            containing up to P1 points of dimension D.
        p2: Tensor of shape (N, P2, D) giving a batch of point clouds, each
            containing up to P2 points of dimension D.
        K: Integer giving the number of nearest neighbors to return.
        lengths1: LongTensor of shape (N,) of values in the range [0, P1], giving the
            length of each pointcloud in p1. Or None to indicate that every cloud has
            length P1.
        lengths2: LongTensor of shape (N,) of values in the range [0, P2], giving the
            length of each pointcloud in p2. Or None to indicate that every cloud has
            length P2.

    Returns:
        idx: LongTensor of shape (N, P1, K) giving the indices of the
            K nearest neighbors from points in p1 to points in p2.
            Concretely, if idx[n, i, k] = j then p2[n, j] is the kth nearest neighbor
            to p1[n, i]. This is padded with zeros both where a cloud in p2 has fewer
            than K points and where a cloud in p1 has fewer than P1 points.
        dists: Tensor of shape (N, P1, K) giving the squared distances to the nearest
            neighbors. This is padded with zeros both where a cloud in p2 has fewer than
            K points and where a cloud in p1 has fewer than P1 points.
    """
    N, P1, D = p1.shape
    _N, P2, _D = p2.shape

    assert N == _N and D == _D

    if lengths1 is None:
        lengths1 = torch.full((N,), P1, dtype=torch.int64, device=p1.device)
    if lengths2 is None:
        lengths2 = torch.full((N,), P2, dtype=torch.int64, device=p1.device)

    p1_copy = p1.clone()
    p2_copy = p2.clone()

    # We pad the values with infinities so that the smallest differences are
    # among actual points.
    inf = float("inf")
    p1_mask = torch.arange(P1, device=p1.device)[None] >= lengths1[:, None]
    p1_copy[p1_mask] = inf
    p2_copy[torch.arange(P2, device=p1.device)[None] >= lengths2[:, None]] = -inf

    # view is safe here: we are merely adding extra dimensions of length 1
    diffs = p1_copy.view(N, P1, 1, D) - p2_copy.view(N, 1, P2, D)
    dists2 = (diffs * diffs).sum(dim=3)

    # We always sort, because this works well with padding.
    out = dists2.topk(min(K, P2), dim=2, largest=False, sorted=True)

    out_indices = out.indices
    out_values = out.values

    if P2 < K:
        # Need to add padding
        pad_shape = (N, P1, K - P2)
        out_indices = torch.cat([out_indices, out_indices.new_zeros(pad_shape)], 2)
        out_values = torch.cat([out_values, out_values.new_zeros(pad_shape)], 2)

    K_mask = torch.arange(K, device=p1.device)[None] >= lengths2[:, None]
    # Create a combined mask for where the points in p1 are padded
    # or the corresponding p2 has fewer than K points.
    p1_K_mask = p1_mask[:, :, None] | K_mask[:, None, :]
    out_indices[p1_K_mask] = 0
    out_values[p1_K_mask] = 0
    return out_indices, out_values
