# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import torch
from pytorch3d import _C


def knn_points_idx(p1, p2, K, sorted=False, version=-1):
    """
    K-Nearest neighbors on point clouds.

    Args:
        p1: Tensor of shape (N, P1, D) giving a batch of point clouds, each
            containing P1 points of dimension D.
        p2: Tensor of shape (N, P2, D) giving a batch of point clouds, each
            containing P2 points of dimension D.
        K: Integer giving the number of nearest neighbors to return
        sorted: Whether to sort the resulting points.
        version: Which KNN implementation to use in the backend. If version=-1,
                 the correct implementation is selected based on the shapes of
                 the inputs.

    Returns:
        idx: LongTensor of shape (N, P1, K) giving the indices of the
             K nearest neighbors from points in p1 to points in p2.
             Concretely, if idx[n, i, k] = j then p2[n, j] is one of the K
             nearest neighbor to p1[n, i] in p2[n]. If sorted=True, then
             p2[n, j] is the kth nearest neighbor to p1[n, i].
    """
    idx, dists = _C.knn_points_idx(p1, p2, K, version)
    if sorted:
        dists, sort_idx = dists.sort(dim=2)
        idx = idx.gather(2, sort_idx)
    return idx, dists


@torch.no_grad()
def _knn_points_idx_naive(p1, p2, K, sorted=False) -> torch.Tensor:
    """
    Naive PyTorch implementation of K-Nearest Neighbors.

    This is much less efficient than _C.knn_points_idx, but we include this
    naive implementation for testing and benchmarking.

    Args:
        p1: Tensor of shape (N, P1, D) giving a batch of point clouds, each
            containing P1 points of dimension D.
        p2: Tensor of shape (N, P2, D) giving a batch of point clouds, each
            containing P2 points of dimension D.
        K: Integer giving the number of nearest neighbors to return
        sorted: Whether to sort the resulting points.

    Returns:
        idx: LongTensor of shape (N, P1, K) giving the indices of the
             K nearest neighbors from points in p1 to points in p2.
             Concretely, if idx[n, i, k] = j then p2[n, j] is one of the K
             nearest neighbor to p1[n, i] in p2[n]. If sorted=True, then
             p2[n, j] is the kth nearest neighbor to p1[n, i].
        dists: Tensor of shape (N, P1, K) giving the distances to the nearest
               neighbors.
    """
    N, P1, D = p1.shape
    _N, P2, _D = p2.shape
    assert N == _N and D == _D
    diffs = p1.view(N, P1, 1, D) - p2.view(N, 1, P2, D)
    dists2 = (diffs * diffs).sum(dim=3)
    out = dists2.topk(K, dim=2, largest=False, sorted=sorted)
    return out.indices, out.values
