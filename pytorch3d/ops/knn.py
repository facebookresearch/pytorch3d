# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections import namedtuple
from typing import Union

import torch
from pytorch3d import _C
from torch.autograd import Function
from torch.autograd.function import once_differentiable


_KNN = namedtuple("KNN", "dists idx knn")


class _knn_points(Function):
    """
    Torch autograd Function wrapper for KNN C++/CUDA implementations.
    """

    @staticmethod
    # pyre-fixme[14]: `forward` overrides method defined in `Function` inconsistently.
    def forward(
        ctx,
        p1,
        p2,
        lengths1,
        lengths2,
        K,
        version,
        norm: int = 2,
        return_sorted: bool = True,
    ):
        """
        K-Nearest neighbors on point clouds.

        Args:
            p1: Tensor of shape (N, P1, D) giving a batch of N point clouds, each
                containing up to P1 points of dimension D.
            p2: Tensor of shape (N, P2, D) giving a batch of N point clouds, each
                containing up to P2 points of dimension D.
            lengths1: LongTensor of shape (N,) of values in the range [0, P1], giving the
                length of each pointcloud in p1. Or None to indicate that every cloud has
                length P1.
            lengths2: LongTensor of shape (N,) of values in the range [0, P2], giving the
                length of each pointcloud in p2. Or None to indicate that every cloud has
                length P2.
            K: Integer giving the number of nearest neighbors to return.
            version: Which KNN implementation to use in the backend. If version=-1,
                the correct implementation is selected based on the shapes of the inputs.
            norm: (int) indicating the norm. Only supports 1 (for L1) and 2 (for L2).
            return_sorted: (bool) whether to return the nearest neighbors sorted in
                ascending order of distance.

        Returns:
            p1_dists: Tensor of shape (N, P1, K) giving the squared distances to
                the nearest neighbors. This is padded with zeros both where a cloud in p2
                has fewer than K points and where a cloud in p1 has fewer than P1 points.

            p1_idx: LongTensor of shape (N, P1, K) giving the indices of the
                K nearest neighbors from points in p1 to points in p2.
                Concretely, if `p1_idx[n, i, k] = j` then `p2[n, j]` is the k-th nearest
                neighbors to `p1[n, i]` in `p2[n]`. This is padded with zeros both where a cloud
                in p2 has fewer than K points and where a cloud in p1 has fewer than P1 points.
        """
        if not ((norm == 1) or (norm == 2)):
            raise ValueError("Support for 1 or 2 norm.")

        idx, dists = _C.knn_points_idx(p1, p2, lengths1, lengths2, norm, K, version)

        # sort KNN in ascending order if K > 1
        if K > 1 and return_sorted:
            if lengths2.min() < K:
                P1 = p1.shape[1]
                mask = lengths2[:, None] <= torch.arange(K, device=dists.device)[None]
                # mask has shape [N, K], true where dists irrelevant
                mask = mask[:, None].expand(-1, P1, -1)
                # mask has shape [N, P1, K], true where dists irrelevant
                dists[mask] = float("inf")
                dists, sort_idx = dists.sort(dim=2)
                dists[mask] = 0
            else:
                dists, sort_idx = dists.sort(dim=2)
            idx = idx.gather(2, sort_idx)

        ctx.save_for_backward(p1, p2, lengths1, lengths2, idx)
        ctx.mark_non_differentiable(idx)
        ctx.norm = norm
        return dists, idx

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_dists, grad_idx):
        p1, p2, lengths1, lengths2, idx = ctx.saved_tensors
        norm = ctx.norm
        # TODO(gkioxari) Change cast to floats once we add support for doubles.
        if not (grad_dists.dtype == torch.float32):
            grad_dists = grad_dists.float()
        if not (p1.dtype == torch.float32):
            p1 = p1.float()
        if not (p2.dtype == torch.float32):
            p2 = p2.float()
        grad_p1, grad_p2 = _C.knn_points_backward(
            p1, p2, lengths1, lengths2, idx, norm, grad_dists
        )
        return grad_p1, grad_p2, None, None, None, None, None, None


def knn_points(
    p1: torch.Tensor,
    p2: torch.Tensor,
    lengths1: Union[torch.Tensor, None] = None,
    lengths2: Union[torch.Tensor, None] = None,
    norm: int = 2,
    K: int = 1,
    version: int = -1,
    return_nn: bool = False,
    return_sorted: bool = True,
) -> _KNN:
    """
    K-Nearest neighbors on point clouds.

    Args:
        p1: Tensor of shape (N, P1, D) giving a batch of N point clouds, each
            containing up to P1 points of dimension D.
        p2: Tensor of shape (N, P2, D) giving a batch of N point clouds, each
            containing up to P2 points of dimension D.
        lengths1: LongTensor of shape (N,) of values in the range [0, P1], giving the
            length of each pointcloud in p1. Or None to indicate that every cloud has
            length P1.
        lengths2: LongTensor of shape (N,) of values in the range [0, P2], giving the
            length of each pointcloud in p2. Or None to indicate that every cloud has
            length P2.
        norm: Integer indicating the norm of the distance. Supports only 1 for L1, 2 for L2.
        K: Integer giving the number of nearest neighbors to return.
        version: Which KNN implementation to use in the backend. If version=-1,
            the correct implementation is selected based on the shapes of the inputs.
        return_nn: If set to True returns the K nearest neighbors in p2 for each point in p1.
        return_sorted: (bool) whether to return the nearest neighbors sorted in
            ascending order of distance.

    Returns:
        dists: Tensor of shape (N, P1, K) giving the squared distances to
            the nearest neighbors. This is padded with zeros both where a cloud in p2
            has fewer than K points and where a cloud in p1 has fewer than P1 points.

        idx: LongTensor of shape (N, P1, K) giving the indices of the
            K nearest neighbors from points in p1 to points in p2.
            Concretely, if `p1_idx[n, i, k] = j` then `p2[n, j]` is the k-th nearest
            neighbors to `p1[n, i]` in `p2[n]`. This is padded with zeros both where a cloud
            in p2 has fewer than K points and where a cloud in p1 has fewer than P1
            points.

        nn: Tensor of shape (N, P1, K, D) giving the K nearest neighbors in p2 for
            each point in p1. Concretely, `p2_nn[n, i, k]` gives the k-th nearest neighbor
            for `p1[n, i]`. Returned if `return_nn` is True.
            The nearest neighbors are collected using `knn_gather`

            .. code-block::

                p2_nn = knn_gather(p2, p1_idx, lengths2)

            which is a helper function that allows indexing any tensor of shape (N, P2, U) with
            the indices `p1_idx` returned by `knn_points`. The output is a tensor
            of shape (N, P1, K, U).

    """
    if p1.shape[0] != p2.shape[0]:
        raise ValueError("pts1 and pts2 must have the same batch dimension.")
    if p1.shape[2] != p2.shape[2]:
        raise ValueError("pts1 and pts2 must have the same point dimension.")

    p1 = p1.contiguous()
    p2 = p2.contiguous()

    P1 = p1.shape[1]
    P2 = p2.shape[1]

    if lengths1 is None:
        lengths1 = torch.full((p1.shape[0],), P1, dtype=torch.int64, device=p1.device)
    if lengths2 is None:
        lengths2 = torch.full((p1.shape[0],), P2, dtype=torch.int64, device=p1.device)

    p1_dists, p1_idx = _knn_points.apply(
        p1, p2, lengths1, lengths2, K, version, norm, return_sorted
    )

    p2_nn = None
    if return_nn:
        p2_nn = knn_gather(p2, p1_idx, lengths2)

    return _KNN(dists=p1_dists, idx=p1_idx, knn=p2_nn if return_nn else None)


def knn_gather(
    x: torch.Tensor, idx: torch.Tensor, lengths: Union[torch.Tensor, None] = None
):
    """
    A helper function for knn that allows indexing a tensor x with the indices `idx`
    returned by `knn_points`.

    For example, if `dists, idx = knn_points(p, x, lengths_p, lengths, K)`
    where p is a tensor of shape (N, L, D) and x a tensor of shape (N, M, D),
    then one can compute the K nearest neighbors of p with `p_nn = knn_gather(x, idx, lengths)`.
    It can also be applied for any tensor x of shape (N, M, U) where U != D.

    Args:
        x: Tensor of shape (N, M, U) containing U-dimensional features to
            be gathered.
        idx: LongTensor of shape (N, L, K) giving the indices returned by `knn_points`.
        lengths: LongTensor of shape (N,) of values in the range [0, M], giving the
            length of each example in the batch in x. Or None to indicate that every
            example has length M.
    Returns:
        x_out: Tensor of shape (N, L, K, U) resulting from gathering the elements of x
            with idx, s.t. `x_out[n, l, k] = x[n, idx[n, l, k]]`.
            If `k > lengths[n]` then `x_out[n, l, k]` is filled with 0.0.
    """
    N, M, U = x.shape
    _N, L, K = idx.shape

    if N != _N:
        raise ValueError("x and idx must have same batch dimension.")

    if lengths is None:
        lengths = torch.full((x.shape[0],), M, dtype=torch.int64, device=x.device)

    idx_expanded = idx[:, :, :, None].expand(-1, -1, -1, U)
    # idx_expanded has shape [N, L, K, U]

    x_out = x[:, :, None].expand(-1, -1, K, -1).gather(1, idx_expanded)
    # p2_nn has shape [N, L, K, U]

    needs_mask = lengths.min() < K
    if needs_mask:
        # mask has shape [N, K], true where idx is irrelevant because
        # there is less number of points in p2 than K
        mask = lengths[:, None] <= torch.arange(K, device=x.device)[None]

        # expand mask to shape [N, L, K, U]
        mask = mask[:, None].expand(-1, L, -1)
        mask = mask[:, :, :, None].expand(-1, -1, -1, U)
        x_out[mask] = 0.0

    return x_out
