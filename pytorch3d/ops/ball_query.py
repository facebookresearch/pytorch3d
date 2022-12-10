# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Union

import torch
from pytorch3d import _C
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from .knn import _KNN
from .utils import masked_gather


class _ball_query(Function):
    """
    Torch autograd Function wrapper for Ball Query C++/CUDA implementations.
    """

    @staticmethod
    def forward(ctx, p1, p2, lengths1, lengths2, K, radius):
        """
        Arguments defintions the same as in the ball_query function
        """
        idx, dists = _C.ball_query(p1, p2, lengths1, lengths2, K, radius)
        ctx.save_for_backward(p1, p2, lengths1, lengths2, idx)
        ctx.mark_non_differentiable(idx)
        return dists, idx

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_dists, grad_idx):
        p1, p2, lengths1, lengths2, idx = ctx.saved_tensors
        # TODO(gkioxari) Change cast to floats once we add support for doubles.
        if not (grad_dists.dtype == torch.float32):
            grad_dists = grad_dists.float()
        if not (p1.dtype == torch.float32):
            p1 = p1.float()
        if not (p2.dtype == torch.float32):
            p2 = p2.float()

        # Reuse the KNN backward function
        # by default, norm is 2
        grad_p1, grad_p2 = _C.knn_points_backward(
            p1, p2, lengths1, lengths2, idx, 2, grad_dists
        )
        return grad_p1, grad_p2, None, None, None, None


def ball_query(
    p1: torch.Tensor,
    p2: torch.Tensor,
    lengths1: Union[torch.Tensor, None] = None,
    lengths2: Union[torch.Tensor, None] = None,
    K: int = 500,
    radius: float = 0.2,
    return_nn: bool = True,
):
    """
    Ball Query is an alternative to KNN. It can be
    used to find all points in p2 that are within a specified radius
    to the query point in p1 (with an upper limit of K neighbors).

    The neighbors returned are not necssarily the *nearest* to the
    point in p1, just the first K values in p2 which are within the
    specified radius.

    This method is faster than kNN when there are large numbers of points
    in p2 and the ordering of neighbors is not important compared to the
    distance being within the radius threshold.

    "Ball query’s local neighborhood guarantees a fixed region scale thus
    making local region features more generalizable across space, which is
    preferred for tasks requiring local pattern recognition
    (e.g. semantic point labeling)" [1].

    [1] Charles R. Qi et al, "PointNet++: Deep Hierarchical Feature Learning
        on Point Sets in a Metric Space", NeurIPS 2017.

    Args:
        p1: Tensor of shape (N, P1, D) giving a batch of N point clouds, each
            containing up to P1 points of dimension D. These represent the centers of
            the ball queries.
        p2: Tensor of shape (N, P2, D) giving a batch of N point clouds, each
            containing up to P2 points of dimension D.
        lengths1: LongTensor of shape (N,) of values in the range [0, P1], giving the
            length of each pointcloud in p1. Or None to indicate that every cloud has
            length P1.
        lengths2: LongTensor of shape (N,) of values in the range [0, P2], giving the
            length of each pointcloud in p2. Or None to indicate that every cloud has
            length P2.
        K: Integer giving the upper bound on the number of samples to take
            within the radius
        radius: the radius around each point within which the neighbors need to be located
        return_nn: If set to True returns the K neighbor points in p2 for each point in p1.

    Returns:
        dists: Tensor of shape (N, P1, K) giving the squared distances to
            the neighbors. This is padded with zeros both where a cloud in p2
            has fewer than S points and where a cloud in p1 has fewer than P1 points
            and also if there are fewer than K points which satisfy the radius threshold.

        idx: LongTensor of shape (N, P1, K) giving the indices of the
            S neighbors in p2 for points in p1.
            Concretely, if `p1_idx[n, i, k] = j` then `p2[n, j]` is the k-th
            neighbor to `p1[n, i]` in `p2[n]`. This is padded with -1 both where a cloud
            in p2 has fewer than S points and where a cloud in p1 has fewer than P1
            points and also if there are fewer than K points which satisfy the radius threshold.

        nn: Tensor of shape (N, P1, K, D) giving the K neighbors in p2 for
            each point in p1. Concretely, `p2_nn[n, i, k]` gives the k-th neighbor
            for `p1[n, i]`. Returned if `return_nn` is True.  The output is a tensor
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
    N = p1.shape[0]

    if lengths1 is None:
        lengths1 = torch.full((N,), P1, dtype=torch.int64, device=p1.device)
    if lengths2 is None:
        lengths2 = torch.full((N,), P2, dtype=torch.int64, device=p1.device)

    dists, idx = _ball_query.apply(p1, p2, lengths1, lengths2, K, radius)

    # Gather the neighbors if needed
    points_nn = masked_gather(p2, idx) if return_nn else None

    return _KNN(dists=dists, idx=idx, knn=points_nn)
