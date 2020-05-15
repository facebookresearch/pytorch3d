// Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

#pragma once
#include <torch/extension.h>
#include <cstdio>
#include <tuple>
#include "utils/pytorch3d_cutils.h"

// ****************************************************************************
// *                      PointEdgeDistance                                   *
// ****************************************************************************

// Computes the squared euclidean distance of each p in points to the closest
// mesh edge belonging to the corresponding example in the batch of size N.
//
// Args:
//    points: FloatTensor of shape (P, 3)
//    points_first_idx: LongTensor of shape (N,) indicating the first point
//         index for each example in the batch
//    segms: FloatTensor of shape (S, 2, 3) of edge segments. The s-th edge
//        segment is spanned by (segms[s, 0], segms[s, 1])
//    segms_first_idx: LongTensor of shape (N,) indicating the first edge
//        index for each example in the batch
//    max_points: Scalar equal to max(P_i) for i in [0, N - 1] containing
//        the maximum number of points in the batch and is used to set
//        the grid dimensions in the CUDA implementation.
//
// Returns:
//    dists: FloatTensor of shape (P,), where dists[p] is the squared euclidean
//        distance of points[p] to the closest edge in the same example in the
//        batch.
//    idxs: LongTensor of shape (P,), where idxs[p] is the index of the closest
//        edge in the batch.
//        So, dists[p] = d(points[p], segms[idxs[p], 0], segms[idxs[p], 1]),
//        where d(u, v0, v1) is the distance of u from the segment spanned by
//        (v0, v1).
//

#ifdef WITH_CUDA

std::tuple<torch::Tensor, torch::Tensor> PointEdgeDistanceForwardCuda(
    const torch::Tensor& points,
    const torch::Tensor& points_first_idx,
    const torch::Tensor& segms,
    const torch::Tensor& segms_first_idx,
    const int64_t max_points);
#endif

std::tuple<torch::Tensor, torch::Tensor> PointEdgeDistanceForward(
    const torch::Tensor& points,
    const torch::Tensor& points_first_idx,
    const torch::Tensor& segms,
    const torch::Tensor& segms_first_idx,
    const int64_t max_points) {
  if (points.is_cuda()) {
#ifdef WITH_CUDA
    CHECK_CUDA(points);
    CHECK_CUDA(points_first_idx);
    CHECK_CUDA(segms);
    CHECK_CUDA(segms_first_idx);
    return PointEdgeDistanceForwardCuda(
        points, points_first_idx, segms, segms_first_idx, max_points);
#else
    AT_ERROR("Not compiled with GPU support.");
#endif
  }
  AT_ERROR("No CPU implementation.");
}

// Backward pass for PointEdgeDistance.
//
// Args:
//    points: FloatTensor of shape (P, 3)
//    segms: FloatTensor of shape (S, 2, 3)
//    idx_points: LongTensor of shape (P,) containing the indices
//        of the closest edge in the example in the batch.
//        This is computed by the forward pass.
//    grad_dists: FloatTensor of shape (P,)
//
// Returns:
//    grad_points: FloatTensor of shape (P, 3)
//    grad_segms: FloatTensor of shape (S, 2, 3)
//

#ifdef WITH_CUDA

std::tuple<torch::Tensor, torch::Tensor> PointEdgeDistanceBackwardCuda(
    const torch::Tensor& points,
    const torch::Tensor& segms,
    const torch::Tensor& idx_points,
    const torch::Tensor& grad_dists);
#endif

std::tuple<torch::Tensor, torch::Tensor> PointEdgeDistanceBackward(
    const torch::Tensor& points,
    const torch::Tensor& segms,
    const torch::Tensor& idx_points,
    const torch::Tensor& grad_dists) {
  if (points.is_cuda()) {
#ifdef WITH_CUDA
    CHECK_CUDA(points);
    CHECK_CUDA(segms);
    CHECK_CUDA(idx_points);
    CHECK_CUDA(grad_dists);
    return PointEdgeDistanceBackwardCuda(points, segms, idx_points, grad_dists);
#else
    AT_ERROR("Not compiled with GPU support.");
#endif
  }
  AT_ERROR("No CPU implementation.");
}

// ****************************************************************************
// *                      EdgePointDistance                                   *
// ****************************************************************************

// Computes the squared euclidean distance of each edge segment to the closest
// point belonging to the corresponding example in the batch of size N.
//
// Args:
//    points: FloatTensor of shape (P, 3)
//    points_first_idx: LongTensor of shape (N,) indicating the first point
//         index for each example in the batch
//    segms: FloatTensor of shape (S, 2, 3) of edge segments. The s-th edge
//        segment is spanned by (segms[s, 0], segms[s, 1])
//    segms_first_idx: LongTensor of shape (N,) indicating the first edge
//        index for each example in the batch
//    max_segms: Scalar equal to max(S_i) for i in [0, N - 1] containing
//        the maximum number of edges in the batch and is used to set
//        the block dimensions in the CUDA implementation.
//
// Returns:
//    dists: FloatTensor of shape (S,), where dists[s] is the squared
//        euclidean distance of s-th edge to the closest point in the
//        corresponding example in the batch.
//    idxs: LongTensor of shape (S,), where idxs[s] is the index of the closest
//        point in the example in the batch.
//        So, dists[s] = d(points[idxs[s]], segms[s, 0], segms[s, 1]), where
//        d(u, v0, v1) is the distance of u from the segment spanned by (v0, v1)
//
//

#ifdef WITH_CUDA

std::tuple<torch::Tensor, torch::Tensor> EdgePointDistanceForwardCuda(
    const torch::Tensor& points,
    const torch::Tensor& points_first_idx,
    const torch::Tensor& segms,
    const torch::Tensor& segms_first_idx,
    const int64_t max_segms);
#endif

std::tuple<torch::Tensor, torch::Tensor> EdgePointDistanceForward(
    const torch::Tensor& points,
    const torch::Tensor& points_first_idx,
    const torch::Tensor& segms,
    const torch::Tensor& segms_first_idx,
    const int64_t max_segms) {
  if (points.is_cuda()) {
#ifdef WITH_CUDA
    CHECK_CUDA(points);
    CHECK_CUDA(points_first_idx);
    CHECK_CUDA(segms);
    CHECK_CUDA(segms_first_idx);
    return EdgePointDistanceForwardCuda(
        points, points_first_idx, segms, segms_first_idx, max_segms);
#else
    AT_ERROR("Not compiled with GPU support.");
#endif
  }
  AT_ERROR("No CPU implementation.");
}

// Backward pass for EdgePointDistance.
//
// Args:
//    points: FloatTensor of shape (P, 3)
//    segms: FloatTensor of shape (S, 2, 3)
//    idx_segms: LongTensor of shape (S,) containing the indices
//        of the closest point in the example in the batch.
//        This is computed by the forward pass
//    grad_dists: FloatTensor of shape (S,)
//
// Returns:
//    grad_points: FloatTensor of shape (P, 3)
//    grad_segms: FloatTensor of shape (S, 2, 3)
//

#ifdef WITH_CUDA

std::tuple<torch::Tensor, torch::Tensor> EdgePointDistanceBackwardCuda(
    const torch::Tensor& points,
    const torch::Tensor& segms,
    const torch::Tensor& idx_segms,
    const torch::Tensor& grad_dists);
#endif

std::tuple<torch::Tensor, torch::Tensor> EdgePointDistanceBackward(
    const torch::Tensor& points,
    const torch::Tensor& segms,
    const torch::Tensor& idx_segms,
    const torch::Tensor& grad_dists) {
  if (points.is_cuda()) {
#ifdef WITH_CUDA
    CHECK_CUDA(points);
    CHECK_CUDA(segms);
    CHECK_CUDA(idx_segms);
    CHECK_CUDA(grad_dists);
    return EdgePointDistanceBackwardCuda(points, segms, idx_segms, grad_dists);
#else
    AT_ERROR("Not compiled with GPU support.");
#endif
  }
  AT_ERROR("No CPU implementation.");
}

// ****************************************************************************
// *                          PointEdgeArrayDistance                          *
// ****************************************************************************

// Computes the squared euclidean distance of each p in points to each edge
// segment in segms.
//
// Args:
//    points: FloatTensor of shape (P, 3)
//    segms: FloatTensor of shape (S, 2, 3) of edge segments. The s-th
//        edge segment is spanned by (segms[s, 0], segms[s, 1])
//
// Returns:
//    dists: FloatTensor of shape (P, S), where dists[p, s] is the squared
//        euclidean distance of points[p] to the segment spanned by
//        (segms[s, 0], segms[s, 1])
//
// For pointcloud and meshes of batch size N, this function requires N
// computations. The memory occupied is O(NPS) which can become quite large.
// For example, a medium sized batch with N = 32 with P = 10000 and S = 5000
// will require for the forward pass 5.8G of memory to store dists.

#ifdef WITH_CUDA
torch::Tensor PointEdgeArrayDistanceForwardCuda(
    const torch::Tensor& points,
    const torch::Tensor& segms);
#endif

torch::Tensor PointEdgeArrayDistanceForward(
    const torch::Tensor& points,
    const torch::Tensor& segms) {
  if (points.is_cuda()) {
#ifdef WITH_CUDA
    CHECK_CUDA(points);
    CHECK_CUDA(segms);
    return PointEdgeArrayDistanceForwardCuda(points, segms);
#else
    AT_ERROR("Not compiled with GPU support.");
#endif
  }
  AT_ERROR("No CPU implementation.");
}

// Backward pass for PointEdgeArrayDistance.
//
// Args:
//    points: FloatTensor of shape (P, 3)
//    segms: FloatTensor of shape (S, 2, 3)
//    grad_dists: FloatTensor of shape (P, S)
//
// Returns:
//   grad_points: FloatTensor of shape (P, 3)
//   grad_segms: FloatTensor of shape (S, 2, 3)
//

#ifdef WITH_CUDA

std::tuple<torch::Tensor, torch::Tensor> PointEdgeArrayDistanceBackwardCuda(
    const torch::Tensor& points,
    const torch::Tensor& segms,
    const torch::Tensor& grad_dists);
#endif

std::tuple<torch::Tensor, torch::Tensor> PointEdgeArrayDistanceBackward(
    const torch::Tensor& points,
    const torch::Tensor& segms,
    const torch::Tensor& grad_dists) {
  if (points.is_cuda()) {
#ifdef WITH_CUDA
    CHECK_CUDA(points);
    CHECK_CUDA(segms);
    CHECK_CUDA(grad_dists);
    return PointEdgeArrayDistanceBackwardCuda(points, segms, grad_dists);
#else
    AT_ERROR("Not compiled with GPU support.");
#endif
  }
  AT_ERROR("No CPU implementation.");
}
