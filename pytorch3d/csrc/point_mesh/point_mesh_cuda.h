/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <torch/extension.h>
#include <cstdio>
#include <tuple>
#include "utils/pytorch3d_cutils.h"

// ****************************************************************************
// *                      PointFaceDistance                                   *
// ****************************************************************************

// Computes the squared euclidean distance of each p in points to it closest
// triangular face belonging to the corresponding mesh example in the batch of
// size N.
//
// Args:
//    points: FloatTensor of shape (P, 3)
//    points_first_idx: LongTensor of shape (N,) indicating the first point
//        index for each example in the batch
//    tris: FloatTensor of shape (T, 3, 3) of the triangular faces. The t-th
//        triangular face is spanned by (tris[t, 0], tris[t, 1], tris[t, 2])
//    tris_first_idx: LongTensor of shape (N,) indicating the first face
//        index for each example in the batch
//    max_points: Scalar equal to max(P_i) for i in [0, N - 1] containing
//        the maximum number of points in the batch and is used to set
//        the block dimensions in the CUDA implementation.
//     min_triangle_area: triangles less than this size are considered
//     points/lines.
//
// Returns:
//    dists: FloatTensor of shape (P,), where dists[p] is the minimum
//        squared euclidean distance of points[p] to the faces in the same
//        example in the batch.
//    idxs: LongTensor of shape (P,), where idxs[p] is the index of the closest
//        face in the batch.
//        So, dists[p] = d(points[p], tris[idxs[p], 0], tris[idxs[p], 1],
//        tris[idxs[p], 2]) where d(u, v0, v1, v2) is the distance of u from the
//        face spanned by (v0, v1, v2)
//
//

#ifdef WITH_CUDA

std::tuple<torch::Tensor, torch::Tensor> PointFaceDistanceForwardCuda(
    const torch::Tensor& points,
    const torch::Tensor& points_first_idx,
    const torch::Tensor& tris,
    const torch::Tensor& tris_first_idx,
    const int64_t max_points,
    const double min_triangle_area);
#endif

std::tuple<torch::Tensor, torch::Tensor> PointFaceDistanceForwardCpu(
    const torch::Tensor& points,
    const torch::Tensor& points_first_idx,
    const torch::Tensor& tris,
    const torch::Tensor& tris_first_idx,
    const double min_triangle_area);

std::tuple<torch::Tensor, torch::Tensor> PointFaceDistanceForward(
    const torch::Tensor& points,
    const torch::Tensor& points_first_idx,
    const torch::Tensor& tris,
    const torch::Tensor& tris_first_idx,
    const int64_t max_points,
    const double min_triangle_area) {
  if (points.is_cuda()) {
#ifdef WITH_CUDA
    CHECK_CUDA(points);
    CHECK_CUDA(points_first_idx);
    CHECK_CUDA(tris);
    CHECK_CUDA(tris_first_idx);
    return PointFaceDistanceForwardCuda(
        points,
        points_first_idx,
        tris,
        tris_first_idx,
        max_points,
        min_triangle_area);
#else
    AT_ERROR("Not compiled with GPU support.");
#endif
  }
  return PointFaceDistanceForwardCpu(
      points, points_first_idx, tris, tris_first_idx, min_triangle_area);
}

// Backward pass for PointFaceDistance.
//
// Args:
//    points: FloatTensor of shape (P, 3)
//    tris: FloatTensor of shape (T, 3, 3)
//    idx_points: LongTensor of shape (P,) containing the indices
//        of the closest face in the example in the batch.
//        This is computed by the forward pass
//    grad_dists: FloatTensor of shape (P,)
//     min_triangle_area: triangles less than this size are considered
//     points/lines.
//
// Returns:
//    grad_points: FloatTensor of shape (P, 3)
//    grad_tris: FloatTensor of shape (T, 3, 3)
//

#ifdef WITH_CUDA

std::tuple<torch::Tensor, torch::Tensor> PointFaceDistanceBackwardCuda(
    const torch::Tensor& points,
    const torch::Tensor& tris,
    const torch::Tensor& idx_points,
    const torch::Tensor& grad_dists,
    const double min_triangle_area);
#endif
std::tuple<torch::Tensor, torch::Tensor> PointFaceDistanceBackwardCpu(
    const torch::Tensor& points,
    const torch::Tensor& tris,
    const torch::Tensor& idx_points,
    const torch::Tensor& grad_dists,
    const double min_triangle_area);

std::tuple<torch::Tensor, torch::Tensor> PointFaceDistanceBackward(
    const torch::Tensor& points,
    const torch::Tensor& tris,
    const torch::Tensor& idx_points,
    const torch::Tensor& grad_dists,
    const double min_triangle_area) {
  if (points.is_cuda()) {
#ifdef WITH_CUDA
    CHECK_CUDA(points);
    CHECK_CUDA(tris);
    CHECK_CUDA(idx_points);
    CHECK_CUDA(grad_dists);
    return PointFaceDistanceBackwardCuda(
        points, tris, idx_points, grad_dists, min_triangle_area);
#else
    AT_ERROR("Not compiled with GPU support.");
#endif
  }
  return PointFaceDistanceBackwardCpu(
      points, tris, idx_points, grad_dists, min_triangle_area);
}

// ****************************************************************************
// *                      FacePointDistance                                   *
// ****************************************************************************

// Computes the squared euclidean distance of each triangular face to its
// closest point belonging to the corresponding example in the batch of size N.
//
// Args:
//    points: FloatTensor of shape (P, 3)
//    points_first_idx: LongTensor of shape (N,) indicating the first point
//        index for each example in the batch
//    tris: FloatTensor of shape (T, 3, 3) of the triangular faces. The t-th
//        triangular face is spanned by (tris[t, 0], tris[t, 1], tris[t, 2])
//    tris_first_idx: LongTensor of shape (N,) indicating the first face
//        index for each example in the batch
//    max_tris: Scalar equal to max(T_i) for i in [0, N - 1] containing
//        the maximum number of faces in the batch and is used to set
//        the block dimensions in the CUDA implementation.
//     min_triangle_area: triangles less than this size are considered
//     points/lines.
//
// Returns:
//    dists: FloatTensor of shape (T,), where dists[t] is the minimum squared
//        euclidean distance of t-th triangular face from the closest point in
//        the batch.
//    idxs: LongTensor of shape (T,), where idxs[t] is the index of the closest
//        point in the batch.
//        So, dists[t] = d(points[idxs[t]], tris[t, 0], tris[t, 1], tris[t, 2])
//        where d(u, v0, v1, v2) is the distance of u from the triangular face
//        spanned by (v0, v1, v2)
//

#ifdef WITH_CUDA

std::tuple<torch::Tensor, torch::Tensor> FacePointDistanceForwardCuda(
    const torch::Tensor& points,
    const torch::Tensor& points_first_idx,
    const torch::Tensor& tris,
    const torch::Tensor& tris_first_idx,
    const int64_t max_tris,
    const double min_triangle_area);
#endif

std::tuple<torch::Tensor, torch::Tensor> FacePointDistanceForwardCpu(
    const torch::Tensor& points,
    const torch::Tensor& points_first_idx,
    const torch::Tensor& tris,
    const torch::Tensor& tris_first_idx,
    const double min_triangle_area);

std::tuple<torch::Tensor, torch::Tensor> FacePointDistanceForward(
    const torch::Tensor& points,
    const torch::Tensor& points_first_idx,
    const torch::Tensor& tris,
    const torch::Tensor& tris_first_idx,
    const int64_t max_tris,
    const double min_triangle_area) {
  if (points.is_cuda()) {
#ifdef WITH_CUDA
    CHECK_CUDA(points);
    CHECK_CUDA(points_first_idx);
    CHECK_CUDA(tris);
    CHECK_CUDA(tris_first_idx);
    return FacePointDistanceForwardCuda(
        points,
        points_first_idx,
        tris,
        tris_first_idx,
        max_tris,
        min_triangle_area);
#else
    AT_ERROR("Not compiled with GPU support.");
#endif
  }
  return FacePointDistanceForwardCpu(
      points, points_first_idx, tris, tris_first_idx, min_triangle_area);
}

// Backward pass for FacePointDistance.
//
// Args:
//    points: FloatTensor of shape (P, 3)
//    tris: FloatTensor of shape (T, 3, 3)
//    idx_tris: LongTensor of shape (T,) containing the indices
//        of the closest point in the example in the batch.
//        This is computed by the forward pass
//    grad_dists: FloatTensor of shape (T,)
//     min_triangle_area: triangles less than this size are considered
//     points/lines.
//
// Returns:
//    grad_points: FloatTensor of shape (P, 3)
//    grad_tris: FloatTensor of shape (T, 3, 3)
//

#ifdef WITH_CUDA

std::tuple<torch::Tensor, torch::Tensor> FacePointDistanceBackwardCuda(
    const torch::Tensor& points,
    const torch::Tensor& tris,
    const torch::Tensor& idx_tris,
    const torch::Tensor& grad_dists,
    const double min_triangle_area);
#endif

std::tuple<torch::Tensor, torch::Tensor> FacePointDistanceBackwardCpu(
    const torch::Tensor& points,
    const torch::Tensor& tris,
    const torch::Tensor& idx_tris,
    const torch::Tensor& grad_dists,
    const double min_triangle_area);

std::tuple<torch::Tensor, torch::Tensor> FacePointDistanceBackward(
    const torch::Tensor& points,
    const torch::Tensor& tris,
    const torch::Tensor& idx_tris,
    const torch::Tensor& grad_dists,
    const double min_triangle_area) {
  if (points.is_cuda()) {
#ifdef WITH_CUDA
    CHECK_CUDA(points);
    CHECK_CUDA(tris);
    CHECK_CUDA(idx_tris);
    CHECK_CUDA(grad_dists);
    return FacePointDistanceBackwardCuda(
        points, tris, idx_tris, grad_dists, min_triangle_area);
#else
    AT_ERROR("Not compiled with GPU support.");
#endif
  }
  return FacePointDistanceBackwardCpu(
      points, tris, idx_tris, grad_dists, min_triangle_area);
}

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

std::tuple<torch::Tensor, torch::Tensor> PointEdgeDistanceForwardCpu(
    const torch::Tensor& points,
    const torch::Tensor& points_first_idx,
    const torch::Tensor& segms,
    const torch::Tensor& segms_first_idx,
    const int64_t max_points);

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
  return PointEdgeDistanceForwardCpu(
      points, points_first_idx, segms, segms_first_idx, max_points);
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

std::tuple<torch::Tensor, torch::Tensor> PointEdgeDistanceBackwardCpu(
    const torch::Tensor& points,
    const torch::Tensor& segms,
    const torch::Tensor& idx_points,
    const torch::Tensor& grad_dists);

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
  return PointEdgeDistanceBackwardCpu(points, segms, idx_points, grad_dists);
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

std::tuple<torch::Tensor, torch::Tensor> EdgePointDistanceForwardCpu(
    const torch::Tensor& points,
    const torch::Tensor& points_first_idx,
    const torch::Tensor& segms,
    const torch::Tensor& segms_first_idx,
    const int64_t max_segms);

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
  return EdgePointDistanceForwardCpu(
      points, points_first_idx, segms, segms_first_idx, max_segms);
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

std::tuple<torch::Tensor, torch::Tensor> EdgePointDistanceBackwardCpu(
    const torch::Tensor& points,
    const torch::Tensor& segms,
    const torch::Tensor& idx_segms,
    const torch::Tensor& grad_dists);

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
  return EdgePointDistanceBackwardCpu(points, segms, idx_segms, grad_dists);
}

// ****************************************************************************
// *                       PointFaceArrayDistance                             *
// ****************************************************************************

// Computes the squared euclidean distance of each p in points to each
// triangular face spanned by (v0, v1, v2) in tris.
//
// Args:
//    points: FloatTensor of shape (P, 3)
//    tris: FloatTensor of shape (T, 3, 3) of the triangular faces. The t-th
//        triangular face is spanned by (tris[t, 0], tris[t, 1], tris[t, 2])
//     min_triangle_area: triangles less than this size are considered
//     points/lines.
//
// Returns:
//    dists: FloatTensor of shape (P, T), where dists[p, t] is the squared
//        euclidean distance of points[p] to the face spanned by (v0, v1, v2)
//        where v0 = tris[t, 0], v1 = tris[t, 1] and v2 = tris[t, 2]
//
// For pointcloud and meshes of batch size N, this function requires N
// computations. The memory occupied is O(NPT) which can become quite large.
// For example, a medium sized batch with N = 32 with P = 10000 and T = 5000
// will require for the forward pass 5.8G of memory to store dists.

#ifdef WITH_CUDA

torch::Tensor PointFaceArrayDistanceForwardCuda(
    const torch::Tensor& points,
    const torch::Tensor& tris,
    const double min_triangle_area);
#endif

torch::Tensor PointFaceArrayDistanceForwardCpu(
    const torch::Tensor& points,
    const torch::Tensor& tris,
    const double min_triangle_area);

torch::Tensor PointFaceArrayDistanceForward(
    const torch::Tensor& points,
    const torch::Tensor& tris,
    const double min_triangle_area) {
  if (points.is_cuda()) {
#ifdef WITH_CUDA
    CHECK_CUDA(points);
    CHECK_CUDA(tris);
    return PointFaceArrayDistanceForwardCuda(points, tris, min_triangle_area);
#else
    AT_ERROR("Not compiled with GPU support.");
#endif
  }
  return PointFaceArrayDistanceForwardCpu(points, tris, min_triangle_area);
}

// Backward pass for PointFaceArrayDistance.
//
// Args:
//    points: FloatTensor of shape (P, 3)
//    tris: FloatTensor of shape (T, 3, 3)
//    grad_dists: FloatTensor of shape (P, T)
//     min_triangle_area: triangles less than this size are considered
//     points/lines.
//
// Returns:
//    grad_points: FloatTensor of shape (P, 3)
//    grad_tris: FloatTensor of shape (T, 3, 3)
//

#ifdef WITH_CUDA
std::tuple<torch::Tensor, torch::Tensor> PointFaceArrayDistanceBackwardCuda(
    const torch::Tensor& points,
    const torch::Tensor& tris,
    const torch::Tensor& grad_dists,
    const double min_triangle_area);
#endif
std::tuple<torch::Tensor, torch::Tensor> PointFaceArrayDistanceBackwardCpu(
    const torch::Tensor& points,
    const torch::Tensor& tris,
    const torch::Tensor& grad_dists,
    const double min_triangle_area);

std::tuple<torch::Tensor, torch::Tensor> PointFaceArrayDistanceBackward(
    const torch::Tensor& points,
    const torch::Tensor& tris,
    const torch::Tensor& grad_dists,
    const double min_triangle_area) {
  if (points.is_cuda()) {
#ifdef WITH_CUDA
    CHECK_CUDA(points);
    CHECK_CUDA(tris);
    CHECK_CUDA(grad_dists);
    return PointFaceArrayDistanceBackwardCuda(
        points, tris, grad_dists, min_triangle_area);
#else
    AT_ERROR("Not compiled with GPU support.");
#endif
  }
  return PointFaceArrayDistanceBackwardCpu(
      points, tris, grad_dists, min_triangle_area);
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

torch::Tensor PointEdgeArrayDistanceForwardCpu(
    const torch::Tensor& points,
    const torch::Tensor& segms);

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
  return PointEdgeArrayDistanceForwardCpu(points, segms);
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

std::tuple<torch::Tensor, torch::Tensor> PointEdgeArrayDistanceBackwardCpu(
    const torch::Tensor& points,
    const torch::Tensor& segms,
    const torch::Tensor& grad_dists);

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
  return PointEdgeArrayDistanceBackwardCpu(points, segms, grad_dists);
}
