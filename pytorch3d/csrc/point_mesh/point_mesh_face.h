// Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

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
//        triangulare face is spanned by (tris[t, 0], tris[t, 1], tris[t, 2])
//    tris_first_idx: LongTensor of shape (N,) indicating the first face
//        index for each example in the batch
//    max_points: Scalar equal to max(P_i) for i in [0, N - 1] containing
//        the maximum number of points in the batch and is used to set
//        the block dimensions in the CUDA implementation.
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
    const int64_t max_points);
#endif

std::tuple<torch::Tensor, torch::Tensor> PointFaceDistanceForward(
    const torch::Tensor& points,
    const torch::Tensor& points_first_idx,
    const torch::Tensor& tris,
    const torch::Tensor& tris_first_idx,
    const int64_t max_points) {
  if (points.is_cuda()) {
#ifdef WITH_CUDA
    CHECK_CUDA(points);
    CHECK_CUDA(points_first_idx);
    CHECK_CUDA(tris);
    CHECK_CUDA(tris_first_idx);
    return PointFaceDistanceForwardCuda(
        points, points_first_idx, tris, tris_first_idx, max_points);
#else
    AT_ERROR("Not compiled with GPU support.");
#endif
  }
  AT_ERROR("No CPU implementation.");
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
    const torch::Tensor& grad_dists);
#endif

std::tuple<torch::Tensor, torch::Tensor> PointFaceDistanceBackward(
    const torch::Tensor& points,
    const torch::Tensor& tris,
    const torch::Tensor& idx_points,
    const torch::Tensor& grad_dists) {
  if (points.is_cuda()) {
#ifdef WITH_CUDA
    CHECK_CUDA(points);
    CHECK_CUDA(tris);
    CHECK_CUDA(idx_points);
    CHECK_CUDA(grad_dists);
    return PointFaceDistanceBackwardCuda(points, tris, idx_points, grad_dists);
#else
    AT_ERROR("Not compiled with GPU support.");
#endif
  }
  AT_ERROR("No CPU implementation.");
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
//        triangulare face is spanned by (tris[t, 0], tris[t, 1], tris[t, 2])
//    tris_first_idx: LongTensor of shape (N,) indicating the first face
//        index for each example in the batch
//    max_tris: Scalar equal to max(T_i) for i in [0, N - 1] containing
//        the maximum number of faces in the batch and is used to set
//        the block dimensions in the CUDA implementation.
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
    const int64_t max_tros);
#endif

std::tuple<torch::Tensor, torch::Tensor> FacePointDistanceForward(
    const torch::Tensor& points,
    const torch::Tensor& points_first_idx,
    const torch::Tensor& tris,
    const torch::Tensor& tris_first_idx,
    const int64_t max_tris) {
  if (points.is_cuda()) {
#ifdef WITH_CUDA
    CHECK_CUDA(points);
    CHECK_CUDA(points_first_idx);
    CHECK_CUDA(tris);
    CHECK_CUDA(tris_first_idx);
    return FacePointDistanceForwardCuda(
        points, points_first_idx, tris, tris_first_idx, max_tris);
#else
    AT_ERROR("Not compiled with GPU support.");
#endif
  }
  AT_ERROR("No CPU implementation.");
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
    const torch::Tensor& grad_dists);
#endif

std::tuple<torch::Tensor, torch::Tensor> FacePointDistanceBackward(
    const torch::Tensor& points,
    const torch::Tensor& tris,
    const torch::Tensor& idx_tris,
    const torch::Tensor& grad_dists) {
  if (points.is_cuda()) {
#ifdef WITH_CUDA
    CHECK_CUDA(points);
    CHECK_CUDA(tris);
    CHECK_CUDA(idx_tris);
    CHECK_CUDA(grad_dists);
    return FacePointDistanceBackwardCuda(points, tris, idx_tris, grad_dists);
#else
    AT_ERROR("Not compiled with GPU support.");
#endif
  }
  AT_ERROR("No CPU implementation.");
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
//        triangulare face is spanned by (tris[t, 0], tris[t, 1], tris[t, 2])
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
    const torch::Tensor& tris);
#endif

torch::Tensor PointFaceArrayDistanceForward(
    const torch::Tensor& points,
    const torch::Tensor& tris) {
  if (points.is_cuda()) {
#ifdef WITH_CUDA
    CHECK_CUDA(points);
    CHECK_CUDA(tris);
    return PointFaceArrayDistanceForwardCuda(points, tris);
#else
    AT_ERROR("Not compiled with GPU support.");
#endif
  }
  AT_ERROR("No CPU implementation.");
}

// Backward pass for PointFaceArrayDistance.
//
// Args:
//    points: FloatTensor of shape (P, 3)
//    tris: FloatTensor of shape (T, 3, 3)
//    grad_dists: FloatTensor of shape (P, T)
//
// Returns:
//    grad_points: FloatTensor of shape (P, 3)
//    grad_tris: FloatTensor of shape (T, 3, 3)
//

#ifdef WITH_CUDA
std::tuple<torch::Tensor, torch::Tensor> PointFaceArrayDistanceBackwardCuda(
    const torch::Tensor& points,
    const torch::Tensor& tris,
    const torch::Tensor& grad_dists);
#endif

std::tuple<torch::Tensor, torch::Tensor> PointFaceArrayDistanceBackward(
    const torch::Tensor& points,
    const torch::Tensor& tris,
    const torch::Tensor& grad_dists) {
  if (points.is_cuda()) {
#ifdef WITH_CUDA
    CHECK_CUDA(points);
    CHECK_CUDA(tris);
    CHECK_CUDA(grad_dists);
    return PointFaceArrayDistanceBackwardCuda(points, tris, grad_dists);
#else
    AT_ERROR("Not compiled with GPU support.");
#endif
  }
  AT_ERROR("No CPU implementation.");
}
