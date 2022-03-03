/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <torch/extension.h>
#include <tuple>
#include "utils/pytorch3d_cutils.h"

// Compute indices of K neighbors in pointcloud p2 to points
// in pointcloud p1 which fall within a specified radius
//
// Args:
//    p1: FloatTensor of shape (N, P1, D) giving a batch of pointclouds each
//        containing P1 points of dimension D.
//    p2: FloatTensor of shape (N, P2, D) giving a batch of pointclouds each
//        containing P2 points of dimension D.
//    lengths1: LongTensor, shape (N,), giving actual length of each P1 cloud.
//    lengths2: LongTensor, shape (N,), giving actual length of each P2 cloud.
//    K: Integer giving the upper bound on the number of samples to take
//      within the radius
//    radius: the radius around each point within which the neighbors need to be
//      located
//
// Returns:
//    p1_neighbor_idx: LongTensor of shape (N, P1, K), where
//        p1_neighbor_idx[n, i, k] = j means that the kth
//        neighbor to p1[n, i] in the cloud p2[n] is p2[n, j].
//        This is padded with -1s both where a cloud in p2 has fewer than
//        S points and where a cloud in p1 has fewer than P1 points and
//        also if there are fewer than K points which satisfy the radius
//        threshold.
//
//    p1_neighbor_dists: FloatTensor of shape (N, P1, K) containing the squared
//        distance from each point p1[n, p, :] to its K neighbors
//        p2[n, p1_neighbor_idx[n, p, k], :].

// CPU implementation
std::tuple<at::Tensor, at::Tensor> BallQueryCpu(
    const at::Tensor& p1,
    const at::Tensor& p2,
    const at::Tensor& lengths1,
    const at::Tensor& lengths2,
    const int K,
    const float radius);

// CUDA implementation
std::tuple<at::Tensor, at::Tensor> BallQueryCuda(
    const at::Tensor& p1,
    const at::Tensor& p2,
    const at::Tensor& lengths1,
    const at::Tensor& lengths2,
    const int K,
    const float radius);

// Implementation which is exposed
// Note: the backward pass reuses the KNearestNeighborBackward kernel
inline std::tuple<at::Tensor, at::Tensor> BallQuery(
    const at::Tensor& p1,
    const at::Tensor& p2,
    const at::Tensor& lengths1,
    const at::Tensor& lengths2,
    int K,
    float radius) {
  if (p1.is_cuda() || p2.is_cuda()) {
#ifdef WITH_CUDA
    CHECK_CUDA(p1);
    CHECK_CUDA(p2);
    return BallQueryCuda(
        p1.contiguous(),
        p2.contiguous(),
        lengths1.contiguous(),
        lengths2.contiguous(),
        K,
        radius);
#else
    AT_ERROR("Not compiled with GPU support.");
#endif
  }
  return BallQueryCpu(
      p1.contiguous(),
      p2.contiguous(),
      lengths1.contiguous(),
      lengths2.contiguous(),
      K,
      radius);
}
