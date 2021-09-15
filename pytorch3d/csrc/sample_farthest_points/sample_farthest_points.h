/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <torch/extension.h>
#include <tuple>
#include "utils/pytorch3d_cutils.h"

// Iterative farthest point sampling algorithm [1] to subsample a set of
// K points from a given pointcloud. At each iteration, a point is selected
// which has the largest nearest neighbor distance to any of the
// already selected points.

// Farthest point sampling provides more uniform coverage of the input
// point cloud compared to uniform random sampling.

// [1] Charles R. Qi et al, "PointNet++: Deep Hierarchical Feature Learning
//     on Point Sets in a Metric Space", NeurIPS 2017.

// Args:
//     points: (N, P, D) float32 Tensor containing the batch of pointclouds.
//     lengths: (N,) long Tensor giving the number of points in each pointcloud
//        (to support heterogeneous batches of pointclouds).
//     K: a tensor of length (N,) giving the number of
//        samples to select for each element in the batch.
//        The number of samples is typically << P.
//     random_start_point: bool, if True, a random point is selected as the
//        starting point for iterative sampling.
// Returns:
//     selected_indices: (N, K) array of selected indices. If the values in
//        K are not all the same, then the shape will be (N, max(K), D), and
//        padded with -1 for batch elements where k_i < max(K). The selected
//        points are gathered in the pytorch autograd wrapper.

at::Tensor FarthestPointSamplingCpu(
    const at::Tensor& points,
    const at::Tensor& lengths,
    const at::Tensor& K,
    const bool random_start_point);

// Exposed implementation.
at::Tensor FarthestPointSampling(
    const at::Tensor& points,
    const at::Tensor& lengths,
    const at::Tensor& K,
    const bool random_start_point) {
  if (points.is_cuda() || lengths.is_cuda() || K.is_cuda()) {
    AT_ERROR("CUDA implementation not yet supported");
  }
  return FarthestPointSamplingCpu(points, lengths, K, random_start_point);
}
