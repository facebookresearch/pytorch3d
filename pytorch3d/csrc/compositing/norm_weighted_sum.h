/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <torch/extension.h>
#include "utils/pytorch3d_cutils.h"

#include <vector>

// Perform normalized weighted sum compositing of points in a z-buffer.
//
// Inputs:
//    features: FloatTensor of shape (C, P) which gives the features
//            of each point where C is the size of the feature and
//            P the number of points.
//    alphas: FloatTensor of shape (N, points_per_pixel, H, W) where
//            points_per_pixel is the number of points in the z-buffer
//            sorted in z-order, and (H, W) is the image size.
//    points_idx: IntTensor of shape (N, points_per_pixel, H, W) giving the
//            indices of the nearest points at each pixel, sorted in z-order.
// Returns:
//    weighted_fs: FloatTensor of shape (N, C, H, W) giving the accumulated
//            feature in each point. Concretely, it gives:
//                 weighted_fs[b,c,i,j] = sum_k alphas[b,k,i,j] *
//                   features[c,points_idx[b,k,i,j]] / sum_k alphas[b,k,i,j]

// CUDA declarations
#ifdef WITH_CUDA
torch::Tensor weightedSumNormCudaForward(
    const torch::Tensor& features,
    const torch::Tensor& alphas,
    const torch::Tensor& points_idx);

std::tuple<torch::Tensor, torch::Tensor> weightedSumNormCudaBackward(
    const torch::Tensor& grad_outputs,
    const torch::Tensor& features,
    const torch::Tensor& alphas,
    const torch::Tensor& points_idx);
#endif

// C++ declarations
torch::Tensor weightedSumNormCpuForward(
    const torch::Tensor& features,
    const torch::Tensor& alphas,
    const torch::Tensor& points_idx);

std::tuple<torch::Tensor, torch::Tensor> weightedSumNormCpuBackward(
    const torch::Tensor& grad_outputs,
    const torch::Tensor& features,
    const torch::Tensor& alphas,
    const torch::Tensor& points_idx);

torch::Tensor weightedSumNormForward(
    torch::Tensor& features,
    torch::Tensor& alphas,
    torch::Tensor& points_idx) {
  features = features.contiguous();
  alphas = alphas.contiguous();
  points_idx = points_idx.contiguous();

  if (features.is_cuda()) {
#ifdef WITH_CUDA
    CHECK_CUDA(features);
    CHECK_CUDA(alphas);
    CHECK_CUDA(points_idx);

    return weightedSumNormCudaForward(features, alphas, points_idx);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  } else {
    return weightedSumNormCpuForward(features, alphas, points_idx);
  }
}

std::tuple<torch::Tensor, torch::Tensor> weightedSumNormBackward(
    torch::Tensor& grad_outputs,
    torch::Tensor& features,
    torch::Tensor& alphas,
    torch::Tensor& points_idx) {
  grad_outputs = grad_outputs.contiguous();
  features = features.contiguous();
  alphas = alphas.contiguous();
  points_idx = points_idx.contiguous();

  if (grad_outputs.is_cuda()) {
#ifdef WITH_CUDA
    CHECK_CUDA(grad_outputs);
    CHECK_CUDA(features);
    CHECK_CUDA(alphas);
    CHECK_CUDA(points_idx);

    return weightedSumNormCudaBackward(
        grad_outputs, features, alphas, points_idx);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  } else {
    return weightedSumNormCpuBackward(
        grad_outputs, features, alphas, points_idx);
  }
}
