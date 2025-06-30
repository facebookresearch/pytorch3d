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

// Perform alpha compositing of points in a z-buffer.
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
//            feature for each point. Concretely, it gives:
//                 weighted_fs[b,c,i,j] = sum_k cum_alpha_k *
//                   features[c,points_idx[b,k,i,j]]
//                 where cum_alpha_k =
//                    alphas[b,k,i,j] * prod_l=0..k-1 (1 - alphas[b,l,i,j])

// CUDA declarations
#ifdef WITH_CUDA
torch::Tensor alphaCompositeCudaForward(
    const torch::Tensor& features,
    const torch::Tensor& alphas,
    const torch::Tensor& points_idx);

std::tuple<torch::Tensor, torch::Tensor> alphaCompositeCudaBackward(
    const torch::Tensor& grad_outputs,
    const torch::Tensor& features,
    const torch::Tensor& alphas,
    const torch::Tensor& points_idx);
#endif

// C++ declarations
torch::Tensor alphaCompositeCpuForward(
    const torch::Tensor& features,
    const torch::Tensor& alphas,
    const torch::Tensor& points_idx);

std::tuple<torch::Tensor, torch::Tensor> alphaCompositeCpuBackward(
    const torch::Tensor& grad_outputs,
    const torch::Tensor& features,
    const torch::Tensor& alphas,
    const torch::Tensor& points_idx);

torch::Tensor alphaCompositeForward(
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
    return alphaCompositeCudaForward(features, alphas, points_idx);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  } else {
    CHECK_CPU(features);
    CHECK_CPU(alphas);
    CHECK_CPU(points_idx);
    return alphaCompositeCpuForward(features, alphas, points_idx);
  }
}

std::tuple<torch::Tensor, torch::Tensor> alphaCompositeBackward(
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

    return alphaCompositeCudaBackward(
        grad_outputs, features, alphas, points_idx);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  } else {
    CHECK_CPU(grad_outputs);
    CHECK_CPU(features);
    CHECK_CPU(alphas);
    CHECK_CPU(points_idx);

    return alphaCompositeCpuBackward(
        grad_outputs, features, alphas, points_idx);
  }
}
