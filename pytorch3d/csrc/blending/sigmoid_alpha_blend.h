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

// clang-format off
// Function to blend the top K faces per pixel based on the 2d euclidean distance
// from the center of the pixel to the face. This method is adapted from [1].
// The output can be used to set the alpha value in an RGBA image.
// Args:
//      pix_to_face: LongTensor of shape (N, H, W, K), indices of faces overlapping
//          with each pixel, where N is the batch size, H, W are the dimensions of the
//          image and K is the number of faces rasterized per pixel.
//      distances: FloatTensor of shape (N, H, W, K), 2d euclidean distance of each pixel
//          relative to the faces in pix_to_face
//      sigma: float, parameter which controls the width of the sigmoid for blending
// Returns:
//      alphas: FloatTensor of shape (N, H, W), the blended values for each pixel
//          in the image.
//
// [1] Shichen Liu et al, 'Soft Rasterizer: A Differentiable Renderer for
// Image-based 3D Reasoning'
// clang-format on
at::Tensor SigmoidAlphaBlendForwardCpu(
    const at::Tensor& distances,
    const at::Tensor& pix_to_face,
    const float sigma);

#ifdef WITH_CUDA
at::Tensor SigmoidAlphaBlendForwardCuda(
    const at::Tensor& distances,
    const at::Tensor& pix_to_face,
    const float sigma);
#endif

// clang-format off
// Args:
//      grad_alphas: FloatTensor of shape (N, H, W), upstream gradients for alphas
//      alphas: FloatTensor of shape (N, H, W), the alpha values from the forward pass
//      pix_to_face: LongTensor of shape (N, H, W, K), indices of faces overlapping
//          with each pixel, where N is the batch size, H, W are the dimensions of the
//          image, and K is the number of faces rasterized per pixel
//      distances: FloatTensor of shape (N, H, W, K), 2d euclidean distance of each pixel
//          to the corresponding faces in pix_to_face
//      sigma: float, parameter which controls the width of the sigmoid for blending
// Returns:
//      grad_distances: FloatTensor of shape (N, H, W, K)
// clang-format on
at::Tensor SigmoidAlphaBlendBackwardCpu(
    const at::Tensor& grad_alphas,
    const at::Tensor& alphas,
    const at::Tensor& distances,
    const at::Tensor& pix_to_face,
    const float sigma);

#ifdef WITH_CUDA
at::Tensor SigmoidAlphaBlendBackwardCuda(
    const at::Tensor& grad_alphas,
    const at::Tensor& alphas,
    const at::Tensor& distances,
    const at::Tensor& pix_to_face,
    const float sigma);
#endif

// Implementation which is exposed.
at::Tensor
SigmoidAlphaBlend(at::Tensor& distances, at::Tensor& pix_to_face, float sigma) {
  if (distances.is_cuda() && pix_to_face.is_cuda()) {
#ifdef WITH_CUDA
    return SigmoidAlphaBlendForwardCuda(distances, pix_to_face, sigma);
#else
    AT_ERROR("Not compiled with GPU support.");
#endif
  }
  return SigmoidAlphaBlendForwardCpu(distances, pix_to_face, sigma);
}

// Implementation which is exposed.
at::Tensor SigmoidAlphaBlendBackward(
    const at::Tensor& grad_alphas,
    const at::Tensor& alphas,
    const at::Tensor& distances,
    const at::Tensor& pix_to_face,
    const float sigma) {
  if (distances.is_cuda() && pix_to_face.is_cuda() && alphas.is_cuda() &&
      grad_alphas.is_cuda()) {
#ifdef WITH_CUDA
    return SigmoidAlphaBlendBackwardCuda(
        grad_alphas, alphas, distances, pix_to_face, sigma);
#else
    AT_ERROR("Not compiled with GPU support.");
#endif
  }
  return SigmoidAlphaBlendBackwardCpu(
      grad_alphas, alphas, distances, pix_to_face, sigma);
}
