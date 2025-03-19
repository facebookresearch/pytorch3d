/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <ATen/core/TensorAccessor.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <stdio.h>
#include <vector>

// TODO(gkioxari) support all data types once AtomicAdd supports doubles.
// Currently, support is for floats only.
__global__ void weightedSumCudaForwardKernel(
    // clang-format off
    at::PackedTensorAccessor64<float, 4, at::RestrictPtrTraits> result,
    const at::PackedTensorAccessor64<float, 2, at::RestrictPtrTraits> features,
    const at::PackedTensorAccessor64<float, 4, at::RestrictPtrTraits> alphas,
    const at::PackedTensorAccessor64<int64_t, 4, at::RestrictPtrTraits> points_idx) {
  // clang-format on
  const int64_t C = features.size(0);
  const int64_t H = points_idx.size(2);
  const int64_t W = points_idx.size(3);

  // Get the batch and index
  const auto batch = blockIdx.x;

  const int num_pixels = C * H * W;
  const auto num_threads = gridDim.y * blockDim.x;
  const auto tid = blockIdx.y * blockDim.x + threadIdx.x;

  // Parallelize over each feature in each pixel in images of size H * W,
  // for each image in the batch of size batch_size
  for (int pid = tid; pid < num_pixels; pid += num_threads) {
    int ch = pid / (H * W);
    int j = (pid % (H * W)) / W;
    int i = (pid % (H * W)) % W;

    // Iterate through the closest K points for this pixel
    for (int k = 0; k < points_idx.size(1); ++k) {
      int n_idx = points_idx[batch][k][j][i];
      // Sentinel value is -1 indicating no point overlaps the pixel
      if (n_idx < 0) {
        continue;
      }

      // Accumulate the values
      float alpha = alphas[batch][k][j][i];
      // TODO(gkioxari) It might be more efficient to have threads write in a
      // local variable, and move atomicAdd outside of the loop such that
      // atomicAdd is executed once per thread.
      atomicAdd(&result[batch][ch][j][i], features[ch][n_idx] * alpha);
    }
  }
}

// TODO(gkioxari) support all data types once AtomicAdd supports doubles.
// Currently, support is for floats only.
__global__ void weightedSumCudaBackwardKernel(
    // clang-format off
    at::PackedTensorAccessor64<float, 2, at::RestrictPtrTraits> grad_features,
    at::PackedTensorAccessor64<float, 4, at::RestrictPtrTraits> grad_alphas,
    const at::PackedTensorAccessor64<float, 4, at::RestrictPtrTraits> grad_outputs,
    const at::PackedTensorAccessor64<float, 2, at::RestrictPtrTraits> features,
    const at::PackedTensorAccessor64<float, 4, at::RestrictPtrTraits> alphas,
    const at::PackedTensorAccessor64<int64_t, 4, at::RestrictPtrTraits> points_idx) {
  // clang-format on
  const int64_t C = features.size(0);
  const int64_t H = points_idx.size(2);
  const int64_t W = points_idx.size(3);

  // Get the batch and index
  const auto batch = blockIdx.x;

  const int num_pixels = C * H * W;
  const auto num_threads = gridDim.y * blockDim.x;
  const auto tid = blockIdx.y * blockDim.x + threadIdx.x;

  // Iterate over each pixel to compute the contribution to the
  // gradient for the features and weights
  for (int pid = tid; pid < num_pixels; pid += num_threads) {
    int ch = pid / (H * W);
    int j = (pid % (H * W)) / W;
    int i = (pid % (H * W)) % W;

    // Iterate through the closest K points for this pixel
    for (int k = 0; k < points_idx.size(1); ++k) {
      int n_idx = points_idx[batch][k][j][i];
      // Sentinel value is -1 indicating no point overlaps the pixel
      if (n_idx < 0) {
        continue;
      }
      float alpha = alphas[batch][k][j][i];

      // TODO(gkioxari) It might be more efficient to have threads write in a
      // local variable, and move atomicAdd outside of the loop such that
      // atomicAdd is executed once per thread.
      atomicAdd(
          &grad_alphas[batch][k][j][i],
          features[ch][n_idx] * grad_outputs[batch][ch][j][i]);
      atomicAdd(
          &grad_features[ch][n_idx], alpha * grad_outputs[batch][ch][j][i]);
    }
  }
}

at::Tensor weightedSumCudaForward(
    const at::Tensor& features,
    const at::Tensor& alphas,
    const at::Tensor& points_idx) {
  // Check inputs are on the same device
  at::TensorArg features_t{features, "features", 1},
      alphas_t{alphas, "alphas", 2}, points_idx_t{points_idx, "points_idx", 3};
  at::CheckedFrom c = "weightedSumCudaForward";
  at::checkAllSameGPU(c, {features_t, alphas_t, points_idx_t});
  at::checkAllSameType(c, {features_t, alphas_t});

  // Set the device for the kernel launch based on the device of the input
  at::cuda::CUDAGuard device_guard(features.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int64_t batch_size = points_idx.size(0);
  const int64_t C = features.size(0);
  const int64_t H = points_idx.size(2);
  const int64_t W = points_idx.size(3);

  auto result = at::zeros({batch_size, C, H, W}, features.options());

  if (result.numel() == 0) {
    AT_CUDA_CHECK(cudaGetLastError());
    return result;
  }

  const dim3 threadsPerBlock(64);
  const dim3 numBlocks(batch_size, 1024 / batch_size + 1);

  // TODO(gkioxari) add AT_DISPATCH_FLOATING_TYPES once atomicAdd supports
  // doubles. Currently, support is for floats only.
  weightedSumCudaForwardKernel<<<numBlocks, threadsPerBlock, 0, stream>>>(
      // clang-format off
      // As we are using packed accessors here the tensors
      // do not need to be made contiguous.
      result.packed_accessor64<float, 4, at::RestrictPtrTraits>(),
      features.packed_accessor64<float, 2, at::RestrictPtrTraits>(),
      alphas.packed_accessor64<float, 4, at::RestrictPtrTraits>(),
      points_idx.packed_accessor64<int64_t, 4, at::RestrictPtrTraits>());
  // clang-format on
  AT_CUDA_CHECK(cudaGetLastError());
  return result;
}

std::tuple<at::Tensor, at::Tensor> weightedSumCudaBackward(
    const at::Tensor& grad_outputs,
    const at::Tensor& features,
    const at::Tensor& alphas,
    const at::Tensor& points_idx) {
  // Check inputs are on the same device
  at::TensorArg grad_outputs_t{grad_outputs, "grad_outputs", 1},
      features_t{features, "features", 2}, alphas_t{alphas, "alphas", 3},
      points_idx_t{points_idx, "points_idx", 4};
  at::CheckedFrom c = "weightedSumCudaBackward";
  at::checkAllSameGPU(c, {grad_outputs_t, features_t, alphas_t, points_idx_t});
  at::checkAllSameType(c, {grad_outputs_t, features_t, alphas_t});

  // Set the device for the kernel launch based on the device of the input
  at::cuda::CUDAGuard device_guard(features.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  auto grad_features = at::zeros_like(features);
  auto grad_alphas = at::zeros_like(alphas);

  if (grad_features.numel() == 0 || grad_alphas.numel() == 0) {
    AT_CUDA_CHECK(cudaGetLastError());
    return std::make_tuple(grad_features, grad_alphas);
  }

  const int64_t bs = points_idx.size(0);

  const dim3 threadsPerBlock(64);
  const dim3 numBlocks(bs, 1024 / bs + 1);

  // TODO(gkioxari) add AT_DISPATCH_FLOATING_TYPES once atomicAdd supports
  // doubles. Currently, support is for floats only.
  weightedSumCudaBackwardKernel<<<numBlocks, threadsPerBlock, 0, stream>>>(
      // clang-format off
      // As we are using packed accessors here the tensors
      // do not need to be made contiguous.
      grad_features.packed_accessor64<float, 2, at::RestrictPtrTraits>(),
      grad_alphas.packed_accessor64<float, 4, at::RestrictPtrTraits>(),
      grad_outputs.packed_accessor64<float, 4, at::RestrictPtrTraits>(),
      features.packed_accessor64<float, 2, at::RestrictPtrTraits>(),
      alphas.packed_accessor64<float, 4, at::RestrictPtrTraits>(),
      points_idx.packed_accessor64<int64_t, 4, at::RestrictPtrTraits>());
  // clang-format on
  AT_CUDA_CHECK(cudaGetLastError());
  return std::make_tuple(grad_features, grad_alphas);
}
