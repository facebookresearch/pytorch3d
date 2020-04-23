// Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

#include <ATen/ATen.h>
#include <ATen/core/TensorAccessor.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <stdio.h>
#include <vector>

__constant__ const float kEpsilon = 1e-4;

// TODO(gkioxari) support all data types once AtomicAdd supports doubles.
// Currently, support is for floats only.
__global__ void weightedSumNormCudaForwardKernel(
    // clang-format off
    at::PackedTensorAccessor64<float, 4, at::RestrictPtrTraits> result,
    const at::PackedTensorAccessor64<float, 2, at::RestrictPtrTraits> features,
    const at::PackedTensorAccessor64<float, 4, at::RestrictPtrTraits> alphas,
    const at::PackedTensorAccessor64<int64_t, 4, at::RestrictPtrTraits> points_idx) {
  // clang-format on
  const int64_t batch_size = result.size(0);
  const int64_t C = features.size(0);
  const int64_t H = points_idx.size(2);
  const int64_t W = points_idx.size(3);

  // Get the batch and index
  const int batch = blockIdx.x;

  const int num_pixels = C * W * H;
  const int num_threads = gridDim.y * blockDim.x;
  const int tid = blockIdx.y * blockDim.x + threadIdx.x;

  // Parallelize over each feature in each pixel in images of size H * W,
  // for each image in the batch of size batch_size
  for (int pid = tid; pid < num_pixels; pid += num_threads) {
    int ch = pid / (W * H);
    int j = (pid % (W * H)) / H;
    int i = (pid % (W * H)) % H;

    // Store the accumulated alpha value
    float cum_alpha = 0.;
    // Iterate through the closest K points for this pixel
    for (int k = 0; k < points_idx.size(1); ++k) {
      int n_idx = points_idx[batch][k][j][i];
      // Sentinel value is -1 indicating no point overlaps the pixel
      if (n_idx < 0) {
        continue;
      }

      cum_alpha += alphas[batch][k][j][i];
    }

    if (cum_alpha < kEpsilon) {
      cum_alpha = kEpsilon;
    }

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
          &result[batch][ch][j][i], features[ch][n_idx] * alpha / cum_alpha);
    }
  }
}

// TODO(gkioxari) support all data types once AtomicAdd supports doubles.
// Currently, support is for floats only.
__global__ void weightedSumNormCudaBackwardKernel(
    // clang-format off
    at::PackedTensorAccessor64<float, 2, at::RestrictPtrTraits> grad_features,
    at::PackedTensorAccessor64<float, 4, at::RestrictPtrTraits> grad_alphas,
    const at::PackedTensorAccessor64<float, 4, at::RestrictPtrTraits> grad_outputs,
    const at::PackedTensorAccessor64<float, 2, at::RestrictPtrTraits> features,
    const at::PackedTensorAccessor64<float, 4, at::RestrictPtrTraits> alphas,
    const at::PackedTensorAccessor64<int64_t, 4, at::RestrictPtrTraits> points_idx) {
  // clang-format on
  const int64_t batch_size = points_idx.size(0);
  const int64_t C = features.size(0);
  const int64_t H = points_idx.size(2);
  const int64_t W = points_idx.size(3);

  // Get the batch and index
  const int batch = blockIdx.x;

  const int num_pixels = C * W * H;
  const int num_threads = gridDim.y * blockDim.x;
  const int tid = blockIdx.y * blockDim.x + threadIdx.x;

  // Parallelize over each feature in each pixel in images of size H * W,
  // for each image in the batch of size batch_size
  for (int pid = tid; pid < num_pixels; pid += num_threads) {
    int ch = pid / (W * H);
    int j = (pid % (W * H)) / H;
    int i = (pid % (W * H)) % H;

    float sum_alpha = 0.;
    float sum_alphafs = 0.;
    // Iterate through the closest K points for this pixel to calculate the
    // cumulative sum of the alphas for this pixel
    for (int k = 0; k < points_idx.size(1); ++k) {
      int n_idx = points_idx[batch][k][j][i];
      // Sentinel value is -1 indicating no point overlaps the pixel
      if (n_idx < 0) {
        continue;
      }

      sum_alpha += alphas[batch][k][j][i];
      sum_alphafs += alphas[batch][k][j][i] * features[ch][n_idx];
    }

    if (sum_alpha < kEpsilon) {
      sum_alpha = kEpsilon;
    }

    // Iterate again through the closest K points for this pixel to calculate
    // the gradient.
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
          (features[ch][n_idx] * sum_alpha - sum_alphafs) /
              (sum_alpha * sum_alpha) * grad_outputs[batch][ch][j][i]);
      atomicAdd(
          &grad_features[ch][n_idx],
          alpha * grad_outputs[batch][ch][j][i] / sum_alpha);
    }
  }
}

at::Tensor weightedSumNormCudaForward(
    const at::Tensor& features,
    const at::Tensor& alphas,
    const at::Tensor& points_idx) {
  const int64_t batch_size = points_idx.size(0);
  const int64_t C = features.size(0);
  const int64_t H = points_idx.size(2);
  const int64_t W = points_idx.size(3);

  auto result = at::zeros({batch_size, C, H, W}, features.options());

  const dim3 threadsPerBlock(64);
  const dim3 numBlocks(batch_size, 1024 / batch_size + 1);

  // TODO(gkioxari) add AT_DISPATCH_FLOATING_TYPES once atomicAdd supports
  // doubles. Currently, support is for floats only.
  // clang-format off
  weightedSumNormCudaForwardKernel<<<numBlocks, threadsPerBlock>>>(
      result.packed_accessor64<float, 4, at::RestrictPtrTraits>(),
      features.packed_accessor64<float, 2, at::RestrictPtrTraits>(),
      alphas.packed_accessor64<float, 4, at::RestrictPtrTraits>(),
      points_idx.packed_accessor64<int64_t, 4, at::RestrictPtrTraits>());
  // clang-format on

  return result;
}

std::tuple<at::Tensor, at::Tensor> weightedSumNormCudaBackward(
    const at::Tensor& grad_outputs,
    const at::Tensor& features,
    const at::Tensor& alphas,
    const at::Tensor& points_idx) {
  auto grad_features = at::zeros_like(features);
  auto grad_alphas = at::zeros_like(alphas);

  const int64_t bs = points_idx.size(0);

  const dim3 threadsPerBlock(64);
  const dim3 numBlocks(bs, 1024 / bs + 1);

  // TODO(gkioxari) add AT_DISPATCH_FLOATING_TYPES once atomicAdd supports
  // doubles. Currently, support is for floats only.
  weightedSumNormCudaBackwardKernel<<<numBlocks, threadsPerBlock>>>(
      // clang-format off
      grad_features.packed_accessor64<float, 2, at::RestrictPtrTraits>(),
      grad_alphas.packed_accessor64<float, 4, at::RestrictPtrTraits>(),
      grad_outputs.packed_accessor64<float, 4, at::RestrictPtrTraits>(),
      features.packed_accessor64<float, 2, at::RestrictPtrTraits>(),
      alphas.packed_accessor64<float, 4, at::RestrictPtrTraits>(),
      points_idx.packed_accessor64<int64_t, 4, at::RestrictPtrTraits>());
  // clang-format on

  return std::make_tuple(grad_features, grad_alphas);
}
