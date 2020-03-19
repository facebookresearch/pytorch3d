// Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <stdio.h>
#include <vector>

// TODO(gkioxari) support all data types once AtomicAdd supports doubles.
// Currently, support is for floats only.
__global__ void weightedSumCudaForwardKernel(
    // clang-format off
    torch::PackedTensorAccessor<float, 4, torch::RestrictPtrTraits, size_t> result,
    const torch::PackedTensorAccessor<float, 2, torch::RestrictPtrTraits, size_t> features,
    const torch::PackedTensorAccessor<float, 4, torch::RestrictPtrTraits, size_t> alphas,
    const torch::PackedTensorAccessor<int64_t, 4, torch::RestrictPtrTraits, size_t> points_idx) {
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
    torch::PackedTensorAccessor<float, 2, torch::RestrictPtrTraits, size_t> grad_features,
    torch::PackedTensorAccessor<float, 4, torch::RestrictPtrTraits, size_t> grad_alphas,
    const torch::PackedTensorAccessor<float, 4, torch::RestrictPtrTraits, size_t> grad_outputs,
    const torch::PackedTensorAccessor<float, 2, torch::RestrictPtrTraits, size_t> features,
    const torch::PackedTensorAccessor<float, 4, torch::RestrictPtrTraits, size_t> alphas,
    const torch::PackedTensorAccessor<int64_t, 4, torch::RestrictPtrTraits, size_t> points_idx) {
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

  // Iterate over each pixel to compute the contribution to the
  // gradient for the features and weights
  for (int pid = tid; pid < num_pixels; pid += num_threads) {
    int ch = pid / (W * H);
    int j = (pid % (W * H)) / H;
    int i = (pid % (W * H)) % H;

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

torch::Tensor weightedSumCudaForward(
    const torch::Tensor& features,
    const torch::Tensor& alphas,
    const torch::Tensor& points_idx) {
  const int64_t batch_size = points_idx.size(0);
  const int64_t C = features.size(0);
  const int64_t H = points_idx.size(2);
  const int64_t W = points_idx.size(3);

  auto result = torch::zeros({batch_size, C, H, W}, features.options());

  const dim3 threadsPerBlock(64);
  const dim3 numBlocks(batch_size, 1024 / batch_size + 1);

  // TODO(gkioxari) add AT_DISPATCH_FLOATING_TYPES once atomicAdd supports
  // doubles. Currently, support is for floats only.
  weightedSumCudaForwardKernel<<<numBlocks, threadsPerBlock>>>(
      // clang-format off
      result.packed_accessor<float, 4, torch::RestrictPtrTraits, size_t>(),
      features.packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>(),
      alphas.packed_accessor<float, 4, torch::RestrictPtrTraits, size_t>(),
      points_idx.packed_accessor<int64_t, 4, torch::RestrictPtrTraits, size_t>());
  // clang-format on

  return result;
}

std::tuple<torch::Tensor, torch::Tensor> weightedSumCudaBackward(
    const torch::Tensor& grad_outputs,
    const torch::Tensor& features,
    const torch::Tensor& alphas,
    const torch::Tensor& points_idx) {
  auto grad_features = torch::zeros_like(features);
  auto grad_alphas = torch::zeros_like(alphas);

  const int64_t bs = points_idx.size(0);

  const dim3 threadsPerBlock(64);
  const dim3 numBlocks(bs, 1024 / bs + 1);

  // TODO(gkioxari) add AT_DISPATCH_FLOATING_TYPES once atomicAdd supports
  // doubles. Currently, support is for floats only.
  weightedSumCudaBackwardKernel<<<numBlocks, threadsPerBlock>>>(
      // clang-format off
      grad_features.packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>(),
      grad_alphas.packed_accessor<float, 4, torch::RestrictPtrTraits, size_t>(),
      grad_outputs.packed_accessor<float, 4, torch::RestrictPtrTraits, size_t>(),
      features.packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>(),
      alphas.packed_accessor<float, 4, torch::RestrictPtrTraits, size_t>(),
      points_idx.packed_accessor<int64_t, 4, torch::RestrictPtrTraits, size_t>());
  // clang-format on

  return std::make_tuple(grad_features, grad_alphas);
}
