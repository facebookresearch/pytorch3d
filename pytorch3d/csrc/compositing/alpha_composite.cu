// Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

#include <ATen/ATen.h>
#include <ATen/core/TensorAccessor.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <stdio.h>
#include <vector>

// TODO(gkioxari) support all data types once AtomicAdd supports doubles.
// Currently, support is for floats only.
__global__ void alphaCompositeCudaForwardKernel(
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

  // Iterate over each feature in each pixel
  for (int pid = tid; pid < num_pixels; pid += num_threads) {
    int ch = pid / (W * H);
    int j = (pid % (W * H)) / H;
    int i = (pid % (W * H)) % H;

    // alphacomposite the different values
    float cum_alpha = 1.;
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
          &result[batch][ch][j][i], features[ch][n_idx] * cum_alpha * alpha);
      cum_alpha = cum_alpha * (1 - alpha);
    }
  }
}

// TODO(gkioxari) support all data types once AtomicAdd supports doubles.
// Currently, support is for floats only.
__global__ void alphaCompositeCudaBackwardKernel(
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

    // alphacomposite the different values
    float cum_alpha = 1.;
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
          cum_alpha * features[ch][n_idx] * grad_outputs[batch][ch][j][i]);
      atomicAdd(
          &grad_features[ch][n_idx],
          cum_alpha * alpha * grad_outputs[batch][ch][j][i]);

      // Iterate over all (K-1) nearest points to update gradient
      for (int t = 0; t < k; ++t) {
        int t_idx = points_idx[batch][t][j][i];
        // Sentinel value is -1, indicating no point overlaps this pixel
        if (t_idx < 0) {
          continue;
        }
        float alpha_tvalue = alphas[batch][t][j][i];
        // TODO(gkioxari) It might be more efficient to have threads write in a
        // local variable, and move atomicAdd outside of the loop such that
        // atomicAdd is executed once per thread.
        atomicAdd(
            &grad_alphas[batch][t][j][i],
            -grad_outputs[batch][ch][j][i] * features[ch][n_idx] * cum_alpha *
                alpha / (1 - alpha_tvalue));
      }

      cum_alpha = cum_alpha * (1 - alphas[batch][k][j][i]);
    }
  }
}

at::Tensor alphaCompositeCudaForward(
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
  alphaCompositeCudaForwardKernel<<<numBlocks, threadsPerBlock>>>(
      // clang-format off
      result.packed_accessor64<float, 4, at::RestrictPtrTraits>(),
      features.packed_accessor64<float, 2, at::RestrictPtrTraits>(),
      alphas.packed_accessor64<float, 4, at::RestrictPtrTraits>(),
      points_idx.packed_accessor64<int64_t, 4, at::RestrictPtrTraits>());
  // clang-format on

  return result;
}

std::tuple<at::Tensor, at::Tensor> alphaCompositeCudaBackward(
    const at::Tensor& grad_outputs,
    const at::Tensor& features,
    const at::Tensor& alphas,
    const at::Tensor& points_idx) {
  auto grad_features = at::zeros_like(features);
  auto grad_alphas = at::zeros_like(alphas);

  const int64_t bs = alphas.size(0);

  const dim3 threadsPerBlock(64);
  const dim3 numBlocks(bs, 1024 / bs + 1);

  // TODO(gkioxari) add AT_DISPATCH_FLOATING_TYPES once atomicAdd supports
  // doubles. Currently, support is for floats only.
  alphaCompositeCudaBackwardKernel<<<numBlocks, threadsPerBlock>>>(
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
