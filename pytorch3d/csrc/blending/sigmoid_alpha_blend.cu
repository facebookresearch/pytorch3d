/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cmath>
#include <vector>

template <typename scalar_t>
__global__ void SigmoidAlphaBlendForwardKernel(
    // clang-format off
    const at::PackedTensorAccessor64<scalar_t, 4, at::RestrictPtrTraits> distances, // (N, H, W, K)
    const at::PackedTensorAccessor64<int64_t, 4, at::RestrictPtrTraits> pix_to_face, // (N, H, W, K)
    at::PackedTensorAccessor64<scalar_t, 3, at::RestrictPtrTraits> alphas, // (N, H, W)
    // clang-format on
    const scalar_t sigma,
    const int N,
    const int H,
    const int W,
    const int K) {
  // Parallelize over each pixel in images of
  // size H * W, for each image in the batch of size N.
  const int num_threads = gridDim.x * blockDim.x;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;

  // TODO: revisit performance of this kernel with shared memory usage

  for (int t_i = tid; t_i < N * H * W; t_i += num_threads) {
    // Convert linear index to 3D index
    const int n = t_i / (H * W); // batch index.
    const int pix_idx = t_i % (H * W);

    // TODO: fix index calculation for non square images.
    const int yi = pix_idx / W;
    const int xi = pix_idx % W;
    scalar_t alpha = 1.0;

    // Loop over all the faces for this pixel.
    for (int k = 0; k < K; k++) {
      // Index into (N, H, W, K) tensors
      const int f = pix_to_face[n][yi][xi][k];
      if (f < 0) {
        // Sentinel value is -1 indicating no face overlaps the pixel.
        continue;
      }
      // The distance is negative if a pixel is inside a face and positive
      // outside the face. Therefore use -1.0 * the distance to get the
      // correct sign.
      scalar_t dist = -1.0 * distances[n][yi][xi][k];

      // Calculate the sigmoid probability.
      scalar_t prob = 1. / (1. + exp(-dist / sigma));

      // The cumulative product ensures that alpha will be 0.0 if at least 1
      // face fully covers the pixel as for that face, prob will be 1.0.
      // This results in a multiplication by 0.0 because of the (1.0 - prob)
      // term. Therefore the final result of (1.0 - alpha) will be 1.0.
      alpha *= (1.0 - prob);
    }
    alphas[n][yi][xi] = 1.0 - alpha;
  }
}

at::Tensor SigmoidAlphaBlendForwardCuda(
    const at::Tensor& distances, // (N, H, W, K)
    const at::Tensor& pix_to_face, // (N, H, W, K)
    const float sigma) {
  const int N = distances.size(0);
  const int H = distances.size(1);
  const int W = distances.size(2);
  const int K = distances.size(3);

  at::Tensor alphas = at::zeros({N, H, W}, distances.options());
  const size_t blocks = 1024;
  const size_t threads = 128;

  // Check inputs are on the same device
  at::TensorArg distances_t{distances, "distances", 1},
      pix_to_face_t{pix_to_face, "pix_to_face", 2};
  at::CheckedFrom c = "SigmoidAlphaBlendForwardCuda";
  at::checkAllSameGPU(c, {distances_t, pix_to_face_t});

  // Set the device for the kernel launch based on the device of distances
  at::cuda::CUDAGuard device_guard(distances.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (distances.numel() == 0) {
    AT_CUDA_CHECK(cudaGetLastError());
    return alphas;
  }

  AT_DISPATCH_FLOATING_TYPES(
      distances.scalar_type(), "sigmoid_alpha_blend_kernel", ([&] {
        // clang-format off
      SigmoidAlphaBlendForwardKernel<scalar_t><<<blocks, threads, 0, stream>>>(
      distances.packed_accessor64<scalar_t, 4, at::RestrictPtrTraits>(),
      pix_to_face.packed_accessor64<int64_t, 4, at::RestrictPtrTraits>(),
      alphas.packed_accessor64<scalar_t, 3, at::RestrictPtrTraits>(),
      sigma,
      N,
      H,
      W,
      K);
        // clang-format on
      }));

  AT_CUDA_CHECK(cudaGetLastError());
  return alphas;
}

template <typename scalar_t>
__global__ void SigmoidAlphaBlendBackwardKernel(
    // clang-format off
    const at::PackedTensorAccessor64<scalar_t, 3, at::RestrictPtrTraits> grad_alphas, // (N, H, W)
    const at::PackedTensorAccessor64<scalar_t, 3, at::RestrictPtrTraits> alphas, // (N, H, W)
    const at::PackedTensorAccessor64<scalar_t, 4, at::RestrictPtrTraits> distances, // (N, H, W, K)
    const at::PackedTensorAccessor64<int64_t, 4, at::RestrictPtrTraits> pix_to_face, // (N, H, W, K)
    at::PackedTensorAccessor64<scalar_t, 4, at::RestrictPtrTraits> grad_distances, // (N, H, W)
    // clang-format on
    const scalar_t sigma,
    const int N,
    const int H,
    const int W,
    const int K) {
  // Parallelize over each of the top K faces for each pixel in images of
  // size H * W * K, for each image in the batch of size N.

  // Get block and thread index.
  const int n = blockIdx.x;
  const int num_pixels = H * W * K;
  const int num_threads = gridDim.y * blockDim.x;
  const int tid = blockIdx.y * blockDim.x + threadIdx.x;

  for (int t_i = tid; t_i < num_pixels; t_i += num_threads) {
    // Convert linear index to 3D index.
    int yi = t_i / (W * K);
    int xi = (t_i % (W * K)) / K;
    int k = (t_i % (W * K)) % K;

    const scalar_t alpha = 1.0 - alphas[n][yi][xi];
    const scalar_t grad_alpha = grad_alphas[n][yi][xi];
    const int f = pix_to_face[n][yi][xi][k];

    // Sentinel value is -1 indicating no face overlaps the pixel.
    if (f >= 0) {
      // The distance is negative if a pixel is inside a face and positive
      // outside the face. Therefore use -1.0 * the distance to get the
      // correct sign.
      scalar_t dist = -1.0 * distances[n][yi][xi][k];

      // Calculate the sigmoid probability.
      scalar_t prob = 1. / (1. + exp(-dist / sigma));

      grad_distances[n][yi][xi][k] = grad_alpha * (-1.0 / sigma) * prob * alpha;
    }
  }
}

at::Tensor SigmoidAlphaBlendBackwardCuda(
    const at::Tensor& grad_alphas, // (N, H, W)
    const at::Tensor& alphas, // (N, H, W)
    const at::Tensor& distances, // (N, H, W, K)
    const at::Tensor& pix_to_face, // (N, H, W, K)
    float sigma) {
  const int N = distances.size(0);
  const int H = distances.size(1);
  const int W = distances.size(2);
  const int K = distances.size(3);

  at::Tensor grad_distances = at::zeros({N, H, W, K}, distances.options());

  const dim3 threads(512);
  const dim3 blocks(N, 1024 / N + 1);

  at::TensorArg grad_alphas_t{grad_alphas, "grad_alphas", 1},
      alphas_t{alphas, "alphas", 2}, distances_t{distances, "distances", 3},
      pix_to_face_t{pix_to_face, "pix_to_face", 4};
  at::CheckedFrom c = "SigmoidAlphaBlendBackwardCuda";
  at::checkAllSameGPU(c, {grad_alphas_t, alphas_t, distances_t, pix_to_face_t});

  // Set the device for the kernel launch based on the device of distances
  at::cuda::CUDAGuard device_guard(alphas.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (alphas.numel() == 0) {
    AT_CUDA_CHECK(cudaGetLastError());
    return grad_alphas;
  }

  AT_DISPATCH_FLOATING_TYPES(
      distances.scalar_type(), "sigmoid_alpha_blend_backward_kernel", ([&] {
        SigmoidAlphaBlendBackwardKernel<
            scalar_t><<<blocks, threads, 0, stream>>>(
            // clang-format off
            grad_alphas.packed_accessor64<scalar_t, 3,at::RestrictPtrTraits>(),
            alphas.packed_accessor64<scalar_t, 3, at::RestrictPtrTraits>(),
            distances.packed_accessor64<scalar_t, 4, at::RestrictPtrTraits>(),
            pix_to_face.packed_accessor64<int64_t, 4, at::RestrictPtrTraits>(),
            grad_distances.packed_accessor64<scalar_t, 4, at::RestrictPtrTraits>(),
            // clang-format on
            sigma,
            N,
            H,
            W,
            K);
      }));

  AT_CUDA_CHECK(cudaGetLastError());
  return grad_distances;
}
