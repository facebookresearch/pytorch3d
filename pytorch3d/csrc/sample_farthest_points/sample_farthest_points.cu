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
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "utils/warp_reduce.cuh"

template <unsigned int block_size>
__global__ void FarthestPointSamplingKernel(
    // clang-format off
    const at::PackedTensorAccessor64<float, 3, at::RestrictPtrTraits> points,
    const at::PackedTensorAccessor64<int64_t, 1, at::RestrictPtrTraits> lengths,
    const at::PackedTensorAccessor64<int64_t, 1, at::RestrictPtrTraits> K,
    at::PackedTensorAccessor64<int64_t, 2, at::RestrictPtrTraits> idxs,
    at::PackedTensorAccessor64<float, 2, at::RestrictPtrTraits> min_point_dist,
    const at::PackedTensorAccessor64<int64_t, 1, at::RestrictPtrTraits> start_idxs
    // clang-format on
) {
  // Get constants
  const int64_t N = points.size(0);
  const int64_t P = points.size(1);
  const int64_t D = points.size(2);

  // Create single shared memory buffer which is split and cast to different
  // types: dists/dists_idx are used to save the maximum distances seen by the
  // points processed by any one thread and the associated point indices.
  // These values only need to be accessed by other threads in this block which
  // are processing the same batch and not by other blocks.
  extern __shared__ char shared_buf[];
  float* dists = (float*)shared_buf; // block_size floats
  int64_t* dists_idx = (int64_t*)&dists[block_size]; // block_size int64_t

  // Get batch index and thread index
  const int64_t batch_idx = blockIdx.x;
  const size_t tid = threadIdx.x;

  // If K is greater than the number of points in the pointcloud
  // we only need to iterate until the smaller value is reached.
  const int64_t k_n = min(K[batch_idx], lengths[batch_idx]);

  // Write the first selected point to global memory in the first thread
  int64_t selected = start_idxs[batch_idx];
  if (tid == 0)
    idxs[batch_idx][0] = selected;

  // Iterate to find k_n sampled points
  for (int64_t k = 1; k < k_n; ++k) {
    // Keep track of the maximum of the minimum distance to previously selected
    // points seen by this thread
    int64_t max_dist_idx = 0;
    float max_dist = -1.0;

    // Iterate through all the points in this pointcloud. For already selected
    // points, the minimum distance to the set of previously selected points
    // will be 0.0 so they won't be selected again.
    for (int64_t p = tid; p < lengths[batch_idx]; p += block_size) {
      // Calculate the distance to the last selected point
      float dist2 = 0.0;
      for (int64_t d = 0; d < D; ++d) {
        float diff = points[batch_idx][selected][d] - points[batch_idx][p][d];
        dist2 += (diff * diff);
      }

      // If the distance of point p to the last selected point is
      // less than the previous minimum distance of p to the set of selected
      // points, then updated the corresponding value in min_point_dist
      // so it always contains the min distance.
      const float p_min_dist = min(dist2, min_point_dist[batch_idx][p]);
      min_point_dist[batch_idx][p] = p_min_dist;

      // Update the max distance and point idx for this thread.
      max_dist_idx = (p_min_dist > max_dist) ? p : max_dist_idx;
      max_dist = (p_min_dist > max_dist) ? p_min_dist : max_dist;
    }

    // After going through all points for this thread, save the max
    // point and idx seen by this thread. Each thread sees P/block_size points.
    dists[tid] = max_dist;
    dists_idx[tid] = max_dist_idx;
    // Sync to ensure all threads in the block have updated their max point.
    __syncthreads();

    // Parallelized block reduction to find the max point seen by
    // all the threads in this block for iteration k.
    // Each block represents one batch element so we can use a divide/conquer
    // approach to find the max, syncing all threads after each step.

    for (int s = block_size / 2; s > 0; s >>= 1) {
      if (tid < s) {
        // Compare the best point seen by two threads and update the shared
        // memory at the location of the first thread index with the max out
        // of the two threads.
        if (dists[tid] < dists[tid + s]) {
          dists[tid] = dists[tid + s];
          dists_idx[tid] = dists_idx[tid + s];
        }
      }
      __syncthreads();
    }

    // TODO(nikhilar): As reduction proceeds, the number of “active” threads
    // decreases. When tid < 32, there should only be one warp left which could
    // be unrolled.

    // The overall max after reducing will be saved
    // at the location of tid = 0.
    selected = dists_idx[0];

    if (tid == 0) {
      // Write the farthest point for iteration k to global memory
      idxs[batch_idx][k] = selected;
    }
  }
}

at::Tensor FarthestPointSamplingCuda(
    const at::Tensor& points, // (N, P, 3)
    const at::Tensor& lengths, // (N,)
    const at::Tensor& K, // (N,)
    const at::Tensor& start_idxs) {
  // Check inputs are on the same device
  at::TensorArg p_t{points, "points", 1}, lengths_t{lengths, "lengths", 2},
      k_t{K, "K", 3}, start_idxs_t{start_idxs, "start_idxs", 4};
  at::CheckedFrom c = "FarthestPointSamplingCuda";
  at::checkAllSameGPU(c, {p_t, lengths_t, k_t, start_idxs_t});
  at::checkAllSameType(c, {lengths_t, k_t, start_idxs_t});

  // Set the device for the kernel launch based on the device of points
  at::cuda::CUDAGuard device_guard(points.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  TORCH_CHECK(
      points.size(0) == lengths.size(0),
      "Point and lengths must have the same batch dimension");

  TORCH_CHECK(
      points.size(0) == K.size(0),
      "Points and K must have the same batch dimension");

  const int64_t N = points.size(0);
  const int64_t P = points.size(1);
  const int64_t max_K = at::max(K).item<int64_t>();

  // Initialize the output tensor with the sampled indices
  auto idxs = at::full({N, max_K}, -1, lengths.options());
  auto min_point_dist = at::full({N, P}, 1e10, points.options());

  if (N == 0 || P == 0) {
    AT_CUDA_CHECK(cudaGetLastError());
    return idxs;
  }

  // Set the number of blocks to the batch size so that the
  // block reduction step can be done for each pointcloud
  // to find the max distance point in the pointcloud at each iteration.
  const size_t blocks = N;

  // Set the threads to the nearest power of 2 of the number of
  // points in the pointcloud (up to the max threads in a block).
  // This will ensure each thread processes the minimum necessary number of
  // points (P/threads).
  const int points_pow_2 = std::log(static_cast<double>(P)) / std::log(2.0);

  // Max possible threads per block
  const int MAX_THREADS_PER_BLOCK = 1024;
  const size_t threads = max(min(1 << points_pow_2, MAX_THREADS_PER_BLOCK), 1);

  // Create the accessors
  auto points_a = points.packed_accessor64<float, 3, at::RestrictPtrTraits>();
  auto lengths_a =
      lengths.packed_accessor64<int64_t, 1, at::RestrictPtrTraits>();
  auto K_a = K.packed_accessor64<int64_t, 1, at::RestrictPtrTraits>();
  auto idxs_a = idxs.packed_accessor64<int64_t, 2, at::RestrictPtrTraits>();
  auto start_idxs_a =
      start_idxs.packed_accessor64<int64_t, 1, at::RestrictPtrTraits>();
  auto min_point_dist_a =
      min_point_dist.packed_accessor64<float, 2, at::RestrictPtrTraits>();

  // Initialize the shared memory which will be used to store the
  // distance/index of the best point seen by each thread.
  size_t shared_mem = threads * sizeof(float) + threads * sizeof(int64_t);
  // TODO: using shared memory for min_point_dist gives an ~2x speed up
  // compared to using a global (N, P) shaped tensor, however for
  // larger pointclouds this may exceed the shared memory limit per block.
  // If a speed up is required for smaller pointclouds, then the storage
  // could be switched to shared memory if the required total shared memory is
  // within the memory limit per block.

  // Support a case for all powers of 2 up to MAX_THREADS_PER_BLOCK possible per
  // block.
  switch (threads) {
    case 1024:
      FarthestPointSamplingKernel<1024>
          <<<blocks, threads, shared_mem, stream>>>(
              points_a, lengths_a, K_a, idxs_a, min_point_dist_a, start_idxs_a);
      break;
    case 512:
      FarthestPointSamplingKernel<512><<<blocks, threads, shared_mem, stream>>>(
          points_a, lengths_a, K_a, idxs_a, min_point_dist_a, start_idxs_a);
      break;
    case 256:
      FarthestPointSamplingKernel<256><<<blocks, threads, shared_mem, stream>>>(
          points_a, lengths_a, K_a, idxs_a, min_point_dist_a, start_idxs_a);
      break;
    case 128:
      FarthestPointSamplingKernel<128><<<blocks, threads, shared_mem, stream>>>(
          points_a, lengths_a, K_a, idxs_a, min_point_dist_a, start_idxs_a);
      break;
    case 64:
      FarthestPointSamplingKernel<64><<<blocks, threads, shared_mem, stream>>>(
          points_a, lengths_a, K_a, idxs_a, min_point_dist_a, start_idxs_a);
      break;
    case 32:
      FarthestPointSamplingKernel<32><<<blocks, threads, shared_mem, stream>>>(
          points_a, lengths_a, K_a, idxs_a, min_point_dist_a, start_idxs_a);
      break;
    case 16:
      FarthestPointSamplingKernel<16><<<blocks, threads, shared_mem, stream>>>(
          points_a, lengths_a, K_a, idxs_a, min_point_dist_a, start_idxs_a);
      break;
    case 8:
      FarthestPointSamplingKernel<8><<<blocks, threads, shared_mem, stream>>>(
          points_a, lengths_a, K_a, idxs_a, min_point_dist_a, start_idxs_a);
      break;
    case 4:
      FarthestPointSamplingKernel<4><<<threads, threads, shared_mem, stream>>>(
          points_a, lengths_a, K_a, idxs_a, min_point_dist_a, start_idxs_a);
      break;
    case 2:
      FarthestPointSamplingKernel<2><<<threads, threads, shared_mem, stream>>>(
          points_a, lengths_a, K_a, idxs_a, min_point_dist_a, start_idxs_a);
      break;
    case 1:
      FarthestPointSamplingKernel<1><<<threads, threads, shared_mem, stream>>>(
          points_a, lengths_a, K_a, idxs_a, min_point_dist_a, start_idxs_a);
      break;
    default:
      FarthestPointSamplingKernel<1024>
          <<<blocks, threads, shared_mem, stream>>>(
              points_a, lengths_a, K_a, idxs_a, min_point_dist_a, start_idxs_a);
  }

  AT_CUDA_CHECK(cudaGetLastError());
  return idxs;
}
