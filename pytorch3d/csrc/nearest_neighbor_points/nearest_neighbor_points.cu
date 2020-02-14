// Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

#include <ATen/ATen.h>
#include <float.h>

template <typename scalar_t>
__device__ void WarpReduce(
    volatile scalar_t* min_dists,
    volatile long* min_idxs,
    const size_t tid) {
  // s = 32
  if (min_dists[tid] > min_dists[tid + 32]) {
    min_idxs[tid] = min_idxs[tid + 32];
    min_dists[tid] = min_dists[tid + 32];
  }
  // s = 16
  if (min_dists[tid] > min_dists[tid + 16]) {
    min_idxs[tid] = min_idxs[tid + 16];
    min_dists[tid] = min_dists[tid + 16];
  }
  // s = 8
  if (min_dists[tid] > min_dists[tid + 8]) {
    min_idxs[tid] = min_idxs[tid + 8];
    min_dists[tid] = min_dists[tid + 8];
  }
  // s = 4
  if (min_dists[tid] > min_dists[tid + 4]) {
    min_idxs[tid] = min_idxs[tid + 4];
    min_dists[tid] = min_dists[tid + 4];
  }
  // s = 2
  if (min_dists[tid] > min_dists[tid + 2]) {
    min_idxs[tid] = min_idxs[tid + 2];
    min_dists[tid] = min_dists[tid + 2];
  }
  // s = 1
  if (min_dists[tid] > min_dists[tid + 1]) {
    min_idxs[tid] = min_idxs[tid + 1];
    min_dists[tid] = min_dists[tid + 1];
  }
}

//  CUDA kernel to compute nearest neighbors between two batches of pointclouds
//  where each point is of dimension D.
//
//  Args:
//    points1: First set of points, of shape (N, P1, D).
//    points2: Second set of points, of shape (N, P2, D).
//    idx: Output memory buffer of shape (N, P1).
//    N: Batch size.
//    P1: Number of points in points1.
//    P2: Number of points in points2.
//    D_2: Size of the shared buffer; this is D rounded up so that memory access
//         is aligned.
//
template <typename scalar_t>
__global__ void NearestNeighborKernel(
    const scalar_t* __restrict__ points1,
    const scalar_t* __restrict__ points2,
    long* __restrict__ idx,
    const size_t N,
    const size_t P1,
    const size_t P2,
    const size_t D,
    const size_t D_2) {
  // Each block will compute one element of the output idx[n, i]. Within the
  // block we will use threads to compute the distances between points1[n, i]
  // and points2[n, j] for all 0 <= j < P2, then use a block reduction to
  // take an argmin of the distances.

  // Shared buffers for the threads in the block. CUDA only allows declaration
  // of a single shared buffer, so it needs to be manually sliced and cast to
  // build several logical shared buffers of different types.
  extern __shared__ char shared_buf[];
  scalar_t* x = (scalar_t*)shared_buf; // scalar_t[DD]
  scalar_t* min_dists = &x[D_2]; // scalar_t[NUM_THREADS]
  long* min_idxs = (long*)&min_dists[blockDim.x]; // long[NUM_THREADS]

  const size_t n = blockIdx.y; // index of batch element.
  const size_t i = blockIdx.x; // index of point within batch element.
  const size_t tid = threadIdx.x;

  // Thread 0 copies points1[n, i, :] into x.
  if (tid == 0) {
    for (size_t d = 0; d < D; d++) {
      x[d] = points1[n * (P1 * D) + i * D + d];
    }
  }
  __syncthreads();

  // Compute the distances between points1[n, i] and points2[n, j] for
  // all 0 <= j < P2. Here each thread will reduce over P2 / blockDim.x
  // in serial, and store its result to shared memory
  scalar_t min_dist = FLT_MAX;
  size_t min_idx = 0;
  for (size_t j = tid; j < P2; j += blockDim.x) {
    scalar_t dist = 0;
    for (size_t d = 0; d < D; d++) {
      scalar_t x_d = x[d];
      scalar_t y_d = points2[n * (P2 * D) + j * D + d];
      scalar_t diff = x_d - y_d;
      dist += diff * diff;
    }
    min_dist = (j == tid) ? dist : min_dist;
    min_idx = (dist <= min_dist) ? j : min_idx;
    min_dist = (dist <= min_dist) ? dist : min_dist;
  }
  min_dists[tid] = min_dist;
  min_idxs[tid] = min_idx;
  __syncthreads();

  // Perform reduction in shared memory.
  for (int s = blockDim.x / 2; s > 32; s >>= 1) {
    if (tid < s) {
      if (min_dists[tid] > min_dists[tid + s]) {
        min_dists[tid] = min_dists[tid + s];
        min_idxs[tid] = min_idxs[tid + s];
      }
    }
    __syncthreads();
  }

  // Unroll the last 6 iterations of the loop since they will happen
  // synchronized within a single warp.
  if (tid < 32)
    WarpReduce<scalar_t>(min_dists, min_idxs, tid);

  // Finally thread 0 writes the result to the output buffer.
  if (tid == 0) {
    idx[n * P1 + i] = min_idxs[0];
  }
}

//  CUDA kernel to compute nearest neighbors between two sets of 3-dimensional
//  pointclouds. This is a specialization of the nearest_neighbor_kernel
//  to the case D=3.
//
//  Args:
//    points1: First set of pointclouds, of shape (N, P1, 3).
//    points2: Second set of pointclouds, of shape (N, P2, 3).
//    idx: Output memory buffer of shape (N, P1).
//    N: Batch size.
//    P1: Number of points in points1.
//    P2: Number of points in points2.
//
template <typename scalar_t>
__global__ void NearestNeighborKernelD3(
    const scalar_t* __restrict__ points1,
    const scalar_t* __restrict__ points2,
    long* __restrict__ idx,
    const size_t N,
    const size_t P1,
    const size_t P2) {
  // Single shared memory buffer which is split and cast to different types.
  extern __shared__ char shared_buf[];
  scalar_t* min_dists = (scalar_t*)shared_buf; // scalar_t[NUM_THREADS]
  long* min_idxs = (long*)&min_dists[blockDim.x]; // long[NUM_THREADS]

  const size_t D = 3;
  const size_t n = blockIdx.y; // index of batch element.
  const size_t i = blockIdx.x; // index of point within batch element.
  const size_t tid = threadIdx.x;

  // Retrieve the coordinates of points1[n, i] from global memory; these
  // will be stored in registers for fast access.
  const scalar_t x = points1[n * (P1 * D) + i * D + 0];
  const scalar_t y = points1[n * (P1 * D) + i * D + 1];
  const scalar_t z = points1[n * (P1 * D) + i * D + 2];

  // Compute distances between points1[n, i] and all points2[n, j]
  // for 0 <= j < P2
  scalar_t min_dist = FLT_MAX;
  size_t min_idx = 0;

  // Distance computation for points in p2 spread across threads in the block.
  for (size_t j = tid; j < P2; j += blockDim.x) {
    scalar_t dx = x - points2[n * (P2 * D) + j * D + 0];
    scalar_t dy = y - points2[n * (P2 * D) + j * D + 1];
    scalar_t dz = z - points2[n * (P2 * D) + j * D + 2];
    scalar_t dist = dx * dx + dy * dy + dz * dz;
    min_dist = (j == tid) ? dist : min_dist;
    min_idx = (dist <= min_dist) ? j : min_idx;
    min_dist = (dist <= min_dist) ? dist : min_dist;
  }
  min_dists[tid] = min_dist;
  min_idxs[tid] = min_idx;

  // Synchronize local threads writing to the shared memory buffer.
  __syncthreads();

  // Perform reduction in shared memory.
  for (int s = blockDim.x / 2; s > 32; s >>= 1) {
    if (tid < s) {
      if (min_dists[tid] > min_dists[tid + s]) {
        min_dists[tid] = min_dists[tid + s];
        min_idxs[tid] = min_idxs[tid + s];
      }
    }

    // Synchronize local threads so that min_dists is correct.
    __syncthreads();
  }

  // Unroll the last 6 iterations of the loop since they will happen
  // synchronized within a single warp.
  if (tid < 32)
    WarpReduce<scalar_t>(min_dists, min_idxs, tid);

  // Finally thread 0 writes the result to the output buffer.
  if (tid == 0) {
    idx[n * P1 + i] = min_idxs[0];
  }
}

at::Tensor NearestNeighborIdxCuda(const at::Tensor& p1, const at::Tensor& p2) {
  const auto N = p1.size(0);
  const auto P1 = p1.size(1);
  const auto P2 = p2.size(1);
  const auto D = p1.size(2);

  AT_ASSERTM(p2.size(2) == D, "Point sets must have same last dimension.");
  auto idx = at::empty({N, P1}, p1.options().dtype(at::kLong));

  // On P100 with pointclouds of size (16, 5000, 3), 128 threads per block
  // gives best results.
  const int threads = 128;
  const dim3 blocks(P1, N);

  if (D == 3) {
    // Use the specialized kernel for D=3.
    AT_DISPATCH_FLOATING_TYPES(p1.type(), "nearest_neighbor_v3_cuda", ([&] {
                                 size_t shared_size = threads * sizeof(size_t) +
                                     threads * sizeof(long);
                                 NearestNeighborKernelD3<scalar_t>
                                     <<<blocks, threads, shared_size>>>(
                                         p1.data_ptr<scalar_t>(),
                                         p2.data_ptr<scalar_t>(),
                                         idx.data_ptr<long>(),
                                         N,
                                         P1,
                                         P2);
                               }));
  } else {
    // Use the general kernel for all other D.
    AT_DISPATCH_FLOATING_TYPES(
        p1.type(), "nearest_neighbor_v3_cuda", ([&] {
          // To avoid misaligned memory access, the size of shared buffers
          // need to be rounded to the next even size.
          size_t D_2 = D + (D % 2);
          size_t shared_size = (D_2 + threads) * sizeof(size_t);
          shared_size += threads * sizeof(long);
          NearestNeighborKernel<scalar_t><<<blocks, threads, shared_size>>>(
              p1.data_ptr<scalar_t>(),
              p2.data_ptr<scalar_t>(),
              idx.data_ptr<long>(),
              N,
              P1,
              P2,
              D,
              D_2);
        }));
  }

  return idx;
}
