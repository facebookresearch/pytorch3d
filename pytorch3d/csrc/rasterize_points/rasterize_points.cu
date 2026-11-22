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
#include <cstdio>
#include <sstream>
#include <tuple>
#include "rasterize_points/rasterization_utils.cuh"

namespace {
// A little structure for holding details about a pixel.
struct Pix {
  float z; // Depth of the reference point.
  int32_t idx; // Index of the reference point.
  float dist2; // Euclidean distance square to the reference point.
};

__device__ inline bool operator<(const Pix& a, const Pix& b) {
  return a.z < b.z;
}

// This function checks if a pixel given by xy location pxy lies within the
// point with index p and batch index n. One of the inputs is a list (q)
// which contains Pixel structs with the indices of the points which intersect
// with this pixel sorted by closest z distance. If the pixel pxy lies in the
// point, the list (q) is updated and re-orderered in place. In addition
// the auxiliary variables q_size, q_max_z and q_max_idx are also modified.
// This code is shared between RasterizePointsNaiveCudaKernel and
// RasterizePointsFineCudaKernel.
template <typename PointQ>
__device__ void CheckPixelInsidePoint(
    const float* points, // (P, 3)
    const int p_idx,
    int& q_size,
    float& q_max_z,
    int& q_max_idx,
    PointQ& q,
    const float* radius,
    const float xf,
    const float yf,
    const int K) {
  const float px = points[p_idx * 3 + 0];
  const float py = points[p_idx * 3 + 1];
  const float pz = points[p_idx * 3 + 2];
  const float p_radius = radius[p_idx];
  const float radius2 = p_radius * p_radius;
  if (pz < 0)
    return; // Don't render points behind the camera
  const float dx = xf - px;
  const float dy = yf - py;
  const float dist2 = dx * dx + dy * dy;
  if (dist2 < radius2) {
    if (q_size < K) {
      // Just insert it
      q[q_size] = {pz, p_idx, dist2};
      if (pz > q_max_z) {
        q_max_z = pz;
        q_max_idx = q_size;
      }
      q_size++;
    } else if (pz < q_max_z) {
      // Overwrite the old max, and find the new max
      q[q_max_idx] = {pz, p_idx, dist2};
      q_max_z = pz;
      for (int i = 0; i < K; i++) {
        if (q[i].z > q_max_z) {
          q_max_z = q[i].z;
          q_max_idx = i;
        }
      }
    }
  }
}
} // namespace
// ****************************************************************************
// *                          NAIVE RASTERIZATION                             *
// ****************************************************************************

__global__ void RasterizePointsNaiveCudaKernel(
    const float* points, // (P, 3)
    const int64_t* cloud_to_packed_first_idx, // (N)
    const int64_t* num_points_per_cloud, // (N)
    const float* radius,
    const int N,
    const int H,
    const int W,
    const int K,
    int32_t* point_idxs, // (N, H, W, K)
    float* zbuf, // (N, H, W, K)
    float* pix_dists) { // (N, H, W, K)
  // Simple version: One thread per output pixel
  const int num_threads = gridDim.x * blockDim.x;
  const int tid = blockDim.x * blockIdx.x + threadIdx.x;
  for (int i = tid; i < N * H * W; i += num_threads) {
    // Convert linear index to 3D index
    const int n = i / (H * W); // Batch index
    const int pix_idx = i % (H * W);

    // Reverse ordering of the X and Y axis as the camera coordinates
    // assume that +Y is pointing up and +X is pointing left.
    const int yi = H - 1 - pix_idx / W;
    const int xi = W - 1 - pix_idx % W;

    // screen coordinates to ndc coordinates of pixel.
    const float xf = PixToNonSquareNdc(xi, W, H);
    const float yf = PixToNonSquareNdc(yi, H, W);

    // For keeping track of the K closest points we want a data structure
    // that (1) gives O(1) access to the closest point for easy comparisons,
    // and (2) allows insertion of new elements. In the CPU version we use
    // std::priority_queue; then (2) is O(log K). We can't use STL
    // containers in CUDA; we could roll our own max heap in an array, but
    // that would likely have a lot of warp divergence so we do something
    // simpler instead: keep the elements in an unsorted array, but keep
    // track of the max value and the index of the max value. Then (1) is
    // still O(1) time, while (2) is O(K) with a clean loop. Since K <= 8
    // this should be fast enough for our purposes.
    // TODO(jcjohns) Abstract this out into a standalone data structure
    Pix q[kMaxPointsPerPixel];
    int q_size = 0;
    float q_max_z = -1000;
    int q_max_idx = -1;

    // Using the batch index of the thread get the start and stop
    // indices for the points.
    const int64_t point_start_idx = cloud_to_packed_first_idx[n];
    const int64_t point_stop_idx = point_start_idx + num_points_per_cloud[n];

    for (int p_idx = point_start_idx; p_idx < point_stop_idx; ++p_idx) {
      CheckPixelInsidePoint(
          points, p_idx, q_size, q_max_z, q_max_idx, q, radius, xf, yf, K);
    }
    BubbleSort(q, q_size);
    int idx = n * H * W * K + pix_idx * K;
    for (int k = 0; k < q_size; ++k) {
      point_idxs[idx + k] = q[k].idx;
      zbuf[idx + k] = q[k].z;
      pix_dists[idx + k] = q[k].dist2;
    }
  }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> RasterizePointsNaiveCuda(
    const at::Tensor& points, // (P. 3)
    const at::Tensor& cloud_to_packed_first_idx, // (N)
    const at::Tensor& num_points_per_cloud, // (N)
    const std::tuple<int, int> image_size,
    const at::Tensor& radius,
    const int points_per_pixel) {
  // Check inputs are on the same device
  at::TensorArg points_t{points, "points", 1},
      cloud_to_packed_first_idx_t{
          cloud_to_packed_first_idx, "cloud_to_packed_first_idx", 2},
      num_points_per_cloud_t{num_points_per_cloud, "num_points_per_cloud", 3};
  at::CheckedFrom c = "RasterizePointsNaiveCuda";
  at::checkAllSameGPU(
      c, {points_t, cloud_to_packed_first_idx_t, num_points_per_cloud_t});

  // Set the device for the kernel launch based on the device of the input
  at::cuda::CUDAGuard device_guard(points.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  TORCH_CHECK(
      points.ndimension() == 2 && points.size(1) == 3,
      "points must have dimensions (num_points, 3)");
  TORCH_CHECK(
      num_points_per_cloud.size(0) == cloud_to_packed_first_idx.size(0),
      "num_points_per_cloud must have same size first dimension as cloud_to_packed_first_idx");

  const int N = num_points_per_cloud.size(0); // batch size.
  const int H = std::get<0>(image_size);
  const int W = std::get<1>(image_size);
  const int K = points_per_pixel;

  if (K > kMaxPointsPerPixel) {
    std::stringstream ss;
    ss << "Must have points_per_pixel <= " << kMaxPointsPerPixel;
    AT_ERROR(ss.str());
  }

  auto int_opts = num_points_per_cloud.options().dtype(at::kInt);
  auto float_opts = points.options().dtype(at::kFloat);
  at::Tensor point_idxs = at::full({N, H, W, K}, -1, int_opts);
  at::Tensor zbuf = at::full({N, H, W, K}, -1, float_opts);
  at::Tensor pix_dists = at::full({N, H, W, K}, -1, float_opts);

  if (point_idxs.numel() == 0) {
    AT_CUDA_CHECK(cudaGetLastError());
    return std::make_tuple(point_idxs, zbuf, pix_dists);
  }

  const size_t blocks = 1024;
  const size_t threads = 64;
  RasterizePointsNaiveCudaKernel<<<blocks, threads, 0, stream>>>(
      points.contiguous().data_ptr<float>(),
      cloud_to_packed_first_idx.contiguous().data_ptr<int64_t>(),
      num_points_per_cloud.contiguous().data_ptr<int64_t>(),
      radius.contiguous().data_ptr<float>(),
      N,
      H,
      W,
      K,
      point_idxs.contiguous().data_ptr<int32_t>(),
      zbuf.contiguous().data_ptr<float>(),
      pix_dists.contiguous().data_ptr<float>());

  AT_CUDA_CHECK(cudaGetLastError());
  return std::make_tuple(point_idxs, zbuf, pix_dists);
}

// ****************************************************************************
// *                            FINE RASTERIZATION                            *
// ****************************************************************************

__global__ void RasterizePointsFineCudaKernel(
    const float* points, // (P, 3)
    const int32_t* bin_points, // (N, BH, BW, T)
    const float* radius,
    const int bin_size,
    const int N,
    const int BH, // num_bins y
    const int BW, // num_bins x
    const int M,
    const int H,
    const int W,
    const int K,
    int32_t* point_idxs, // (N, H, W, K)
    float* zbuf, // (N, H, W, K)
    float* pix_dists) { // (N, H, W, K)
  // This can be more than H * W if H or W are not divisible by bin_size.
  const int num_pixels = N * BH * BW * bin_size * bin_size;
  const int num_threads = gridDim.x * blockDim.x;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;

  for (int pid = tid; pid < num_pixels; pid += num_threads) {
    // Convert linear index into bin and pixel indices. We make the within
    // block pixel ids move the fastest, so that adjacent threads will fall
    // into the same bin; this should give them coalesced memory reads when
    // they read from points and bin_points.
    int i = pid;
    const int n = i / (BH * BW * bin_size * bin_size);
    i %= BH * BW * bin_size * bin_size;
    const int by = i / (BW * bin_size * bin_size);
    i %= BW * bin_size * bin_size;
    const int bx = i / (bin_size * bin_size);
    i %= bin_size * bin_size;

    const int yi = i / bin_size + by * bin_size;
    const int xi = i % bin_size + bx * bin_size;

    if (yi >= H || xi >= W)
      continue;

    const float xf = PixToNonSquareNdc(xi, W, H);
    const float yf = PixToNonSquareNdc(yi, H, W);

    // This part looks like the naive rasterization kernel, except we use
    // bin_points to only look at a subset of points already known to fall
    // in this bin. TODO abstract out this logic into some data structure
    // that is shared by both kernels?
    Pix q[kMaxPointsPerPixel];
    int q_size = 0;
    float q_max_z = -1000;
    int q_max_idx = -1;
    for (int m = 0; m < M; ++m) {
      const int p = bin_points[n * BH * BW * M + by * BW * M + bx * M + m];
      if (p < 0) {
        // bin_points uses -1 as a sentinal value
        continue;
      }
      CheckPixelInsidePoint(
          points, p, q_size, q_max_z, q_max_idx, q, radius, xf, yf, K);
    }
    // Now we've looked at all the points for this bin, so we can write
    // output for the current pixel.
    BubbleSort(q, q_size);

    // Reverse ordering of the X and Y axis as the camera coordinates
    // assume that +Y is pointing up and +X is pointing left.
    const int yidx = H - 1 - yi;
    const int xidx = W - 1 - xi;

    const int pix_idx = n * H * W * K + yidx * W * K + xidx * K;
    for (int k = 0; k < q_size; ++k) {
      point_idxs[pix_idx + k] = q[k].idx;
      zbuf[pix_idx + k] = q[k].z;
      pix_dists[pix_idx + k] = q[k].dist2;
    }
  }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> RasterizePointsFineCuda(
    const at::Tensor& points, // (P, 3)
    const at::Tensor& bin_points,
    const std::tuple<int, int> image_size,
    const at::Tensor& radius,
    const int bin_size,
    const int points_per_pixel) {
  // Check inputs are on the same device
  at::TensorArg points_t{points, "points", 1},
      bin_points_t{bin_points, "bin_points", 2};
  at::CheckedFrom c = "RasterizePointsFineCuda";
  at::checkAllSameGPU(c, {points_t, bin_points_t});

  // Set the device for the kernel launch based on the device of the input
  at::cuda::CUDAGuard device_guard(points.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int N = bin_points.size(0);
  const int BH = bin_points.size(1);
  const int BW = bin_points.size(2);
  const int M = bin_points.size(3);
  const int K = points_per_pixel;

  const int H = std::get<0>(image_size);
  const int W = std::get<1>(image_size);

  if (K > kMaxPointsPerPixel) {
    AT_ERROR("Must have num_closest <= 150");
  }
  auto int_opts = bin_points.options().dtype(at::kInt);
  auto float_opts = points.options().dtype(at::kFloat);
  at::Tensor point_idxs = at::full({N, H, W, K}, -1, int_opts);
  at::Tensor zbuf = at::full({N, H, W, K}, -1, float_opts);
  at::Tensor pix_dists = at::full({N, H, W, K}, -1, float_opts);

  if (point_idxs.numel() == 0) {
    AT_CUDA_CHECK(cudaGetLastError());
    return std::make_tuple(point_idxs, zbuf, pix_dists);
  }

  const size_t blocks = 1024;
  const size_t threads = 64;
  RasterizePointsFineCudaKernel<<<blocks, threads, 0, stream>>>(
      points.contiguous().data_ptr<float>(),
      bin_points.contiguous().data_ptr<int32_t>(),
      radius.contiguous().data_ptr<float>(),
      bin_size,
      N,
      BH,
      BW,
      M,
      H,
      W,
      K,
      point_idxs.contiguous().data_ptr<int32_t>(),
      zbuf.contiguous().data_ptr<float>(),
      pix_dists.contiguous().data_ptr<float>());

  AT_CUDA_CHECK(cudaGetLastError());
  return std::make_tuple(point_idxs, zbuf, pix_dists);
}

// ****************************************************************************
// *                            BACKWARD PASS                                 *
// ****************************************************************************
// TODO(T55115174) Add more documentation for backward kernel.
__global__ void RasterizePointsBackwardCudaKernel(
    const float* points, // (P, 3)
    const int32_t* idxs, // (N, H, W, K)
    const int N,
    const int P,
    const int H,
    const int W,
    const int K,
    const float* grad_zbuf, // (N, H, W, K)
    const float* grad_dists, // (N, H, W, K)
    float* grad_points) { // (P, 3)
  // Parallelized over each of K points per pixel, for each pixel in images of
  // size H * W, for each image in the batch of size N.
  int num_threads = gridDim.x * blockDim.x;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = tid; i < N * H * W * K; i += num_threads) {
    // const int n = i / (H * W * K); // batch index (not needed).
    const int yxk = i % (H * W * K);
    const int yi = yxk / (W * K);
    const int xk = yxk % (W * K);
    const int xi = xk / K;
    // k = xk % K (We don't actually need k, but this would be it.)
    // Reverse ordering of X and Y axes.
    const int yidx = H - 1 - yi;
    const int xidx = W - 1 - xi;

    const float xf = PixToNonSquareNdc(xidx, W, H);
    const float yf = PixToNonSquareNdc(yidx, H, W);

    const int p = idxs[i];
    if (p < 0)
      continue;
    const float grad_dist2 = grad_dists[i];
    const int p_ind = p * 3; // index into packed points tensor
    const float px = points[p_ind + 0];
    const float py = points[p_ind + 1];
    const float dx = px - xf;
    const float dy = py - yf;
    const float grad_px = 2.0f * grad_dist2 * dx;
    const float grad_py = 2.0f * grad_dist2 * dy;
    const float grad_pz = grad_zbuf[i];
    atomicAdd(grad_points + p_ind + 0, grad_px);
    atomicAdd(grad_points + p_ind + 1, grad_py);
    atomicAdd(grad_points + p_ind + 2, grad_pz);
  }
}

at::Tensor RasterizePointsBackwardCuda(
    const at::Tensor& points, // (N, P, 3)
    const at::Tensor& idxs, // (N, H, W, K)
    const at::Tensor& grad_zbuf, // (N, H, W, K)
    const at::Tensor& grad_dists) { // (N, H, W, K)

  // Check inputs are on the same device
  at::TensorArg points_t{points, "points", 1}, idxs_t{idxs, "idxs", 2},
      grad_zbuf_t{grad_zbuf, "grad_zbuf", 3},
      grad_dists_t{grad_dists, "grad_dists", 4};
  at::CheckedFrom c = "RasterizePointsBackwardCuda";
  at::checkAllSameGPU(c, {points_t, idxs_t, grad_zbuf_t, grad_dists_t});
  at::checkAllSameType(c, {points_t, grad_zbuf_t, grad_dists_t});
  // This is nondeterministic because atomicAdd
  at::globalContext().alertNotDeterministic("RasterizePointsBackwardCuda");
  // Set the device for the kernel launch based on the device of the input
  at::cuda::CUDAGuard device_guard(points.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int P = points.size(0);
  const int N = idxs.size(0);
  const int H = idxs.size(1);
  const int W = idxs.size(2);
  const int K = idxs.size(3);

  at::Tensor grad_points = at::zeros({P, 3}, points.options());

  if (grad_points.numel() == 0) {
    AT_CUDA_CHECK(cudaGetLastError());
    return grad_points;
  }

  const size_t blocks = 1024;
  const size_t threads = 64;

  RasterizePointsBackwardCudaKernel<<<blocks, threads, 0, stream>>>(
      points.contiguous().data_ptr<float>(),
      idxs.contiguous().data_ptr<int32_t>(),
      N,
      P,
      H,
      W,
      K,
      grad_zbuf.contiguous().data_ptr<float>(),
      grad_dists.contiguous().data_ptr<float>(),
      grad_points.contiguous().data_ptr<float>());

  AT_CUDA_CHECK(cudaGetLastError());
  return grad_points;
}
