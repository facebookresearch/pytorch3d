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
#include <algorithm>
#include <list>
#include <queue>
#include <tuple>
#include "utils/float_math.cuh"
#include "utils/geometry_utils.cuh"
#include "utils/warp_reduce.cuh"

// ****************************************************************************
// *                   Generic Forward/Backward Kernels                       *
// ****************************************************************************

__global__ void DistanceForwardKernel(
    const float* __restrict__ objects, // (O * oD * 3)
    const size_t objects_size, // O
    const size_t objects_dim, // oD
    const float* __restrict__ targets, // (T * tD * 3)
    const size_t targets_size, // T
    const size_t targets_dim, // tD
    const int64_t* __restrict__ objects_first_idx, // (B,)
    const int64_t* __restrict__ targets_first_idx, // (B,)
    const size_t batch_size, // B
    float* __restrict__ dist_objects, // (O,)
    int64_t* __restrict__ idx_objects, // (O,)
    const double min_triangle_area) {
  // This kernel is used interchangeably to compute bi-directional distances
  // between points and triangles/lines. The direction of the distance computed,
  // i.e. point to triangle/line or triangle/line to point, depends on the order
  // of the input arguments and is inferred based on their shape. The shape is
  // used to distinguish between triangles and lines

  // Single shared memory buffer which is split and cast to different types.
  extern __shared__ char shared_buf[];
  float* min_dists = (float*)shared_buf; // float[NUM_THREADS]
  int64_t* min_idxs = (int64_t*)&min_dists[blockDim.x]; // int64_t[NUM_THREADS]

  const size_t batch_idx = blockIdx.y; // index of batch element.

  // start and end for objects in batch_idx
  const int64_t starto = objects_first_idx[batch_idx];
  const int64_t endo = batch_idx + 1 < batch_size
      ? objects_first_idx[batch_idx + 1]
      : objects_size;

  // start and end for targets in batch_idx
  const int64_t startt = targets_first_idx[batch_idx];
  const int64_t endt = batch_idx + 1 < batch_size
      ? targets_first_idx[batch_idx + 1]
      : targets_size;

  const size_t i = blockIdx.x; // index within batch element.
  const size_t tid = threadIdx.x; // thread index

  // Set references to points/face based on which of objects/targets refer to
  // points/faces
  float3* points_f3 = objects_dim == 1 ? (float3*)objects : (float3*)targets;
  float3* face_f3 = objects_dim == 1 ? (float3*)targets : (float3*)objects;
  // Distinguishes whether we're computing distance against triangle vs edge
  bool isTriangle = objects_dim == 3 || targets_dim == 3;

  // Each block will compute one element of the output idx_objects[starto + i],
  // dist_objects[starto + i]. Within the block we will use threads to compute
  // the distances between objects[starto + i] and targets[j] for all j
  // belonging in the same batch as i, i.e. j in [startt, endt]. Then use a
  // block reduction to take an argmin of the distances.

  // If i exceeds the number of objects in batch_idx, then do nothing
  if (i < (endo - starto)) {
    // Compute the distances between objects[starto + i] and targets[j] for
    // all j belonging in the same batch as i, i.e. j in [startt, endt].
    // Here each thread will reduce over (endt-startt) / blockDim.x in serial,
    // and store its result to shared memory
    float min_dist = FLT_MAX;
    size_t min_idx = 0;
    for (size_t j = tid; j < (endt - startt); j += blockDim.x) {
      size_t point_idx = objects_dim == 1 ? starto + i : startt + j;
      size_t face_idx = objects_dim == 1 ? (startt + j) * targets_dim
                                         : (starto + i) * objects_dim;

      float dist;
      if (isTriangle) {
        dist = PointTriangle3DistanceForward(
            points_f3[point_idx],
            face_f3[face_idx],
            face_f3[face_idx + 1],
            face_f3[face_idx + 2],
            min_triangle_area);
      } else {
        dist = PointLine3DistanceForward(
            points_f3[point_idx], face_f3[face_idx], face_f3[face_idx + 1]);
      }

      min_dist = (j == tid) ? dist : min_dist;
      min_idx = (dist <= min_dist) ? (startt + j) : min_idx;
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
      WarpReduceMin<float>(min_dists, min_idxs, tid);

    // Finally thread 0 writes the result to the output buffer.
    if (tid == 0) {
      idx_objects[starto + i] = min_idxs[0];
      dist_objects[starto + i] = min_dists[0];
    }
  }
}

std::tuple<at::Tensor, at::Tensor> DistanceForwardCuda(
    const at::Tensor& objects,
    const size_t objects_dim,
    const at::Tensor& objects_first_idx,
    const at::Tensor& targets,
    const size_t targets_dim,
    const at::Tensor& targets_first_idx,
    const int64_t max_objects,
    const double min_triangle_area) {
  // Check inputs are on the same device
  at::TensorArg objects_t{objects, "objects", 1},
      objects_first_idx_t{objects_first_idx, "objects_first_idx", 2},
      targets_t{targets, "targets", 3},
      targets_first_idx_t{targets_first_idx, "targets_first_idx", 4};
  at::CheckedFrom c = "DistanceForwardCuda";
  at::checkAllSameGPU(
      c, {objects_t, objects_first_idx_t, targets_t, targets_first_idx_t});
  at::checkAllSameType(c, {objects_t, targets_t});

  // Set the device for the kernel launch based on the device of the input
  at::cuda::CUDAGuard device_guard(objects.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int64_t objects_size = objects.size(0);
  const int64_t targets_size = targets.size(0);
  const int64_t batch_size = objects_first_idx.size(0);

  TORCH_CHECK(targets_first_idx.size(0) == batch_size);
  if (objects_dim == 1) {
    TORCH_CHECK(
        targets_dim >= 2 && targets_dim <= 3,
        "either object or target must be edge or face");
    TORCH_CHECK(objects.size(1) == 3, "points must be of shape Px3");
    TORCH_CHECK(
        targets.size(2) == 3,
        "face must be of shape Tx3x3, lines must be of shape Tx2x3");
  } else {
    TORCH_CHECK(targets_dim == 1, "either object or target must be point");
    TORCH_CHECK(
        objects_dim >= 2 && objects_dim <= 3,
        "either object or target must be edge or face");
    TORCH_CHECK(targets.size(1) == 3, "points must be of shape Px3");
    TORCH_CHECK(
        objects.size(2) == 3,
        "face must be of shape Tx3x3, lines must be of shape Tx2x3");
  }

  // clang-format off
  at::Tensor dists = at::zeros({objects_size,}, objects.options());
  at::Tensor idxs = at::zeros({objects_size,}, objects_first_idx.options());
  // clang-format on

  if (dists.numel() == 0) {
    AT_CUDA_CHECK(cudaGetLastError());
    return std::make_tuple(dists, idxs);
  }

  const int threads = 128;
  const dim3 blocks(max_objects, batch_size);
  size_t shared_size = threads * sizeof(size_t) + threads * sizeof(int64_t);

  DistanceForwardKernel<<<blocks, threads, shared_size, stream>>>(
      objects.contiguous().data_ptr<float>(),
      objects_size,
      objects_dim,
      targets.contiguous().data_ptr<float>(),
      targets_size,
      targets_dim,
      objects_first_idx.contiguous().data_ptr<int64_t>(),
      targets_first_idx.contiguous().data_ptr<int64_t>(),
      batch_size,
      dists.data_ptr<float>(),
      idxs.data_ptr<int64_t>(),
      min_triangle_area);

  AT_CUDA_CHECK(cudaGetLastError());
  return std::make_tuple(dists, idxs);
}

__global__ void DistanceBackwardKernel(
    const float* __restrict__ objects, // (O * oD * 3)
    const size_t objects_size, // O
    const size_t objects_dim, // oD
    const float* __restrict__ targets, // (T * tD * 3)
    const size_t targets_dim, // tD
    const int64_t* __restrict__ idx_objects, // (O,)
    const float* __restrict__ grad_dists, // (O,)
    float* __restrict__ grad_points, // ((O or T) * 3)
    float* __restrict__ grad_face, // ((O or T) * max(oD, tD) * 3)
    const double min_triangle_area) {
  // This kernel is used interchangeably to compute bi-directional backward
  // distances between points and triangles/lines. The direction of the distance
  // computed, i.e. point to triangle/line or triangle/line to point, depends on
  // the order of the input arguments and is inferred based on their shape. The
  // shape is used to distinguish between triangles and lines. Note that
  // grad_points will always be used for the point data and grad_face for the
  // edge/triangle

  // Set references to points/face based on whether objects/targets are which
  float3* points_f3 = objects_dim == 1 ? (float3*)objects : (float3*)targets;
  float3* face_f3 = objects_dim == 1 ? (float3*)targets : (float3*)objects;

  const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = gridDim.x * blockDim.x;

  for (size_t o = tid; o < objects_size; o += stride) {
    const int64_t tidx = idx_objects[o];

    size_t point_index = objects_dim == 1 ? o : tidx;
    size_t face_index = objects_dim == 1 ? tidx * targets_dim : o * objects_dim;
    bool isTriangle = objects_dim == 3 || targets_dim == 3;

    float3 grad_point, grad_v0, grad_v1, grad_v2;
    if (isTriangle) {
      const auto grads = PointTriangle3DistanceBackward(
          points_f3[point_index],
          face_f3[face_index],
          face_f3[face_index + 1],
          face_f3[face_index + 2],
          grad_dists[o],
          min_triangle_area);
      grad_point = thrust::get<0>(grads);
      grad_v0 = thrust::get<1>(grads);
      grad_v1 = thrust::get<2>(grads);
      grad_v2 = thrust::get<3>(grads);
    } else {
      const auto grads = PointLine3DistanceBackward(
          points_f3[point_index],
          face_f3[face_index],
          face_f3[face_index + 1],
          grad_dists[o]);
      grad_point = thrust::get<0>(grads);
      grad_v0 = thrust::get<1>(grads);
      grad_v1 = thrust::get<2>(grads);
    }

    atomicAdd(grad_points + point_index * 3 + 0, grad_point.x);
    atomicAdd(grad_points + point_index * 3 + 1, grad_point.y);
    atomicAdd(grad_points + point_index * 3 + 2, grad_point.z);

    atomicAdd(grad_face + face_index * 3 + 0 * 3 + 0, grad_v0.x);
    atomicAdd(grad_face + face_index * 3 + 0 * 3 + 1, grad_v0.y);
    atomicAdd(grad_face + face_index * 3 + 0 * 3 + 2, grad_v0.z);

    atomicAdd(grad_face + face_index * 3 + 1 * 3 + 0, grad_v1.x);
    atomicAdd(grad_face + face_index * 3 + 1 * 3 + 1, grad_v1.y);
    atomicAdd(grad_face + face_index * 3 + 1 * 3 + 2, grad_v1.z);

    if (isTriangle) {
      atomicAdd(grad_face + face_index * 3 + 2 * 3 + 0, grad_v2.x);
      atomicAdd(grad_face + face_index * 3 + 2 * 3 + 1, grad_v2.y);
      atomicAdd(grad_face + face_index * 3 + 2 * 3 + 2, grad_v2.z);
    }
  }
}

std::tuple<at::Tensor, at::Tensor> DistanceBackwardCuda(
    const at::Tensor& objects,
    const size_t objects_dim,
    const at::Tensor& targets,
    const size_t targets_dim,
    const at::Tensor& idx_objects,
    const at::Tensor& grad_dists,
    const double min_triangle_area) {
  // Check inputs are on the same device
  at::TensorArg objects_t{objects, "objects", 1},
      targets_t{targets, "targets", 2},
      idx_objects_t{idx_objects, "idx_objects", 3},
      grad_dists_t{grad_dists, "grad_dists", 4};
  at::CheckedFrom c = "DistanceBackwardCuda";
  at::checkAllSameGPU(c, {objects_t, targets_t, idx_objects_t, grad_dists_t});
  at::checkAllSameType(c, {objects_t, targets_t, grad_dists_t});
  // This is nondeterministic because atomicAdd
  at::globalContext().alertNotDeterministic("DistanceBackwardCuda");

  // Set the device for the kernel launch based on the device of the input
  at::cuda::CUDAGuard device_guard(objects.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int64_t objects_size = objects.size(0);
  const int64_t targets_size = targets.size(0);

  at::Tensor grad_points;
  at::Tensor grad_tris;

  TORCH_CHECK(idx_objects.size(0) == objects_size);
  TORCH_CHECK(grad_dists.size(0) == objects_size);
  if (objects_dim == 1) {
    TORCH_CHECK(
        targets_dim >= 2 && targets_dim <= 3,
        "either object or target must be edge or face");
    TORCH_CHECK(objects.size(1) == 3, "points must be of shape Px3");
    TORCH_CHECK(
        targets.size(2) == 3,
        "face must be of shape Tx3x3, lines must be of shape Tx2x3");
    // clang-format off
    grad_points = at::zeros({objects_size, 3}, objects.options());
    grad_tris = at::zeros({targets_size, int64_t(targets_dim), 3}, targets.options());
    // clang-format on
  } else {
    TORCH_CHECK(targets_dim == 1, "either object or target must be point");
    TORCH_CHECK(
        objects_dim >= 2 && objects_dim <= 3,
        "either object or target must be edge or face");
    TORCH_CHECK(targets.size(1) == 3, "points must be of shape Px3");
    TORCH_CHECK(
        objects.size(2) == 3,
        "face must be of shape Tx3x3, lines must be of shape Tx2x3");
    // clang-format off
    grad_points = at::zeros({targets_size, 3}, targets.options());
    grad_tris = at::zeros({objects_size, int64_t(objects_dim), 3}, objects.options());
    // clang-format on
  }

  if (grad_points.numel() == 0 || grad_tris.numel() == 0) {
    AT_CUDA_CHECK(cudaGetLastError());
    return std::make_tuple(grad_points, grad_tris);
  }

  const int blocks = 64;
  const int threads = 512;

  DistanceBackwardKernel<<<blocks, threads, 0, stream>>>(
      objects.contiguous().data_ptr<float>(),
      objects_size,
      objects_dim,
      targets.contiguous().data_ptr<float>(),
      targets_dim,
      idx_objects.contiguous().data_ptr<int64_t>(),
      grad_dists.contiguous().data_ptr<float>(),
      grad_points.data_ptr<float>(),
      grad_tris.data_ptr<float>(),
      min_triangle_area);

  AT_CUDA_CHECK(cudaGetLastError());
  return std::make_tuple(grad_points, grad_tris);
}

// ****************************************************************************
// *                          PointFaceDistance                               *
// ****************************************************************************

std::tuple<at::Tensor, at::Tensor> PointFaceDistanceForwardCuda(
    const at::Tensor& points,
    const at::Tensor& points_first_idx,
    const at::Tensor& tris,
    const at::Tensor& tris_first_idx,
    const int64_t max_points,
    const double min_triangle_area) {
  return DistanceForwardCuda(
      points,
      1,
      points_first_idx,
      tris,
      3,
      tris_first_idx,
      max_points,
      min_triangle_area);
}

std::tuple<at::Tensor, at::Tensor> PointFaceDistanceBackwardCuda(
    const at::Tensor& points,
    const at::Tensor& tris,
    const at::Tensor& idx_points,
    const at::Tensor& grad_dists,
    const double min_triangle_area) {
  return DistanceBackwardCuda(
      points, 1, tris, 3, idx_points, grad_dists, min_triangle_area);
}

// ****************************************************************************
// *                          FacePointDistance                               *
// ****************************************************************************

std::tuple<at::Tensor, at::Tensor> FacePointDistanceForwardCuda(
    const at::Tensor& points,
    const at::Tensor& points_first_idx,
    const at::Tensor& tris,
    const at::Tensor& tris_first_idx,
    const int64_t max_tris,
    const double min_triangle_area) {
  return DistanceForwardCuda(
      tris,
      3,
      tris_first_idx,
      points,
      1,
      points_first_idx,
      max_tris,
      min_triangle_area);
}

std::tuple<at::Tensor, at::Tensor> FacePointDistanceBackwardCuda(
    const at::Tensor& points,
    const at::Tensor& tris,
    const at::Tensor& idx_tris,
    const at::Tensor& grad_dists,
    const double min_triangle_area) {
  return DistanceBackwardCuda(
      tris, 3, points, 1, idx_tris, grad_dists, min_triangle_area);
}

// ****************************************************************************
// *                          PointEdgeDistance                               *
// ****************************************************************************

std::tuple<at::Tensor, at::Tensor> PointEdgeDistanceForwardCuda(
    const at::Tensor& points,
    const at::Tensor& points_first_idx,
    const at::Tensor& segms,
    const at::Tensor& segms_first_idx,
    const int64_t max_points) {
  return DistanceForwardCuda(
      points,
      1,
      points_first_idx,
      segms,
      2,
      segms_first_idx,
      max_points,
      1); // todo: unused parameter handling for min_triangle_area
}

std::tuple<at::Tensor, at::Tensor> PointEdgeDistanceBackwardCuda(
    const at::Tensor& points,
    const at::Tensor& segms,
    const at::Tensor& idx_points,
    const at::Tensor& grad_dists) {
  return DistanceBackwardCuda(points, 1, segms, 2, idx_points, grad_dists, 1);
}

// ****************************************************************************
// *                          EdgePointDistance                               *
// ****************************************************************************

std::tuple<at::Tensor, at::Tensor> EdgePointDistanceForwardCuda(
    const at::Tensor& points,
    const at::Tensor& points_first_idx,
    const at::Tensor& segms,
    const at::Tensor& segms_first_idx,
    const int64_t max_segms) {
  return DistanceForwardCuda(
      segms, 2, segms_first_idx, points, 1, points_first_idx, max_segms, 1);
}

std::tuple<at::Tensor, at::Tensor> EdgePointDistanceBackwardCuda(
    const at::Tensor& points,
    const at::Tensor& segms,
    const at::Tensor& idx_segms,
    const at::Tensor& grad_dists) {
  return DistanceBackwardCuda(segms, 2, points, 1, idx_segms, grad_dists, 1);
}

// ****************************************************************************
// *                     PointFaceArrayDistance                               *
// ****************************************************************************
// TODO: Create wrapper function and merge kernel with other array kernel

__global__ void PointFaceArrayForwardKernel(
    const float* __restrict__ points, // (P, 3)
    const float* __restrict__ tris, // (T, 3, 3)
    float* __restrict__ dists, // (P, T)
    const size_t P,
    const size_t T,
    const double min_triangle_area) {
  const float3* points_f3 = (float3*)points;
  const float3* tris_f3 = (float3*)tris;

  // Parallelize over P * S computations
  const int num_threads = gridDim.x * blockDim.x;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;

  for (int t_i = tid; t_i < P * T; t_i += num_threads) {
    const int t = t_i / P; // segment index.
    const int p = t_i % P; // point index
    const float3 v0 = tris_f3[t * 3 + 0];
    const float3 v1 = tris_f3[t * 3 + 1];
    const float3 v2 = tris_f3[t * 3 + 2];

    const float3 point = points_f3[p];
    float dist =
        PointTriangle3DistanceForward(point, v0, v1, v2, min_triangle_area);
    dists[p * T + t] = dist;
  }
}

at::Tensor PointFaceArrayDistanceForwardCuda(
    const at::Tensor& points,
    const at::Tensor& tris,
    const double min_triangle_area) {
  // Check inputs are on the same device
  at::TensorArg points_t{points, "points", 1}, tris_t{tris, "tris", 2};
  at::CheckedFrom c = "PointFaceArrayDistanceForwardCuda";
  at::checkAllSameGPU(c, {points_t, tris_t});
  at::checkAllSameType(c, {points_t, tris_t});

  // Set the device for the kernel launch based on the device of the input
  at::cuda::CUDAGuard device_guard(points.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int64_t P = points.size(0);
  const int64_t T = tris.size(0);

  TORCH_CHECK(points.size(1) == 3, "points must be of shape Px3");
  TORCH_CHECK(
      (tris.size(1) == 3) && (tris.size(2) == 3),
      "tris must be of shape Tx3x3");

  at::Tensor dists = at::zeros({P, T}, points.options());

  if (dists.numel() == 0) {
    AT_CUDA_CHECK(cudaGetLastError());
    return dists;
  }

  const size_t blocks = 1024;
  const size_t threads = 64;

  PointFaceArrayForwardKernel<<<blocks, threads, 0, stream>>>(
      points.contiguous().data_ptr<float>(),
      tris.contiguous().data_ptr<float>(),
      dists.data_ptr<float>(),
      P,
      T,
      min_triangle_area);

  AT_CUDA_CHECK(cudaGetLastError());
  return dists;
}

__global__ void PointFaceArrayBackwardKernel(
    const float* __restrict__ points, // (P, 3)
    const float* __restrict__ tris, // (T, 3, 3)
    const float* __restrict__ grad_dists, // (P, T)
    float* __restrict__ grad_points, // (P, 3)
    float* __restrict__ grad_tris, // (T, 3, 3)
    const size_t P,
    const size_t T,
    const double min_triangle_area) {
  const float3* points_f3 = (float3*)points;
  const float3* tris_f3 = (float3*)tris;

  // Parallelize over P * S computations
  const int num_threads = gridDim.x * blockDim.x;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;

  for (int t_i = tid; t_i < P * T; t_i += num_threads) {
    const int t = t_i / P; // triangle index.
    const int p = t_i % P; // point index
    const float3 v0 = tris_f3[t * 3 + 0];
    const float3 v1 = tris_f3[t * 3 + 1];
    const float3 v2 = tris_f3[t * 3 + 2];

    const float3 point = points_f3[p];

    const float grad_dist = grad_dists[p * T + t];
    const auto grad = PointTriangle3DistanceBackward(
        point, v0, v1, v2, grad_dist, min_triangle_area);

    const float3 grad_point = thrust::get<0>(grad);
    const float3 grad_v0 = thrust::get<1>(grad);
    const float3 grad_v1 = thrust::get<2>(grad);
    const float3 grad_v2 = thrust::get<3>(grad);

    atomicAdd(grad_points + 3 * p + 0, grad_point.x);
    atomicAdd(grad_points + 3 * p + 1, grad_point.y);
    atomicAdd(grad_points + 3 * p + 2, grad_point.z);

    atomicAdd(grad_tris + t * 3 * 3 + 0 * 3 + 0, grad_v0.x);
    atomicAdd(grad_tris + t * 3 * 3 + 0 * 3 + 1, grad_v0.y);
    atomicAdd(grad_tris + t * 3 * 3 + 0 * 3 + 2, grad_v0.z);

    atomicAdd(grad_tris + t * 3 * 3 + 1 * 3 + 0, grad_v1.x);
    atomicAdd(grad_tris + t * 3 * 3 + 1 * 3 + 1, grad_v1.y);
    atomicAdd(grad_tris + t * 3 * 3 + 1 * 3 + 2, grad_v1.z);

    atomicAdd(grad_tris + t * 3 * 3 + 2 * 3 + 0, grad_v2.x);
    atomicAdd(grad_tris + t * 3 * 3 + 2 * 3 + 1, grad_v2.y);
    atomicAdd(grad_tris + t * 3 * 3 + 2 * 3 + 2, grad_v2.z);
  }
}

std::tuple<at::Tensor, at::Tensor> PointFaceArrayDistanceBackwardCuda(
    const at::Tensor& points,
    const at::Tensor& tris,
    const at::Tensor& grad_dists,
    const double min_triangle_area) {
  // Check inputs are on the same device
  at::TensorArg points_t{points, "points", 1}, tris_t{tris, "tris", 2},
      grad_dists_t{grad_dists, "grad_dists", 3};
  at::CheckedFrom c = "PointFaceArrayDistanceBackwardCuda";
  at::checkAllSameGPU(c, {points_t, tris_t, grad_dists_t});
  at::checkAllSameType(c, {points_t, tris_t, grad_dists_t});
  // This is nondeterministic because atomicAdd
  at::globalContext().alertNotDeterministic(
      "PointFaceArrayDistanceBackwardCuda");

  // Set the device for the kernel launch based on the device of the input
  at::cuda::CUDAGuard device_guard(points.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int64_t P = points.size(0);
  const int64_t T = tris.size(0);

  TORCH_CHECK(points.size(1) == 3, "points must be of shape Px3");
  TORCH_CHECK(
      (tris.size(1) == 3) && (tris.size(2) == 3),
      "tris must be of shape Tx3x3");
  TORCH_CHECK((grad_dists.size(0) == P) && (grad_dists.size(1) == T));

  at::Tensor grad_points = at::zeros({P, 3}, points.options());
  at::Tensor grad_tris = at::zeros({T, 3, 3}, tris.options());

  if (grad_points.numel() == 0 || grad_tris.numel() == 0) {
    AT_CUDA_CHECK(cudaGetLastError());
    return std::make_tuple(grad_points, grad_tris);
  }

  const size_t blocks = 1024;
  const size_t threads = 64;

  PointFaceArrayBackwardKernel<<<blocks, threads, 0, stream>>>(
      points.contiguous().data_ptr<float>(),
      tris.contiguous().data_ptr<float>(),
      grad_dists.contiguous().data_ptr<float>(),
      grad_points.data_ptr<float>(),
      grad_tris.data_ptr<float>(),
      P,
      T,
      min_triangle_area);

  AT_CUDA_CHECK(cudaGetLastError());
  return std::make_tuple(grad_points, grad_tris);
}

// ****************************************************************************
// *                     PointEdgeArrayDistance                               *
// ****************************************************************************
// TODO: Create wrapper function and merge kernel with other array kernel

__global__ void PointEdgeArrayForwardKernel(
    const float* __restrict__ points, // (P, 3)
    const float* __restrict__ segms, // (S, 2, 3)
    float* __restrict__ dists, // (P, S)
    const size_t P,
    const size_t S) {
  float3* points_f3 = (float3*)points;
  float3* segms_f3 = (float3*)segms;

  // Parallelize over P * S computations
  const int num_threads = gridDim.x * blockDim.x;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;

  for (int t_i = tid; t_i < P * S; t_i += num_threads) {
    const int s = t_i / P; // segment index.
    const int p = t_i % P; // point index
    float3 a = segms_f3[s * 2 + 0];
    float3 b = segms_f3[s * 2 + 1];

    float3 point = points_f3[p];
    float dist = PointLine3DistanceForward(point, a, b);
    dists[p * S + s] = dist;
  }
}

at::Tensor PointEdgeArrayDistanceForwardCuda(
    const at::Tensor& points,
    const at::Tensor& segms) {
  // Check inputs are on the same device
  at::TensorArg points_t{points, "points", 1}, segms_t{segms, "segms", 2};
  at::CheckedFrom c = "PointEdgeArrayDistanceForwardCuda";
  at::checkAllSameGPU(c, {points_t, segms_t});
  at::checkAllSameType(c, {points_t, segms_t});

  // Set the device for the kernel launch based on the device of the input
  at::cuda::CUDAGuard device_guard(points.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int64_t P = points.size(0);
  const int64_t S = segms.size(0);

  TORCH_CHECK(points.size(1) == 3, "points must be of shape Px3");
  TORCH_CHECK(
      (segms.size(1) == 2) && (segms.size(2) == 3),
      "segms must be of shape Sx2x3");

  at::Tensor dists = at::zeros({P, S}, points.options());

  if (dists.numel() == 0) {
    AT_CUDA_CHECK(cudaGetLastError());
    return dists;
  }

  const size_t blocks = 1024;
  const size_t threads = 64;

  PointEdgeArrayForwardKernel<<<blocks, threads, 0, stream>>>(
      points.contiguous().data_ptr<float>(),
      segms.contiguous().data_ptr<float>(),
      dists.data_ptr<float>(),
      P,
      S);

  AT_CUDA_CHECK(cudaGetLastError());
  return dists;
}

__global__ void PointEdgeArrayBackwardKernel(
    const float* __restrict__ points, // (P, 3)
    const float* __restrict__ segms, // (S, 2, 3)
    const float* __restrict__ grad_dists, // (P, S)
    float* __restrict__ grad_points, // (P, 3)
    float* __restrict__ grad_segms, // (S, 2, 3)
    const size_t P,
    const size_t S) {
  float3* points_f3 = (float3*)points;
  float3* segms_f3 = (float3*)segms;

  // Parallelize over P * S computations
  const int num_threads = gridDim.x * blockDim.x;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;

  for (int t_i = tid; t_i < P * S; t_i += num_threads) {
    const int s = t_i / P; // segment index.
    const int p = t_i % P; // point index
    const float3 a = segms_f3[s * 2 + 0];
    const float3 b = segms_f3[s * 2 + 1];

    const float3 point = points_f3[p];
    const float grad_dist = grad_dists[p * S + s];
    const auto grads = PointLine3DistanceBackward(point, a, b, grad_dist);
    const float3 grad_point = thrust::get<0>(grads);
    const float3 grad_a = thrust::get<1>(grads);
    const float3 grad_b = thrust::get<2>(grads);

    atomicAdd(grad_points + p * 3 + 0, grad_point.x);
    atomicAdd(grad_points + p * 3 + 1, grad_point.y);
    atomicAdd(grad_points + p * 3 + 2, grad_point.z);

    atomicAdd(grad_segms + s * 2 * 3 + 0 * 3 + 0, grad_a.x);
    atomicAdd(grad_segms + s * 2 * 3 + 0 * 3 + 1, grad_a.y);
    atomicAdd(grad_segms + s * 2 * 3 + 0 * 3 + 2, grad_a.z);

    atomicAdd(grad_segms + s * 2 * 3 + 1 * 3 + 0, grad_b.x);
    atomicAdd(grad_segms + s * 2 * 3 + 1 * 3 + 1, grad_b.y);
    atomicAdd(grad_segms + s * 2 * 3 + 1 * 3 + 2, grad_b.z);
  }
}

std::tuple<at::Tensor, at::Tensor> PointEdgeArrayDistanceBackwardCuda(
    const at::Tensor& points,
    const at::Tensor& segms,
    const at::Tensor& grad_dists) {
  // Check inputs are on the same device
  at::TensorArg points_t{points, "points", 1}, segms_t{segms, "segms", 2},
      grad_dists_t{grad_dists, "grad_dists", 3};
  at::CheckedFrom c = "PointEdgeArrayDistanceBackwardCuda";
  at::checkAllSameGPU(c, {points_t, segms_t, grad_dists_t});
  at::checkAllSameType(c, {points_t, segms_t, grad_dists_t});
  // This is nondeterministic because atomicAdd
  at::globalContext().alertNotDeterministic(
      "PointEdgeArrayDistanceBackwardCuda");

  // Set the device for the kernel launch based on the device of the input
  at::cuda::CUDAGuard device_guard(points.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int64_t P = points.size(0);
  const int64_t S = segms.size(0);

  TORCH_CHECK(points.size(1) == 3, "points must be of shape Px3");
  TORCH_CHECK(
      (segms.size(1) == 2) && (segms.size(2) == 3),
      "segms must be of shape Sx2x3");
  TORCH_CHECK((grad_dists.size(0) == P) && (grad_dists.size(1) == S));

  at::Tensor grad_points = at::zeros({P, 3}, points.options());
  at::Tensor grad_segms = at::zeros({S, 2, 3}, segms.options());

  if (grad_points.numel() == 0 || grad_segms.numel() == 0) {
    AT_CUDA_CHECK(cudaGetLastError());
    return std::make_tuple(grad_points, grad_segms);
  }

  const size_t blocks = 1024;
  const size_t threads = 64;

  PointEdgeArrayBackwardKernel<<<blocks, threads, 0, stream>>>(
      points.contiguous().data_ptr<float>(),
      segms.contiguous().data_ptr<float>(),
      grad_dists.contiguous().data_ptr<float>(),
      grad_points.data_ptr<float>(),
      grad_segms.data_ptr<float>(),
      P,
      S);
  AT_CUDA_CHECK(cudaGetLastError());
  return std::make_tuple(grad_points, grad_segms);
}
