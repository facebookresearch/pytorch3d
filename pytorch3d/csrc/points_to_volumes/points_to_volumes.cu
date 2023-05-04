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

using at::PackedTensorAccessor64;
using at::RestrictPtrTraits;

// A chunk of work is blocksize-many points.
// There are N clouds in the batch, and P points in each cloud.
// The number of potential chunks to do per cloud is (1+(P-1)/blocksize),
// which we call chunks_per_cloud.
// These (N*chunks_per_cloud) chunks are divided among the gridSize-many blocks.
// In block b, we work on chunks b, b+gridSize, b+2*gridSize etc .
// In chunk i, we work on cloud (i/chunks_per_cloud) on points starting from
// blocksize*(i%chunks_per_cloud).

// Explanation of the calculation is in the cpp file.

// EightDirections(t) runs t(a,b,c) for every combination of boolean a, b, c.
template <class T>
static __device__ void EightDirections(T&& t) {
  t(false, false, false);
  t(false, false, true);
  t(false, true, false);
  t(false, true, true);
  t(true, false, false);
  t(true, false, true);
  t(true, true, false);
  t(true, true, true);
}

__global__ void PointsToVolumesForwardKernel(
    const PackedTensorAccessor64<float, 3, RestrictPtrTraits> points_3d,
    const PackedTensorAccessor64<float, 3, RestrictPtrTraits> points_features,
    PackedTensorAccessor64<float, 5, RestrictPtrTraits> volume_densities,
    PackedTensorAccessor64<float, 5, RestrictPtrTraits> volume_features,
    PackedTensorAccessor64<int64_t, 2, RestrictPtrTraits> grid_sizes,
    PackedTensorAccessor64<float, 2, RestrictPtrTraits> mask,
    const float point_weight,
    const bool align_corners,
    const bool splat,
    const int64_t batch_size,
    const int64_t P,
    const int64_t n_features) {
  const int64_t chunks_per_cloud = (1 + (P - 1) / blockDim.x);
  const int64_t chunks_to_do = batch_size * chunks_per_cloud;
  const int scale_offset = align_corners ? 1 : 0;
  const float offset = align_corners ? 0 : 0.5;
  for (int64_t chunk = blockIdx.x; chunk < chunks_to_do; chunk += gridDim.x) {
    const int64_t batch_index = chunk / chunks_per_cloud;
    const int64_t start_point = blockDim.x * (chunk % chunks_per_cloud);
    int64_t point_idx = start_point + threadIdx.x;
    if (point_idx >= P) {
      continue;
    }
    if (mask[batch_index][point_idx] == 0) {
      continue;
    }
    auto volume_densities_aa = volume_densities[batch_index][0];
    auto volume_features_aa = volume_features[batch_index];
    auto point = points_3d[batch_index][point_idx];
    auto point_features = points_features[batch_index][point_idx];
    const int64_t grid_size_x = grid_sizes[batch_index][2];
    const int64_t grid_size_y = grid_sizes[batch_index][1];
    const int64_t grid_size_z = grid_sizes[batch_index][0];
    auto increment_location =
        [&](int64_t x, int64_t y, int64_t z, float weight) {
          if (x >= grid_size_x || y >= grid_size_y || z >= grid_size_z) {
            return;
          }
          if (x < 0 || y < 0 || z < 0) {
            return;
          }

          atomicAdd(&volume_densities_aa[z][y][x], weight * point_weight);

          for (int64_t feature_idx = 0; feature_idx < n_features;
               ++feature_idx) {
            atomicAdd(
                &volume_features_aa[feature_idx][z][y][x],
                point_features[feature_idx] * weight * point_weight);
          }
        };
    if (!splat) {
      long x = std::lround(
          (point[0] + 1) * 0.5 * (grid_size_x - scale_offset) - offset);
      long y = std::lround(
          (point[1] + 1) * 0.5 * (grid_size_y - scale_offset) - offset);
      long z = std::lround(
          (point[2] + 1) * 0.5 * (grid_size_z - scale_offset) - offset);
      increment_location(x, y, z, 1);
    } else {
      float x = 0, y = 0, z = 0;
      float rx = std::modf(
          (point[0] + 1) * 0.5 * (grid_size_x - scale_offset) - offset, &x);
      float ry = std::modf(
          (point[1] + 1) * 0.5 * (grid_size_y - scale_offset) - offset, &y);
      float rz = std::modf(
          (point[2] + 1) * 0.5 * (grid_size_z - scale_offset) - offset, &z);
      auto handle_point = [&](bool up_x, bool up_y, bool up_z) {
        float weight =
            (up_x ? rx : 1 - rx) * (up_y ? ry : 1 - ry) * (up_z ? rz : 1 - rz);
        increment_location(x + up_x, y + up_y, z + up_z, weight);
      };
      EightDirections(handle_point);
    }
  }
}

void PointsToVolumesForwardCuda(
    const at::Tensor& points_3d,
    const at::Tensor& points_features,
    const at::Tensor& volume_densities,
    const at::Tensor& volume_features,
    const at::Tensor& grid_sizes,
    const at::Tensor& mask,
    const float point_weight,
    const bool align_corners,
    const bool splat) {
  // Check inputs are on the same device
  at::TensorArg points_3d_t{points_3d, "points_3d", 1},
      points_features_t{points_features, "points_features", 2},
      volume_densities_t{volume_densities, "volume_densities", 3},
      volume_features_t{volume_features, "volume_features", 4},
      grid_sizes_t{grid_sizes, "grid_sizes", 5}, mask_t{mask, "mask", 6};
  at::CheckedFrom c = "PointsToVolumesForwardCuda";
  at::checkAllSameGPU(
      c,
      {points_3d_t,
       points_features_t,
       volume_densities_t,
       volume_features_t,
       grid_sizes_t,
       mask_t});

  // This is nondeterministic because atomicAdd
  at::globalContext().alertNotDeterministic("PointsToVolumesForwardCuda");

  // Set the device for the kernel launch based on the device of the input
  at::cuda::CUDAGuard device_guard(points_3d.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int blocks = 1024;
  const int threads = 32;

  const int64_t batch_size = points_3d.size(0);
  const int64_t P = points_3d.size(1);
  const int64_t n_features = points_features.size(2);

  PointsToVolumesForwardKernel<<<blocks, threads, 0, stream>>>(
      points_3d.packed_accessor64<float, 3, RestrictPtrTraits>(),
      points_features.packed_accessor64<float, 3, RestrictPtrTraits>(),
      volume_densities.packed_accessor64<float, 5, RestrictPtrTraits>(),
      volume_features.packed_accessor64<float, 5, RestrictPtrTraits>(),
      grid_sizes.packed_accessor64<int64_t, 2, RestrictPtrTraits>(),
      mask.packed_accessor64<float, 2, RestrictPtrTraits>(),
      point_weight,
      align_corners,
      splat,
      batch_size,
      P,
      n_features);
}

__global__ void PointsToVolumesBackwardKernel(
    const PackedTensorAccessor64<float, 3, RestrictPtrTraits> points_3d,
    const PackedTensorAccessor64<float, 3, RestrictPtrTraits> points_features,
    const PackedTensorAccessor64<int64_t, 2, RestrictPtrTraits> grid_sizes,
    const PackedTensorAccessor64<float, 2, RestrictPtrTraits> mask,
    PackedTensorAccessor64<float, 5, RestrictPtrTraits> grad_volume_densities,
    PackedTensorAccessor64<float, 5, RestrictPtrTraits> grad_volume_features,
    PackedTensorAccessor64<float, 3, RestrictPtrTraits> grad_points_3d,
    PackedTensorAccessor64<float, 3, RestrictPtrTraits> grad_points_features,
    const float point_weight,
    const bool align_corners,
    const bool splat,
    const int64_t batch_size,
    const int64_t P,
    const int64_t n_features) {
  const int64_t chunks_per_cloud = (1 + (P - 1) / blockDim.x);
  const int64_t chunks_to_do = batch_size * chunks_per_cloud;
  const int scale_offset = align_corners ? 1 : 0;
  const float offset = align_corners ? 0 : 0.5;
  // Note that the gradients belonging to each point are only touched by
  // a single thread in one of our "chunks", which is in a single block.
  // So unlike in the forward pass, there's no need for atomics here.
  for (int64_t chunk = blockIdx.x; chunk < chunks_to_do; chunk += gridDim.x) {
    const int64_t batch_index = chunk / chunks_per_cloud;
    const int64_t start_point = blockDim.x * (chunk % chunks_per_cloud);
    int64_t point_idx = start_point + threadIdx.x;
    if (point_idx >= P) {
      continue;
    }
    if (mask[batch_index][point_idx] == 0) {
      continue;
    }
    auto point = points_3d[batch_index][point_idx];
    auto point_features = points_features[batch_index][point_idx];
    auto grad_point = grad_points_3d[batch_index][point_idx];
    auto grad_point_features = grad_points_features[batch_index][point_idx];
    auto grad_volume_densities_a = grad_volume_densities[batch_index][0];
    auto grad_volume_features_a = grad_volume_features[batch_index];
    const int64_t grid_size_x = grid_sizes[batch_index][2];
    const int64_t grid_size_y = grid_sizes[batch_index][1];
    const int64_t grid_size_z = grid_sizes[batch_index][0];

    auto increment_location =
        [&](int64_t x, int64_t y, int64_t z, float weight) {
          if (x >= grid_size_x || y >= grid_size_y || z >= grid_size_z) {
            return false;
          }
          if (x < 0 || y < 0 || z < 0) {
            return false;
          }

          // This is a forward line, for comparison
          // volume_densities_aa[z][y][x] += weight * point_weight;

          for (int64_t feature_idx = 0; feature_idx < n_features;
               ++feature_idx) {
            // This is a forward line, for comparison
            // volume_features_aa[feature_idx][z][y][x] +=
            //    point_features[feature_idx] * weight * point_weight;
            grad_point_features[feature_idx] +=
                grad_volume_features_a[feature_idx][z][y][x] * weight *
                point_weight;
          }
          return true;
        };

    if (!splat) {
      long x = std::lround(
          (point[0] + 1) * 0.5 * (grid_size_x - scale_offset) - offset);
      long y = std::lround(
          (point[1] + 1) * 0.5 * (grid_size_y - scale_offset) - offset);
      long z = std::lround(
          (point[2] + 1) * 0.5 * (grid_size_z - scale_offset) - offset);
      increment_location(x, y, z, 1);
    } else {
      float x = 0, y = 0, z = 0;
      float rx = std::modf(
          (point[0] + 1) * 0.5 * (grid_size_x - scale_offset) - offset, &x);
      float ry = std::modf(
          (point[1] + 1) * 0.5 * (grid_size_y - scale_offset) - offset, &y);
      float rz = std::modf(
          (point[2] + 1) * 0.5 * (grid_size_z - scale_offset) - offset, &z);
      auto handle_point = [&](bool up_x, bool up_y, bool up_z) {
        float weight_x = (up_x ? rx : 1 - rx);
        float weight_y = (up_y ? ry : 1 - ry);
        float weight_z = (up_z ? rz : 1 - rz);
        float weight = weight_x * weight_y * weight_z;
        if (increment_location(x + up_x, y + up_y, z + up_z, weight)) {
          // weight * point_weight has been added to
          // volume_densities_aa[z+up_z][y+up_y][x+up_x]
          // Also for each feature_idx,
          //   point_features[feature_idx] * weight * point_weight
          // has been added to
          // volume_features_aa[feature_idx][z+up_z][y+up_y][x+up_x]

          double source_gradient =
              grad_volume_densities_a[z + up_z][y + up_y][x + up_x];
          for (int64_t feature_idx = 0; feature_idx < n_features;
               ++feature_idx) {
            source_gradient += point_features[feature_idx] *
                grad_volume_features_a[feature_idx][z + up_z][y + up_y]
                                      [x + up_x];
          }
          grad_point[0] += source_gradient * (up_x ? 1 : -1) * weight_y *
              weight_z * 0.5 * (grid_size_x - scale_offset) * point_weight;
          grad_point[1] += source_gradient * (up_y ? 1 : -1) * weight_x *
              weight_z * 0.5 * (grid_size_y - scale_offset) * point_weight;
          grad_point[2] += source_gradient * (up_z ? 1 : -1) * weight_x *
              weight_y * 0.5 * (grid_size_z - scale_offset) * point_weight;
        }
      };
      EightDirections(handle_point);
    }
  }
}

void PointsToVolumesBackwardCuda(
    const at::Tensor& points_3d,
    const at::Tensor& points_features,
    const at::Tensor& grid_sizes,
    const at::Tensor& mask,
    const float point_weight,
    const bool align_corners,
    const bool splat,
    const at::Tensor& grad_volume_densities,
    const at::Tensor& grad_volume_features,
    const at::Tensor& grad_points_3d,
    const at::Tensor& grad_points_features) {
  // Check inputs are on the same device
  at::TensorArg points_3d_t{points_3d, "points_3d", 1},
      points_features_t{points_features, "points_features", 2},
      grid_sizes_t{grid_sizes, "grid_sizes", 3}, mask_t{mask, "mask", 4},
      grad_volume_densities_t{
          grad_volume_densities, "grad_volume_densities", 8},
      grad_volume_features_t{grad_volume_features, "grad_volume_features", 9},
      grad_points_3d_t{grad_points_3d, "grad_points_3d", 10},
      grad_points_features_t{grad_points_features, "grad_points_features", 11};

  at::CheckedFrom c = "PointsToVolumesBackwardCuda";
  at::checkAllSameGPU(
      c,
      {points_3d_t,
       points_features_t,
       grid_sizes_t,
       mask_t,
       grad_volume_densities_t,
       grad_volume_features_t,
       grad_points_3d_t,
       grad_points_features_t});

  // Set the device for the kernel launch based on the device of the input
  at::cuda::CUDAGuard device_guard(points_3d.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int blocks = 1024;
  const int threads = 32;

  const int64_t batch_size = points_3d.size(0);
  const int64_t P = points_3d.size(1);
  const int64_t n_features = points_features.size(2);

  PointsToVolumesBackwardKernel<<<blocks, threads, 0, stream>>>(
      points_3d.packed_accessor64<float, 3, RestrictPtrTraits>(),
      points_features.packed_accessor64<float, 3, RestrictPtrTraits>(),
      grid_sizes.packed_accessor64<int64_t, 2, RestrictPtrTraits>(),
      mask.packed_accessor64<float, 2, RestrictPtrTraits>(),
      grad_volume_densities.packed_accessor64<float, 5, RestrictPtrTraits>(),
      grad_volume_features.packed_accessor64<float, 5, RestrictPtrTraits>(),
      grad_points_3d.packed_accessor64<float, 3, RestrictPtrTraits>(),
      grad_points_features.packed_accessor64<float, 3, RestrictPtrTraits>(),
      point_weight,
      align_corners,
      splat,
      batch_size,
      P,
      n_features);
}
