/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <torch/csrc/autograd/VariableTypeUtils.h>
#include <torch/extension.h>
#include <cstdio>
#include <tuple>
#include "utils/pytorch3d_cutils.h"

/*
    volume_features and volume_densities are modified in place.

    Args:
        points_3d: Batch of 3D point cloud coordinates of shape
            `(minibatch, N, 3)` where N is the number of points
            in each point cloud. Coordinates have to be specified in the
            local volume coordinates (ranging in [-1, 1]).
        points_features: Features of shape `(minibatch, N, feature_dim)`
            corresponding to the points of the input point cloud `points_3d`.
        volume_features: Batch of input feature volumes
            of shape `(minibatch, feature_dim, D, H, W)`
        volume_densities: Batch of input feature volume densities
            of shape `(minibatch, 1, D, H, W)`. Each voxel should
            contain a non-negative number corresponding to its
            opaqueness (the higher, the less transparent).

        grid_sizes: `LongTensor` of shape (minibatch, 3) representing the
            spatial resolutions of each of the the non-flattened `volumes`
            tensors. Note that the following has to hold:
                `torch.prod(grid_sizes, dim=1)==N_voxels`.

        point_weight: A scalar controlling how much weight a single point has.

        mask: A binary mask of shape `(minibatch, N)` determining
            which 3D points are going to be converted to the resulting
            volume. Set to `None` if all points are valid.

        align_corners: as for grid_sample.

        splat: if true, trilinear interpolation. If false all the weight goes in
            the nearest voxel.
*/

void PointsToVolumesForwardCpu(
    const torch::Tensor& points_3d,
    const torch::Tensor& points_features,
    const torch::Tensor& volume_densities,
    const torch::Tensor& volume_features,
    const torch::Tensor& grid_sizes,
    const torch::Tensor& mask,
    float point_weight,
    bool align_corners,
    bool splat);

void PointsToVolumesForwardCuda(
    const torch::Tensor& points_3d,
    const torch::Tensor& points_features,
    const torch::Tensor& volume_densities,
    const torch::Tensor& volume_features,
    const torch::Tensor& grid_sizes,
    const torch::Tensor& mask,
    float point_weight,
    bool align_corners,
    bool splat);

inline void PointsToVolumesForward(
    const torch::Tensor& points_3d,
    const torch::Tensor& points_features,
    const torch::Tensor& volume_densities,
    const torch::Tensor& volume_features,
    const torch::Tensor& grid_sizes,
    const torch::Tensor& mask,
    float point_weight,
    bool align_corners,
    bool splat) {
  if (points_3d.is_cuda()) {
#ifdef WITH_CUDA
    CHECK_CUDA(points_3d);
    CHECK_CUDA(points_features);
    CHECK_CUDA(volume_densities);
    CHECK_CUDA(volume_features);
    CHECK_CUDA(grid_sizes);
    CHECK_CUDA(mask);
    PointsToVolumesForwardCuda(
        points_3d,
        points_features,
        volume_densities,
        volume_features,
        grid_sizes,
        mask,
        point_weight,
        align_corners,
        splat);
    torch::autograd::increment_version(volume_features);
    torch::autograd::increment_version(volume_densities);
    return;
#else
    AT_ERROR("Not compiled with GPU support.");
#endif
  }
  PointsToVolumesForwardCpu(
      points_3d,
      points_features,
      volume_densities,
      volume_features,
      grid_sizes,
      mask,
      point_weight,
      align_corners,
      splat);
}

// grad_points_3d and grad_points_features are modified in place.

void PointsToVolumesBackwardCpu(
    const torch::Tensor& points_3d,
    const torch::Tensor& points_features,
    const torch::Tensor& grid_sizes,
    const torch::Tensor& mask,
    float point_weight,
    bool align_corners,
    bool splat,
    const torch::Tensor& grad_volume_densities,
    const torch::Tensor& grad_volume_features,
    const torch::Tensor& grad_points_3d,
    const torch::Tensor& grad_points_features);

void PointsToVolumesBackwardCuda(
    const torch::Tensor& points_3d,
    const torch::Tensor& points_features,
    const torch::Tensor& grid_sizes,
    const torch::Tensor& mask,
    float point_weight,
    bool align_corners,
    bool splat,
    const torch::Tensor& grad_volume_densities,
    const torch::Tensor& grad_volume_features,
    const torch::Tensor& grad_points_3d,
    const torch::Tensor& grad_points_features);

inline void PointsToVolumesBackward(
    const torch::Tensor& points_3d,
    const torch::Tensor& points_features,
    const torch::Tensor& grid_sizes,
    const torch::Tensor& mask,
    float point_weight,
    bool align_corners,
    bool splat,
    const torch::Tensor& grad_volume_densities,
    const torch::Tensor& grad_volume_features,
    const torch::Tensor& grad_points_3d,
    const torch::Tensor& grad_points_features) {
  if (points_3d.is_cuda()) {
#ifdef WITH_CUDA
    CHECK_CUDA(points_3d);
    CHECK_CUDA(points_features);
    CHECK_CUDA(grid_sizes);
    CHECK_CUDA(mask);
    CHECK_CUDA(grad_volume_densities);
    CHECK_CUDA(grad_volume_features);
    CHECK_CUDA(grad_points_3d);
    CHECK_CUDA(grad_points_features);
    PointsToVolumesBackwardCuda(
        points_3d,
        points_features,
        grid_sizes,
        mask,
        point_weight,
        align_corners,
        splat,
        grad_volume_densities,
        grad_volume_features,
        grad_points_3d,
        grad_points_features);
    return;
#else
    AT_ERROR("Not compiled with GPU support.");
#endif
  }
  PointsToVolumesBackwardCpu(
      points_3d,
      points_features,
      grid_sizes,
      mask,
      point_weight,
      align_corners,
      splat,
      grad_volume_densities,
      grad_volume_features,
      grad_points_3d,
      grad_points_features);
}
