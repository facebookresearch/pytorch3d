/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <torch/csrc/autograd/VariableTypeUtils.h>
#include <torch/extension.h>
#include <algorithm>
#include <cmath>
#include <thread>
#include <vector>

// In the x direction, the location {0, ..., grid_size_x - 1} correspond to
// points px in [-1, 1]. There are two ways to do this.

// If align_corners=True, px=-1 is the exact location 0 and px=1 is the exact
// location grid_size_x - 1.
// So the location of px is {(px + 1) * 0.5} * (grid_size_x - 1).
// Note that if you generate random points within the bounds you are less likely
// to hit the edge locations than other locations.
// This can be thought of as saying "location i" means a specific point.

// If align_corners=False, px=-1 is half way between the exact location 0 and
// the non-existent location -1, i.e. location -0.5.
// Similarly px=1 is is half way between the exact location grid_size_x-1 and
// the non-existent location grid_size, i.e. the location grid_size_x - 0.5.
// So the location of px is ({(px + 1) * 0.5} * grid_size_x) - 0.5.
// Note that if you generate random points within the bounds you are equally
// likely to hit any location.
// This can be thought of as saying "location i" means the whole box from
// (i-0.5) to (i+0.5)

// EightDirections(t) runs t(a,b,c) for every combination of boolean a, b, c.
template <class T>
static void EightDirections(T&& t) {
  t(false, false, false);
  t(false, false, true);
  t(false, true, false);
  t(false, true, true);
  t(true, false, false);
  t(true, false, true);
  t(true, true, false);
  t(true, true, true);
}

void PointsToVolumesForwardCpu(
    const torch::Tensor& points_3d,
    const torch::Tensor& points_features,
    const torch::Tensor& volume_densities,
    const torch::Tensor& volume_features,
    const torch::Tensor& grid_sizes,
    const torch::Tensor& mask,
    const float point_weight,
    const bool align_corners,
    const bool splat) {
  const int64_t batch_size = points_3d.size(0);
  const int64_t P = points_3d.size(1);
  const int64_t n_features = points_features.size(2);

  // We unify the formula for the location of px in the comment above as
  // ({(px + 1) * 0.5} * (grid_size_x-scale_offset)) - offset.
  const int scale_offset = align_corners ? 1 : 0;
  const float offset = align_corners ? 0 : 0.5;

  auto points_3d_a = points_3d.accessor<float, 3>();
  auto points_features_a = points_features.accessor<float, 3>();
  auto volume_densities_a = volume_densities.accessor<float, 5>();
  auto volume_features_a = volume_features.accessor<float, 5>();
  auto grid_sizes_a = grid_sizes.accessor<int64_t, 2>();
  auto mask_a = mask.accessor<float, 2>();

  // For each batch element
  for (int64_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
    auto points_3d_aa = points_3d_a[batch_idx];
    auto points_features_aa = points_features_a[batch_idx];
    auto volume_densities_aa = volume_densities_a[batch_idx][0];
    auto volume_features_aa = volume_features_a[batch_idx];
    auto grid_sizes_aa = grid_sizes_a[batch_idx];
    auto mask_aa = mask_a[batch_idx];

    const int64_t grid_size_x = grid_sizes_aa[2];
    const int64_t grid_size_y = grid_sizes_aa[1];
    const int64_t grid_size_z = grid_sizes_aa[0];

    // For each point
    for (int64_t point_idx = 0; point_idx < P; ++point_idx) {
      // Ignore point if mask is 0
      if (mask_aa[point_idx] == 0) {
        continue;
      }
      auto point = points_3d_aa[point_idx];
      auto point_features = points_features_aa[point_idx];

      // Define how to increment a location in the volume by an amount. The need
      // for this depends on the interpolation method:
      // once per point for nearest, eight times for splat.
      auto increment_location =
          [&](int64_t x, int64_t y, int64_t z, float weight) {
            if (x >= grid_size_x || y >= grid_size_y || z >= grid_size_z) {
              return;
            }
            if (x < 0 || y < 0 || z < 0) {
              return;
            }

            volume_densities_aa[z][y][x] += weight * point_weight;

            for (int64_t feature_idx = 0; feature_idx < n_features;
                 ++feature_idx) {
              volume_features_aa[feature_idx][z][y][x] +=
                  point_features[feature_idx] * weight * point_weight;
            }
          };

      if (!splat) {
        // Increment the location nearest the point.
        long x = std::lround(
            (point[0] + 1) * 0.5 * (grid_size_x - scale_offset) - offset);
        long y = std::lround(
            (point[1] + 1) * 0.5 * (grid_size_y - scale_offset) - offset);
        long z = std::lround(
            (point[2] + 1) * 0.5 * (grid_size_z - scale_offset) - offset);
        increment_location(x, y, z, 1);
      } else {
        // There are 8 locations around the point which we need to worry about.
        // Their coordinates are (x or x+1, y or y+1, z or z+1).
        // rx is a number between 0 and 1 for the proportion in the x direction:
        // rx==0 means weight all on the lower bound, x, rx=1-eps means most
        // weight on x+1. Ditto for ry and yz.
        float x = 0, y = 0, z = 0;
        float rx = std::modf(
            (point[0] + 1) * 0.5 * (grid_size_x - scale_offset) - offset, &x);
        float ry = std::modf(
            (point[1] + 1) * 0.5 * (grid_size_y - scale_offset) - offset, &y);
        float rz = std::modf(
            (point[2] + 1) * 0.5 * (grid_size_z - scale_offset) - offset, &z);
        // Define how to fractionally increment one of the 8 locations around
        // the point.
        auto handle_point = [&](bool up_x, bool up_y, bool up_z) {
          float weight = (up_x ? rx : 1 - rx) * (up_y ? ry : 1 - ry) *
              (up_z ? rz : 1 - rz);
          increment_location(x + up_x, y + up_y, z + up_z, weight);
        };
        // and do so.
        EightDirections(handle_point);
      }
    }
  }
  torch::autograd::increment_version(volume_features);
  torch::autograd::increment_version(volume_densities);
}

// With nearest, the only smooth dependence is that volume features
// depend on points features.
//
// With splat, the dependencies are as follows, with gradients passing
// in the opposite direction.
//
//    points_3d         points_features
//         │  │                  │
//         │  │                  │
//         │  └───────────┐      │
//         │              │      │
//         │              │      │
//         ▼              ▼      ▼
// volume_densities    volume_features

// It is also the case that the input volume_densities and
// volume_features affect the corresponding outputs (they are
// modified in place).
// But the forward pass just increments these by a value which
// does not depend on them. So our autograd backwards pass needs
// to copy the gradient for each of those outputs to the
// corresponding input. We just do that in the Python layer.

void PointsToVolumesBackwardCpu(
    const torch::Tensor& points_3d,
    const torch::Tensor& points_features,
    const torch::Tensor& grid_sizes,
    const torch::Tensor& mask,
    const float point_weight,
    const bool align_corners,
    const bool splat,
    const torch::Tensor& grad_volume_densities,
    const torch::Tensor& grad_volume_features,
    const torch::Tensor& grad_points_3d,
    const torch::Tensor& grad_points_features) {
  const int64_t batch_size = points_3d.size(0);
  const int64_t P = points_3d.size(1);
  const int64_t n_features = grad_points_features.size(2);
  const int scale_offset = align_corners ? 1 : 0;
  const float offset = align_corners ? 0 : 0.5;

  auto points_3d_a = points_3d.accessor<float, 3>();
  auto points_features_a = points_features.accessor<float, 3>();
  auto grid_sizes_a = grid_sizes.accessor<int64_t, 2>();
  auto mask_a = mask.accessor<float, 2>();
  auto grad_volume_densities_a = grad_volume_densities.accessor<float, 5>();
  auto grad_volume_features_a = grad_volume_features.accessor<float, 5>();
  auto grad_points_3d_a = grad_points_3d.accessor<float, 3>();
  auto grad_points_features_a = grad_points_features.accessor<float, 3>();

  // For each batch element
  for (int64_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
    auto points_3d_aa = points_3d_a[batch_idx];
    auto points_features_aa = points_features_a[batch_idx];
    auto grid_sizes_aa = grid_sizes_a[batch_idx];
    auto mask_aa = mask_a[batch_idx];
    auto grad_volume_densities_aa = grad_volume_densities_a[batch_idx][0];
    auto grad_volume_features_aa = grad_volume_features_a[batch_idx];
    auto grad_points_3d_aa = grad_points_3d_a[batch_idx];
    auto grad_points_features_aa = grad_points_features_a[batch_idx];

    const int64_t grid_size_x = grid_sizes_aa[2];
    const int64_t grid_size_y = grid_sizes_aa[1];
    const int64_t grid_size_z = grid_sizes_aa[0];

    // For each point
    for (int64_t point_idx = 0; point_idx < P; ++point_idx) {
      if (mask_aa[point_idx] == 0) {
        continue;
      }
      auto point = points_3d_aa[point_idx];
      auto point_features = points_features_aa[point_idx];
      auto grad_point_features = grad_points_features_aa[point_idx];
      auto grad_point = grad_points_3d_aa[point_idx];

      // Define how to (backwards) increment a location in the point cloud,
      // to take gradients to the features.
      // We return false if the location does not really exist, so there was
      // nothing to do.
      // This happens once per point for nearest, eight times for splat.
      auto increment_location =
          [&](int64_t x, int64_t y, int64_t z, float weight) {
            if (x >= grid_size_x || y >= grid_size_y || z >= grid_size_z) {
              return false;
            }
            if (x < 0 || y < 0 || z < 0) {
              return false;
            }

            for (int64_t feature_idx = 0; feature_idx < n_features;
                 ++feature_idx) {
              // This is a forward line, for comparison
              // volume_features_aa[feature_idx][z][y][x] +=
              //    point_features[feature_idx] * weight * point_weight;
              grad_point_features[feature_idx] +=
                  grad_volume_features_aa[feature_idx][z][y][x] * weight *
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
          // For each of the eight locations, we first increment the feature
          // gradient.
          if (increment_location(x + up_x, y + up_y, z + up_z, weight)) {
            // If the location is a real location, we also (in this splat
            // case) need to update the gradient w.r.t. the point position.
            // - the amount in this location is controlled by the weight.
            // There are two contributions:
            //  (1) The point position affects how much density we added
            //      to the location's density, so we have a contribution
            //      from grad_volume_density. Specifically,
            //      weight * point_weight has been added to
            //      volume_densities_aa[z+up_z][y+up_y][x+up_x]
            //
            //  (2) The point position affects how much of each of the
            //      point's features were added to the corresponding feature
            //      of this location, so we have a contribution from
            //      grad_volume_features. Specifically, for each feature_idx,
            //      point_features[feature_idx] * weight * point_weight
            //      has been added to
            //      volume_features_aa[feature_idx][z+up_z][y+up_y][x+up_x]

            float source_gradient =
                grad_volume_densities_aa[z + up_z][y + up_y][x + up_x];
            for (int64_t feature_idx = 0; feature_idx < n_features;
                 ++feature_idx) {
              source_gradient += point_features[feature_idx] *
                  grad_volume_features_aa[feature_idx][z + up_z][y + up_y]
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
}
