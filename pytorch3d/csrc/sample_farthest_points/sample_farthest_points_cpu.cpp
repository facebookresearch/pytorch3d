/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <torch/extension.h>
#include <iterator>
#include <random>
#include <vector>

at::Tensor FarthestPointSamplingCpu(
    const at::Tensor& points,
    const at::Tensor& lengths,
    const at::Tensor& K,
    const at::Tensor& start_idxs) {
  // Get constants
  const int64_t N = points.size(0);
  const int64_t P = points.size(1);
  const int64_t D = points.size(2);
  const int64_t max_K = torch::max(K).item<int64_t>();

  // Initialize an output array for the sampled indices
  // of shape (N, max_K)
  auto long_opts = lengths.options();
  torch::Tensor sampled_indices = torch::full({N, max_K}, -1, long_opts);

  // Create accessors for all tensors
  auto points_a = points.accessor<float, 3>();
  auto lengths_a = lengths.accessor<int64_t, 1>();
  auto k_a = K.accessor<int64_t, 1>();
  auto sampled_indices_a = sampled_indices.accessor<int64_t, 2>();
  auto start_idxs_a = start_idxs.accessor<int64_t, 1>();

  // Initialize a mask to prevent duplicates
  // If true, the point has already been selected.
  std::vector<unsigned char> selected_points_mask(P, false);

  // Initialize to infinity a vector of
  // distances from each point to any of the previously selected points
  std::vector<float> dists(P, std::numeric_limits<float>::max());

  for (int64_t n = 0; n < N; ++n) {
    // Resize and reset points mask and distances for each batch
    selected_points_mask.resize(lengths_a[n]);
    dists.resize(lengths_a[n]);
    std::fill(selected_points_mask.begin(), selected_points_mask.end(), false);
    std::fill(dists.begin(), dists.end(), std::numeric_limits<float>::max());

    // Get the starting point index and save it
    int64_t last_idx = start_idxs_a[n];
    sampled_indices_a[n][0] = last_idx;

    // Set the value of the mask at this point to false
    selected_points_mask[last_idx] = true;

    // For heterogeneous pointclouds, use the minimum of the
    // length for that cloud compared to K as the number of
    // points to sample
    const int64_t batch_k = std::min(lengths_a[n], k_a[n]);

    // Iteratively select batch_k points per batch
    for (int64_t k = 1; k < batch_k; ++k) {
      // Iterate through all the points
      for (int64_t p = 0; p < lengths_a[n]; ++p) {
        if (selected_points_mask[p]) {
          // For already selected points set the distance to 0.0
          dists[p] = 0.0;
          continue;
        }

        // Calculate the distance to the last selected point
        float dist2 = 0.0;
        for (int64_t d = 0; d < D; ++d) {
          float diff = points_a[n][last_idx][d] - points_a[n][p][d];
          dist2 += diff * diff;
        }

        // If the distance of this point to the last selected point is closer
        // than the distance to any of the previously selected points, then
        // update this distance
        if (dist2 < dists[p]) {
          dists[p] = dist2;
        }
      }

      // The aim is to pick the point that has the largest
      // nearest neighbour distance to any of the already selected points
      auto itr = std::max_element(dists.begin(), dists.end());
      last_idx = std::distance(dists.begin(), itr);

      // Save selected point
      sampled_indices_a[n][k] = last_idx;

      // Set the mask value to true to prevent duplicates.
      selected_points_mask[last_idx] = true;
    }
  }

  return sampled_indices;
}
