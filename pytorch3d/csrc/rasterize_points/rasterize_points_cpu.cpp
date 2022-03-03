/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <torch/extension.h>
#include <queue>
#include <tuple>
#include "rasterization_utils.h"

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> RasterizePointsNaiveCpu(
    const torch::Tensor& points, // (P, 3)
    const torch::Tensor& cloud_to_packed_first_idx, // (N)
    const torch::Tensor& num_points_per_cloud, // (N)
    const std::tuple<int, int> image_size,
    const torch::Tensor& radius,
    const int points_per_pixel) {
  const int32_t N = cloud_to_packed_first_idx.size(0); // batch_size.

  const int H = std::get<0>(image_size);
  const int W = std::get<1>(image_size);
  const int K = points_per_pixel;

  // Initialize output tensors.
  auto int_opts = num_points_per_cloud.options().dtype(torch::kInt32);
  auto float_opts = points.options().dtype(torch::kFloat32);
  torch::Tensor point_idxs = torch::full({N, H, W, K}, -1, int_opts);
  torch::Tensor zbuf = torch::full({N, H, W, K}, -1, float_opts);
  torch::Tensor pix_dists = torch::full({N, H, W, K}, -1, float_opts);

  auto points_a = points.accessor<float, 2>();
  auto point_idxs_a = point_idxs.accessor<int32_t, 4>();
  auto zbuf_a = zbuf.accessor<float, 4>();
  auto pix_dists_a = pix_dists.accessor<float, 4>();
  auto radius_a = radius.accessor<float, 1>();

  for (int n = 0; n < N; ++n) {
    // Loop through each pointcloud in the batch.
    // Get the start index of the points in points_packed and the num points
    // in the point cloud.
    const int point_start_idx =
        cloud_to_packed_first_idx[n].item().to<int32_t>();
    const int point_stop_idx =
        (point_start_idx + num_points_per_cloud[n].item().to<int32_t>());

    for (int yi = 0; yi < H; ++yi) {
      // Reverse the order of yi so that +Y is pointing upwards in the image.
      const int yidx = H - 1 - yi;
      const float yf = PixToNonSquareNdc(yidx, H, W);

      for (int xi = 0; xi < W; ++xi) {
        // Reverse the order of xi so that +X is pointing to the left in the
        // image.
        const int xidx = W - 1 - xi;
        const float xf = PixToNonSquareNdc(xidx, W, H);

        // Use a priority queue to hold (z, idx, r)
        std::priority_queue<std::tuple<float, int, float>> q;
        for (int p = point_start_idx; p < point_stop_idx; ++p) {
          const float px = points_a[p][0];
          const float py = points_a[p][1];
          const float pz = points_a[p][2];
          const float p_radius = radius_a[p];
          const float radius2 = p_radius * p_radius;
          if (pz < 0) {
            continue;
          }
          const float dx = px - xf;
          const float dy = py - yf;
          const float dist2 = dx * dx + dy * dy;
          if (dist2 < radius2) {
            // The current point hit the current pixel
            q.emplace(pz, p, dist2);
            if ((int)q.size() > K) {
              q.pop();
            }
          }
        }
        // Now all the points have been seen, so pop elements off the queue
        // one by one and write them into the output tensors.
        while (!q.empty()) {
          auto t = q.top();
          q.pop();
          int i = q.size();
          zbuf_a[n][yi][xi][i] = std::get<0>(t);
          point_idxs_a[n][yi][xi][i] = std::get<1>(t);
          pix_dists_a[n][yi][xi][i] = std::get<2>(t);
        }
      }
    }
  }
  return std::make_tuple(point_idxs, zbuf, pix_dists);
}

torch::Tensor RasterizePointsCoarseCpu(
    const torch::Tensor& points, // (P, 3)
    const torch::Tensor& cloud_to_packed_first_idx, // (N)
    const torch::Tensor& num_points_per_cloud, // (N)
    const std::tuple<int, int> image_size,
    const torch::Tensor& radius,
    const int bin_size,
    const int max_points_per_bin) {
  const int32_t N = cloud_to_packed_first_idx.size(0); // batch_size.
  const int M = max_points_per_bin;

  const float H = std::get<0>(image_size);
  const float W = std::get<1>(image_size);

  // Integer division round up.
  const int BH = 1 + (H - 1) / bin_size;
  const int BW = 1 + (W - 1) / bin_size;

  auto opts = num_points_per_cloud.options().dtype(torch::kInt32);
  torch::Tensor points_per_bin = torch::zeros({N, BH, BW}, opts);
  torch::Tensor bin_points = torch::full({N, BH, BW, M}, -1, opts);

  auto points_a = points.accessor<float, 2>();
  auto points_per_bin_a = points_per_bin.accessor<int32_t, 3>();
  auto bin_points_a = bin_points.accessor<int32_t, 4>();
  auto radius_a = radius.accessor<float, 1>();

  const float ndc_x_range = NonSquareNdcRange(W, H);
  const float pixel_width_x = ndc_x_range / W;
  const float bin_width_x = pixel_width_x * bin_size;

  const float ndc_y_range = NonSquareNdcRange(H, W);
  const float pixel_width_y = ndc_y_range / H;
  const float bin_width_y = pixel_width_y * bin_size;

  for (int n = 0; n < N; ++n) {
    // Loop through each pointcloud in the batch.
    // Get the start index of the points in points_packed and the num points
    // in the point cloud.
    const int point_start_idx =
        cloud_to_packed_first_idx[n].item().to<int32_t>();
    const int point_stop_idx =
        (point_start_idx + num_points_per_cloud[n].item().to<int32_t>());

    float bin_y_min = -1.0f;
    float bin_y_max = bin_y_min + bin_width_y;

    // Iterate through the horizontal bins from top to bottom.
    for (int by = 0; by < BH; by++) {
      float bin_x_min = -1.0f;
      float bin_x_max = bin_x_min + bin_width_x;

      // Iterate through bins on this horizontal line, left to right.
      for (int bx = 0; bx < BW; bx++) {
        int32_t points_hit = 0;
        for (int p = point_start_idx; p < point_stop_idx; ++p) {
          float px = points_a[p][0];
          float py = points_a[p][1];
          float pz = points_a[p][2];
          const float p_radius = radius_a[p];
          if (pz < 0) {
            continue;
          }
          float point_x_min = px - p_radius;
          float point_x_max = px + p_radius;
          float point_y_min = py - p_radius;
          float point_y_max = py + p_radius;

          // Use a half-open interval so that points exactly on the
          // boundary between bins will fall into exactly one bin.
          bool x_hit = (point_x_min <= bin_x_max) && (bin_x_min <= point_x_max);
          bool y_hit = (point_y_min <= bin_y_max) && (bin_y_min <= point_y_max);
          if (x_hit && y_hit) {
            // Got too many points for this bin, so throw an error.
            if (points_hit >= max_points_per_bin) {
              AT_ERROR("Got too many points per bin");
            }
            // The current point falls in the current bin, so
            // record it.
            bin_points_a[n][by][bx][points_hit] = p;
            points_hit++;
          }
        }
        // Record the number of points found in this bin
        points_per_bin_a[n][by][bx] = points_hit;

        // Shift the bin to the right for the next loop iteration
        bin_x_min = bin_x_max;
        bin_x_max = bin_x_min + bin_width_x;
      }
      // Shift the bin down for the next loop iteration
      bin_y_min = bin_y_max;
      bin_y_max = bin_y_min + bin_width_y;
    }
  }
  return bin_points;
}

torch::Tensor RasterizePointsBackwardCpu(
    const torch::Tensor& points, // (P, 3)
    const torch::Tensor& idxs, // (N, H, W, K)
    const torch::Tensor& grad_zbuf, // (N, H, W, K)
    const torch::Tensor& grad_dists) { // (N, H, W, K)

  const int N = idxs.size(0);
  const int P = points.size(0);
  const int H = idxs.size(1);
  const int W = idxs.size(2);
  const int K = idxs.size(3);

  torch::Tensor grad_points = torch::zeros({P, 3}, points.options());

  auto points_a = points.accessor<float, 2>();
  auto idxs_a = idxs.accessor<int32_t, 4>();
  auto grad_dists_a = grad_dists.accessor<float, 4>();
  auto grad_zbuf_a = grad_zbuf.accessor<float, 4>();
  auto grad_points_a = grad_points.accessor<float, 2>();

  for (int n = 0; n < N; ++n) { // Loop over images in the batch
    for (int y = 0; y < H; ++y) { // Loop over rows in the image
      // Reverse the order of yi so that +Y is pointing upwards in the image.
      const int yidx = H - 1 - y;
      // Y coordinate of the top of the pixel.
      const float yf = PixToNonSquareNdc(yidx, H, W);

      // Iterate through pixels on this horizontal line, left to right.
      for (int x = 0; x < W; ++x) { // Loop over pixels in the row

        // Reverse the order of xi so that +X is pointing to the left in the
        // image.
        const int xidx = W - 1 - x;
        const float xf = PixToNonSquareNdc(xidx, W, H);
        for (int k = 0; k < K; ++k) { // Loop over points for the pixel
          const int p = idxs_a[n][y][x][k];
          if (p < 0) {
            break;
          }
          const float grad_dist2 = grad_dists_a[n][y][x][k];
          const float px = points_a[p][0];
          const float py = points_a[p][1];
          const float dx = px - xf;
          const float dy = py - yf;
          // Remember: dists[n][y][x][k] = dx * dx + dy * dy;
          const float grad_px = 2.0f * grad_dist2 * dx;
          const float grad_py = 2.0f * grad_dist2 * dy;
          grad_points_a[p][0] += grad_px;
          grad_points_a[p][1] += grad_py;
          grad_points_a[p][2] += grad_zbuf_a[n][y][x][k];
        }
      }
    }
  }
  return grad_points;
}
