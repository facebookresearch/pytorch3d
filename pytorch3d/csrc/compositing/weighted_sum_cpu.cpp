/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <torch/extension.h>

#include <cmath>
#include <vector>

torch::Tensor weightedSumCpuForward(
    const torch::Tensor& features,
    const torch::Tensor& alphas,
    const torch::Tensor& points_idx) {
  const int64_t B = points_idx.size(0);
  const int64_t K = points_idx.size(1);
  const int64_t H = points_idx.size(2);
  const int64_t W = points_idx.size(3);
  const int64_t C = features.size(0);

  torch::Tensor result = torch::zeros({B, C, H, W}, features.options());

  auto features_a = features.accessor<float, 2>();
  auto alphas_a = alphas.accessor<float, 4>();
  auto points_idx_a = points_idx.accessor<int64_t, 4>();
  auto result_a = result.accessor<float, 4>();

  // Iterate over the batch
  for (int b = 0; b < B; ++b) {
    // Iterate over the features
    for (int c = 0; c < C; ++c) {
      // Iterate through the horizontal lines of the image from top to bottom
      for (int j = 0; j < H; ++j) {
        // Iterate over pixels in a horizontal line, left to right
        for (int i = 0; i < W; ++i) {
          // Iterate through the closest K points for this pixel
          for (int k = 0; k < K; ++k) {
            int64_t n_idx = points_idx_a[b][k][j][i];
            // Sentinel value is -1 indicating no point overlaps the pixel
            if (n_idx < 0) {
              continue;
            }

            float alpha = alphas_a[b][k][j][i];
            result_a[b][c][j][i] += alpha * features_a[c][n_idx];
          }
        }
      }
    }
  }
  return result;
}

std::tuple<torch::Tensor, torch::Tensor> weightedSumCpuBackward(
    const torch::Tensor& grad_outputs,
    const torch::Tensor& features,
    const torch::Tensor& alphas,
    const torch::Tensor& points_idx) {
  const int64_t B = points_idx.size(0);
  const int64_t K = points_idx.size(1);
  const int64_t H = points_idx.size(2);
  const int64_t W = points_idx.size(3);
  const int64_t C = features.size(0);

  torch::Tensor grad_features = torch::zeros_like(features);
  torch::Tensor grad_alphas = torch::zeros_like(alphas);

  auto grad_outputs_a = grad_outputs.accessor<float, 4>();
  auto features_a = features.accessor<float, 2>();
  auto alphas_a = alphas.accessor<float, 4>();
  auto points_idx_a = points_idx.accessor<int64_t, 4>();
  auto grad_features_a = grad_features.accessor<float, 2>();
  auto grad_alphas_a = grad_alphas.accessor<float, 4>();

  // Iterate over the batch
  for (int b = 0; b < B; ++b) {
    // Iterate oer the features
    for (int c = 0; c < C; ++c) {
      // Iterate through the horizontal lines of the image from top to bottom
      for (int j = 0; j < H; ++j) {
        // Iterate over pixels in a horizontal line, left to right
        for (int i = 0; i < W; ++i) {
          // Iterate through the closest K points for this pixel
          for (int k = 0; k < K; ++k) {
            int64_t n_idx = points_idx_a[b][k][j][i];
            // Sentinal value is -1, indicating no point overlaps this pixel
            if (n_idx < 0) {
              continue;
            }

            float alpha = alphas_a[b][k][j][i];
            grad_alphas_a[b][k][j][i] +=
                grad_outputs_a[b][c][j][i] * features_a[c][n_idx];
            grad_features_a[c][n_idx] += grad_outputs_a[b][c][j][i] * alpha;
          }
        }
      }
    }
  }
  return std::make_tuple(grad_features, grad_alphas);
}
