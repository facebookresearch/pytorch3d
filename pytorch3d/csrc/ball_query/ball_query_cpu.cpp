/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <torch/extension.h>
#include <tuple>

std::tuple<at::Tensor, at::Tensor> BallQueryCpu(
    const at::Tensor& p1,
    const at::Tensor& p2,
    const at::Tensor& lengths1,
    const at::Tensor& lengths2,
    int K,
    float radius) {
  const int N = p1.size(0);
  const int P1 = p1.size(1);
  const int D = p1.size(2);

  auto long_opts = lengths1.options().dtype(torch::kInt64);
  torch::Tensor idxs = torch::full({N, P1, K}, -1, long_opts);
  torch::Tensor dists = torch::full({N, P1, K}, 0, p1.options());
  const float radius2 = radius * radius;

  auto p1_a = p1.accessor<float, 3>();
  auto p2_a = p2.accessor<float, 3>();
  auto lengths1_a = lengths1.accessor<int64_t, 1>();
  auto lengths2_a = lengths2.accessor<int64_t, 1>();
  auto idxs_a = idxs.accessor<int64_t, 3>();
  auto dists_a = dists.accessor<float, 3>();

  for (int n = 0; n < N; ++n) {
    const int64_t length1 = lengths1_a[n];
    const int64_t length2 = lengths2_a[n];
    for (int64_t i = 0; i < length1; ++i) {
      for (int64_t j = 0, count = 0; j < length2 && count < K; ++j) {
        float dist2 = 0;
        for (int d = 0; d < D; ++d) {
          float diff = p1_a[n][i][d] - p2_a[n][j][d];
          dist2 += diff * diff;
        }
        if (dist2 < radius2) {
          dists_a[n][i][count] = dist2;
          idxs_a[n][i][count] = j;
          ++count;
        }
      }
    }
  }
  return std::make_tuple(idxs, dists);
}
