// Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

#include <torch/extension.h>

at::Tensor NearestNeighborIdxCpu(at::Tensor p1, at::Tensor p2) {
  const int N = p1.size(0);
  const int P1 = p1.size(1);
  const int D = p1.size(2);
  const int P2 = p2.size(1);

  auto long_opts = p1.options().dtype(torch::kInt64);
  torch::Tensor out = torch::empty({N, P1}, long_opts);

  auto p1_a = p1.accessor<float, 3>();
  auto p2_a = p2.accessor<float, 3>();
  auto out_a = out.accessor<int64_t, 2>();

  for (int n = 0; n < N; ++n) {
    for (int i1 = 0; i1 < P1; ++i1) {
      // TODO: support other floating-point types?
      float min_dist = -1;
      int64_t min_idx = -1;
      for (int i2 = 0; i2 < P2; ++i2) {
        float dist = 0;
        for (int d = 0; d < D; ++d) {
          float diff = p1_a[n][i1][d] - p2_a[n][i2][d];
          dist += diff * diff;
        }
        if (min_dist == -1 || dist < min_dist) {
          min_dist = dist;
          min_idx = i2;
        }
      }
      out_a[n][i1] = min_idx;
    }
  }
  return out;
}
