// Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

#include <torch/extension.h>
#include <ATen/ATen.h>

template <typename scalar_t>
void NearestNeighborIdxCpuKernel(
    const at::Tensor& p1,
    const at::Tensor& p2,
    at::Tensor& out,
    const size_t N,
    const size_t D,
    const size_t P1,
    const size_t P2) {

  auto p1_a = p1.accessor<scalar_t, 3>();
  auto p2_a = p2.accessor<scalar_t, 3>();
  auto out_a = out.accessor<int64_t, 2>();

  for (size_t n = 0; n < N; ++n) {
    for (size_t i1 = 0; i1 < P1; ++i1) {
      scalar_t min_dist = -1;
      int64_t min_idx = -1;
      for (int64_t i2 = 0; i2 < P2; ++i2) {
        scalar_t dist = 0;
        for (size_t d = 0; d < D; ++d) {
          scalar_t diff = p1_a[n][i1][d] - p2_a[n][i2][d];
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
}

at::Tensor NearestNeighborIdxCpu(const at::Tensor& p1, const at::Tensor& p2) {
  const size_t N  = p1.size(0);
  const size_t P1 = p1.size(1);
  const size_t D  = p1.size(2);
  const size_t P2 = p2.size(1);

  auto long_opts = p1.options().dtype(torch::kInt64);
  torch::Tensor out = torch::empty({N, P1}, long_opts);

  AT_DISPATCH_FLOATING_TYPES(p1.type(), "nearest_neighbor_idx_cpu", [&] {
      NearestNeighborIdxCpuKernel<scalar_t>(
          p1,
          p2,
          out,
          N,
          D,
          P1,
          P2
      );
  });

  return out;
}
