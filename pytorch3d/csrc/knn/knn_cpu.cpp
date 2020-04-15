// Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

#include <torch/extension.h>
#include <queue>
#include <tuple>

std::tuple<at::Tensor, at::Tensor> KNearestNeighborIdxCpu(
    const at::Tensor& p1,
    const at::Tensor& p2,
    const at::Tensor& lengths1,
    const at::Tensor& lengths2,
    int K) {
  const int N = p1.size(0);
  const int P1 = p1.size(1);
  const int D = p1.size(2);
  const int P2 = p2.size(1);

  auto long_opts = p1.options().dtype(torch::kInt64);
  torch::Tensor idxs = torch::full({N, P1, K}, 0, long_opts);
  torch::Tensor dists = torch::full({N, P1, K}, 0, p1.options());

  auto p1_a = p1.accessor<float, 3>();
  auto p2_a = p2.accessor<float, 3>();
  auto lengths1_a = lengths1.accessor<int64_t, 1>();
  auto lengths2_a = lengths2.accessor<int64_t, 1>();
  auto idxs_a = idxs.accessor<int64_t, 3>();
  auto dists_a = dists.accessor<float, 3>();

  for (int n = 0; n < N; ++n) {
    const int64_t length1 = lengths1_a[n];
    const int64_t length2 = lengths2_a[n];
    for (int64_t i1 = 0; i1 < length1; ++i1) {
      // Use a priority queue to store (distance, index) tuples.
      std::priority_queue<std::tuple<float, int>> q;
      for (int64_t i2 = 0; i2 < length2; ++i2) {
        float dist = 0;
        for (int d = 0; d < D; ++d) {
          float diff = p1_a[n][i1][d] - p2_a[n][i2][d];
          dist += diff * diff;
        }
        int size = static_cast<int>(q.size());
        if (size < K || dist < std::get<0>(q.top())) {
          q.emplace(dist, i2);
          if (size >= K) {
            q.pop();
          }
        }
      }
      while (!q.empty()) {
        auto t = q.top();
        q.pop();
        const int k = q.size();
        dists_a[n][i1][k] = std::get<0>(t);
        idxs_a[n][i1][k] = std::get<1>(t);
      }
    }
  }
  return std::make_tuple(idxs, dists);
}

// ------------------------------------------------------------- //
//                   Backward Operators                          //
// ------------------------------------------------------------- //

std::tuple<at::Tensor, at::Tensor> KNearestNeighborBackwardCpu(
    const at::Tensor& p1,
    const at::Tensor& p2,
    const at::Tensor& lengths1,
    const at::Tensor& lengths2,
    const at::Tensor& idxs,
    const at::Tensor& grad_dists) {
  const int N = p1.size(0);
  const int P1 = p1.size(1);
  const int D = p1.size(2);
  const int P2 = p2.size(1);
  const int K = idxs.size(2);

  torch::Tensor grad_p1 = torch::full({N, P1, D}, 0, p1.options());
  torch::Tensor grad_p2 = torch::full({N, P2, D}, 0, p2.options());

  auto p1_a = p1.accessor<float, 3>();
  auto p2_a = p2.accessor<float, 3>();
  auto lengths1_a = lengths1.accessor<int64_t, 1>();
  auto lengths2_a = lengths2.accessor<int64_t, 1>();
  auto idxs_a = idxs.accessor<int64_t, 3>();
  auto grad_dists_a = grad_dists.accessor<float, 3>();
  auto grad_p1_a = grad_p1.accessor<float, 3>();
  auto grad_p2_a = grad_p2.accessor<float, 3>();

  for (int n = 0; n < N; ++n) {
    const int64_t length1 = lengths1_a[n];
    int64_t length2 = lengths2_a[n];
    length2 = (length2 < K) ? length2 : K;
    for (int64_t i1 = 0; i1 < length1; ++i1) {
      for (int64_t k = 0; k < length2; ++k) {
        const int64_t i2 = idxs_a[n][i1][k];
        for (int64_t d = 0; d < D; ++d) {
          const float diff =
              2.0f * grad_dists_a[n][i1][k] * (p1_a[n][i1][d] - p2_a[n][i2][d]);
          grad_p1_a[n][i1][d] += diff;
          grad_p2_a[n][i2][d] += -1.0f * diff;
        }
      }
    }
  }
  return std::make_tuple(grad_p1, grad_p2);
}
