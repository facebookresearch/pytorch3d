/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>

at::Tensor GatherScatterCpu(
    const at::Tensor& input,
    const at::Tensor& edges,
    bool directed,
    bool backward) {
  const auto num_vertices = input.size(0);
  const auto input_feature_dim = input.size(1);
  const auto num_edges = edges.size(0);

  auto output = at::zeros({num_vertices, input_feature_dim}, input.options());

  auto input_a = input.accessor<float, 2>();
  auto edges_a = edges.accessor<int64_t, 2>();
  auto output_a = output.accessor<float, 2>();
  const int v0_idx = backward ? 1 : 0;
  const int v1_idx = backward ? 0 : 1;

  for (int e = 0; e < num_edges; ++e) {
    // Get indices of vertices which form the edge.
    const int64_t v0 = edges_a[e][v0_idx];
    const int64_t v1 = edges_a[e][v1_idx];

    for (int d = 0; d < input_feature_dim; ++d) {
      output_a[v0][d] += input_a[v1][d];
      if (!directed) {
        output_a[v1][d] += input_a[v0][d];
      }
    }
  }
  return output;
}
