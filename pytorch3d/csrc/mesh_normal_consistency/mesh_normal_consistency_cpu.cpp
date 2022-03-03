/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <utility>
#include <vector>

at::Tensor MeshNormalConsistencyFindVerticesCpu(const at::Tensor& edge_num) {
  // We take a LongTensor of shape (E,) giving the number of things intersecting
  // each edge. The things are taken to be numbered in order.
  // (In fact, the "things" are opposite vertices to edges, renumbered).
  // We return a tensor of shape (?, 2) where for every pair of things which
  // intersect the same edge there is a row of their numbers in the output.

  // Example possible inputs and outputs (order of output is not specified):
  //  [1,0,1,1,0] => [[]]
  //          [3] => [[0,1], [0,2], [1,2]]
  //        [0,3] => [[0,1], [0,2], [1,2]]
  //        [1,3] => [[1,2], [1,3], [2,3]]
  //[1,0,2,1,0,2] => [[1,2], [4,5]]

  const auto num_edges = edge_num.size(0);
  auto edges_a = edge_num.accessor<int64_t, 1>();

  int64_t vert_idx = 0;
  std::vector<std::pair<int64_t, int64_t>> pairs;
  for (int64_t i_edge = 0; i_edge < num_edges; ++i_edge) {
    int64_t e = edges_a[i_edge];
    for (int64_t j = 0; j < e; ++j) {
      for (int64_t i = 0; i < j; ++i) {
        pairs.emplace_back(vert_idx + i, vert_idx + j);
      }
    }
    vert_idx += e;
  }

  // Convert from std::vector by copying over the items to a new empty torch
  // tensor.
  auto pairs_tensor = at::empty({(int64_t)pairs.size(), 2}, edge_num.options());
  auto pairs_a = pairs_tensor.accessor<int64_t, 2>();
  for (int64_t i_pair = 0; i_pair < pairs.size(); ++i_pair) {
    auto accessor = pairs_a[i_pair];
    accessor[0] = pairs[i_pair].first;
    accessor[1] = pairs[i_pair].second;
  }

  return pairs_tensor;
}
