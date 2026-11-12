/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/Parallel.h>
#include <torch/extension.h>
#include <tuple>
#include "iou_box3d/iou_utils.h"

std::tuple<at::Tensor, at::Tensor> IoUBox3DCpu(
    const at::Tensor& boxes1,
    const at::Tensor& boxes2) {
  TORCH_CHECK(
      boxes1.dim() == 3 && boxes2.dim() == 3 && boxes1.size(1) == 8 &&
          boxes1.size(2) == 3 && boxes2.size(1) == 8 && boxes2.size(2) == 3,
      "Boxes must have shape (N, 8, 3)");
  const int64_t N = boxes1.size(0);
  const int64_t M = boxes2.size(0);
  auto float_opts = boxes1.options().dtype(torch::kFloat32);
  torch::Tensor vols = torch::zeros({N, M}, float_opts);
  torch::Tensor ious = torch::zeros({N, M}, float_opts);

  const float* boxes1_data = boxes1.data_ptr<float>();
  const float* boxes2_data = boxes2.data_ptr<float>();
  float* vols_data = vols.data_ptr<float>();
  float* ious_data = ious.data_ptr<float>();

  at::parallel_for(0, N * M, 64, [&](int64_t begin, int64_t end) {
    int64_t n = begin / M;
    int64_t m = begin % M;
    for (int64_t i = begin; i < end; ++i) {
      iou_box3d::BoxOverlap(
          boxes1_data + 24 * n,
          boxes2_data + 24 * m,
          vols_data[i],
          ious_data[i]);
      if (++m == M) {
        m = 0;
        ++n;
      }
    }
  });
  return std::make_tuple(vols, ious);
}
