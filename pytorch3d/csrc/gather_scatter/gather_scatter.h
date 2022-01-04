/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <torch/extension.h>
#include "utils/pytorch3d_cutils.h"

// Fused gather scatter operation for aggregating features of neighbor nodes
// in a graph. This gather scatter operation is specific to graphs as edge
// indices are used as input.
//
// Args:
//   input: float32 Tensor of shape (V, D) where V is the number of vertices
//          and D is the feature dimension.
//   edges: int64 Tensor of shape (E, 2) giving the indices of the vertices that
//          make up the edge. E is the number of edges.
//  directed: Bool indicating if edges in the graph are directed. For a
//            directed graph v0 -> v1 the updated feature for v0 depends on v1.
//  backward: Bool indicating if the operation is the backward pass.
//
// Returns:
//   output: float32 Tensor of same shape as input.

at::Tensor GatherScatterCuda(
    const at::Tensor& input,
    const at::Tensor& edges,
    bool directed,
    bool backward);

at::Tensor GatherScatterCpu(
    const at::Tensor& input,
    const at::Tensor& edges,
    bool directed,
    bool backward);

// Exposed implementation.
at::Tensor GatherScatter(
    const at::Tensor& input,
    const at::Tensor& edges,
    bool directed,
    bool backward) {
  if (input.is_cuda() && edges.is_cuda()) {
#ifdef WITH_CUDA
    CHECK_CUDA(input);
    CHECK_CUDA(edges);
    return GatherScatterCuda(input, edges, directed, backward);
#else
    AT_ERROR("Not compiled with GPU support.");
#endif
  }
  return GatherScatterCpu(input, edges, directed, backward);
}
