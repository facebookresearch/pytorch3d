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

// For mesh_normal_consistency, find pairs of vertices opposite the same edge.
//
// Args:
//   edge_num: int64 Tensor of shape (E,) giving the number of vertices
//              corresponding to each edge.
//
// Returns:
//    pairs: int64 Tensor of shape (N,2)

at::Tensor MeshNormalConsistencyFindVerticesCpu(const at::Tensor& edge_num);

// Exposed implementation.
at::Tensor MeshNormalConsistencyFindVertices(const at::Tensor& edge_num) {
  if (edge_num.is_cuda()) {
    AT_ERROR("This function needs a CPU tensor.");
  }
  return MeshNormalConsistencyFindVerticesCpu(edge_num);
}
