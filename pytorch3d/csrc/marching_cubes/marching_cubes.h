/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <torch/extension.h>
#include <tuple>
#include <vector>
#include "utils/pytorch3d_cutils.h"

// Run Marching Cubes algorithm over a batch of volume scalar fields
// with a pre-defined threshold and return a mesh composed of vertices
// and faces for the mesh.
//
// Args:
//    vol: FloatTensor of shape (D, H, W) giving a volume
//    scalar grids.
//    isolevel: isosurface value to use as the threshoold to determine whether
//    the points are within a volume.
//
// Returns:
//    vertices: List of N FloatTensors of vertices
//    faces:    List of N LongTensors of faces

// CPU implementation
std::tuple<at::Tensor, at::Tensor> MarchingCubesCpu(
    const at::Tensor& vol,
    const float isolevel);

// Implementation which is exposed
inline std::tuple<at::Tensor, at::Tensor> MarchingCubes(
    const at::Tensor& vol,
    const float isolevel) {
  return MarchingCubesCpu(vol.contiguous(), isolevel);
}
