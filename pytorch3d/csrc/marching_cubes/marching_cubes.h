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
//    vertices: (N_verts, 3) FloatTensor of vertices
//    faces: (N_faces, 3) LongTensor of faces
//    ids: (N_verts,) LongTensor used to identify each vertex and deduplication
//         to avoid floating point precision issues.
//         For Cuda, will be used to dedupe redundant vertices.
//         For cpp implementation, this tensor is just a placeholder.

// CPU implementation
std::tuple<at::Tensor, at::Tensor, at::Tensor> MarchingCubesCpu(
    const at::Tensor& vol,
    const float isolevel);

// CUDA implementation
std::tuple<at::Tensor, at::Tensor, at::Tensor> MarchingCubesCuda(
    const at::Tensor& vol,
    const float isolevel);

// Implementation which is exposed
inline std::tuple<at::Tensor, at::Tensor, at::Tensor> MarchingCubes(
    const at::Tensor& vol,
    const float isolevel) {
  if (vol.is_cuda()) {
#ifdef WITH_CUDA
    CHECK_CUDA(vol);
    const int D = vol.size(0);
    const int H = vol.size(1);
    const int W = vol.size(2);
    if (D > 1024 || H > 1024 || W > 1024) {
      AT_ERROR("Maximum volume size allowed 1K x 1K x 1K");
    }
    return MarchingCubesCuda(vol.contiguous(), isolevel);
#else
    AT_ERROR("Not compiled with GPU support.");
#endif
  }
  return MarchingCubesCpu(vol.contiguous(), isolevel);
}
