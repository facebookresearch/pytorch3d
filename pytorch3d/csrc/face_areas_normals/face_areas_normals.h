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
#include "utils/pytorch3d_cutils.h"

// Compute areas of mesh faces using packed representation.
//
// Inputs:
//    verts: FloatTensor of shape (V, 3) giving vertex positions.
//    faces: LongTensor of shape (F, 3) giving faces.
//
// Returns:
//    areas: FloatTensor of shape (F,) where areas[f] is the area of faces[f].
//    normals: FloatTensor of shape (F, 3) where normals[f] is the normal of
//    faces[f]
//

// Cpu implementation.
std::tuple<at::Tensor, at::Tensor> FaceAreasNormalsForwardCpu(
    const at::Tensor verts,
    const at::Tensor faces);
// Cpu implementation
at::Tensor FaceAreasNormalsBackwardCpu(
    const at::Tensor grad_areas,
    const at::Tensor grad_normals,
    const at::Tensor verts,
    const at::Tensor faces);

#ifdef WITH_CUDA
// Cuda implementation.
std::tuple<at::Tensor, at::Tensor> FaceAreasNormalsForwardCuda(
    const at::Tensor verts,
    const at::Tensor faces);
// Cuda implementation.
at::Tensor FaceAreasNormalsBackwardCuda(
    const at::Tensor grad_areas,
    const at::Tensor grad_normals,
    const at::Tensor verts,
    const at::Tensor faces);
#endif

// Implementation which is exposed.
std::tuple<at::Tensor, at::Tensor> FaceAreasNormalsForward(
    const at::Tensor verts,
    const at::Tensor faces) {
  if (verts.is_cuda() && faces.is_cuda()) {
#ifdef WITH_CUDA
    CHECK_CUDA(verts);
    CHECK_CUDA(faces);
    return FaceAreasNormalsForwardCuda(verts, faces);
#else
    AT_ERROR("Not compiled with GPU support.");
#endif
  }
  CHECK_CPU(verts);
  CHECK_CPU(faces);
  return FaceAreasNormalsForwardCpu(verts, faces);
}

// Implementation which is exposed.
at::Tensor FaceAreasNormalsBackward(
    const at::Tensor grad_areas,
    const at::Tensor grad_normals,
    const at::Tensor verts,
    const at::Tensor faces) {
  if (verts.is_cuda() && faces.is_cuda()) {
#ifdef WITH_CUDA
    CHECK_CUDA(verts);
    CHECK_CUDA(faces);
    CHECK_CUDA(grad_areas);
    CHECK_CUDA(grad_normals);
    return FaceAreasNormalsBackwardCuda(grad_areas, grad_normals, verts, faces);
#else
    AT_ERROR("Not compiled with GPU support.");
#endif
  }
  CHECK_CPU(grad_areas);
  CHECK_CPU(grad_normals);
  CHECK_CPU(verts);
  CHECK_CPU(faces);
  return FaceAreasNormalsBackwardCpu(grad_areas, grad_normals, verts, faces);
}
