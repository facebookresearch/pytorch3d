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

// Calculate the intersection volume and IoU metric for two batches of boxes
//
// Args:
//     boxes1: tensor of shape (N, 8, 3) of the coordinates of the 1st boxes
//     boxes2: tensor of shape (M, 8, 3) of the coordinates of the 2nd boxes
// Returns:
//     vol: (N, M) tensor of the volume of the intersecting convex shapes
//     iou: (N, M) tensor of the intersection over union which is
//          defined as: `iou = vol / (vol1 + vol2 - vol)`

// CPU implementation
std::tuple<at::Tensor, at::Tensor> IoUBox3DCpu(
    const at::Tensor& boxes1,
    const at::Tensor& boxes2);

// CUDA implementation
std::tuple<at::Tensor, at::Tensor> IoUBox3DCuda(
    const at::Tensor& boxes1,
    const at::Tensor& boxes2);

// Implementation which is exposed
inline std::tuple<at::Tensor, at::Tensor> IoUBox3D(
    const at::Tensor& boxes1,
    const at::Tensor& boxes2) {
  if (boxes1.is_cuda() || boxes2.is_cuda()) {
#ifdef WITH_CUDA
    CHECK_CUDA(boxes1);
    CHECK_CUDA(boxes2);
    return IoUBox3DCuda(boxes1.contiguous(), boxes2.contiguous());
#else
    AT_ERROR("Not compiled with GPU support.");
#endif
  }
  return IoUBox3DCpu(boxes1.contiguous(), boxes2.contiguous());
}
