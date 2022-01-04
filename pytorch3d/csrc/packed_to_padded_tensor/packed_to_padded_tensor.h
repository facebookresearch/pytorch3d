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

// PackedToPadded
// Converts a packed tensor into a padded tensor, restoring the batch dimension.
// Refer to pytorch3d/structures/meshes.py for details on packed/padded tensors.
//
// Inputs:
//    inputs_packed: FloatTensor of shape (F, D), representing the packed batch
//                      tensor, e.g. areas for faces in a batch of meshes.
//    first_idxs: LongTensor of shape (N,) where N is the number of
//                       elements in the batch and `first_idxs[i] = f`
//                       means that the inputs for batch element i begin at
//                       `inputs[f]`.
//    max_size: Max length of an element in the batch.
// Returns:
//   inputs_padded: FloatTensor of shape (N, max_size, D) where max_size is max
//                 of `sizes`. The values for batch element i which start at
//                 `inputs_packed[first_idxs[i]]` will be copied to
//                 `inputs_padded[i, :]`, with zeros padding out the extra
//                  inputs.
//

// PaddedToPacked
// Converts a padded tensor into a packed tensor.
// Refer to pytorch3d/structures/meshes.py for details on packed/padded tensors.
//
// Inputs:
//    inputs_padded: FloatTensor of shape (N, max_size, D), representing the
//                padded tensor, e.g. areas for faces in a batch of meshes.
//    first_idxs: LongTensor of shape (N,) where N is the number of
//                       elements in the batch and `first_idxs[i] = f`
//                       means that the inputs for batch element i begin at
//                       `inputs_packed[f]`.
//    num_inputs: Number of packed entries (= F)
// Returns:
//   inputs_packed: FloatTensor of shape (F, D), where
//                      `inputs_packed[first_idx[i]:] = inputs_padded[i, :]`.
//
//

// Cpu implementation.
at::Tensor PackedToPaddedCpu(
    const at::Tensor inputs_packed,
    const at::Tensor first_idxs,
    const int64_t max_size);

// Cpu implementation.
at::Tensor PaddedToPackedCpu(
    const at::Tensor inputs_padded,
    const at::Tensor first_idxs,
    const int64_t num_inputs);

#ifdef WITH_CUDA
// Cuda implementation.
at::Tensor PackedToPaddedCuda(
    const at::Tensor inputs_packed,
    const at::Tensor first_idxs,
    const int64_t max_size);

// Cuda implementation.
at::Tensor PaddedToPackedCuda(
    const at::Tensor inputs_padded,
    const at::Tensor first_idxs,
    const int64_t num_inputs);
#endif

// Implementation which is exposed.
at::Tensor PackedToPadded(
    const at::Tensor inputs_packed,
    const at::Tensor first_idxs,
    const int64_t max_size) {
  if (inputs_packed.is_cuda()) {
#ifdef WITH_CUDA
    CHECK_CUDA(inputs_packed);
    CHECK_CUDA(first_idxs);
    return PackedToPaddedCuda(inputs_packed, first_idxs, max_size);
#else
    AT_ERROR("Not compiled with GPU support.");
#endif
  }
  return PackedToPaddedCpu(inputs_packed, first_idxs, max_size);
}

// Implementation which is exposed.
at::Tensor PaddedToPacked(
    const at::Tensor inputs_padded,
    const at::Tensor first_idxs,
    const int64_t num_inputs) {
  if (inputs_padded.is_cuda()) {
#ifdef WITH_CUDA
    CHECK_CUDA(inputs_padded);
    CHECK_CUDA(first_idxs);
    return PaddedToPackedCuda(inputs_padded, first_idxs, num_inputs);
#else
    AT_ERROR("Not compiled with GPU support.");
#endif
  }
  return PaddedToPackedCpu(inputs_padded, first_idxs, num_inputs);
}
