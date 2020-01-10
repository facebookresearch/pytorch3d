// Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once
#include <torch/extension.h>

// Converts a packed tensor into a padded tensor, restoring the batch dimension.
// Refer to pytorch3d/structures/meshes.py for details on packed/padded tensors.
//
// Inputs:
//    inputs: FloatTensor of shape (F,), representing the packed batch tensor.
//           e.g. areas for faces in a batch of meshes.
//    first_idxs: LongTensor of shape (N,) where N is the number of
//                       elements in the batch and `packed_first_idxs[i] = f`
//                       means that the inputs for batch element i begin at
//                       `inputs[f]`.
//   max_size: Max length of an element in the batch.
// Returns:
//   inputs_padded: FloatTensor of shape (N, max_size) where max_size is max
//                 of `sizes`. The values for batch element i which start at
//                 `inputs[packed_first_idxs[i]]` will be copied to
//                 `inputs_padded[i, :]``, with zeros padding out the extra
//                  inputs.
//

// Cuda implementation.
at::Tensor packed_to_padded_tensor_cuda(
    at::Tensor inputs,
    at::Tensor first_idxs,
    const long max_size);

// Implementation which is exposed.
at::Tensor packed_to_padded_tensor(
    at::Tensor inputs,
    at::Tensor first_idxs,
    const long max_size) {
  if (inputs.type().is_cuda()) {
#ifdef WITH_CUDA
    return packed_to_padded_tensor_cuda(inputs, first_idxs, max_size);
#else
    AT_ERROR("Not compiled with GPU support.");
#endif
  }
  AT_ERROR("Not implemented on the CPU.");
}
