// Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include <ATen/ATen.h>

template <typename scalar_t>
__global__ void packed_to_padded_tensor_kernel(
    const scalar_t* __restrict__ inputs,
    const long* __restrict__ first_idxs,
    scalar_t* __restrict__ inputs_padded,
    const size_t batch_size,
    const size_t max_size,
    const size_t num_inputs) {
  // Batch elements split evenly across blocks (num blocks = batch_size) and
  // values for each element split across threads in the block. Each thread adds
  // the values of its respective input elements to the global inputs_padded
  // tensor.
  const size_t tid = threadIdx.x;
  const size_t batch_idx = blockIdx.x;

  const long start = first_idxs[batch_idx];
  const long end =
      batch_idx + 1 < batch_size ? first_idxs[batch_idx + 1] : num_inputs;
  const int num_faces = end - start;
  for (size_t f = tid; f < num_faces; f += blockDim.x) {
    inputs_padded[batch_idx * max_size + f] = inputs[start + f];
  }
}

at::Tensor packed_to_padded_tensor_cuda(
    at::Tensor inputs,
    at::Tensor first_idxs,
    const long max_size) {
  const auto num_inputs = inputs.size(0);
  const auto batch_size = first_idxs.size(0);
  at::Tensor inputs_padded =
      at::zeros({batch_size, max_size}, inputs.options());

  const int threads = 512;
  const int blocks = batch_size;
  AT_DISPATCH_FLOATING_TYPES(
      inputs.type(), "packed_to_padded_tensor_kernel", ([&] {
        packed_to_padded_tensor_kernel<scalar_t><<<blocks, threads>>>(
            inputs.data_ptr<scalar_t>(),
            first_idxs.data_ptr<long>(),
            inputs_padded.data_ptr<scalar_t>(),
            batch_size,
            max_size,
            num_inputs);
      }));

  return inputs_padded;
}
