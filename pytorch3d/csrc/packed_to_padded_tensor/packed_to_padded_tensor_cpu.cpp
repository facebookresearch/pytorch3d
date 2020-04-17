// Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

#include <torch/extension.h>

at::Tensor PackedToPaddedCpu(
    const at::Tensor inputs_packed,
    const at::Tensor first_idxs,
    const int64_t max_size) {
  const int64_t num_inputs = inputs_packed.size(0);
  const int64_t batch_size = first_idxs.size(0);

  AT_ASSERTM(
      inputs_packed.dim() == 2, "inputs_packed must be a 2-dimensional tensor");
  const int64_t D = inputs_packed.size(1);

  torch::Tensor inputs_padded =
      torch::zeros({batch_size, max_size, D}, inputs_packed.options());

  auto inputs_packed_a = inputs_packed.accessor<float, 2>();
  auto first_idxs_a = first_idxs.accessor<int64_t, 1>();
  auto inputs_padded_a = inputs_padded.accessor<float, 3>();

  for (int b = 0; b < batch_size; ++b) {
    const int64_t start = first_idxs_a[b];
    const int64_t end = b + 1 < batch_size ? first_idxs_a[b + 1] : num_inputs;
    const int64_t num = end - start;
    for (int i = 0; i < num; ++i) {
      for (int j = 0; j < D; ++j) {
        inputs_padded_a[b][i][j] = inputs_packed_a[start + i][j];
      }
    }
  }
  return inputs_padded;
}

at::Tensor PaddedToPackedCpu(
    const at::Tensor inputs_padded,
    const at::Tensor first_idxs,
    const int64_t num_inputs) {
  const int64_t batch_size = inputs_padded.size(0);

  AT_ASSERTM(
      inputs_padded.dim() == 3, "inputs_padded must be a 3-dimensional tensor");
  const int64_t D = inputs_padded.size(2);

  torch::Tensor inputs_packed =
      torch::zeros({num_inputs, D}, inputs_padded.options());

  auto inputs_padded_a = inputs_padded.accessor<float, 3>();
  auto first_idxs_a = first_idxs.accessor<int64_t, 1>();
  auto inputs_packed_a = inputs_packed.accessor<float, 2>();

  for (int b = 0; b < batch_size; ++b) {
    const int64_t start = first_idxs_a[b];
    const int64_t end = b + 1 < batch_size ? first_idxs_a[b + 1] : num_inputs;
    const int64_t num = end - start;
    for (int i = 0; i < num; ++i) {
      for (int j = 0; j < D; ++j) {
        inputs_packed_a[start + i][j] = inputs_padded_a[b][i][j];
      }
    }
  }
  return inputs_packed;
}
