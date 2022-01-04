/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

// Kernel for inputs_packed of shape (F, D), where D > 1
template <typename scalar_t>
__global__ void PackedToPaddedKernel(
    const scalar_t* __restrict__ inputs_packed,
    const int64_t* __restrict__ first_idxs,
    scalar_t* __restrict__ inputs_padded,
    const size_t batch_size,
    const size_t max_size,
    const size_t num_inputs,
    const size_t D) {
  // Batch elements split evenly across blocks (num blocks = batch_size) and
  // values for each element split across threads in the block. Each thread adds
  // the values of its respective input elements to the global inputs_padded
  // tensor.
  const size_t tid = threadIdx.x;
  const size_t batch_idx = blockIdx.x;

  const int64_t start = first_idxs[batch_idx];
  const int64_t end =
      batch_idx + 1 < batch_size ? first_idxs[batch_idx + 1] : num_inputs;
  const int num = end - start;
  for (size_t f = tid; f < num; f += blockDim.x) {
    for (size_t j = 0; j < D; ++j) {
      inputs_padded[batch_idx * max_size * D + f * D + j] =
          inputs_packed[(start + f) * D + j];
    }
  }
}

// Kernel for inputs of shape (F, 1)
template <typename scalar_t>
__global__ void PackedToPaddedKernelD1(
    const scalar_t* __restrict__ inputs_packed,
    const int64_t* __restrict__ first_idxs,
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

  const int64_t start = first_idxs[batch_idx];
  const int64_t end =
      batch_idx + 1 < batch_size ? first_idxs[batch_idx + 1] : num_inputs;
  const int num = end - start;
  for (size_t f = tid; f < num; f += blockDim.x) {
    inputs_padded[batch_idx * max_size + f] = inputs_packed[start + f];
  }
}

// Kernel for inputs_padded of shape (B, F, D), where D > 1
template <typename scalar_t>
__global__ void PaddedToPackedKernel(
    const scalar_t* __restrict__ inputs_padded,
    const int64_t* __restrict__ first_idxs,
    scalar_t* __restrict__ inputs_packed,
    const size_t batch_size,
    const size_t max_size,
    const size_t num_inputs,
    const size_t D) {
  // Batch elements split evenly across blocks (num blocks = batch_size) and
  // values for each element split across threads in the block. Each thread adds
  // the values of its respective input elements to the global inputs_packed
  // tensor.
  const size_t tid = threadIdx.x;
  const size_t batch_idx = blockIdx.x;

  const int64_t start = first_idxs[batch_idx];
  const int64_t end =
      batch_idx + 1 < batch_size ? first_idxs[batch_idx + 1] : num_inputs;
  const int num = end - start;
  for (size_t f = tid; f < num; f += blockDim.x) {
    for (size_t j = 0; j < D; ++j) {
      inputs_packed[(start + f) * D + j] =
          inputs_padded[batch_idx * max_size * D + f * D + j];
    }
  }
}

// Kernel for inputs_padded of shape (B, F, 1)
template <typename scalar_t>
__global__ void PaddedToPackedKernelD1(
    const scalar_t* __restrict__ inputs_padded,
    const int64_t* __restrict__ first_idxs,
    scalar_t* __restrict__ inputs_packed,
    const size_t batch_size,
    const size_t max_size,
    const size_t num_inputs) {
  // Batch elements split evenly across blocks (num blocks = batch_size) and
  // values for each element split across threads in the block. Each thread adds
  // the values of its respective input elements to the global inputs_packed
  // tensor.
  const size_t tid = threadIdx.x;
  const size_t batch_idx = blockIdx.x;

  const int64_t start = first_idxs[batch_idx];
  const int64_t end =
      batch_idx + 1 < batch_size ? first_idxs[batch_idx + 1] : num_inputs;
  const int num = end - start;
  for (size_t f = tid; f < num; f += blockDim.x) {
    inputs_packed[start + f] = inputs_padded[batch_idx * max_size + f];
  }
}

at::Tensor PackedToPaddedCuda(
    const at::Tensor inputs_packed,
    const at::Tensor first_idxs,
    const int64_t max_size) {
  // Check inputs are on the same device
  at::TensorArg inputs_packed_t{inputs_packed, "inputs_packed", 1},
      first_idxs_t{first_idxs, "first_idxs", 2};
  at::CheckedFrom c = "PackedToPaddedCuda";
  at::checkAllSameGPU(c, {inputs_packed_t, first_idxs_t});

  // Set the device for the kernel launch based on the device of the input
  at::cuda::CUDAGuard device_guard(inputs_packed.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int64_t num_inputs = inputs_packed.size(0);
  const int64_t batch_size = first_idxs.size(0);

  TORCH_CHECK(
      inputs_packed.dim() == 2, "inputs_packed must be a 2-dimensional tensor");
  const int64_t D = inputs_packed.size(1);
  at::Tensor inputs_padded =
      at::zeros({batch_size, max_size, D}, inputs_packed.options());

  if (inputs_padded.numel() == 0) {
    AT_CUDA_CHECK(cudaGetLastError());
    return inputs_padded;
  }

  const int threads = 512;
  const int blocks = batch_size;
  if (D == 1) {
    AT_DISPATCH_FLOATING_TYPES(
        inputs_packed.scalar_type(), "packed_to_padded_d1_kernel", ([&] {
          PackedToPaddedKernelD1<scalar_t><<<blocks, threads, 0, stream>>>(
              inputs_packed.contiguous().data_ptr<scalar_t>(),
              first_idxs.contiguous().data_ptr<int64_t>(),
              inputs_padded.data_ptr<scalar_t>(),
              batch_size,
              max_size,
              num_inputs);
        }));
  } else {
    AT_DISPATCH_FLOATING_TYPES(
        inputs_packed.scalar_type(), "packed_to_padded_kernel", ([&] {
          PackedToPaddedKernel<scalar_t><<<blocks, threads, 0, stream>>>(
              inputs_packed.contiguous().data_ptr<scalar_t>(),
              first_idxs.contiguous().data_ptr<int64_t>(),
              inputs_padded.data_ptr<scalar_t>(),
              batch_size,
              max_size,
              num_inputs,
              D);
        }));
  }

  AT_CUDA_CHECK(cudaGetLastError());
  return inputs_padded;
}

at::Tensor PaddedToPackedCuda(
    const at::Tensor inputs_padded,
    const at::Tensor first_idxs,
    const int64_t num_inputs) {
  // Check inputs are on the same device
  at::TensorArg inputs_padded_t{inputs_padded, "inputs_padded", 1},
      first_idxs_t{first_idxs, "first_idxs", 2};
  at::CheckedFrom c = "PaddedToPackedCuda";
  at::checkAllSameGPU(c, {inputs_padded_t, first_idxs_t});

  // Set the device for the kernel launch based on the device of the input
  at::cuda::CUDAGuard device_guard(inputs_padded.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int64_t batch_size = inputs_padded.size(0);
  const int64_t max_size = inputs_padded.size(1);

  TORCH_CHECK(batch_size == first_idxs.size(0), "sizes mismatch");
  TORCH_CHECK(
      inputs_padded.dim() == 3,
      "inputs_padded  must be a 3-dimensional tensor");
  const int64_t D = inputs_padded.size(2);

  at::Tensor inputs_packed =
      at::zeros({num_inputs, D}, inputs_padded.options());

  if (inputs_packed.numel() == 0) {
    AT_CUDA_CHECK(cudaGetLastError());
    return inputs_packed;
  }

  const int threads = 512;
  const int blocks = batch_size;

  if (D == 1) {
    AT_DISPATCH_FLOATING_TYPES(
        inputs_padded.scalar_type(), "padded_to_packed_d1_kernel", ([&] {
          PaddedToPackedKernelD1<scalar_t><<<blocks, threads, 0, stream>>>(
              inputs_padded.contiguous().data_ptr<scalar_t>(),
              first_idxs.contiguous().data_ptr<int64_t>(),
              inputs_packed.data_ptr<scalar_t>(),
              batch_size,
              max_size,
              num_inputs);
        }));
  } else {
    AT_DISPATCH_FLOATING_TYPES(
        inputs_padded.scalar_type(), "padded_to_packed_kernel", ([&] {
          PaddedToPackedKernel<scalar_t><<<blocks, threads, 0, stream>>>(
              inputs_padded.contiguous().data_ptr<scalar_t>(),
              first_idxs.contiguous().data_ptr<int64_t>(),
              inputs_packed.data_ptr<scalar_t>(),
              batch_size,
              max_size,
              num_inputs,
              D);
        }));
  }

  AT_CUDA_CHECK(cudaGetLastError());
  return inputs_packed;
}
