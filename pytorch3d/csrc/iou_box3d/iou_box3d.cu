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
#include <algorithm>
#include "iou_box3d/iou_utils.h"

// Parallelize over N*M computations which can each be done
// independently
__global__ void IoUBox3DKernel(
    const float* __restrict__ boxes1, // (N, 8, 3)
    const float* __restrict__ boxes2, // (M, 8, 3)
    const int64_t N,
    const int64_t M,
    float* __restrict__ vols, // (N, M)
    float* __restrict__ ious) { // (N, M)
  const int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t stride = gridDim.x * blockDim.x;

  for (int64_t i = tid; i < N * M; i += stride) {
    const int64_t n = i / M;
    const int64_t m = i % M;
    iou_box3d::BoxOverlap(boxes1 + 24 * n, boxes2 + 24 * m, vols[i], ious[i]);
  }
}

std::tuple<at::Tensor, at::Tensor> IoUBox3DCuda(
    const at::Tensor& boxes1, // (N, 8, 3)
    const at::Tensor& boxes2) { // (M, 8, 3)
  // Check inputs are on the same device
  at::TensorArg boxes1_t{boxes1, "boxes1", 1}, boxes2_t{boxes2, "boxes2", 2};
  at::CheckedFrom c = "IoUBox3DCuda";
  at::checkAllSameGPU(c, {boxes1_t, boxes2_t});
  at::checkAllSameType(c, {boxes1_t, boxes2_t});

  // Set the device for the kernel launch based on the device of boxes1
  at::cuda::CUDAGuard device_guard(boxes1.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  TORCH_CHECK(
      boxes1.dim() == 3 && boxes2.dim() == 3 && boxes1.size(1) == 8 &&
          boxes1.size(2) == 3 && boxes2.size(1) == 8 && boxes2.size(2) == 3,
      "Boxes must have shape (N, 8, 3)");

  const int64_t N = boxes1.size(0);
  const int64_t M = boxes2.size(0);

  auto vols = at::empty({N, M}, boxes1.options());
  auto ious = at::empty({N, M}, boxes1.options());

  if (vols.numel() == 0) {
    return std::make_tuple(vols, ious);
  }

  const int64_t threads = 128;
  const int64_t blocks =
      std::min<int64_t>((N * M + threads - 1) / threads, 65536);

  IoUBox3DKernel<<<blocks, threads, 0, stream>>>(
      boxes1.data_ptr<float>(),
      boxes2.data_ptr<float>(),
      N,
      M,
      vols.data_ptr<float>(),
      ious.data_ptr<float>());

  AT_CUDA_CHECK(cudaGetLastError());

  return std::make_tuple(vols, ious);
}
