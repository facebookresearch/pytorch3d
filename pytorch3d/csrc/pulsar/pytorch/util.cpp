/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifdef WITH_CUDA
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime_api.h>
#include <limits>

namespace pulsar {
namespace pytorch {

namespace {
constexpr size_t kMaxCudaMemcpyBytes =
    static_cast<size_t>(std::numeric_limits<int>::max());

inline void checkCudaMemcpyArgs(void* trg, const void* src, const size_t& size) {
  TORCH_CHECK(size <= kMaxCudaMemcpyBytes, "Invalid cudaMemcpyAsync size: ", size);
  TORCH_CHECK(size == 0 || trg != nullptr, "cudaMemcpyAsync target pointer is null.");
  TORCH_CHECK(size == 0 || src != nullptr, "cudaMemcpyAsync source pointer is null.");
}
} // namespace

void cudaDevToDev(
    void* trg,
    const void* src,
    const size_t& size,
    const cudaStream_t& stream) {
  checkCudaMemcpyArgs(trg, src, size);
  C10_CUDA_CHECK(
      cudaMemcpyAsync(trg, src, size, cudaMemcpyDeviceToDevice, stream));
}

void cudaDevToDev(
    void* trg,
    const void* src,
    const int& size,
    const cudaStream_t& stream) {
  TORCH_CHECK(size >= 0, "Invalid cudaMemcpyAsync size: ", size);
  cudaDevToDev(trg, src, static_cast<size_t>(size), stream);
}

void cudaDevToHost(
    void* trg,
    const void* src,
    const size_t& size,
    const cudaStream_t& stream) {
  checkCudaMemcpyArgs(trg, src, size);
  C10_CUDA_CHECK(
      cudaMemcpyAsync(trg, src, size, cudaMemcpyDeviceToHost, stream));
}

void cudaDevToHost(
    void* trg,
    const void* src,
    const int& size,
    const cudaStream_t& stream) {
  TORCH_CHECK(size >= 0, "Invalid cudaMemcpyAsync size: ", size);
  cudaDevToHost(trg, src, static_cast<size_t>(size), stream);
}

} // namespace pytorch
} // namespace pulsar
#endif
