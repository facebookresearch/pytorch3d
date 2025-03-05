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

namespace pulsar {
namespace pytorch {

void cudaDevToDev(
    void* trg,
    const void* src,
    const int& size,
    const cudaStream_t& stream) {
  C10_CUDA_CHECK(
      cudaMemcpyAsync(trg, src, size, cudaMemcpyDeviceToDevice, stream));
}

void cudaDevToHost(
    void* trg,
    const void* src,
    const int& size,
    const cudaStream_t& stream) {
  C10_CUDA_CHECK(
      cudaMemcpyAsync(trg, src, size, cudaMemcpyDeviceToHost, stream));
}

} // namespace pytorch
} // namespace pulsar
#endif
