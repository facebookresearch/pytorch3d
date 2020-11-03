// Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#ifdef WITH_CUDA
#include <cuda_runtime_api.h>

namespace pulsar {
namespace pytorch {

void cudaDevToDev(
    void* trg,
    const void* src,
    const int& size,
    const cudaStream_t& stream) {
  cudaMemcpyAsync(trg, src, size, cudaMemcpyDeviceToDevice, stream);
}

void cudaDevToHost(
    void* trg,
    const void* src,
    const int& size,
    const cudaStream_t& stream) {
  cudaMemcpyAsync(trg, src, size, cudaMemcpyDeviceToHost, stream);
}

} // namespace pytorch
} // namespace pulsar
#endif
