/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef PULSAR_NATIVE_PYTORCH_UTIL_H_
#define PULSAR_NATIVE_PYTORCH_UTIL_H_

#include <ATen/ATen.h>
#include "../global.h"

namespace pulsar {
namespace pytorch {

void cudaDevToDev(
    void* trg,
    const void* src,
    const int& size,
    const cudaStream_t& stream);
void cudaDevToHost(
    void* trg,
    const void* src,
    const int& size,
    const cudaStream_t& stream);

/**
 * This method takes a memory pointer and wraps it into a pytorch tensor.
 *
 * This is preferred over `torch::from_blob`, since that requires a CUDA
 * managed pointer. However, working with these for high performance
 * operations is slower. Most of the rendering operations should stay
 * local to the respective GPU anyways, so unmanaged pointers are
 * preferred.
 */
template <typename T>
torch::Tensor from_blob(
    const T* ptr,
    const torch::IntArrayRef& shape,
    const c10::DeviceType& device_type,
    const c10::DeviceIndex& device_index,
    const torch::Dtype& dtype,
    const cudaStream_t& stream) {
  torch::Tensor ret = torch::zeros(
      shape, torch::device({device_type, device_index}).dtype(dtype));
  const int num_elements =
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>{});
  if (device_type == c10::DeviceType::CUDA) {
#ifdef WITH_CUDA
    cudaDevToDev(
        ret.data_ptr(),
        static_cast<const void*>(ptr),
        sizeof(T) * num_elements,
        stream);
#else
    throw std::runtime_error(
        "Initiating devToDev copy on a build without CUDA.");
#endif
    // TODO: check for synchronization.
  } else {
    memcpy(ret.data_ptr(), ptr, sizeof(T) * num_elements);
  }
  return ret;
};

} // namespace pytorch
} // namespace pulsar

#endif
