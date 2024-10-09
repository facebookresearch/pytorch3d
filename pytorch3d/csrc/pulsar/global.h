/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef PULSAR_GLOBAL_H
#define PULSAR_GLOBAL_H

#include "./constants.h"
#ifndef WIN32
#include <csignal>
#endif

#if defined(_WIN64) || defined(_WIN32)
#define uint unsigned int
#define ushort unsigned short
#endif

#include "./logging.h" // <- include before torch/extension.h

#define MAX_GRAD_SPHERES 128

#ifdef __CUDACC__
#define INLINE __forceinline__
#define HOST __host__
#define DEVICE __device__
#define GLOBAL __global__
#define RESTRICT __restrict__
#define DEBUGBREAK()
#ifdef __NVCC_DIAG_PRAGMA_SUPPORT__
#pragma nv_diag_suppress 1866
#pragma nv_diag_suppress 2941
#pragma nv_diag_suppress 2951
#pragma nv_diag_suppress 2967
#else
#if !defined(USE_ROCM)
#pragma diag_suppress = attribute_not_allowed
#pragma diag_suppress = 1866
#pragma diag_suppress = 2941
#pragma diag_suppress = 2951
#pragma diag_suppress = 2967
#endif //! USE_ROCM
#endif
#else // __CUDACC__
#define INLINE inline
#define HOST
#define DEVICE
#define GLOBAL
#define RESTRICT
#define DEBUGBREAK() std::raise(SIGINT)
// Don't care about pytorch warnings; they shouldn't clutter our warnings.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Weverything"
#include <torch/extension.h>
#pragma clang diagnostic pop
#ifdef WITH_CUDA
#include <ATen/cuda/CUDAContext.h>
#if !defined(USE_ROCM)
#include <vector_functions.h>
#endif //! USE_ROCM
#else
#ifndef cudaStream_t
typedef void* cudaStream_t;
#endif
struct int2 {
  int x, y;
};
struct ushort2 {
  unsigned short x, y;
};
struct float2 {
  float x, y;
};
struct float3 {
  float x, y, z;
};
inline float3 make_float3(const float& x, const float& y, const float& z) {
  float3 res;
  res.x = x;
  res.y = y;
  res.z = z;
  return res;
}
#endif
namespace py = pybind11;

inline bool operator==(const float3& a, const float3& b) {
  return a.x == b.x && a.y == b.y && a.z == b.z;
}
#endif // __CUDACC__
#define IHD INLINE HOST DEVICE

// An assertion command that can be used on host and device.
#ifdef PULSAR_ASSERTIONS
#ifdef __CUDACC__
#define PASSERT(VAL)                                     \
  if (!(VAL)) {                                          \
    printf(                                              \
        "Pulsar assertion failed in %s, line %d: %s.\n", \
        __FILE__,                                        \
        __LINE__,                                        \
        #VAL);                                           \
  }
#else
#define PASSERT(VAL)                                     \
  if (!(VAL)) {                                          \
    printf(                                              \
        "Pulsar assertion failed in %s, line %d: %s.\n", \
        __FILE__,                                        \
        __LINE__,                                        \
        #VAL);                                           \
    std::raise(SIGINT);                                  \
  }
#endif
#else
#define PASSERT(VAL)
#endif

#endif
