/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <float.h>
#include <math.h>
#include <cstdio>

// Helper functions WarpReduceMin and WarpReduceMax used in .cu files.
// Starting in Volta, instructions are no longer synchronous within a warp,
// so on CUDA __syncwarp() is required between dependent shared-memory
// accesses in the unrolled tail reduction.
//
// On AMD/HIP no __syncwarp() is needed here: all wavefront lanes execute
// in lockstep (AMD has no equivalent of NVIDIA's Independent Thread
// Scheduling), and the AMDGPU memory model guarantees that LDS operations
// issued by the same wavefront are observed in program order without an
// explicit s_waitcnt — see the LLVM AMDGPU backend memory-model rules
// (https://llvm.org/docs/AMDGPUUsage.html) and the HIP hardware-
// implementation docs. This holds for both wave32 (RDNA, gfx10xx/11xx/
// 12xx) and wave64 (CDNA, gfx9xx), so the USE_ROCM skip is
// architecture-independent.

template <typename scalar_t>
__device__ void
WarpReduceMin(scalar_t* min_dists, int64_t* min_idxs, const size_t tid) {
  // s = 32
  if (min_dists[tid] > min_dists[tid + 32]) {
    min_idxs[tid] = min_idxs[tid + 32];
    min_dists[tid] = min_dists[tid + 32];
  }
#if !defined(USE_ROCM)
  __syncwarp();
#endif
  // s = 16
  if (min_dists[tid] > min_dists[tid + 16]) {
    min_idxs[tid] = min_idxs[tid + 16];
    min_dists[tid] = min_dists[tid + 16];
  }
#if !defined(USE_ROCM)
  __syncwarp();
#endif
  // s = 8
  if (min_dists[tid] > min_dists[tid + 8]) {
    min_idxs[tid] = min_idxs[tid + 8];
    min_dists[tid] = min_dists[tid + 8];
  }
#if !defined(USE_ROCM)
  __syncwarp();
#endif
  // s = 4
  if (min_dists[tid] > min_dists[tid + 4]) {
    min_idxs[tid] = min_idxs[tid + 4];
    min_dists[tid] = min_dists[tid + 4];
  }
#if !defined(USE_ROCM)
  __syncwarp();
#endif
  // s = 2
  if (min_dists[tid] > min_dists[tid + 2]) {
    min_idxs[tid] = min_idxs[tid + 2];
    min_dists[tid] = min_dists[tid + 2];
  }
#if !defined(USE_ROCM)
  __syncwarp();
#endif
  // s = 1
  if (min_dists[tid] > min_dists[tid + 1]) {
    min_idxs[tid] = min_idxs[tid + 1];
    min_dists[tid] = min_dists[tid + 1];
  }
#if !defined(USE_ROCM)
  __syncwarp();
#endif
}

template <typename scalar_t>
__device__ void WarpReduceMax(
    volatile scalar_t* dists,
    volatile int64_t* dists_idx,
    const size_t tid) {
  if (dists[tid] < dists[tid + 32]) {
    dists[tid] = dists[tid + 32];
    dists_idx[tid] = dists_idx[tid + 32];
  }
#if !defined(USE_ROCM)
  __syncwarp();
#endif
  if (dists[tid] < dists[tid + 16]) {
    dists[tid] = dists[tid + 16];
    dists_idx[tid] = dists_idx[tid + 16];
  }
#if !defined(USE_ROCM)
  __syncwarp();
#endif
  if (dists[tid] < dists[tid + 8]) {
    dists[tid] = dists[tid + 8];
    dists_idx[tid] = dists_idx[tid + 8];
  }
#if !defined(USE_ROCM)
  __syncwarp();
#endif
  if (dists[tid] < dists[tid + 4]) {
    dists[tid] = dists[tid + 4];
    dists_idx[tid] = dists_idx[tid + 4];
  }
#if !defined(USE_ROCM)
  __syncwarp();
#endif
  if (dists[tid] < dists[tid + 2]) {
    dists[tid] = dists[tid + 2];
    dists_idx[tid] = dists_idx[tid + 2];
  }
#if !defined(USE_ROCM)
  __syncwarp();
#endif
  if (dists[tid] < dists[tid + 1]) {
    dists[tid] = dists[tid + 1];
    dists_idx[tid] = dists_idx[tid + 1];
  }
#if !defined(USE_ROCM)
  __syncwarp();
#endif
}
