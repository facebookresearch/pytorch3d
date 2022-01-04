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

// Helper functions WarpReduceMin and WarpReduceMax used in .cu files
// Starting in Volta, instructions are no longer synchronous within a warp.
// We need to call __syncwarp() to sync the 32 threads in the warp
// instead of all the threads in the block.

template <typename scalar_t>
__device__ void
WarpReduceMin(scalar_t* min_dists, int64_t* min_idxs, const size_t tid) {
  // s = 32
  if (min_dists[tid] > min_dists[tid + 32]) {
    min_idxs[tid] = min_idxs[tid + 32];
    min_dists[tid] = min_dists[tid + 32];
  }
  __syncwarp();
  // s = 16
  if (min_dists[tid] > min_dists[tid + 16]) {
    min_idxs[tid] = min_idxs[tid + 16];
    min_dists[tid] = min_dists[tid + 16];
  }
  __syncwarp();
  // s = 8
  if (min_dists[tid] > min_dists[tid + 8]) {
    min_idxs[tid] = min_idxs[tid + 8];
    min_dists[tid] = min_dists[tid + 8];
  }
  __syncwarp();
  // s = 4
  if (min_dists[tid] > min_dists[tid + 4]) {
    min_idxs[tid] = min_idxs[tid + 4];
    min_dists[tid] = min_dists[tid + 4];
  }
  __syncwarp();
  // s = 2
  if (min_dists[tid] > min_dists[tid + 2]) {
    min_idxs[tid] = min_idxs[tid + 2];
    min_dists[tid] = min_dists[tid + 2];
  }
  __syncwarp();
  // s = 1
  if (min_dists[tid] > min_dists[tid + 1]) {
    min_idxs[tid] = min_idxs[tid + 1];
    min_dists[tid] = min_dists[tid + 1];
  }
  __syncwarp();
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
  __syncwarp();
  if (dists[tid] < dists[tid + 16]) {
    dists[tid] = dists[tid + 16];
    dists_idx[tid] = dists_idx[tid + 16];
  }
  __syncwarp();
  if (dists[tid] < dists[tid + 8]) {
    dists[tid] = dists[tid + 8];
    dists_idx[tid] = dists_idx[tid + 8];
  }
  __syncwarp();
  if (dists[tid] < dists[tid + 4]) {
    dists[tid] = dists[tid + 4];
    dists_idx[tid] = dists_idx[tid + 4];
  }
  __syncwarp();
  if (dists[tid] < dists[tid + 2]) {
    dists[tid] = dists[tid + 2];
    dists_idx[tid] = dists_idx[tid + 2];
  }
  __syncwarp();
  if (dists[tid] < dists[tid + 1]) {
    dists[tid] = dists[tid + 1];
    dists_idx[tid] = dists_idx[tid + 1];
  }
  __syncwarp();
}
