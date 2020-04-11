// Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

#include <float.h>
#include <math.h>
#include <cstdio>

// helper WarpReduce used in .cu files

template <typename scalar_t>
__device__ void WarpReduce(
    volatile scalar_t* min_dists,
    volatile int64_t* min_idxs,
    const size_t tid) {
  // s = 32
  if (min_dists[tid] > min_dists[tid + 32]) {
    min_idxs[tid] = min_idxs[tid + 32];
    min_dists[tid] = min_dists[tid + 32];
  }
  // s = 16
  if (min_dists[tid] > min_dists[tid + 16]) {
    min_idxs[tid] = min_idxs[tid + 16];
    min_dists[tid] = min_dists[tid + 16];
  }
  // s = 8
  if (min_dists[tid] > min_dists[tid + 8]) {
    min_idxs[tid] = min_idxs[tid + 8];
    min_dists[tid] = min_dists[tid + 8];
  }
  // s = 4
  if (min_dists[tid] > min_dists[tid + 4]) {
    min_idxs[tid] = min_idxs[tid + 4];
    min_dists[tid] = min_dists[tid + 4];
  }
  // s = 2
  if (min_dists[tid] > min_dists[tid + 2]) {
    min_idxs[tid] = min_idxs[tid + 2];
    min_dists[tid] = min_dists[tid + 2];
  }
  // s = 1
  if (min_dists[tid] > min_dists[tid + 1]) {
    min_idxs[tid] = min_idxs[tid + 1];
    min_dists[tid] = min_dists[tid + 1];
  }
}
