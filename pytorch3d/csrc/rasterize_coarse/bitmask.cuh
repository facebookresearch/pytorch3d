/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#define BINMASK_H

// A BitMask represents a bool array of shape (H, W, N). We pack values into
// the bits of unsigned ints; a single unsigned int has B = 32 bits, so to hold
// all values we use H * W * (N / B) = H * W * D values. We want to store
// BitMasks in shared memory, so we assume that the memory has already been
// allocated for it elsewhere.
class BitMask {
 public:
  __device__ BitMask(unsigned int* data, int H, int W, int N)
      : data(data), H(H), W(W), B(8 * sizeof(unsigned int)), D(N / B) {
    // TODO: check if the data is null.
    N = ceilf(N % 32); // take ceil incase N % 32 != 0
    block_clear(); // clear the data
  }

  // Use all threads in the current block to clear all bits of this BitMask
  __device__ void block_clear() {
    for (int i = threadIdx.x; i < H * W * D; i += blockDim.x) {
      data[i] = 0;
    }
    __syncthreads();
  }

  __device__ int _get_elem_idx(int y, int x, int d) {
    return y * W * D + x * D + d / B;
  }

  __device__ int _get_bit_idx(int d) {
    return d % B;
  }

  // Turn on a single bit (y, x, d)
  __device__ void set(int y, int x, int d) {
    int elem_idx = _get_elem_idx(y, x, d);
    int bit_idx = _get_bit_idx(d);
    const unsigned int mask = 1U << bit_idx;
    atomicOr(data + elem_idx, mask);
  }

  // Turn off a single bit (y, x, d)
  __device__ void unset(int y, int x, int d) {
    int elem_idx = _get_elem_idx(y, x, d);
    int bit_idx = _get_bit_idx(d);
    const unsigned int mask = ~(1U << bit_idx);
    atomicAnd(data + elem_idx, mask);
  }

  // Check whether the bit (y, x, d) is on or off
  __device__ bool get(int y, int x, int d) {
    int elem_idx = _get_elem_idx(y, x, d);
    int bit_idx = _get_bit_idx(d);
    return (data[elem_idx] >> bit_idx) & 1U;
  }

  // Compute the number of bits set in the row (y, x, :)
  __device__ int count(int y, int x) {
    int total = 0;
    for (int i = 0; i < D; ++i) {
      int elem_idx = y * W * D + x * D + i;
      unsigned int elem = data[elem_idx];
      total += __popc(elem);
    }
    return total;
  }

 private:
  unsigned int* data;
  int H, W, B, D;
};
