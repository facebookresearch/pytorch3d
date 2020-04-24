// Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

#pragma once

// Given a pixel coordinate 0 <= i < S, convert it to a normalized device
// coordinate in the range [-1, 1]. We divide the NDC range into S evenly-sized
// pixels, and assume that each pixel falls in the *center* of its range.
__device__ inline float PixToNdc(int i, int S) {
  // NDC x-offset + (i * pixel_width + half_pixel_width)
  return -1 + (2 * i + 1.0f) / S;
}

// The maximum number of points per pixel that we can return. Since we use
// thread-local arrays to hold and sort points, the maximum size of the array
// needs to be known at compile time. There might be some fancy template magic
// we could use to make this more dynamic, but for now just fix a constant.
// TODO: is 8 enough? Would increasing have performance considerations?
const int32_t kMaxPointsPerPixel = 150;

const int32_t kMaxFacesPerBin = 22;

template <typename T>
__device__ inline void BubbleSort(T* arr, int n) {
  // Bubble sort. We only use it for tiny thread-local arrays (n < 8); in this
  // regime we care more about warp divergence than computational complexity.
  for (int i = 0; i < n - 1; ++i) {
    for (int j = 0; j < n - i - 1; ++j) {
      if (arr[j + 1] < arr[j]) {
        T temp = arr[j];
        arr[j] = arr[j + 1];
        arr[j + 1] = temp;
      }
    }
  }
}
