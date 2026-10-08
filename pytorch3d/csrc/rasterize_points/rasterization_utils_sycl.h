#pragma once

#include <sycl/sycl.hpp>

namespace pytorch3d {
namespace rasterization_utils_sycl {

  
  inline float NonSquareNdcRange(const int S1, const int S2) {
    float range = 2.0f;
    if (S1 > S2) {
      range = (S1 * range) / S2;
    }
  return range;
}

inline float PixToNonSquareNdc(const int i, const int S1, const int S2) {
  const float range = NonSquareNdcRange(S1, S2);
  const float offset = range / 2.0f;
  // Explicit double cast required for SYCL to align with CUDA behavior
  // Without it, will fail test_simple_xpu_naive and test_simple_xpu_binned
  // in tests/test_rasterize_meshes.py
  const float numerator = range * i + offset;
  const float quotient = static_cast<float>(
      static_cast<double>(numerator) / static_cast<double>(S1));
  return -offset + quotient;
  }
  
  template <typename T>
  inline void BubbleSort(T* arr, int n) {
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
  
  const int32_t kMaxPointsPerPixel = 150;
  
  const int32_t kMaxItemsPerBin = 22;
} // namespace rasterization_utils_sycl
} // namespace pytorch3d