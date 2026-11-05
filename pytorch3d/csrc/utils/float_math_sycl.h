#pragma once

#include "../pulsar/global.h"
#include <sycl/sycl.hpp>

namespace pytorch3d {
namespace float_math_sycl {

// Common functions and operators for float2.

inline float2 operator+(const float2& a, const float2& b) {
  return make_float2(a.x + b.x, a.y + b.y);
}

inline float2 operator-(const float2& a, const float2& b) {
  return make_float2(a.x - b.x, a.y - b.y);
}

inline float2 operator/(const float2& a, const float2& b) {
  return make_float2(a.x / b.x, a.y / b.y);
}

inline float2 operator/(const float2& a, const float scalar) {
  return make_float2(a.x / scalar, a.y / scalar);
}

inline float2 operator*(const float2& a, const float2& b) {
  return make_float2(a.x * b.x, a.y * b.y);
}

inline float2 operator*(const float scalar, const float2& vector) {
  return make_float2(scalar * vector.x, scalar * vector.y);
}

inline float FloatMin3(const float a, const float b, const float c) {
  return sycl::fmin(a, sycl::fmin(b, c));
}

inline float FloatMax3(const float a, const float b, const float c) {
  return sycl::fmax(a, sycl::fmax(b, c));
}

inline float dot(const float2& a, const float2& b) {
  return a.x * b.x + a.y * b.y;
}

// Common functions and operators for float3.

inline float3 operator-(const float3& a, const float3& b) {
  return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline float3 operator+(const float3& a, const float3& b) {
  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline float3 operator/(const float3& a, const float3& b) {
  return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}

inline float3 operator/(const float3& a, const float scalar) {
  return make_float3(a.x / scalar, a.y / scalar, a.z / scalar);
}

inline float3 operator*(const float3& a, const float3& b) {
  return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

inline float3 operator*(const float scalar, const float3& vector) {
  return make_float3(scalar * vector.x, scalar * vector.y, scalar * vector.z);
}

inline float dot(const float3& a, const float3& b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

} // namespace float_math_sycl
} // namespace pytorch3d