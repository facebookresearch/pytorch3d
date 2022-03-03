/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <thrust/tuple.h>

// Set epsilon
#ifdef _MSC_VER
#define vEpsilon 1e-8f
#else
const auto vEpsilon = 1e-8;
#endif

// Common functions and operators for float2.

__device__ inline float2 operator-(const float2& a, const float2& b) {
  return make_float2(a.x - b.x, a.y - b.y);
}

__device__ inline float2 operator+(const float2& a, const float2& b) {
  return make_float2(a.x + b.x, a.y + b.y);
}

__device__ inline float2 operator/(const float2& a, const float2& b) {
  return make_float2(a.x / b.x, a.y / b.y);
}

__device__ inline float2 operator/(const float2& a, const float b) {
  return make_float2(a.x / b, a.y / b);
}

__device__ inline float2 operator*(const float2& a, const float2& b) {
  return make_float2(a.x * b.x, a.y * b.y);
}

__device__ inline float2 operator*(const float a, const float2& b) {
  return make_float2(a * b.x, a * b.y);
}

__device__ inline float FloatMin3(const float a, const float b, const float c) {
  return fminf(a, fminf(b, c));
}

__device__ inline float FloatMax3(const float a, const float b, const float c) {
  return fmaxf(a, fmaxf(b, c));
}

__device__ inline float dot(const float2& a, const float2& b) {
  return a.x * b.x + a.y * b.y;
}

// Backward pass for the dot product.
// Args:
//     a, b: Coordinates of two points.
//     grad_dot: Upstream gradient for the output.
//
// Returns:
//    tuple of gradients for each of the input points:
//      (float2 grad_a, float2 grad_b)
//
__device__ inline thrust::tuple<float2, float2>
DotBackward(const float2& a, const float2& b, const float& grad_dot) {
  return thrust::make_tuple(grad_dot * b, grad_dot * a);
}

__device__ inline float sum(const float2& a) {
  return a.x + a.y;
}

// Common functions and operators for float3.

__device__ inline float3 operator-(const float3& a, const float3& b) {
  return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ inline float3 operator+(const float3& a, const float3& b) {
  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ inline float3 operator/(const float3& a, const float3& b) {
  return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}

__device__ inline float3 operator/(const float3& a, const float b) {
  return make_float3(a.x / b, a.y / b, a.z / b);
}

__device__ inline float3 operator*(const float3& a, const float3& b) {
  return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

__device__ inline float3 operator*(const float a, const float3& b) {
  return make_float3(a * b.x, a * b.y, a * b.z);
}

__device__ inline float dot(const float3& a, const float3& b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ inline float sum(const float3& a) {
  return a.x + a.y + a.z;
}

__device__ inline float3 cross(const float3& a, const float3& b) {
  return make_float3(
      a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

__device__ inline thrust::tuple<float3, float3>
cross_backward(const float3& a, const float3& b, const float3& grad_cross) {
  const float grad_ax = -grad_cross.y * b.z + grad_cross.z * b.y;
  const float grad_ay = grad_cross.x * b.z - grad_cross.z * b.x;
  const float grad_az = -grad_cross.x * b.y + grad_cross.y * b.x;
  const float3 grad_a = make_float3(grad_ax, grad_ay, grad_az);

  const float grad_bx = grad_cross.y * a.z - grad_cross.z * a.y;
  const float grad_by = -grad_cross.x * a.z + grad_cross.z * a.x;
  const float grad_bz = grad_cross.x * a.y - grad_cross.y * a.x;
  const float3 grad_b = make_float3(grad_bx, grad_by, grad_bz);

  return thrust::make_tuple(grad_a, grad_b);
}

__device__ inline float norm(const float3& a) {
  return sqrt(dot(a, a));
}

__device__ inline float3 normalize(const float3& a) {
  return a / (norm(a) + vEpsilon);
}

__device__ inline float3 normalize_backward(
    const float3& a,
    const float3& grad_normz) {
  const float a_norm = norm(a) + vEpsilon;
  const float3 out = a / a_norm;

  const float grad_ax = grad_normz.x * (1.0f - out.x * out.x) / a_norm +
      grad_normz.y * (-out.x * out.y) / a_norm +
      grad_normz.z * (-out.x * out.z) / a_norm;
  const float grad_ay = grad_normz.x * (-out.x * out.y) / a_norm +
      grad_normz.y * (1.0f - out.y * out.y) / a_norm +
      grad_normz.z * (-out.y * out.z) / a_norm;
  const float grad_az = grad_normz.x * (-out.x * out.z) / a_norm +
      grad_normz.y * (-out.y * out.z) / a_norm +
      grad_normz.z * (1.0f - out.z * out.z) / a_norm;
  return make_float3(grad_ax, grad_ay, grad_az);
}
