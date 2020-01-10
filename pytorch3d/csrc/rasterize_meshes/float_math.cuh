// Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once
#include <thrust/tuple.h>

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
