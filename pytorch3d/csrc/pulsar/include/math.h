/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef PULSAR_NATIVE_IMPL_MATH_H_
#define PULSAR_NATIVE_IMPL_MATH_H_

#include "./camera.h"
#include "./commands.h"
#include "./fastermath.h"

/**
 * Get the direction of val.
 *
 * Returns +1 if val is positive, -1 if val is zero or negative.
 */
IHD int sign_dir(const int& val) {
  return -(static_cast<int>((val <= 0)) << 1) + 1;
};

/**
 * Get the direction of val.
 *
 * Returns +1 if val is positive, -1 if val is zero or negative.
 */
IHD float sign_dir(const float& val) {
  return static_cast<float>(1 - (static_cast<int>((val <= 0)) << 1));
};

/**
 * Integer ceil division.
 */
IHD uint iDivCeil(uint a, uint b) {
  return (a % b != 0) ? (a / b + 1) : (a / b);
}

IHD float3 outer_product_sum(const float3& a) {
  return make_float3(
      a.x * a.x + a.x * a.y + a.x * a.z,
      a.x * a.y + a.y * a.y + a.y * a.z,
      a.x * a.z + a.y * a.z + a.z * a.z);
}

// TODO: put intrinsics here.
IHD float3 operator+(const float3& a, const float3& b) {
  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

IHD void operator+=(float3& a, const float3& b) {
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
}

IHD void operator-=(float3& a, const float3& b) {
  a.x -= b.x;
  a.y -= b.y;
  a.z -= b.z;
}

IHD void operator/=(float3& a, const float& b) {
  a.x /= b;
  a.y /= b;
  a.z /= b;
}

IHD void operator*=(float3& a, const float& b) {
  a.x *= b;
  a.y *= b;
  a.z *= b;
}

IHD float3 operator/(const float3& a, const float& b) {
  return make_float3(a.x / b, a.y / b, a.z / b);
}

IHD float3 operator-(const float3& a, const float3& b) {
  return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

IHD float3 operator*(const float3& a, const float& b) {
  return make_float3(a.x * b, a.y * b, a.z * b);
}

IHD float3 operator*(const float3& a, const float3& b) {
  return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

IHD float3 operator*(const float& a, const float3& b) {
  return b * a;
}

INLINE DEVICE float length(const float3& v) {
  // TODO: benchmark what's faster.
  return NORM3DF(v.x, v.y, v.z);
  // return __fsqrt_rn(v.x * v.x + v.y * v.y + v.z * v.z);
}

/**
 * Left-hand multiplication of the constructed rotation matrix with the vector.
 */
IHD float3 rotate(
    const float3& v,
    const float3& dir_x,
    const float3& dir_y,
    const float3& dir_z) {
  return make_float3(
      dir_x.x * v.x + dir_x.y * v.y + dir_x.z * v.z,
      dir_y.x * v.x + dir_y.y * v.y + dir_y.z * v.z,
      dir_z.x * v.x + dir_z.y * v.y + dir_z.z * v.z);
}

INLINE DEVICE float3 normalize(const float3& v) {
  return v * RNORM3DF(v.x, v.y, v.z);
}

INLINE DEVICE float dot(const float3& a, const float3& b) {
  return FADD(FADD(FMUL(a.x, b.x), FMUL(a.y, b.y)), FMUL(a.z, b.z));
}

INLINE DEVICE float3 cross(const float3& a, const float3& b) {
  // TODO: faster
  return make_float3(
      a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

namespace pulsar {
IHD CamGradInfo operator+(const CamGradInfo& a, const CamGradInfo& b) {
  CamGradInfo res;
  res.cam_pos = a.cam_pos + b.cam_pos;
  res.pixel_0_0_center = a.pixel_0_0_center + b.pixel_0_0_center;
  res.pixel_dir_x = a.pixel_dir_x + b.pixel_dir_x;
  res.pixel_dir_y = a.pixel_dir_y + b.pixel_dir_y;
  return res;
}

IHD CamGradInfo operator*(const CamGradInfo& a, const float& b) {
  CamGradInfo res;
  res.cam_pos = a.cam_pos * b;
  res.pixel_0_0_center = a.pixel_0_0_center * b;
  res.pixel_dir_x = a.pixel_dir_x * b;
  res.pixel_dir_y = a.pixel_dir_y * b;
  return res;
}

IHD IntWrapper operator+(const IntWrapper& a, const IntWrapper& b) {
  IntWrapper res;
  res.val = a.val + b.val;
  return res;
}
} // namespace pulsar

#endif
