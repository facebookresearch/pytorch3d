/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <type_traits>

// A fixed-sized vector with basic arithmetic operators useful for
// representing 2D coordinates.
// TODO: switch to Eigen if more functionality is needed.

template <
    typename T,
    typename = std::enable_if_t<
        std::is_same<T, double>::value || std::is_same<T, float>::value>>
struct vec2 {
  T x, y;
  typedef T scalar_t;
  vec2(T x, T y) : x(x), y(y) {}
};

template <typename T>
inline vec2<T> operator+(const vec2<T>& a, const vec2<T>& b) {
  return vec2<T>(a.x + b.x, a.y + b.y);
}

template <typename T>
inline vec2<T> operator-(const vec2<T>& a, const vec2<T>& b) {
  return vec2<T>(a.x - b.x, a.y - b.y);
}

template <typename T>
inline vec2<T> operator*(const T a, const vec2<T>& b) {
  return vec2<T>(a * b.x, a * b.y);
}

template <typename T>
inline vec2<T> operator/(const vec2<T>& a, const T b) {
  if (b == 0.0) {
    AT_ERROR(
        "denominator in vec2 division is 0"); // prevent divide by 0 errors.
  }
  return vec2<T>(a.x / b, a.y / b);
}

template <typename T>
inline T dot(const vec2<T>& a, const vec2<T>& b) {
  return a.x * b.x + a.y * b.y;
}

template <typename T>
inline T norm(const vec2<T>& a, const vec2<T>& b) {
  const vec2<T> ba = b - a;
  return sqrt(dot(ba, ba));
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const vec2<T>& v) {
  os << "vec2(" << v.x << ", " << v.y << ")";
  return os;
}
