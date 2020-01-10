// Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include <float.h>
#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include "float_math.cuh"

// Set epsilon for preventing floating point errors and division by 0.
const auto kEpsilon = 1e-30;

// Determines whether a point p is on the right side of a 2D line segment
// given by the end points v0, v1.
//
// Args:
//     p: vec2 Coordinates of a point.
//     v0, v1: vec2 Coordinates of the end points of the edge.
//
// Returns:
//     area: The signed area of the parallelogram given by the vectors
//           A = p - v0
//           B = v1 - v0
//
__device__ inline float
EdgeFunctionForward(const float2& p, const float2& v0, const float2& v1) {
  return (p.x - v0.x) * (v1.y - v0.y) - (p.y - v0.y) * (v1.x - v0.x);
}

// Backward pass for the edge function returning partial dervivatives for each
// of the input points.
//
// Args:
//     p: vec2 Coordinates of a point.
//     v0, v1: vec2 Coordinates of the end points of the edge.
//     grad_edge: Upstream gradient for output from edge function.
//
// Returns:
//     tuple of  gradients for each of the input points:
//     (float2 d_edge_dp, float2 d_edge_dv0, float2 d_edge_dv1)
//
__device__ inline thrust::tuple<float2, float2, float2> EdgeFunctionBackward(
    const float2& p,
    const float2& v0,
    const float2& v1,
    const float& grad_edge) {
  const float2 dedge_dp = make_float2(v1.y - v0.y, v0.x - v1.x);
  const float2 dedge_dv0 = make_float2(p.y - v1.y, v1.x - p.x);
  const float2 dedge_dv1 = make_float2(v0.y - p.y, p.x - v0.x);
  return thrust::make_tuple(
      grad_edge * dedge_dp, grad_edge * dedge_dv0, grad_edge * dedge_dv1);
}

// The forward pass for computing the barycentric coordinates of a point
// relative to a triangle.
//
// Args:
//     p: Coordinates of a point.
//     v0, v1, v2: Coordinates of the triangle vertices.
//
// Returns
//     bary: (w0, w1, w2) barycentric coordinates in the range [0, 1].
//
__device__ inline float3 BarycentricCoordsForward(
    const float2& p,
    const float2& v0,
    const float2& v1,
    const float2& v2) {
  const float area = EdgeFunctionForward(v2, v0, v1) + kEpsilon;
  const float w0 = EdgeFunctionForward(p, v1, v2) / area;
  const float w1 = EdgeFunctionForward(p, v2, v0) / area;
  const float w2 = EdgeFunctionForward(p, v0, v1) / area;
  return make_float3(w0, w1, w2);
}

// The backward pass for computing the barycentric coordinates of a point
// relative to a triangle.
//
// Args:
//     p: Coordinates of a point.
//     v0, v1, v2: (x, y) coordinates of the triangle vertices.
//     grad_bary_upstream: vec3<T> Upstream gradient for each of the
//                         barycentric coordaintes [grad_w0, grad_w1, grad_w2].
//
// Returns
//    tuple of gradients for each of the triangle vertices:
//    (float2 grad_v0, float2 grad_v1, float2 grad_v2)
//
__device__ inline thrust::tuple<float2, float2, float2, float2>
BarycentricCoordsBackward(
    const float2& p,
    const float2& v0,
    const float2& v1,
    const float2& v2,
    const float3& grad_bary_upstream) {
  const float area = EdgeFunctionForward(v2, v0, v1) + kEpsilon;
  const float area2 = pow(area, 2.0);
  const float e0 = EdgeFunctionForward(p, v1, v2);
  const float e1 = EdgeFunctionForward(p, v2, v0);
  const float e2 = EdgeFunctionForward(p, v0, v1);

  const float grad_w0 = grad_bary_upstream.x;
  const float grad_w1 = grad_bary_upstream.y;
  const float grad_w2 = grad_bary_upstream.z;

  // Calculate component of the gradient from each of w0, w1 and w2.
  // e.g. for w0:
  // dloss/dw0_v = dl/dw0 * dw0/dw0_top * dw0_top/dv
  //               + dl/dw0 * dw0/dw0_bot * dw0_bot/dv
  const float dw0_darea = -e0 / (area2);
  const float dw0_e0 = 1 / area;
  const float dloss_d_w0area = grad_w0 * dw0_darea;
  const float dloss_e0 = grad_w0 * dw0_e0;
  auto de0_dv = EdgeFunctionBackward(p, v1, v2, dloss_e0);
  auto dw0area_dv = EdgeFunctionBackward(v2, v0, v1, dloss_d_w0area);
  const float2 dw0_p = thrust::get<0>(de0_dv);
  const float2 dw0_dv0 = thrust::get<1>(dw0area_dv);
  const float2 dw0_dv1 = thrust::get<1>(de0_dv) + thrust::get<2>(dw0area_dv);
  const float2 dw0_dv2 = thrust::get<2>(de0_dv) + thrust::get<0>(dw0area_dv);

  const float dw1_darea = -e1 / (area2);
  const float dw1_e1 = 1 / area;
  const float dloss_d_w1area = grad_w1 * dw1_darea;
  const float dloss_e1 = grad_w1 * dw1_e1;
  auto de1_dv = EdgeFunctionBackward(p, v2, v0, dloss_e1);
  auto dw1area_dv = EdgeFunctionBackward(v2, v0, v1, dloss_d_w1area);
  const float2 dw1_p = thrust::get<0>(de1_dv);
  const float2 dw1_dv0 = thrust::get<2>(de1_dv) + thrust::get<1>(dw1area_dv);
  const float2 dw1_dv1 = thrust::get<2>(dw1area_dv);
  const float2 dw1_dv2 = thrust::get<1>(de1_dv) + thrust::get<0>(dw1area_dv);

  const float dw2_darea = -e2 / (area2);
  const float dw2_e2 = 1 / area;
  const float dloss_d_w2area = grad_w2 * dw2_darea;
  const float dloss_e2 = grad_w2 * dw2_e2;
  auto de2_dv = EdgeFunctionBackward(p, v0, v1, dloss_e2);
  auto dw2area_dv = EdgeFunctionBackward(v2, v0, v1, dloss_d_w2area);
  const float2 dw2_p = thrust::get<0>(de2_dv);
  const float2 dw2_dv0 = thrust::get<1>(de2_dv) + thrust::get<1>(dw2area_dv);
  const float2 dw2_dv1 = thrust::get<2>(de2_dv) + thrust::get<2>(dw2area_dv);
  const float2 dw2_dv2 = thrust::get<0>(dw2area_dv);

  const float2 dbary_p = dw0_p + dw1_p + dw2_p;
  const float2 dbary_dv0 = dw0_dv0 + dw1_dv0 + dw2_dv0;
  const float2 dbary_dv1 = dw0_dv1 + dw1_dv1 + dw2_dv1;
  const float2 dbary_dv2 = dw0_dv2 + dw1_dv2 + dw2_dv2;

  return thrust::make_tuple(dbary_p, dbary_dv0, dbary_dv1, dbary_dv2);
}

// Return minimum distance between line segment (v1 - v0) and point p.
//
// Args:
//     p: Coordinates of a point.
//     v0, v1: Coordinates of the end points of the line segment.
//
// Returns:
//     non-square distance to the boundary of the triangle.
//
__device__ inline float
PointLineDistanceForward(const float2& p, const float2& a, const float2& b) {
  const float2 ba = b - a;
  float l2 = dot(ba, ba);
  float t = dot(ba, p - a) / l2;
  if (l2 <= kEpsilon) {
    return dot(p - b, p - b);
  }
  t = __saturatef(t); // clamp to the interval [+0.0, 1.0]
  const float2 p_proj = a + t * ba;
  const float2 d = (p_proj - p);
  return dot(d, d); // squared distance
}

// Backward pass for point to line distance in 2D.
//
// Args:
//     p: Coordinates of a point.
//     v0, v1: Coordinates of the end points of the line segment.
//     grad_dist: Upstream gradient for the distance.
//
// Returns:
//    tuple of gradients for each of the input points:
//      (float2 grad_p, float2 grad_v0, float2 grad_v1)
//
__device__ inline thrust::tuple<float2, float2, float2>
PointLineDistanceBackward(
    const float2& p,
    const float2& v0,
    const float2& v1,
    const float& grad_dist) {
  // Redo some of the forward pass calculations.
  const float2 v1v0 = v1 - v0;
  const float2 pv0 = p - v0;
  const float t_bot = dot(v1v0, v1v0);
  const float t_top = dot(v1v0, pv0);
  float tt = t_top / t_bot;
  tt = __saturatef(tt);
  const float2 p_proj = (1.0f - tt) * v0 + tt * v1;
  const float2 d = p - p_proj;
  const float dist = sqrt(dot(d, d));

  const float2 grad_p = -1.0f * grad_dist * 2.0f * (p_proj - p);
  const float2 grad_v0 = grad_dist * (1.0f - tt) * 2.0f * (p_proj - p);
  const float2 grad_v1 = grad_dist * tt * 2.0f * (p_proj - p);

  return thrust::make_tuple(grad_p, grad_v0, grad_v1);
}

// The forward pass for calculating the shortest distance between a point
// and a triangle.
//
// Args:
//     p: Coordinates of a point.
//     v0, v1, v2: Coordinates of the three triangle vertices.
//
// Returns:
//     shortest absolute distance from a point to a triangle.
//
__device__ inline float PointTriangleDistanceForward(
    const float2& p,
    const float2& v0,
    const float2& v1,
    const float2& v2) {
  // Compute distance to all 3 edges of the triangle and return the min.
  const float e01_dist = PointLineDistanceForward(p, v0, v1);
  const float e02_dist = PointLineDistanceForward(p, v0, v2);
  const float e12_dist = PointLineDistanceForward(p, v1, v2);
  const float edge_dist = fminf(fminf(e01_dist, e02_dist), e12_dist);
  return edge_dist;
}

// Backward pass for point triangle distance.
//
// Args:
//     p: Coordinates of a point.
//     v0, v1, v2: Coordinates of the three triangle vertices.
//     grad_dist: Upstream gradient for the distance.
//
// Returns:
//    tuple of gradients for each of the triangle vertices:
//      (float2 grad_v0, float2 grad_v1, float2 grad_v2)
//
__device__ inline thrust::tuple<float2, float2, float2, float2>
PointTriangleDistanceBackward(
    const float2& p,
    const float2& v0,
    const float2& v1,
    const float2& v2,
    const float& grad_dist) {
  // Compute distance to all 3 edges of the triangle.
  const float e01_dist = PointLineDistanceForward(p, v0, v1);
  const float e02_dist = PointLineDistanceForward(p, v0, v2);
  const float e12_dist = PointLineDistanceForward(p, v1, v2);

  // Initialize output tensors.
  float2 grad_v0 = make_float2(0.0f, 0.0f);
  float2 grad_v1 = make_float2(0.0f, 0.0f);
  float2 grad_v2 = make_float2(0.0f, 0.0f);
  float2 grad_p = make_float2(0.0f, 0.0f);

  // Find which edge is the closest and return PointLineDistanceBackward for
  // that edge.
  if (e01_dist <= e02_dist && e01_dist <= e12_dist) {
    // Closest edge is v1 - v0.
    auto grad_e01 = PointLineDistanceBackward(p, v0, v1, grad_dist);
    grad_p = thrust::get<0>(grad_e01);
    grad_v0 = thrust::get<1>(grad_e01);
    grad_v1 = thrust::get<2>(grad_e01);
  } else if (e02_dist <= e01_dist && e02_dist <= e12_dist) {
    // Closest edge is v2 - v0.
    auto grad_e02 = PointLineDistanceBackward(p, v0, v2, grad_dist);
    grad_p = thrust::get<0>(grad_e02);
    grad_v0 = thrust::get<1>(grad_e02);
    grad_v2 = thrust::get<2>(grad_e02);
  } else if (e12_dist <= e01_dist && e12_dist <= e02_dist) {
    // Closest edge is v2 - v1.
    auto grad_e12 = PointLineDistanceBackward(p, v1, v2, grad_dist);
    grad_p = thrust::get<0>(grad_e12);
    grad_v1 = thrust::get<1>(grad_e12);
    grad_v2 = thrust::get<2>(grad_e12);
  }

  return thrust::make_tuple(grad_p, grad_v0, grad_v1, grad_v2);
}
