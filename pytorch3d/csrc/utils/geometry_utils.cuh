/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <float.h>
#include <math.h>
#include <cstdio>
#include "float_math.cuh"

// Set epsilon for preventing floating point errors and division by 0.
#ifdef _MSC_VER
#define kEpsilon 1e-8f
#else
const auto kEpsilon = 1e-8;
#endif

// ************************************************************* //
//                          vec2 utils                           //
// ************************************************************* //

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
//     tuple of gradients for each of the input points:
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
  const float area2 = pow(area, 2.0f);
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

// Forward pass for applying perspective correction to barycentric coordinates.
//
// Args:
//     bary: Screen-space barycentric coordinates for a point
//     z0, z1, z2: Camera-space z-coordinates of the triangle vertices
//
// Returns
//     World-space barycentric coordinates
//
__device__ inline float3 BarycentricPerspectiveCorrectionForward(
    const float3& bary,
    const float z0,
    const float z1,
    const float z2) {
  const float w0_top = bary.x * z1 * z2;
  const float w1_top = z0 * bary.y * z2;
  const float w2_top = z0 * z1 * bary.z;
  const float denom = fmaxf(w0_top + w1_top + w2_top, kEpsilon);
  const float w0 = w0_top / denom;
  const float w1 = w1_top / denom;
  const float w2 = w2_top / denom;
  return make_float3(w0, w1, w2);
}

// Backward pass for applying perspective correction to barycentric coordinates.
//
// Args:
//     bary: Screen-space barycentric coordinates for a point
//     z0, z1, z2: Camera-space z-coordinates of the triangle vertices
//     grad_out: Upstream gradient of the loss with respect to the corrected
//               barycentric coordinates.
//
// Returns a tuple of:
//      grad_bary: Downstream gradient of the loss with respect to the the
//                 uncorrected barycentric coordinates.
//      grad_z0, grad_z1, grad_z2: Downstream gradient of the loss with respect
//                                 to the z-coordinates of the triangle verts
__device__ inline thrust::tuple<float3, float, float, float>
BarycentricPerspectiveCorrectionBackward(
    const float3& bary,
    const float z0,
    const float z1,
    const float z2,
    const float3& grad_out) {
  // Recompute forward pass
  const float w0_top = bary.x * z1 * z2;
  const float w1_top = z0 * bary.y * z2;
  const float w2_top = z0 * z1 * bary.z;
  const float denom = fmaxf(w0_top + w1_top + w2_top, kEpsilon);

  // Now do backward pass
  const float grad_denom_top =
      -w0_top * grad_out.x - w1_top * grad_out.y - w2_top * grad_out.z;
  const float grad_denom = grad_denom_top / (denom * denom);
  const float grad_w0_top = grad_denom + grad_out.x / denom;
  const float grad_w1_top = grad_denom + grad_out.y / denom;
  const float grad_w2_top = grad_denom + grad_out.z / denom;
  const float grad_bary_x = grad_w0_top * z1 * z2;
  const float grad_bary_y = grad_w1_top * z0 * z2;
  const float grad_bary_z = grad_w2_top * z0 * z1;
  const float3 grad_bary = make_float3(grad_bary_x, grad_bary_y, grad_bary_z);
  const float grad_z0 = grad_w1_top * bary.y * z2 + grad_w2_top * bary.z * z1;
  const float grad_z1 = grad_w0_top * bary.x * z2 + grad_w2_top * bary.z * z0;
  const float grad_z2 = grad_w0_top * bary.x * z1 + grad_w1_top * bary.y * z0;
  return thrust::make_tuple(grad_bary, grad_z0, grad_z1, grad_z2);
}

// Clip negative barycentric coordinates to 0.0 and renormalize so
// the barycentric coordinates for a point sum to 1. When the blur_radius
// is greater than 0, a face will still be recorded as overlapping a pixel
// if the pixel is outside the face. In this case at least one of the
// barycentric coordinates for the pixel relative to the face will be negative.
// Clipping will ensure that the texture and z buffer are interpolated
// correctly.
//
//  Args
//     bary: (w0, w1, w2) barycentric coordinates which can be outside the
//            range [0, 1].
//
//  Returns
//     bary: (w0, w1, w2) barycentric coordinates in the range [0, 1] which
//           satisfy the condition: sum(w0, w1, w2) = 1.0.
//
__device__ inline float3 BarycentricClipForward(const float3 bary) {
  float3 w = make_float3(0.0f, 0.0f, 0.0f);
  // Clamp lower bound only
  w.x = max(bary.x, 0.0);
  w.y = max(bary.y, 0.0);
  w.z = max(bary.z, 0.0);
  float w_sum = w.x + w.y + w.z;
  w_sum = fmaxf(w_sum, 1e-5);
  w.x /= w_sum;
  w.y /= w_sum;
  w.z /= w_sum;

  return w;
}

// Backward pass for barycentric coordinate clipping.
//
//  Args
//     bary: (w0, w1, w2) barycentric coordinates which can be outside the
//            range [0, 1].
//     grad_baryclip_upstream: vec3<T> Upstream gradient for each of the clipped
//                         barycentric coordinates [grad_w0, grad_w1, grad_w2].
//
// Returns
//    vec3<T> of gradients for the unclipped barycentric coordinates:
//    (grad_w0, grad_w1, grad_w2)
//
__device__ inline float3 BarycentricClipBackward(
    const float3 bary,
    const float3 grad_baryclip_upstream) {
  // Redo some of the forward pass calculations
  float3 w = make_float3(0.0f, 0.0f, 0.0f);
  // Clamp lower bound only
  w.x = max(bary.x, 0.0);
  w.y = max(bary.y, 0.0);
  w.z = max(bary.z, 0.0);
  float w_sum = w.x + w.y + w.z;

  float3 grad_bary = make_float3(1.0f, 1.0f, 1.0f);
  float3 grad_clip = make_float3(1.0f, 1.0f, 1.0f);
  float3 grad_sum = make_float3(1.0f, 1.0f, 1.0f);

  // Check if sum was clipped.
  float grad_sum_clip = 1.0f;
  if (w_sum < 1e-5) {
    grad_sum_clip = 0.0f;
    w_sum = 1e-5;
  }

  // Check if any of bary values have been clipped.
  if (bary.x < 0.0f) {
    grad_clip.x = 0.0f;
  }
  if (bary.y < 0.0f) {
    grad_clip.y = 0.0f;
  }
  if (bary.z < 0.0f) {
    grad_clip.z = 0.0f;
  }

  // Gradients of the sum.
  grad_sum.x = -w.x / (pow(w_sum, 2.0f)) * grad_sum_clip;
  grad_sum.y = -w.y / (pow(w_sum, 2.0f)) * grad_sum_clip;
  grad_sum.z = -w.z / (pow(w_sum, 2.0f)) * grad_sum_clip;

  // Gradients for each of the bary coordinates including the cross terms
  // from the sum.
  grad_bary.x = grad_clip.x *
      (grad_baryclip_upstream.x * (1.0f / w_sum + grad_sum.x) +
       grad_baryclip_upstream.y * (grad_sum.y) +
       grad_baryclip_upstream.z * (grad_sum.z));

  grad_bary.y = grad_clip.y *
      (grad_baryclip_upstream.y * (1.0f / w_sum + grad_sum.y) +
       grad_baryclip_upstream.x * (grad_sum.x) +
       grad_baryclip_upstream.z * (grad_sum.z));

  grad_bary.z = grad_clip.z *
      (grad_baryclip_upstream.z * (1.0f / w_sum + grad_sum.z) +
       grad_baryclip_upstream.x * (grad_sum.x) +
       grad_baryclip_upstream.y * (grad_sum.y));

  return grad_bary;
}

// Return minimum distance between line segment (v1 - v0) and point p.
//
// Args:
//     p: Coordinates of a point.
//     v0, v1: Coordinates of the end points of the line segment.
//
// Returns:
//     squared distance to the boundary of the triangle.
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
//     shortest squared distance from a point to a triangle.
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

// ************************************************************* //
//                          vec3 utils                           //
// ************************************************************* //

// Computes the area of a triangle (v0, v1, v2).
//
// Args:
//     v0, v1, v2: vec3 coordinates of the triangle vertices
//
// Returns
//     area: float: The area of the triangle
//
__device__ inline float
AreaOfTriangle(const float3& v0, const float3& v1, const float3& v2) {
  float3 p0 = v1 - v0;
  float3 p1 = v2 - v0;

  // compute the hypotenus of the scross product (p0 x p1)
  float dd = hypot(
      p0.y * p1.z - p0.z * p1.y,
      hypot(p0.z * p1.x - p0.x * p1.z, p0.x * p1.y - p0.y * p1.x));

  return dd / 2.0;
}

// Computes the barycentric coordinates of a point p relative
// to a triangle (v0, v1, v2), i.e. p = w0 * v0 + w1 * v1 + w2 * v2
// s.t. w0 + w1 + w2 = 1.0
//
// NOTE that this function assumes that p lives on the space spanned
// by (v0, v1, v2).
// TODO(gkioxari) explicitly check whether p is coplanar with (v0, v1, v2)
// and throw an error if check fails
//
// Args:
//     p: vec3 coordinates of a point
//     v0, v1, v2: vec3 coordinates of the triangle vertices
//
// Returns
//     bary: (w0, w1, w2) barycentric coordinates
//
__device__ inline float3 BarycentricCoords3Forward(
    const float3& p,
    const float3& v0,
    const float3& v1,
    const float3& v2) {
  float3 p0 = v1 - v0;
  float3 p1 = v2 - v0;
  float3 p2 = p - v0;

  const float d00 = dot(p0, p0);
  const float d01 = dot(p0, p1);
  const float d11 = dot(p1, p1);
  const float d20 = dot(p2, p0);
  const float d21 = dot(p2, p1);

  const float denom = d00 * d11 - d01 * d01 + kEpsilon;
  const float w1 = (d11 * d20 - d01 * d21) / denom;
  const float w2 = (d00 * d21 - d01 * d20) / denom;
  const float w0 = 1.0f - w1 - w2;

  return make_float3(w0, w1, w2);
}

// Checks whether the point p is inside the triangle (v0, v1, v2).
// A point is inside the triangle, if all barycentric coordinates
// wrt the triangle are >= 0 & <= 1.
// If the triangle is degenerate, aka line or point, then return False.
//
// NOTE that this function assumes that p lives on the space spanned
// by (v0, v1, v2).
// TODO(gkioxari) explicitly check whether p is coplanar with (v0, v1, v2)
// and throw an error if check fails
//
// Args:
//     p: vec3 coordinates of a point
//     v0, v1, v2: vec3 coordinates of the triangle vertices
//     min_triangle_area: triangles less than this size are considered
//     points/lines, IsInsideTriangle returns False
//
// Returns:
//     inside: bool indicating wether p is inside triangle
//
__device__ inline bool IsInsideTriangle(
    const float3& p,
    const float3& v0,
    const float3& v1,
    const float3& v2,
    const double min_triangle_area) {
  bool inside;
  if (AreaOfTriangle(v0, v1, v2) < min_triangle_area) {
    inside = 0;
  } else {
    float3 bary = BarycentricCoords3Forward(p, v0, v1, v2);
    bool x_in = 0.0f <= bary.x && bary.x <= 1.0f;
    bool y_in = 0.0f <= bary.y && bary.y <= 1.0f;
    bool z_in = 0.0f <= bary.z && bary.z <= 1.0f;
    inside = x_in && y_in && z_in;
  }
  return inside;
}

// Computes the minimum squared Euclidean distance between the point p
// and the segment spanned by (v0, v1).
// To find this we parametrize p as: x(t) = v0 + t * (v1 - v0)
// and find t which minimizes (x(t) - p) ^ 2.
// Note that p does not need to live in the space spanned by (v0, v1)
//
// Args:
//     p: vec3 coordinates of a point
//     v0, v1: vec3 coordinates of start and end of segment
//
// Returns:
//     dist: the minimum squared distance of p from segment (v0, v1)
//

__device__ inline float
PointLine3DistanceForward(const float3& p, const float3& v0, const float3& v1) {
  const float3 v1v0 = v1 - v0;
  const float3 pv0 = p - v0;
  const float t_bot = dot(v1v0, v1v0);
  const float t_top = dot(pv0, v1v0);
  // if t_bot small, then v0 == v1, set tt to 0.
  float tt = (t_bot < kEpsilon) ? 0.0f : (t_top / t_bot);

  tt = __saturatef(tt); // clamps to [0, 1]

  const float3 p_proj = v0 + tt * v1v0;
  const float3 diff = p - p_proj;
  const float dist = dot(diff, diff);
  return dist;
}

// Backward function of the minimum squared Euclidean distance between the point
// p and the line segment (v0, v1).
//
// Args:
//     p: vec3 coordinates of a point
//     v0, v1: vec3 coordinates of start and end of segment
//     grad_dist: Float of the gradient wrt dist
//
// Returns:
//    tuple of gradients for the point and line segment (v0, v1):
//      (float3 grad_p, float3 grad_v0, float3 grad_v1)

__device__ inline thrust::tuple<float3, float3, float3>
PointLine3DistanceBackward(
    const float3& p,
    const float3& v0,
    const float3& v1,
    const float& grad_dist) {
  const float3 v1v0 = v1 - v0;
  const float3 pv0 = p - v0;
  const float t_bot = dot(v1v0, v1v0);
  const float t_top = dot(v1v0, pv0);

  float3 grad_p = make_float3(0.0f, 0.0f, 0.0f);
  float3 grad_v0 = make_float3(0.0f, 0.0f, 0.0f);
  float3 grad_v1 = make_float3(0.0f, 0.0f, 0.0f);

  const float tt = t_top / t_bot;

  if (t_bot < kEpsilon) {
    // if t_bot small, then v0 == v1,
    // and dist = 0.5 * dot(pv0, pv0) + 0.5 * dot(pv1, pv1)
    grad_p = grad_dist * 2.0f * pv0;
    grad_v0 = -0.5f * grad_p;
    grad_v1 = grad_v0;
  } else if (tt < 0.0f) {
    grad_p = grad_dist * 2.0f * pv0;
    grad_v0 = -1.0f * grad_p;
    // no gradients wrt v1
  } else if (tt > 1.0f) {
    grad_p = grad_dist * 2.0f * (p - v1);
    grad_v1 = -1.0f * grad_p;
    // no gradients wrt v0
  } else {
    const float3 p_proj = v0 + tt * v1v0;
    const float3 diff = p - p_proj;
    const float3 grad_base = grad_dist * 2.0f * diff;
    grad_p = grad_base - dot(grad_base, v1v0) * v1v0 / t_bot;
    const float3 dtt_v0 = (-1.0f * v1v0 - pv0 + 2.0f * tt * v1v0) / t_bot;
    grad_v0 = (-1.0f + tt) * grad_base - dot(grad_base, v1v0) * dtt_v0;
    const float3 dtt_v1 = (pv0 - 2.0f * tt * v1v0) / t_bot;
    grad_v1 = -dot(grad_base, v1v0) * dtt_v1 - tt * grad_base;
  }

  return thrust::make_tuple(grad_p, grad_v0, grad_v1);
}

// Computes the squared distance of a point p relative to a triangle (v0, v1,
// v2). If the point's projection p0 on the plane spanned by (v0, v1, v2) is
// inside the triangle with vertices (v0, v1, v2), then the returned value is
// the squared distance of p to its projection p0. Otherwise, the returned value
// is the smallest squared distance of p from the line segments (v0, v1), (v0,
// v2) and (v1, v2).
//
// Args:
//     p: vec3 coordinates of a point
//     v0, v1, v2: vec3 coordinates of the triangle vertices
//     min_triangle_area: triangles less than this size are considered
//     points/lines, IsInsideTriangle returns False
//
// Returns:
//     dist: Float of the squared distance
//

__device__ inline float PointTriangle3DistanceForward(
    const float3& p,
    const float3& v0,
    const float3& v1,
    const float3& v2,
    const double min_triangle_area) {
  float3 normal = cross(v2 - v0, v1 - v0);
  const float norm_normal = norm(normal);
  normal = normalize(normal);

  // p0 is the projection of p on the plane spanned by (v0, v1, v2)
  // i.e. p0 = p + t * normal, s.t. (p0 - v0) is orthogonal to normal
  const float t = dot(v0 - p, normal);
  const float3 p0 = p + t * normal;

  bool is_inside = IsInsideTriangle(p0, v0, v1, v2, min_triangle_area);
  float dist = 0.0f;

  if ((is_inside) && (norm_normal > kEpsilon)) {
    // if projection p0 is inside triangle spanned by (v0, v1, v2)
    // then distance is equal to norm(p0 - p)^2
    dist = t * t;
  } else {
    const float e01 = PointLine3DistanceForward(p, v0, v1);
    const float e02 = PointLine3DistanceForward(p, v0, v2);
    const float e12 = PointLine3DistanceForward(p, v1, v2);

    dist = (e01 > e02) ? e02 : e01;
    dist = (dist > e12) ? e12 : dist;
  }

  return dist;
}

// The backward pass for computing the squared distance of a point
// to the triangle (v0, v1, v2).
//
// Args:
//     p: xyz coordinates of a point
//     v0, v1, v2: xyz coordinates of the triangle vertices
//     grad_dist: Float of the gradient wrt dist
//     min_triangle_area: triangles less than this size are considered
//     points/lines, IsInsideTriangle returns False
//
// Returns:
//     tuple of gradients for the point and triangle:
//        (float3 grad_p, float3 grad_v0, float3 grad_v1, float3 grad_v2)
//

__device__ inline thrust::tuple<float3, float3, float3, float3>
PointTriangle3DistanceBackward(
    const float3& p,
    const float3& v0,
    const float3& v1,
    const float3& v2,
    const float& grad_dist,
    const double min_triangle_area) {
  const float3 v2v0 = v2 - v0;
  const float3 v1v0 = v1 - v0;
  const float3 v0p = v0 - p;
  float3 raw_normal = cross(v2v0, v1v0);
  const float norm_normal = norm(raw_normal);
  float3 normal = normalize(raw_normal);

  // p0 is the projection of p on the plane spanned by (v0, v1, v2)
  // i.e. p0 = p + t * normal, s.t. (p0 - v0) is orthogonal to normal
  const float t = dot(v0 - p, normal);
  const float3 p0 = p + t * normal;
  const float3 diff = t * normal;

  bool is_inside = IsInsideTriangle(p0, v0, v1, v2, min_triangle_area);

  float3 grad_p = make_float3(0.0f, 0.0f, 0.0f);
  float3 grad_v0 = make_float3(0.0f, 0.0f, 0.0f);
  float3 grad_v1 = make_float3(0.0f, 0.0f, 0.0f);
  float3 grad_v2 = make_float3(0.0f, 0.0f, 0.0f);

  if ((is_inside) && (norm_normal > kEpsilon)) {
    // derivative of dist wrt p
    grad_p = -2.0f * grad_dist * t * normal;
    // derivative of dist wrt normal
    const float3 grad_normal = 2.0f * grad_dist * t * (v0p + diff);
    // derivative of dist wrt raw_normal
    const float3 grad_raw_normal = normalize_backward(raw_normal, grad_normal);
    // derivative of dist wrt v2v0 and v1v0
    const auto grad_cross = cross_backward(v2v0, v1v0, grad_raw_normal);
    const float3 grad_cross_v2v0 = thrust::get<0>(grad_cross);
    const float3 grad_cross_v1v0 = thrust::get<1>(grad_cross);
    grad_v0 =
        grad_dist * 2.0f * t * normal - (grad_cross_v2v0 + grad_cross_v1v0);
    grad_v1 = grad_cross_v1v0;
    grad_v2 = grad_cross_v2v0;
  } else {
    const float e01 = PointLine3DistanceForward(p, v0, v1);
    const float e02 = PointLine3DistanceForward(p, v0, v2);
    const float e12 = PointLine3DistanceForward(p, v1, v2);

    if ((e01 <= e02) && (e01 <= e12)) {
      // e01 is smallest
      const auto grads = PointLine3DistanceBackward(p, v0, v1, grad_dist);
      grad_p = thrust::get<0>(grads);
      grad_v0 = thrust::get<1>(grads);
      grad_v1 = thrust::get<2>(grads);
    } else if ((e02 <= e01) && (e02 <= e12)) {
      // e02 is smallest
      const auto grads = PointLine3DistanceBackward(p, v0, v2, grad_dist);
      grad_p = thrust::get<0>(grads);
      grad_v0 = thrust::get<1>(grads);
      grad_v2 = thrust::get<2>(grads);
    } else if ((e12 <= e01) && (e12 <= e02)) {
      // e12 is smallest
      const auto grads = PointLine3DistanceBackward(p, v1, v2, grad_dist);
      grad_p = thrust::get<0>(grads);
      grad_v1 = thrust::get<1>(grads);
      grad_v2 = thrust::get<2>(grads);
    }
  }

  return thrust::make_tuple(grad_p, grad_v0, grad_v1, grad_v2);
}
