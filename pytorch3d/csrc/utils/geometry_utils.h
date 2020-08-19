// Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

#include <ATen/ATen.h>
#include <algorithm>
#include <tuple>
#include <type_traits>
#include "vec2.h"
#include "vec3.h"

// Set epsilon for preventing floating point errors and division by 0.
const auto kEpsilon = 1e-8;

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
//                 v1 ________
//                   /\      /
//               A  /  \    /
//                 /    \  /
//             v0 /______\/
//                   B    p
//
//          The area can also be interpreted as the cross product A x B.
//          If the sign of the area is positive, the point p is on the
//          right side of the edge. Negative area indicates the point is on
//          the left side of the edge. i.e. for an edge v1 - v0:
//
//                      v1
//                     /
//                    /
//             -     /    +
//                  /
//                 /
//               v0
//
template <typename T>
T EdgeFunctionForward(const vec2<T>& p, const vec2<T>& v0, const vec2<T>& v1) {
  const T edge = (p.x - v0.x) * (v1.y - v0.y) - (p.y - v0.y) * (v1.x - v0.x);
  return edge;
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
//     (vec2<T> d_edge_dp, vec2<T> d_edge_dv0, vec2<T> d_edge_dv1)
//
template <typename T>
inline std::tuple<vec2<T>, vec2<T>, vec2<T>> EdgeFunctionBackward(
    const vec2<T>& p,
    const vec2<T>& v0,
    const vec2<T>& v1,
    const T grad_edge) {
  const vec2<T> dedge_dp(v1.y - v0.y, v0.x - v1.x);
  const vec2<T> dedge_dv0(p.y - v1.y, v1.x - p.x);
  const vec2<T> dedge_dv1(v0.y - p.y, p.x - v0.x);
  return std::make_tuple(
      grad_edge * dedge_dp, grad_edge * dedge_dv0, grad_edge * dedge_dv1);
}

// The forward pass for computing the barycentric coordinates of a point
// relative to a triangle.
// Ref:
// https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/barycentric-coordinates
//
// Args:
//     p: Coordinates of a point.
//     v0, v1, v2: Coordinates of the triangle vertices.
//
// Returns
//     bary: (w0, w1, w2) barycentric coordinates in the range [0, 1].
//
template <typename T>
vec3<T> BarycentricCoordinatesForward(
    const vec2<T>& p,
    const vec2<T>& v0,
    const vec2<T>& v1,
    const vec2<T>& v2) {
  const T area = EdgeFunctionForward(v2, v0, v1) + kEpsilon;
  const T w0 = EdgeFunctionForward(p, v1, v2) / area;
  const T w1 = EdgeFunctionForward(p, v2, v0) / area;
  const T w2 = EdgeFunctionForward(p, v0, v1) / area;
  return vec3<T>(w0, w1, w2);
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
//    (vec2<T> grad_v0, vec2<T> grad_v1, vec2<T> grad_v2)
//
template <typename T>
inline std::tuple<vec2<T>, vec2<T>, vec2<T>, vec2<T>> BarycentricCoordsBackward(
    const vec2<T>& p,
    const vec2<T>& v0,
    const vec2<T>& v1,
    const vec2<T>& v2,
    const vec3<T>& grad_bary_upstream) {
  const T area = EdgeFunctionForward(v2, v0, v1) + kEpsilon;
  const T area2 = pow(area, 2.0f);
  const T area_inv = 1.0f / area;
  const T e0 = EdgeFunctionForward(p, v1, v2);
  const T e1 = EdgeFunctionForward(p, v2, v0);
  const T e2 = EdgeFunctionForward(p, v0, v1);

  const T grad_w0 = grad_bary_upstream.x;
  const T grad_w1 = grad_bary_upstream.y;
  const T grad_w2 = grad_bary_upstream.z;

  // Calculate component of the gradient from each of w0, w1 and w2.
  // e.g. for w0:
  // dloss/dw0_v = dl/dw0 * dw0/dw0_top * dw0_top/dv
  //               + dl/dw0 * dw0/dw0_bot * dw0_bot/dv
  const T dw0_darea = -e0 / (area2);
  const T dw0_e0 = area_inv;
  const T dloss_d_w0area = grad_w0 * dw0_darea;
  const T dloss_e0 = grad_w0 * dw0_e0;
  auto de0_dv = EdgeFunctionBackward(p, v1, v2, dloss_e0);
  auto dw0area_dv = EdgeFunctionBackward(v2, v0, v1, dloss_d_w0area);
  const vec2<T> dw0_p = std::get<0>(de0_dv);
  const vec2<T> dw0_dv0 = std::get<1>(dw0area_dv);
  const vec2<T> dw0_dv1 = std::get<1>(de0_dv) + std::get<2>(dw0area_dv);
  const vec2<T> dw0_dv2 = std::get<2>(de0_dv) + std::get<0>(dw0area_dv);

  const T dw1_darea = -e1 / (area2);
  const T dw1_e1 = area_inv;
  const T dloss_d_w1area = grad_w1 * dw1_darea;
  const T dloss_e1 = grad_w1 * dw1_e1;
  auto de1_dv = EdgeFunctionBackward(p, v2, v0, dloss_e1);
  auto dw1area_dv = EdgeFunctionBackward(v2, v0, v1, dloss_d_w1area);
  const vec2<T> dw1_p = std::get<0>(de1_dv);
  const vec2<T> dw1_dv0 = std::get<2>(de1_dv) + std::get<1>(dw1area_dv);
  const vec2<T> dw1_dv1 = std::get<2>(dw1area_dv);
  const vec2<T> dw1_dv2 = std::get<1>(de1_dv) + std::get<0>(dw1area_dv);

  const T dw2_darea = -e2 / (area2);
  const T dw2_e2 = area_inv;
  const T dloss_d_w2area = grad_w2 * dw2_darea;
  const T dloss_e2 = grad_w2 * dw2_e2;
  auto de2_dv = EdgeFunctionBackward(p, v0, v1, dloss_e2);
  auto dw2area_dv = EdgeFunctionBackward(v2, v0, v1, dloss_d_w2area);
  const vec2<T> dw2_p = std::get<0>(de2_dv);
  const vec2<T> dw2_dv0 = std::get<1>(de2_dv) + std::get<1>(dw2area_dv);
  const vec2<T> dw2_dv1 = std::get<2>(de2_dv) + std::get<2>(dw2area_dv);
  const vec2<T> dw2_dv2 = std::get<0>(dw2area_dv);

  const vec2<T> dbary_p = dw0_p + dw1_p + dw2_p;
  const vec2<T> dbary_dv0 = dw0_dv0 + dw1_dv0 + dw2_dv0;
  const vec2<T> dbary_dv1 = dw0_dv1 + dw1_dv1 + dw2_dv1;
  const vec2<T> dbary_dv2 = dw0_dv2 + dw1_dv2 + dw2_dv2;

  return std::make_tuple(dbary_p, dbary_dv0, dbary_dv1, dbary_dv2);
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
template <typename T>
inline vec3<T> BarycentricPerspectiveCorrectionForward(
    const vec3<T>& bary,
    const T z0,
    const T z1,
    const T z2) {
  const T w0_top = bary.x * z1 * z2;
  const T w1_top = bary.y * z0 * z2;
  const T w2_top = bary.z * z0 * z1;
  const T denom = w0_top + w1_top + w2_top;
  const T w0 = w0_top / denom;
  const T w1 = w1_top / denom;
  const T w2 = w2_top / denom;
  return vec3<T>(w0, w1, w2);
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
template <typename T>
inline std::tuple<vec3<T>, T, T, T> BarycentricPerspectiveCorrectionBackward(
    const vec3<T>& bary,
    const T z0,
    const T z1,
    const T z2,
    const vec3<T>& grad_out) {
  // Recompute forward pass
  const T w0_top = bary.x * z1 * z2;
  const T w1_top = bary.y * z0 * z2;
  const T w2_top = bary.z * z0 * z1;
  const T denom = w0_top + w1_top + w2_top;

  // Now do backward pass
  const T grad_denom_top =
      -w0_top * grad_out.x - w1_top * grad_out.y - w2_top * grad_out.z;
  const T grad_denom = grad_denom_top / (denom * denom);
  const T grad_w0_top = grad_denom + grad_out.x / denom;
  const T grad_w1_top = grad_denom + grad_out.y / denom;
  const T grad_w2_top = grad_denom + grad_out.z / denom;
  const T grad_bary_x = grad_w0_top * z1 * z2;
  const T grad_bary_y = grad_w1_top * z0 * z2;
  const T grad_bary_z = grad_w2_top * z0 * z1;
  const vec3<T> grad_bary(grad_bary_x, grad_bary_y, grad_bary_z);
  const T grad_z0 = grad_w1_top * bary.y * z2 + grad_w2_top * bary.z * z1;
  const T grad_z1 = grad_w0_top * bary.x * z2 + grad_w2_top * bary.z * z0;
  const T grad_z2 = grad_w0_top * bary.x * z1 + grad_w1_top * bary.y * z0;
  return std::make_tuple(grad_bary, grad_z0, grad_z1, grad_z2);
}

// Clip negative barycentric coordinates to 0.0 and renormalize so
// the barycentric coordinates for a point sum to 1. When the blur_radius
// is greater than 0, a face will still be recorded as overlapping a pixel
// if the pixel is outisde the face. In this case at least one of the
// barycentric coordinates for the pixel relative to the face will be negative.
// Clipping will ensure that the texture and z buffer are interpolated
// correctly.
//
//  Args
//     bary: (w0, w1, w2) barycentric coordinates which can contain values < 0.
//
//  Returns
//     bary: (w0, w1, w2) barycentric coordinates in the range [0, 1] which
//           satisfy the condition: sum(w0, w1, w2) = 1.0.
//
template <typename T>
vec3<T> BarycentricClipForward(const vec3<T> bary) {
  vec3<T> w(0.0f, 0.0f, 0.0f);
  // Only clamp negative values to 0.0.
  // No need to clamp values > 1.0 as they will be renormalized.
  w.x = std::max(bary.x, 0.0f);
  w.y = std::max(bary.y, 0.0f);
  w.z = std::max(bary.z, 0.0f);
  float w_sum = w.x + w.y + w.z;
  w_sum = std::fmaxf(w_sum, 1e-5);
  w.x /= w_sum;
  w.y /= w_sum;
  w.z /= w_sum;
  return w;
}

// Backward pass for barycentric coordinate clipping.
//
//  Args
//     bary: (w0, w1, w2) barycentric coordinates which can contain values < 0.
//     grad_baryclip_upstream: vec3<T> Upstream gradient for each of the clipped
//                         barycentric coordinates [grad_w0, grad_w1, grad_w2].
//
// Returns
//    vec3<T> of gradients for the unclipped barycentric coordinates:
//    (grad_w0, grad_w1, grad_w2)
//
template <typename T>
vec3<T> BarycentricClipBackward(
    const vec3<T> bary,
    const vec3<T> grad_baryclip_upstream) {
  // Redo some of the forward pass calculations
  vec3<T> w(0.0f, 0.0f, 0.0f);
  w.x = std::max(bary.x, 0.0f);
  w.y = std::max(bary.y, 0.0f);
  w.z = std::max(bary.z, 0.0f);
  float w_sum = w.x + w.y + w.z;

  vec3<T> grad_bary(1.0f, 1.0f, 1.0f);
  vec3<T> grad_clip(1.0f, 1.0f, 1.0f);
  vec3<T> grad_sum(1.0f, 1.0f, 1.0f);

  // Check if the sum was clipped.
  float grad_sum_clip = 1.0f;
  if (w_sum < 1e-5) {
    grad_sum_clip = 0.0f;
    w_sum = 1e-5;
  }

  // Check if any of the bary coordinates have been clipped.
  // Only negative values are clamped to 0.0.
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

// Calculate minimum distance between a line segment (v1 - v0) and point p.
//
// Args:
//     p: Coordinates of a point.
//     v0, v1: Coordinates of the end points of the line segment.
//
// Returns:
//     squared distance of the point to the line.
//
// Consider the line extending the segment - this can be parameterized as:
// v0 + t (v1 - v0).
//
// First find the projection of point p onto the line. It falls where:
// t = [(p - v0) . (v1 - v0)] / |v1 - v0|^2
// where . is the dot product.
//
// The parameter t is clamped from [0, 1] to handle points outside the
// segment (v1 - v0).
//
// Once the projection of the point on the segment is known, the distance from
// p to the projection gives the minimum distance to the segment.
//
template <typename T>
T PointLineDistanceForward(
    const vec2<T>& p,
    const vec2<T>& v0,
    const vec2<T>& v1) {
  const vec2<T> v1v0 = v1 - v0;
  const T l2 = dot(v1v0, v1v0);
  if (l2 <= kEpsilon) {
    return dot(p - v1, p - v1);
  }

  const T t = dot(v1v0, p - v0) / l2;
  const T tt = std::min(std::max(t, 0.00f), 1.00f);
  const vec2<T> p_proj = v0 + tt * v1v0;
  return dot(p - p_proj, p - p_proj);
}

template <typename T>
T PointLine3DistanceForward(
    const vec3<T>& p,
    const vec3<T>& v0,
    const vec3<T>& v1) {
  const vec3<T> v1v0 = v1 - v0;
  const T l2 = dot(v1v0, v1v0);
  if (l2 <= kEpsilon) {
    return dot(p - v1, p - v1);
  }

  const T t = dot(v1v0, p - v0) / l2;
  const T tt = std::min(std::max(t, 0.00f), 1.00f);
  const vec3<T> p_proj = v0 + tt * v1v0;
  return dot(p - p_proj, p - p_proj);
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
//      (vec2<T> grad_p, vec2<T> grad_v0, vec2<T> grad_v1)
//
template <typename T>
inline std::tuple<vec2<T>, vec2<T>, vec2<T>> PointLineDistanceBackward(
    const vec2<T>& p,
    const vec2<T>& v0,
    const vec2<T>& v1,
    const T& grad_dist) {
  // Redo some of the forward pass calculations.
  const vec2<T> v1v0 = v1 - v0;
  const vec2<T> pv0 = p - v0;
  const T t_bot = dot(v1v0, v1v0);
  const T t_top = dot(v1v0, pv0);
  const T t = t_top / t_bot;
  const T tt = std::min(std::max(t, 0.00f), 1.00f);
  const vec2<T> p_proj = (1.0f - tt) * v0 + tt * v1;

  const vec2<T> grad_v0 = grad_dist * (1.0f - tt) * 2.0f * (p_proj - p);
  const vec2<T> grad_v1 = grad_dist * tt * 2.0f * (p_proj - p);
  const vec2<T> grad_p = -1.0f * grad_dist * 2.0f * (p_proj - p);

  return std::make_tuple(grad_p, grad_v0, grad_v1);
}

template <typename T>
std::tuple<vec3<T>, vec3<T>, vec3<T>> PointLine3DistanceBackward(
    const vec3<T>& p,
    const vec3<T>& v0,
    const vec3<T>& v1,
    const T& grad_dist) {
  const vec3<T> v1v0 = v1 - v0;
  const vec3<T> pv0 = p - v0;
  const T t_bot = dot(v1v0, v1v0);
  const T t_top = dot(v1v0, pv0);

  vec3<T> grad_p{0.0f, 0.0f, 0.0f};
  vec3<T> grad_v0{0.0f, 0.0f, 0.0f};
  vec3<T> grad_v1{0.0f, 0.0f, 0.0f};

  const T tt = t_top / t_bot;

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
    const vec3<T> p_proj = v0 + tt * v1v0;
    const vec3<T> diff = p - p_proj;
    const vec3<T> grad_base = grad_dist * 2.0f * diff;
    grad_p = grad_base - dot(grad_base, v1v0) * v1v0 / t_bot;
    const vec3<T> dtt_v0 = (-1.0f * v1v0 - pv0 + 2.0f * tt * v1v0) / t_bot;
    grad_v0 = (-1.0f + tt) * grad_base - dot(grad_base, v1v0) * dtt_v0;
    const vec3<T> dtt_v1 = (pv0 - 2.0f * tt * v1v0) / t_bot;
    grad_v1 = -dot(grad_base, v1v0) * dtt_v1 - tt * grad_base;
  }

  return std::make_tuple(grad_p, grad_v0, grad_v1);
}

// The forward pass for calculating the shortest distance between a point
// and a triangle.
// Ref: https://www.randygaul.net/2014/07/23/distance-point-to-line-segment/
//
// Args:
//     p: Coordinates of a point.
//     v0, v1, v2: Coordinates of the three triangle vertices.
//
// Returns:
//     shortest squared distance from a point to a triangle.
//
//
template <typename T>
T PointTriangleDistanceForward(
    const vec2<T>& p,
    const vec2<T>& v0,
    const vec2<T>& v1,
    const vec2<T>& v2) {
  // Compute distance of point to 3 edges of the triangle and return the
  // minimum value.
  const T e01_dist = PointLineDistanceForward(p, v0, v1);
  const T e02_dist = PointLineDistanceForward(p, v0, v2);
  const T e12_dist = PointLineDistanceForward(p, v1, v2);
  const T edge_dist = std::min(std::min(e01_dist, e02_dist), e12_dist);

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
//      (vec2<T> grad_v0, vec2<T> grad_v1, vec2<T> grad_v2)
//
template <typename T>
inline std::tuple<vec2<T>, vec2<T>, vec2<T>, vec2<T>>
PointTriangleDistanceBackward(
    const vec2<T>& p,
    const vec2<T>& v0,
    const vec2<T>& v1,
    const vec2<T>& v2,
    const T& grad_dist) {
  // Compute distance to all 3 edges of the triangle.
  const T e01_dist = PointLineDistanceForward(p, v0, v1);
  const T e02_dist = PointLineDistanceForward(p, v0, v2);
  const T e12_dist = PointLineDistanceForward(p, v1, v2);

  // Initialize output tensors.
  vec2<T> grad_v0(0.0f, 0.0f);
  vec2<T> grad_v1(0.0f, 0.0f);
  vec2<T> grad_v2(0.0f, 0.0f);
  vec2<T> grad_p(0.0f, 0.0f);

  // Find which edge is the closest and return PointLineDistanceBackward for
  // that edge.
  if (e01_dist <= e02_dist && e01_dist <= e12_dist) {
    // Closest edge is v1 - v0.
    auto grad_e01 = PointLineDistanceBackward(p, v0, v1, grad_dist);
    grad_p = std::get<0>(grad_e01);
    grad_v0 = std::get<1>(grad_e01);
    grad_v1 = std::get<2>(grad_e01);
  } else if (e02_dist <= e01_dist && e02_dist <= e12_dist) {
    // Closest edge is v2 - v0.
    auto grad_e02 = PointLineDistanceBackward(p, v0, v2, grad_dist);
    grad_p = std::get<0>(grad_e02);
    grad_v0 = std::get<1>(grad_e02);
    grad_v2 = std::get<2>(grad_e02);
  } else if (e12_dist <= e01_dist && e12_dist <= e02_dist) {
    // Closest edge is v2 - v1.
    auto grad_e12 = PointLineDistanceBackward(p, v1, v2, grad_dist);
    grad_p = std::get<0>(grad_e12);
    grad_v1 = std::get<1>(grad_e12);
    grad_v2 = std::get<2>(grad_e12);
  }

  return std::make_tuple(grad_p, grad_v0, grad_v1, grad_v2);
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
//
// Returns:
//     dist: Float of the squared distance
//

const float vEpsilon = 1e-8;

template <typename T>
vec3<T> BarycentricCoords3Forward(
    const vec3<T>& p,
    const vec3<T>& v0,
    const vec3<T>& v1,
    const vec3<T>& v2) {
  vec3<T> p0 = v1 - v0;
  vec3<T> p1 = v2 - v0;
  vec3<T> p2 = p - v0;

  const T d00 = dot(p0, p0);
  const T d01 = dot(p0, p1);
  const T d11 = dot(p1, p1);
  const T d20 = dot(p2, p0);
  const T d21 = dot(p2, p1);

  const T denom = d00 * d11 - d01 * d01 + kEpsilon;
  const T w1 = (d11 * d20 - d01 * d21) / denom;
  const T w2 = (d00 * d21 - d01 * d20) / denom;
  const T w0 = 1.0f - w1 - w2;

  return vec3<T>(w0, w1, w2);
}

// Checks whether the point p is inside the triangle (v0, v1, v2).
// A point is inside the triangle, if all barycentric coordinates
// wrt the triangle are >= 0 & <= 1.
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
// Returns:
//     inside: bool indicating wether p is inside triangle
//
template <typename T>
static bool IsInsideTriangle(
    const vec3<T>& p,
    const vec3<T>& v0,
    const vec3<T>& v1,
    const vec3<T>& v2) {
  vec3<T> bary = BarycentricCoords3Forward(p, v0, v1, v2);
  bool x_in = 0.0f <= bary.x && bary.x <= 1.0f;
  bool y_in = 0.0f <= bary.y && bary.y <= 1.0f;
  bool z_in = 0.0f <= bary.z && bary.z <= 1.0f;
  bool inside = x_in && y_in && z_in;
  return inside;
}

template <typename T>
T PointTriangle3DistanceForward(
    const vec3<T>& p,
    const vec3<T>& v0,
    const vec3<T>& v1,
    const vec3<T>& v2) {
  vec3<T> normal = cross(v2 - v0, v1 - v0);
  const T norm_normal = norm(normal);
  normal = normal / (norm_normal + vEpsilon);

  // p0 is the projection of p on the plane spanned by (v0, v1, v2)
  // i.e. p0 = p + t * normal, s.t. (p0 - v0) is orthogonal to normal
  const T t = dot(v0 - p, normal);
  const vec3<T> p0 = p + t * normal;

  bool is_inside = IsInsideTriangle(p0, v0, v1, v2);
  T dist = 0.0f;

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

template <typename T>
std::tuple<vec3<T>, vec3<T>>
cross_backward(const vec3<T>& a, const vec3<T>& b, const vec3<T>& grad_cross) {
  const float grad_ax = -grad_cross.y * b.z + grad_cross.z * b.y;
  const float grad_ay = grad_cross.x * b.z - grad_cross.z * b.x;
  const float grad_az = -grad_cross.x * b.y + grad_cross.y * b.x;
  const vec3<T> grad_a = vec3<T>(grad_ax, grad_ay, grad_az);

  const float grad_bx = grad_cross.y * a.z - grad_cross.z * a.y;
  const float grad_by = -grad_cross.x * a.z + grad_cross.z * a.x;
  const float grad_bz = grad_cross.x * a.y - grad_cross.y * a.x;
  const vec3<T> grad_b = vec3<T>(grad_bx, grad_by, grad_bz);

  return std::make_tuple(grad_a, grad_b);
}

template <typename T>
vec3<T> normalize_backward(const vec3<T>& a, const vec3<T>& grad_normz) {
  const float a_norm = norm(a) + vEpsilon;
  const vec3<T> out = a / a_norm;

  const float grad_ax = grad_normz.x * (1.0f - out.x * out.x) / a_norm +
      grad_normz.y * (-out.x * out.y) / a_norm +
      grad_normz.z * (-out.x * out.z) / a_norm;
  const float grad_ay = grad_normz.x * (-out.x * out.y) / a_norm +
      grad_normz.y * (1.0f - out.y * out.y) / a_norm +
      grad_normz.z * (-out.y * out.z) / a_norm;
  const float grad_az = grad_normz.x * (-out.x * out.z) / a_norm +
      grad_normz.y * (-out.y * out.z) / a_norm +
      grad_normz.z * (1.0f - out.z * out.z) / a_norm;
  return vec3<T>(grad_ax, grad_ay, grad_az);
}

// The backward pass for computing the squared distance of a point
// to the triangle (v0, v1, v2).
//
// Args:
//     p: xyz coordinates of a point
//     v0, v1, v2: xyz coordinates of the triangle vertices
//     grad_dist: Float of the gradient wrt dist
//
// Returns:
//     tuple of gradients for the point and triangle:
//        (float3 grad_p, float3 grad_v0, float3 grad_v1, float3 grad_v2)
//

template <typename T>
static std::tuple<vec3<T>, vec3<T>, vec3<T>, vec3<T>>
PointTriangle3DistanceBackward(
    const vec3<T>& p,
    const vec3<T>& v0,
    const vec3<T>& v1,
    const vec3<T>& v2,
    const T& grad_dist) {
  const vec3<T> v2v0 = v2 - v0;
  const vec3<T> v1v0 = v1 - v0;
  const vec3<T> v0p = v0 - p;
  vec3<T> raw_normal = cross(v2v0, v1v0);
  const T norm_normal = norm(raw_normal);
  vec3<T> normal = raw_normal / (norm_normal + vEpsilon);

  // p0 is the projection of p on the plane spanned by (v0, v1, v2)
  // i.e. p0 = p + t * normal, s.t. (p0 - v0) is orthogonal to normal
  const T t = dot(v0 - p, normal);
  const vec3<T> p0 = p + t * normal;
  const vec3<T> diff = t * normal;

  bool is_inside = IsInsideTriangle(p0, v0, v1, v2);

  vec3<T> grad_p(0.0f, 0.0f, 0.0f);
  vec3<T> grad_v0(0.0f, 0.0f, 0.0f);
  vec3<T> grad_v1(0.0f, 0.0f, 0.0f);
  vec3<T> grad_v2(0.0f, 0.0f, 0.0f);

  if ((is_inside) && (norm_normal > kEpsilon)) {
    // derivative of dist wrt p
    grad_p = -2.0f * grad_dist * t * normal;
    // derivative of dist wrt normal
    const vec3<T> grad_normal = 2.0f * grad_dist * t * (v0p + diff);
    // derivative of dist wrt raw_normal
    const vec3<T> grad_raw_normal = normalize_backward(raw_normal, grad_normal);
    // derivative of dist wrt v2v0 and v1v0
    const auto grad_cross = cross_backward(v2v0, v1v0, grad_raw_normal);
    const vec3<T> grad_cross_v2v0 = std::get<0>(grad_cross);
    const vec3<T> grad_cross_v1v0 = std::get<1>(grad_cross);
    grad_v0 =
        grad_dist * 2.0f * t * normal - (grad_cross_v2v0 + grad_cross_v1v0);
    grad_v1 = grad_cross_v1v0;
    grad_v2 = grad_cross_v2v0;
  } else {
    const T e01 = PointLine3DistanceForward(p, v0, v1);
    const T e02 = PointLine3DistanceForward(p, v0, v2);
    const T e12 = PointLine3DistanceForward(p, v1, v2);

    if ((e01 <= e02) && (e01 <= e12)) {
      // e01 is smallest
      const auto grads = PointLine3DistanceBackward(p, v0, v1, grad_dist);
      grad_p = std::get<0>(grads);
      grad_v0 = std::get<1>(grads);
      grad_v1 = std::get<2>(grads);
    } else if ((e02 <= e01) && (e02 <= e12)) {
      // e02 is smallest
      const auto grads = PointLine3DistanceBackward(p, v0, v2, grad_dist);
      grad_p = std::get<0>(grads);
      grad_v0 = std::get<1>(grads);
      grad_v2 = std::get<2>(grads);
    } else if ((e12 <= e01) && (e12 <= e02)) {
      // e12 is smallest
      const auto grads = PointLine3DistanceBackward(p, v1, v2, grad_dist);
      grad_p = std::get<0>(grads);
      grad_v1 = std::get<1>(grads);
      grad_v2 = std::get<2>(grads);
    }
  }

  return std::make_tuple(grad_p, grad_v0, grad_v1, grad_v2);
}
