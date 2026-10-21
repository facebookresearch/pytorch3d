#pragma once

#include <oneapi/dpl/tuple>
#include <sycl/sycl.hpp>
#include "float_math_sycl.h"

namespace pytorch3d {
namespace geometry_utils_sycl {

using namespace pytorch3d::float_math_sycl;

constexpr double kEpsilon = 1e-8;

inline float 
EdgeFunctionForward(const float2& p, const float2& v0, const float2& v1) {
  return (p.x - v0.x) * (v1.y - v0.y) - (p.y - v0.y) * (v1.x - v0.x);
}

inline oneapi::dpl::tuple<float2, float2, float2> EdgeFunctionBackward(
    const float2& p,
    const float2& v0,
    const float2& v1,
    const float& grad_edge) {
  const float2 dedge_dp = make_float2(v1.y - v0.y, v0.x - v1.x);
  const float2 dedge_dv0 = make_float2(p.y - v1.y, v1.x - p.x);
  const float2 dedge_dv1 = make_float2(v0.y - p.y, p.x - v0.x);
  return oneapi::dpl::make_tuple(
      grad_edge * dedge_dp, grad_edge * dedge_dv0, grad_edge * dedge_dv1);
}

inline float3 BarycentricCoordsForward(
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

inline oneapi::dpl::tuple<float2, float2, float2, float2>
BarycentricCoordsBackward(
    const float2& p,
    const float2& v0,
    const float2& v1,
    const float2& v2,
    const float3& grad_bary_upstream) {
  const float area = EdgeFunctionForward(v2, v0, v1) + kEpsilon;
  const float area2 = sycl::pow(area, 2.0f);
  const float e0 = EdgeFunctionForward(p, v1, v2);
  const float e1 = EdgeFunctionForward(p, v2, v0);
  const float e2 = EdgeFunctionForward(p, v0, v1);

  const float grad_w0 = grad_bary_upstream.x;
  const float grad_w1 = grad_bary_upstream.y;
  const float grad_w2 = grad_bary_upstream.z;

  const float dw0_darea = -e0 / area2;
  const float dw0_e0 = 1 / area;
  const float dloss_d_w0area = grad_w0 * dw0_darea;
  const float dloss_e0 = grad_w0 * dw0_e0;
  auto de0_dv = EdgeFunctionBackward(p, v1, v2, dloss_e0);
  auto dw0area_dv = EdgeFunctionBackward(v2, v0, v1, dloss_d_w0area);
  const float2 dw0_p = oneapi::dpl::get<0>(de0_dv);
  const float2 dw0_dv0 = oneapi::dpl::get<1>(dw0area_dv);
  const float2 dw0_dv1 = oneapi::dpl::get<1>(de0_dv) + oneapi::dpl::get<2>(dw0area_dv);
  const float2 dw0_dv2 = oneapi::dpl::get<2>(de0_dv) + oneapi::dpl::get<0>(dw0area_dv);

  const float dw1_darea = -e1 / area2;
  const float dw1_e1 = 1 / area;
  const float dloss_d_w1area = grad_w1 * dw1_darea;
  const float dloss_e1 = grad_w1 * dw1_e1;
  auto de1_dv = EdgeFunctionBackward(p, v2, v0, dloss_e1);
  auto dw1area_dv = EdgeFunctionBackward(v2, v0, v1, dloss_d_w1area);
  const float2 dw1_p = oneapi::dpl::get<0>(de1_dv);
  const float2 dw1_dv0 = oneapi::dpl::get<2>(de1_dv) + oneapi::dpl::get<1>(dw1area_dv);
  const float2 dw1_dv1 = oneapi::dpl::get<2>(dw1area_dv);
  const float2 dw1_dv2 = oneapi::dpl::get<1>(de1_dv) + oneapi::dpl::get<0>(dw1area_dv);

  const float dw2_darea = -e2 / area2;
  const float dw2_e2 = 1 / area;
  const float dloss_d_w2area = grad_w2 * dw2_darea;
  const float dloss_e2 = grad_w2 * dw2_e2;
  auto de2_dv = EdgeFunctionBackward(p, v0, v1, dloss_e2);
  auto dw2area_dv = EdgeFunctionBackward(v2, v0, v1, dloss_d_w2area);
  const float2 dw2_p = oneapi::dpl::get<0>(de2_dv);
  const float2 dw2_dv0 = oneapi::dpl::get<1>(de2_dv) + oneapi::dpl::get<1>(dw2area_dv);
  const float2 dw2_dv1 = oneapi::dpl::get<2>(de2_dv) + oneapi::dpl::get<2>(dw2area_dv);
  const float2 dw2_dv2 = oneapi::dpl::get<0>(dw2area_dv);

  const float2 dbary_p = dw0_p + dw1_p + dw2_p;
  const float2 dbary_dv0 = dw0_dv0 + dw1_dv0 + dw2_dv0;
  const float2 dbary_dv1 = dw0_dv1 + dw1_dv1 + dw2_dv1;
  const float2 dbary_dv2 = dw0_dv2 + dw1_dv2 + dw2_dv2;

  return oneapi::dpl::make_tuple(dbary_p, dbary_dv0, dbary_dv1, dbary_dv2);
}


inline float3 BarycentricPerspectiveCorrectionForward(
    const float3& bary,
    const float z0,
    const float z1,
    const float z2) {
  const float w0_top = bary.x * z1 * z2;
  const float w1_top = z0 * bary.y * z2;
  const float w2_top = z0 * z1 * bary.z;
    const float denom = sycl::fmax(
      w0_top + w1_top + w2_top, static_cast<float>(kEpsilon));
  const float w0 = w0_top / denom;
  const float w1 = w1_top / denom;
  const float w2 = w2_top / denom;
  return make_float3(w0, w1, w2);
}

inline oneapi::dpl::tuple<float3, float, float, float>
BarycentricPerspectiveCorrectionBackward(
    const float3& bary,
    const float z0,
    const float z1,
    const float z2,
    const float3& grad_out) {
  const float w0_top = bary.x * z1 * z2;
  const float w1_top = z0 * bary.y * z2;
  const float w2_top = z0 * z1 * bary.z;
  const float denom = sycl::fmax(
      static_cast<double>(w0_top + w1_top + w2_top), kEpsilon);
  
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
  return oneapi::dpl::make_tuple(grad_bary, grad_z0, grad_z1, grad_z2);
}

inline float3 BarycentricClipForward(const float3 bary) {
  float3 w = make_float3(0.0f, 0.0f, 0.0f);
  w.x = sycl::fmax(bary.x, 0.0f);
  w.y = sycl::fmax(bary.y, 0.0f);
  w.z = sycl::fmax(bary.z, 0.0f);
  float w_sum = w.x + w.y + w.z;
  w_sum = sycl::fmax(w_sum, 1e-5f);
  w.x /= w_sum;
  w.y /= w_sum;
  w.z /= w_sum;

  return w;
}

inline float3 BarycentricClipBackward(
    const float3& bary,
    const float3& grad_baryclip_upstream) {
  float3 w = make_float3(0.0f, 0.0f, 0.0f);
  
  w.x = sycl::fmax(bary.x, 0.0f);
  w.y = sycl::fmax(bary.y, 0.0f);
  w.z = sycl::fmax(bary.z, 0.0f);
  float w_sum = w.x + w.y + w.z;

  float3 grad_bary = make_float3(1.0f, 1.0f, 1.0f);
  float3 grad_clip = make_float3(1.0f, 1.0f, 1.0f);
  float3 grad_sum = make_float3(1.0f, 1.0f, 1.0f);

  float grad_sum_clip = 1.0f;
  if (w_sum < 1e-5f) {
    grad_sum_clip = 0.0f;
    w_sum = 1e-5f;
  }

  if (bary.x < 0.0f) {
    grad_clip.x = 0.0f;
  }
  if (bary.y < 0.0f) {
    grad_clip.y = 0.0f;
  }
  if (bary.z < 0.0f) {
    grad_clip.z = 0.0f;
  }

  grad_sum.x = -w.x / sycl::pow(w_sum, 2.0f) * grad_sum_clip;
  grad_sum.y = -w.y / sycl::pow(w_sum, 2.0f) * grad_sum_clip;
  grad_sum.z = -w.z / sycl::pow(w_sum, 2.0f) * grad_sum_clip;

  grad_bary.x = grad_clip.x *
      (grad_baryclip_upstream.x * (1.0f / w_sum + grad_sum.x) +
       grad_baryclip_upstream.y * grad_sum.y +
       grad_baryclip_upstream.z * grad_sum.z);

  grad_bary.y = grad_clip.y *
      (grad_baryclip_upstream.y * (1.0f / w_sum + grad_sum.y) +
       grad_baryclip_upstream.x * grad_sum.x +
       grad_baryclip_upstream.z * grad_sum.z);

  grad_bary.z = grad_clip.z *
      (grad_baryclip_upstream.z * (1.0f / w_sum + grad_sum.z) +
       grad_baryclip_upstream.x * grad_sum.x +
       grad_baryclip_upstream.y * grad_sum.y);

  return grad_bary;
}

inline float
PointLineDistanceForward(const float2& p, const float2& a, const float2& b) {
  const float2 ba = b - a;
  float l2 = dot(ba, ba);
  float t = dot(ba, p - a) / l2;
  if (l2 <= kEpsilon) {
    return dot(p - b, p - b);
  }
  t = sycl::fmin(sycl::fmax(t, 0.0f), 1.0f);
  const float2 p_proj = a + t * ba;
  const float2 d = p_proj - p;
  return dot(d, d);
}

inline oneapi::dpl::tuple<float2, float2, float2>
PointLineDistanceBackward(
    const float2& p,
    const float2& v0,
    const float2& v1,
    const float& grad_dist) {
  const float2 v1v0 = v1 - v0;
  const float2 pv0 = p - v0;
  const float t_bot = dot(v1v0, v1v0);
  const float t_top = dot(v1v0, pv0);
  float tt = t_top / t_bot;
  tt = sycl::fmin(sycl::fmax(tt, 0.0f), 1.0f);
  const float2 p_proj = (1.0f - tt) * v0 + tt * v1;

  const float2 grad_p = -1.0f * grad_dist * 2.0f * (p_proj - p);
  const float2 grad_v0 = grad_dist * (1.0f - tt) * 2.0f * (p_proj - p);
  const float2 grad_v1 = grad_dist * tt * 2.0f * (p_proj - p);

  return oneapi::dpl::make_tuple(grad_p, grad_v0, grad_v1);
}

inline float PointTriangleDistanceForward(
    const float2& p,
    const float2& v0,
    const float2& v1,
    const float2& v2) {
  const float e01_dist = PointLineDistanceForward(p, v0, v1);
  const float e02_dist = PointLineDistanceForward(p, v0, v2);
  const float e12_dist = PointLineDistanceForward(p, v1, v2);
  return sycl::fmin(sycl::fmin(e01_dist, e02_dist), e12_dist);
}

inline oneapi::dpl::tuple<float2, float2, float2, float2>
PointTriangleDistanceBackward(
    const float2& p,
    const float2& v0,
    const float2& v1,
    const float2& v2,
    const float grad_dist) {
  const float e01_dist = PointLineDistanceForward(p, v0, v1);
  const float e02_dist = PointLineDistanceForward(p, v0, v2);
  const float e12_dist = PointLineDistanceForward(p, v1, v2);

  float2 grad_p = make_float2(0.0f, 0.0f);
  float2 grad_v0 = make_float2(0.0f, 0.0f);
  float2 grad_v1 = make_float2(0.0f, 0.0f);
  float2 grad_v2 = make_float2(0.0f, 0.0f);
  
  if (e01_dist <= e02_dist && e01_dist <= e12_dist) {
    const auto grad_e01 = PointLineDistanceBackward(p, v0, v1, grad_dist);
    grad_p = oneapi::dpl::get<0>(grad_e01);
    grad_v0 = oneapi::dpl::get<1>(grad_e01);
    grad_v1 = oneapi::dpl::get<2>(grad_e01);
  } else if (e02_dist <= e01_dist && e02_dist <= e12_dist) {
    const auto grad_e02 = PointLineDistanceBackward(p, v0, v2, grad_dist);
    grad_p = oneapi::dpl::get<0>(grad_e02);
    grad_v0 = oneapi::dpl::get<1>(grad_e02);
    grad_v2 = oneapi::dpl::get<2>(grad_e02);
  } else if (e12_dist <= e01_dist && e12_dist <= e02_dist) {
    const auto grad_e12 = PointLineDistanceBackward(p, v1, v2, grad_dist);
    grad_p = oneapi::dpl::get<0>(grad_e12);
    grad_v1 = oneapi::dpl::get<1>(grad_e12);
    grad_v2 = oneapi::dpl::get<2>(grad_e12);
  }

  return oneapi::dpl::make_tuple(grad_p, grad_v0, grad_v1, grad_v2);
}

} // namespace geometry_utils_sycl
} // namespace pytorch3d