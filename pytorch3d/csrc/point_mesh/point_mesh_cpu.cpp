/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <torch/extension.h>
#include <array>
#include <limits>
#include "utils/geometry_utils.h"
#include "utils/vec3.h"

// - We start with implementations of simple operations on points, edges and
// faces. The hull of H points is a point if H=1, an edge if H=2, a face if H=3.

template <typename T>
vec3<T> ExtractPoint(const at::TensorAccessor<T, 1>& t) {
  return vec3<T>(t[0], t[1], t[2]);
}

template <typename Accessor>
static std::array<vec3<std::remove_pointer_t<typename Accessor::PtrType>>, 1>
ExtractHullHelper(const Accessor& t, std::array<char, 1> /*tag*/) {
  return {ExtractPoint(t)};
}

template <typename Accessor>
static std::array<vec3<std::remove_pointer_t<typename Accessor::PtrType>>, 2>
ExtractHullHelper(const Accessor& t, std::array<char, 2> /*tag*/) {
  return {ExtractPoint(t[0]), ExtractPoint(t[1])};
}

template <typename Accessor>
static std::array<vec3<std::remove_pointer_t<typename Accessor::PtrType>>, 3>
ExtractHullHelper(const Accessor& t, std::array<char, 3> /*tag*/) {
  return {ExtractPoint(t[0]), ExtractPoint(t[1]), ExtractPoint(t[2])};
}

template <int H, typename Accessor>
std::array<vec3<std::remove_pointer_t<typename Accessor::PtrType>>, H>
ExtractHull(const Accessor& t) {
  std::array<char, H> tag;
  return ExtractHullHelper(t, tag);
}

template <typename T>
void IncrementPoint(at::TensorAccessor<T, 1>&& t, const vec3<T>& point) {
  t[0] += point.x;
  t[1] += point.y;
  t[2] += point.z;
}

// distance between the convex hull of A points and B points
// this could be done in c++17 with tuple_cat and invoke
template <typename T>
T HullDistance(
    const std::array<vec3<T>, 1>& a,
    const std::array<vec3<T>, 2>& b,
    const double /*min_triangle_area*/) {
  using std::get;
  return PointLine3DistanceForward(get<0>(a), get<0>(b), get<1>(b));
}
template <typename T>
T HullDistance(
    const std::array<vec3<T>, 1>& a,
    const std::array<vec3<T>, 3>& b,
    const double min_triangle_area) {
  using std::get;
  return PointTriangle3DistanceForward(
      get<0>(a), get<0>(b), get<1>(b), get<2>(b), min_triangle_area);
}
template <typename T>
T HullDistance(
    const std::array<vec3<T>, 2>& a,
    const std::array<vec3<T>, 1>& b,
    const double /*min_triangle_area*/) {
  return HullDistance(b, a, 1);
}
template <typename T>
T HullDistance(
    const std::array<vec3<T>, 3>& a,
    const std::array<vec3<T>, 1>& b,
    const double min_triangle_area) {
  return HullDistance(b, a, min_triangle_area);
}

template <typename T>
void HullHullDistanceBackward(
    const std::array<vec3<T>, 1>& a,
    const std::array<vec3<T>, 2>& b,
    T grad_dist,
    at::TensorAccessor<T, 1>&& grad_a,
    at::TensorAccessor<T, 2>&& grad_b,
    const double /*min_triangle_area*/) {
  using std::get;
  auto res =
      PointLine3DistanceBackward(get<0>(a), get<0>(b), get<1>(b), grad_dist);
  IncrementPoint(std::move(grad_a), get<0>(res));
  IncrementPoint(grad_b[0], get<1>(res));
  IncrementPoint(grad_b[1], get<2>(res));
}
template <typename T>
void HullHullDistanceBackward(
    const std::array<vec3<T>, 1>& a,
    const std::array<vec3<T>, 3>& b,
    T grad_dist,
    at::TensorAccessor<T, 1>&& grad_a,
    at::TensorAccessor<T, 2>&& grad_b,
    const double min_triangle_area) {
  using std::get;
  auto res = PointTriangle3DistanceBackward(
      get<0>(a), get<0>(b), get<1>(b), get<2>(b), grad_dist, min_triangle_area);
  IncrementPoint(std::move(grad_a), get<0>(res));
  IncrementPoint(grad_b[0], get<1>(res));
  IncrementPoint(grad_b[1], get<2>(res));
  IncrementPoint(grad_b[2], get<3>(res));
}
template <typename T>
void HullHullDistanceBackward(
    const std::array<vec3<T>, 3>& a,
    const std::array<vec3<T>, 1>& b,
    T grad_dist,
    at::TensorAccessor<T, 2>&& grad_a,
    at::TensorAccessor<T, 1>&& grad_b,
    const double min_triangle_area) {
  return HullHullDistanceBackward(
      b, a, grad_dist, std::move(grad_b), std::move(grad_a), min_triangle_area);
}
template <typename T>
void HullHullDistanceBackward(
    const std::array<vec3<T>, 2>& a,
    const std::array<vec3<T>, 1>& b,
    T grad_dist,
    at::TensorAccessor<T, 2>&& grad_a,
    at::TensorAccessor<T, 1>&& grad_b,
    const double /*min_triangle_area*/) {
  return HullHullDistanceBackward(
      b, a, grad_dist, std::move(grad_b), std::move(grad_a), 1);
}

template <int H>
void ValidateShape(const at::Tensor& as) {
  if (H == 1) {
    TORCH_CHECK(as.size(1) == 3);
  } else {
    TORCH_CHECK(as.size(2) == 3 && as.size(1) == H);
  }
}

// ----------- Here begins the implementation of each top-level
//             function using non-type template parameters to
//             implement all the cases in one go. ----------- //

template <int H1, int H2>
std::tuple<at::Tensor, at::Tensor> HullHullDistanceForwardCpu(
    const at::Tensor& as,
    const at::Tensor& as_first_idx,
    const at::Tensor& bs,
    const at::Tensor& bs_first_idx,
    const double min_triangle_area) {
  const int64_t A_N = as.size(0);
  const int64_t B_N = bs.size(0);
  const int64_t BATCHES = as_first_idx.size(0);

  ValidateShape<H1>(as);
  ValidateShape<H2>(bs);

  TORCH_CHECK(bs_first_idx.size(0) == BATCHES);

  // clang-format off
  at::Tensor dists = at::zeros({A_N,}, as.options());
  at::Tensor idxs = at::zeros({A_N,}, as_first_idx.options());
  // clang-format on

  auto as_a = as.accessor < float, H1 == 1 ? 2 : 3 > ();
  auto bs_a = bs.accessor < float, H2 == 1 ? 2 : 3 > ();
  auto as_first_idx_a = as_first_idx.accessor<int64_t, 1>();
  auto bs_first_idx_a = bs_first_idx.accessor<int64_t, 1>();
  auto dists_a = dists.accessor<float, 1>();
  auto idxs_a = idxs.accessor<int64_t, 1>();
  int64_t a_batch_end = 0;
  int64_t b_batch_start = 0, b_batch_end = 0;
  int64_t batch_idx = 0;
  for (int64_t a_n = 0; a_n < A_N; ++a_n) {
    if (a_n == a_batch_end) {
      ++batch_idx;
      b_batch_start = b_batch_end;
      if (batch_idx == BATCHES) {
        a_batch_end = std::numeric_limits<int64_t>::max();
        b_batch_end = B_N;
      } else {
        a_batch_end = as_first_idx_a[batch_idx];
        b_batch_end = bs_first_idx_a[batch_idx];
      }
    }
    float min_dist = std::numeric_limits<float>::max();
    size_t min_idx = 0;
    auto a = ExtractHull<H1>(as_a[a_n]);
    for (int64_t b_n = b_batch_start; b_n < b_batch_end; ++b_n) {
      float dist =
          HullDistance(a, ExtractHull<H2>(bs_a[b_n]), min_triangle_area);
      if (dist <= min_dist) {
        min_dist = dist;
        min_idx = b_n;
      }
    }
    dists_a[a_n] = min_dist;
    idxs_a[a_n] = min_idx;
  }

  return std::make_tuple(dists, idxs);
}

template <int H1, int H2>
std::tuple<at::Tensor, at::Tensor> HullHullDistanceBackwardCpu(
    const at::Tensor& as,
    const at::Tensor& bs,
    const at::Tensor& idx_bs,
    const at::Tensor& grad_dists,
    const double min_triangle_area) {
  const int64_t A_N = as.size(0);

  TORCH_CHECK(idx_bs.size(0) == A_N);
  TORCH_CHECK(grad_dists.size(0) == A_N);
  ValidateShape<H1>(as);
  ValidateShape<H2>(bs);

  at::Tensor grad_as = at::zeros_like(as);
  at::Tensor grad_bs = at::zeros_like(bs);

  auto as_a = as.accessor < float, H1 == 1 ? 2 : 3 > ();
  auto bs_a = bs.accessor < float, H2 == 1 ? 2 : 3 > ();
  auto grad_as_a = grad_as.accessor < float, H1 == 1 ? 2 : 3 > ();
  auto grad_bs_a = grad_bs.accessor < float, H2 == 1 ? 2 : 3 > ();
  auto idx_bs_a = idx_bs.accessor<int64_t, 1>();
  auto grad_dists_a = grad_dists.accessor<float, 1>();

  for (int64_t a_n = 0; a_n < A_N; ++a_n) {
    auto a = ExtractHull<H1>(as_a[a_n]);
    auto b = ExtractHull<H2>(bs_a[idx_bs_a[a_n]]);
    HullHullDistanceBackward(
        a,
        b,
        grad_dists_a[a_n],
        grad_as_a[a_n],
        grad_bs_a[idx_bs_a[a_n]],
        min_triangle_area);
  }
  return std::make_tuple(grad_as, grad_bs);
}

template <int H>
torch::Tensor PointHullArrayDistanceForwardCpu(
    const torch::Tensor& points,
    const torch::Tensor& bs,
    const double min_triangle_area) {
  const int64_t P = points.size(0);
  const int64_t B_N = bs.size(0);

  TORCH_CHECK(points.size(1) == 3, "points must be of shape Px3");
  ValidateShape<H>(bs);

  at::Tensor dists = at::zeros({P, B_N}, points.options());
  auto points_a = points.accessor<float, 2>();
  auto bs_a = bs.accessor<float, 3>();
  auto dists_a = dists.accessor<float, 2>();
  for (int64_t p = 0; p < P; ++p) {
    auto point = ExtractHull<1>(points_a[p]);
    auto dest = dists_a[p];
    for (int64_t b_n = 0; b_n < B_N; ++b_n) {
      auto b = ExtractHull<H>(bs_a[b_n]);
      dest[b_n] = HullDistance(point, b, min_triangle_area);
    }
  }
  return dists;
}

template <int H>
std::tuple<at::Tensor, at::Tensor> PointHullArrayDistanceBackwardCpu(
    const at::Tensor& points,
    const at::Tensor& bs,
    const at::Tensor& grad_dists,
    const double min_triangle_area) {
  const int64_t P = points.size(0);
  const int64_t B_N = bs.size(0);

  TORCH_CHECK(points.size(1) == 3, "points must be of shape Px3");
  ValidateShape<H>(bs);
  TORCH_CHECK((grad_dists.size(0) == P) && (grad_dists.size(1) == B_N));

  at::Tensor grad_points = at::zeros({P, 3}, points.options());
  at::Tensor grad_bs = at::zeros({B_N, H, 3}, bs.options());

  auto points_a = points.accessor<float, 2>();
  auto bs_a = bs.accessor<float, 3>();
  auto grad_dists_a = grad_dists.accessor<float, 2>();
  auto grad_points_a = grad_points.accessor<float, 2>();
  auto grad_bs_a = grad_bs.accessor<float, 3>();
  for (int64_t p = 0; p < P; ++p) {
    auto point = ExtractHull<1>(points_a[p]);
    auto grad_point = grad_points_a[p];
    auto grad_dist = grad_dists_a[p];
    for (int64_t b_n = 0; b_n < B_N; ++b_n) {
      auto b = ExtractHull<H>(bs_a[b_n]);
      HullHullDistanceBackward(
          point,
          b,
          grad_dist[b_n],
          std::move(grad_point),
          grad_bs_a[b_n],
          min_triangle_area);
    }
  }
  return std::make_tuple(grad_points, grad_bs);
}

// ---------- Here begin the exported functions ------------ //

std::tuple<torch::Tensor, torch::Tensor> PointFaceDistanceForwardCpu(
    const torch::Tensor& points,
    const torch::Tensor& points_first_idx,
    const torch::Tensor& tris,
    const torch::Tensor& tris_first_idx,
    const double min_triangle_area) {
  return HullHullDistanceForwardCpu<1, 3>(
      points, points_first_idx, tris, tris_first_idx, min_triangle_area);
}

std::tuple<torch::Tensor, torch::Tensor> PointFaceDistanceBackwardCpu(
    const torch::Tensor& points,
    const torch::Tensor& tris,
    const torch::Tensor& idx_points,
    const torch::Tensor& grad_dists,
    const double min_triangle_area) {
  return HullHullDistanceBackwardCpu<1, 3>(
      points, tris, idx_points, grad_dists, min_triangle_area);
}

std::tuple<torch::Tensor, torch::Tensor> FacePointDistanceForwardCpu(
    const torch::Tensor& points,
    const torch::Tensor& points_first_idx,
    const torch::Tensor& tris,
    const torch::Tensor& tris_first_idx,
    const double min_triangle_area) {
  return HullHullDistanceForwardCpu<3, 1>(
      tris, tris_first_idx, points, points_first_idx, min_triangle_area);
}

std::tuple<torch::Tensor, torch::Tensor> FacePointDistanceBackwardCpu(
    const torch::Tensor& points,
    const torch::Tensor& tris,
    const torch::Tensor& idx_tris,
    const torch::Tensor& grad_dists,
    const double min_triangle_area) {
  auto res = HullHullDistanceBackwardCpu<3, 1>(
      tris, points, idx_tris, grad_dists, min_triangle_area);
  return std::make_tuple(std::get<1>(res), std::get<0>(res));
}

torch::Tensor PointEdgeArrayDistanceForwardCpu(
    const torch::Tensor& points,
    const torch::Tensor& segms) {
  return PointHullArrayDistanceForwardCpu<2>(points, segms, 1);
}

std::tuple<at::Tensor, at::Tensor> PointFaceArrayDistanceBackwardCpu(
    const at::Tensor& points,
    const at::Tensor& tris,
    const at::Tensor& grad_dists,
    const double min_triangle_area) {
  return PointHullArrayDistanceBackwardCpu<3>(
      points, tris, grad_dists, min_triangle_area);
}

torch::Tensor PointFaceArrayDistanceForwardCpu(
    const torch::Tensor& points,
    const torch::Tensor& tris,
    const double min_triangle_area) {
  return PointHullArrayDistanceForwardCpu<3>(points, tris, min_triangle_area);
}

std::tuple<at::Tensor, at::Tensor> PointEdgeArrayDistanceBackwardCpu(
    const at::Tensor& points,
    const at::Tensor& segms,
    const at::Tensor& grad_dists) {
  return PointHullArrayDistanceBackwardCpu<2>(points, segms, grad_dists, 1);
}

std::tuple<torch::Tensor, torch::Tensor> PointEdgeDistanceForwardCpu(
    const torch::Tensor& points,
    const torch::Tensor& points_first_idx,
    const torch::Tensor& segms,
    const torch::Tensor& segms_first_idx,
    const int64_t /*max_points*/) {
  return HullHullDistanceForwardCpu<1, 2>(
      points, points_first_idx, segms, segms_first_idx, 1);
}

std::tuple<torch::Tensor, torch::Tensor> PointEdgeDistanceBackwardCpu(
    const torch::Tensor& points,
    const torch::Tensor& segms,
    const torch::Tensor& idx_points,
    const torch::Tensor& grad_dists) {
  return HullHullDistanceBackwardCpu<1, 2>(
      points, segms, idx_points, grad_dists, 1);
}

std::tuple<torch::Tensor, torch::Tensor> EdgePointDistanceForwardCpu(
    const torch::Tensor& points,
    const torch::Tensor& points_first_idx,
    const torch::Tensor& segms,
    const torch::Tensor& segms_first_idx,
    const int64_t /*max_segms*/) {
  return HullHullDistanceForwardCpu<2, 1>(
      segms, segms_first_idx, points, points_first_idx, 1);
}

std::tuple<torch::Tensor, torch::Tensor> EdgePointDistanceBackwardCpu(
    const torch::Tensor& points,
    const torch::Tensor& segms,
    const torch::Tensor& idx_segms,
    const torch::Tensor& grad_dists) {
  auto res = HullHullDistanceBackwardCpu<2, 1>(
      segms, points, idx_segms, grad_dists, 1);
  return std::make_tuple(std::get<1>(res), std::get<0>(res));
}
