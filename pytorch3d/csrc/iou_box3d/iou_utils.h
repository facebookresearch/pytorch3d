/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <math.h>
#include <stdint.h>

// Intersection volume of two boxes, shared by the CPU and CUDA
// implementations.
//
// Box1 is treated as a closed convex polyhedron and is clipped by the six
// half-spaces of box2 one at a time. Each cut closes the polyhedron again with
// a cap face built from the same vertex classifications, so the surface stays
// closed without any coplanarity tests or tolerances. The volume comes from the
// divergence theorem with signed terms, which does not depend on a reference
// point inside the shape.

#ifdef __CUDACC__
#define IOU_HOST_DEVICE __host__ __device__
#else
#define IOU_HOST_DEVICE
#endif

namespace iou_box3d {

// In exact arithmetic the clipped polyhedron has at most 6 + 6 faces (each
// cut adds a single cap face), an original face has at most 4 + 6 vertices and
// a cap at most 12. Rounding on boxes whose faces are not quite planar can add
// slivers with more vertices; the most seen in testing was 20.
constexpr int kMaxFaces = 12;
constexpr int kMaxFaceVerts = 32;
constexpr int kMaxCapEdges = 32;

template <typename T>
struct Point {
  T x, y, z;
};

template <typename T>
IOU_HOST_DEVICE inline Point<T> operator+(
    const Point<T>& a,
    const Point<T>& b) {
  return {a.x + b.x, a.y + b.y, a.z + b.z};
}

template <typename T>
IOU_HOST_DEVICE inline Point<T> operator-(
    const Point<T>& a,
    const Point<T>& b) {
  return {a.x - b.x, a.y - b.y, a.z - b.z};
}

template <typename T>
IOU_HOST_DEVICE inline Point<T> operator*(const Point<T>& a, const T b) {
  return {a.x * b, a.y * b, a.z * b};
}

template <typename T>
IOU_HOST_DEVICE inline bool operator==(const Point<T>& a, const Point<T>& b) {
  return a.x == b.x && a.y == b.y && a.z == b.z;
}

template <typename T>
IOU_HOST_DEVICE inline T Dot(const Point<T>& a, const Point<T>& b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

IOU_HOST_DEVICE inline float Fma(const float a, const float b, const float c) {
  return fmaf(a, b, c);
}

IOU_HOST_DEVICE inline double
Fma(const double a, const double b, const double c) {
  return fma(a, b, c);
}

template <typename T>
IOU_HOST_DEVICE inline Point<T> Cross(const Point<T>& a, const Point<T>& b) {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

// Corner indices of face f of a box, counterclockwise when seen from outside
// for the corner order documented in box3d_overlap.
IOU_HOST_DEVICE inline int BoxFaceCorner(const int f, const int i) {
  const int faces[6][4] = {
      {0, 3, 2, 1},
      {4, 5, 6, 7},
      {0, 1, 5, 4},
      {3, 7, 6, 2},
      {0, 4, 7, 3},
      {1, 2, 6, 5},
  };
  return faces[f][i];
}

// A polyhedron stored as faces, each an ordered loop of vertices.
template <typename T>
struct Polyhedron {
  Point<T> verts[kMaxFaces][kMaxFaceVerts];
  int num_verts[kMaxFaces];
  int num_faces;
};

template <typename T>
IOU_HOST_DEVICE inline void LoadBox(const float* box, Point<T> corners[8]) {
  for (int i = 0; i < 8; ++i) {
    corners[i] = {T(box[3 * i]), T(box[3 * i + 1]), T(box[3 * i + 2])};
  }
}

template <typename T>
IOU_HOST_DEVICE inline Point<T> BoxCenter(const Point<T> corners[8]) {
  Point<T> sum = corners[0];
  for (int i = 1; i < 8; ++i) {
    sum = sum + corners[i];
  }
  return sum * T(0.125);
}

// Positive for boxes with the documented corner order, negative for their
// mirror images.
template <typename T>
IOU_HOST_DEVICE inline T BoxSignedVolume(
    const Point<T> corners[8],
    const Point<T>& center) {
  T vol = 0;
  for (int f = 0; f < 6; ++f) {
    const Point<T> v0 = corners[BoxFaceCorner(f, 0)] - center;
    const Point<T> v1 = corners[BoxFaceCorner(f, 1)] - center;
    const Point<T> v2 = corners[BoxFaceCorner(f, 2)] - center;
    const Point<T> v3 = corners[BoxFaceCorner(f, 3)] - center;
    vol += Dot(v0, Cross(v1, v2)) + Dot(v0, Cross(v2, v3));
  }
  return vol / 6;
}

template <typename T>
IOU_HOST_DEVICE inline T PolyhedronSignedVolume(const Polyhedron<T>& poly) {
  T vol = 0;
  for (int f = 0; f < poly.num_faces; ++f) {
    const Point<T> v0 = poly.verts[f][0];
    for (int i = 1; i + 1 < poly.num_verts[f]; ++i) {
      vol += Dot(v0, Cross(poly.verts[f][i], poly.verts[f][i + 1]));
    }
  }
  return vol / 6;
}

// Signed distance of p from the plane, scaled by |normal|. Every copy of a
// vertex must be classified identically, so this uses explicit fma, which
// the compiler cannot contract differently at different call sites.
template <typename T>
IOU_HOST_DEVICE inline T
PlaneDistance(const Point<T>& normal, const T offset, const Point<T>& p) {
  return Fma(normal.z, p.z, Fma(normal.y, p.y, Fma(normal.x, p.x, -offset)));
}

// The point where the edge from vin (s_in <= 0) to vout (s_out > 0) crosses
// the plane. The two faces sharing an edge call this with the same arguments,
// so they get bitwise identical points.
template <typename T>
IOU_HOST_DEVICE inline Point<T> EdgeCrossing(
    const Point<T>& vin,
    const Point<T>& vout,
    const T s_in,
    const T s_out) {
  const T t = s_in / (s_in - s_out);
  return {
      Fma(vout.x - vin.x, t, vin.x),
      Fma(vout.y - vin.y, t, vin.y),
      Fma(vout.z - vin.z, t, vin.z)};
}

// Appends p to a vertex loop unless it repeats the previous vertex, which
// happens when a vertex lies exactly on the plane.
template <typename T>
IOU_HOST_DEVICE inline void
PushVertex(Point<T>* loop, int& len, const Point<T>& p) {
  if (len == 0 || !(loop[len - 1] == p)) {
    loop[len++] = p;
  }
}

// Records the cut edge from -> to of the cap face. Returns false on buffer
// overflow.
template <typename T>
IOU_HOST_DEVICE inline bool AddCapEdge(
    const Point<T>& from,
    const Point<T>& to,
    Point<T>* cap_from,
    Point<T>* cap_to,
    int& num_cap_edges) {
  if (from == to) {
    return true;
  }
  if (num_cap_edges == kMaxCapEdges) {
    return false;
  }
  cap_from[num_cap_edges] = from;
  cap_to[num_cap_edges] = to;
  ++num_cap_edges;
  return true;
}

// Clips one face to the half-space s <= 0 (Sutherland-Hodgman). Writes the
// clipped loop to out. Where the loop leaves the half-space at X and comes
// back at E, the face gets the edge X -> E and the cap face the edge E -> X,
// which is appended to cap_from/cap_to. Returns false on buffer overflow.
template <typename T>
IOU_HOST_DEVICE inline bool ClipFace(
    const Point<T>* verts,
    const int num_verts,
    const Point<T>& normal,
    const T offset,
    Point<T>* out,
    int& num_out,
    Point<T>* cap_from,
    Point<T>* cap_to,
    int& num_cap_edges) {
  T s[kMaxFaceVerts];
  for (int i = 0; i < num_verts; ++i) {
    s[i] = PlaneDistance(normal, offset, verts[i]);
  }
  bool have_exit = false;
  bool have_first_entry = false;
  Point<T> exit;
  Point<T> first_entry;
  num_out = 0;
  int prev = num_verts - 1;
  for (int cur = 0; cur < num_verts; prev = cur++) {
    const bool prev_in = s[prev] <= 0;
    const bool cur_in = s[cur] <= 0;
    if (num_out + 2 > kMaxFaceVerts) {
      return false;
    }
    if (prev_in && !cur_in) {
      exit = EdgeCrossing(verts[prev], verts[cur], s[prev], s[cur]);
      have_exit = true;
      PushVertex(out, num_out, exit);
    } else if (!prev_in && cur_in) {
      const Point<T> entry =
          EdgeCrossing(verts[cur], verts[prev], s[cur], s[prev]);
      if (have_exit) {
        if (!AddCapEdge(entry, exit, cap_from, cap_to, num_cap_edges)) {
          return false;
        }
        have_exit = false;
      } else {
        first_entry = entry;
        have_first_entry = true;
      }
      PushVertex(out, num_out, entry);
    }
    if (cur_in) {
      PushVertex(out, num_out, verts[cur]);
    }
  }
  if (have_exit && have_first_entry &&
      !AddCapEdge(first_entry, exit, cap_from, cap_to, num_cap_edges)) {
    return false;
  }
  if (num_out > 1 && out[num_out - 1] == out[0]) {
    --num_out;
  }
  return true;
}

// Chains the cap edges into loops and appends them as one face. In exact
// arithmetic there is a single convex loop, but non-planar input faces can
// give several. Each extra loop is joined to the first vertex by a bridge
// edge traversed in both directions, which adds no volume and is handled
// consistently by later clipping.
template <typename T>
IOU_HOST_DEVICE inline bool AddCapFace(
    Polyhedron<T>& poly,
    const Point<T>* cap_from,
    const Point<T>* cap_to,
    const int num_cap_edges) {
  if (num_cap_edges == 0) {
    return true;
  }
  if (poly.num_faces == kMaxFaces) {
    return false;
  }
  Point<T>* cap = poly.verts[poly.num_faces];
  int len = 0;
  uint32_t used = 0;
  for (int start = 0; start < num_cap_edges; ++start) {
    if (used & (1u << start)) {
      continue;
    }
    Point<T> loop[kMaxFaceVerts];
    int loop_len = 0;
    for (int cur = start; cur >= 0;) {
      if (loop_len == kMaxFaceVerts) {
        return false;
      }
      used |= 1u << cur;
      loop[loop_len++] = cap_from[cur];
      const int edge = cur;
      cur = -1;
      for (int j = 0; j < num_cap_edges; ++j) {
        if (!(used & (1u << j)) && cap_from[j] == cap_to[edge]) {
          cur = j;
          break;
        }
      }
    }
    if (loop_len < 3) {
      continue;
    }
    const bool bridge = len > 0;
    if (len + loop_len + 2 * bridge > kMaxFaceVerts) {
      return false;
    }
    if (bridge) {
      cap[len++] = cap[0];
    }
    for (int i = 0; i < loop_len; ++i) {
      cap[len++] = loop[i];
    }
    if (bridge) {
      cap[len++] = loop[0];
    }
  }
  if (len >= 3) {
    poly.num_verts[poly.num_faces++] = len;
  }
  return true;
}

// Clips poly to the half-space dot(normal, p) <= offset.
template <typename T>
IOU_HOST_DEVICE inline bool
ClipByPlane(Polyhedron<T>& poly, const Point<T>& normal, const T offset) {
  Point<T> cap_from[kMaxCapEdges];
  Point<T> cap_to[kMaxCapEdges];
  int num_cap_edges = 0;
  int num_faces = 0;
  for (int f = 0; f < poly.num_faces; ++f) {
    Point<T> out[kMaxFaceVerts];
    int num_out = 0;
    const bool ok = ClipFace(
        poly.verts[f],
        poly.num_verts[f],
        normal,
        offset,
        out,
        num_out,
        cap_from,
        cap_to,
        num_cap_edges);
    if (!ok) {
      return false;
    }
    if (num_out >= 3) {
      for (int i = 0; i < num_out; ++i) {
        poly.verts[num_faces][i] = out[i];
      }
      poly.num_verts[num_faces++] = num_out;
    }
  }
  poly.num_faces = num_faces;
  return AddCapFace(poly, cap_from, cap_to, num_cap_edges);
}

template <typename T>
IOU_HOST_DEVICE inline bool BoxesDisjoint(
    const Point<T> corners1[8],
    const Point<T> corners2[8]) {
  Point<T> lo1 = corners1[0], hi1 = corners1[0];
  Point<T> lo2 = corners2[0], hi2 = corners2[0];
  for (int i = 1; i < 8; ++i) {
    lo1 = {
        fmin(lo1.x, corners1[i].x),
        fmin(lo1.y, corners1[i].y),
        fmin(lo1.z, corners1[i].z)};
    hi1 = {
        fmax(hi1.x, corners1[i].x),
        fmax(hi1.y, corners1[i].y),
        fmax(hi1.z, corners1[i].z)};
    lo2 = {
        fmin(lo2.x, corners2[i].x),
        fmin(lo2.y, corners2[i].y),
        fmin(lo2.z, corners2[i].z)};
    hi2 = {
        fmax(hi2.x, corners2[i].x),
        fmax(hi2.y, corners2[i].y),
        fmax(hi2.z, corners2[i].z)};
  }
  return hi1.x < lo2.x || hi2.x < lo1.x || hi1.y < lo2.y || hi2.y < lo1.y ||
      hi1.z < lo2.z || hi2.z < lo1.z;
}

// Computes the intersection volume and IoU of two boxes, each given as 8
// corners (24 floats). Gives NaN if a fixed-size buffer would overflow, which
// should not happen for valid boxes.
template <typename T>
IOU_HOST_DEVICE inline void
BoxOverlap(const float* box1, const float* box2, T& vol, T& iou) {
  vol = 0;
  iou = 0;
  Point<T> corners1[8];
  Point<T> corners2[8];
  LoadBox(box1, corners1);
  LoadBox(box2, corners2);
  if (BoxesDisjoint(corners1, corners2)) {
    return;
  }

  const Point<T> center1 = BoxCenter(corners1);
  const Point<T> center2 = BoxCenter(corners2);
  const T signed_vol1 = BoxSignedVolume(corners1, center1);
  const T vol1 = fabs(signed_vol1);
  const T signed_vol2 = BoxSignedVolume(corners2, center2);
  const T vol2 = fabs(signed_vol2);

  // Work relative to the center of box1 for accuracy far from the origin.
  for (int i = 0; i < 8; ++i) {
    corners1[i] = corners1[i] - center1;
    corners2[i] = corners2[i] - center1;
  }

  Polyhedron<T> poly;
  poly.num_faces = 6;
  for (int f = 0; f < 6; ++f) {
    poly.num_verts[f] = 4;
    for (int i = 0; i < 4; ++i) {
      poly.verts[f][i] = corners1[BoxFaceCorner(f, i)];
    }
  }

  for (int f = 0; f < 6 && poly.num_faces > 0; ++f) {
    const Point<T> v0 = corners2[BoxFaceCorner(f, 0)];
    const Point<T> v1 = corners2[BoxFaceCorner(f, 1)];
    const Point<T> v2 = corners2[BoxFaceCorner(f, 2)];
    const Point<T> v3 = corners2[BoxFaceCorner(f, 3)];
    // Outward for the documented corner order, inward for its mirror image.
    Point<T> normal = Cross(v2 - v0, v3 - v1);
    if (signed_vol2 < 0) {
      normal = normal * T(-1);
    }
    const Point<T> mid = (v0 + v1 + v2 + v3) * T(0.25);
    if (!ClipByPlane(poly, normal, Dot(normal, mid))) {
      vol = iou = T(NAN);
      return;
    }
  }

  // The clipped polyhedron keeps the orientation of box1.
  const T signed_vol = PolyhedronSignedVolume(poly);
  vol = fmax(signed_vol1 < 0 ? -signed_vol : signed_vol, T(0));
  const T union_vol = vol1 + vol2 - vol;
  iou = union_vol > 0 ? vol / union_vol : T(0);
}

} // namespace iou_box3d
