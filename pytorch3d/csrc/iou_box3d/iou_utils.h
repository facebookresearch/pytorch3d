/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <assert.h>
#include <torch/extension.h>
#include <torch/torch.h>
#include <algorithm>
#include <list>
#include <numeric>
#include <queue>
#include <tuple>
#include <type_traits>
#include "utils/vec3.h"

// dEpsilon: Used in dot products and is used to assess whether two unit vectors
// are orthogonal (or coplanar). It's an epsilon on cos(θ).
// It is also the distance, relative to the face size, within which two box
// faces are considered to lie in the same plane.
const auto dEpsilon = 1e-3;
// kEpsilon: Used only for norm(u) = u/max(||u||, kEpsilon)
const auto kEpsilon = 1e-8;

/*
_PLANES and _TRIS define the 4- and 3-connectivity
of the 8 box corners.
_PLANES gives the quad faces of the 3D box
_TRIS gives the triangle faces of the 3D box
*/
const int NUM_PLANES = 6;
const int NUM_TRIS = 12;
const int _PLANES[6][4] = {
    {0, 1, 2, 3},
    {3, 2, 6, 7},
    {0, 1, 5, 4},
    {0, 3, 7, 4},
    {1, 5, 6, 2},
    {4, 5, 6, 7},
};
const int _TRIS[12][3] = {
    {0, 1, 2},
    {0, 3, 2},
    {4, 5, 6},
    {4, 6, 7},
    {1, 5, 6},
    {1, 6, 2},
    {0, 4, 7},
    {0, 7, 3},
    {3, 2, 6},
    {3, 6, 7},
    {0, 1, 5},
    {0, 4, 5},
};

// Create a new data type for representing the
// verts for each face which can be triangle or plane.
// This helps make the code more readable.
using face_verts = std::vector<std::vector<vec3<float>>>;

// Args
//    box: (8, 3) tensor accessor for the box vertices
//    plane_idx: index of the plane in the box
//    vert_idx: index of the vertex in the plane
//
// Returns
//    vec3<T> (x, y, x) vertex coordinates
//
template <typename Box>
inline vec3<float>
ExtractVertsPlane(const Box& box, const int plane_idx, const int vert_idx) {
  return vec3<float>(
      box[_PLANES[plane_idx][vert_idx]][0],
      box[_PLANES[plane_idx][vert_idx]][1],
      box[_PLANES[plane_idx][vert_idx]][2]);
}

// Args
//    box: (8, 3) tensor accessor for the box vertices
//    tri_idx: index of the triangle face in the box
//    vert_idx: index of the vertex in the triangle
//
// Returns
//    vec3<T> (x, y, x) vertex coordinates
//
template <typename Box>
inline vec3<float>
ExtractVertsTri(const Box& box, const int tri_idx, const int vert_idx) {
  return vec3<float>(
      box[_TRIS[tri_idx][vert_idx]][0],
      box[_TRIS[tri_idx][vert_idx]][1],
      box[_TRIS[tri_idx][vert_idx]][2]);
}

// Args
//    box: (8, 3) tensor accessor for the box vertices
//
// Returns
//    std::vector<std::vector<vec3<T>>> effectively (F, 3, 3)
//      coordinates of the verts for each face
//
template <typename Box>
inline face_verts GetBoxTris(const Box& box) {
  face_verts box_tris;
  for (int t = 0; t < NUM_TRIS; ++t) {
    vec3<float> v0 = ExtractVertsTri(box, t, 0);
    vec3<float> v1 = ExtractVertsTri(box, t, 1);
    vec3<float> v2 = ExtractVertsTri(box, t, 2);
    box_tris.push_back({v0, v1, v2});
  }
  return box_tris;
}

// Args
//    box: (8, 3) tensor accessor for the box vertices
//
// Returns
//    std::vector<std::vector<vec3<T>>> effectively (P, 3, 3)
//      coordinates of the 4 verts for each plane
//
template <typename Box>
inline face_verts GetBoxPlanes(const Box& box) {
  face_verts box_planes;
  for (int t = 0; t < NUM_PLANES; ++t) {
    vec3<float> v0 = ExtractVertsPlane(box, t, 0);
    vec3<float> v1 = ExtractVertsPlane(box, t, 1);
    vec3<float> v2 = ExtractVertsPlane(box, t, 2);
    vec3<float> v3 = ExtractVertsPlane(box, t, 3);
    box_planes.push_back({v0, v1, v2, v3});
  }
  return box_planes;
}

// The normal of a plane spanned by vectors e0 and e1
//
// Args
//    e0, e1: vec3 vectors defining a plane
//
// Returns
//    vec3: normal of the plane
//
inline vec3<float> GetNormal(const vec3<float> e0, const vec3<float> e1) {
  vec3<float> n = cross(e0, e1);
  n = n / std::fmaxf(norm(n), kEpsilon);
  return n;
}

// The center of a triangle tri
//
// Args
//    tri: vec3 coordinates of the vertices of the triangle
//
// Returns
//    vec3: center of the triangle
//
inline vec3<float> TriCenter(const std::vector<vec3<float>>& tri) {
  // Vertices of the triangle
  const vec3<float> v0 = tri[0];
  const vec3<float> v1 = tri[1];
  const vec3<float> v2 = tri[2];

  return (v0 + v1 + v2) / 3.0f;
}

// The normal of the triangle defined by vertices (v0, v1, v2)
// We find the "best" edges connecting the face center to the vertices,
// such that the cross product between the edges is maximized.
//
// Args
//    tri: vec3 coordinates of the vertices of the face
//
// Returns
//    vec3: normal for the face
//
inline vec3<float> TriNormal(const std::vector<vec3<float>>& tri) {
  // Get center of triangle
  const vec3<float> ctr = TriCenter(tri);

  // find the "best" normal as cross product of edges from center
  float max_dist = -1.0f;
  vec3<float> n = {0.0f, 0.0f, 0.0f};
  for (int i = 0; i < 2; ++i) {
    for (int j = i + 1; j < 3; ++j) {
      const float dist = norm(cross(tri[i] - ctr, tri[j] - ctr));
      if (dist > max_dist) {
        n = GetNormal(tri[i] - ctr, tri[j] - ctr);
      }
    }
  }
  return n;
}

// The center of a plane
//
// Args
//    plane: vec3 coordinates of the vertices of the plane
//
// Returns
//    vec3: center of the plane
//
inline vec3<float> PlaneCenter(const std::vector<vec3<float>>& plane) {
  // Vertices of the plane
  const vec3<float> v0 = plane[0];
  const vec3<float> v1 = plane[1];
  const vec3<float> v2 = plane[2];
  const vec3<float> v3 = plane[3];

  return (v0 + v1 + v2 + v3) / 4.0f;
}

// The normal of a planar face with vertices (v0, v1, v2, v3)
// We find the "best" edges connecting the face center to the vertices,
// such that the cross product between the edges is maximized.
//
// Args
//    plane: vec3 coordinates of the vertices of the planar face
//
// Returns
//    vec3: normal of the planar face
//
inline vec3<float> PlaneNormal(const std::vector<vec3<float>>& plane) {
  // Get center of planar face
  vec3<float> ctr = PlaneCenter(plane);

  // find the "best" normal as cross product of edges from center
  float max_dist = -1.0f;
  vec3<float> n = {0.0f, 0.0f, 0.0f};
  for (int i = 0; i < 3; ++i) {
    for (int j = i + 1; j < 4; ++j) {
      const float dist = norm(cross(plane[i] - ctr, plane[j] - ctr));
      if (dist > max_dist) {
        n = GetNormal(plane[i] - ctr, plane[j] - ctr);
      }
    }
  }
  return n;
}

// The normal of a box plane defined by the verts in `plane` such that it
// points toward the centroid of the box given by `center`.
//
// Args
//    plane: vec3 coordinates of the vertices of the plane
//    center: vec3 coordinates of the center of the box from
//        which the plane originated
//
// Returns
//    vec3: normal for the plane such that it points towards
//      the center of the box
//
inline vec3<float> PlaneNormalDirection(
    const std::vector<vec3<float>>& plane,
    const vec3<float>& center) {
  // The plane's center & normal
  const vec3<float> plane_center = PlaneCenter(plane);
  vec3<float> n = PlaneNormal(plane);

  // We project the center on the plane defined by (v0, v1, v2, v3)
  // We can write center = plane_center + a * e0 + b * e1 + c * n
  // We know that <e0, n> = 0 and <e1, n> = 0 and
  // <a, b> is the dot product between a and b.
  // This means we can solve for c as:
  // c = <center - plane_center - a * e0 - b * e1, n>
  //   = <center - plane_center, n>
  const float c = dot((center - plane_center), n);

  // If c is negative, then we revert the direction of n such that n
  // points "inside"
  if (c < 0.0f) {
    n = -1.0f * n;
  }

  return n;
}

// Calculate the volume of the box by summing the volume of
// each of the tetrahedrons formed with a triangle face and
// the box centroid.
//
// Args
//    box_tris: vector of vec3 coordinates of the vertices of each
//       of the triangles in the box
//    box_center: vec3 coordinates of the center of the box
//
// Returns
//    float: volume of the box
//
inline float BoxVolume(
    const face_verts& box_tris,
    const vec3<float>& box_center) {
  float box_vol = 0.0;
  // Iterate through each triange, calculate the area of the
  // tetrahedron formed with the box_center and sum them
  for (int t = 0; t < box_tris.size(); ++t) {
    // Subtract the center:
    const vec3<float> v0 = box_tris[t][0] - box_center;
    const vec3<float> v1 = box_tris[t][1] - box_center;
    const vec3<float> v2 = box_tris[t][2] - box_center;

    // Compute the area
    const float area = dot(v0, cross(v1, v2));
    const float vol = std::abs(area) / 6.0;
    box_vol = box_vol + vol;
  }
  return box_vol;
}

// Compute the box center as the mean of the verts
//
// Args
//    box_verts: (8, 3) tensor of the corner vertices of the box
//
// Returns
//    vec3: coordinates of the center of the box
//
inline vec3<float> BoxCenter(const at::Tensor& box_verts) {
  const auto& box_center_t = at::mean(box_verts, 0);
  const vec3<float> box_center(
      box_center_t[0].item<float>(),
      box_center_t[1].item<float>(),
      box_center_t[2].item<float>());
  return box_center;
}

// Find the point of intersection between a plane
// and a line given by the end points (p0, p1)
//
// Args
//    plane_ctr: vec3 coordinates of the center of the plane
//    normal: vec3 of the direction of the plane normal
//    p0, p1: vec3 of the start and end point of the line
//
// Returns
//    vec3: position of the intersection point
//
inline vec3<float> PlaneEdgeIntersection(
    const vec3<float>& plane_ctr,
    const vec3<float>& normal,
    const vec3<float>& p0,
    const vec3<float>& p1) {
  // The point of intersection can be parametrized
  // p = p0 + a (p1 - p0) where a in [0, 1]
  // We want to find a such that p is on plane
  // <p - ctr, n> = 0

  vec3<float> direc = p1 - p0;
  direc = direc / std::fmaxf(norm(direc), kEpsilon);

  vec3<float> p = (p1 + p0) / 2.0f;

  if (std::abs(dot(direc, normal)) >= dEpsilon) {
    const float top = -1.0f * dot(p0 - plane_ctr, normal);
    const float bot = dot(p1 - p0, normal);
    const float a = top / bot;
    p = p0 + a * (p1 - p0);
  }
  return p;
}

// The face of the box that each triangle in _TRIS belongs to.
const int _TRI_FACE[NUM_TRIS] = {0, 0, 5, 5, 4, 4, 3, 3, 1, 1, 2, 2};

// A convex polygon lying in a box triangle. Clipping a triangle by the six
// planes of a box adds at most one vertex per plane.
using face_poly = std::vector<vec3<float>>;

// A box face with the plane it spans.
struct FacePlane {
  // The center of the corners
  vec3<float> center;
  // The unit normal pointing inside the box
  vec3<float> normal;
  // The longer diagonal of the face
  float size;
  // The largest distance of a corner from the plane, i.e. how far the face
  // is from being planar
  float deviation;
};

// Args
//    plane: vec3 coordinates of the corners of the box face
//    center: vec3 coordinates of the center of the box
//
// Returns
//    FacePlane of the face
//
inline FacePlane GetFacePlane(
    const std::vector<vec3<float>>& plane,
    const vec3<float>& center) {
  FacePlane face{
      PlaneCenter(plane),
      PlaneNormalDirection(plane, center),
      std::fmaxf(norm(plane[2] - plane[0]), norm(plane[3] - plane[1])),
      0.0f};
  for (const auto& v : plane) {
    face.deviation =
        std::fmaxf(face.deviation, std::abs(dot(v - face.center, face.normal)));
  }
  return face;
}

// The mean signed distance of the corners of a face from a plane, along the
// plane normal pointing inside.
inline float MeanDistance(
    const std::vector<vec3<float>>& q,
    const FacePlane& plane) {
  float dist = 0.0f;
  for (const auto& v : q) {
    dist = dist + dot(v - plane.center, plane.normal);
  }
  return dist / 4.0f;
}

// Compute a boolean indicator for whether two box faces lie in the same
// plane with their boxes on the same side of it: the inside normals point the
// same way and the corners of each face are within the tolerance of the plane
// of the other. The tolerance is dEpsilon of the face size, plus how far both
// faces are from planar. Faces of boxes on opposite sides of a plane both
// bound the intersection, which is then a thin slab, and are clipped as usual.
//
// Args
//    q1, q2: vec3 coordinates of the corners of the two faces
//    face1, face2: FacePlane of the two faces
//
// Returns
//    bool: whether or not the two faces lie in the same plane
//
inline bool IsCoplanarFacePlanes(
    const std::vector<vec3<float>>& q1,
    const FacePlane& face1,
    const std::vector<vec3<float>>& q2,
    const FacePlane& face2) {
  if (dot(face1.normal, face2.normal) <= 0.0f) {
    return false;
  }
  const float tol = face1.deviation + face2.deviation +
      dEpsilon * std::fmaxf(face1.size, face2.size);
  for (const auto& v : q1) {
    if (std::abs(dot(v - face2.center, face2.normal)) > tol) {
      return false;
    }
  }
  for (const auto& v : q2) {
    if (std::abs(dot(v - face1.center, face1.normal)) > tol) {
      return false;
    }
  }
  return true;
}

// Clip a convex polygon so that it lies inside a plane.
//
// The polygon lies in a triangle, so the signed distance to the plane is
// affine over it and the vertices inside the plane form one contiguous run.
// The run is taken from the vertex furthest inside, which bounds the output
// to one more vertex than the input even when rounding misclassifies
// vertices close to the plane.
//
// Args
//    plane: FacePlane of the clipping plane
//    poly: the polygon to clip
//
// Returns
//    face_poly: the clipped polygon; empty if poly is entirely outside
//
inline face_poly ClipPolyByPlane(
    const FacePlane& plane,
    const face_poly& poly) {
  const int n = poly.size();
  if (n == 0) {
    return {};
  }

  // Signed distance of each vertex along the normal pointing inside
  std::vector<float> dist(n);
  int top = 0;
  for (int v = 0; v < n; ++v) {
    dist[v] = dot(poly[v] - plane.center, plane.normal);
    top = dist[v] > dist[top] ? v : top;
  }

  // All out
  if (dist[top] < 0.0f) {
    return {};
  }

  // The first vertex outside the plane after and before the top vertex
  int exit = -1;
  int entry = -1;
  for (int k = 1; k < n && exit < 0; ++k) {
    const int v = (top + k) % n;
    exit = dist[v] < 0.0f ? v : -1;
  }
  for (int k = 1; k < n && entry < 0; ++k) {
    const int v = (top + n - k) % n;
    entry = dist[v] < 0.0f ? v : -1;
  }

  // All in
  if (exit < 0) {
    return poly;
  }

  // Keep the run between entry and exit, adding the points where its first
  // and last edges cross the plane.
  const int first = (entry + 1) % n;
  const int last = (exit + n - 1) % n;
  face_poly out;
  out.push_back(PlaneEdgeIntersection(
      plane.center, plane.normal, poly[first], poly[entry]));
  for (int v = first; v != exit; v = (v + 1) % n) {
    out.push_back(poly[v]);
  }
  out.push_back(PlaneEdgeIntersection(
      plane.center, plane.normal, poly[last], poly[exit]));
  return out;
}

// Clip a triangle of one box by the planes of another box.
//
// Args
//    tri: vec3 coordinates of the vertices of the triangle
//    planes: FacePlane of each face of the other box
//    skip: bit p is set if the triangle is not clipped by plane p
//
// Returns
//    face_poly: the clipped polygon
//
inline face_poly ClipTriByBox(
    const std::vector<vec3<float>>& tri,
    const std::vector<FacePlane>& planes,
    const unsigned int skip) {
  face_poly poly(tri.begin(), tri.end());
  for (int p = 0; p < NUM_PLANES; ++p) {
    if (!((skip >> p) & 1u)) {
      poly = ClipPolyByPlane(planes[p], poly);
    }
  }
  return poly;
}

// The volume of the cone from center to a convex polygon, summed over the
// tetrahedrons of a fan triangulation as in BoxVolume.
inline float PolyVolume(const face_poly& poly, const vec3<float>& center) {
  float vol = 0.0;
  for (int v = 1; v + 1 < static_cast<int>(poly.size()); ++v) {
    const vec3<float> v0 = poly[0] - center;
    const vec3<float> v1 = poly[v] - center;
    const vec3<float> v2 = poly[v + 1] - center;
    vol = vol + std::abs(dot(v0, cross(v1, v2))) / 6.0;
  }
  return vol;
}
