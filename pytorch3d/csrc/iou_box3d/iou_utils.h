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

const auto kEpsilon = 1e-5;

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

// The normal of the face defined by vertices (v0, v1, v2)
// Define e0 to be the edge connecting (v1, v0)
// Define e1 to be the edge connecting (v2, v0)
// normal is the cross product of e0, e1
//
// Args
//    v0, v1, v2: vec3 coordinates of the vertices of the face
//
// Returns
//    vec3: normal for the face
//
inline vec3<float> FaceNormal(vec3<float> v0, vec3<float> v1, vec3<float> v2) {
  vec3<float> n = cross(v1 - v0, v2 - v0);
  n = n / std::fmaxf(norm(n), kEpsilon);
  return n;
}

// The area of the face defined by vertices (v0, v1, v2)
// Define e0 to be the edge connecting (v1, v0)
// Define e1 to be the edge connecting (v2, v0)
// Area is the norm of the cross product of e0, e1 divided by 2.0
//
// Args
//    tri: vec3 coordinates of the vertices of the face
//
// Returns
//    float: area for the face
//
inline float FaceArea(const std::vector<vec3<float>>& tri) {
  // Get verts for face
  const vec3<float> v0 = tri[0];
  const vec3<float> v1 = tri[1];
  const vec3<float> v2 = tri[2];
  const vec3<float> n = cross(v1 - v0, v2 - v0);
  return norm(n) / 2.0;
}

// The normal of a box plane defined by the verts in `plane` with
// the centroid of the box given by `center`.
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
  // Only need the first 3 verts of the plane
  const vec3<float> v0 = plane[0];
  const vec3<float> v1 = plane[1];
  const vec3<float> v2 = plane[2];

  // We project the center on the plane defined by (v0, v1, v2)
  // We can write center = v0 + a * e0 + b * e1 + c * n
  // We know that <e0, n> = 0 and <e1, n> = 0 and
  // <a, b> is the dot product between a and b.
  // This means we can solve for c as:
  // c = <center - v0 - a * e0 - b * e1, n> = <center - v0, n>
  vec3<float> n = FaceNormal(v0, v1, v2);
  const float c = dot((center - v0), n);

  // If c is negative, then we revert the direction of n such that n
  // points "inside"
  if (c < kEpsilon) {
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

// Compute the polyhedron center as the mean of the face centers
// of the triangle faces
//
// Args
//    tris: vector of vec3 coordinates of the
//       vertices of each of the triangles in the polyhedron
//
// Returns
//    vec3: coordinates of the center of the polyhedron
//
inline vec3<float> PolyhedronCenter(const face_verts& tris) {
  float x = 0.0;
  float y = 0.0;
  float z = 0.0;
  const int num_tris = tris.size();

  // Find the center point of each face
  for (int t = 0; t < num_tris; ++t) {
    const vec3<float> v0 = tris[t][0];
    const vec3<float> v1 = tris[t][1];
    const vec3<float> v2 = tris[t][2];
    const float x_face = (v0.x + v1.x + v2.x) / 3.0;
    const float y_face = (v0.y + v1.y + v2.y) / 3.0;
    const float z_face = (v0.z + v1.z + v2.z) / 3.0;
    x = x + x_face;
    y = y + y_face;
    z = z + z_face;
  }

  // Take the mean of the centers of all faces
  x = x / num_tris;
  y = y / num_tris;
  z = z / num_tris;

  const vec3<float> center(x, y, z);
  return center;
}

// Compute a boolean indicator for whether a point
// is inside a plane, where inside refers to whether
// or not the point has a component in the
// normal direction of the plane.
//
// Args
//    plane: vector of vec3 coordinates of the
//       vertices of each of the triangles in the box
//    normal: vec3 of the direction of the plane normal
//    point: vec3 of the position of the point of interest
//
// Returns
//    bool: whether or not the point is inside the plane
//
inline bool IsInside(
    const std::vector<vec3<float>>& plane,
    const vec3<float>& normal,
    const vec3<float>& point) {
  // Get one vert of the plane
  const vec3<float> v0 = plane[0];

  // Every point p can be written as p = v0 + a e0 + b e1 + c n
  // Solving for c:
  // c = (point - v0 - a * e0 - b * e1).dot(n)
  // We know that <e0, n> = 0 and <e1, n> = 0
  // So the calculation can be simplified as:
  const float c = dot((point - v0), normal);
  const bool inside = c > -1.0f * kEpsilon;
  return inside;
}

// Find the point of intersection between a plane
// and a line given by the end points (p0, p1)
//
// Args
//    plane: vector of vec3 coordinates of the
//       vertices of each of the triangles in the box
//    normal: vec3 of the direction of the plane normal
//    p0, p1: vec3 of the start and end point of the line
//
// Returns
//    vec3: position of the intersection point
//
inline vec3<float> PlaneEdgeIntersection(
    const std::vector<vec3<float>>& plane,
    const vec3<float>& normal,
    const vec3<float>& p0,
    const vec3<float>& p1) {
  // Get one vert of the plane
  const vec3<float> v0 = plane[0];

  // The point of intersection can be parametrized
  // p = p0 + a (p1 - p0) where a in [0, 1]
  // We want to find a such that p is on plane
  // <p - v0, n> = 0
  const float top = dot(-1.0f * (p0 - v0), normal);
  const float bot = dot(p1 - p0, normal);
  const float a = top / bot;
  const vec3<float> p = p0 + a * (p1 - p0);
  return p;
}

// Triangle is clipped into a quadrilateral
// based on the intersection points with the plane.
// Then the quadrilateral is divided into two triangles.
//
// Args
//    plane: vector of vec3 coordinates of the
//        vertices of each of the triangles in the box
//    normal: vec3 of the direction of the plane normal
//    vout: vec3 of the point in the triangle which is outside
//       the plane
//    vin1, vin2: vec3 of the points in the triangle which are
//        inside the plane
//
// Returns
//    std::vector<std::vector<vec3>>: vector of vertex coordinates
//      of the new triangle faces
//
inline face_verts ClipTriByPlaneOneOut(
    const std::vector<vec3<float>>& plane,
    const vec3<float>& normal,
    const vec3<float>& vout,
    const vec3<float>& vin1,
    const vec3<float>& vin2) {
  // point of intersection between plane and (vin1, vout)
  const vec3<float> pint1 = PlaneEdgeIntersection(plane, normal, vin1, vout);
  // point of intersection between plane and (vin2, vout)
  const vec3<float> pint2 = PlaneEdgeIntersection(plane, normal, vin2, vout);
  const face_verts face_verts = {{vin1, pint1, pint2}, {vin1, pint2, vin2}};
  return face_verts;
}

// Triangle is clipped into a smaller triangle based
// on the intersection points with the plane.
//
// Args
//    plane: vector of vec3 coordinates of the
//       vertices of each of the triangles in the box
//    normal: vec3 of the direction of the plane normal
//    vout1, vout2: vec3 of the points in the triangle which are
//       outside the plane
//    vin: vec3 of the point in the triangle which is inside
//        the plane
// Returns
//    std::vector<std::vector<vec3>>: vector of vertex coordinates
//      of the new triangle face
//
inline face_verts ClipTriByPlaneTwoOut(
    const std::vector<vec3<float>>& plane,
    const vec3<float>& normal,
    const vec3<float>& vout1,
    const vec3<float>& vout2,
    const vec3<float>& vin) {
  // point of intersection between plane and (vin, vout1)
  const vec3<float> pint1 = PlaneEdgeIntersection(plane, normal, vin, vout1);
  // point of intersection between plane and (vin, vout2)
  const vec3<float> pint2 = PlaneEdgeIntersection(plane, normal, vin, vout2);
  const face_verts face_verts = {{vin, pint1, pint2}};
  return face_verts;
}

// Clip the triangle faces so that they lie within the
// plane, creating new triangle faces where necessary.
//
// Args
//    plane: vector of vec3 coordinates of the
//       vertices of each of the triangles in the box
//    tri: std:vector<vec3> of the vertex coordinates of the
//       triangle faces
//    normal: vec3 of the direction of the plane normal
//
// Returns
//    std::vector<std::vector<vec3>>: vector of vertex coordinates
//      of the new triangle faces formed after clipping.
//      All triangles are now "inside" the plane.
//
inline face_verts ClipTriByPlane(
    const std::vector<vec3<float>>& plane,
    const std::vector<vec3<float>>& tri,
    const vec3<float>& normal) {
  // Get Triangle vertices
  const vec3<float> v0 = tri[0];
  const vec3<float> v1 = tri[1];
  const vec3<float> v2 = tri[2];

  // Check each of the triangle vertices to see if it is inside the plane
  const bool isin0 = IsInside(plane, normal, v0);
  const bool isin1 = IsInside(plane, normal, v1);
  const bool isin2 = IsInside(plane, normal, v2);

  // All in
  if (isin0 && isin1 && isin2) {
    // Return input vertices
    face_verts tris = {{v0, v1, v2}};
    return tris;
  }

  face_verts empty_tris = {};
  // All out
  if (!isin0 && !isin1 && !isin2) {
    return empty_tris;
  }

  // One vert out
  if (isin0 && isin1 && !isin2) {
    return ClipTriByPlaneOneOut(plane, normal, v2, v0, v1);
  }
  if (isin0 && !isin1 && isin2) {
    return ClipTriByPlaneOneOut(plane, normal, v1, v0, v2);
  }
  if (!isin0 && isin1 && isin2) {
    return ClipTriByPlaneOneOut(plane, normal, v0, v1, v2);
  }

  // Two verts out
  if (isin0 && !isin1 && !isin2) {
    return ClipTriByPlaneTwoOut(plane, normal, v1, v2, v0);
  }
  if (!isin0 && !isin1 && isin2) {
    return ClipTriByPlaneTwoOut(plane, normal, v0, v1, v2);
  }
  if (!isin0 && isin1 && !isin2) {
    return ClipTriByPlaneTwoOut(plane, normal, v0, v2, v1);
  }

  // Else return empty (should not be reached)
  return empty_tris;
}

// Compute a boolean indicator for whether or not two faces
// are coplanar
//
// Args
//    tri1, tri2: std:vector<vec3> of the vertex coordinates of
//        triangle faces
//
// Returns
//    bool: whether or not the two faces are coplanar
//
inline bool IsCoplanarFace(
    const std::vector<vec3<float>>& tri1,
    const std::vector<vec3<float>>& tri2) {
  // Get verts for face 1
  const vec3<float> v0 = tri1[0];
  const vec3<float> v1 = tri1[1];
  const vec3<float> v2 = tri1[2];

  const vec3<float> n1 = FaceNormal(v0, v1, v2);
  int coplanar_count = 0;
  for (int i = 0; i < 3; ++i) {
    float d = std::abs(dot(tri2[i] - v0, n1));
    if (d < kEpsilon) {
      coplanar_count = coplanar_count + 1;
    }
  }
  return (coplanar_count == 3);
}

// Get the triangles from each box which are part of the
// intersecting polyhedron by computing the intersection
// points with each of the planes.
//
// Args
//    tris: vertex coordinates of all the triangle faces
//       in the box
//    planes: vertex coordinates of all the planes in the box
//    center: vec3 coordinates of the center of the box from which
//        the planes originate
//
// Returns
//    std::vector<std::vector<vec3>>> vector of vertex coordinates
//      of the new triangle faces formed after clipping.
//      All triangles are now "inside" the planes.
//
inline face_verts BoxIntersections(
    const face_verts& tris,
    const face_verts& planes,
    const vec3<float>& center) {
  // Create a new vector to avoid modifying in place
  face_verts out_tris = tris;
  for (int p = 0; p < NUM_PLANES; ++p) {
    // Get plane normal direction
    const vec3<float> n2 = PlaneNormalDirection(planes[p], center);
    // Iterate through triangles in tris
    // Create intermediate vector to store the updated tris
    face_verts tri_verts_updated;
    for (int t = 0; t < out_tris.size(); ++t) {
      // Clip tri by plane
      const face_verts tri_updated = ClipTriByPlane(planes[p], out_tris[t], n2);
      // Add to the tri_verts_updated output if not empty
      for (int v = 0; v < tri_updated.size(); ++v) {
        tri_verts_updated.push_back(tri_updated[v]);
      }
    }
    // Update the tris
    out_tris = tri_verts_updated;
  }
  return out_tris;
}
