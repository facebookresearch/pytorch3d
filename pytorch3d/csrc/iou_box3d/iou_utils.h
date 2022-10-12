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
// With dEpsilon = 0.001, two unit vectors are considered co-planar
// if their θ = 2.5 deg.
const auto dEpsilon = 1e-3;
// aEpsilon: Used once in main function to check for small face areas
const auto aEpsilon = 1e-4;
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
  // The center of the plane
  const vec3<float> plane_ctr = PlaneCenter(plane);

  // Every point p can be written as p = plane_ctr + a e0 + b e1 + c n
  // Solving for c:
  // c = (point - plane_ctr - a * e0 - b * e1).dot(n)
  // We know that <e0, n> = 0 and <e1, n> = 0
  // So the calculation can be simplified as:
  const float c = dot((point - plane_ctr), normal);
  const bool inside = c >= 0.0f;
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
  // The center of the plane
  const vec3<float> plane_ctr = PlaneCenter(plane);

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

// Compute the most distant points between two sets of vertices
//
// Args
//    verts1, verts2: vec3 defining the list of vertices
//
// Returns
//    v1m, v2m: vec3 vectors of the most distant points
//          in verts1 and verts2 respectively
//
inline std::tuple<vec3<float>, vec3<float>> ArgMaxVerts(
    const std::vector<vec3<float>>& verts1,
    const std::vector<vec3<float>>& verts2) {
  vec3<float> v1m = {0.0f, 0.0f, 0.0f};
  vec3<float> v2m = {0.0f, 0.0f, 0.0f};
  float maxdist = -1.0f;

  for (const auto& v1 : verts1) {
    for (const auto& v2 : verts2) {
      if (norm(v1 - v2) > maxdist) {
        v1m = v1;
        v2m = v2;
        maxdist = norm(v1 - v2);
      }
    }
  }
  return std::make_tuple(v1m, v2m);
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
inline bool IsCoplanarTriTri(
    const std::vector<vec3<float>>& tri1,
    const std::vector<vec3<float>>& tri2) {
  // Get normal for tri 1
  const vec3<float> n1 = TriNormal(tri1);

  // Get normal for tri 2
  const vec3<float> n2 = TriNormal(tri2);

  // Check if parallel
  const bool check1 = std::abs(dot(n1, n2)) > 1 - dEpsilon;

  // Compute most distant points
  auto argvs = ArgMaxVerts(tri1, tri2);
  const auto v1m = std::get<0>(argvs);
  const auto v2m = std::get<1>(argvs);

  vec3<float> n12m = v1m - v2m;
  n12m = n12m / std::fmaxf(norm(n12m), kEpsilon);

  const bool check2 = (std::abs(dot(n12m, n1)) < dEpsilon) ||
      (std::abs(dot(n12m, n2)) < dEpsilon);

  return (check1 && check2);
}

// Compute a boolean indicator for whether or not a triangular and a planar
// face are coplanar
//
// Args
//    tri, plane: std:vector<vec3> of the vertex coordinates of
//        triangular face and planar face
//    normal: the normal direction of the plane pointing "inside"
//
// Returns
//    bool: whether or not the two faces are coplanar
//
inline bool IsCoplanarTriPlane(
    const std::vector<vec3<float>>& tri,
    const std::vector<vec3<float>>& plane,
    const vec3<float>& normal) {
  // Get normal for tri
  const vec3<float> nt = TriNormal(tri);

  // check if parallel
  const bool check1 = std::abs(dot(nt, normal)) > 1 - dEpsilon;

  // Compute most distant points
  auto argvs = ArgMaxVerts(tri, plane);
  const auto v1m = std::get<0>(argvs);
  const auto v2m = std::get<1>(argvs);

  vec3<float> n12m = v1m - v2m;
  n12m = n12m / std::fmaxf(norm(n12m), kEpsilon);

  const bool check2 = std::abs(dot(n12m, normal)) < dEpsilon;

  return (check1 && check2);
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

  // Check coplanar
  const bool iscoplanar = IsCoplanarTriPlane(tri, plane, normal);
  if (iscoplanar) {
    // Return input vertices
    face_verts tris = {{v0, v1, v2}};
    return tris;
  }

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
