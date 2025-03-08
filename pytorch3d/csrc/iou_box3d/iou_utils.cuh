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
#include "utils/float_math.cuh"

// dEpsilon: Used in dot products and is used to assess whether two unit vectors
// are orthogonal (or coplanar). It's an epsilon on cos(θ).
// With dEpsilon = 0.001, two unit vectors are considered co-planar
// if their θ = 2.5 deg.
__constant__ const float dEpsilon = 1e-3;
// aEpsilon: Used once in main function to check for small face areas
__constant__ const float aEpsilon = 1e-4;
// kEpsilon: Used only for norm(u) = u/max(||u||, kEpsilon)
__constant__ const float kEpsilon = 1e-8;

/*
_PLANES and _TRIS define the 4- and 3-connectivity
of the 8 box corners.
_PLANES gives the quad faces of the 3D box
_TRIS gives the triangle faces of the 3D box
*/
const int NUM_PLANES = 6;
const int NUM_TRIS = 12;
// This is required for iniitalizing the faces
// in the intersecting shape
const int MAX_TRIS = 100;

// Create data types for representing the
// verts for each face and the indices.
// We will use struct arrays for representing
// the data for each box and intersecting
// triangles
struct FaceVerts {
  float3 v0;
  float3 v1;
  float3 v2;
  float3 v3; // Can be empty for triangles
};

struct FaceVertsIdx {
  int v0;
  int v1;
  int v2;
  int v3; // Can be empty for triangles
};

// This is used when deciding which faces to
// keep that are not coplanar
struct Keep {
  bool keep;
};

__device__ FaceVertsIdx _PLANES[] = {
    {0, 1, 2, 3},
    {3, 2, 6, 7},
    {0, 1, 5, 4},
    {0, 3, 7, 4},
    {1, 5, 6, 2},
    {4, 5, 6, 7},
};
__device__ FaceVertsIdx _TRIS[] = {
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

// Args
//    box: (8, 3) tensor accessor for the box vertices
//    box_tris: Array of structs of type FaceVerts,
//      effectively (F, 3, 3) where the coordinates of the
//      verts for each face will be saved to.
//
// Returns: None (output saved to box_tris)
//
template <typename Box, typename BoxTris>
__device__ inline void GetBoxTris(const Box& box, BoxTris& box_tris) {
  for (int t = 0; t < NUM_TRIS; ++t) {
    const float3 v0 = make_float3(
        box[_TRIS[t].v0][0], box[_TRIS[t].v0][1], box[_TRIS[t].v0][2]);
    const float3 v1 = make_float3(
        box[_TRIS[t].v1][0], box[_TRIS[t].v1][1], box[_TRIS[t].v1][2]);
    const float3 v2 = make_float3(
        box[_TRIS[t].v2][0], box[_TRIS[t].v2][1], box[_TRIS[t].v2][2]);
    box_tris[t] = {v0, v1, v2};
  }
}

// Args
//    box: (8, 3) tensor accessor for the box vertices
//    box_planes: Array of structs of type FaceVerts, effectively (P, 4, 3)
//      where the coordinates of the verts for the four corners of each plane
//      will be saved to
//
// Returns: None (output saved to box_planes)
//
template <typename Box, typename FaceVertsBoxPlanes>
__device__ inline void GetBoxPlanes(
    const Box& box,
    FaceVertsBoxPlanes& box_planes) {
  for (int t = 0; t < NUM_PLANES; ++t) {
    const float3 v0 = make_float3(
        box[_PLANES[t].v0][0], box[_PLANES[t].v0][1], box[_PLANES[t].v0][2]);
    const float3 v1 = make_float3(
        box[_PLANES[t].v1][0], box[_PLANES[t].v1][1], box[_PLANES[t].v1][2]);
    const float3 v2 = make_float3(
        box[_PLANES[t].v2][0], box[_PLANES[t].v2][1], box[_PLANES[t].v2][2]);
    const float3 v3 = make_float3(
        box[_PLANES[t].v3][0], box[_PLANES[t].v3][1], box[_PLANES[t].v3][2]);
    box_planes[t] = {v0, v1, v2, v3};
  }
}

// The geometric center of a list of vertices.
//
// Args
//    vertices: A list of float3 vertices {v0, ..., vN}.
//
// Returns
//    float3: Geometric center of the vertices.
//
__device__ inline float3 FaceCenter(
    std::initializer_list<const float3> vertices) {
  auto sumVertices = float3{};
  for (const auto& vertex : vertices) {
    sumVertices = sumVertices + vertex;
  }
  return sumVertices / vertices.size();
}

// The normal of a plane spanned by vectors e0 and e1
//
// Args
//    e0, e1: float3 vectors defining a plane
//
// Returns
//    float3: normal of the plane
//
__device__ inline float3 GetNormal(const float3 e0, const float3 e1) {
  float3 n = cross(e0, e1);
  n = n / std::fmaxf(norm(n), kEpsilon);
  return n;
}

// The normal of a face with vertices (v0, v1, v2) or (v0, ..., v3).
// We find the "best" edges connecting the face center to the vertices,
// such that the cross product between the edges is maximized.
//
// Args
//    vertices: a list of float3 coordinates of the vertices.
//
// Returns
//    float3: center of the plane
//
__device__ inline float3 FaceNormal(
    std::initializer_list<const float3> vertices) {
  const auto faceCenter = FaceCenter(vertices);
  auto normal = float3();
  auto maxDist = -1;
  for (auto v1 = vertices.begin(); v1 != vertices.end() - 1; ++v1) {
    for (auto v2 = v1 + 1; v2 != vertices.end(); ++v2) {
      const auto v1ToCenter = *v1 - faceCenter;
      const auto v2ToCenter = *v2 - faceCenter;
      const auto dist = norm(cross(v1ToCenter, v2ToCenter));
      if (dist > maxDist) {
        normal = GetNormal(v1ToCenter, v2ToCenter);
        maxDist = dist;
      }
    }
  }
  return normal;
}

// The area of the face defined by vertices (v0, v1, v2)
// Define e0 to be the edge connecting (v1, v0)
// Define e1 to be the edge connecting (v2, v0)
// Area is the norm of the cross product of e0, e1 divided by 2.0
//
// Args
//    tri: FaceVerts of float3 coordinates of the vertices of the face
//
// Returns
//    float: area for the face
//
__device__ inline float FaceArea(const FaceVerts& tri) {
  // Get verts for face 1
  const float3 n = cross(tri.v1 - tri.v0, tri.v2 - tri.v0);
  return norm(n) / 2.0;
}

// The normal of a box plane defined by the verts in `plane` such that it
// points toward the centroid of the box given by `center`.
//
// Args
//    plane: float3 coordinates of the vertices of the plane
//    center: float3 coordinates of the center of the box from
//        which the plane originated
//
// Returns
//    float3: normal for the plane such that it points towards
//      the center of the box
//
template <typename FaceVertsPlane>
__device__ inline float3 PlaneNormalDirection(
    const FaceVertsPlane& plane,
    const float3& center) {
  // The plane's center
  const float3 plane_center =
      FaceCenter({plane.v0, plane.v1, plane.v2, plane.v3});

  // The plane's normal
  float3 n = FaceNormal({plane.v0, plane.v1, plane.v2, plane.v3});

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
//    box_tris: vector of float3 coordinates of the vertices of each
//       of the triangles in the box
//    box_center: float3 coordinates of the center of the box
//
// Returns
//    float: volume of the box
//
template <typename BoxTris>
__device__ inline float BoxVolume(
    const BoxTris& box_tris,
    const float3& box_center,
    const int num_tris) {
  float box_vol = 0.0;
  // Iterate through each triange, calculate the area of the
  // tetrahedron formed with the box_center and sum them
  for (int t = 0; t < num_tris; ++t) {
    // Subtract the center:
    float3 v0 = box_tris[t].v0;
    float3 v1 = box_tris[t].v1;
    float3 v2 = box_tris[t].v2;

    v0 = v0 - box_center;
    v1 = v1 - box_center;
    v2 = v2 - box_center;

    // Compute the area
    const float area = dot(v0, cross(v1, v2));
    const float vol = abs(area) / 6.0;
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
//    float3: coordinates of the center of the box
//
template <typename Box>
__device__ inline float3 BoxCenter(const Box box_verts) {
  float x = 0.0;
  float y = 0.0;
  float z = 0.0;
  const int num_verts = box_verts.size(0); // Should be 8
  // Sum all x, y, z, and take the mean
  for (int t = 0; t < num_verts; ++t) {
    x = x + box_verts[t][0];
    y = y + box_verts[t][1];
    z = z + box_verts[t][2];
  }
  // Take the mean of all the vertex positions
  x = x / num_verts;
  y = y / num_verts;
  z = z / num_verts;
  const float3 center = make_float3(x, y, z);
  return center;
}

// Compute the polyhedron center as the mean of the face centers
// of the triangle faces
//
// Args
//    tris: vector of float3 coordinates of the
//       vertices of each of the triangles in the polyhedron
//
// Returns
//    float3: coordinates of the center of the polyhedron
//
template <typename Tris>
__device__ inline float3 PolyhedronCenter(
    const Tris& tris,
    const int num_tris) {
  float x = 0.0;
  float y = 0.0;
  float z = 0.0;

  // Find the center point of each face
  for (int t = 0; t < num_tris; ++t) {
    const float3 v0 = tris[t].v0;
    const float3 v1 = tris[t].v1;
    const float3 v2 = tris[t].v2;
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

  const float3 center = make_float3(x, y, z);
  return center;
}

// Compute a boolean indicator for whether a point
// is inside a plane, where inside refers to whether
// or not the point has a component in the
// normal direction of the plane.
//
// Args
//    plane: vector of float3 coordinates of the
//       vertices of each of the triangles in the box
//    normal: float3 of the direction of the plane normal
//    point: float3 of the position of the point of interest
//
// Returns
//    bool: whether or not the point is inside the plane
//
__device__ inline bool
IsInside(const FaceVerts& plane, const float3& normal, const float3& point) {
  // The center of the plane
  const float3 plane_ctr = FaceCenter({plane.v0, plane.v1, plane.v2, plane.v3});

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
//    plane: vector of float3 coordinates of the
//       vertices of each of the triangles in the box
//    normal: float3 of the direction of the plane normal
//    p0, p1: float3 of the start and end point of the line
//
// Returns
//    float3: position of the intersection point
//
__device__ inline float3 PlaneEdgeIntersection(
    const FaceVerts& plane,
    const float3& normal,
    const float3& p0,
    const float3& p1) {
  // The center of the plane
  const float3 plane_ctr = FaceCenter({plane.v0, plane.v1, plane.v2, plane.v3});

  // The point of intersection can be parametrized
  // p = p0 + a (p1 - p0) where a in [0, 1]
  // We want to find a such that p is on plane
  // <p - plane_ctr, n> = 0

  float3 direc = p1 - p0;
  direc = direc / fmaxf(norm(direc), kEpsilon);

  float3 p = (p1 + p0) / 2.0f;

  if (abs(dot(direc, normal)) >= dEpsilon) {
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
//    verts1, verts2: list of float3 defining the list of vertices
//
// Returns
//    v1m, v2m: float3 vectors of the most distant points
//          in verts1 and verts2 respectively
//
__device__ inline std::tuple<float3, float3> ArgMaxVerts(
    std::initializer_list<float3> verts1,
    std::initializer_list<float3> verts2) {
  auto v1m = float3();
  auto v2m = float3();
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
//    tri1, tri2: FaceVerts struct of the vertex coordinates of
//       the triangle face
//
// Returns
//    bool: whether or not the two faces are coplanar
//
__device__ inline bool IsCoplanarTriTri(
    const FaceVerts& tri1,
    const FaceVerts& tri2) {
  const float3 tri1_n = FaceNormal({tri1.v0, tri1.v1, tri1.v2});

  const float3 tri2_n = FaceNormal({tri2.v0, tri2.v1, tri2.v2});

  // Check if parallel
  const bool check1 = abs(dot(tri1_n, tri2_n)) > 1 - dEpsilon;

  // Compute most distant points
  const auto v1mAndv2m =
      ArgMaxVerts({tri1.v0, tri1.v1, tri1.v2}, {tri2.v0, tri2.v1, tri2.v2});
  const auto v1m = std::get<0>(v1mAndv2m);
  const auto v2m = std::get<1>(v1mAndv2m);

  float3 n12m = v1m - v2m;
  n12m = n12m / fmaxf(norm(n12m), kEpsilon);

  const bool check2 = (abs(dot(n12m, tri1_n)) < dEpsilon) ||
      (abs(dot(n12m, tri2_n)) < dEpsilon);

  return (check1 && check2);
}

// Compute a boolean indicator for whether or not a triangular and a planar
// face are coplanar
//
// Args
//    tri, plane: FaceVerts struct of the vertex coordinates of
//       the triangle and planar face
//  normal: the normal direction of the plane pointing "inside"
//
// Returns
//    bool: whether or not the two faces are coplanar
//
__device__ inline bool IsCoplanarTriPlane(
    const FaceVerts& tri,
    const FaceVerts& plane,
    const float3& normal) {
  const float3 nt = FaceNormal({tri.v0, tri.v1, tri.v2});

  // check if parallel
  const bool check1 = abs(dot(nt, normal)) > 1 - dEpsilon;

  // Compute most distant points
  const auto v1mAndv2m = ArgMaxVerts(
      {tri.v0, tri.v1, tri.v2}, {plane.v0, plane.v1, plane.v2, plane.v3});
  const auto v1m = std::get<0>(v1mAndv2m);
  const auto v2m = std::get<1>(v1mAndv2m);

  float3 n12m = v1m - v2m;
  n12m = n12m / fmaxf(norm(n12m), kEpsilon);

  const bool check2 = abs(dot(n12m, normal)) < dEpsilon;

  return (check1 && check2);
}

// Triangle is clipped into a quadrilateral
// based on the intersection points with the plane.
// Then the quadrilateral is divided into two triangles.
//
// Args
//    plane: vector of float3 coordinates of the
//        vertices of each of the triangles in the box
//    normal: float3 of the direction of the plane normal
//    vout: float3 of the point in the triangle which is outside
//       the plane
//    vin1, vin2: float3 of the points in the triangle which are
//        inside the plane
//    face_verts_out: Array of structs of type FaceVerts,
//       with the coordinates of the new triangle faces
//       formed after clipping.
//       All triangles are now "inside" the plane.
//
// Returns:
//    count: (int) number of new faces after clipping the triangle
//      i.e. the valid faces which have been saved
//      to face_verts_out
//
template <typename FaceVertsBox>
__device__ inline int ClipTriByPlaneOneOut(
    const FaceVerts& plane,
    const float3& normal,
    const float3& vout,
    const float3& vin1,
    const float3& vin2,
    FaceVertsBox& face_verts_out) {
  // point of intersection between plane and (vin1, vout)
  const float3 pint1 = PlaneEdgeIntersection(plane, normal, vin1, vout);
  // point of intersection between plane and (vin2, vout)
  const float3 pint2 = PlaneEdgeIntersection(plane, normal, vin2, vout);

  face_verts_out[0] = {vin1, pint1, pint2};
  face_verts_out[1] = {vin1, pint2, vin2};

  return 2;
}

// Triangle is clipped into a smaller triangle based
// on the intersection points with the plane.
//
// Args
//    plane: vector of float3 coordinates of the
//       vertices of each of the triangles in the box
//    normal: float3 of the direction of the plane normal
//    vout1, vout2: float3 of the points in the triangle which are
//       outside the plane
//    vin: float3 of the point in the triangle which is inside
//        the plane
//    face_verts_out: Array of structs of type FaceVerts,
//       with the coordinates of the new triangle faces
//       formed after clipping.
//       All triangles are now "inside" the plane.
//
// Returns:
//    count: (int) number of new faces after clipping the triangle
//      i.e. the valid faces which have been saved
//      to face_verts_out
//
template <typename FaceVertsBox>
__device__ inline int ClipTriByPlaneTwoOut(
    const FaceVerts& plane,
    const float3& normal,
    const float3& vout1,
    const float3& vout2,
    const float3& vin,
    FaceVertsBox& face_verts_out) {
  // point of intersection between plane and (vin, vout1)
  const float3 pint1 = PlaneEdgeIntersection(plane, normal, vin, vout1);
  // point of intersection between plane and (vin, vout2)
  const float3 pint2 = PlaneEdgeIntersection(plane, normal, vin, vout2);

  face_verts_out[0] = {vin, pint1, pint2};

  return 1;
}

// Clip the triangle faces so that they lie within the
// plane, creating new triangle faces where necessary.
//
// Args
//    plane: Array of structs of type FaceVerts with the coordinates
//       of the vertices of each of the triangles in the box
//    tri: Array of structs of type FaceVerts with the vertex
//       coordinates of the triangle faces
//    normal: float3 of the direction of the plane normal
//    face_verts_out: Array of structs of type FaceVerts,
//       with the coordinates of the new triangle faces
//       formed after clipping.
//       All triangles are now "inside" the plane.
//
// Returns:
//    count: (int) number of new faces after clipping the triangle
//      i.e. the valid faces which have been saved
//      to face_verts_out
//
template <typename FaceVertsBox>
__device__ inline int ClipTriByPlane(
    const FaceVerts& plane,
    const FaceVerts& tri,
    const float3& normal,
    FaceVertsBox& face_verts_out) {
  // Get Triangle vertices
  const float3 v0 = tri.v0;
  const float3 v1 = tri.v1;
  const float3 v2 = tri.v2;

  // Check each of the triangle vertices to see if it is inside the plane
  const bool isin0 = IsInside(plane, normal, v0);
  const bool isin1 = IsInside(plane, normal, v1);
  const bool isin2 = IsInside(plane, normal, v2);

  // Check coplanar
  const bool iscoplanar = IsCoplanarTriPlane(tri, plane, normal);
  if (iscoplanar) {
    // Return input vertices
    face_verts_out[0] = {v0, v1, v2};
    return 1;
  }

  // All in
  if (isin0 && isin1 && isin2) {
    // Return input vertices
    face_verts_out[0] = {v0, v1, v2};
    return 1;
  }

  // All out
  if (!isin0 && !isin1 && !isin2) {
    return 0;
  }

  // One vert out
  if (isin0 && isin1 && !isin2) {
    return ClipTriByPlaneOneOut(plane, normal, v2, v0, v1, face_verts_out);
  }
  if (isin0 && !isin1 && isin2) {
    return ClipTriByPlaneOneOut(plane, normal, v1, v0, v2, face_verts_out);
  }
  if (!isin0 && isin1 && isin2) {
    return ClipTriByPlaneOneOut(plane, normal, v0, v1, v2, face_verts_out);
  }

  // Two verts out
  if (isin0 && !isin1 && !isin2) {
    return ClipTriByPlaneTwoOut(plane, normal, v1, v2, v0, face_verts_out);
  }
  if (!isin0 && !isin1 && isin2) {
    return ClipTriByPlaneTwoOut(plane, normal, v0, v1, v2, face_verts_out);
  }
  if (!isin0 && isin1 && !isin2) {
    return ClipTriByPlaneTwoOut(plane, normal, v0, v2, v1, face_verts_out);
  }

  // Else return empty (should not be reached)
  return 0;
}

// Get the triangles from each box which are part of the
// intersecting polyhedron by computing the intersection
// points with each of the planes.
//
// Args
//    planes: Array of structs of type FaceVerts with the coordinates
//       of the vertices of each of the triangles in the box
//    center: float3 coordinates of the center of the box from which
//        the planes originate
//    face_verts_out: Array of structs of type FaceVerts,
//       where the coordinates of the new triangle faces
//       formed after clipping will be saved to.
//       All triangles are now "inside" the plane.
//
// Returns:
//    count: (int) number of faces in the intersecting shape
//      i.e. the valid faces which have been saved
//      to face_verts_out
//
template <typename FaceVertsPlane, typename FaceVertsBox>
__device__ inline int BoxIntersections(
    const FaceVertsPlane& planes,
    const float3& center,
    FaceVertsBox& face_verts_out) {
  // Initialize num tris to 12
  int num_tris = NUM_TRIS;
  for (int p = 0; p < NUM_PLANES; ++p) {
    // Get plane normal direction
    const float3 n2 = PlaneNormalDirection(planes[p], center);
    // Create intermediate vector to store the updated tris
    FaceVerts tri_verts_updated[MAX_TRIS];
    int offset = 0;

    // Iterate through triangles in face_verts_out
    // for the valid tris given by num_tris
    for (int t = 0; t < num_tris; ++t) {
      // Clip tri by plane, can max be split into 2 triangles
      FaceVerts tri_updated[2];
      const int count =
          ClipTriByPlane(planes[p], face_verts_out[t], n2, tri_updated);
      // Add to the tri_verts_updated output if not empty
      for (int v = 0; v < count; ++v) {
        tri_verts_updated[offset] = tri_updated[v];
        offset++;
      }
    }
    // Update the face_verts_out tris
    num_tris = min(MAX_TRIS, offset);
    for (int j = 0; j < num_tris; ++j) {
      face_verts_out[j] = tri_verts_updated[j];
    }
  }
  return num_tris;
}
