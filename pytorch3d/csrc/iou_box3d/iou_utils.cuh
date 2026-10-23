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
// It is also the distance, relative to the face size, within which two box
// faces are considered to lie in the same plane.
__constant__ const float dEpsilon = 1e-3;
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

// The corners of a box, from which its triangles and faces are formed as
// needed instead of being stored.
struct BoxCorners {
  float3 v[8];

  // The triangle t of the box, as given by _TRIS
  __device__ FaceVerts Tri(const int t) const {
    return {v[_TRIS[t].v0], v[_TRIS[t].v1], v[_TRIS[t].v2]};
  }

  // The face p of the box, as given by _PLANES
  __device__ FaceVerts Face(const int p) const {
    return {
        v[_PLANES[p].v0], v[_PLANES[p].v1], v[_PLANES[p].v2], v[_PLANES[p].v3]};
  }
};

// The triangles of a box, indexable like an array of FaceVerts.
struct BoxTris {
  const BoxCorners& corners;

  __device__ FaceVerts operator[](const int t) const {
    return corners.Tri(t);
  }
};

// Args
//    box: (8, 3) tensor accessor for the box vertices
//    corners: BoxCorners where the vertices will be saved to
//
// Returns: None (output saved to corners)
//
template <typename Box>
__device__ inline void GetBoxCorners(const Box& box, BoxCorners& corners) {
  for (int i = 0; i < 8; ++i) {
    corners.v[i] = make_float3(box[i][0], box[i][1], box[i][2]);
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

// Find the point of intersection between a plane
// and a line given by the end points (p0, p1)
//
// Args
//    plane_ctr: float3 coordinates of the center of the plane
//    normal: float3 of the direction of the plane normal
//    p0, p1: float3 of the start and end point of the line
//
// Returns
//    float3: position of the intersection point
//
__device__ inline float3 PlaneEdgeIntersection(
    const float3& plane_ctr,
    const float3& normal,
    const float3& p0,
    const float3& p1) {
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

// The face of the box that each triangle in _TRIS belongs to.
__device__ const int _TRI_FACE[NUM_TRIS] = {0, 0, 5, 5, 4, 4, 3, 3, 1, 1, 2, 2};

// Clipping a triangle by the six planes of a box adds at most one vertex per
// plane.
const int MAX_POLY_VERTS = 3 + NUM_PLANES;

// A convex polygon lying in a box triangle.
struct FacePoly {
  float3 verts[MAX_POLY_VERTS];
  int num_verts;
};

// A box face with the plane it spans.
struct FacePlane {
  // The center of the corners
  float3 center;
  // The unit normal pointing inside the box
  float3 normal;
  // The longer diagonal of the face
  float size;
  // The largest distance of a corner from the plane, i.e. how far the face
  // is from being planar
  float deviation;
};

// Args
//    corners: BoxCorners of the box
//    center: float3 coordinates of the center of the box
//    faces: Array of FacePlane where the faces will be saved to
//
template <typename FacePlanes>
__device__ inline void GetFacePlanes(
    const BoxCorners& corners,
    const float3& center,
    FacePlanes& faces) {
  for (int p = 0; p < NUM_PLANES; ++p) {
    const FaceVerts q = corners.Face(p);
    FacePlane& face = faces[p];
    face.center = FaceCenter({q.v0, q.v1, q.v2, q.v3});
    face.normal = PlaneNormalDirection(q, center);
    face.size = fmaxf(norm(q.v2 - q.v0), norm(q.v3 - q.v1));
    face.deviation = 0.0f;
    for (const float3& v : {q.v0, q.v1, q.v2, q.v3}) {
      face.deviation =
          fmaxf(face.deviation, abs(dot(v - face.center, face.normal)));
    }
  }
}

// The mean signed distance of the corners of a face from a plane, along the
// plane normal pointing inside.
__device__ inline float MeanDistance(
    const FaceVerts& q,
    const FacePlane& plane) {
  float dist = 0.0f;
  for (const float3& v : {q.v0, q.v1, q.v2, q.v3}) {
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
//    q1, q2: FaceVerts struct of the corners of the two faces
//    face1, face2: FacePlane of the two faces
//
// Returns
//    bool: whether or not the two faces lie in the same plane
//
__device__ inline bool IsCoplanarFacePlanes(
    const FaceVerts& q1,
    const FacePlane& face1,
    const FaceVerts& q2,
    const FacePlane& face2) {
  if (dot(face1.normal, face2.normal) <= 0.0f) {
    return false;
  }
  const float tol = face1.deviation + face2.deviation +
      dEpsilon * fmaxf(face1.size, face2.size);
  for (const float3& v : {q1.v0, q1.v1, q1.v2, q1.v3}) {
    if (abs(dot(v - face2.center, face2.normal)) > tol) {
      return false;
    }
  }
  for (const float3& v : {q2.v0, q2.v1, q2.v2, q2.v3}) {
    if (abs(dot(v - face1.center, face1.normal)) > tol) {
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
//    poly_in: the polygon to clip
//    poly_out: the clipped polygon; empty if poly_in is entirely outside
//
__device__ inline void ClipPolyByPlane(
    const FacePlane& plane,
    const FacePoly& poly_in,
    FacePoly& poly_out) {
  const int n = poly_in.num_verts;
  poly_out.num_verts = 0;
  if (n == 0) {
    return;
  }

  // Signed distance of each vertex along the normal pointing inside
  float dist[MAX_POLY_VERTS];
  int top = 0;
  for (int v = 0; v < n; ++v) {
    dist[v] = dot(poly_in.verts[v] - plane.center, plane.normal);
    top = dist[v] > dist[top] ? v : top;
  }

  // All out
  if (dist[top] < 0.0f) {
    return;
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
    poly_out = poly_in;
    return;
  }

  // Keep the run between entry and exit, adding the points where its first
  // and last edges cross the plane.
  int count = 0;
  const int first = (entry + 1) % n;
  const int last = (exit + n - 1) % n;
  poly_out.verts[count++] = PlaneEdgeIntersection(
      plane.center, plane.normal, poly_in.verts[first], poly_in.verts[entry]);
  for (int v = first; v != exit; v = (v + 1) % n) {
    poly_out.verts[count++] = poly_in.verts[v];
  }
  poly_out.verts[count++] = PlaneEdgeIntersection(
      plane.center, plane.normal, poly_in.verts[last], poly_in.verts[exit]);
  poly_out.num_verts = count;
}

// Clip a triangle of one box by the planes of another box.
//
// Args
//    tri: FaceVerts struct of the triangle
//    planes: Array of FacePlane of the other box
//    skip: bit p is set if the triangle is not clipped by plane p
//    poly: the clipped polygon
//
template <typename FacePlanes>
__device__ inline void ClipTriByBox(
    const FaceVerts& tri,
    const FacePlanes& planes,
    const unsigned int skip,
    FacePoly& poly) {
  FacePoly scratch;
  FacePoly* poly_in = &poly;
  FacePoly* poly_out = &scratch;
  poly.verts[0] = tri.v0;
  poly.verts[1] = tri.v1;
  poly.verts[2] = tri.v2;
  poly.num_verts = 3;
  for (int p = 0; p < NUM_PLANES; ++p) {
    if ((skip >> p) & 1u) {
      continue;
    }
    ClipPolyByPlane(planes[p], *poly_in, *poly_out);
    FacePoly* const poly_tmp = poly_in;
    poly_in = poly_out;
    poly_out = poly_tmp;
  }
  if (poly_in != &poly) {
    poly = *poly_in;
  }
}

// The volume of the cone from center to a convex polygon, summed over the
// tetrahedrons of a fan triangulation as in BoxVolume.
__device__ inline float PolyVolume(const FacePoly& poly, const float3& center) {
  float vol = 0.0;
  const float3 v0 = poly.verts[0] - center;
  for (int v = 1; v + 1 < poly.num_verts; ++v) {
    const float3 v1 = poly.verts[v] - center;
    const float3 v2 = poly.verts[v + 1] - center;
    vol = vol + abs(dot(v0, cross(v1, v2))) / 6.0;
  }
  return vol;
}
