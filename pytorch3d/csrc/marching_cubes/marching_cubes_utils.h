/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <torch/extension.h>
#include <cmath>
#include <cstdint>
#include <vector>
#include "ATen/core/TensorAccessor.h"
#include "marching_cubes/tables.h"

// EPS: Used to assess whether two float values are close
const float EPS = 1e-5;

// Data structures for the marching cubes
struct Vertex {
  // Constructor used when performing marching cube in each cell
  explicit Vertex(float x = 0.0f, float y = 0.0f, float z = 0.0f)
      : x(x), y(y), z(z) {}

  // The */+ operator overrides are used for vertex interpolation
  Vertex operator*(float s) const {
    return Vertex(x * s, y * s, z * s);
  }
  Vertex operator+(const Vertex& xyz) const {
    return Vertex(x + xyz.x, y + xyz.y, z + xyz.z);
  }
  // The =/!= operator overrides is used for checking degenerate triangles
  bool operator==(const Vertex& xyz) const {
    return (
        std::abs(x - xyz.x) < EPS && std::abs(y - xyz.y) < EPS &&
        std::abs(z - xyz.z) < EPS);
  }
  bool operator!=(const Vertex& xyz) const {
    return (
        std::abs(x - xyz.x) >= EPS || std::abs(y - xyz.y) >= EPS ||
        std::abs(z - xyz.z) >= EPS);
  }
  // vertex position
  float x, y, z;
};

struct Cube {
  // Edge and vertex convention:
  //                     v4_______e4____________v5
  //                     /|                    /|
  //                    / |                   / |
  //                 e7/  |                e5/  |
  //                  /___|______e6_________/   |
  //               v7|    |                 |v6 |e9
  //                 |    |                 |   |
  //                 |    |e8               |e10|
  //              e11|    |                 |   |
  //                 |    |_________________|___|
  //                 |   / v0      e0       |   /v1
  //                 |  /                   |  /
  //                 | /e3                  | /e1
  //                 |/_____________________|/
  //                 v3         e2          v2

  Vertex p[8];
  int x, y, z;
  int cubeindex = 0;
  Cube(
      int x,
      int y,
      int z,
      const at::TensorAccessor<float, 3>& vol_a,
      const float isolevel)
      : x(x), y(y), z(z) {
    // vertex position (x, y, z) for v0-v1-v4-v5-v3-v2-v7-v6
    for (int v = 0; v < 8; v++) {
      p[v] = Vertex(x + (v & 1), y + (v >> 1 & 1), z + (v >> 2 & 1));
    }
    // Calculates cube configuration index given values of the cube vertices
    for (int i = 0; i < 8; i++) {
      const int idx = _INDEX_TABLE[i];
      Vertex v = p[idx];
      if (vol_a[v.z][v.y][v.x] < isolevel) {
        cubeindex |= (1 << i);
      }
    }
  }

  // Linearly interpolate the position where an isosurface cuts an edge
  // between two vertices, based on their scalar values
  //
  // Args:
  //    isolevel: float value used as threshold
  //    edge: edge (ID) to interpolate
  //    cube: current cube vertices
  //    vol_a: 3D scalar field
  //
  // Returns:
  //    point: interpolated vertex
  Vertex VertexInterp(
      float isolevel,
      const int edge,
      const at::TensorAccessor<float, 3>& vol_a) {
    const int v1 = _EDGE_TO_VERTICES[edge][0];
    const int v2 = _EDGE_TO_VERTICES[edge][1];
    Vertex p1 = p[v1];
    Vertex p2 = p[v2];
    float val1 = vol_a[p1.z][p1.y][p1.x];
    float val2 = vol_a[p2.z][p2.y][p2.x];

    float ratio = 1.0f;
    if (std::abs(isolevel - val1) < EPS) {
      return p1;
    } else if (std::abs(isolevel - val2) < EPS) {
      return p2;
    } else if (std::abs(val1 - val2) < EPS) {
      return p1;
    }
    // interpolate vertex p based on two vertices on the edge
    ratio = (isolevel - val1) / (val2 - val1);
    return p1 * (1 - ratio) + p2 * ratio;
  }

  // Hash an edge into a global edge_id. The function binds an
  // edge with an integer to address floating point precision issue.
  //
  // Args:
  //    v1_id: global id of vertex 1
  //    v2_id: global id of vertex 2
  //    W: width of the 3d grid
  //    H: height of the 3d grid
  //    D: depth of the 3d grid
  //
  // Returns:
  //    hashing for a pair of vertex ids
  //
  int64_t HashVpair(const int edge, int W, int H, int D) {
    const int v1 = _EDGE_TO_VERTICES[edge][0];
    const int v2 = _EDGE_TO_VERTICES[edge][1];
    const int v1_id = p[v1].x + p[v1].y * W + p[v1].z * W * H;
    const int v2_id = p[v2].x + p[v2].y * W + p[v2].z * W * H;
    return (int64_t)v1_id * (W + W * H + W * H * D) + (int64_t)v2_id;
  }
};
