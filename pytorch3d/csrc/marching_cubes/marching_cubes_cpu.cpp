/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <torch/extension.h>
#include <algorithm>
#include <array>
#include <cstring>
#include <unordered_map>
#include <vector>
#include "marching_cubes/marching_cubes_utils.h"
#include "marching_cubes/tables.h"

// Cpu implementation for Marching Cubes
// Args:
//    vol: a Tensor of size (D, H, W) corresponding to a 3D scalar field
//    isolevel: the isosurface value to use as the threshold to determine
//          whether points are within a volume.
//
// Returns:
//    vertices: a float tensor of shape (N_verts, 3) for positions of the mesh
//    faces: a long tensor of shape (N_faces, 3) for indices of the face
//    ids: a long tensor of shape (N_verts) as placeholder
//
std::tuple<at::Tensor, at::Tensor, at::Tensor> MarchingCubesCpu(
    const at::Tensor& vol,
    const float isolevel) {
  // volume shapes
  const int D = vol.size(0);
  const int H = vol.size(1);
  const int W = vol.size(2);

  // Create tensor accessors
  auto vol_a = vol.accessor<float, 3>();
  // edge_id_to_v maps from an edge id to a vertex position
  std::unordered_map<int64_t, Vertex> edge_id_to_v;
  // uniq_edge_id: used to remove redundant edge ids
  std::unordered_map<int64_t, int64_t> uniq_edge_id;
  std::vector<int64_t> faces; // store face indices
  std::vector<Vertex> verts; // store vertex positions
  // enumerate each cell in the 3d grid
  for (int z = 0; z < D - 1; z++) {
    for (int y = 0; y < H - 1; y++) {
      for (int x = 0; x < W - 1; x++) {
        Cube cube(x, y, z, vol_a, isolevel);
        // Cube is entirely in/out of the surface
        if (_FACE_TABLE[cube.cubeindex][0] == 255) {
          continue;
        }
        // store all boundary vertices that intersect with the edges
        std::array<Vertex, 12> interp_points;
        // triangle vertex IDs and positions
        std::vector<int64_t> tri;
        std::vector<Vertex> ps;

        // Interpolate the vertices where the surface intersects with the cube
        for (int j = 0; _FACE_TABLE[cube.cubeindex][j] != 255; j++) {
          const int e = _FACE_TABLE[cube.cubeindex][j];
          interp_points[e] = cube.VertexInterp(isolevel, e, vol_a);

          int64_t edge = cube.HashVpair(e, W, H, D);
          tri.push_back(edge);
          ps.push_back(interp_points[e]);

          // Check if the triangle face is degenerate. A triangle face
          // is degenerate if any of the two verices share the same 3D position
          if ((j + 1) % 3 == 0 && ps[0] != ps[1] && ps[1] != ps[2] &&
              ps[2] != ps[0]) {
            for (int k = 0; k < 3; k++) {
              int64_t v = tri.at(k);
              edge_id_to_v[v] = ps.at(k);
              if (!uniq_edge_id.count(v)) {
                uniq_edge_id[v] = verts.size();
                verts.push_back(edge_id_to_v[v]);
              }
              faces.push_back(uniq_edge_id[v]);
            }
            tri.clear();
            ps.clear();
          } // endif
        } // endfor edge enumeration
      } // endfor x
    } // endfor y
  } // endfor z
  // Collect returning tensor
  const int n_vertices = verts.size();
  const int64_t n_faces = (int64_t)faces.size() / 3;
  auto vert_tensor = torch::zeros({n_vertices, 3}, torch::kFloat);
  auto id_tensor = torch::zeros({n_vertices}, torch::kInt64); // placeholder
  auto face_tensor = torch::zeros({n_faces, 3}, torch::kInt64);

  auto vert_a = vert_tensor.accessor<float, 2>();
  for (int i = 0; i < n_vertices; i++) {
    vert_a[i][0] = verts.at(i).x;
    vert_a[i][1] = verts.at(i).y;
    vert_a[i][2] = verts.at(i).z;
  }

  auto face_a = face_tensor.accessor<int64_t, 2>();
  for (int64_t i = 0; i < n_faces; i++) {
    face_a[i][0] = faces.at(i * 3 + 0);
    face_a[i][1] = faces.at(i * 3 + 1);
    face_a[i][2] = faces.at(i * 3 + 2);
  }

  return std::make_tuple(vert_tensor, face_tensor, id_tensor);
}
