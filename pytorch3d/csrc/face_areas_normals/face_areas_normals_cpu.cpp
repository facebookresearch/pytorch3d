// Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

#include <torch/extension.h>
#include <tuple>

std::tuple<at::Tensor, at::Tensor> FaceAreasNormalsCpu(
    at::Tensor verts,
    at::Tensor faces) {
  const int V = verts.size(0);
  const int F = faces.size(0);

  at::Tensor areas = at::empty({F}, verts.options());
  at::Tensor normals = at::empty({F, 3}, verts.options());

  auto verts_a = verts.accessor<float, 2>();
  auto faces_a = faces.accessor<int64_t, 2>();
  auto areas_a = areas.accessor<float, 1>();
  auto normals_a = normals.accessor<float, 2>();

  for (int f = 0; f < F; ++f) {
    const int64_t i0 = faces_a[f][0];
    const int64_t i1 = faces_a[f][1];
    const int64_t i2 = faces_a[f][2];

    const float v0_x = verts_a[i0][0];
    const float v0_y = verts_a[i0][1];
    const float v0_z = verts_a[i0][2];

    const float v1_x = verts_a[i1][0];
    const float v1_y = verts_a[i1][1];
    const float v1_z = verts_a[i1][2];

    const float v2_x = verts_a[i2][0];
    const float v2_y = verts_a[i2][1];
    const float v2_z = verts_a[i2][2];

    const float ax = v1_x - v0_x;
    const float ay = v1_y - v0_y;
    const float az = v1_z - v0_z;

    const float bx = v2_x - v0_x;
    const float by = v2_y - v0_y;
    const float bz = v2_z - v0_z;

    const float cx = ay * bz - az * by;
    const float cy = az * bx - ax * bz;
    const float cz = ax * by - ay * bx;

    float norm = sqrt(cx * cx + cy * cy + cz * cz);
    areas_a[f] = norm / 2.0;
    norm = (norm < 1e-6) ? 1e-6 : norm; // max(norm, 1e-6)
    normals_a[f][0] = cx / norm;
    normals_a[f][1] = cy / norm;
    normals_a[f][2] = cz / norm;
  }
  return std::make_tuple(areas, normals);
}
