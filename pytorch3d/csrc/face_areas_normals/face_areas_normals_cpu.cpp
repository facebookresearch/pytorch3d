// Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

#include <torch/extension.h>
#include <tuple>

std::tuple<at::Tensor, at::Tensor> FaceAreasNormalsForwardCpu(
    const at::Tensor verts,
    const at::Tensor faces) {
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

at::Tensor FaceAreasNormalsBackwardCpu(
    const at::Tensor grad_areas,
    const at::Tensor grad_normals,
    const at::Tensor verts,
    const at::Tensor faces) {
  const int V = verts.size(0);
  const int F = faces.size(0);

  at::Tensor grad_verts = at::zeros({V, 3}, grad_areas.options());

  auto grad_areas_a = grad_areas.accessor<float, 1>();
  auto grad_normals_a = grad_normals.accessor<float, 2>();
  auto verts_a = verts.accessor<float, 2>();
  auto faces_a = faces.accessor<int64_t, 2>();
  auto grad_verts_a = grad_verts.accessor<float, 2>();

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
    norm = (norm < 1e-6) ? 1e-6 : norm; // max(norm, 1e-6)
    float inv_norm = 1. / norm;
    float inv_norm_2 = pow(inv_norm, 2.0f);
    float inv_norm_3 = pow(inv_norm, 3.0f);

    // We compute gradients with respect to the input vertices.
    // For each vertex, gradients come from grad_areas and grad_normals.
    // eg, grad_v0_x = (d / d v0_x)
    //       = \sum_f (d / d areas[f]) * (d areas[f] / d v0_x)
    //              + (d / d normals[f, 0]) * (d normals[f, 0] / d v0_x)
    //              + (d / d normals[f, 1]) * (d normals[f, 1] / d v0_x)
    //              + (d / d normals[f, 2]) * (d normals[f, 2] / d v0_x)
    // with (d / d areas[f]) = grad_areas[f] and
    //      (d / d normals[f, j]) = grad_normals[f][j].
    // The equations below are derived after taking
    // derivatives wrt to the vertices (fun times!).

    // grad v0 coming from grad areas and grad normals
    const float grad_v0_x =
        ((-az + bz) * cy + (-by + ay) * cz) / 2.0 * inv_norm * grad_areas_a[f] +
        -cx * ((-az + bz) * cy + (-by + ay) * cz) * inv_norm_3 *
            grad_normals_a[f][0] +
        ((-az + bz) - cy * ((-az + bz) * cy + (-by + ay) * cz) * inv_norm_2) *
            inv_norm * grad_normals_a[f][1] +
        ((-by + ay) - cz * ((-az + bz) * cy + (-by + ay) * cz) * inv_norm_2) *
            inv_norm * grad_normals_a[f][2];
    grad_verts_a[i0][0] += grad_v0_x;

    const float grad_v0_y =
        ((-bz + az) * cx + (-ax + bx) * cz) / 2.0 * inv_norm * grad_areas_a[f] +
        ((-bz + az) - cx * ((-bz + az) * cx + (-ax + bx) * cz) * inv_norm_2) *
            inv_norm * grad_normals_a[f][0] +
        -cy * ((-bz + az) * cx + (-ax + bx) * cz) * inv_norm_3 *
            grad_normals_a[f][1] +
        ((-ax + bx) - cz * ((-bz + az) * cx + (-ax + bx) * cz) * inv_norm_2) *
            inv_norm * grad_normals_a[f][2];
    grad_verts[i0][1] += grad_v0_y;

    const float grad_v0_z =
        ((-ay + by) * cx + (-bx + ax) * cy) / 2.0 * inv_norm * grad_areas_a[f] +
        ((-ay + by) - cx * ((-ay + by) * cx + (-bx + ax) * cy) * inv_norm_2) *
            inv_norm * grad_normals_a[f][0] +
        ((-bx + ax) - cy * ((-ay + by) * cx + (-bx + ax) * cy) * inv_norm_2) *
            inv_norm * grad_normals_a[f][1] +
        -cz * ((-ay + by) * cx + (-bx + ax) * cy) * inv_norm_3 *
            grad_normals_a[f][2];
    grad_verts[i0][2] += grad_v0_z;

    // grad v1 coming from grad areas and grad normals
    const float grad_v1_x =
        (by * cz - bz * cy) / 2.0 * inv_norm * grad_areas_a[f] +
        -cx * (by * cz - bz * cy) * inv_norm_3 * grad_normals_a[f][0] +
        (-bz - cy * (by * cz - bz * cy) * inv_norm_2) * inv_norm *
            grad_normals_a[f][1] +
        (by - cz * (by * cz - bz * cy) * inv_norm_2) * inv_norm *
            grad_normals_a[f][2];
    grad_verts[i1][0] += grad_v1_x;

    const float grad_v1_y =
        (bz * cx - bx * cz) / 2.0 * inv_norm * grad_areas_a[f] +
        (bz - cx * (bz * cx - bx * cz) * inv_norm_2) * inv_norm *
            grad_normals_a[f][0] +
        -cy * (bz * cx - bx * cz) * inv_norm_3 * grad_normals_a[f][1] +
        (-bx - cz * (bz * cx - bx * cz) * inv_norm_2) * inv_norm *
            grad_normals_a[f][2];
    grad_verts[i1][1] += grad_v1_y;

    const float grad_v1_z =
        (bx * cy - by * cx) / 2.0 * inv_norm * grad_areas_a[f] +
        (-by - cx * (bx * cy - by * cx) * inv_norm_2) * inv_norm *
            grad_normals_a[f][0] +
        (bx - cx * (bx * cy - by * cx) * inv_norm_2) * inv_norm *
            grad_normals_a[f][1] +
        -cz * (bx * cy - by * cx) * inv_norm_3 * grad_normals_a[f][2];
    grad_verts[i1][2] += grad_v1_z;

    // grad v2 coming from grad areas
    const float grad_v2_x =
        (az * cy - ay * cz) / 2.0 * inv_norm * grad_areas_a[f] +
        -cx * (az * cy - ay * cz) * inv_norm_3 * grad_normals_a[f][0] +
        (az - cy * (az * cy - ay * cz) * inv_norm_2) * inv_norm *
            grad_normals_a[f][1] +
        (-ay - cz * (az * cy - ay * cz) * inv_norm_2) * inv_norm *
            grad_normals_a[f][2];
    grad_verts[i2][0] += grad_v2_x;

    const float grad_v2_y =
        (ax * cz - az * cx) / 2.0 * inv_norm * grad_areas_a[f] +
        (-az - cx * (ax * cz - az * cx) * inv_norm_2) * inv_norm *
            grad_normals_a[f][0] +
        -cy * (ax * cz - az * cx) * inv_norm_3 * grad_normals_a[f][1] +
        (ax - cz * (ax * cz - az * cx) * inv_norm_2) * inv_norm *
            grad_normals_a[f][2];
    grad_verts[i2][1] += grad_v2_y;

    const float grad_v2_z =
        (ay * cx - ax * cy) / 2.0 * inv_norm * grad_areas_a[f] +
        (ay - cx * (ay * cx - ax * cy) * inv_norm_2) * inv_norm *
            grad_normals_a[f][0] +
        (-ax - cy * (ay * cx - ax * cy) * inv_norm_2) * inv_norm *
            grad_normals_a[f][1] +
        -cz * (ay * cx - ax * cy) * inv_norm_3 * grad_normals_a[f][2];
    grad_verts[i2][2] += grad_v2_z;
  }
  return grad_verts;
}
