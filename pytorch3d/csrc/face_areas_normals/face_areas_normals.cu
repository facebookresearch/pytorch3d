// Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include <ATen/ATen.h>
#include <tuple>

template <typename scalar_t>
__global__ void face_areas_kernel(
    const scalar_t* __restrict__ verts,
    const long* __restrict__ faces,
    scalar_t* __restrict__ face_areas,
    scalar_t* __restrict__ face_normals,
    const size_t V,
    const size_t F) {
  const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = gridDim.x * blockDim.x;

  // Faces split evenly over the number of threads in the grid.
  // Each thread computes the area & normal of its respective faces and adds it
  // to the global face_areas tensor.
  for (size_t f = tid; f < F; f += stride) {
    const long i0 = faces[3 * f + 0];
    const long i1 = faces[3 * f + 1];
    const long i2 = faces[3 * f + 2];

    const scalar_t v0_x = verts[3 * i0 + 0];
    const scalar_t v0_y = verts[3 * i0 + 1];
    const scalar_t v0_z = verts[3 * i0 + 2];

    const scalar_t v1_x = verts[3 * i1 + 0];
    const scalar_t v1_y = verts[3 * i1 + 1];
    const scalar_t v1_z = verts[3 * i1 + 2];

    const scalar_t v2_x = verts[3 * i2 + 0];
    const scalar_t v2_y = verts[3 * i2 + 1];
    const scalar_t v2_z = verts[3 * i2 + 2];

    const scalar_t ax = v1_x - v0_x;
    const scalar_t ay = v1_y - v0_y;
    const scalar_t az = v1_z - v0_z;

    const scalar_t bx = v2_x - v0_x;
    const scalar_t by = v2_y - v0_y;
    const scalar_t bz = v2_z - v0_z;

    const scalar_t cx = ay * bz - az * by;
    const scalar_t cy = az * bx - ax * bz;
    const scalar_t cz = ax * by - ay * bx;

    scalar_t norm = sqrt(cx * cx + cy * cy + cz * cz);
    face_areas[f] = norm / 2.0;
    norm = (norm < 1e-6) ? 1e-6 : norm; // max(norm, 1e-6)
    face_normals[3 * f + 0] = cx / norm;
    face_normals[3 * f + 1] = cy / norm;
    face_normals[3 * f + 2] = cz / norm;
  }
}

std::tuple<at::Tensor, at::Tensor> face_areas_cuda(
    at::Tensor verts,
    at::Tensor faces) {
  const auto V = verts.size(0);
  const auto F = faces.size(0);

  at::Tensor areas = at::empty({F}, verts.options());
  at::Tensor normals = at::empty({F, 3}, verts.options());

  const int blocks = 64;
  const int threads = 512;
  AT_DISPATCH_FLOATING_TYPES(verts.type(), "face_areas_kernel", ([&] {
                               face_areas_kernel<scalar_t><<<blocks, threads>>>(
                                   verts.data_ptr<scalar_t>(),
                                   faces.data_ptr<long>(),
                                   areas.data_ptr<scalar_t>(),
                                   normals.data_ptr<scalar_t>(),
                                   V,
                                   F);
                             }));

  return std::make_tuple(areas, normals);
}
