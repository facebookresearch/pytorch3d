// Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <tuple>

template <typename scalar_t>
__global__ void FaceAreasNormalsForwardKernel(
    const scalar_t* __restrict__ verts,
    const int64_t* __restrict__ faces,
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
    const int64_t i0 = faces[3 * f + 0];
    const int64_t i1 = faces[3 * f + 1];
    const int64_t i2 = faces[3 * f + 2];

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

// TODO(gkioxari) support all data types once AtomicAdd supports doubles.
// Currently, support is for floats only.
__global__ void FaceAreasNormalsBackwardKernel(
    const float* __restrict__ grad_areas,
    const float* __restrict__ grad_normals,
    const float* __restrict__ verts,
    const int64_t* __restrict__ faces,
    float* __restrict__ grad_verts,
    const size_t V,
    const size_t F) {
  const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = gridDim.x * blockDim.x;

  // Faces split evenly over the number of threads in the grid.
  // Each thread computes the area & normal of its respective faces and adds it
  // to the global face_areas tensor.
  for (size_t f = tid; f < F; f += stride) {
    const int64_t i0 = faces[3 * f + 0];
    const int64_t i1 = faces[3 * f + 1];
    const int64_t i2 = faces[3 * f + 2];

    const float v0_x = verts[3 * i0 + 0];
    const float v0_y = verts[3 * i0 + 1];
    const float v0_z = verts[3 * i0 + 2];

    const float v1_x = verts[3 * i1 + 0];
    const float v1_y = verts[3 * i1 + 1];
    const float v1_z = verts[3 * i1 + 2];

    const float v2_x = verts[3 * i2 + 0];
    const float v2_y = verts[3 * i2 + 1];
    const float v2_z = verts[3 * i2 + 2];

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
        ((-az + bz) * cy + (-by + ay) * cz) / 2.0 * inv_norm * grad_areas[f] +
        -cx * ((-az + bz) * cy + (-by + ay) * cz) * inv_norm_3 *
            grad_normals[3 * f + 0] +
        ((-az + bz) - cy * ((-az + bz) * cy + (-by + ay) * cz) * inv_norm_2) *
            inv_norm * grad_normals[3 * f + 1] +
        ((-by + ay) - cz * ((-az + bz) * cy + (-by + ay) * cz) * inv_norm_2) *
            inv_norm * grad_normals[3 * f + 2];
    atomicAdd(grad_verts + 3 * i0 + 0, grad_v0_x);

    const float grad_v0_y =
        ((-bz + az) * cx + (-ax + bx) * cz) / 2.0 * inv_norm * grad_areas[f] +
        ((-bz + az) - cx * ((-bz + az) * cx + (-ax + bx) * cz) * inv_norm_2) *
            inv_norm * grad_normals[3 * f + 0] +
        -cy * ((-bz + az) * cx + (-ax + bx) * cz) * inv_norm_3 *
            grad_normals[3 * f + 1] +
        ((-ax + bx) - cz * ((-bz + az) * cx + (-ax + bx) * cz) * inv_norm_2) *
            inv_norm * grad_normals[3 * f + 2];
    atomicAdd(grad_verts + 3 * i0 + 1, grad_v0_y);

    const float grad_v0_z =
        ((-ay + by) * cx + (-bx + ax) * cy) / 2.0 * inv_norm * grad_areas[f] +
        ((-ay + by) - cx * ((-ay + by) * cx + (-bx + ax) * cy) * inv_norm_2) *
            inv_norm * grad_normals[3 * f + 0] +
        ((-bx + ax) - cy * ((-ay + by) * cx + (-bx + ax) * cy) * inv_norm_2) *
            inv_norm * grad_normals[3 * f + 1] +
        -cz * ((-ay + by) * cx + (-bx + ax) * cy) * inv_norm_3 *
            grad_normals[3 * f + 2];
    atomicAdd(grad_verts + 3 * i0 + 2, grad_v0_z);

    // grad v1 coming from grad areas and grad normals
    const float grad_v1_x =
        (by * cz - bz * cy) / 2.0 * inv_norm * grad_areas[f] +
        -cx * (by * cz - bz * cy) * inv_norm_3 * grad_normals[3 * f + 0] +
        (-bz - cy * (by * cz - bz * cy) * inv_norm_2) * inv_norm *
            grad_normals[3 * f + 1] +
        (by - cz * (by * cz - bz * cy) * inv_norm_2) * inv_norm *
            grad_normals[3 * f + 2];
    atomicAdd(grad_verts + 3 * i1 + 0, grad_v1_x);

    const float grad_v1_y =
        (bz * cx - bx * cz) / 2.0 * inv_norm * grad_areas[f] +
        (bz - cx * (bz * cx - bx * cz) * inv_norm_2) * inv_norm *
            grad_normals[3 * f + 0] +
        -cy * (bz * cx - bx * cz) * inv_norm_3 * grad_normals[3 * f + 1] +
        (-bx - cz * (bz * cx - bx * cz) * inv_norm_2) * inv_norm *
            grad_normals[3 * f + 2];
    atomicAdd(grad_verts + 3 * i1 + 1, grad_v1_y);

    const float grad_v1_z =
        (bx * cy - by * cx) / 2.0 * inv_norm * grad_areas[f] +
        (-by - cx * (bx * cy - by * cx) * inv_norm_2) * inv_norm *
            grad_normals[3 * f + 0] +
        (bx - cx * (bx * cy - by * cx) * inv_norm_2) * inv_norm *
            grad_normals[3 * f + 1] +
        -cz * (bx * cy - by * cx) * inv_norm_3 * grad_normals[3 * f + 2];
    atomicAdd(grad_verts + 3 * i1 + 2, grad_v1_z);

    // grad v2 coming from grad areas
    const float grad_v2_x =
        (az * cy - ay * cz) / 2.0 * inv_norm * grad_areas[f] +
        -cx * (az * cy - ay * cz) * inv_norm_3 * grad_normals[3 * f + 0] +
        (az - cy * (az * cy - ay * cz) * inv_norm_2) * inv_norm *
            grad_normals[3 * f + 1] +
        (-ay - cz * (az * cy - ay * cz) * inv_norm_2) * inv_norm *
            grad_normals[3 * f + 2];
    atomicAdd(grad_verts + 3 * i2 + 0, grad_v2_x);

    const float grad_v2_y =
        (ax * cz - az * cx) / 2.0 * inv_norm * grad_areas[f] +
        (-az - cx * (ax * cz - az * cx) * inv_norm_2) * inv_norm *
            grad_normals[3 * f + 0] +
        -cy * (ax * cz - az * cx) * inv_norm_3 * grad_normals[3 * f + 1] +
        (ax - cz * (ax * cz - az * cx) * inv_norm_2) * inv_norm *
            grad_normals[3 * f + 2];
    atomicAdd(grad_verts + 3 * i2 + 1, grad_v2_y);

    const float grad_v2_z =
        (ay * cx - ax * cy) / 2.0 * inv_norm * grad_areas[f] +
        (ay - cx * (ay * cx - ax * cy) * inv_norm_2) * inv_norm *
            grad_normals[3 * f + 0] +
        (-ax - cy * (ay * cx - ax * cy) * inv_norm_2) * inv_norm *
            grad_normals[3 * f + 1] +
        -cz * (ay * cx - ax * cy) * inv_norm_3 * grad_normals[3 * f + 2];
    atomicAdd(grad_verts + 3 * i2 + 2, grad_v2_z);
  }
}

std::tuple<at::Tensor, at::Tensor> FaceAreasNormalsForwardCuda(
    const at::Tensor verts,
    const at::Tensor faces) {
  const auto V = verts.size(0);
  const auto F = faces.size(0);

  // Check inputs are on the same device
  at::TensorArg verts_t{verts, "verts", 1}, faces_t{faces, "faces", 2};
  at::CheckedFrom c = "FaceAreasNormalsForwardCuda";
  at::checkAllSameGPU(c, {verts_t, faces_t});

  // Set the device for the kernel launch based on the device of verts
  at::cuda::CUDAGuard device_guard(verts.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  at::Tensor areas = at::empty({F}, verts.options());
  at::Tensor normals = at::empty({F, 3}, verts.options());

  if (areas.numel() == 0) {
    AT_CUDA_CHECK(cudaGetLastError());
    return std::make_tuple(areas, normals);
  }

  const int blocks = 64;
  const int threads = 512;

  AT_DISPATCH_FLOATING_TYPES(
      verts.scalar_type(), "face_areas_normals_forward_cuda", ([&] {
        FaceAreasNormalsForwardKernel<scalar_t><<<blocks, threads, 0, stream>>>(
            verts.contiguous().data_ptr<scalar_t>(),
            faces.contiguous().data_ptr<int64_t>(),
            areas.data_ptr<scalar_t>(),
            normals.data_ptr<scalar_t>(),
            V,
            F);
      }));
  AT_CUDA_CHECK(cudaGetLastError());
  return std::make_tuple(areas, normals);
}

at::Tensor FaceAreasNormalsBackwardCuda(
    const at::Tensor grad_areas,
    const at::Tensor grad_normals,
    const at::Tensor verts,
    const at::Tensor faces) {
  // Check inputs are on the same device
  at::TensorArg verts_t{verts, "verts", 1}, faces_t{faces, "faces", 2},
      grad_areas_t{grad_areas, "grad_areas", 3},
      grad_normals_t{grad_normals, "grad_normals", 4};
  at::CheckedFrom c = "FaceAreasNormalsBackwardCuda";
  at::checkAllSameGPU(c, {verts_t, faces_t, grad_areas_t, grad_normals_t});

  // Set the device for the kernel launch based on the device of verts
  at::cuda::CUDAGuard device_guard(verts.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const auto V = verts.size(0);
  const auto F = faces.size(0);

  at::Tensor grad_verts = at::zeros({V, 3}, grad_areas.options());

  if (grad_verts.numel() == 0) {
    AT_CUDA_CHECK(cudaGetLastError());
    return grad_verts;
  }

  const int blocks = 64;
  const int threads = 512;
  // TODO(gkioxari) add AT_DISPATCH_FLOATING_TYPES once atomicAdd supports
  // doubles. Currently, support is for floats only.
  FaceAreasNormalsBackwardKernel<<<blocks, threads, 0, stream>>>(
      grad_areas.contiguous().data_ptr<float>(),
      grad_normals.contiguous().data_ptr<float>(),
      verts.contiguous().data_ptr<float>(),
      faces.contiguous().data_ptr<int64_t>(),
      grad_verts.data_ptr<float>(),
      V,
      F);

  AT_CUDA_CHECK(cudaGetLastError());
  return grad_verts;
}
