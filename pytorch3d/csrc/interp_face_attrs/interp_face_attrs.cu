/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <tuple>

template <typename scalar_t>
__global__ void InterpFaceAttrsForwardKernel(
    const int64_t* __restrict__ pix_to_face, // (P,)
    const scalar_t* __restrict__ barycentric_coords, // (P, 3)
    const scalar_t* __restrict__ face_attrs, // (F, 3, D)
    scalar_t* pix_attrs, // (P, D)
    const size_t P,
    const size_t F,
    const size_t D) {
  const auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  const auto num_threads = blockDim.x * gridDim.x;
  for (int pd = tid; pd < P * D; pd += num_threads) {
    const int p = pd / D;
    const int d = pd % D;
    const int64_t f = pix_to_face[p];
    if (f < 0) {
      continue;
    }
    scalar_t pix_attr = 0.0;
    for (int i = 0; i < 3; ++i) {
      scalar_t weight = barycentric_coords[p * 3 + i];
      scalar_t vert_attr = face_attrs[f * 3 * D + i * D + d];
      pix_attr += weight * vert_attr;
    }
    pix_attrs[p * D + d] = pix_attr;
  }
}

at::Tensor InterpFaceAttrsForwardCuda(
    const at::Tensor& pix_to_face,
    const at::Tensor& barycentric_coords,
    const at::Tensor& face_attrs) {
  // Make sure all inputs are on the same device
  at::TensorArg pix_to_face_t{pix_to_face, "pix_to_face", 1},
      barycentric_coords_t{barycentric_coords, "barycentric_coords", 2},
      face_attrs_t{face_attrs, "face_attributes", 3};
  at::CheckedFrom c = "InterpFaceAttrsForwardCuda";
  at::checkAllSameGPU(c, {pix_to_face_t, barycentric_coords_t, face_attrs_t});
  at::checkAllSameType(c, {barycentric_coords_t, face_attrs_t});

  // Set the device for the kernel launch based on the input
  at::cuda::CUDAGuard device_guard(pix_to_face.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const auto P = pix_to_face.size(0);
  const auto F = face_attrs.size(0);
  const auto D = face_attrs.size(2);

  TORCH_CHECK(
      barycentric_coords.size(0) == P && barycentric_coords.size(1) == 3,
      "barycentric_coords must have size (P, 3)");
  TORCH_CHECK(face_attrs.size(1) == 3, "face_attrs must have size (F, 3, D)");

  auto pix_attrs = at::zeros({P, D}, face_attrs.options());
  const int threads = 1024;
  const int blocks = 512;
  AT_DISPATCH_FLOATING_TYPES(
      face_attrs.scalar_type(), "interp_face_attrs_cuda", ([&] {
        InterpFaceAttrsForwardKernel<<<blocks, threads, 0, stream>>>(
            pix_to_face.contiguous().data_ptr<int64_t>(),
            barycentric_coords.contiguous().data_ptr<scalar_t>(),
            face_attrs.contiguous().data_ptr<scalar_t>(),
            pix_attrs.contiguous().data_ptr<scalar_t>(),
            P,
            F,
            D);
      }));
  AT_CUDA_CHECK(cudaGetLastError());
  return pix_attrs;
}

template <typename scalar_t>
__global__ void InterpFaceAttrsBackwardKernel(
    const int64_t* __restrict__ pix_to_face, // (P,)
    const scalar_t* __restrict__ barycentric_coords, // (P, 3)
    const scalar_t* __restrict__ face_attrs, // (F, 3, D)
    const scalar_t* __restrict__ grad_pix_attrs, // (P, D)
    scalar_t* __restrict__ grad_barycentric_coords, // (P, 3)
    scalar_t* __restrict__ grad_face_attrs, // (F, 3, D)
    const size_t P,
    const size_t F,
    const size_t D) {
  const auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  const auto num_threads = blockDim.x * gridDim.x;
  for (int pd = tid; pd < P * D; pd += num_threads) {
    const int p = pd / D;
    const int d = pd % D;
    const int64_t f = pix_to_face[p];
    if (f < 0) {
      continue;
    }
    scalar_t upstream_grad = grad_pix_attrs[p * D + d];
    for (int i = 0; i < 3; ++i) {
      scalar_t weight = barycentric_coords[p * 3 + i];
      scalar_t vert_attr = face_attrs[f * 3 * D + i * D + d];
      scalar_t grad_bary_down = vert_attr * upstream_grad;
      scalar_t grad_face_down = weight * upstream_grad;
      atomicAdd(grad_barycentric_coords + p * 3 + i, grad_bary_down);
      atomicAdd(grad_face_attrs + f * 3 * D + i * D + d, grad_face_down);
    }
  }
}

std::tuple<at::Tensor, at::Tensor> InterpFaceAttrsBackwardCuda(
    const at::Tensor& pix_to_face,
    const at::Tensor& barycentric_coords,
    const at::Tensor& face_attrs,
    const at::Tensor& grad_pix_attrs) {
  // Make sure all inputs are on the same device
  at::TensorArg pix_to_face_t{pix_to_face, "pix_to_face", 1},
      barycentric_coords_t{barycentric_coords, "barycentric_coords", 2},
      face_attrs_t{face_attrs, "face_attributes", 3},
      grad_pix_attrs_t{grad_pix_attrs, "pix_attrs", 4};
  at::CheckedFrom c = "InterpFaceAttrsBackwarduda";
  at::checkAllSameGPU(
      c, {pix_to_face_t, barycentric_coords_t, face_attrs_t, grad_pix_attrs_t});
  at::checkAllSameType(
      c, {barycentric_coords_t, face_attrs_t, grad_pix_attrs_t});

  // This is nondeterministic because atomicAdd
  at::globalContext().alertNotDeterministic("InterpFaceAttrsBackwardCuda");

  // Set the device for the kernel launch based on the input
  at::cuda::CUDAGuard device_guard(pix_to_face.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const auto P = pix_to_face.size(0);
  const auto F = face_attrs.size(0);
  const auto D = face_attrs.size(2);

  TORCH_CHECK(
      barycentric_coords.size(0) == P && barycentric_coords.size(1) == 3,
      "barycentric_coords must have size (P, 3)");
  TORCH_CHECK(face_attrs.size(1) == 3, "face_attrs must have size (F, 3, D)");
  TORCH_CHECK(
      grad_pix_attrs.size(0) == P && grad_pix_attrs.size(1) == D,
      "grad_pix_attrs must have size (P, D)");

  auto grad_barycentric_coords = at::zeros_like(barycentric_coords);
  auto grad_face_attrs = at::zeros_like(face_attrs);
  const int threads = 1024;
  const int blocks = 512;
  // Only allow float for now.
  // TODO: Add support for double once we fix atomicAdd
  // clang-format off
  InterpFaceAttrsBackwardKernel<<<blocks, threads, 0, stream>>>(
    pix_to_face.contiguous().data_ptr<int64_t>(),
    barycentric_coords.contiguous().data_ptr<float>(),
    face_attrs.contiguous().data_ptr<float>(),
    grad_pix_attrs.contiguous().data_ptr<float>(),
    grad_barycentric_coords.contiguous().data_ptr<float>(),
    grad_face_attrs.contiguous().data_ptr<float>(),
    P, F, D);
  AT_CUDA_CHECK(cudaGetLastError());
  // clang-format on
  return std::make_tuple(grad_barycentric_coords, grad_face_attrs);
}
