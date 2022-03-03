/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <torch/extension.h>
#include <tuple>
#include "utils/pytorch3d_cutils.h"

// Interpolates per-face attributes (forward pass)
//
// Inputs:
//    pix_to_face: LongTensor of shape (P,) giving a face index for each pixel.
//        Each element should be < F, the total number of faces.
//        Face indices < 0 indicate that the pixel is not covered by a face.
//    barycentric_coords: FloatTensor of shape (P, 3) giving barycentric coords.
//    face_attrs: FloatTensor of shape (F, 3, D) giving a D-dimensional
//        value for each vertex of each face.
//
// Returns:
//    pix_attributes: FloatTensor of shape (P, D) giving an interpolated value
//    for each pixel.

// CPU implementation
at::Tensor InterpFaceAttrsForwardCpu(
    const at::Tensor& pix_to_face,
    const at::Tensor& barycentric_coords,
    const at::Tensor& face_attrs) {
  AT_ERROR("Not Implemented");
  return pix_to_face;
}

#ifdef WITH_CUDA
// Cuda implementation.
at::Tensor InterpFaceAttrsForwardCuda(
    const at::Tensor& pix_to_face,
    const at::Tensor& barycentric_coords,
    const at::Tensor& face_attrs);
#endif

// General implementation
at::Tensor InterpFaceAttrsForward(
    const at::Tensor& pix_to_face,
    const at::Tensor& barycentric_coords,
    const at::Tensor& face_attrs) {
  if (pix_to_face.is_cuda()) {
#ifdef WITH_CUDA
    CHECK_CUDA(face_attrs);
    CHECK_CUDA(barycentric_coords);
    return InterpFaceAttrsForwardCuda(
        pix_to_face, barycentric_coords, face_attrs);
#else
    AT_ERROR("Not compiled with GPU support.");
#endif
  }
  return InterpFaceAttrsForwardCpu(pix_to_face, barycentric_coords, face_attrs);
}

// Interpolates per-face attributes (backward pass)
//
// Inputs:
//    pix_to_face: LongTensor of shape (P,) giving a face index for each pixel.
//        Each element should be < F, the total number of faces.
//        Face indices < 0 indicate that the pixel is not covered by a face.
//    barycentric_coords: FloatTensor of shape (P, 3) giving barycentric coords.
//    face_attrs: FloatTensor of shape (F, 3, D) giving a D-dimensional
//        value for each vertex of each face.
//    grad_pix_attrs: Upstream gradients of shape (P, D)
//
// Returns a tuple of:
//    grad_barycentric_coords: FloatTensor of shape (P, 3)
//    grad_face_attrs: FloatTensor of shape (F, 3, D)

std::tuple<at::Tensor, at::Tensor> InterpFaceAttrsBackwardCpu(
    const at::Tensor& pix_to_face,
    const at::Tensor& barycentric_coords,
    const at::Tensor& face_attrs,
    const at::Tensor& grad_pix_attrs) {
  AT_ERROR("Not Implemented");
  return std::make_tuple(pix_to_face, pix_to_face);
}

std::tuple<at::Tensor, at::Tensor> InterpFaceAttrsBackwardCuda(
    const at::Tensor& pix_to_face,
    const at::Tensor& barycentric_coords,
    const at::Tensor& face_attrs,
    const at::Tensor& grad_pix_attrs);

std::tuple<at::Tensor, at::Tensor> InterpFaceAttrsBackward(
    const at::Tensor& pix_to_face,
    const at::Tensor& barycentric_coords,
    const at::Tensor& face_attrs,
    const at::Tensor& grad_pix_attrs) {
  if (pix_to_face.is_cuda()) {
#ifdef WITH_CUDA
    CHECK_CUDA(face_attrs);
    CHECK_CUDA(barycentric_coords);
    CHECK_CUDA(grad_pix_attrs);
    return InterpFaceAttrsBackwardCuda(
        pix_to_face, barycentric_coords, face_attrs, grad_pix_attrs);
#else
    AT_ERROR("Not compiled with GPU support.");
#endif
  }
  return InterpFaceAttrsBackwardCpu(
      pix_to_face, barycentric_coords, face_attrs, grad_pix_attrs);
}
