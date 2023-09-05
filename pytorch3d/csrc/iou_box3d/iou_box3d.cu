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
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "iou_box3d/iou_utils.cuh"

// Parallelize over N*M computations which can each be done
// independently
__global__ void IoUBox3DKernel(
    const at::PackedTensorAccessor64<float, 3, at::RestrictPtrTraits> boxes1,
    const at::PackedTensorAccessor64<float, 3, at::RestrictPtrTraits> boxes2,
    at::PackedTensorAccessor64<float, 2, at::RestrictPtrTraits> vols,
    at::PackedTensorAccessor64<float, 2, at::RestrictPtrTraits> ious) {
  const size_t N = boxes1.size(0);
  const size_t M = boxes2.size(0);

  const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = gridDim.x * blockDim.x;

  FaceVerts box1_tris[NUM_TRIS];
  FaceVerts box2_tris[NUM_TRIS];
  FaceVerts box1_planes[NUM_PLANES];
  FaceVerts box2_planes[NUM_PLANES];

  for (size_t i = tid; i < N * M; i += stride) {
    const size_t n = i / M; // box1 index
    const size_t m = i % M; // box2 index

    // Convert to array of structs of face vertices i.e. effectively (F, 3, 3)
    // FaceVerts is a data type defined in iou_utils.cuh
    GetBoxTris(boxes1[n], box1_tris);
    GetBoxTris(boxes2[m], box2_tris);

    // Calculate the position of the center of the box which is used in
    // several calculations. This requires a tensor as input.
    const float3 box1_center = BoxCenter(boxes1[n]);
    const float3 box2_center = BoxCenter(boxes2[m]);

    // Convert to an array of face vertices
    GetBoxPlanes(boxes1[n], box1_planes);
    GetBoxPlanes(boxes2[m], box2_planes);

    // Get Box Volumes
    const float box1_vol = BoxVolume(box1_tris, box1_center, NUM_TRIS);
    const float box2_vol = BoxVolume(box2_tris, box2_center, NUM_TRIS);

    // Tris in Box1 intersection with Planes in Box2
    // Initialize box1 intersecting faces. MAX_TRIS is the
    // max faces possible in the intersecting shape.
    // TODO: determine if the value of MAX_TRIS is sufficient or
    // if we should store the max tris for each NxM computation
    // and throw an error if any exceeds the max.
    FaceVerts box1_intersect[MAX_TRIS];
    for (int j = 0; j < NUM_TRIS; ++j) {
      // Initialize the faces from the box
      box1_intersect[j] = box1_tris[j];
    }
    // Get the count of the actual number of faces in the intersecting shape
    int box1_count = BoxIntersections(box2_planes, box2_center, box1_intersect);

    // Tris in Box2 intersection with Planes in Box1
    FaceVerts box2_intersect[MAX_TRIS];
    for (int j = 0; j < NUM_TRIS; ++j) {
      box2_intersect[j] = box2_tris[j];
    }
    const int box2_count =
        BoxIntersections(box1_planes, box1_center, box2_intersect);

    // If there are overlapping regions in Box2, remove any coplanar faces
    if (box2_count > 0) {
      // Identify if any triangles in Box2 are coplanar with Box1
      Keep tri2_keep[MAX_TRIS];
      for (int j = 0; j < MAX_TRIS; ++j) {
        // Initialize the valid faces to be true
        tri2_keep[j].keep = j < box2_count ? true : false;
      }
      for (int b1 = 0; b1 < box1_count; ++b1) {
        for (int b2 = 0; b2 < box2_count; ++b2) {
          const bool is_coplanar =
              IsCoplanarTriTri(box1_intersect[b1], box2_intersect[b2]);
          const float area = FaceArea(box1_intersect[b1]);
          if ((is_coplanar) && (area > aEpsilon)) {
            tri2_keep[b2].keep = false;
          }
        }
      }

      // Keep only the non coplanar triangles in Box2 - add them to the
      // Box1 triangles.
      for (int b2 = 0; b2 < box2_count; ++b2) {
        if (tri2_keep[b2].keep) {
          box1_intersect[box1_count] = box2_intersect[b2];
          // box1_count will determine the total faces in the
          // intersecting shape
          box1_count++;
        }
      }
    }

    // Initialize the vol and iou to 0.0 in case there are no triangles
    // in the intersecting shape.
    float vol = 0.0;
    float iou = 0.0;

    // If there are triangles in the intersecting shape
    if (box1_count > 0) {
      // The intersecting shape is a polyhedron made up of the
      // triangular faces that are all now in box1_intersect.
      // Calculate the polyhedron center
      const float3 poly_center = PolyhedronCenter(box1_intersect, box1_count);
      // Compute intersecting polyhedron volume
      vol = BoxVolume(box1_intersect, poly_center, box1_count);
      // Compute IoU
      iou = vol / (box1_vol + box2_vol - vol);
    }

    // Write the volume and IoU to global memory
    vols[n][m] = vol;
    ious[n][m] = iou;
  }
}

std::tuple<at::Tensor, at::Tensor> IoUBox3DCuda(
    const at::Tensor& boxes1, // (N, 8, 3)
    const at::Tensor& boxes2) { // (M, 8, 3)
  // Check inputs are on the same device
  at::TensorArg boxes1_t{boxes1, "boxes1", 1}, boxes2_t{boxes2, "boxes2", 2};
  at::CheckedFrom c = "IoUBox3DCuda";
  at::checkAllSameGPU(c, {boxes1_t, boxes2_t});
  at::checkAllSameType(c, {boxes1_t, boxes2_t});

  // Set the device for the kernel launch based on the device of boxes1
  at::cuda::CUDAGuard device_guard(boxes1.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  TORCH_CHECK(boxes2.size(2) == boxes1.size(2), "Boxes must have shape (8, 3)");

  TORCH_CHECK(
      (boxes2.size(1) == 8) && (boxes1.size(1) == 8),
      "Boxes must have shape (8, 3)");

  const int64_t N = boxes1.size(0);
  const int64_t M = boxes2.size(0);

  auto vols = at::zeros({N, M}, boxes1.options());
  auto ious = at::zeros({N, M}, boxes1.options());

  if (vols.numel() == 0) {
    AT_CUDA_CHECK(cudaGetLastError());
    return std::make_tuple(vols, ious);
  }

  const size_t blocks = 512;
  const size_t threads = 256;

  IoUBox3DKernel<<<blocks, threads, 0, stream>>>(
      boxes1.packed_accessor64<float, 3, at::RestrictPtrTraits>(),
      boxes2.packed_accessor64<float, 3, at::RestrictPtrTraits>(),
      vols.packed_accessor64<float, 2, at::RestrictPtrTraits>(),
      ious.packed_accessor64<float, 2, at::RestrictPtrTraits>());

  AT_CUDA_CHECK(cudaGetLastError());

  return std::make_tuple(vols, ious);
}
