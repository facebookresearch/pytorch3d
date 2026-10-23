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

  BoxCorners box1;
  BoxCorners box2;
  FacePlane box1_faces[NUM_PLANES];
  FacePlane box2_faces[NUM_PLANES];

  for (size_t i = tid; i < N * M; i += stride) {
    const size_t n = i / M; // box1 index
    const size_t m = i % M; // box2 index

    GetBoxCorners(boxes1[n], box1);
    GetBoxCorners(boxes2[m], box2);

    // Calculate the position of the center of the box which is used in
    // several calculations. This requires a tensor as input.
    const float3 box1_center = BoxCenter(boxes1[n]);
    const float3 box2_center = BoxCenter(boxes2[m]);

    // Get Box Volumes
    const float box1_vol = BoxVolume(BoxTris{box1}, box1_center, NUM_TRIS);
    const float box2_vol = BoxVolume(BoxTris{box2}, box2_center, NUM_TRIS);

    // Planes, inside normals and flatness of the faces of each box
    GetFacePlanes(box1, box1_center, box1_faces);
    GetFacePlanes(box2, box2_center, box2_faces);

    // Where a face of each box lies in the same plane, only the face further
    // inside the other box is part of the intersecting polyhedron. It is not
    // clipped by that plane, and the other face is left out. Bit (6 * f + p)
    // of a skip mask is set if face f is not clipped by plane p of the other
    // box, and bit f of a drop mask if face f is left out.
    unsigned long long box1_skip = 0;
    unsigned long long box2_skip = 0;
    unsigned int box1_drop = 0;
    unsigned int box2_drop = 0;
    for (int f1 = 0; f1 < NUM_PLANES; ++f1) {
      const FaceVerts q1 = box1.Face(f1);
      for (int f2 = 0; f2 < NUM_PLANES; ++f2) {
        const FaceVerts q2 = box2.Face(f2);
        if (!IsCoplanarFacePlanes(q1, box1_faces[f1], q2, box2_faces[f2])) {
          continue;
        }
        if (MeanDistance(q1, box2_faces[f2]) >= 0.0f) {
          box1_skip |= 1ull << (NUM_PLANES * f1 + f2);
          box2_drop |= 1u << f2;
        } else {
          box2_skip |= 1ull << (NUM_PLANES * f2 + f1);
          box1_drop |= 1u << f1;
        }
      }
    }
    const auto skip_planes = [](const unsigned long long skip, const int f) {
      return static_cast<unsigned int>(skip >> (NUM_PLANES * f)) &
          ((1u << NUM_PLANES) - 1);
    };

    // The intersecting polyhedron is made of the triangles of each box
    // clipped by the planes of the other box. The volume is summed from a
    // point inside the polyhedron, so the triangles are clipped once to find
    // that point and again to sum the volume instead of being stored.
    FacePoly poly;
    float3 vert_sum = make_float3(0.0f, 0.0f, 0.0f);
    int vert_count = 0;
    for (int t = 0; t < NUM_TRIS; ++t) {
      const int f = _TRI_FACE[t];
      if (!((box1_drop >> f) & 1u)) {
        ClipTriByBox(box1.Tri(t), box2_faces, skip_planes(box1_skip, f), poly);
        for (int v = 0; v < poly.num_verts; ++v) {
          vert_sum = vert_sum + poly.verts[v];
        }
        vert_count += poly.num_verts;
      }
      if (!((box2_drop >> f) & 1u)) {
        ClipTriByBox(box2.Tri(t), box1_faces, skip_planes(box2_skip, f), poly);
        for (int v = 0; v < poly.num_verts; ++v) {
          vert_sum = vert_sum + poly.verts[v];
        }
        vert_count += poly.num_verts;
      }
    }

    // Initialize the vol and iou to 0.0 in case there are no triangles
    // in the intersecting shape.
    float vol = 0.0;
    float iou = 0.0;

    // If there are triangles in the intersecting shape
    if (vert_count > 0) {
      // Calculate the polyhedron center
      const float3 poly_center = vert_sum / vert_count;
      // Compute intersecting polyhedron volume
      for (int t = 0; t < NUM_TRIS; ++t) {
        const int f = _TRI_FACE[t];
        if (!((box1_drop >> f) & 1u)) {
          ClipTriByBox(
              box1.Tri(t), box2_faces, skip_planes(box1_skip, f), poly);
          vol = vol + PolyVolume(poly, poly_center);
        }
        if (!((box2_drop >> f) & 1u)) {
          ClipTriByBox(
              box2.Tri(t), box1_faces, skip_planes(box2_skip, f), poly);
          vol = vol + PolyVolume(poly, poly_center);
        }
      }
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
