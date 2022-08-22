/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <torch/extension.h>
#include <torch/torch.h>
#include <list>
#include <numeric>
#include <queue>
#include <tuple>
#include "iou_box3d/iou_utils.h"

std::tuple<at::Tensor, at::Tensor> IoUBox3DCpu(
    const at::Tensor& boxes1,
    const at::Tensor& boxes2) {
  const int N = boxes1.size(0);
  const int M = boxes2.size(0);
  auto float_opts = boxes1.options().dtype(torch::kFloat32);
  torch::Tensor vols = torch::zeros({N, M}, float_opts);
  torch::Tensor ious = torch::zeros({N, M}, float_opts);

  // Create tensor accessors
  auto boxes1_a = boxes1.accessor<float, 3>();
  auto boxes2_a = boxes2.accessor<float, 3>();
  auto vols_a = vols.accessor<float, 2>();
  auto ious_a = ious.accessor<float, 2>();

  // Iterate through the N boxes in boxes1
  for (int n = 0; n < N; ++n) {
    const auto& box1 = boxes1_a[n];
    // Convert to vector of face vertices i.e. effectively (F, 3, 3)
    // face_verts is a data type defined in iou_utils.h
    const face_verts box1_tris = GetBoxTris(box1);

    // Calculate the position of the center of the box which is used in
    // several calculations. This requires a tensor as input.
    const vec3<float> box1_center = BoxCenter(boxes1[n]);

    // Convert to vector of face vertices i.e. effectively (P, 4, 3)
    const face_verts box1_planes = GetBoxPlanes(box1);

    // Get Box Volumes
    const float box1_vol = BoxVolume(box1_tris, box1_center);

    // Iterate through the M boxes in boxes2
    for (int m = 0; m < M; ++m) {
      // Repeat above steps for box2
      // TODO: check if caching these value helps performance.
      const auto& box2 = boxes2_a[m];
      const face_verts box2_tris = GetBoxTris(box2);
      const vec3<float> box2_center = BoxCenter(boxes2[m]);
      const face_verts box2_planes = GetBoxPlanes(box2);
      const float box2_vol = BoxVolume(box2_tris, box2_center);

      // Every triangle in one box will be compared to each plane in the other
      // box. There are 3 possible outcomes:
      // 1. If the triangle is fully inside, then it will
      //    remain as is.
      // 2. If the triagnle it is fully outside, it will be removed.
      // 3. If the triangle intersects with the (infinite) plane, it
      //    will be broken into subtriangles such that each subtriangle is full
      //    inside the plane and part of the intersecting tetrahedron.

      // Tris in Box1 -> Planes in Box2
      face_verts box1_intersect =
          BoxIntersections(box1_tris, box2_planes, box2_center);
      // Tris in Box2 -> Planes in Box1
      face_verts box2_intersect =
          BoxIntersections(box2_tris, box1_planes, box1_center);

      // If there are overlapping regions in Box2, remove any coplanar faces
      if (box2_intersect.size() > 0) {
        // Identify if any triangles in Box2 are coplanar with Box1
        std::vector<int> tri2_keep(box2_intersect.size());
        std::fill(tri2_keep.begin(), tri2_keep.end(), 1);
        for (int b1 = 0; b1 < box1_intersect.size(); ++b1) {
          for (int b2 = 0; b2 < box2_intersect.size(); ++b2) {
            const bool is_coplanar =
                IsCoplanarTriTri(box1_intersect[b1], box2_intersect[b2]);
            const float area = FaceArea(box1_intersect[b1]);
            if ((is_coplanar) && (area > aEpsilon)) {
              tri2_keep[b2] = 0;
            }
          }
        }

        // Keep only the non coplanar triangles in Box2 - add them to the
        // Box1 triangles.
        for (int b2 = 0; b2 < box2_intersect.size(); ++b2) {
          if (tri2_keep[b2] == 1) {
            box1_intersect.push_back((box2_intersect[b2]));
          }
        }
      }

      // Initialize the vol and iou to 0.0 in case there are no triangles
      // in the intersecting shape.
      float vol = 0.0;
      float iou = 0.0;

      // If there are triangles in the intersecting shape
      if (box1_intersect.size() > 0) {
        // The intersecting shape is a polyhedron made up of the
        // triangular faces that are all now in box1_intersect.
        // Calculate the polyhedron center
        const vec3<float> polyhedron_center = PolyhedronCenter(box1_intersect);
        // Compute intersecting polyhedron volume
        vol = BoxVolume(box1_intersect, polyhedron_center);
        // Compute IoU
        iou = vol / (box1_vol + box2_vol - vol);
      }
      // Save out volume and IoU
      vols_a[n][m] = vol;
      ious_a[n][m] = iou;
    }
  }
  return std::make_tuple(vols, ious);
}
