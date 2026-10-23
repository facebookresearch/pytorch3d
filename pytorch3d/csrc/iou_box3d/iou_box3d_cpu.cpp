/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <torch/extension.h>
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
    std::vector<FacePlane> box1_faces;
    for (const auto& plane : box1_planes) {
      box1_faces.push_back(GetFacePlane(plane, box1_center));
    }

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
      std::vector<FacePlane> box2_faces;
      for (const auto& plane : box2_planes) {
        box2_faces.push_back(GetFacePlane(plane, box2_center));
      }
      const float box2_vol = BoxVolume(box2_tris, box2_center);

      // Where a face of each box lies in the same plane, only the face
      // further inside the other box is part of the intersecting polyhedron.
      // It is not clipped by that plane, and the other face is left out.
      // Bit p of a skip mask is set if face f is not clipped by plane p of
      // the other box, and bit f of a drop mask if face f is left out.
      std::vector<unsigned int> box1_skip(NUM_PLANES, 0);
      std::vector<unsigned int> box2_skip(NUM_PLANES, 0);
      unsigned int box1_drop = 0;
      unsigned int box2_drop = 0;
      for (int f1 = 0; f1 < NUM_PLANES; ++f1) {
        for (int f2 = 0; f2 < NUM_PLANES; ++f2) {
          if (!IsCoplanarFacePlanes(
                  box1_planes[f1],
                  box1_faces[f1],
                  box2_planes[f2],
                  box2_faces[f2])) {
            continue;
          }
          if (MeanDistance(box1_planes[f1], box2_faces[f2]) >= 0.0f) {
            box1_skip[f1] |= 1u << f2;
            box2_drop |= 1u << f2;
          } else {
            box2_skip[f2] |= 1u << f1;
            box1_drop |= 1u << f1;
          }
        }
      }

      // The intersecting polyhedron is made of the triangles of each box
      // clipped by the planes of the other box: each triangle that is fully
      // inside remains as is, one that is fully outside is removed, and one
      // that crosses a plane is cut to the part inside it.
      std::vector<face_poly> polys;
      for (int t = 0; t < NUM_TRIS; ++t) {
        const int f = _TRI_FACE[t];
        if (!((box1_drop >> f) & 1u)) {
          polys.push_back(ClipTriByBox(box1_tris[t], box2_faces, box1_skip[f]));
        }
        if (!((box2_drop >> f) & 1u)) {
          polys.push_back(ClipTriByBox(box2_tris[t], box1_faces, box2_skip[f]));
        }
      }

      // Initialize the vol and iou to 0.0 in case there are no triangles
      // in the intersecting shape.
      float vol = 0.0;
      float iou = 0.0;

      // The polyhedron center as the mean of the polygon vertices
      vec3<float> vert_sum(0.0f, 0.0f, 0.0f);
      int vert_count = 0;
      for (const auto& poly : polys) {
        for (const auto& v : poly) {
          vert_sum = vert_sum + v;
        }
        vert_count += poly.size();
      }

      // If there are triangles in the intersecting shape
      if (vert_count > 0) {
        const vec3<float> polyhedron_center = vert_sum / float(vert_count);
        // Compute intersecting polyhedron volume
        for (const auto& poly : polys) {
          vol = vol + PolyVolume(poly, polyhedron_center);
        }
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
