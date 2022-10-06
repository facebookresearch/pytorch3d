/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// clang-format off
#include "./pulsar/global.h" // Include before <torch/extension.h>.
#include <torch/extension.h>
// clang-format on
#include "./pulsar/pytorch/renderer.h"
#include "./pulsar/pytorch/tensor_util.h"
#include "ball_query/ball_query.h"
#include "blending/sigmoid_alpha_blend.h"
#include "compositing/alpha_composite.h"
#include "compositing/norm_weighted_sum.h"
#include "compositing/weighted_sum.h"
#include "face_areas_normals/face_areas_normals.h"
#include "gather_scatter/gather_scatter.h"
#include "interp_face_attrs/interp_face_attrs.h"
#include "iou_box3d/iou_box3d.h"
#include "knn/knn.h"
#include "marching_cubes/marching_cubes.h"
#include "mesh_normal_consistency/mesh_normal_consistency.h"
#include "packed_to_padded_tensor/packed_to_padded_tensor.h"
#include "point_mesh/point_mesh_cuda.h"
#include "points_to_volumes/points_to_volumes.h"
#include "rasterize_meshes/rasterize_meshes.h"
#include "rasterize_points/rasterize_points.h"
#include "sample_farthest_points/sample_farthest_points.h"
#include "sample_pdf/sample_pdf.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("face_areas_normals_forward", &FaceAreasNormalsForward);
  m.def("face_areas_normals_backward", &FaceAreasNormalsBackward);
  m.def("packed_to_padded", &PackedToPadded);
  m.def("padded_to_packed", &PaddedToPacked);
  m.def("interp_face_attrs_forward", &InterpFaceAttrsForward);
  m.def("interp_face_attrs_backward", &InterpFaceAttrsBackward);
#ifdef WITH_CUDA
  m.def("knn_check_version", &KnnCheckVersion);
#endif
  m.def("knn_points_idx", &KNearestNeighborIdx);
  m.def("knn_points_backward", &KNearestNeighborBackward);
  m.def("ball_query", &BallQuery);
  m.def("sample_farthest_points", &FarthestPointSampling);
  m.def(
      "mesh_normal_consistency_find_verts", &MeshNormalConsistencyFindVertices);
  m.def("gather_scatter", &GatherScatter);
  m.def("points_to_volumes_forward", PointsToVolumesForward);
  m.def("points_to_volumes_backward", PointsToVolumesBackward);
  m.def("rasterize_points", &RasterizePoints);
  m.def("rasterize_points_backward", &RasterizePointsBackward);
  m.def("rasterize_meshes_backward", &RasterizeMeshesBackward);
  m.def("rasterize_meshes", &RasterizeMeshes);
  m.def("sigmoid_alpha_blend", &SigmoidAlphaBlend);
  m.def("sigmoid_alpha_blend_backward", &SigmoidAlphaBlendBackward);

  // Accumulation functions
  m.def("accum_weightedsumnorm", &weightedSumNormForward);
  m.def("accum_weightedsum", &weightedSumForward);
  m.def("accum_alphacomposite", &alphaCompositeForward);
  m.def("accum_weightedsumnorm_backward", &weightedSumNormBackward);
  m.def("accum_weightedsum_backward", &weightedSumBackward);
  m.def("accum_alphacomposite_backward", &alphaCompositeBackward);

  // These are only visible for testing; users should not call them directly
  m.def("_rasterize_points_coarse", &RasterizePointsCoarse);
  m.def("_rasterize_points_naive", &RasterizePointsNaive);
  m.def("_rasterize_meshes_naive", &RasterizeMeshesNaive);
  m.def("_rasterize_meshes_coarse", &RasterizeMeshesCoarse);
  m.def("_rasterize_meshes_fine", &RasterizeMeshesFine);

  // PointEdge distance functions
  m.def("point_edge_dist_forward", &PointEdgeDistanceForward);
  m.def("point_edge_dist_backward", &PointEdgeDistanceBackward);
  m.def("edge_point_dist_forward", &EdgePointDistanceForward);
  m.def("edge_point_dist_backward", &EdgePointDistanceBackward);
  m.def("point_edge_array_dist_forward", &PointEdgeArrayDistanceForward);
  m.def("point_edge_array_dist_backward", &PointEdgeArrayDistanceBackward);

  // PointFace distance functions
  m.def("point_face_dist_forward", &PointFaceDistanceForward);
  m.def("point_face_dist_backward", &PointFaceDistanceBackward);
  m.def("face_point_dist_forward", &FacePointDistanceForward);
  m.def("face_point_dist_backward", &FacePointDistanceBackward);
  m.def("point_face_array_dist_forward", &PointFaceArrayDistanceForward);
  m.def("point_face_array_dist_backward", &PointFaceArrayDistanceBackward);

  // Sample PDF
  m.def("sample_pdf", &SamplePdf);

  // 3D IoU
  m.def("iou_box3d", &IoUBox3D);

  // Marching cubes
  m.def("marching_cubes", &MarchingCubes);

  // Pulsar.
#ifdef PULSAR_LOGGING_ENABLED
  c10::ShowLogInfoToStderr();
#endif
  py::class_<
      pulsar::pytorch::Renderer,
      std::shared_ptr<pulsar::pytorch::Renderer>>(m, "PulsarRenderer")
      .def(py::init<
           const uint&,
           const uint&,
           const uint&,
           const bool&,
           const bool&,
           const float&,
           const uint&,
           const uint&>())
      .def(
          "__eq__",
          [](const pulsar::pytorch::Renderer& a,
             const pulsar::pytorch::Renderer& b) { return a == b; },
          py::is_operator())
      .def(
          "__ne__",
          [](const pulsar::pytorch::Renderer& a,
             const pulsar::pytorch::Renderer& b) { return !(a == b); },
          py::is_operator())
      .def(
          "__repr__",
          [](const pulsar::pytorch::Renderer& self) {
            std::stringstream ss;
            ss << self;
            return ss.str();
          })
      .def(
          "forward",
          &pulsar::pytorch::Renderer::forward,
          py::arg("vert_pos"),
          py::arg("vert_col"),
          py::arg("vert_radii"),

          py::arg("cam_pos"),
          py::arg("pixel_0_0_center"),
          py::arg("pixel_vec_x"),
          py::arg("pixel_vec_y"),
          py::arg("focal_length"),
          py::arg("principal_point_offsets"),

          py::arg("gamma"),
          py::arg("max_depth"),
          py::arg("min_depth") /* = 0.f*/,
          py::arg(
              "bg_col") /* = at::nullopt not exposed properly in pytorch 1.1. */
          ,
          py::arg("opacity") /* = at::nullopt ... */,
          py::arg("percent_allowed_difference") = 0.01f,
          py::arg("max_n_hits") = MAX_UINT,
          py::arg("mode") = 0)
      .def("backward", &pulsar::pytorch::Renderer::backward)
      .def_property(
          "device_tracker",
          [](const pulsar::pytorch::Renderer& self) {
            return self.device_tracker;
          },
          [](pulsar::pytorch::Renderer& self, const torch::Tensor& val) {
            self.device_tracker = val;
          })
      .def_property_readonly("width", &pulsar::pytorch::Renderer::width)
      .def_property_readonly("height", &pulsar::pytorch::Renderer::height)
      .def_property_readonly(
          "max_num_balls", &pulsar::pytorch::Renderer::max_num_balls)
      .def_property_readonly(
          "orthogonal", &pulsar::pytorch::Renderer::orthogonal)
      .def_property_readonly(
          "right_handed", &pulsar::pytorch::Renderer::right_handed)
      .def_property_readonly("n_track", &pulsar::pytorch::Renderer::n_track);
  m.def(
      "pulsar_sphere_ids_from_result_info_nograd",
      &pulsar::pytorch::sphere_ids_from_result_info_nograd);
  // Constants.
  m.attr("EPS") = py::float_(EPS);
  m.attr("MAX_FLOAT") = py::float_(MAX_FLOAT);
  m.attr("MAX_INT") = py::int_(MAX_INT);
  m.attr("MAX_UINT") = py::int_(MAX_UINT);
  m.attr("MAX_USHORT") = py::int_(MAX_USHORT);
  m.attr("PULSAR_MAX_GRAD_SPHERES") = py::int_(MAX_GRAD_SPHERES);
}
