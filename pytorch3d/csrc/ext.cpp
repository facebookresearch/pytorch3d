// Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

#include <torch/extension.h>
#include "compositing/alpha_composite.h"
#include "compositing/norm_weighted_sum.h"
#include "compositing/weighted_sum.h"
#include "face_areas_normals/face_areas_normals.h"
#include "gather_scatter/gather_scatter.h"
#include "knn/knn.h"
#include "packed_to_padded_tensor/packed_to_padded_tensor.h"
#include "point_mesh/point_mesh_edge.h"
#include "point_mesh/point_mesh_face.h"
#include "rasterize_meshes/rasterize_meshes.h"
#include "rasterize_points/rasterize_points.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("face_areas_normals_forward", &FaceAreasNormalsForward);
  m.def("face_areas_normals_backward", &FaceAreasNormalsBackward);
  m.def("packed_to_padded", &PackedToPadded);
  m.def("padded_to_packed", &PaddedToPacked);
#ifdef WITH_CUDA
  m.def("knn_check_version", &KnnCheckVersion);
#endif
  m.def("knn_points_idx", &KNearestNeighborIdx);
  m.def("knn_points_backward", &KNearestNeighborBackward);
  m.def("gather_scatter", &GatherScatter);
  m.def("rasterize_points", &RasterizePoints);
  m.def("rasterize_points_backward", &RasterizePointsBackward);
  m.def("rasterize_meshes_backward", &RasterizeMeshesBackward);
  m.def("rasterize_meshes", &RasterizeMeshes);

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
}
