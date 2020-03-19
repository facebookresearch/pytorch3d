// Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

#include <torch/extension.h>
#include "compositing/alpha_composite.h"
#include "compositing/norm_weighted_sum.h"
#include "compositing/weighted_sum.h"
#include "face_areas_normals/face_areas_normals.h"
#include "gather_scatter/gather_scatter.h"
#include "nearest_neighbor_points/nearest_neighbor_points.h"
#include "packed_to_padded_tensor/packed_to_padded_tensor.h"
#include "rasterize_meshes/rasterize_meshes.h"
#include "rasterize_points/rasterize_points.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("face_areas_normals_forward", &FaceAreasNormalsForward);
  m.def("face_areas_normals_backward", &FaceAreasNormalsBackward);
  m.def("packed_to_padded", &PackedToPadded);
  m.def("padded_to_packed", &PaddedToPacked);
  m.def("nn_points_idx", &NearestNeighborIdx);
  m.def("gather_scatter", &gather_scatter);
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
}
