/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef PULSAR_NATIVE_INCLUDE_RENDERER_NORM_SPHERE_GRADIENTS_H_
#define PULSAR_NATIVE_INCLUDE_RENDERER_NORM_SPHERE_GRADIENTS_H_

#include "../global.h"
#include "./commands.h"
#include "./math.h"
#include "./renderer.h"

namespace pulsar {
namespace Renderer {

/**
 * Normalize the sphere gradients.
 *
 * We're assuming that the samples originate from a Monte Carlo
 * sampling process and normalize by number and sphere area.
 */
template <bool DEV>
GLOBAL void norm_sphere_gradients(Renderer renderer, const int num_balls) {
  GET_PARALLEL_IDX_1D(idx, num_balls);
  float norm_fac = 0.f;
  IntersectInfo ii;
  if (renderer.ids_sorted_d[idx] > 0) {
    ii = renderer.ii_d[idx];
    // Normalize the sphere gradients as averages.
    // This avoids the case that there are small spheres in a scene with still
    // un-converged colors whereas the big spheres already converged, just
    // because their integrated learning rate is 'higher'.
    norm_fac = FRCP(static_cast<float>(renderer.ids_sorted_d[idx]));
  }
  PULSAR_LOG_DEV_NODE(
      PULSAR_LOG_NORMALIZE,
      "ids_sorted_d[idx]: %d, norm_fac: %.9f.\n",
      renderer.ids_sorted_d[idx],
      norm_fac);
  renderer.grad_rad_d[idx] *= norm_fac;
  for (uint c_idx = 0; c_idx < renderer.cam.n_channels; ++c_idx) {
    renderer.grad_col_d[idx * renderer.cam.n_channels + c_idx] *= norm_fac;
  }
  renderer.grad_pos_d[idx] *= norm_fac;
  renderer.grad_opy_d[idx] *= norm_fac;

  if (renderer.ids_sorted_d[idx] > 0) {
    // For the camera, we need to be more correct and have the gradients
    // be proportional to the area they cover in the image.
    // This leads to a formulation very much like in monte carlo integration:
    norm_fac = FRCP(static_cast<float>(renderer.ids_sorted_d[idx])) *
        (static_cast<float>(ii.max.x) - static_cast<float>(ii.min.x)) *
        (static_cast<float>(ii.max.y) - static_cast<float>(ii.min.y)) *
        1e-3f; // for better numerics.
  }
  renderer.grad_cam_buf_d[idx].cam_pos *= norm_fac;
  renderer.grad_cam_buf_d[idx].pixel_0_0_center *= norm_fac;
  renderer.grad_cam_buf_d[idx].pixel_dir_x *= norm_fac;
  renderer.grad_cam_buf_d[idx].pixel_dir_y *= norm_fac;
  // The sphere only contributes to the camera gradients if it is
  // large enough in screen space.
  if (renderer.ids_sorted_d[idx] > 0 && ii.max.x >= ii.min.x + 3 &&
      ii.max.y >= ii.min.y + 3)
    renderer.ids_sorted_d[idx] = 1;
  END_PARALLEL_NORET();
};

} // namespace Renderer
} // namespace pulsar

#endif
