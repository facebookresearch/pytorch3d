/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef PULSAR_NATIVE_INCLUDE_RENDERER_FILL_BG_DEVICE_H_
#define PULSAR_NATIVE_INCLUDE_RENDERER_FILL_BG_DEVICE_H_

#include "../global.h"
#include "./camera.h"
#include "./commands.h"
#include "./renderer.h"

namespace pulsar {
namespace Renderer {

template <bool DEV>
GLOBAL void fill_bg(
    Renderer renderer,
    const CamInfo cam,
    float const* const bg_col_d,
    const float gamma,
    const uint mode) {
  GET_PARALLEL_IDS_2D(coord_x, coord_y, cam.film_width, cam.film_height);
  int write_loc = coord_y * cam.film_width * (3 + 2 * renderer.n_track) +
      coord_x * (3 + 2 * renderer.n_track);
  if (renderer.forw_info_d[write_loc + 1] // sm_d
      == 0.f) {
    // This location has not been processed yet.
    // Write first the forw_info:
    // sm_m
    renderer.forw_info_d[write_loc] =
        cam.background_normalization_depth / gamma;
    // sm_d
    renderer.forw_info_d[write_loc + 1] = 1.f;
    // max_closest_possible_intersection_hit
    renderer.forw_info_d[write_loc + 2] = -1.f;
    // sphere IDs and intersection depths.
    for (int i = 0; i < renderer.n_track; ++i) {
      int sphere_id = -1;
      IASF(sphere_id, renderer.forw_info_d[write_loc + 3 + i * 2]);
      renderer.forw_info_d[write_loc + 3 + i * 2 + 1] = -1.f;
    }
    if (mode == 0) {
      // Image background.
      for (int i = 0; i < cam.n_channels; ++i) {
        renderer.result_d
            [coord_y * cam.film_width * cam.n_channels +
             coord_x * cam.n_channels + i] = bg_col_d[i];
      }
    }
  }
  END_PARALLEL_2D_NORET();
};

} // namespace Renderer
} // namespace pulsar

#endif
