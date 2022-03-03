/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef PULSAR_NATIVE_INCLUDE_RENDERER_CONSTRUCT_DEVICE_H_
#define PULSAR_NATIVE_INCLUDE_RENDERER_CONSTRUCT_DEVICE_H_

#include "../global.h"
#include "./camera.device.h"
#include "./commands.h"
#include "./math.h"
#include "./renderer.h"

namespace pulsar {
namespace Renderer {

template <bool DEV>
HOST void construct(
    Renderer* self,
    const size_t& max_num_balls,
    const int& width,
    const int& height,
    const bool& orthogonal_projection,
    const bool& right_handed_system,
    const float& background_normalization_depth,
    const uint& n_channels,
    const uint& n_track) {
  ARGCHECK(
      (max_num_balls > 0 && max_num_balls < MAX_INT),
      2,
      ("the maximum number of balls must be >0 and <" +
       std::to_string(MAX_INT) + ". Is " + std::to_string(max_num_balls) + ".")
          .c_str());
  ARGCHECK(width > 1, 3, "the image width must be > 1");
  ARGCHECK(height > 1, 4, "the image height must be > 1");
  ARGCHECK(
      background_normalization_depth > 0.f &&
          background_normalization_depth < 1.f,
      6,
      "background_normalization_depth must be in ]0., 1.[.");
  ARGCHECK(n_channels > 0, 7, "n_channels must be >0!");
  ARGCHECK(
      n_track > 0 && n_track <= MAX_GRAD_SPHERES,
      8,
      ("n_track must be >0 and <" + std::to_string(MAX_GRAD_SPHERES) + ". Is " +
       std::to_string(n_track) + ".")
          .c_str());
  self->cam.film_width = width;
  self->cam.film_height = height;
  self->max_num_balls = max_num_balls;
  MALLOC(self->result_d, float, width* height* n_channels);
  self->cam.orthogonal_projection = orthogonal_projection;
  self->cam.right_handed = right_handed_system;
  self->cam.background_normalization_depth = background_normalization_depth;
  self->cam.n_channels = n_channels;
  MALLOC(self->min_depth_d, float, max_num_balls);
  MALLOC(self->min_depth_sorted_d, float, max_num_balls);
  MALLOC(self->ii_d, IntersectInfo, max_num_balls);
  MALLOC(self->ii_sorted_d, IntersectInfo, max_num_balls);
  MALLOC(self->ids_d, int, max_num_balls);
  MALLOC(self->ids_sorted_d, int, max_num_balls);
  size_t sort_id_size = 0;
  GET_SORT_WS_SIZE(&sort_id_size, float, int, max_num_balls);
  CHECKLAUNCH();
  size_t sort_ii_size = 0;
  GET_SORT_WS_SIZE(&sort_ii_size, float, IntersectInfo, max_num_balls);
  CHECKLAUNCH();
  size_t sort_di_size = 0;
  GET_SORT_WS_SIZE(&sort_di_size, float, DrawInfo, max_num_balls);
  CHECKLAUNCH();
  size_t select_ii_size = 0;
  GET_SELECT_WS_SIZE(&select_ii_size, char, IntersectInfo, max_num_balls);
  size_t select_di_size = 0;
  GET_SELECT_WS_SIZE(&select_di_size, char, DrawInfo, max_num_balls);
  size_t sum_size = 0;
  GET_SUM_WS_SIZE(&sum_size, CamGradInfo, max_num_balls);
  size_t sum_cont_size = 0;
  GET_SUM_WS_SIZE(&sum_cont_size, int, max_num_balls);
  size_t reduce_size = 0;
  GET_REDUCE_WS_SIZE(
      &reduce_size, IntersectInfo, IntersectInfoMinMax(), max_num_balls);
  self->workspace_size = IMAX(
      IMAX(IMAX(sort_id_size, sort_ii_size), sort_di_size),
      IMAX(
          IMAX(select_di_size, select_ii_size),
          IMAX(IMAX(sum_size, sum_cont_size), reduce_size)));
  MALLOC(self->workspace_d, char, self->workspace_size);
  MALLOC(self->di_d, DrawInfo, max_num_balls);
  MALLOC(self->di_sorted_d, DrawInfo, max_num_balls);
  MALLOC(self->region_flags_d, char, max_num_balls);
  MALLOC(self->num_selected_d, size_t, 1);
  MALLOC(self->forw_info_d, float, width* height*(3 + 2 * n_track));
  MALLOC(self->min_max_pixels_d, IntersectInfo, 1);
  MALLOC(self->grad_pos_d, float3, max_num_balls);
  MALLOC(self->grad_col_d, float, max_num_balls* n_channels);
  MALLOC(self->grad_rad_d, float, max_num_balls);
  MALLOC(self->grad_cam_d, float, 12);
  MALLOC(self->grad_cam_buf_d, CamGradInfo, max_num_balls);
  MALLOC(self->grad_opy_d, float, max_num_balls);
  MALLOC(self->n_grad_contributions_d, int, 1);
  self->n_track = static_cast<int>(n_track);
}

} // namespace Renderer
} // namespace pulsar

#endif
