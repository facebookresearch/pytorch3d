/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef PULSAR_NATIVE_INCLUDE_RENDERER_FORWARD_DEVICE_H_
#define PULSAR_NATIVE_INCLUDE_RENDERER_FORWARD_DEVICE_H_

#include "../global.h"
#include "./camera.device.h"
#include "./commands.h"
#include "./math.h"
#include "./renderer.h"

namespace pulsar {
namespace Renderer {

template <bool DEV>
void forward(
    Renderer* self,
    const float* vert_pos,
    const float* vert_col,
    const float* vert_rad,
    const CamInfo& cam,
    const float& gamma,
    float percent_allowed_difference,
    const uint& max_n_hits,
    const float* bg_col_d,
    const float* opacity_d,
    const size_t& num_balls,
    const uint& mode,
    cudaStream_t stream) {
  ARGCHECK(gamma > 0.f && gamma <= 1.f, 6, "gamma must be in [0., 1.]");
  ARGCHECK(
      percent_allowed_difference >= 0.f && percent_allowed_difference <= 1.f,
      7,
      "percent_allowed_difference must be in [0., 1.]");
  ARGCHECK(max_n_hits >= 1u, 8, "max_n_hits must be >= 1");
  ARGCHECK(
      num_balls > 0 && num_balls <= self->max_num_balls,
      9,
      ("num_balls must be >0 and <= max num balls! (" +
       std::to_string(num_balls) + " vs. " +
       std::to_string(self->max_num_balls) + ")")
          .c_str());
  ARGCHECK(
      cam.film_width == self->cam.film_width &&
          cam.film_height == self->cam.film_height,
      5,
      "cam result width and height must agree");
  ARGCHECK(mode <= 1, 10, "mode must be <= 1!");
  if (percent_allowed_difference > 1.f - FEPS) {
    LOG(WARNING) << "percent_allowed_difference > " << (1.f - FEPS)
                 << "! Clamping to " << (1.f - FEPS) << ".";
    percent_allowed_difference = 1.f - FEPS;
  }
  LOG_IF(INFO, PULSAR_LOG_RENDER) << "Rendering forward pass...";
  // Update camera and transform into a new virtual camera system with
  // centered principal point and subsection rendering.
  self->cam.eye = cam.eye;
  self->cam.pixel_0_0_center = cam.pixel_0_0_center - cam.eye;
  self->cam.pixel_dir_x = cam.pixel_dir_x;
  self->cam.pixel_dir_y = cam.pixel_dir_y;
  self->cam.sensor_dir_z = cam.sensor_dir_z;
  self->cam.half_pixel_size = cam.half_pixel_size;
  self->cam.focal_length = cam.focal_length;
  self->cam.aperture_width = cam.aperture_width;
  self->cam.aperture_height = cam.aperture_height;
  self->cam.min_dist = cam.min_dist;
  self->cam.max_dist = cam.max_dist;
  self->cam.norm_fac = cam.norm_fac;
  self->cam.principal_point_offset_x = cam.principal_point_offset_x;
  self->cam.principal_point_offset_y = cam.principal_point_offset_y;
  self->cam.film_border_left = cam.film_border_left;
  self->cam.film_border_top = cam.film_border_top;
#ifdef PULSAR_TIMINGS_ENABLED
  START_TIME(calc_signature);
#endif
  LAUNCH_MAX_PARALLEL_1D(
      calc_signature<DEV>,
      num_balls,
      stream,
      *self,
      reinterpret_cast<const float3*>(vert_pos),
      vert_col,
      vert_rad,
      num_balls);
  CHECKLAUNCH();
#ifdef PULSAR_TIMINGS_ENABLED
  STOP_TIME(calc_signature);
  START_TIME(sort);
#endif
  SORT_ASCENDING_WS(
      self->min_depth_d,
      self->min_depth_sorted_d,
      self->ids_d,
      self->ids_sorted_d,
      num_balls,
      self->workspace_d,
      self->workspace_size,
      stream);
  CHECKLAUNCH();
  SORT_ASCENDING_WS(
      self->min_depth_d,
      self->min_depth_sorted_d,
      self->ii_d,
      self->ii_sorted_d,
      num_balls,
      self->workspace_d,
      self->workspace_size,
      stream);
  CHECKLAUNCH();
  SORT_ASCENDING_WS(
      self->min_depth_d,
      self->min_depth_sorted_d,
      self->di_d,
      self->di_sorted_d,
      num_balls,
      self->workspace_d,
      self->workspace_size,
      stream);
  CHECKLAUNCH();
#ifdef PULSAR_TIMINGS_ENABLED
  STOP_TIME(sort);
  START_TIME(minmax);
#endif
  IntersectInfo pixel_minmax;
  pixel_minmax.min.x = MAX_USHORT;
  pixel_minmax.min.y = MAX_USHORT;
  pixel_minmax.max.x = 0;
  pixel_minmax.max.y = 0;
  REDUCE_WS(
      self->ii_sorted_d,
      self->min_max_pixels_d,
      num_balls,
      IntersectInfoMinMax(),
      pixel_minmax,
      self->workspace_d,
      self->workspace_size,
      stream);
  COPY_DEV_HOST(&pixel_minmax, self->min_max_pixels_d, IntersectInfo, 1);
  LOG_IF(INFO, PULSAR_LOG_RENDER)
      << "Region with pixels to render: " << pixel_minmax.min.x << ":"
      << pixel_minmax.max.x << " (x), " << pixel_minmax.min.y << ":"
      << pixel_minmax.max.y << " (y).";
#ifdef PULSAR_TIMINGS_ENABLED
  STOP_TIME(minmax);
  START_TIME(render);
#endif
  MEMSET(
      self->result_d,
      0,
      float,
      self->cam.film_width * self->cam.film_height * self->cam.n_channels,
      stream);
  MEMSET(
      self->forw_info_d,
      0,
      float,
      self->cam.film_width * self->cam.film_height * (3 + 2 * self->n_track),
      stream);
  if (pixel_minmax.max.y > pixel_minmax.min.y &&
      pixel_minmax.max.x > pixel_minmax.min.x) {
    PASSERT(
        pixel_minmax.min.x >= static_cast<ushort>(self->cam.film_border_left) &&
        pixel_minmax.min.x <
            static_cast<ushort>(
                self->cam.film_border_left + self->cam.film_width) &&
        pixel_minmax.max.x <=
            static_cast<ushort>(
                self->cam.film_border_left + self->cam.film_width) &&
        pixel_minmax.min.y >= static_cast<ushort>(self->cam.film_border_top) &&
        pixel_minmax.min.y <
            static_cast<ushort>(
                self->cam.film_border_top + self->cam.film_height) &&
        pixel_minmax.max.y <=
            static_cast<ushort>(
                self->cam.film_border_top + self->cam.film_height));
    // Cut the image in 3x3 regions.
    int y_step = RENDER_BLOCK_SIZE *
        iDivCeil(pixel_minmax.max.y - pixel_minmax.min.y,
                 3u * RENDER_BLOCK_SIZE);
    int x_step = RENDER_BLOCK_SIZE *
        iDivCeil(pixel_minmax.max.x - pixel_minmax.min.x,
                 3u * RENDER_BLOCK_SIZE);
    LOG_IF(INFO, PULSAR_LOG_RENDER) << "Using image slices of size " << x_step
                                    << ", " << y_step << " (W, H).";
    for (int y_min = pixel_minmax.min.y; y_min < pixel_minmax.max.y;
         y_min += y_step) {
      for (int x_min = pixel_minmax.min.x; x_min < pixel_minmax.max.x;
           x_min += x_step) {
        // Create region selection.
        LAUNCH_MAX_PARALLEL_1D(
            create_selector<DEV>,
            num_balls,
            stream,
            self->ii_sorted_d,
            num_balls,
            x_min,
            x_min + x_step,
            y_min,
            y_min + y_step,
            self->region_flags_d);
        CHECKLAUNCH();
        SELECT_FLAGS_WS(
            self->region_flags_d,
            self->ii_sorted_d,
            self->ii_d,
            self->num_selected_d,
            num_balls,
            self->workspace_d,
            self->workspace_size,
            stream);
        CHECKLAUNCH();
        SELECT_FLAGS_WS(
            self->region_flags_d,
            self->di_sorted_d,
            self->di_d,
            self->num_selected_d,
            num_balls,
            self->workspace_d,
            self->workspace_size,
            stream);
        CHECKLAUNCH();
        SELECT_FLAGS_WS(
            self->region_flags_d,
            self->ids_sorted_d,
            self->ids_d,
            self->num_selected_d,
            num_balls,
            self->workspace_d,
            self->workspace_size,
            stream);
        CHECKLAUNCH();
        LAUNCH_PARALLEL_2D(
            render<DEV>,
            x_step,
            y_step,
            RENDER_BLOCK_SIZE,
            RENDER_BLOCK_SIZE,
            stream,
            self->num_selected_d,
            self->ii_d,
            self->di_d,
            self->min_depth_d,
            self->ids_d,
            opacity_d,
            self->cam,
            gamma,
            percent_allowed_difference,
            max_n_hits,
            bg_col_d,
            mode,
            x_min,
            y_min,
            x_step,
            y_step,
            self->result_d,
            self->forw_info_d,
            self->n_track);
        CHECKLAUNCH();
      }
    }
  }
  if (mode == 0) {
    LAUNCH_MAX_PARALLEL_2D(
        fill_bg<DEV>,
        static_cast<int64_t>(self->cam.film_width),
        static_cast<int64_t>(self->cam.film_height),
        stream,
        *self,
        self->cam,
        bg_col_d,
        gamma,
        mode);
    CHECKLAUNCH();
  }
#ifdef PULSAR_TIMINGS_ENABLED
  STOP_TIME(render);
  float time_ms;
  // This blocks the result and prevents batch-processing from parallelizing.
  GET_TIME(calc_signature, &time_ms);
  std::cout << "Time for signature calculation: " << time_ms << " ms"
            << std::endl;
  GET_TIME(sort, &time_ms);
  std::cout << "Time for sorting: " << time_ms << " ms" << std::endl;
  GET_TIME(minmax, &time_ms);
  std::cout << "Time for minmax pixel calculation: " << time_ms << " ms"
            << std::endl;
  GET_TIME(render, &time_ms);
  std::cout << "Time for rendering: " << time_ms << " ms" << std::endl;
#endif
  LOG_IF(INFO, PULSAR_LOG_RENDER) << "Forward pass complete.";
}

} // namespace Renderer
} // namespace pulsar

#endif
