/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef PULSAR_NATIVE_RENDERER_BACKWARD_DEVICE_H_
#define PULSAR_NATIVE_RENDERER_BACKWARD_DEVICE_H_

#include "./camera.device.h"
#include "./math.h"
#include "./renderer.h"

namespace pulsar {
namespace Renderer {

template <bool DEV>
void backward(
    Renderer* self,
    const float* grad_im,
    const float* image,
    const float* forw_info,
    const float* vert_pos,
    const float* vert_col,
    const float* vert_rad,
    const CamInfo& cam,
    const float& gamma,
    float percent_allowed_difference,
    const uint& max_n_hits,
    const float* vert_opy_d,
    const size_t& num_balls,
    const uint& mode,
    const bool& dif_pos,
    const bool& dif_col,
    const bool& dif_rad,
    const bool& dif_cam,
    const bool& dif_opy,
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
      "num_balls must be >0 and less than max num balls!");
  ARGCHECK(
      cam.film_width == self->cam.film_width &&
          cam.film_height == self->cam.film_height,
      5,
      "cam film size must agree");
  ARGCHECK(mode <= 1, 10, "mode must be <= 1!");
  if (percent_allowed_difference < EPS) {
    LOG(WARNING) << "percent_allowed_difference < " << FEPS << "! Clamping to "
                 << FEPS << ".";
    percent_allowed_difference = FEPS;
  }
  if (percent_allowed_difference > 1.f - FEPS) {
    LOG(WARNING) << "percent_allowed_difference > " << (1.f - FEPS)
                 << "! Clamping to " << (1.f - FEPS) << ".";
    percent_allowed_difference = 1.f - FEPS;
  }
  LOG_IF(INFO, PULSAR_LOG_RENDER) << "Rendering backward pass...";
  // Update camera.
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
  START_TIME(calc_gradients);
#endif
  MEMSET(self->grad_pos_d, 0, float3, num_balls, stream);
  MEMSET(self->grad_col_d, 0, float, num_balls * self->cam.n_channels, stream);
  MEMSET(self->grad_rad_d, 0, float, num_balls, stream);
  MEMSET(self->grad_cam_d, 0, float, 12, stream);
  MEMSET(self->grad_cam_buf_d, 0, CamGradInfo, num_balls, stream);
  MEMSET(self->grad_opy_d, 0, float, num_balls, stream);
  MEMSET(self->ids_sorted_d, 0, int, num_balls, stream);
  LAUNCH_PARALLEL_2D(
      calc_gradients<DEV>,
      self->cam.film_width,
      self->cam.film_height,
      GRAD_BLOCK_SIZE,
      GRAD_BLOCK_SIZE,
      stream,
      self->cam,
      grad_im,
      gamma,
      reinterpret_cast<const float3*>(vert_pos),
      vert_col,
      vert_rad,
      vert_opy_d,
      num_balls,
      image,
      forw_info,
      self->di_d,
      self->ii_d,
      dif_pos,
      dif_col,
      dif_rad,
      dif_cam,
      dif_opy,
      self->grad_rad_d,
      self->grad_col_d,
      self->grad_pos_d,
      self->grad_cam_buf_d,
      self->grad_opy_d,
      self->ids_sorted_d,
      self->n_track);
  CHECKLAUNCH();
#ifdef PULSAR_TIMINGS_ENABLED
  STOP_TIME(calc_gradients);
  START_TIME(normalize);
#endif
  LAUNCH_MAX_PARALLEL_1D(
      norm_sphere_gradients<DEV>, num_balls, stream, *self, num_balls);
  CHECKLAUNCH();
  if (dif_cam) {
    SUM_WS(
        self->grad_cam_buf_d,
        reinterpret_cast<CamGradInfo*>(self->grad_cam_d),
        static_cast<int>(num_balls),
        self->workspace_d,
        self->workspace_size,
        stream);
    CHECKLAUNCH();
    SUM_WS(
        (IntWrapper*)(self->ids_sorted_d),
        (IntWrapper*)(self->n_grad_contributions_d),
        static_cast<int>(num_balls),
        self->workspace_d,
        self->workspace_size,
        stream);
    CHECKLAUNCH();
    LAUNCH_MAX_PARALLEL_1D(
        norm_cam_gradients<DEV>, static_cast<int64_t>(1), stream, *self);
    CHECKLAUNCH();
  }
#ifdef PULSAR_TIMINGS_ENABLED
  STOP_TIME(normalize);
  float time_ms;
  // This blocks the result and prevents batch-processing from parallelizing.
  GET_TIME(calc_signature, &time_ms);
  std::cout << "Time for signature calculation: " << time_ms << " ms"
            << std::endl;
  GET_TIME(calc_gradients, &time_ms);
  std::cout << "Time for gradient calculation: " << time_ms << " ms"
            << std::endl;
  GET_TIME(normalize, &time_ms);
  std::cout << "Time for aggregation and normalization: " << time_ms << " ms"
            << std::endl;
#endif
  LOG_IF(INFO, PULSAR_LOG_RENDER) << "Backward pass complete.";
}

} // namespace Renderer
} // namespace pulsar

#endif
