/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "./camera.h"
#include "../include/math.h"

namespace pulsar {
namespace pytorch {

CamInfo cam_info_from_params(
    const torch::Tensor& cam_pos,
    const torch::Tensor& pixel_0_0_center,
    const torch::Tensor& pixel_vec_x,
    const torch::Tensor& pixel_vec_y,
    const torch::Tensor& principal_point_offset,
    const float& focal_length,
    const uint& width,
    const uint& height,
    const float& min_dist,
    const float& max_dist,
    const bool& right_handed) {
  CamInfo res;
  fill_cam_vecs(
      cam_pos.detach().cpu(),
      pixel_0_0_center.detach().cpu(),
      pixel_vec_x.detach().cpu(),
      pixel_vec_y.detach().cpu(),
      principal_point_offset.detach().cpu(),
      right_handed,
      &res);
  res.half_pixel_size = 0.5f * length(res.pixel_dir_x);
  if (length(res.pixel_dir_y) * 0.5f - res.half_pixel_size > EPS) {
    throw std::runtime_error("Pixel sizes must agree in x and y direction!");
  }
  res.focal_length = focal_length;
  res.aperture_width =
      width + 2u * static_cast<uint>(abs(res.principal_point_offset_x));
  res.aperture_height =
      height + 2u * static_cast<uint>(abs(res.principal_point_offset_y));
  res.pixel_0_0_center -=
      res.pixel_dir_x * static_cast<float>(abs(res.principal_point_offset_x));
  res.pixel_0_0_center -=
      res.pixel_dir_y * static_cast<float>(abs(res.principal_point_offset_y));
  res.film_width = width;
  res.film_height = height;
  res.film_border_left =
      static_cast<uint>(std::max(0, 2 * res.principal_point_offset_x));
  res.film_border_top =
      static_cast<uint>(std::max(0, 2 * res.principal_point_offset_y));
  LOG_IF(INFO, PULSAR_LOG_INIT)
      << "Aperture width, height: " << res.aperture_width << ", "
      << res.aperture_height;
  LOG_IF(INFO, PULSAR_LOG_INIT)
      << "Film width, height: " << res.film_width << ", " << res.film_height;
  LOG_IF(INFO, PULSAR_LOG_INIT)
      << "Film border left, top: " << res.film_border_left << ", "
      << res.film_border_top;
  res.min_dist = min_dist;
  res.max_dist = max_dist;
  res.norm_fac = 1.f / (max_dist - min_dist);
  return res;
};

} // namespace pytorch
} // namespace pulsar
