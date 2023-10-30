/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef PULSAR_NATIVE_CAMERA_H_
#define PULSAR_NATIVE_CAMERA_H_

#include <tuple>
#include "../global.h"

#include "../include/camera.h"

namespace pulsar {
namespace pytorch {

inline void fill_cam_vecs(
    const torch::Tensor& pos_vec,
    const torch::Tensor& pixel_0_0_center,
    const torch::Tensor& pixel_dir_x,
    const torch::Tensor& pixel_dir_y,
    const torch::Tensor& principal_point_offset,
    const bool& right_handed,
    CamInfo* res) {
  res->eye.x = pos_vec.data_ptr<float>()[0];
  res->eye.y = pos_vec.data_ptr<float>()[1];
  res->eye.z = pos_vec.data_ptr<float>()[2];
  res->pixel_0_0_center.x = pixel_0_0_center.data_ptr<float>()[0];
  res->pixel_0_0_center.y = pixel_0_0_center.data_ptr<float>()[1];
  res->pixel_0_0_center.z = pixel_0_0_center.data_ptr<float>()[2];
  res->pixel_dir_x.x = pixel_dir_x.data_ptr<float>()[0];
  res->pixel_dir_x.y = pixel_dir_x.data_ptr<float>()[1];
  res->pixel_dir_x.z = pixel_dir_x.data_ptr<float>()[2];
  res->pixel_dir_y.x = pixel_dir_y.data_ptr<float>()[0];
  res->pixel_dir_y.y = pixel_dir_y.data_ptr<float>()[1];
  res->pixel_dir_y.z = pixel_dir_y.data_ptr<float>()[2];
  auto sensor_dir_z = pixel_dir_y.cross(pixel_dir_x, -1);
  sensor_dir_z /= sensor_dir_z.norm();
  if (right_handed) {
    sensor_dir_z *= -1.f;
  }
  res->sensor_dir_z.x = sensor_dir_z.data_ptr<float>()[0];
  res->sensor_dir_z.y = sensor_dir_z.data_ptr<float>()[1];
  res->sensor_dir_z.z = sensor_dir_z.data_ptr<float>()[2];
  res->principal_point_offset_x = principal_point_offset.data_ptr<int32_t>()[0];
  res->principal_point_offset_y = principal_point_offset.data_ptr<int32_t>()[1];
}

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
    const bool& right_handed);

} // namespace pytorch
} // namespace pulsar

#endif
