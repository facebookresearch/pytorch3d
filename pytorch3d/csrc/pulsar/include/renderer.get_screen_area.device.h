/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef PULSAR_NATIVE_INCLUDE_RENDERER_GET_SCREEN_AREA_DEVICE_H_
#define PULSAR_NATIVE_INCLUDE_RENDERER_GET_SCREEN_AREA_DEVICE_H_

#include "../global.h"
#include "./camera.device.h"
#include "./commands.h"
#include "./math.h"

namespace pulsar {
namespace Renderer {

/**
 * Find the closest enclosing screen area rectangle in pixels that encloses a
 * ball.
 *
 * The method returns the two x and the two y values of the boundaries. They
 * are not ordered yet and you need to find min and max for the left/right and
 * lower/upper boundary.
 *
 * The return values are floats and need to be rounded appropriately.
 */
INLINE DEVICE bool get_screen_area(
    const float3& ball_center_cam,
    const float3& ray_center_norm,
    const float& vert_rad,
    const CamInfo& cam,
    const uint& idx,
    /* Out variables. */
    float* x_1,
    float* x_2,
    float* y_1,
    float* y_2) {
  float cos_alpha = dot(cam.sensor_dir_z, ray_center_norm);
  float2 o__c_, alpha, theta;
  if (cos_alpha < EPS) {
    PULSAR_LOG_DEV(
        PULSAR_LOG_CALC_SIGNATURE,
        "signature %d|ball not visible. cos_alpha: %.9f.\n",
        idx,
        cos_alpha);
    // No intersection, ball won't be visible.
    return false;
  }
  // Multiply the direction vector with the camera rotation matrix
  // to have the optical axis being the canonical z vector (0, 0, 1).
  // TODO: optimize.
  const float3 ball_center_cam_rot = rotate(
      ball_center_cam,
      cam.pixel_dir_x / length(cam.pixel_dir_x),
      cam.pixel_dir_y / length(cam.pixel_dir_y),
      cam.sensor_dir_z);
  PULSAR_LOG_DEV(
      PULSAR_LOG_CALC_SIGNATURE,
      "signature %d|ball_center_cam_rot: %f, %f, %f.\n",
      idx,
      ball_center_cam.x,
      ball_center_cam.y,
      ball_center_cam.z);
  const float pixel_size_norm_fac = FRCP(2.f * cam.half_pixel_size);
  const float optical_offset_x =
      (static_cast<float>(cam.aperture_width) - 1.f) * .5f;
  const float optical_offset_y =
      (static_cast<float>(cam.aperture_height) - 1.f) * .5f;
  if (cam.orthogonal_projection) {
    *x_1 =
        FMA(ball_center_cam_rot.x - vert_rad,
            pixel_size_norm_fac,
            optical_offset_x);
    *x_2 =
        FMA(ball_center_cam_rot.x + vert_rad,
            pixel_size_norm_fac,
            optical_offset_x);
    *y_1 =
        FMA(ball_center_cam_rot.y - vert_rad,
            pixel_size_norm_fac,
            optical_offset_y);
    *y_2 =
        FMA(ball_center_cam_rot.y + vert_rad,
            pixel_size_norm_fac,
            optical_offset_y);
    return true;
  } else {
    o__c_.x = FMAX(
        FSQRT(
            ball_center_cam_rot.x * ball_center_cam_rot.x +
            ball_center_cam_rot.z * ball_center_cam_rot.z),
        FEPS);
    o__c_.y = FMAX(
        FSQRT(
            ball_center_cam_rot.y * ball_center_cam_rot.y +
            ball_center_cam_rot.z * ball_center_cam_rot.z),
        FEPS);
    PULSAR_LOG_DEV(
        PULSAR_LOG_CALC_SIGNATURE,
        "signature %d|o__c_: %f, %f.\n",
        idx,
        o__c_.x,
        o__c_.y);
    alpha.x = sign_dir(ball_center_cam_rot.x) *
        acos(FMIN(FMAX(ball_center_cam_rot.z / o__c_.x, -1.f), 1.f));
    alpha.y = -sign_dir(ball_center_cam_rot.y) *
        acos(FMIN(FMAX(ball_center_cam_rot.z / o__c_.y, -1.f), 1.f));
    theta.x = asin(FMIN(FMAX(vert_rad / o__c_.x, -1.f), 1.f));
    theta.y = asin(FMIN(FMAX(vert_rad / o__c_.y, -1.f), 1.f));
    PULSAR_LOG_DEV(
        PULSAR_LOG_CALC_SIGNATURE,
        "signature %d|alpha.x: %f, alpha.y: %f, theta.x: %f, theta.y: %f.\n",
        idx,
        alpha.x,
        alpha.y,
        theta.x,
        theta.y);
    *x_1 = tan(alpha.x - theta.x) * cam.focal_length;
    *x_2 = tan(alpha.x + theta.x) * cam.focal_length;
    *y_1 = tan(alpha.y - theta.y) * cam.focal_length;
    *y_2 = tan(alpha.y + theta.y) * cam.focal_length;
    PULSAR_LOG_DEV(
        PULSAR_LOG_CALC_SIGNATURE,
        "signature %d|in sensor plane: x_1: %f, x_2: %f, y_1: %f, y_2: %f.\n",
        idx,
        *x_1,
        *x_2,
        *y_1,
        *y_2);
    *x_1 = FMA(*x_1, pixel_size_norm_fac, optical_offset_x);
    *x_2 = FMA(*x_2, pixel_size_norm_fac, optical_offset_x);
    *y_1 = FMA(*y_1, -pixel_size_norm_fac, optical_offset_y);
    *y_2 = FMA(*y_2, -pixel_size_norm_fac, optical_offset_y);
    return true;
  }
};

} // namespace Renderer
} // namespace pulsar

#endif
