/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef PULSAR_NATIVE_INCLUDE_RENDERER_CALC_SIGNATURE_DEVICE_H_
#define PULSAR_NATIVE_INCLUDE_RENDERER_CALC_SIGNATURE_DEVICE_H_

#include "../global.h"
#include "./camera.device.h"
#include "./commands.h"
#include "./math.h"
#include "./renderer.get_screen_area.device.h"
#include "./renderer.h"

namespace pulsar {
namespace Renderer {

template <bool DEV>
GLOBAL void calc_signature(
    Renderer renderer,
    float3 const* const RESTRICT vert_poss,
    float const* const RESTRICT vert_cols,
    float const* const RESTRICT vert_rads,
    const uint num_balls) {
  /* We're not using RESTRICT here for the pointers within `renderer`. Just one
     value is being read from each of the pointers, so the effect would be
     negligible or non-existent. */
  GET_PARALLEL_IDX_1D(idx, num_balls);
  // Create aliases.
  // For reading...
  const float3& vert_pos = vert_poss[idx]; /** Vertex position. */
  const float* vert_col =
      vert_cols + idx * renderer.cam.n_channels; /** Vertex color. */
  const float& vert_rad = vert_rads[idx]; /** Vertex radius. */
  const CamInfo& cam = renderer.cam; /** Camera in world coordinates. */
  // For writing...
  /** Ball ID (either original index of the ball or -1 if not visible). */
  int& id_out = renderer.ids_d[idx];
  /** Intersection helper structure for the ball. */
  IntersectInfo& intersect_helper_out = renderer.ii_d[idx];
  /** Draw helper structure for this ball. */
  DrawInfo& draw_helper_out = renderer.di_d[idx];
  /** Minimum possible intersection depth for this ball. */
  float& closest_possible_intersect_out = renderer.min_depth_d[idx];
  PULSAR_LOG_DEV(
      PULSAR_LOG_CALC_SIGNATURE,
      "signature %d|vert_pos: %.9f, %.9f, %.9f, vert_col (first three): "
      "%.9f, %.9f, %.9f.\n",
      idx,
      vert_pos.x,
      vert_pos.y,
      vert_pos.z,
      vert_col[0],
      vert_col[1],
      vert_col[2]);
  // Set flags to invalid for a potential early return.
  id_out = -1; // Invalid ID.
  closest_possible_intersect_out =
      MAX_FLOAT; // These spheres are sorted to the very end.
  intersect_helper_out.max.x = MAX_USHORT; // No intersection possible.
  intersect_helper_out.min.x = MAX_USHORT;
  intersect_helper_out.max.y = MAX_USHORT;
  intersect_helper_out.min.y = MAX_USHORT;
  // Start processing.
  /** Ball center in the camera coordinate system. */
  const float3 ball_center_cam = vert_pos - cam.eye;
  /** Distance to the ball center in the camera coordinate system. */
  const float t_center = length(ball_center_cam);
  /** Closest possible intersection with this ball from the camera. */
  float closest_possible_intersect;
  if (cam.orthogonal_projection) {
    const float3 ball_center_cam_rot = rotate(
        ball_center_cam,
        cam.pixel_dir_x / length(cam.pixel_dir_x),
        cam.pixel_dir_y / length(cam.pixel_dir_y),
        cam.sensor_dir_z);
    closest_possible_intersect = ball_center_cam_rot.z - vert_rad;
  } else {
    closest_possible_intersect = t_center - vert_rad;
  }
  PULSAR_LOG_DEV(
      PULSAR_LOG_CALC_SIGNATURE,
      "signature %d|t_center: %f. vert_rad: %f. "
      "closest_possible_intersect: %f.\n",
      idx,
      t_center,
      vert_rad,
      closest_possible_intersect);
  /**
   * Corner points of the enclosing projected rectangle of the ball.
   * They are first calculated in the camera coordinate system, then
   * converted to the pixel coordinate system.
   */
  float x_1, x_2, y_1, y_2;
  bool hits_screen_plane;
  float3 ray_center_norm = ball_center_cam / t_center;
  PASSERT(vert_rad >= 0.f);
  if (closest_possible_intersect < cam.min_dist ||
      closest_possible_intersect > cam.max_dist) {
    PULSAR_LOG_DEV(
        PULSAR_LOG_CALC_SIGNATURE,
        "signature %d|ignoring sphere out of min/max bounds: %.9f, "
        "min: %.9f, max: %.9f.\n",
        idx,
        closest_possible_intersect,
        cam.min_dist,
        cam.max_dist);
    RETURN_PARALLEL();
  }
  // Find the relevant region on the screen plane.
  hits_screen_plane = get_screen_area(
      ball_center_cam,
      ray_center_norm,
      vert_rad,
      cam,
      idx,
      &x_1,
      &x_2,
      &y_1,
      &y_2);
  if (!hits_screen_plane)
    RETURN_PARALLEL();
  PULSAR_LOG_DEV(
      PULSAR_LOG_CALC_SIGNATURE,
      "signature %d|in pixels: x_1: %f, x_2: %f, y_1: %f, y_2: %f.\n",
      idx,
      x_1,
      x_2,
      y_1,
      y_2);
  // Check whether the pixel coordinates are on screen.
  if (FMAX(x_1, x_2) <= static_cast<float>(cam.film_border_left) ||
      FMIN(x_1, x_2) >=
          static_cast<float>(cam.film_border_left + cam.film_width) - 0.5f ||
      FMAX(y_1, y_2) <= static_cast<float>(cam.film_border_top) ||
      FMIN(y_1, y_2) >
          static_cast<float>(cam.film_border_top + cam.film_height) - 0.5f)
    RETURN_PARALLEL();
  // Write results.
  id_out = idx;
  intersect_helper_out.min.x = static_cast<ushort>(
      FMAX(FMIN(x_1, x_2), static_cast<float>(cam.film_border_left)));
  intersect_helper_out.min.y = static_cast<ushort>(
      FMAX(FMIN(y_1, y_2), static_cast<float>(cam.film_border_top)));
  // In the following calculations, the max that needs to be stored is
  // exclusive.
  // That means that the calculated value needs to be `ceil`ed and incremented
  // to find the correct value.
  intersect_helper_out.max.x = static_cast<ushort>(FMIN(
      FCEIL(FMAX(x_1, x_2)) + 1,
      static_cast<float>(cam.film_border_left + cam.film_width)));
  intersect_helper_out.max.y = static_cast<ushort>(FMIN(
      FCEIL(FMAX(y_1, y_2)) + 1,
      static_cast<float>(cam.film_border_top + cam.film_height)));
  PULSAR_LOG_DEV(
      PULSAR_LOG_CALC_SIGNATURE,
      "signature %d|limits after refining: x_1: %u, x_2: %u, "
      "y_1: %u, y_2: %u.\n",
      idx,
      intersect_helper_out.min.x,
      intersect_helper_out.max.x,
      intersect_helper_out.min.y,
      intersect_helper_out.max.y);
  if (intersect_helper_out.min.x == MAX_USHORT) {
    id_out = -1;
    RETURN_PARALLEL();
  }
  PULSAR_LOG_DEV(
      PULSAR_LOG_CALC_SIGNATURE,
      "signature %d|writing info. closest_possible_intersect: %.9f. "
      "ray_center_norm: %.9f, %.9f, %.9f. t_center: %.9f. radius: %.9f.\n",
      idx,
      closest_possible_intersect,
      ray_center_norm.x,
      ray_center_norm.y,
      ray_center_norm.z,
      t_center,
      vert_rad);
  closest_possible_intersect_out = closest_possible_intersect;
  draw_helper_out.ray_center_norm = ray_center_norm;
  draw_helper_out.t_center = t_center;
  draw_helper_out.radius = vert_rad;
  if (cam.n_channels <= 3) {
    draw_helper_out.first_color = vert_col[0];
    for (uint c_id = 1; c_id < cam.n_channels; ++c_id) {
      draw_helper_out.color_union.color[c_id - 1] = vert_col[c_id];
    }
  } else {
    draw_helper_out.color_union.ptr = const_cast<float*>(vert_col);
  }
  END_PARALLEL();
};

} // namespace Renderer
} // namespace pulsar

#endif
