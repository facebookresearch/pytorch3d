/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef PULSAR_NATIVE_INCLUDE_RENDERER_CALC_GRADIENTS_H_
#define PULSAR_NATIVE_INCLUDE_RENDERER_CALC_GRADIENTS_H_

#include "../global.h"
#include "./commands.h"
#include "./renderer.h"

#include "./renderer.draw.device.h"

namespace pulsar {
namespace Renderer {

template <bool DEV>
GLOBAL void calc_gradients(
    const CamInfo cam, /** Camera in world coordinates. */
    float const* const RESTRICT grad_im, /** The gradient image. */
    const float
        gamma, /** The transparency parameter used in the forward pass. */
    float3 const* const RESTRICT vert_poss, /** Vertex position vector. */
    float const* const RESTRICT vert_cols, /** Vertex color vector. */
    float const* const RESTRICT vert_rads, /** Vertex radius vector. */
    float const* const RESTRICT opacity, /** Vertex opacity. */
    const uint num_balls, /** Number of balls. */
    float const* const RESTRICT result_d, /** Result image. */
    float const* const RESTRICT forw_info_d, /** Forward pass info. */
    DrawInfo const* const RESTRICT di_d, /** Draw information. */
    IntersectInfo const* const RESTRICT ii_d, /** Intersect information. */
    // Mode switches.
    const bool calc_grad_pos,
    const bool calc_grad_col,
    const bool calc_grad_rad,
    const bool calc_grad_cam,
    const bool calc_grad_opy,
    // Out variables.
    float* const RESTRICT grad_rad_d, /** Radius gradients. */
    float* const RESTRICT grad_col_d, /** Color gradients. */
    float3* const RESTRICT grad_pos_d, /** Position gradients. */
    CamGradInfo* const RESTRICT grad_cam_buf_d, /** Camera gradient buffer. */
    float* const RESTRICT grad_opy_d, /** Opacity gradient buffer. */
    int* const RESTRICT
        grad_contributed_d, /** Gradient contribution counter. */
    // Infrastructure.
    const int n_track,
    const uint offs_x,
    const uint offs_y /** Debug offsets. */
) {
  uint limit_x = cam.film_width, limit_y = cam.film_height;
  if (offs_x != 0) {
    // We're in debug mode.
    limit_x = 1;
    limit_y = 1;
  }
  GET_PARALLEL_IDS_2D(coord_x_base, coord_y_base, limit_x, limit_y);
  // coord_x_base and coord_y_base are in the film coordinate system.
  // We now need to translate to the aperture coordinate system. If
  // the principal point was shifted left/up nothing has to be
  // subtracted - only shift needs to be added in case it has been
  // shifted down/right.
  const uint film_coord_x = coord_x_base + offs_x;
  const uint ap_coord_x = film_coord_x +
      2 * static_cast<uint>(std::max(0, cam.principal_point_offset_x));
  const uint film_coord_y = coord_y_base + offs_y;
  const uint ap_coord_y = film_coord_y +
      2 * static_cast<uint>(std::max(0, cam.principal_point_offset_y));
  const float3 ray_dir = /** Ray cast through the pixel, normalized. */
      cam.pixel_0_0_center + ap_coord_x * cam.pixel_dir_x +
      ap_coord_y * cam.pixel_dir_y;
  const float norm_ray_dir = length(ray_dir);
  // ray_dir_norm *must* be calculated here in the same way as in the draw
  // function to have the same values withno other numerical instabilities
  // (for example, ray_dir * FRCP(norm_ray_dir) does not work)!
  float3 ray_dir_norm; /** Ray cast through the pixel, normalized. */
  float2 projected_ray; /** Ray intersection with the sensor. */
  if (cam.orthogonal_projection) {
    ray_dir_norm = cam.sensor_dir_z;
    projected_ray.x = static_cast<float>(ap_coord_x);
    projected_ray.y = static_cast<float>(ap_coord_y);
  } else {
    ray_dir_norm = normalize(
        cam.pixel_0_0_center + ap_coord_x * cam.pixel_dir_x +
        ap_coord_y * cam.pixel_dir_y);
    // This is a reasonable assumption for normal focal lengths and image sizes.
    PASSERT(FABS(ray_dir_norm.z) > FEPS);
    projected_ray.x = ray_dir_norm.x / ray_dir_norm.z * cam.focal_length;
    projected_ray.y = ray_dir_norm.y / ray_dir_norm.z * cam.focal_length;
  }
  float* result = const_cast<float*>(
      result_d + film_coord_y * cam.film_width * cam.n_channels +
      film_coord_x * cam.n_channels);
  const float* grad_im_l = grad_im +
      film_coord_y * cam.film_width * cam.n_channels +
      film_coord_x * cam.n_channels;
  // For writing...
  float3 grad_pos;
  float grad_rad, grad_opy;
  CamGradInfo grad_cam_local = CamGradInfo();
  // Set up shared infrastructure.
  const int fwi_loc = film_coord_y * cam.film_width * (3 + 2 * n_track) +
      film_coord_x * (3 + 2 * n_track);
  float sm_m = forw_info_d[fwi_loc];
  float sm_d = forw_info_d[fwi_loc + 1];
  PULSAR_LOG_DEV_APIX(
      PULSAR_LOG_GRAD,
      "grad|sm_m: %f, sm_d: %f, result: "
      "%f, %f, %f; grad_im: %f, %f, %f.\n",
      sm_m,
      sm_d,
      result[0],
      result[1],
      result[2],
      grad_im_l[0],
      grad_im_l[1],
      grad_im_l[2]);
  // Start processing.
  for (int grad_idx = 0; grad_idx < n_track; ++grad_idx) {
    int sphere_idx;
    FASI(forw_info_d[fwi_loc + 3 + 2 * grad_idx], sphere_idx);
    PASSERT(
        sphere_idx == -1 ||
        sphere_idx >= 0 && static_cast<uint>(sphere_idx) < num_balls);
    if (sphere_idx >= 0) {
      // TODO: make more efficient.
      grad_pos = make_float3(0.f, 0.f, 0.f);
      grad_rad = 0.f;
      grad_cam_local = CamGradInfo();
      const DrawInfo di = di_d[sphere_idx];
      grad_opy = 0.f;
      draw(
          di,
          opacity == NULL ? 1.f : opacity[sphere_idx],
          cam,
          gamma,
          ray_dir_norm,
          projected_ray,
          // Mode switches.
          false, // draw only
          calc_grad_pos,
          calc_grad_col,
          calc_grad_rad,
          calc_grad_cam,
          calc_grad_opy,
          // Position info.
          ap_coord_x,
          ap_coord_y,
          sphere_idx,
          // Optional in.
          &ii_d[sphere_idx],
          &ray_dir,
          &norm_ray_dir,
          grad_im_l,
          NULL,
          // In/out
          &sm_d,
          &sm_m,
          result,
          // Optional out.
          NULL,
          NULL,
          &grad_pos,
          grad_col_d + sphere_idx * cam.n_channels,
          &grad_rad,
          &grad_cam_local,
          &grad_opy);
      ATOMICADD(&(grad_rad_d[sphere_idx]), grad_rad);
      // Color has been added directly.
      ATOMICADD_F3(&(grad_pos_d[sphere_idx]), grad_pos);
      ATOMICADD_F3(
          &(grad_cam_buf_d[sphere_idx].cam_pos), grad_cam_local.cam_pos);
      if (!cam.orthogonal_projection) {
        ATOMICADD_F3(
            &(grad_cam_buf_d[sphere_idx].pixel_0_0_center),
            grad_cam_local.pixel_0_0_center);
      }
      ATOMICADD_F3(
          &(grad_cam_buf_d[sphere_idx].pixel_dir_x),
          grad_cam_local.pixel_dir_x);
      ATOMICADD_F3(
          &(grad_cam_buf_d[sphere_idx].pixel_dir_y),
          grad_cam_local.pixel_dir_y);
      ATOMICADD(&(grad_opy_d[sphere_idx]), grad_opy);
      ATOMICADD(&(grad_contributed_d[sphere_idx]), 1);
    }
  }
  END_PARALLEL_2D_NORET();
};

} // namespace Renderer
} // namespace pulsar

#endif
