/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef PULSAR_NATIVE_INCLUDE_RENDERER_RENDER_DEVICE_H_
#define PULSAR_NATIVE_INCLUDE_RENDERER_RENDER_DEVICE_H_

#include "../global.h"
#include "./camera.device.h"
#include "./commands.h"
#include "./math.h"
#include "./renderer.h"

#include "./closest_sphere_tracker.device.h"
#include "./renderer.draw.device.h"

namespace pulsar {
namespace Renderer {

template <bool DEV>
GLOBAL void render(
    size_t const* const RESTRICT
        num_balls, /** Number of balls relevant for this pass. */
    IntersectInfo const* const RESTRICT ii_d, /** Intersect information. */
    DrawInfo const* const RESTRICT di_d, /** Draw information. */
    float const* const RESTRICT min_depth_d, /** Minimum depth per sphere. */
    int const* const RESTRICT ids_d, /** IDs. */
    float const* const RESTRICT op_d, /** Opacity. */
    const CamInfo cam_norm, /** Camera normalized with all vectors to be in the
                             * camera coordinate system.
                             */
    const float gamma, /** Transparency parameter. **/
    const float percent_allowed_difference, /** Maximum allowed
                                               error in color. */
    const uint max_n_hits,
    const float* bg_col,
    const uint mode,
    const int x_min,
    const int y_min,
    const int x_step,
    const int y_step,
    // Out variables.
    float* const RESTRICT result_d, /** The result image. */
    float* const RESTRICT forw_info_d, /** Additional information needed for the
                                           grad computation. */
    const int n_track /** The number of spheres to track for backprop. */
) {
  // Do not early stop threads in this block here. They can all contribute to
  // the scanning process, we just have to prevent from writing their result.
  GET_PARALLEL_IDS_2D(offs_x, offs_y, x_step, y_step);
  // Variable declarations and const initializations.
  const float ln_pad_over_1minuspad =
      FLN(percent_allowed_difference / (1.f - percent_allowed_difference));
  /** A facility to track the closest spheres to the camera
      (in preparation for gradient calculation). */
  ClosestSphereTracker tracker(n_track);
  const uint coord_x = x_min + offs_x; /** Ray coordinate x. */
  const uint coord_y = y_min + offs_y; /** Ray coordinate y. */
  float3 ray_dir_norm; /** Ray cast through the pixel, normalized. */
  float2 projected_ray; /** Ray intersection with the sensor. */
  if (cam_norm.orthogonal_projection) {
    ray_dir_norm = cam_norm.sensor_dir_z;
    projected_ray.x = static_cast<float>(coord_x);
    projected_ray.y = static_cast<float>(coord_y);
  } else {
    ray_dir_norm = normalize(
        cam_norm.pixel_0_0_center + coord_x * cam_norm.pixel_dir_x +
        coord_y * cam_norm.pixel_dir_y);
    // This is a reasonable assumption for normal focal lengths and image sizes.
    PASSERT(FABS(ray_dir_norm.z) > FEPS);
    projected_ray.x = ray_dir_norm.x / ray_dir_norm.z * cam_norm.focal_length;
    projected_ray.y = ray_dir_norm.y / ray_dir_norm.z * cam_norm.focal_length;
  }
  PULSAR_LOG_DEV_PIX(
      PULSAR_LOG_RENDER_PIX,
      "render|ray_dir_norm: %.9f, %.9f, %.9f. projected_ray: %.9f, %.9f.\n",
      ray_dir_norm.x,
      ray_dir_norm.y,
      ray_dir_norm.z,
      projected_ray.x,
      projected_ray.y);
  // Set up shared infrastructure.
  /** This entire thread block. */
  cg::thread_block thread_block = cg::this_thread_block();
  /** The collaborators within a warp. */
  cg::coalesced_group thread_warp = cg::coalesced_threads();
  /** The number of loaded balls in the load buffer di_l. */
  SHARED uint n_loaded;
  /** Draw information buffer. */
  SHARED DrawInfo di_l[RENDER_BUFFER_SIZE];
  /** The original sphere id of each loaded sphere. */
  SHARED uint sphere_id_l[RENDER_BUFFER_SIZE];
  /** The number of pixels in this block that are done. */
  SHARED int n_pixels_done;
  /** Whether loading of balls is completed. */
  SHARED bool loading_done;
  /** The number of balls loaded overall (just for statistics). */
  SHARED int n_balls_loaded;
  /** The area this thread block covers. */
  SHARED IntersectInfo block_area;
  if (thread_block.thread_rank() == 0) {
    // Initialize the shared variables.
    n_loaded = 0;
    block_area.min.x = static_cast<ushort>(coord_x);
    block_area.max.x = static_cast<ushort>(IMIN(
        coord_x + blockDim.x, cam_norm.film_border_left + cam_norm.film_width));
    block_area.min.y = static_cast<ushort>(coord_y);
    block_area.max.y = static_cast<ushort>(IMIN(
        coord_y + blockDim.y, cam_norm.film_border_top + cam_norm.film_height));
    n_pixels_done = 0;
    loading_done = false;
    n_balls_loaded = 0;
  }
  PULSAR_LOG_DEV_PIX(
      PULSAR_LOG_RENDER_PIX,
      "render|block_area.min: %d, %d. block_area.max: %d, %d.\n",
      block_area.min.x,
      block_area.min.y,
      block_area.max.x,
      block_area.max.y);
  // Initialization of the pixel with the background color.
  /**
   * The result of this very pixel.
   * the offset calculation might overflow if this thread is out of
   * bounds of the film. However, in this case result is not
   * accessed, so this is fine.
   */
  float* result = result_d +
      (coord_y - cam_norm.film_border_top) * cam_norm.film_width *
          cam_norm.n_channels +
      (coord_x - cam_norm.film_border_left) * cam_norm.n_channels;
  if (coord_x >= cam_norm.film_border_left &&
      coord_x < cam_norm.film_border_left + cam_norm.film_width &&
      coord_y >= cam_norm.film_border_top &&
      coord_y < cam_norm.film_border_top + cam_norm.film_height) {
    // Initialize the result.
    if (mode == 0u) {
      for (uint c_id = 0; c_id < cam_norm.n_channels; ++c_id)
        result[c_id] = bg_col[c_id];
    } else {
      result[0] = 0.f;
    }
  }
  /** Normalization denominator. */
  float sm_d = 1.f;
  /** Normalization tracker for stable softmax. The maximum observed value. */
  float sm_m = cam_norm.background_normalization_depth / gamma;
  /** Whether this pixel has had all information needed for drawing. */
  bool done =
      (coord_x < cam_norm.film_border_left ||
       coord_x >= cam_norm.film_border_left + cam_norm.film_width ||
       coord_y < cam_norm.film_border_top ||
       coord_y >= cam_norm.film_border_top + cam_norm.film_height);
  /** The depth threshold for a new point to have at least
   * `percent_allowed_difference` influence on the result color. All points that
   * are further away than this are ignored.
   */
  float depth_threshold = done ? -1.f : MAX_FLOAT;
  /** The closest intersection possible of a ball that was hit by this pixel
   * ray. */
  float max_closest_possible_intersection_hit = -1.f;
  bool hit; /** Whether a sphere was hit. */
  float intersection_depth; /** The intersection_depth for a sphere at this
                               pixel. */
  float closest_possible_intersection; /** The closest possible intersection
    for this sphere. */
  float max_closest_possible_intersection;
  // Sync up threads so that everyone is similarly initialized.
  thread_block.sync();
  //! Coalesced loading and intersection analysis of balls.
  for (uint ball_idx = thread_block.thread_rank();
       ball_idx < iDivCeil(static_cast<uint>(*num_balls), thread_block.size()) *
               thread_block.size() &&
       !loading_done && n_pixels_done < thread_block.size();
       ball_idx += thread_block.size()) {
    if (ball_idx < static_cast<uint>(*num_balls)) { // Account for overflow.
      const IntersectInfo& ii = ii_d[ball_idx];
      hit = (ii.min.x <= block_area.max.x) && (ii.max.x > block_area.min.x) &&
          (ii.min.y <= block_area.max.y) && (ii.max.y > block_area.min.y);
      if (hit) {
        uint write_idx = ATOMICADD_B(&n_loaded, 1u);
        di_l[write_idx] = di_d[ball_idx];
        sphere_id_l[write_idx] = static_cast<uint>(ids_d[ball_idx]);
        PULSAR_LOG_DEV_PIXB(
            PULSAR_LOG_RENDER_PIX,
            "render|found intersection with sphere %u.\n",
            sphere_id_l[write_idx]);
      }
      if (ii.min.x == MAX_USHORT)
        // This is an invalid sphere (out of image). These spheres have
        // maximum depth. Since we ordered the spheres by earliest possible
        // intersection depth we re certain that there will no other sphere
        // that is relevant after this one.
        loading_done = true;
    }
    // Reset n_pixels_done.
    n_pixels_done = 0;
    thread_block.sync(); // Make sure n_loaded is updated.
    if (n_loaded > RENDER_BUFFER_LOAD_THRESH) {
      // The load buffer is full enough. Draw.
      if (thread_block.thread_rank() == 0)
        n_balls_loaded += n_loaded;
      max_closest_possible_intersection = 0.f;
      // This excludes threads outside of the image boundary. Also, it reduces
      // block artifacts.
      if (!done) {
        for (uint draw_idx = 0; draw_idx < n_loaded; ++draw_idx) {
          intersection_depth = 0.f;
          if (cam_norm.orthogonal_projection) {
            // The closest possible intersection is the distance to the camera
            // plane.
            closest_possible_intersection = min_depth_d[sphere_id_l[draw_idx]];
          } else {
            closest_possible_intersection =
                di_l[draw_idx].t_center - di_l[draw_idx].radius;
          }
          PULSAR_LOG_DEV_PIX(
              PULSAR_LOG_RENDER_PIX,
              "render|drawing sphere %u (depth: %f, "
              "closest possible intersection: %f).\n",
              sphere_id_l[draw_idx],
              di_l[draw_idx].t_center,
              closest_possible_intersection);
          hit = draw(
              di_l[draw_idx], // Sphere to draw.
              op_d == NULL ? 1.f : op_d[sphere_id_l[draw_idx]], // Opacity.
              cam_norm, // Cam.
              gamma, // Gamma.
              ray_dir_norm, // Ray direction.
              projected_ray, // Ray intersection with the image.
              // Mode switches.
              true, // Draw.
              false,
              false,
              false,
              false,
              false, // No gradients.
              // Position info.
              coord_x,
              coord_y,
              sphere_id_l[draw_idx],
              // Optional in variables.
              NULL, // intersect information.
              NULL, // ray_dir.
              NULL, // norm_ray_dir.
              NULL, // grad_pix.
              &ln_pad_over_1minuspad,
              // in/out variables
              &sm_d,
              &sm_m,
              result,
              // Optional out.
              &depth_threshold,
              &intersection_depth,
              NULL,
              NULL,
              NULL,
              NULL,
              NULL // gradients.
          );
          if (hit) {
            max_closest_possible_intersection_hit = FMAX(
                max_closest_possible_intersection_hit,
                closest_possible_intersection);
            tracker.track(
                sphere_id_l[draw_idx], intersection_depth, coord_x, coord_y);
          }
          max_closest_possible_intersection = FMAX(
              max_closest_possible_intersection, closest_possible_intersection);
        }
        PULSAR_LOG_DEV_PIX(
            PULSAR_LOG_RENDER_PIX,
            "render|max_closest_possible_intersection: %f, "
            "depth_threshold: %f.\n",
            max_closest_possible_intersection,
            depth_threshold);
      }
      done = done ||
          (percent_allowed_difference > 0.f &&
           max_closest_possible_intersection > depth_threshold) ||
          tracker.get_n_hits() >= max_n_hits;
      uint warp_done = thread_warp.ballot(done);
      if (thread_warp.thread_rank() == 0)
        ATOMICADD_B(&n_pixels_done, POPC(warp_done));
      // This sync is necessary to keep n_loaded until all threads are done with
      // painting.
      thread_block.sync();
      n_loaded = 0;
    }
    thread_block.sync();
  }
  if (thread_block.thread_rank() == 0)
    n_balls_loaded += n_loaded;
  PULSAR_LOG_DEV_PIX(
      PULSAR_LOG_RENDER_PIX,
      "render|loaded %d balls in total.\n",
      n_balls_loaded);
  if (!done) {
    for (uint draw_idx = 0; draw_idx < n_loaded; ++draw_idx) {
      intersection_depth = 0.f;
      if (cam_norm.orthogonal_projection) {
        // The closest possible intersection is the distance to the camera
        // plane.
        closest_possible_intersection = min_depth_d[sphere_id_l[draw_idx]];
      } else {
        closest_possible_intersection =
            di_l[draw_idx].t_center - di_l[draw_idx].radius;
      }
      PULSAR_LOG_DEV_PIX(
          PULSAR_LOG_RENDER_PIX,
          "render|drawing sphere %u (depth: %f, "
          "closest possible intersection: %f).\n",
          sphere_id_l[draw_idx],
          di_l[draw_idx].t_center,
          closest_possible_intersection);
      hit = draw(
          di_l[draw_idx], // Sphere to draw.
          op_d == NULL ? 1.f : op_d[sphere_id_l[draw_idx]], // Opacity.
          cam_norm, // Cam.
          gamma, // Gamma.
          ray_dir_norm, // Ray direction.
          projected_ray, // Ray intersection with the image.
          // Mode switches.
          true, // Draw.
          false,
          false,
          false,
          false,
          false, // No gradients.
          // Logging info.
          coord_x,
          coord_y,
          sphere_id_l[draw_idx],
          // Optional in variables.
          NULL, // intersect information.
          NULL, // ray_dir.
          NULL, // norm_ray_dir.
          NULL, // grad_pix.
          &ln_pad_over_1minuspad,
          // in/out variables
          &sm_d,
          &sm_m,
          result,
          // Optional out.
          &depth_threshold,
          &intersection_depth,
          NULL,
          NULL,
          NULL,
          NULL,
          NULL // gradients.
      );
      if (hit) {
        max_closest_possible_intersection_hit = FMAX(
            max_closest_possible_intersection_hit,
            closest_possible_intersection);
        tracker.track(
            sphere_id_l[draw_idx], intersection_depth, coord_x, coord_y);
      }
    }
  }
  if (coord_x < cam_norm.film_border_left ||
      coord_y < cam_norm.film_border_top ||
      coord_x >= cam_norm.film_border_left + cam_norm.film_width ||
      coord_y >= cam_norm.film_border_top + cam_norm.film_height) {
    RETURN_PARALLEL();
  }
  if (mode == 1u) {
    // The subtractions, for example coord_y - cam_norm.film_border_left, are
    // safe even though both components are uints. We checked their relation
    // just above.
    result_d
        [(coord_y - cam_norm.film_border_top) * cam_norm.film_width *
             cam_norm.n_channels +
         (coord_x - cam_norm.film_border_left) * cam_norm.n_channels] =
            static_cast<float>(tracker.get_n_hits());
  } else {
    float sm_d_normfac = FRCP(FMAX(sm_d, FEPS));
    for (uint c_id = 0; c_id < cam_norm.n_channels; ++c_id)
      result[c_id] *= sm_d_normfac;
    int write_loc = (coord_y - cam_norm.film_border_top) * cam_norm.film_width *
            (3 + 2 * n_track) +
        (coord_x - cam_norm.film_border_left) * (3 + 2 * n_track);
    forw_info_d[write_loc] = sm_m;
    forw_info_d[write_loc + 1] = sm_d;
    forw_info_d[write_loc + 2] = max_closest_possible_intersection_hit;
    PULSAR_LOG_DEV_PIX(
        PULSAR_LOG_RENDER_PIX,
        "render|writing the %d most important ball infos.\n",
        IMIN(n_track, tracker.get_n_hits()));
    for (int i = 0; i < n_track; ++i) {
      int sphere_id = tracker.get_closest_sphere_id(i);
      IASF(sphere_id, forw_info_d[write_loc + 3 + i * 2]);
      forw_info_d[write_loc + 3 + i * 2 + 1] =
          tracker.get_closest_sphere_depth(i) == MAX_FLOAT
          ? -1.f
          : tracker.get_closest_sphere_depth(i);
      PULSAR_LOG_DEV_PIX(
          PULSAR_LOG_RENDER_PIX,
          "render|writing %d most important: id: %d, normalized depth: %f.\n",
          i,
          tracker.get_closest_sphere_id(i),
          tracker.get_closest_sphere_depth(i));
    }
  }
  END_PARALLEL_2D();
}

} // namespace Renderer
} // namespace pulsar

#endif
