/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef PULSAR_NATIVE_INCLUDE_CAMERA_H_
#define PULSAR_NATIVE_INCLUDE_CAMERA_H_

#include <stdint.h>
#include "../global.h"

namespace pulsar {
/**
 * Everything that's needed to raycast with our camera model.
 */
struct CamInfo {
  float3 eye; /** Position in world coordinates. */
  float3 pixel_0_0_center; /** LUC center of pixel position in world
                              coordinates. */
  float3 pixel_dir_x; /** Direction for increasing x for one pixel to the next,
                       * in  world coordinates. */
  float3 pixel_dir_y; /** Direction for increasing y for one pixel to the next,
                       * in  world coordinates. */
  float3 sensor_dir_z; /** Normalized direction vector from eye through the
                        * sensor in z direction (optical axis). */
  float half_pixel_size; /** Half size of a pixel, in world coordinates. This
                          * must be consistent with pixel_dir_x and pixel_dir_y!
                          */
  float focal_length; /** The focal length, if applicable. */
  uint aperture_width; /** Full image width in px, possibly not fully used
                        * in case of a shifted principal point. */
  uint aperture_height; /** Full image height in px, possibly not fully used
                         * in case of a shifted principal point. */
  uint film_width; /** Resulting image width. */
  uint film_height; /** Resulting image height. */
  /** The top left coordinates (inclusive) of the film in the full aperture. */
  uint film_border_left, film_border_top;
  int32_t principal_point_offset_x; /** Horizontal principal point offset. */
  int32_t principal_point_offset_y; /** Vertical principal point offset. */
  float min_dist; /** Minimum distance for a ball to be rendered. */
  float max_dist; /** Maximum distance for a ball to be rendered. */
  float norm_fac; /** 1 / (max_dist - min_dist), pre-computed. */
  /** The depth where to place the background, in normalized coordinates where
   * 0. is the backmost depth and 1. the frontmost. */
  float background_normalization_depth;
  /** The number of image content channels to use. Usually three. */
  uint n_channels;
  /** Whether to use an orthogonal instead of a perspective projection. */
  bool orthogonal_projection;
  /** Whether to use a right-handed system (inverts the z axis). */
  bool right_handed;
};

inline bool operator==(const CamInfo& a, const CamInfo& b) {
  return a.film_width == b.film_width && a.film_height == b.film_height &&
      a.background_normalization_depth == b.background_normalization_depth &&
      a.n_channels == b.n_channels &&
      a.orthogonal_projection == b.orthogonal_projection &&
      a.right_handed == b.right_handed;
};

struct CamGradInfo {
  HOST DEVICE CamGradInfo(int = 0);
  float3 cam_pos;
  float3 pixel_0_0_center;
  float3 pixel_dir_x;
  float3 pixel_dir_y;
};

} // namespace pulsar

#endif
