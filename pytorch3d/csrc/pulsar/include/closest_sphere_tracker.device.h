/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef PULSAR_NATIVE_INCLUDE_CLOSEST_SPHERE_TRACKER_DEVICE_H_
#define PULSAR_NATIVE_INCLUDE_CLOSEST_SPHERE_TRACKER_DEVICE_H_

#include "../global.h"

namespace pulsar {
namespace Renderer {

/**
 * A facility to track the closest spheres to the camera.
 *
 * Their max number is defined by MAX_GRAD_SPHERES (this is defined in
 * `pulsar/native/global.h`). This is done to keep the performance as high as
 * possible because this struct needs to do updates continuously on the GPU.
 */
struct ClosestSphereTracker {
 public:
  IHD ClosestSphereTracker(const int& n_track) : n_hits(0), n_track(n_track) {
    PASSERT(n_track < MAX_GRAD_SPHERES);
    // Initialize the sphere IDs to -1 and the weights to 0.
    for (int i = 0; i < n_track; ++i) {
      this->most_important_sphere_ids[i] = -1;
      this->closest_sphere_intersection_depths[i] = MAX_FLOAT;
    }
  };

  IHD void track(
      const uint& sphere_idx,
      const float& intersection_depth,
      const uint& coord_x,
      const uint& coord_y) {
    PULSAR_LOG_DEV_PIX(
        PULSAR_LOG_TRACKER_PIX,
        "tracker|tracking sphere %u (depth: %f).\n",
        sphere_idx,
        intersection_depth);
    for (int i = IMIN(this->n_hits, n_track) - 1; i >= -1; --i) {
      if (i < 0 ||
          this->closest_sphere_intersection_depths[i] < intersection_depth) {
        // Write position is i+1.
        PULSAR_LOG_DEV_PIX(
            PULSAR_LOG_TRACKER_PIX,
            "tracker|determined writing position: %d.\n",
            i + 1);
        if (i + 1 < n_track) {
          // Shift every other sphere back.
          for (int j = n_track - 1; j > i + 1; --j) {
            this->closest_sphere_intersection_depths[j] =
                this->closest_sphere_intersection_depths[j - 1];
            this->most_important_sphere_ids[j] =
                this->most_important_sphere_ids[j - 1];
          }
          this->closest_sphere_intersection_depths[i + 1] = intersection_depth;
          this->most_important_sphere_ids[i + 1] = sphere_idx;
        }
        break;
      }
    }
#if PULSAR_LOG_TRACKER_PIX
    PULSAR_LOG_DEV_PIX(
        PULSAR_LOG_TRACKER_PIX,
        "tracker|sphere list after adding sphere %u:\n",
        sphere_idx);
    for (int i = 0; i < n_track; ++i) {
      PULSAR_LOG_DEV_PIX(
          PULSAR_LOG_TRACKER_PIX,
          "tracker|sphere %d: %d (depth: %f).\n",
          i,
          this->most_important_sphere_ids[i],
          this->closest_sphere_intersection_depths[i]);
    }
#endif // PULSAR_LOG_TRACKER_PIX
    this->n_hits += 1;
  }

  /**
   * Get the number of hits registered.
   */
  IHD int get_n_hits() const {
    return this->n_hits;
  }

  /**
   * Get the idx closest sphere ID.
   *
   * For example, get_closest_sphere_id(0) gives the overall closest
   * sphere id.
   *
   * This method is implemented for highly optimized scenarios and will *not*
   * perform an index check at runtime if assertions are disabled. idx must be
   * >=0 and < IMIN(n_hits, n_track) for a valid result, if it is >=
   * n_hits it will return -1.
   */
  IHD int get_closest_sphere_id(const int& idx) {
    PASSERT(idx >= 0 && idx < n_track);
    return this->most_important_sphere_ids[idx];
  }

  /**
   * Get the idx closest sphere normalized_depth.
   *
   * For example, get_closest_sphere_depth(0) gives the overall closest
   * sphere depth (normalized).
   *
   * This method is implemented for highly optimized scenarios and will *not*
   * perform an index check at runtime if assertions are disabled. idx must be
   * >=0 and < IMIN(n_hits, n_track) for a valid result, if it is >=
   * n_hits it will return 1. + FEPS.
   */
  IHD float get_closest_sphere_depth(const int& idx) {
    PASSERT(idx >= 0 && idx < n_track);
    return this->closest_sphere_intersection_depths[idx];
  }

 private:
  /** The number of registered hits so far. */
  int n_hits;
  /** The number of intersections to track. Must be <MAX_GRAD_SPHERES. */
  int n_track;
  /** The sphere ids of the n_track spheres with the highest color
   * contribution. */
  int most_important_sphere_ids[MAX_GRAD_SPHERES];
  /** The normalized depths of the closest n_track spheres. */
  float closest_sphere_intersection_depths[MAX_GRAD_SPHERES];
};

} // namespace Renderer
} // namespace pulsar

#endif
