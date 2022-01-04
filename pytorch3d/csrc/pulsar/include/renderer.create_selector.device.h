/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef PULSAR_NATIVE_INCLUDE_RENDERER_CREATE_SELECTOR_DEVICE_H_
#define PULSAR_NATIVE_INCLUDE_RENDERER_CREATE_SELECTOR_DEVICE_H_

#include "../global.h"
#include "./commands.h"
#include "./renderer.h"

namespace pulsar {
namespace Renderer {

template <bool DEV>
GLOBAL void create_selector(
    IntersectInfo const* const RESTRICT ii_sorted_d,
    const uint num_balls,
    const int min_x,
    const int max_x,
    const int min_y,
    const int max_y,
    /* Out variables. */
    char* RESTRICT region_flags_d) {
  GET_PARALLEL_IDX_1D(idx, num_balls);
  bool hit = (static_cast<int>(ii_sorted_d[idx].min.x) <= max_x) &&
      (static_cast<int>(ii_sorted_d[idx].max.x) > min_x) &&
      (static_cast<int>(ii_sorted_d[idx].min.y) <= max_y) &&
      (static_cast<int>(ii_sorted_d[idx].max.y) > min_y);
  region_flags_d[idx] = hit;
  END_PARALLEL_NORET();
}

} // namespace Renderer
} // namespace pulsar

#endif
