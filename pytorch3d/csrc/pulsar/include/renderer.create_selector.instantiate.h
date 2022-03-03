/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef PULSAR_NATIVE_INCLUDE_RENDERER_CREATE_SELECTOR_INSTANTIATE_H_
#define PULSAR_NATIVE_INCLUDE_RENDERER_CREATE_SELECTOR_INSTANTIATE_H_

#include "./renderer.create_selector.device.h"

namespace pulsar {
namespace Renderer {

template GLOBAL void create_selector<ISONDEVICE>(
    IntersectInfo const* const RESTRICT ii_sorted_d,
    const uint num_balls,
    const int min_x,
    const int max_x,
    const int min_y,
    const int max_y,
    /* Out variables. */
    char* RESTRICT region_flags_d);

}
} // namespace pulsar

#endif
