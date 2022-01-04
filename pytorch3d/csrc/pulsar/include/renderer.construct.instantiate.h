/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef PULSAR_NATIVE_INCLUDE_RENDERER_CONSTRUCT_INSTANTIATE_H_
#define PULSAR_NATIVE_INCLUDE_RENDERER_CONSTRUCT_INSTANTIATE_H_

#include "./renderer.construct.device.h"

namespace pulsar {
namespace Renderer {
template void construct<ISONDEVICE>(
    Renderer* self,
    const size_t& max_num_balls,
    const int& width,
    const int& height,
    const bool& orthogonal_projection,
    const bool& right_handed_system,
    const float& background_normalization_depth,
    const uint& n_channels,
    const uint& n_track);
}
} // namespace pulsar

#endif
