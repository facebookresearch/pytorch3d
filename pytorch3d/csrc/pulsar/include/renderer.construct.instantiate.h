// Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
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
