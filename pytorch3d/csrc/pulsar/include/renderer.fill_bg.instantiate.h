// Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#include "./renderer.fill_bg.device.h"

namespace pulsar {
namespace Renderer {

template GLOBAL void fill_bg<ISONDEVICE>(
    Renderer renderer,
    const CamInfo norm,
    float const* const bg_col_d,
    const float gamma,
    const uint mode);

} // namespace Renderer
} // namespace pulsar
