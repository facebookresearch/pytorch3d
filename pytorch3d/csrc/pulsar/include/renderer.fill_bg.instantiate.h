/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

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
