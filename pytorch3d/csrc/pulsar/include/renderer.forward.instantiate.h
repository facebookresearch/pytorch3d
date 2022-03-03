/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "./renderer.forward.device.h"

namespace pulsar {
namespace Renderer {

template void forward<ISONDEVICE>(
    Renderer* self,
    const float* vert_pos,
    const float* vert_col,
    const float* vert_rad,
    const CamInfo& cam,
    const float& gamma,
    float percent_allowed_difference,
    const uint& max_n_hits,
    const float* bg_col_d,
    const float* opacity_d,
    const size_t& num_balls,
    const uint& mode,
    cudaStream_t stream);

} // namespace Renderer
} // namespace pulsar
