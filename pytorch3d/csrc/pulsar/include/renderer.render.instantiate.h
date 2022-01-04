/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef PULSAR_NATIVE_INCLUDE_RENDERER_RENDER_INSTANTIATE_H_
#define PULSAR_NATIVE_INCLUDE_RENDERER_RENDER_INSTANTIATE_H_

#include "./renderer.render.device.h"

namespace pulsar {
namespace Renderer {
template GLOBAL void render<ISONDEVICE>(
    size_t const* const RESTRICT
        num_balls, /** Number of balls relevant for this pass. */
    IntersectInfo const* const RESTRICT ii_d, /** Intersect information. */
    DrawInfo const* const RESTRICT di_d, /** Draw information. */
    float const* const RESTRICT min_depth_d, /** Minimum depth per sphere. */
    int const* const RESTRICT id_d, /** IDs. */
    float const* const RESTRICT op_d, /** Opacity. */
    const CamInfo cam_norm, /** Camera normalized with all vectors to be in the
                             * camera coordinate system.
                             */
    const float gamma, /** Transparency parameter. **/
    const float percent_allowed_difference, /** Maximum allowed
                                               error in color. */
    const uint max_n_hits,
    const float* bg_col_d,
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
);
}
} // namespace pulsar

#endif
