/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "./renderer.calc_gradients.device.h"

namespace pulsar {
namespace Renderer {

template GLOBAL void calc_gradients<ISONDEVICE>(
    const CamInfo cam, /** Camera in world coordinates. */
    float const* const RESTRICT grad_im, /** The gradient image. */
    const float
        gamma, /** The transparency parameter used in the forward pass. */
    float3 const* const RESTRICT vert_poss, /** Vertex position vector. */
    float const* const RESTRICT vert_cols, /** Vertex color vector. */
    float const* const RESTRICT vert_rads, /** Vertex radius vector. */
    float const* const RESTRICT opacity, /** Vertex opacity. */
    const uint num_balls, /** Number of balls. */
    float const* const RESTRICT result_d, /** Result image. */
    float const* const RESTRICT forw_info_d, /** Forward pass info. */
    DrawInfo const* const RESTRICT di_d, /** Draw information. */
    IntersectInfo const* const RESTRICT ii_d, /** Intersect information. */
    // Mode switches.
    const bool calc_grad_pos,
    const bool calc_grad_col,
    const bool calc_grad_rad,
    const bool calc_grad_cam,
    const bool calc_grad_opy,
    // Out variables.
    float* const RESTRICT grad_rad_d, /** Radius gradients. */
    float* const RESTRICT grad_col_d, /** Color gradients. */
    float3* const RESTRICT grad_pos_d, /** Position gradients. */
    CamGradInfo* const RESTRICT grad_cam_buf_d, /** Camera gradient buffer. */
    float* const RESTRICT grad_opy_d, /** Opacity gradient buffer. */
    int* const RESTRICT
        grad_contributed_d, /** Gradient contribution counter. */
    // Infrastructure.
    const int n_track,
    const uint offs_x,
    const uint offs_y);

} // namespace Renderer
} // namespace pulsar
