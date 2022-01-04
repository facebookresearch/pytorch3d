/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef PULSAR_NATIVE_INCLUDE_RENDERER_NORM_CAM_GRADIENTS_DEVICE_H_
#define PULSAR_NATIVE_INCLUDE_RENDERER_NORM_CAM_GRADIENTS_DEVICE_H_

#include "../global.h"
#include "./camera.device.h"
#include "./commands.h"
#include "./math.h"
#include "./renderer.h"

namespace pulsar {
namespace Renderer {

/**
 * Normalize the camera gradients by the number of spheres that contributed.
 */
template <bool DEV>
GLOBAL void norm_cam_gradients(Renderer renderer) {
  GET_PARALLEL_IDX_1D(idx, 1);
  CamGradInfo* cgi = reinterpret_cast<CamGradInfo*>(renderer.grad_cam_d);
  *cgi = *cgi * FRCP(static_cast<float>(*renderer.n_grad_contributions_d));
  END_PARALLEL_NORET();
};

} // namespace Renderer
} // namespace pulsar

#endif
