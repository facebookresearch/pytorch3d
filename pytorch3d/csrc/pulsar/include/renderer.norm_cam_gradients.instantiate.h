/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "./renderer.norm_cam_gradients.device.h"

namespace pulsar {
namespace Renderer {

template GLOBAL void norm_cam_gradients<ISONDEVICE>(Renderer renderer);

} // namespace Renderer
} // namespace pulsar
