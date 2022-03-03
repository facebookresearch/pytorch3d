/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "./renderer.norm_sphere_gradients.device.h"

namespace pulsar {
namespace Renderer {

template GLOBAL void norm_sphere_gradients<ISONDEVICE>(
    Renderer renderer,
    const int num_balls);

} // namespace Renderer
} // namespace pulsar
