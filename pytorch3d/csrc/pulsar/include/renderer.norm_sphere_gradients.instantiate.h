// Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#include "./renderer.norm_sphere_gradients.device.h"

namespace pulsar {
namespace Renderer {

template GLOBAL void norm_sphere_gradients<ISONDEVICE>(
    Renderer renderer,
    const int num_balls);

} // namespace Renderer
} // namespace pulsar
