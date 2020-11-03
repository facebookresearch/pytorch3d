// Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#ifndef PULSAR_NATIVE_INCLUDE_RENDERER_CALC_SIGNATURE_INSTANTIATE_H_
#define PULSAR_NATIVE_INCLUDE_RENDERER_CALC_SIGNATURE_INSTANTIATE_H_

#include "./renderer.calc_signature.device.h"

namespace pulsar {
namespace Renderer {
template GLOBAL void calc_signature<ISONDEVICE>(
    Renderer renderer,
    float3 const* const RESTRICT vert_poss,
    float const* const RESTRICT vert_cols,
    float const* const RESTRICT vert_rads,
    const uint num_balls);
}
} // namespace pulsar

#endif
