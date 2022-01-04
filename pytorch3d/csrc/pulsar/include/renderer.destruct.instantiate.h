/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef PULSAR_NATIVE_INCLUDE_RENDERER_DESTRUCT_INSTANTIATE_H_
#define PULSAR_NATIVE_INCLUDE_RENDERER_DESTRUCT_INSTANTIATE_H_

#include "./renderer.destruct.device.h"

namespace pulsar {
namespace Renderer {
template void destruct<ISONDEVICE>(Renderer* self);
}
} // namespace pulsar

#endif
