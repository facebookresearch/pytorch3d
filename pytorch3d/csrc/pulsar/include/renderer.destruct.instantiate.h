// Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#ifndef PULSAR_NATIVE_INCLUDE_RENDERER_DESTRUCT_INSTANTIATE_H_
#define PULSAR_NATIVE_INCLUDE_RENDERER_DESTRUCT_INSTANTIATE_H_

#include "./renderer.destruct.device.h"

namespace pulsar {
namespace Renderer {
template void destruct<ISONDEVICE>(Renderer* self);
}
} // namespace pulsar

#endif
