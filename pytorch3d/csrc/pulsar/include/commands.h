/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef PULSAR_NATIVE_COMMANDS_ROUTING_H_
#define PULSAR_NATIVE_COMMANDS_ROUTING_H_

#include "../global.h"

// Commands available everywhere.
#define MALLOC_HOST(VAR, TYPE, SIZE) \
  VAR = static_cast<TYPE*>(malloc(sizeof(TYPE) * (SIZE)))
#define FREE_HOST(PTR) free(PTR)

/* Include command definitions depending on CPU or GPU use. */

#ifdef __CUDACC__
// TODO: find out which compiler we're using here and use the suppression.
// #pragma push
// #pragma diag_suppress = 68
#include <ATen/cuda/CUDAContext.h>
// #pragma pop
#include "../gpu/commands.h"
#else
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Weverything"
#pragma clang diagnostic pop
#include "../host/commands.h"
#endif

#endif
