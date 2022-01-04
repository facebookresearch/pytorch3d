/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "./global.h"
#include "./logging.h"

/**
 * A compilation unit to provide warnings about the code and avoid
 * repeated messages.
 */
#ifdef PULSAR_ASSERTIONS
#pragma message("WARNING: assertions are enabled in Pulsar.")
#endif
#ifdef PULSAR_LOGGING_ENABLED
#pragma message("WARNING: logging is enabled in Pulsar.")
#endif
