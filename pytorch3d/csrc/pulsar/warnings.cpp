// Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
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
