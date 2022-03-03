/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef PULSAR_LOGGING_H_
#define PULSAR_LOGGING_H_

// #define PULSAR_LOGGING_ENABLED
/**
 * Enable detailed per-operation timings.
 *
 * This timing scheme is not appropriate to measure batched calculations.
 * Use `PULSAR_TIMINGS_BATCHED_ENABLED` for that.
 */
// #define PULSAR_TIMINGS_ENABLED
/**
 * Time batched operations.
 */
// #define PULSAR_TIMINGS_BATCHED_ENABLED
#if defined(PULSAR_TIMINGS_BATCHED_ENABLED) && defined(PULSAR_TIMINGS_ENABLED)
#pragma message("Pulsar|batched and unbatched timings enabled. This will not")
#pragma message("Pulsar|create meaningful results.")
#endif

#ifdef PULSAR_LOGGING_ENABLED

// Control logging.
// 0: INFO, 1: WARNING, 2: ERROR, 3: FATAL (Abort after logging).
#define CAFFE2_LOG_THRESHOLD 0
#define PULSAR_LOG_INIT false
#define PULSAR_LOG_FORWARD false
#define PULSAR_LOG_CALC_SIGNATURE false
#define PULSAR_LOG_RENDER false
#define PULSAR_LOG_RENDER_PIX false
#define PULSAR_LOG_RENDER_PIX_X 428
#define PULSAR_LOG_RENDER_PIX_Y 669
#define PULSAR_LOG_RENDER_PIX_ALL false
#define PULSAR_LOG_TRACKER_PIX false
#define PULSAR_LOG_TRACKER_PIX_X 428
#define PULSAR_LOG_TRACKER_PIX_Y 669
#define PULSAR_LOG_TRACKER_PIX_ALL false
#define PULSAR_LOG_DRAW_PIX false
#define PULSAR_LOG_DRAW_PIX_X 428
#define PULSAR_LOG_DRAW_PIX_Y 669
#define PULSAR_LOG_DRAW_PIX_ALL false
#define PULSAR_LOG_BACKWARD false
#define PULSAR_LOG_GRAD false
#define PULSAR_LOG_GRAD_X 509
#define PULSAR_LOG_GRAD_Y 489
#define PULSAR_LOG_GRAD_ALL false
#define PULSAR_LOG_NORMALIZE false
#define PULSAR_LOG_NORMALIZE_X 0
#define PULSAR_LOG_NORMALIZE_ALL false

#define PULSAR_LOG_DEV(ID, ...) \
  if ((ID)) {                   \
    printf(__VA_ARGS__);        \
  }
#define PULSAR_LOG_DEV_APIX(ID, MSG, ...)                               \
  if ((ID) && (film_coord_x == (ID##_X) && film_coord_y == (ID##_Y)) || \
      ID##_ALL) {                                                       \
    printf(                                                             \
        "%u %u (ap %u %u)|" MSG,                                        \
        film_coord_x,                                                   \
        film_coord_y,                                                   \
        ap_coord_x,                                                     \
        ap_coord_y,                                                     \
        __VA_ARGS__);                                                   \
  }
#define PULSAR_LOG_DEV_PIX(ID, MSG, ...)                                  \
  if ((ID) && (coord_x == (ID##_X) && coord_y == (ID##_Y)) || ID##_ALL) { \
    printf("%u %u|" MSG, coord_x, coord_y, __VA_ARGS__);                  \
  }
#ifdef __CUDACC__
#define PULSAR_LOG_DEV_PIXB(ID, MSG, ...)                       \
  if ((ID) && static_cast<int>(block_area.min.x) <= (ID##_X) && \
      static_cast<int>(block_area.max.x) > (ID##_X) &&          \
      static_cast<int>(block_area.min.y) <= (ID##_Y) &&         \
      static_cast<int>(block_area.max.y) > (ID##_Y)) {          \
    printf("%u %u|" MSG, coord_x, coord_y, __VA_ARGS__);        \
  }
#else
#define PULSAR_LOG_DEV_PIXB(ID, MSG, ...)                   \
  if ((ID) && coord_x == (ID##_X) && coord_y == (ID##_Y)) { \
    printf("%u %u|" MSG, coord_x, coord_y, __VA_ARGS__);    \
  }
#endif
#define PULSAR_LOG_DEV_NODE(ID, MSG, ...)      \
  if ((ID) && idx == (ID##_X) || (ID##_ALL)) { \
    printf("%u|" MSG, idx, __VA_ARGS__);       \
  }

#else

#define CAFFE2_LOG_THRESHOLD 2

#define PULSAR_LOG_RENDER false
#define PULSAR_LOG_INIT false
#define PULSAR_LOG_FORWARD false
#define PULSAR_LOG_BACKWARD false
#define PULSAR_LOG_TRACKER_PIX false

#define PULSAR_LOG_DEV(...)
#define PULSAR_LOG_DEV_APIX(...)
#define PULSAR_LOG_DEV_PIX(...)
#define PULSAR_LOG_DEV_PIXB(...)
#define PULSAR_LOG_DEV_NODE(...)

#endif

#endif
