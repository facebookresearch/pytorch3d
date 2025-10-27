/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef PULSAR_NATIVE_COMMANDS_H_
#define PULSAR_NATIVE_COMMANDS_H_

#ifdef _MSC_VER
#include <intrin.h>
#define __builtin_popcount (int)__popcnt
#endif

// Definitions for CPU commands.
// #include <execution>
// #include <numeric>

namespace cg {
struct coalesced_group {
  INLINE uint thread_rank() const {
    return 0u;
  }
  INLINE uint size() const {
    return 1u;
  }
  INLINE uint ballot(uint val) const {
    return static_cast<uint>(val > 0);
  }
};

struct thread_block {
  INLINE uint thread_rank() const {
    return 0u;
  }
  INLINE uint size() const {
    return 1u;
  }
  INLINE void sync() const {}
};

INLINE coalesced_group coalesced_threads() {
  coalesced_group ret;
  return ret;
}

INLINE thread_block this_thread_block() {
  thread_block ret;
  return ret;
}
} // namespace cg
#define SHFL_SYNC(a, b, c) (b)
template <typename T>
T WARP_CUMSUM(
    const cg::coalesced_group& group,
    const uint& mask,
    const T& base) {
  return base;
}

template <typename T>
DEVICE T
WARP_MAX(const cg::coalesced_group& group, const uint& mask, const T& base) {
  return base;
}

template <typename T>
DEVICE T
WARP_SUM(const cg::coalesced_group& group, const uint& mask, const T& base) {
  return base;
}

INLINE DEVICE float3 WARP_SUM_FLOAT3(
    const cg::coalesced_group& group,
    const uint& mask,
    const float3& base) {
  return base;
}

#define ACTIVEMASK() (1u << 31)
#define ALIGN(VAL)
#define SYNC()
#define THREADFENCE_B()
#define BALLOT(mask, val) (val != 0)
#define SHARED
// Floating point.
#define FMAX(a, b) std::fmax((a), (b))
#define FMIN(a, b) std::fmin((a), (b))
INLINE float atomicMax(float* address, float val) {
  *address = std::max(*address, val);
  return *address;
}
INLINE float atomicMin(float* address, float val) {
  *address = std::min(*address, val);
  return *address;
}
#define FMUL(a, b) ((a) * (b))
#define FDIV(a, b) ((a) / (b))
#define FSUB(a, b) ((a) - (b))
#define FABSLEQAS(a, b, c) \
  ((a) <= (b) ? FSUB((b), (a)) <= (c) : FSUB((a), (b)) < (c))
#define FADD(a, b) ((a) + (b))
#define FSQRT(a) sqrtf(a)
#define FEXP(a) fasterexp(a)
#define FLN(a) fasterlog(a)
#define FPOW(a, b) powf((a), (b))
#define FROUND(x) roundf(x)
#define FCEIL(a) ceilf(a)
#define FFLOOR(a) floorf(a)
#define FSATURATE(x) std::max(0.f, std::min(1.f, x))
#define FABS(a) abs(a)
#define FMA(x, y, z) ((x) * (y) + (z))
#define I2F(a) static_cast<float>(a)
#define FRCP(x) (1.f / (x))
#define IASF(x, loc) memcpy(&(loc), &(x), sizeof(x))
#define FASI(x, loc) memcpy(&(loc), &(x), sizeof(x))
#define DMAX(a, b) std::max((a), (b))
#define DMIN(a, b) std::min((a), (b))
#define DSATURATE(a) DMIN(1., DMAX(0., (a)))
#define DSQRT(a) sqrt(a)
//
//
//
//
//
//
//
//
//
//
//
//
// uint.
#define CLZ(VAL) _clz(VAL)
template <typename T>
INLINE T ATOMICADD(T* address, T val) {
  T old = *address;
  *address += val;
  return old;
}
template <typename T>
INLINE void ATOMICADD_F3(T* address, T val) {
  ATOMICADD(&(address->x), val.x);
  ATOMICADD(&(address->y), val.y);
  ATOMICADD(&(address->z), val.z);
}
#define ATOMICADD_B(a, b) ATOMICADD((a), (b))
#define POPC(a) __builtin_popcount(a)

// int.
#define IMIN(a, b) std::min((a), (b))
#define IMAX(a, b) std::max((a), (b))
#define IABS(a) abs(a)

// Checks.
// like TORCH_CHECK_ARG in PyTorch > 1.10
#define ARGCHECK(cond, argN, ...) \
  TORCH_CHECK(cond, "invalid argument ", argN, ": ", __VA_ARGS__)

// Math.
#define NORM3DF(x, y, z) sqrtf(x* x + y * y + z * z)
#define RNORM3DF(x, y, z) (1.f / sqrtf(x * x + y * y + z * z))

// High level.
#define PREFETCH(PTR)
#define GET_SORT_WS_SIZE(RES_PTR, KEY_TYPE, VAL_TYPE, NUM_OBJECTS) \
  *(RES_PTR) = 0;
#define GET_REDUCE_WS_SIZE(RES_PTR, TYPE, REDUCE_OP, NUM_OBJECTS) \
  *(RES_PTR) = 0;
#define GET_SELECT_WS_SIZE(                              \
    RES_PTR, TYPE_SELECTOR, TYPE_SELECTION, NUM_OBJECTS) \
  *(RES_PTR) = 0;
#define GET_SUM_WS_SIZE(RES_PTR, TYPE_SUM, NUM_OBJECTS) *(RES_PTR) = 0;
#define GET_MM_WS_SIZE(RES_PTR, TYPE, NUM_OBJECTS) *(RES_PTR) = 0;

#define SORT_DESCENDING(                                                     \
    TMPN1, SORT_PTR, SORTED_PTR, VAL_PTR, VAL_SORTED_PTR, NUM_OBJECTS)       \
  std::vector<size_t> TMPN1(NUM_OBJECTS);                                    \
  std::iota(TMPN1.begin(), TMPN1.end(), 0);                                  \
  const auto TMPN1##_val_ptr = (SORT_PTR);                                   \
  std::sort(                                                                 \
      TMPN1.begin(), TMPN1.end(), [&TMPN1##_val_ptr](size_t i1, size_t i2) { \
        return TMPN1##_val_ptr[i1] > TMPN1##_val_ptr[i2];                    \
      });                                                                    \
  for (int i = 0; i < (NUM_OBJECTS); ++i) {                                  \
    (SORTED_PTR)[i] = (SORT_PTR)[TMPN1[i]];                                  \
  }                                                                          \
  for (int i = 0; i < (NUM_OBJECTS); ++i) {                                  \
    (VAL_SORTED_PTR)[i] = (VAL_PTR)[TMPN1[i]];                               \
  }

#define SORT_ASCENDING(                                                 \
    SORT_PTR, SORTED_PTR, VAL_PTR, VAL_SORTED_PTR, NUM_OBJECTS, STREAM) \
  {                                                                     \
    std::vector<size_t> TMPN1(NUM_OBJECTS);                             \
    std::iota(TMPN1.begin(), TMPN1.end(), 0);                           \
    const auto TMPN1_val_ptr = (SORT_PTR);                              \
    std::sort(                                                          \
        TMPN1.begin(),                                                  \
        TMPN1.end(),                                                    \
        [&TMPN1_val_ptr](size_t i1, size_t i2) -> bool {                \
          return TMPN1_val_ptr[i1] < TMPN1_val_ptr[i2];                 \
        });                                                             \
    for (int i = 0; i < (NUM_OBJECTS); ++i) {                           \
      (SORTED_PTR)[i] = (SORT_PTR)[TMPN1[i]];                           \
    }                                                                   \
    for (int i = 0; i < (NUM_OBJECTS); ++i) {                           \
      (VAL_SORTED_PTR)[i] = (VAL_PTR)[TMPN1[i]];                        \
    }                                                                   \
  }

#define SORT_DESCENDING_WS( \
    TMPN1,                  \
    SORT_PTR,               \
    SORTED_PTR,             \
    VAL_PTR,                \
    VAL_SORTED_PTR,         \
    NUM_OBJECTS,            \
    WORSPACE_PTR,           \
    WORKSPACE_SIZE)         \
  SORT_DESCENDING(          \
      TMPN1, SORT_PTR, SORTED_PTR, VAL_PTR, VAL_SORTED_PTR, NUM_OBJECTS)

#define SORT_ASCENDING_WS( \
    SORT_PTR,              \
    SORTED_PTR,            \
    VAL_PTR,               \
    VAL_SORTED_PTR,        \
    NUM_OBJECTS,           \
    WORSPACE_PTR,          \
    WORKSPACE_SIZE,        \
    STREAM)                \
  SORT_ASCENDING(          \
      SORT_PTR, SORTED_PTR, VAL_PTR, VAL_SORTED_PTR, NUM_OBJECTS, STREAM)

#define REDUCE(REDUCE_PTR, RESULT_PTR, NUM_ITEMS, REDUCE_OP, REDUCE_INIT) \
  {                                                                       \
    *(RESULT_PTR) = (REDUCE_INIT);                                        \
    for (int i = 0; i < (NUM_ITEMS); ++i) {                               \
      *(RESULT_PTR) = REDUCE_OP(*(RESULT_PTR), (REDUCE_PTR)[i]);          \
    }                                                                     \
  }
#define REDUCE_WS(  \
    REDUCE_PTR,     \
    RESULT_PTR,     \
    NUM_ITEMS,      \
    REDUCE_OP,      \
    REDUCE_INIT,    \
    WORKSPACE_PTR,  \
    WORKSPACE_SIZE, \
    STREAM)         \
  REDUCE(REDUCE_PTR, RESULT_PTR, NUM_ITEMS, REDUCE_OP, REDUCE_INIT)

#define SELECT_FLAGS_WS(                    \
    FLAGS_PTR,                              \
    ITEM_PTR,                               \
    OUT_PTR,                                \
    NUM_SELECTED_PTR,                       \
    NUM_ITEMS,                              \
    WORKSPACE_PTR,                          \
    WORSPACE_BYTES,                         \
    STREAM)                                 \
  {                                         \
    *NUM_SELECTED_PTR = 0;                  \
    ptrdiff_t write_pos = 0;                \
    for (int i = 0; i < NUM_ITEMS; ++i) {   \
      if (FLAGS_PTR[i]) {                   \
        OUT_PTR[write_pos++] = ITEM_PTR[i]; \
        *NUM_SELECTED_PTR += 1;             \
      }                                     \
    }                                       \
  }

template <typename T>
void SUM_WS(
    T* SUM_PTR,
    T* OUT_PTR,
    size_t NUM_OBJECTS,
    char* WORKSPACE_PTR,
    size_t WORKSPACE_BYTES,
    cudaStream_t STREAM) {
  *(OUT_PTR) = T();
  for (int i = 0; i < (NUM_OBJECTS); ++i) {
    *(OUT_PTR) = *(OUT_PTR) + (SUM_PTR)[i];
  }
}

template <typename T>
void MIN_WS(
    T* MIN_PTR,
    T* OUT_PTR,
    size_t NUM_OBJECTS,
    char* WORKSPACE_PTR,
    size_t WORKSPACE_BYTES,
    cudaStream_t STREAM) {
  *(OUT_PTR) = T();
  for (int i = 0; i < (NUM_OBJECTS); ++i) {
    *(OUT_PTR) = std::min<T>(*(OUT_PTR), (MIN_PTR)[i]);
  }
}

template <typename T>
void MAX_WS(
    T* MAX_PTR,
    T* OUT_PTR,
    size_t NUM_OBJECTS,
    char* WORKSPACE_PTR,
    size_t WORKSPACE_BYTES,
    cudaStream_t STREAM) {
  *(OUT_PTR) = T();
  for (int i = 0; i < (NUM_OBJECTS); ++i) {
    *(OUT_PTR) = std::max<T>(*(OUT_PTR), (MAX_PTR)[i]);
  }
}
//
//
//
//
#define COPY_HOST_DEV(PTR_D, PTR_H, TYPE, SIZE) \
  std::memcpy((PTR_D), (PTR_H), sizeof(TYPE) * (SIZE))
//
#define COPY_DEV_HOST(PTR_H, PTR_D, TYPE, SIZE) \
  std::memcpy((PTR_H), (PTR_D), sizeof(TYPE) * (SIZE))
//
#define COPY_DEV_DEV(PTR_T, PTR_S, TYPE, SIZE) \
  std::memcpy((PTR_T), (PTR_S), sizeof(TYPE) * SIZE)
//

#define MALLOC(VAR, TYPE, SIZE) MALLOC_HOST(VAR, TYPE, SIZE)
#define FREE(PTR) FREE_HOST(PTR)
#define MEMSET(VAR, VAL, TYPE, SIZE, STREAM) \
  memset((VAR), (VAL), sizeof(TYPE) * (SIZE))
//

#define LAUNCH_MAX_PARALLEL_1D(FUNC, N, STREAM, ...) FUNC(__VA_ARGS__);
#define LAUNCH_PARALLEL_1D(FUNC, N, TN, STREAM, ...) FUNC(__VA_ARGS__);
#define LAUNCH_MAX_PARALLEL_2D(FUNC, NX, NY, STREAM, ...) FUNC(__VA_ARGS__);
#define LAUNCH_PARALLEL_2D(FUNC, NX, NY, TX, TY, STREAM, ...) FUNC(__VA_ARGS__);
//
//
//
//
//
#define GET_PARALLEL_IDX_1D(VARNAME, N) \
  for (uint VARNAME = 0; VARNAME < (N); ++VARNAME) {
#define GET_PARALLEL_IDS_2D(VAR_X, VAR_Y, WIDTH, HEIGHT)          \
  int2 blockDim;                                                  \
  blockDim.x = 1;                                                 \
  blockDim.y = 1;                                                 \
  uint __parallel_2d_width = WIDTH;                               \
  uint __parallel_2d_height = HEIGHT;                             \
  for (uint VAR_Y = 0; VAR_Y < __parallel_2d_height; ++(VAR_Y)) { \
    for (uint VAR_X = 0; VAR_X < __parallel_2d_width; ++(VAR_X)) {
//
//
//
#define END_PARALLEL() \
  end_parallel:;       \
  }
#define END_PARALLEL_NORET() }
#define END_PARALLEL_2D() \
  end_parallel:;          \
  }                       \
  }
#define END_PARALLEL_2D_NORET() \
  }                             \
  }
#define RETURN_PARALLEL() goto end_parallel;
#define CHECKLAUNCH()
#define ISONDEVICE false
#define SYNCDEVICE()
#define START_TIME(TN) \
  auto __time_start_##TN = std::chrono::steady_clock::now();
#define STOP_TIME(TN) auto __time_stop_##TN = std::chrono::steady_clock::now();
#define GET_TIME(TN, TOPTR)                                       \
  *TOPTR = std::chrono::duration_cast<std::chrono::milliseconds>( \
               __time_stop_##TN - __time_start_##TN)              \
               .count()
#define START_TIME_CU(TN)                          \
  cudaEvent_t __time_start_##TN, __time_stop_##TN; \
  cudaEventCreate(&__time_start_##TN);             \
  cudaEventCreate(&__time_stop_##TN);              \
  cudaEventRecord(__time_start_##TN);
#define STOP_TIME_CU(TN) cudaEventRecord(__time_stop_##TN);
#define GET_TIME_CU(TN, TOPTR)            \
  cudaEventSynchronize(__time_stop_##TN); \
  cudaEventElapsedTime((TOPTR), __time_start_##TN, __time_stop_##TN);

#endif
