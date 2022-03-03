/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef PULSAR_NATIVE_CUDA_COMMANDS_H_
#define PULSAR_NATIVE_CUDA_COMMANDS_H_

// Definitions for GPU commands.
#include <cooperative_groups.h>
#include <cub/cub.cuh>
namespace cg = cooperative_groups;

#ifdef __DRIVER_TYPES_H__
#ifndef DEVICE_RESET
#define DEVICE_RESET cudaDeviceReset();
#endif
#else
#ifndef DEVICE_RESET
#define DEVICE_RESET
#endif
#endif

#define HANDLECUDA(CMD) CMD
// handleCudaError((CMD), __FILE__, __LINE__)
inline void
handleCudaError(const cudaError_t err, const char* file, const int line) {
  if (err != cudaSuccess) {
#ifndef __NVCC__
    fprintf(
        stderr,
        "%s(%i) : getLastCudaError() CUDA error :"
        " (%d) %s.\n",
        file,
        line,
        static_cast<int>(err),
        cudaGetErrorString(err));
    DEVICE_RESET
    exit(1);
#endif
  }
}
inline void
getLastCudaError(const char* errorMessage, const char* file, const int line) {
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "Error: %s.", errorMessage);
    handleCudaError(err, file, line);
  }
}

#define ALIGN(VAL) __align__(VAL)
#define SYNC() HANDLECUDE(cudaDeviceSynchronize())
#define THREADFENCE_B() __threadfence_block()
#define SHFL_SYNC(a, b, c) __shfl_sync((a), (b), (c))
#define SHARED __shared__
#define ACTIVEMASK() __activemask()
#define BALLOT(mask, val) __ballot_sync((mask), val)
/**
 * Find the cumulative sum within a warp up to the current
 * thread lane, with each mask thread contributing base.
 */
template <typename T>
DEVICE T
WARP_CUMSUM(const cg::coalesced_group& group, const uint& mask, const T& base) {
  T ret = base;
  T shfl_val;
  shfl_val = __shfl_down_sync(mask, ret, 1u); // Deactivate the rightmost lane.
  ret += (group.thread_rank() < 31) * shfl_val;
  shfl_val = __shfl_down_sync(mask, ret, 2u);
  ret += (group.thread_rank() < 30) * shfl_val;
  shfl_val = __shfl_down_sync(mask, ret, 4u); // ...4
  ret += (group.thread_rank() < 28) * shfl_val;
  shfl_val = __shfl_down_sync(mask, ret, 8u); // ...8
  ret += (group.thread_rank() < 24) * shfl_val;
  shfl_val = __shfl_down_sync(mask, ret, 16u); // ...16
  ret += (group.thread_rank() < 16) * shfl_val;
  return ret;
}

template <typename T>
DEVICE T
WARP_MAX(const cg::coalesced_group& group, const uint& mask, const T& base) {
  T ret = base;
  ret = max(ret, __shfl_down_sync(mask, ret, 16u));
  ret = max(ret, __shfl_down_sync(mask, ret, 8u));
  ret = max(ret, __shfl_down_sync(mask, ret, 4u));
  ret = max(ret, __shfl_down_sync(mask, ret, 2u));
  ret = max(ret, __shfl_down_sync(mask, ret, 1u));
  return ret;
}

template <typename T>
DEVICE T
WARP_SUM(const cg::coalesced_group& group, const uint& mask, const T& base) {
  T ret = base;
  ret = ret + __shfl_down_sync(mask, ret, 16u);
  ret = ret + __shfl_down_sync(mask, ret, 8u);
  ret = ret + __shfl_down_sync(mask, ret, 4u);
  ret = ret + __shfl_down_sync(mask, ret, 2u);
  ret = ret + __shfl_down_sync(mask, ret, 1u);
  return ret;
}

INLINE DEVICE float3 WARP_SUM_FLOAT3(
    const cg::coalesced_group& group,
    const uint& mask,
    const float3& base) {
  float3 ret = base;
  ret.x = WARP_SUM(group, mask, base.x);
  ret.y = WARP_SUM(group, mask, base.y);
  ret.z = WARP_SUM(group, mask, base.z);
  return ret;
}

// Floating point.
// #define FMUL(a, b) __fmul_rn((a), (b))
#define FMUL(a, b) ((a) * (b))
#define FDIV(a, b) __fdiv_rn((a), (b))
// #define FSUB(a, b) __fsub_rn((a), (b))
#define FSUB(a, b) ((a) - (b))
#define FADD(a, b) __fadd_rn((a), (b))
#define FSQRT(a) __fsqrt_rn(a)
#define FEXP(a) fasterexp(a)
#define FLN(a) fasterlog(a)
#define FPOW(a, b) __powf((a), (b))
#define FMAX(a, b) fmax((a), (b))
#define FMIN(a, b) fmin((a), (b))
#define FCEIL(a) ceilf(a)
#define FFLOOR(a) floorf(a)
#define FROUND(x) nearbyintf(x)
#define FSATURATE(x) __saturatef(x)
#define FABS(a) abs(a)
#define IASF(a, loc) (loc) = __int_as_float(a)
#define FASI(a, loc) (loc) = __float_as_int(a)
#define FABSLEQAS(a, b, c) \
  ((a) <= (b) ? FSUB((b), (a)) <= (c) : FSUB((a), (b)) < (c))
/** Calculates x*y+z. */
#define FMA(x, y, z) __fmaf_rn((x), (y), (z))
#define I2F(a) __int2float_rn(a)
#define FRCP(x) __frcp_rn(x)
__device__ static float atomicMax(float* address, float val) {
  int* address_as_i = (int*)address;
  int old = *address_as_i, assumed;
  do {
    assumed = old;
    old = ::atomicCAS(
        address_as_i,
        assumed,
        __float_as_int(::fmaxf(val, __int_as_float(assumed))));
  } while (assumed != old);
  return __int_as_float(old);
}
__device__ static float atomicMin(float* address, float val) {
  int* address_as_i = (int*)address;
  int old = *address_as_i, assumed;
  do {
    assumed = old;
    old = ::atomicCAS(
        address_as_i,
        assumed,
        __float_as_int(::fminf(val, __int_as_float(assumed))));
  } while (assumed != old);
  return __int_as_float(old);
}
#define DMAX(a, b) FMAX(a, b)
#define DMIN(a, b) FMIN(a, b)
#define DSQRT(a) sqrt(a)
#define DSATURATE(a) DMIN(1., DMAX(0., (a)))
// half
#define HADD(a, b) __hadd((a), (b))
#define HSUB2(a, b) __hsub2((a), (b))
#define HMUL2(a, b) __hmul2((a), (b))
#define HSQRT(a) hsqrt(a)

// uint.
#define CLZ(VAL) __clz(VAL)
#define POPC(a) __popc(a)
//
//
//
//
//
//
//
//
//
#define ATOMICADD(PTR, VAL) atomicAdd((PTR), (VAL))
#define ATOMICADD_F3(PTR, VAL)   \
  ATOMICADD(&((PTR)->x), VAL.x); \
  ATOMICADD(&((PTR)->y), VAL.y); \
  ATOMICADD(&((PTR)->z), VAL.z);
#if (CUDART_VERSION >= 10000) && (__CUDA_ARCH__ >= 600)
#define ATOMICADD_B(PTR, VAL) atomicAdd_block((PTR), (VAL))
#else
#define ATOMICADD_B(PTR, VAL) ATOMICADD(PTR, VAL)
#endif
//
//
//
//
// int.
#define IMIN(a, b) min((a), (b))
#define IMAX(a, b) max((a), (b))
#define IABS(a) abs(a)

// Checks.
// like TORCH_CHECK_ARG in PyTorch > 1.10
#define ARGCHECK(cond, argN, ...) \
  TORCH_CHECK(cond, "invalid argument ", argN, ": ", __VA_ARGS__)

// Math.
#define NORM3DF(x, y, z) norm3df(x, y, z)
#define RNORM3DF(x, y, z) rnorm3df(x, y, z)

// High level.
#define GET_SORT_WS_SIZE(RES_PTR, KEY_TYPE, VAL_TYPE, NUM_OBJECTS) \
  cub::DeviceRadixSort::SortPairsDescending(                       \
      (void*)NULL,                                                 \
      *(RES_PTR),                                                  \
      reinterpret_cast<KEY_TYPE*>(NULL),                           \
      reinterpret_cast<KEY_TYPE*>(NULL),                           \
      reinterpret_cast<VAL_TYPE*>(NULL),                           \
      reinterpret_cast<VAL_TYPE*>(NULL),                           \
      (NUM_OBJECTS));
#define GET_REDUCE_WS_SIZE(RES_PTR, TYPE, REDUCE_OP, NUM_OBJECTS) \
  {                                                               \
    TYPE init = TYPE();                                           \
    cub::DeviceReduce::Reduce(                                    \
        (void*)NULL,                                              \
        *(RES_PTR),                                               \
        (TYPE*)NULL,                                              \
        (TYPE*)NULL,                                              \
        (NUM_OBJECTS),                                            \
        (REDUCE_OP),                                              \
        init);                                                    \
  }
#define GET_SELECT_WS_SIZE(                              \
    RES_PTR, TYPE_SELECTOR, TYPE_SELECTION, NUM_OBJECTS) \
  {                                                      \
    cub::DeviceSelect::Flagged(                          \
        (void*)NULL,                                     \
        *(RES_PTR),                                      \
        (TYPE_SELECTION*)NULL,                           \
        (TYPE_SELECTOR*)NULL,                            \
        (TYPE_SELECTION*)NULL,                           \
        (int*)NULL,                                      \
        (NUM_OBJECTS));                                  \
  }
#define GET_SUM_WS_SIZE(RES_PTR, TYPE_SUM, NUM_OBJECTS) \
  {                                                     \
    cub::DeviceReduce::Sum(                             \
        (void*)NULL,                                    \
        *(RES_PTR),                                     \
        (TYPE_SUM*)NULL,                                \
        (TYPE_SUM*)NULL,                                \
        NUM_OBJECTS);                                   \
  }
#define GET_MM_WS_SIZE(RES_PTR, TYPE, NUM_OBJECTS)                         \
  {                                                                        \
    TYPE init = TYPE();                                                    \
    cub::DeviceReduce::Max(                                                \
        (void*)NULL, *(RES_PTR), (TYPE*)NULL, (TYPE*)NULL, (NUM_OBJECTS)); \
  }
#define SORT_DESCENDING(                                               \
    TMPN1, SORT_PTR, SORTED_PTR, VAL_PTR, VAL_SORTED_PTR, NUM_OBJECTS) \
  void* TMPN1 = NULL;                                                  \
  size_t TMPN1##_bytes = 0;                                            \
  cub::DeviceRadixSort::SortPairsDescending(                           \
      TMPN1,                                                           \
      TMPN1##_bytes,                                                   \
      (SORT_PTR),                                                      \
      (SORTED_PTR),                                                    \
      (VAL_PTR),                                                       \
      (VAL_SORTED_PTR),                                                \
      (NUM_OBJECTS));                                                  \
  HANDLECUDA(cudaMalloc(&TMPN1, TMPN1##_bytes));                       \
  cub::DeviceRadixSort::SortPairsDescending(                           \
      TMPN1,                                                           \
      TMPN1##_bytes,                                                   \
      (SORT_PTR),                                                      \
      (SORTED_PTR),                                                    \
      (VAL_PTR),                                                       \
      (VAL_SORTED_PTR),                                                \
      (NUM_OBJECTS));                                                  \
  HANDLECUDA(cudaFree(TMPN1));
#define SORT_DESCENDING_WS(                  \
    TMPN1,                                   \
    SORT_PTR,                                \
    SORTED_PTR,                              \
    VAL_PTR,                                 \
    VAL_SORTED_PTR,                          \
    NUM_OBJECTS,                             \
    WORKSPACE_PTR,                           \
    WORKSPACE_BYTES)                         \
  cub::DeviceRadixSort::SortPairsDescending( \
      (WORKSPACE_PTR),                       \
      (WORKSPACE_BYTES),                     \
      (SORT_PTR),                            \
      (SORTED_PTR),                          \
      (VAL_PTR),                             \
      (VAL_SORTED_PTR),                      \
      (NUM_OBJECTS));
#define SORT_ASCENDING_WS(         \
    SORT_PTR,                      \
    SORTED_PTR,                    \
    VAL_PTR,                       \
    VAL_SORTED_PTR,                \
    NUM_OBJECTS,                   \
    WORKSPACE_PTR,                 \
    WORKSPACE_BYTES,               \
    STREAM)                        \
  cub::DeviceRadixSort::SortPairs( \
      (WORKSPACE_PTR),             \
      (WORKSPACE_BYTES),           \
      (SORT_PTR),                  \
      (SORTED_PTR),                \
      (VAL_PTR),                   \
      (VAL_SORTED_PTR),            \
      (NUM_OBJECTS),               \
      0,                           \
      sizeof(*(SORT_PTR)) * 8,     \
      (STREAM));
#define SUM_WS(                                                            \
    SUM_PTR, OUT_PTR, NUM_OBJECTS, WORKSPACE_PTR, WORKSPACE_BYTES, STREAM) \
  cub::DeviceReduce::Sum(                                                  \
      (WORKSPACE_PTR),                                                     \
      (WORKSPACE_BYTES),                                                   \
      (SUM_PTR),                                                           \
      (OUT_PTR),                                                           \
      (NUM_OBJECTS),                                                       \
      (STREAM));
#define MIN_WS(                                                            \
    MIN_PTR, OUT_PTR, NUM_OBJECTS, WORKSPACE_PTR, WORKSPACE_BYTES, STREAM) \
  cub::DeviceReduce::Min(                                                  \
      (WORKSPACE_PTR),                                                     \
      (WORKSPACE_BYTES),                                                   \
      (MIN_PTR),                                                           \
      (OUT_PTR),                                                           \
      (NUM_OBJECTS),                                                       \
      (STREAM));
#define MAX_WS(                                                            \
    MAX_PTR, OUT_PTR, NUM_OBJECTS, WORKSPACE_PTR, WORKSPACE_BYTES, STREAM) \
  cub::DeviceReduce::Min(                                                  \
      (WORKSPACE_PTR),                                                     \
      (WORKSPACE_BYTES),                                                   \
      (MAX_PTR),                                                           \
      (OUT_PTR),                                                           \
      (NUM_OBJECTS),                                                       \
      (STREAM));
//
//
//
// TODO: rewrite using nested contexts instead of temporary names.
#define REDUCE(REDUCE_PTR, RESULT_PTR, NUM_ITEMS, REDUCE_OP, REDUCE_INIT) \
  cub::DeviceReduce::Reduce(                                              \
      TMPN1,                                                              \
      TMPN1##_bytes,                                                      \
      (REDUCE_PTR),                                                       \
      (RESULT_PTR),                                                       \
      (NUM_ITEMS),                                                        \
      (REDUCE_OP),                                                        \
      (REDUCE_INIT));                                                     \
  HANDLECUDA(cudaMalloc(&TMPN1, TMPN1##_bytes));                          \
  cub::DeviceReduce::Reduce(                                              \
      TMPN1,                                                              \
      TMPN1##_bytes,                                                      \
      (REDUCE_PTR),                                                       \
      (RESULT_PTR),                                                       \
      (NUM_ITEMS),                                                        \
      (REDUCE_OP),                                                        \
      (REDUCE_INIT));                                                     \
  HANDLECUDA(cudaFree(TMPN1));
#define REDUCE_WS(           \
    REDUCE_PTR,              \
    RESULT_PTR,              \
    NUM_ITEMS,               \
    REDUCE_OP,               \
    REDUCE_INIT,             \
    WORKSPACE_PTR,           \
    WORSPACE_BYTES,          \
    STREAM)                  \
  cub::DeviceReduce::Reduce( \
      (WORKSPACE_PTR),       \
      (WORSPACE_BYTES),      \
      (REDUCE_PTR),          \
      (RESULT_PTR),          \
      (NUM_ITEMS),           \
      (REDUCE_OP),           \
      (REDUCE_INIT),         \
      (STREAM));
#define SELECT_FLAGS_WS(      \
    FLAGS_PTR,                \
    ITEM_PTR,                 \
    OUT_PTR,                  \
    NUM_SELECTED_PTR,         \
    NUM_ITEMS,                \
    WORKSPACE_PTR,            \
    WORSPACE_BYTES,           \
    STREAM)                   \
  cub::DeviceSelect::Flagged( \
      (WORKSPACE_PTR),        \
      (WORSPACE_BYTES),       \
      (ITEM_PTR),             \
      (FLAGS_PTR),            \
      (OUT_PTR),              \
      (NUM_SELECTED_PTR),     \
      (NUM_ITEMS),            \
      stream = (STREAM));

#define COPY_HOST_DEV(PTR_D, PTR_H, TYPE, SIZE) \
  HANDLECUDA(cudaMemcpy(                        \
      (PTR_D), (PTR_H), sizeof(TYPE) * (SIZE), cudaMemcpyHostToDevice))
#define COPY_DEV_HOST(PTR_H, PTR_D, TYPE, SIZE) \
  HANDLECUDA(cudaMemcpy(                        \
      (PTR_H), (PTR_D), sizeof(TYPE) * (SIZE), cudaMemcpyDeviceToHost))
#define COPY_DEV_DEV(PTR_T, PTR_S, TYPE, SIZE) \
  HANDLECUDA(cudaMemcpy(                       \
      (PTR_T), (PTR_S), sizeof(TYPE) * (SIZE), cudaMemcpyDeviceToDevice))
//
// We *must* use cudaMallocManaged for pointers on device that should
// interact with pytorch. However, this comes at a significant speed penalty.
// We're using plain CUDA pointers for the rendering operations and
// explicitly copy results to managed pointers wrapped for pytorch (see
// pytorch/util.h).
#define MALLOC(VAR, TYPE, SIZE) cudaMalloc(&(VAR), sizeof(TYPE) * (SIZE))
#define FREE(PTR) HANDLECUDA(cudaFree(PTR))
#define MEMSET(VAR, VAL, TYPE, SIZE, STREAM) \
  HANDLECUDA(cudaMemsetAsync((VAR), (VAL), sizeof(TYPE) * (SIZE), (STREAM)))

#define LAUNCH_MAX_PARALLEL_1D(FUNC, N, STREAM, ...)                \
  {                                                                 \
    int64_t max_threads =                                           \
        at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock; \
    uint num_threads = min((N), max_threads);                       \
    uint num_blocks = iDivCeil((N), num_threads);                   \
    FUNC<<<num_blocks, num_threads, 0, (STREAM)>>>(__VA_ARGS__);    \
  }
#define LAUNCH_PARALLEL_1D(FUNC, N, TN, STREAM, ...)                   \
  {                                                                    \
    uint num_threads = min(static_cast<int>(N), static_cast<int>(TN)); \
    uint num_blocks = iDivCeil((N), num_threads);                      \
    FUNC<<<num_blocks, num_threads, 0, (STREAM)>>>(__VA_ARGS__);       \
  }
#define LAUNCH_MAX_PARALLEL_2D(FUNC, NX, NY, STREAM, ...)               \
  {                                                                     \
    int64_t max_threads =                                               \
        at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock;     \
    int64_t max_threads_sqrt = static_cast<int64_t>(sqrt(max_threads)); \
    dim3 num_threads, num_blocks;                                       \
    num_threads.x = min((NX), max_threads_sqrt);                        \
    num_blocks.x = iDivCeil((NX), num_threads.x);                       \
    num_threads.y = min((NY), max_threads_sqrt);                        \
    num_blocks.y = iDivCeil((NY), num_threads.y);                       \
    num_threads.z = 1;                                                  \
    num_blocks.z = 1;                                                   \
    FUNC<<<num_blocks, num_threads, 0, (STREAM)>>>(__VA_ARGS__);        \
  }
#define LAUNCH_PARALLEL_2D(FUNC, NX, NY, TX, TY, STREAM, ...)    \
  {                                                              \
    dim3 num_threads, num_blocks;                                \
    num_threads.x = min((NX), (TX));                             \
    num_blocks.x = iDivCeil((NX), num_threads.x);                \
    num_threads.y = min((NY), (TY));                             \
    num_blocks.y = iDivCeil((NY), num_threads.y);                \
    num_threads.z = 1;                                           \
    num_blocks.z = 1;                                            \
    FUNC<<<num_blocks, num_threads, 0, (STREAM)>>>(__VA_ARGS__); \
  }

#define GET_PARALLEL_IDX_1D(VARNAME, N)                               \
  const uint VARNAME = __mul24(blockIdx.x, blockDim.x) + threadIdx.x; \
  if (VARNAME >= (N)) {                                               \
    return;                                                           \
  }
#define GET_PARALLEL_IDS_2D(VAR_X, VAR_Y, WIDTH, HEIGHT)            \
  const uint VAR_X = __mul24(blockIdx.x, blockDim.x) + threadIdx.x; \
  const uint VAR_Y = __mul24(blockIdx.y, blockDim.y) + threadIdx.y; \
  if (VAR_X >= (WIDTH) || VAR_Y >= (HEIGHT))                        \
    return;
#define END_PARALLEL()
#define END_PARALLEL_NORET()
#define END_PARALLEL_2D_NORET()
#define END_PARALLEL_2D()
#define RETURN_PARALLEL() return
#define CHECKLAUNCH() C10_CUDA_CHECK(cudaGetLastError());
#define ISONDEVICE true
#define SYNCDEVICE() HANDLECUDA(cudaDeviceSynchronize())
#define START_TIME(TN)                             \
  cudaEvent_t __time_start_##TN, __time_stop_##TN; \
  cudaEventCreate(&__time_start_##TN);             \
  cudaEventCreate(&__time_stop_##TN);              \
  cudaEventRecord(__time_start_##TN);
#define STOP_TIME(TN) cudaEventRecord(__time_stop_##TN);
#define GET_TIME(TN, TOPTR)               \
  cudaEventSynchronize(__time_stop_##TN); \
  cudaEventElapsedTime((TOPTR), __time_start_##TN, __time_stop_##TN);
#define START_TIME_CU(TN) START_TIME(CN)
#define STOP_TIME_CU(TN) STOP_TIME(TN)
#define GET_TIME_CU(TN, TOPTR) GET_TIME(TN, TOPTR)

#endif
