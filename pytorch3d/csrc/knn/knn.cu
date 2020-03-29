// Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

#include <ATen/ATen.h>
#include <float.h>
#include <iostream>
#include <tuple>

#include "dispatch.cuh"
#include "mink.cuh"

template <typename scalar_t>
__global__ void KNearestNeighborKernelV0(
    const scalar_t* __restrict__ points1,
    const scalar_t* __restrict__ points2,
    scalar_t* __restrict__ dists,
    int64_t* __restrict__ idxs,
    const size_t N,
    const size_t P1,
    const size_t P2,
    const size_t D,
    const size_t K) {
  // Stupid version: Make each thread handle one query point and loop over
  // all P2 target points. There are N * P1 input points to handle, so
  // do a trivial parallelization over threads.
  // Store both dists and indices for knn in global memory.
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  const int num_threads = blockDim.x * gridDim.x;
  for (int np = tid; np < N * P1; np += num_threads) {
    int n = np / P1;
    int p1 = np % P1;
    int offset = n * P1 * K + p1 * K;
    MinK<scalar_t, int64_t> mink(dists + offset, idxs + offset, K);
    for (int p2 = 0; p2 < P2; ++p2) {
      // Find the distance between points1[n, p1] and points[n, p2]
      scalar_t dist = 0;
      for (int d = 0; d < D; ++d) {
        scalar_t coord1 = points1[n * P1 * D + p1 * D + d];
        scalar_t coord2 = points2[n * P2 * D + p2 * D + d];
        scalar_t diff = coord1 - coord2;
        dist += diff * diff;
      }
      mink.add(dist, p2);
    }
  }
}

template <typename scalar_t, int64_t D>
__global__ void KNearestNeighborKernelV1(
    const scalar_t* __restrict__ points1,
    const scalar_t* __restrict__ points2,
    scalar_t* __restrict__ dists,
    int64_t* __restrict__ idxs,
    const size_t N,
    const size_t P1,
    const size_t P2,
    const size_t K) {
  // Same idea as the previous version, but hoist D into a template argument
  // so we can cache the current point in a thread-local array. We still store
  // the current best K dists and indices in global memory, so this should work
  // for very large K and fairly large D.
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  const int num_threads = blockDim.x * gridDim.x;
  scalar_t cur_point[D];
  for (int np = tid; np < N * P1; np += num_threads) {
    int n = np / P1;
    int p1 = np % P1;
    for (int d = 0; d < D; ++d) {
      cur_point[d] = points1[n * P1 * D + p1 * D + d];
    }
    int offset = n * P1 * K + p1 * K;
    MinK<scalar_t, int64_t> mink(dists + offset, idxs + offset, K);
    for (int p2 = 0; p2 < P2; ++p2) {
      // Find the distance between cur_point and points[n, p2]
      scalar_t dist = 0;
      for (int d = 0; d < D; ++d) {
        scalar_t diff = cur_point[d] - points2[n * P2 * D + p2 * D + d];
        dist += diff * diff;
      }
      mink.add(dist, p2);
    }
  }
}

// This is a shim functor to allow us to dispatch using DispatchKernel1D
template <typename scalar_t, int64_t D>
struct KNearestNeighborV1Functor {
  static void run(
      size_t blocks,
      size_t threads,
      const scalar_t* __restrict__ points1,
      const scalar_t* __restrict__ points2,
      scalar_t* __restrict__ dists,
      int64_t* __restrict__ idxs,
      const size_t N,
      const size_t P1,
      const size_t P2,
      const size_t K) {
    KNearestNeighborKernelV1<scalar_t, D>
        <<<blocks, threads>>>(points1, points2, dists, idxs, N, P1, P2, K);
  }
};

template <typename scalar_t, int64_t D, int64_t K>
__global__ void KNearestNeighborKernelV2(
    const scalar_t* __restrict__ points1,
    const scalar_t* __restrict__ points2,
    scalar_t* __restrict__ dists,
    int64_t* __restrict__ idxs,
    const int64_t N,
    const int64_t P1,
    const int64_t P2) {
  // Same general implementation as V2, but also hoist K into a template arg.
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  const int num_threads = blockDim.x * gridDim.x;
  scalar_t cur_point[D];
  scalar_t min_dists[K];
  int min_idxs[K];
  for (int np = tid; np < N * P1; np += num_threads) {
    int n = np / P1;
    int p1 = np % P1;
    for (int d = 0; d < D; ++d) {
      cur_point[d] = points1[n * P1 * D + p1 * D + d];
    }
    MinK<scalar_t, int> mink(min_dists, min_idxs, K);
    for (int p2 = 0; p2 < P2; ++p2) {
      scalar_t dist = 0;
      for (int d = 0; d < D; ++d) {
        int offset = n * P2 * D + p2 * D + d;
        scalar_t diff = cur_point[d] - points2[offset];
        dist += diff * diff;
      }
      mink.add(dist, p2);
    }
    for (int k = 0; k < mink.size(); ++k) {
      idxs[n * P1 * K + p1 * K + k] = min_idxs[k];
      dists[n * P1 * K + p1 * K + k] = min_dists[k];
    }
  }
}

// This is a shim so we can dispatch using DispatchKernel2D
template <typename scalar_t, int64_t D, int64_t K>
struct KNearestNeighborKernelV2Functor {
  static void run(
      size_t blocks,
      size_t threads,
      const scalar_t* __restrict__ points1,
      const scalar_t* __restrict__ points2,
      scalar_t* __restrict__ dists,
      int64_t* __restrict__ idxs,
      const int64_t N,
      const int64_t P1,
      const int64_t P2) {
    KNearestNeighborKernelV2<scalar_t, D, K>
        <<<blocks, threads>>>(points1, points2, dists, idxs, N, P1, P2);
  }
};

template <typename scalar_t, int D, int K>
__global__ void KNearestNeighborKernelV3(
    const scalar_t* __restrict__ points1,
    const scalar_t* __restrict__ points2,
    scalar_t* __restrict__ dists,
    int64_t* __restrict__ idxs,
    const size_t N,
    const size_t P1,
    const size_t P2) {
  // Same idea as V2, but use register indexing for thread-local arrays.
  // Enabling sorting for this version leads to huge slowdowns; I suspect
  // that it forces min_dists into local memory rather than registers.
  // As a result this version is always unsorted.
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  const int num_threads = blockDim.x * gridDim.x;
  scalar_t cur_point[D];
  scalar_t min_dists[K];
  int min_idxs[K];
  for (int np = tid; np < N * P1; np += num_threads) {
    int n = np / P1;
    int p1 = np % P1;
    for (int d = 0; d < D; ++d) {
      cur_point[d] = points1[n * P1 * D + p1 * D + d];
    }
    RegisterMinK<scalar_t, int, K> mink(min_dists, min_idxs);
    for (int p2 = 0; p2 < P2; ++p2) {
      scalar_t dist = 0;
      for (int d = 0; d < D; ++d) {
        int offset = n * P2 * D + p2 * D + d;
        scalar_t diff = cur_point[d] - points2[offset];
        dist += diff * diff;
      }
      mink.add(dist, p2);
    }
    for (int k = 0; k < mink.size(); ++k) {
      idxs[n * P1 * K + p1 * K + k] = min_idxs[k];
      dists[n * P1 * K + p1 * K + k] = min_dists[k];
    }
  }
}

// This is a shim so we can dispatch using DispatchKernel2D
template <typename scalar_t, int64_t D, int64_t K>
struct KNearestNeighborKernelV3Functor {
  static void run(
      size_t blocks,
      size_t threads,
      const scalar_t* __restrict__ points1,
      const scalar_t* __restrict__ points2,
      scalar_t* __restrict__ dists,
      int64_t* __restrict__ idxs,
      const size_t N,
      const size_t P1,
      const size_t P2) {
    KNearestNeighborKernelV3<scalar_t, D, K>
        <<<blocks, threads>>>(points1, points2, dists, idxs, N, P1, P2);
  }
};

constexpr int V1_MIN_D = 1;
constexpr int V1_MAX_D = 32;

constexpr int V2_MIN_D = 1;
constexpr int V2_MAX_D = 8;
constexpr int V2_MIN_K = 1;
constexpr int V2_MAX_K = 32;

constexpr int V3_MIN_D = 1;
constexpr int V3_MAX_D = 8;
constexpr int V3_MIN_K = 1;
constexpr int V3_MAX_K = 4;

bool InBounds(const int64_t min, const int64_t x, const int64_t max) {
  return min <= x && x <= max;
}

bool CheckVersion(int version, const int64_t D, const int64_t K) {
  if (version == 0) {
    return true;
  } else if (version == 1) {
    return InBounds(V1_MIN_D, D, V1_MAX_D);
  } else if (version == 2) {
    return InBounds(V2_MIN_D, D, V2_MAX_D) && InBounds(V2_MIN_K, K, V2_MAX_K);
  } else if (version == 3) {
    return InBounds(V3_MIN_D, D, V3_MAX_D) && InBounds(V3_MIN_K, K, V3_MAX_K);
  }
  return false;
}

int ChooseVersion(const int64_t D, const int64_t K) {
  for (int version = 3; version >= 1; version--) {
    if (CheckVersion(version, D, K)) {
      return version;
    }
  }
  return 0;
}

std::tuple<at::Tensor, at::Tensor> KNearestNeighborIdxCuda(
    const at::Tensor& p1,
    const at::Tensor& p2,
    int K,
    int version) {
  const auto N = p1.size(0);
  const auto P1 = p1.size(1);
  const auto P2 = p2.size(1);
  const auto D = p2.size(2);
  const int64_t K_64 = K;

  AT_ASSERTM(p2.size(2) == D, "Point sets must have the same last dimension");
  auto long_dtype = p1.options().dtype(at::kLong);
  auto idxs = at::full({N, P1, K}, -1, long_dtype);
  auto dists = at::full({N, P1, K}, -1, p1.options());

  if (version < 0) {
    version = ChooseVersion(D, K);
  } else if (!CheckVersion(version, D, K)) {
    int new_version = ChooseVersion(D, K);
    std::cout << "WARNING: Requested KNN version " << version
              << " is not compatible with D = " << D << "; K = " << K
              << ". Falling back to version = " << new_version << std::endl;
    version = new_version;
  }

  // At this point we should have a valid version no matter what data the user
  // gave us. But we can check once more to be sure; however this time
  // assert fail since failing at this point means we have a bug in our version
  // selection or checking code.
  AT_ASSERTM(CheckVersion(version, D, K), "Invalid version");

  const size_t threads = 256;
  const size_t blocks = 256;
  if (version == 0) {
    AT_DISPATCH_FLOATING_TYPES(p1.scalar_type(), "knn_kernel_cuda", ([&] {
                                 KNearestNeighborKernelV0<scalar_t>
                                     <<<blocks, threads>>>(
                                         p1.data_ptr<scalar_t>(),
                                         p2.data_ptr<scalar_t>(),
                                         dists.data_ptr<scalar_t>(),
                                         idxs.data_ptr<int64_t>(),
                                         N,
                                         P1,
                                         P2,
                                         D,
                                         K);
                               }));
  } else if (version == 1) {
    AT_DISPATCH_FLOATING_TYPES(p1.scalar_type(), "knn_kernel_cuda", ([&] {
                                 DispatchKernel1D<
                                     KNearestNeighborV1Functor,
                                     scalar_t,
                                     V1_MIN_D,
                                     V1_MAX_D>(
                                     D,
                                     blocks,
                                     threads,
                                     p1.data_ptr<scalar_t>(),
                                     p2.data_ptr<scalar_t>(),
                                     dists.data_ptr<scalar_t>(),
                                     idxs.data_ptr<int64_t>(),
                                     N,
                                     P1,
                                     P2,
                                     K);
                               }));
  } else if (version == 2) {
    AT_DISPATCH_FLOATING_TYPES(p1.scalar_type(), "knn_kernel_cuda", ([&] {
                                 DispatchKernel2D<
                                     KNearestNeighborKernelV2Functor,
                                     scalar_t,
                                     V2_MIN_D,
                                     V2_MAX_D,
                                     V2_MIN_K,
                                     V2_MAX_K>(
                                     D,
                                     K_64,
                                     blocks,
                                     threads,
                                     p1.data_ptr<scalar_t>(),
                                     p2.data_ptr<scalar_t>(),
                                     dists.data_ptr<scalar_t>(),
                                     idxs.data_ptr<int64_t>(),
                                     N,
                                     P1,
                                     P2);
                               }));
  } else if (version == 3) {
    AT_DISPATCH_FLOATING_TYPES(p1.scalar_type(), "knn_kernel_cuda", ([&] {
                                 DispatchKernel2D<
                                     KNearestNeighborKernelV3Functor,
                                     scalar_t,
                                     V3_MIN_D,
                                     V3_MAX_D,
                                     V3_MIN_K,
                                     V3_MAX_K>(
                                     D,
                                     K_64,
                                     blocks,
                                     threads,
                                     p1.data_ptr<scalar_t>(),
                                     p2.data_ptr<scalar_t>(),
                                     dists.data_ptr<scalar_t>(),
                                     idxs.data_ptr<int64_t>(),
                                     N,
                                     P1,
                                     P2);
                               }));
  }

  return std::make_tuple(idxs, dists);
}
