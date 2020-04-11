// Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

#pragma once
#include <torch/extension.h>
#include <tuple>
#include "utils/pytorch3d_cutils.h"

// Compute indices of K nearest neighbors in pointcloud p2 to points
// in pointcloud p1.
//
// Args:
//    p1: FloatTensor of shape (N, P1, D) giving a batch of pointclouds each
//        containing P1 points of dimension D.
//    p2: FloatTensor of shape (N, P2, D) giving a batch of pointclouds each
//        containing P2 points of dimension D.
//    lengths1: LongTensor, shape (N,), giving actual length of each P1 cloud.
//    lengths2: LongTensor, shape (N,), giving actual length of each P2 cloud.
//    K: int giving the number of nearest points to return.
//    sorted: bool telling whether to sort the K returned points by their
//        distance.
//    version: Integer telling which implementation to use.
//
// Returns:
//    p1_neighbor_idx: LongTensor of shape (N, P1, K), where
//        p1_neighbor_idx[n, i, k] = j means that the kth nearest
//        neighbor to p1[n, i] in the cloud p2[n] is p2[n, j].
//        It is padded with zeros so that it can be used easily in a later
//        gather() operation.
//
//    p1_neighbor_dists: FloatTensor of shape (N, P1, K) containing the squared
//        distance from each point p1[n, p, :] to its K neighbors
//        p2[n, p1_neighbor_idx[n, p, k], :].

// CPU implementation.
std::tuple<at::Tensor, at::Tensor> KNearestNeighborIdxCpu(
    const at::Tensor& p1,
    const at::Tensor& p2,
    const at::Tensor& lengths1,
    const at::Tensor& lengths2,
    int K);

// CUDA implementation
std::tuple<at::Tensor, at::Tensor> KNearestNeighborIdxCuda(
    const at::Tensor& p1,
    const at::Tensor& p2,
    const at::Tensor& lengths1,
    const at::Tensor& lengths2,
    int K,
    int version);

// Implementation which is exposed.
std::tuple<at::Tensor, at::Tensor> KNearestNeighborIdx(
    const at::Tensor& p1,
    const at::Tensor& p2,
    const at::Tensor& lengths1,
    const at::Tensor& lengths2,
    int K,
    int version) {
  if (p1.is_cuda() || p2.is_cuda()) {
#ifdef WITH_CUDA
    CHECK_CONTIGUOUS_CUDA(p1);
    CHECK_CONTIGUOUS_CUDA(p2);
    return KNearestNeighborIdxCuda(p1, p2, lengths1, lengths2, K, version);
#else
    AT_ERROR("Not compiled with GPU support.");
#endif
  }
  return KNearestNeighborIdxCpu(p1, p2, lengths1, lengths2, K);
}
