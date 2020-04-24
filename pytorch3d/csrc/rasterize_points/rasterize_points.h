// Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

#pragma once
#include <torch/extension.h>
#include <cstdio>
#include <tuple>
#include "utils/pytorch3d_cutils.h"

// ****************************************************************************
// *                          NAIVE RASTERIZATION                             *
// ****************************************************************************

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> RasterizePointsNaiveCpu(
    const torch::Tensor& points,
    const torch::Tensor& cloud_to_packed_first_idx,
    const torch::Tensor& num_points_per_cloud,
    const int image_size,
    const float radius,
    const int points_per_pixel);

#ifdef WITH_CUDA
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
RasterizePointsNaiveCuda(
    const torch::Tensor& points,
    const torch::Tensor& cloud_to_packed_first_idx,
    const torch::Tensor& num_points_per_cloud,
    const int image_size,
    const float radius,
    const int points_per_pixel);
#endif
// Naive (forward) pointcloud rasterization: For each pixel, for each point,
// check whether that point hits the pixel.
//
// Args:
//  points: Tensor of shape (P, 3) giving (packed) positions for
//          points in all N pointclouds in the batch where P is the total
//          number of points in the batch across all pointclouds. These points
//          are expected to be in NDC coordinates in the range [-1, 1].
//  cloud_to_packed_first_idx: LongTensor of shape (N) giving the index in
//                          points_packed of the first point in each pointcloud
//                          in the batch where N is the batch size.
//  num_points_per_cloud: LongTensor of shape (N) giving the number of points
//                        for each pointcloud in the batch.
//  radius: Radius of each point (in NDC units)
//  image_size: (S) Size of the image to return (in pixels)
//  points_per_pixel: (K) The number closest of points to return for each pixel
//
// Returns:
//  A 4 element tuple of:
//  idxs: int32 Tensor of shape (N, S, S, K) giving the indices of the
//        closest K points along the z-axis for each pixel, padded with -1 for
//        pixels hit by fewer than K points. The indices refer to points in
//        points packed i.e a tensor of shape (P, 3) representing the flattened
//        points for all pointclouds in the batch.
//  zbuf: float32 Tensor of shape (N, S, S, K) giving the depth of each
//        closest point for each pixel.
//  dists: float32 Tensor of shape (N, S, S, K) giving squared Euclidean
//          distance in the (NDC) x/y plane between each pixel and its K closest
//          points along the z axis.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> RasterizePointsNaive(
    const torch::Tensor& points,
    const torch::Tensor& cloud_to_packed_first_idx,
    const torch::Tensor& num_points_per_cloud,
    const int image_size,
    const float radius,
    const int points_per_pixel) {
  if (points.is_cuda() && cloud_to_packed_first_idx.is_cuda() &&
      num_points_per_cloud.is_cuda()) {
#ifdef WITH_CUDA
    CHECK_CONTIGUOUS_CUDA(points);
    CHECK_CONTIGUOUS_CUDA(cloud_to_packed_first_idx);
    CHECK_CONTIGUOUS_CUDA(num_points_per_cloud);
    return RasterizePointsNaiveCuda(
        points,
        cloud_to_packed_first_idx,
        num_points_per_cloud,
        image_size,
        radius,
        points_per_pixel);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  } else {
    return RasterizePointsNaiveCpu(
        points,
        cloud_to_packed_first_idx,
        num_points_per_cloud,
        image_size,
        radius,
        points_per_pixel);
  }
}

// ****************************************************************************
// *                          COARSE RASTERIZATION                            *
// ****************************************************************************

torch::Tensor RasterizePointsCoarseCpu(
    const torch::Tensor& points,
    const torch::Tensor& cloud_to_packed_first_idx,
    const torch::Tensor& num_points_per_cloud,
    const int image_size,
    const float radius,
    const int bin_size,
    const int max_points_per_bin);

#ifdef WITH_CUDA
torch::Tensor RasterizePointsCoarseCuda(
    const torch::Tensor& points,
    const torch::Tensor& cloud_to_packed_first_idx,
    const torch::Tensor& num_points_per_cloud,
    const int image_size,
    const float radius,
    const int bin_size,
    const int max_points_per_bin);
#endif
// Args:
//  points: Tensor of shape (P, 3) giving (packed) positions for
//          points in all N pointclouds in the batch where P is the total
//          number of points in the batch across all pointclouds. These points
//          are expected to be in NDC coordinates in the range [-1, 1].
//  cloud_to_packed_first_idx: LongTensor of shape (N) giving the index in
//                          points_packed of the first point in each pointcloud
//                          in the batch where N is the batch size.
//  num_points_per_cloud: LongTensor of shape (N) giving the number of points
//                        for each pointcloud in the batch.
//  radius: Radius of points to rasterize (in NDC units)
//  image_size: Size of the image to generate (in pixels)
//  bin_size: Size of each bin within the image (in pixels)
//
// Returns:
//  points_per_bin: Tensor of shape (N, num_bins, num_bins) giving the number
//                  of points that fall in each bin
//  bin_points: Tensor of shape (N, num_bins, num_bins, K) giving the indices
//              of points that fall into each bin.
torch::Tensor RasterizePointsCoarse(
    const torch::Tensor& points,
    const torch::Tensor& cloud_to_packed_first_idx,
    const torch::Tensor& num_points_per_cloud,
    const int image_size,
    const float radius,
    const int bin_size,
    const int max_points_per_bin) {
  if (points.is_cuda() && cloud_to_packed_first_idx.is_cuda() &&
      num_points_per_cloud.is_cuda()) {
#ifdef WITH_CUDA
    CHECK_CONTIGUOUS_CUDA(points);
    CHECK_CONTIGUOUS_CUDA(cloud_to_packed_first_idx);
    CHECK_CONTIGUOUS_CUDA(num_points_per_cloud);
    return RasterizePointsCoarseCuda(
        points,
        cloud_to_packed_first_idx,
        num_points_per_cloud,
        image_size,
        radius,
        bin_size,
        max_points_per_bin);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  } else {
    return RasterizePointsCoarseCpu(
        points,
        cloud_to_packed_first_idx,
        num_points_per_cloud,
        image_size,
        radius,
        bin_size,
        max_points_per_bin);
  }
}

// ****************************************************************************
// *                            FINE RASTERIZATION                            *
// ****************************************************************************

#ifdef WITH_CUDA
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> RasterizePointsFineCuda(
    const torch::Tensor& points,
    const torch::Tensor& bin_points,
    const int image_size,
    const float radius,
    const int bin_size,
    const int points_per_pixel);
#endif
// Args:
//  points: Tensor of shape (P, 3) giving (packed) positions for
//          points in all N pointclouds in the batch where P is the total
//          number of points in the batch across all pointclouds. These points
//          are expected to be in NDC coordinates in the range [-1, 1].
//  bin_points: int32 Tensor of shape (N, B, B, M) giving the indices of points
//              that fall into each bin (output from coarse rasterization)
//  image_size: Size of image to generate (in pixels)
//  radius: Radius of points to rasterize (NDC units)
//  bin_size: Size of each bin (in pixels)
//  points_per_pixel: How many points to rasterize for each pixel
//
// Returns (same as rasterize_points):
//  idxs: int32 Tensor of shape (N, S, S, K) giving the indices of the
//        closest K points along the z-axis for each pixel, padded with -1 for
//        pixels hit by fewer than K points. The indices refer to points in
//        points packed i.e a tensor of shape (P, 3) representing the flattened
//        points for all pointclouds in the batch.
//  zbuf: float32 Tensor of shape (N, S, S, K) giving the depth of each of each
//        closest point for each pixel
//  dists: float32 Tensor of shape (N, S, S, K) giving squared Euclidean
//         distance in the (NDC) x/y plane between each pixel and its K closest
//         points along the z axis.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> RasterizePointsFine(
    const torch::Tensor& points,
    const torch::Tensor& bin_points,
    const int image_size,
    const float radius,
    const int bin_size,
    const int points_per_pixel) {
  if (points.is_cuda()) {
#ifdef WITH_CUDA
    CHECK_CONTIGUOUS_CUDA(points);
    CHECK_CONTIGUOUS_CUDA(bin_points);
    return RasterizePointsFineCuda(
        points, bin_points, image_size, radius, bin_size, points_per_pixel);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  } else {
    AT_ERROR("NOT IMPLEMENTED");
  }
}

// ****************************************************************************
// *                            BACKWARD PASS                                 *
// ****************************************************************************

torch::Tensor RasterizePointsBackwardCpu(
    const torch::Tensor& points,
    const torch::Tensor& idxs,
    const torch::Tensor& grad_zbuf,
    const torch::Tensor& grad_dists);

#ifdef WITH_CUDA
torch::Tensor RasterizePointsBackwardCuda(
    const torch::Tensor& points,
    const torch::Tensor& idxs,
    const torch::Tensor& grad_zbuf,
    const torch::Tensor& grad_dists);
#endif
// Args:
//  points: Tensor of shape (P, 3) giving (packed) positions for
//          points in all N pointclouds in the batch where P is the total
//          number of points in the batch across all pointclouds. These points
//          are expected to be in NDC coordinates in the range [-1, 1].
//  idxs: int32 Tensor of shape (N, H, W, K) (from forward pass)
//  grad_zbuf: float32 Tensor of shape (N, H, W, K) giving upstream gradient
//             d(loss)/d(zbuf) of the distances from each pixel to its nearest
//             points.
//  grad_dists: Tensor of shape (N, H, W, K) giving upstream gradient
//              d(loss)/d(dists) of the dists tensor returned by the forward
//              pass.
//
// Returns:
//  grad_points: float32 Tensor of shape (N, P, 3) giving downstream gradients
torch::Tensor RasterizePointsBackward(
    const torch::Tensor& points,
    const torch::Tensor& idxs,
    const torch::Tensor& grad_zbuf,
    const torch::Tensor& grad_dists) {
  if (points.is_cuda()) {
#ifdef WITH_CUDA
    CHECK_CONTIGUOUS_CUDA(points);
    CHECK_CONTIGUOUS_CUDA(idxs);
    CHECK_CONTIGUOUS_CUDA(grad_zbuf);
    CHECK_CONTIGUOUS_CUDA(grad_dists);
    return RasterizePointsBackwardCuda(points, idxs, grad_zbuf, grad_dists);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  } else {
    return RasterizePointsBackwardCpu(points, idxs, grad_zbuf, grad_dists);
  }
}

// ****************************************************************************
// *                         MAIN ENTRY POINT                                 *
// ****************************************************************************

// This is the main entry point for the forward pass of the point rasterizer;
// it uses either naive or coarse-to-fine rasterization based on bin_size.
//
// Args:
//  points: Tensor of shape (P, 3) giving (packed) positions for
//          points in all N pointclouds in the batch where P is the total
//          number of points in the batch across all pointclouds. These points
//          are expected to be in NDC coordinates in the range [-1, 1].
//  cloud_to_packed_first_idx: LongTensor of shape (N) giving the index in
//                          points_packed of the first point in each pointcloud
//                          in the batch where N is the batch size.
//  num_points_per_cloud: LongTensor of shape (N) giving the number of points
//                        for each pointcloud in the batch.
//  radius: Radius of each point (in NDC units)
//  image_size:  (S) Size of the image to return (in pixels)
//  points_per_pixel: (K) The number of points to return for each pixel
//  bin_size: Bin size (in pixels) for coarse-to-fine rasterization. Setting
//            bin_size=0 uses naive rasterization instead.
//  max_points_per_bin: The maximum number of points allowed to fall into each
//                      bin when using coarse-to-fine rasterization.
//
// Returns:
//  idxs: int32 Tensor of shape (N, S, S, K) giving the indices of the
//        closest K points along the z-axis for each pixel, padded with -1 for
//        pixels hit by fewer than K points. The indices refer to points in
//        points packed i.e a tensor of shape (P, 3) representing the flattened
//        points for all pointclouds in the batch.
//  zbuf: float32 Tensor of shape (N, S, S, K) giving the depth of each of each
//        closest point for each pixel
//  dists: float32 Tensor of shape (N, S, S, K) giving squared Euclidean
//         distance in the (NDC) x/y plane between each pixel and its K closest
//         points along the z axis.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> RasterizePoints(
    const torch::Tensor& points,
    const torch::Tensor& cloud_to_packed_first_idx,
    const torch::Tensor& num_points_per_cloud,
    const int image_size,
    const float radius,
    const int points_per_pixel,
    const int bin_size,
    const int max_points_per_bin) {
  if (bin_size == 0) {
    // Use the naive per-pixel implementation
    return RasterizePointsNaive(
        points,
        cloud_to_packed_first_idx,
        num_points_per_cloud,
        image_size,
        radius,
        points_per_pixel);
  } else {
    // Use coarse-to-fine rasterization
    const auto bin_points = RasterizePointsCoarse(
        points,
        cloud_to_packed_first_idx,
        num_points_per_cloud,
        image_size,
        radius,
        bin_size,
        max_points_per_bin);
    return RasterizePointsFine(
        points, bin_points, image_size, radius, bin_size, points_per_pixel);
  }
}
