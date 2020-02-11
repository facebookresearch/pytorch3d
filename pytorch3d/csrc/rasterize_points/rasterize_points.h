// Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

#pragma once
#include <torch/extension.h>
#include <cstdio>
#include <tuple>

// ****************************************************************************
// *                          NAIVE RASTERIZATION                             *
// ****************************************************************************

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> RasterizePointsNaiveCpu(
    const torch::Tensor& points,
    const int image_size,
    const float radius,
    const int points_per_pixel);

#ifdef WITH_CUDA
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
RasterizePointsNaiveCuda(
    const torch::Tensor& points,
    const int image_size,
    const float radius,
    const int points_per_pixel);
#endif
// Naive (forward) pointcloud rasterization: For each pixel, for each point,
// check whether that point hits the pixel.
//
// Args:
//  points: Tensor of shape (N, P, 3) (in NDC)
//  radius: Radius of each point (in NDC units)
//  image_size:  (S) Size of the image to return (in pixels)
//  points_per_pixel: (K) The number closest of points to return for each pixel
//
// Returns:
//  idxs: int32 Tensor of shape (N, S, S, K) giving the indices of the
//        closest K points along the z-axis for each pixel, padded with -1 for
//        pixels
//         hit by fewer than K points.
//  zbuf: float32 Tensor of shape (N, S, S, K) giving the depth of each
//        closest point for each pixel.
//  dists: float32 Tensor of shape (N, S, S, K) giving squared Euclidean
//          distance in the (NDC) x/y plane between each pixel and its K closest
//          points along the z axis.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> RasterizePointsNaive(
    const torch::Tensor& points,
    const int image_size,
    const float radius,
    const int points_per_pixel) {
  if (points.type().is_cuda()) {
#ifdef WITH_CUDA
    return RasterizePointsNaiveCuda(
        points, image_size, radius, points_per_pixel);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  } else {
    return RasterizePointsNaiveCpu(
        points, image_size, radius, points_per_pixel);
  }
}

// ****************************************************************************
// *                          COARSE RASTERIZATION                            *
// ****************************************************************************

torch::Tensor RasterizePointsCoarseCpu(
    const torch::Tensor& points,
    const int image_size,
    const float radius,
    const int bin_size,
    const int max_points_per_bin);

#ifdef WITH_CUDA
torch::Tensor RasterizePointsCoarseCuda(
    const torch::Tensor& points,
    const int image_size,
    const float radius,
    const int bin_size,
    const int max_points_per_bin);
#endif
// Args:
//  points: Tensor of shape (N, P, 3)
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
    const int image_size,
    const float radius,
    const int bin_size,
    const int max_points_per_bin) {
  if (points.type().is_cuda()) {
#ifdef WITH_CUDA
    return RasterizePointsCoarseCuda(
        points, image_size, radius, bin_size, max_points_per_bin);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  } else {
    return RasterizePointsCoarseCpu(
        points, image_size, radius, bin_size, max_points_per_bin);
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
//  points: float32 Tensor of shape (N, P, 3)
//  bin_points: int32 Tensor of shape (N, B, B, M) giving the indices of points
//              that fall into each bin (output from coarse rasterization)
//  image_size: Size of image to generate (in pixels)
//  radius: Radius of points to rasterize (NDC units)
//  bin_size: Size of each bin (in pixels)
//  points_per_pixel: How many points to rasterize for each pixel
//
// Returns (same as rasterize_points):
//  idxs: int32 Tensor of shape (N, S, S, K) giving the indices of the closest
//        points_per_pixel points along the z-axis for each pixel, padded with
//        -1 for pixels hit by fewer than points_per_pixel points
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
  if (points.type().is_cuda()) {
#ifdef WITH_CUDA
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
//  points: float32 Tensor of shape (N, P, 3)
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
  if (points.type().is_cuda()) {
#ifdef WITH_CUDA
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
//  points: Tensor of shape (N, P, 3) (in NDC)
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
//        closest points_per_pixel points along the z-axis for each pixel,
//        padded with -1 for pixels hit by fewer than points_per_pixel points
//  zbuf: float32 Tensor of shape (N, S, S, K) giving the depth of each of each
//        closest point for each pixel
//  dists: float32 Tensor of shape (N, S, S, K) giving squared Euclidean
//         distance in the (NDC) x/y plane between each pixel and its K closest
//         points along the z axis.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> RasterizePoints(
    const torch::Tensor& points,
    const int image_size,
    const float radius,
    const int points_per_pixel,
    const int bin_size,
    const int max_points_per_bin) {
  if (bin_size == 0) {
    // Use the naive per-pixel implementation
    return RasterizePointsNaive(points, image_size, radius, points_per_pixel);
  } else {
    // Use coarse-to-fine rasterization
    const auto bin_points = RasterizePointsCoarse(
        points, image_size, radius, bin_size, max_points_per_bin);
    return RasterizePointsFine(
        points, bin_points, image_size, radius, bin_size, points_per_pixel);
  }
}
