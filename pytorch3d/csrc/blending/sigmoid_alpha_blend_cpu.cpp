/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <torch/extension.h>
#include <cmath>
#include <vector>

at::Tensor SigmoidAlphaBlendForwardCpu(
    const at::Tensor& distances, // (N, H, W, K)
    const at::Tensor& pix_to_face, // (N, H, W, K)
    const float sigma) {
  const int N = distances.size(0);
  const int H = distances.size(1);
  const int W = distances.size(2);
  const int K = distances.size(3);

  torch::Tensor out = torch::empty({N, H, W}, distances.options());

  auto distances_a = distances.accessor<float, 4>();
  auto pix_to_face_a = pix_to_face.accessor<int64_t, 4>();
  auto out_a = out.accessor<float, 3>();

  // Iterate over the images in the batch.
  for (int n = 0; n < N; ++n) {
    // Iterate through the horizontal lines of the image from top to bottom.
    for (int h = 0; h < H; ++h) {
      // Iterate over the pixels on this horizontal line, left to right.
      for (int w = 0; w < W; ++w) {
        float alpha = 1.0;

        // Loop through the top K faces for each pixel.
        for (int k = 0; k < K; ++k) {
          const int f = pix_to_face_a[n][h][w][k];
          if (f < 0) {
            // Sentinel value is -1 indicating no face overlaps the pixel.
            continue;
          }
          // The distance is negative if a pixel is inside a face and positive
          // outside the face. Therefore use -1.0 * the distance to get the
          // correct sign.
          float dist = -1.0 * distances_a[n][h][w][k];

          // Calculate the sigmoid probability.
          float prob = 1. / (1. + exp(-dist / sigma));

          // The product ensures that alpha will be 0.0 if at least 1
          // face fully covers the pixel as for that face, prob will be 1.0.
          // This results in a multiplication by 0.0 because of the (1.0 - prob)
          // term. Therefore 1.0 - alpha will be 1.0.
          alpha *= 1.0 - prob;
        }
        out_a[n][h][w] = 1.0 - alpha;
      }
    }
  }
  return out;
}

at::Tensor SigmoidAlphaBlendBackwardCpu(
    const at::Tensor& grad_alphas, // (N, H, W)
    const at::Tensor& alphas, // (N, H, W)
    const at::Tensor& distances, // (N, H, W, K)
    const at::Tensor& pix_to_face, // (N, H, W, K)
    const float sigma) {
  const int N = distances.size(0);
  const int H = distances.size(1);
  const int W = distances.size(2);
  const int K = distances.size(3);

  auto distances_a = distances.accessor<float, 4>();
  auto pix_to_face_a = pix_to_face.accessor<int64_t, 4>();
  auto alphas_a = alphas.accessor<float, 3>();
  auto grad_alphas_a = grad_alphas.accessor<float, 3>();

  torch::Tensor grad_distances =
      torch::zeros({N, H, W, K}, distances.options());
  auto grad_distances_a = grad_distances.accessor<float, 4>();

  // Iterate over the images in the batch.
  for (int n = 0; n < N; ++n) {
    // Iterate through the horizontal lines of the image from top to bottom.
    for (int h = 0; h < H; ++h) {
      // Iterate over the pixels on this horizontal line, left to right.
      for (int w = 0; w < W; ++w) {
        // Get the alpha value from the forward pass and the
        // upstream gradient.
        const float alpha = 1.0 - alphas_a[n][h][w];
        const float grad_alpha = grad_alphas_a[n][h][w];

        // Loop through the top K faces for each pixel.
        for (int k = 0; k < K; ++k) {
          const int f = pix_to_face_a[n][h][w][k];
          if (f < 0) {
            // Sentinel value is -1 indicating no face overlaps the pixel
            continue;
          }
          // The distance is negative if a pixel is inside a face and positive
          // outside the face. Therefore use -1.0 * distance to get the
          // correct sign.
          float dist = -1.0 * distances_a[n][h][w][k];

          // Calculate the sigmoid probability.
          float prob = 1. / (1. + exp(-dist / sigma));

          // clang-format off
          // We need to take the derivative of alpha w.r.t to the distance.
          // alpha = 1.0 - (1.0- sigmoid(-x)) * (1.0 - sigmoid(-x2)) * ... * (1.0 - sigmoid(-xn))
          //
          // Note that d/dx sigmoid(x) = sigmoid(x) * (1.0 - sigmoid(x))
          //
          // This gives:
          // d_alpha/d_dist = -1.0 * -1.0 * sigmoid(-x)(1. - sigmoid(-x)) * (-1.0/sigma)
          //        * ((1.0 - sigmoid(-x2) * ... * (1.0 - sigmoid(-xn))
          //    = (-1.0/sigma) * prob * (1.0 - prob) * alpha/(1.0 - prob)
          //    = (-1.0/sigma) * prob * alpha
          // clang-format on
          grad_distances_a[n][h][w][k] =
              grad_alpha * (-1.0 / sigma) * prob * alpha;
        }
      }
    }
  }
  return grad_distances;
}
