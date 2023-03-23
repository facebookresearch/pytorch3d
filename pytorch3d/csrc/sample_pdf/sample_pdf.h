/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <torch/csrc/autograd/VariableTypeUtils.h>
#include <torch/extension.h>
#include <cstdio>
#include <tuple>
#include "utils/pytorch3d_cutils.h"

// ****************************************************************************
// *                          SamplePdf                                       *
// ****************************************************************************

//  Samples a probability density functions defined by bin edges `bins` and
//  the non-negative per-bin probabilities `weights`.

//  Args:
//      bins: FloatTensor of shape `(batch_size, n_bins+1)` denoting the edges
//      of the sampling bins.

//      weights: FloatTensor of shape `(batch_size, n_bins)` containing
//      non-negative numbers representing the probability of sampling the
//      corresponding bin.

//      uniforms: The quantiles to draw, FloatTensor of shape
//      `(batch_size, n_samples)`.

//      outputs: On call, this contains the quantiles to draw. It is overwritten
//              with the drawn samples. FloatTensor of shape
//              `(batch_size, n_samples), where `n_samples are drawn from each
//               distribution.

//      eps: A constant preventing division by zero in case empty bins are
//      present.

//  Not differentiable

#ifdef WITH_CUDA
void SamplePdfCuda(
    const torch::Tensor& bins,
    const torch::Tensor& weights,
    const torch::Tensor& outputs,
    float eps);
#endif

void SamplePdfCpu(
    const torch::Tensor& bins,
    const torch::Tensor& weights,
    const torch::Tensor& outputs,
    float eps);

inline void SamplePdf(
    const torch::Tensor& bins,
    const torch::Tensor& weights,
    const torch::Tensor& outputs,
    float eps) {
  if (bins.is_cuda()) {
#ifdef WITH_CUDA
    CHECK_CUDA(weights);
    CHECK_CONTIGUOUS_CUDA(outputs);
    torch::autograd::increment_version(outputs);
    SamplePdfCuda(bins, weights, outputs, eps);
    return;
#else
    AT_ERROR("Not compiled with GPU support.");
#endif
  }
  CHECK_CONTIGUOUS(outputs);
  SamplePdfCpu(bins, weights, outputs, eps);
}
