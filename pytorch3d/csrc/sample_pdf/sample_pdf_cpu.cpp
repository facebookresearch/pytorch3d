/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <torch/csrc/autograd/VariableTypeUtils.h>
#include <torch/extension.h>
#include <algorithm>
#include <thread>
#include <vector>

// If the number of bins is the typical 64, it is
// quicker to use binary search than linear scan.
// With more bins, it is more important.
// There is no equivalent CUDA implementation yet.
#define USE_BINARY_SEARCH

namespace {
// This worker function does the job of SamplePdf but only on
// batch elements in [start_batch, end_batch).
void SamplePdfCpu_worker(
    const torch::Tensor& bins,
    const torch::Tensor& weights,
    const torch::Tensor& outputs,
    float eps,
    int64_t start_batch,
    int64_t end_batch) {
  const int64_t n_bins = weights.size(1);
  const int64_t n_samples = outputs.size(1);

  auto bins_a = bins.accessor<float, 2>();
  auto weights_a = weights.accessor<float, 2>();
  float* output_p = outputs.data_ptr<float>() + start_batch * n_samples;

#ifdef USE_BINARY_SEARCH
  std::vector<float> partial_sums(n_bins);
#endif

  for (int64_t i_batch_elt = start_batch; i_batch_elt < end_batch;
       ++i_batch_elt) {
    auto bin_a = bins_a[i_batch_elt];
    auto weight_a = weights_a[i_batch_elt];

    // Here we do the work which has to be done once per batch element.
    // i.e. (1) finding the total weight. (2) If using binary search,
    // precompute the partial sums of the weights.

    float total_weight = 0;
    for (int64_t i_bin = 0; i_bin < n_bins; ++i_bin) {
      total_weight += weight_a[i_bin];
#ifdef USE_BINARY_SEARCH
      partial_sums[i_bin] = total_weight;
#endif
    }
    total_weight += eps;

    for (int64_t i_sample = 0; i_sample < n_samples; ++i_sample) {
      // Here we are taking a single random quantile (which is stored
      // in *output_p) and using it to make a single sample, which we
      // write back to the same location. First we find which bin
      // the quantile lives in, either by binary search in the
      // precomputed partial sums, or by scanning through the weights.

      float uniform = total_weight * *output_p;
#ifdef USE_BINARY_SEARCH
      int64_t i_bin = std::lower_bound(
                          partial_sums.begin(), --partial_sums.end(), uniform) -
          partial_sums.begin();
      if (i_bin > 0) {
        uniform -= partial_sums[i_bin - 1];
      }
#else
      int64_t i_bin = 0;
      while (i_bin + 1 < n_bins && uniform > weight_a[i_bin]) {
        uniform -= weight_a[i_bin];
        ++i_bin;
      }
#endif

      // Now i_bin identifies the bin the quantile lives in, we use
      // straight line interpolation to find the position of the
      // quantile within the bin, and write it to *output_p.

      float bin_start = bin_a[i_bin];
      float bin_end = bin_a[i_bin + 1];
      float bin_weight = weight_a[i_bin];
      float output_value = bin_start;
      if (uniform > bin_weight) {
        output_value = bin_end;
      } else if (bin_weight > eps) {
        output_value += (uniform / bin_weight) * (bin_end - bin_start);
      }
      *output_p = output_value;
      ++output_p;
    }
  }
}

} // anonymous namespace

void SamplePdfCpu(
    const torch::Tensor& bins,
    const torch::Tensor& weights,
    const torch::Tensor& outputs,
    float eps) {
  const int64_t batch_size = bins.size(0);
  const int64_t max_threads = std::min(4, at::get_num_threads());
  const int64_t n_threads = std::min(max_threads, batch_size);
  if (batch_size == 0) {
    return;
  }

  // SamplePdfCpu_worker does the work of this function. We send separate ranges
  // of batch elements to that function in nThreads-1 separate threads.

  std::vector<std::thread> threads;
  threads.reserve(n_threads - 1);
  const int64_t batch_elements_per_thread = 1 + (batch_size - 1) / n_threads;
  int64_t start_batch = 0;
  for (int iThread = 0; iThread < n_threads - 1; ++iThread) {
    threads.emplace_back(
        SamplePdfCpu_worker,
        bins,
        weights,
        outputs,
        eps,
        start_batch,
        start_batch + batch_elements_per_thread);
    start_batch += batch_elements_per_thread;
  }

  // The remaining batch elements are calculated in this threads. If nThreads is
  // 1 then all the work happens in this line.
  SamplePdfCpu_worker(bins, weights, outputs, eps, start_batch, batch_size);
  for (auto&& thread : threads) {
    thread.join();
  }
  torch::autograd::increment_version(outputs);
}
