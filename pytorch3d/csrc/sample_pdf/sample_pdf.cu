/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

// There is no intermediate memory, so no reason not to have blocksize=32.
// 256 is a reasonable number of blocks.

// DESIGN
// We exploit the fact that n_samples is not tiny.
// A chunk of work is T*blocksize many samples from
// a single batch elememt.
// For each batch element there will be
// chunks_per_batch = 1 + (n_samples-1)/(T*blocksize) of them.
// The number of potential chunks to do is
// n_chunks = chunks_per_batch * n_batches.
// These chunks are divided among the gridSize-many blocks.
// In block b, we work on chunks b, b+gridSize, b+2*gridSize etc .
// In chunk i, we work on batch_element i/chunks_per_batch
// on samples starting from (i%chunks_per_batch) * (T*blocksize)

// BEGIN HYPOTHETICAL
// Another option (not implemented) if batch_size was always large
// would be as follows.

// A chunk of work is S samples from each of blocksize-many
// batch elements.
// For each batch element there will be
// chunks_per_batch = (1+(n_samples-1)/S) of them.
// The number of potential chunks to do is
// n_chunks = chunks_per_batch * (1+(n_batches-1)/blocksize)
// These chunks are divided among the gridSize-many blocks.
// In block b, we work on chunks b, b+gridSize, b+2*gridSize etc .
// In chunk i, we work on samples starting from S*(i%chunks_per_batch)
// on batch elements starting from blocksize*(i/chunks_per_batch).
// END HYPOTHETICAL

__global__ void SamplePdfCudaKernel(
    const float* __restrict__ bins,
    const float* __restrict__ weights,
    float* __restrict__ outputs,
    float eps,
    const int T,
    const int64_t batch_size,
    const int64_t n_bins,
    const int64_t n_samples) {
  const int64_t chunks_per_batch = 1 + (n_samples - 1) / (T * blockDim.x);
  const int64_t n_chunks = chunks_per_batch * batch_size;

  for (int64_t i_chunk = blockIdx.x; i_chunk < n_chunks; i_chunk += gridDim.x) {
    // Loop over the chunks.
    int64_t i_batch_element = i_chunk / chunks_per_batch;
    int64_t sample_start = (i_chunk % chunks_per_batch) * (T * blockDim.x);
    const float* const weight_startp = weights + n_bins * i_batch_element;
    const float* const bin_startp = bins + (1 + n_bins) * i_batch_element;

    // Each chunk looks at a single batch element, so we do the preprocessing
    // which depends on the batch element, namely finding the total weight.
    // Idenntical work is being done in sync here by every thread of the block.
    float total_weight = eps;
    for (int64_t i_bin = 0; i_bin < n_bins; ++i_bin) {
      total_weight += weight_startp[i_bin];
    }

    float* const output_startp =
        outputs + n_samples * i_batch_element + sample_start;

    for (int t = 0; t < T; ++t) {
      // Loop over T, which is the number of samples each thread makes within
      // the chunk.
      const int64_t i_sample_within_chunk = threadIdx.x + t * blockDim.x;
      if (sample_start + i_sample_within_chunk >= n_samples) {
        // Some threads need to exit early because the sample they would
        // make is unwanted.
        continue;
      }
      // output_startp[i_sample_within_chunk] contains the quantile we (i.e.
      // this thread) are calcvulating.
      float uniform = total_weight * output_startp[i_sample_within_chunk];
      int64_t i_bin = 0;
      // We find the bin containing the quantile by walking along the weights.
      // This loop must be thread dependent. I.e. the whole warp will wait until
      // every thread has found the bin for its quantile.
      // It may be best to write it differently.
      while (i_bin + 1 < n_bins && uniform > weight_startp[i_bin]) {
        uniform -= weight_startp[i_bin];
        ++i_bin;
      }

      // Now we know which bin to look in, we use linear interpolation
      // to find the location of the quantile within the bin, and
      // write the answer back.
      float bin_start = bin_startp[i_bin];
      float bin_end = bin_startp[i_bin + 1];
      float bin_weight = weight_startp[i_bin];
      float output_value = bin_start;
      if (uniform > bin_weight) {
        output_value = bin_end;
      } else if (bin_weight > eps) {
        output_value += (uniform / bin_weight) * (bin_end - bin_start);
      }
      output_startp[i_sample_within_chunk] = output_value;
    }
  }
}

void SamplePdfCuda(
    const at::Tensor& bins,
    const at::Tensor& weights,
    const at::Tensor& outputs,
    float eps) {
  // Check inputs are on the same device
  at::TensorArg bins_t{bins, "bins", 1}, weights_t{weights, "weights", 2},
      outputs_t{outputs, "outputs", 3};
  at::CheckedFrom c = "SamplePdfCuda";
  at::checkAllSameGPU(c, {bins_t, weights_t, outputs_t});
  at::checkAllSameType(c, {bins_t, weights_t, outputs_t});

  // Set the device for the kernel launch based on the device of the input
  at::cuda::CUDAGuard device_guard(bins.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int64_t batch_size = bins.size(0);
  const int64_t n_bins = weights.size(1);
  const int64_t n_samples = outputs.size(1);

  const int64_t threads = 32;
  const int64_t T = n_samples <= threads ? 1 : 2;
  const int64_t chunks_per_batch = 1 + (n_samples - 1) / (T * threads);
  const int64_t n_chunks = chunks_per_batch * batch_size;

  const int64_t max_blocks = 1024;
  const int64_t blocks = n_chunks < max_blocks ? n_chunks : max_blocks;

  SamplePdfCudaKernel<<<blocks, threads, 0, stream>>>(
      bins.contiguous().data_ptr<float>(),
      weights.contiguous().data_ptr<float>(),
      outputs.data_ptr<float>(), // Checked contiguous in header file.
      eps,
      T,
      batch_size,
      n_bins,
      n_samples);

  AT_CUDA_CHECK(cudaGetLastError());
}
