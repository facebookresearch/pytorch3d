// Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

// TODO(T47953967) to make this cuda kernel support all datatypes.
__global__ void GatherScatterCudaKernel(
    const float* __restrict__ input,
    const int64_t* __restrict__ edges,
    float* __restrict__ output,
    bool directed,
    bool backward,
    const size_t V,
    const size_t D,
    const size_t E) {
  const int tid = threadIdx.x;

  // Reverse the vertex order if backward.
  const int v0_idx = backward ? 1 : 0;
  const int v1_idx = backward ? 0 : 1;

  // Edges are split evenly across the blocks.
  for (int e = blockIdx.x; e < E; e += gridDim.x) {
    // Get indices of vertices which form the edge.
    const int64_t v0 = edges[2 * e + v0_idx];
    const int64_t v1 = edges[2 * e + v1_idx];

    // Split vertex features evenly across threads.
    // This implementation will be quite wasteful when D<128 since there will be
    // a lot of threads doing nothing.
    for (int d = tid; d < D; d += blockDim.x) {
      const float val = input[v1 * D + d];
      float* address = output + v0 * D + d;
      atomicAdd(address, val);
      if (!directed) {
        const float val = input[v0 * D + d];
        float* address = output + v1 * D + d;
        atomicAdd(address, val);
      }
    }
    __syncthreads();
  }
}

at::Tensor GatherScatterCuda(
    const at::Tensor input,
    const at::Tensor edges,
    bool directed,
    bool backward) {
  // Check inputs are on the same device
  at::TensorArg input_t{input, "input", 1}, edges_t{edges, "edges", 2};
  at::CheckedFrom c = "GatherScatterCuda";
  at::checkAllSameGPU(c, {input_t, edges_t});

  // Set the device for the kernel launch based on the device of the input
  at::cuda::CUDAGuard device_guard(input.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const auto num_vertices = input.size(0);
  const auto input_feature_dim = input.size(1);
  const auto num_edges = edges.size(0);

  auto output = at::zeros({num_vertices, input_feature_dim}, input.options());
  const size_t threads = 128;
  const size_t max_blocks = 1920;
  const size_t blocks = num_edges < max_blocks ? num_edges : max_blocks;

  if (output.numel() == 0) {
    AT_CUDA_CHECK(cudaGetLastError());
    return output;
  }

  GatherScatterCudaKernel<<<blocks, threads, 0, stream>>>(
      input.data_ptr<float>(),
      edges.data_ptr<int64_t>(),
      output.data_ptr<float>(),
      directed,
      backward,
      num_vertices,
      input_feature_dim,
      num_edges);
  AT_CUDA_CHECK(cudaGetLastError());
  return output;
}
