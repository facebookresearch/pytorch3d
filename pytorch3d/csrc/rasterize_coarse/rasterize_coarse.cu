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
#include <float.h>
#include <math.h>
#include <tuple>
#include "rasterize_coarse/bitmask.cuh"
#include "rasterize_points/rasterization_utils.cuh"
#include "utils/float_math.cuh"
#include "utils/geometry_utils.cuh" // For kEpsilon -- gross

__global__ void TriangleBoundingBoxKernel(
    const float* face_verts, // (F, 3, 3)
    const int F,
    const float blur_radius,
    float* bboxes, // (4, F)
    bool* skip_face) { // (F,)
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int num_threads = blockDim.x * gridDim.x;
  const float sqrt_radius = sqrt(blur_radius);
  for (int f = tid; f < F; f += num_threads) {
    const float v0x = face_verts[f * 9 + 0 * 3 + 0];
    const float v0y = face_verts[f * 9 + 0 * 3 + 1];
    const float v0z = face_verts[f * 9 + 0 * 3 + 2];
    const float v1x = face_verts[f * 9 + 1 * 3 + 0];
    const float v1y = face_verts[f * 9 + 1 * 3 + 1];
    const float v1z = face_verts[f * 9 + 1 * 3 + 2];
    const float v2x = face_verts[f * 9 + 2 * 3 + 0];
    const float v2y = face_verts[f * 9 + 2 * 3 + 1];
    const float v2z = face_verts[f * 9 + 2 * 3 + 2];
    const float xmin = FloatMin3(v0x, v1x, v2x) - sqrt_radius;
    const float xmax = FloatMax3(v0x, v1x, v2x) + sqrt_radius;
    const float ymin = FloatMin3(v0y, v1y, v2y) - sqrt_radius;
    const float ymax = FloatMax3(v0y, v1y, v2y) + sqrt_radius;
    const float zmin = FloatMin3(v0z, v1z, v2z);
    const bool skip = zmin < kEpsilon;
    bboxes[0 * F + f] = xmin;
    bboxes[1 * F + f] = xmax;
    bboxes[2 * F + f] = ymin;
    bboxes[3 * F + f] = ymax;
    skip_face[f] = skip;
  }
}

__global__ void PointBoundingBoxKernel(
    const float* points, // (P, 3)
    const float* radius, // (P,)
    const int P,
    float* bboxes, // (4, P)
    bool* skip_points) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int num_threads = blockDim.x * gridDim.x;
  for (int p = tid; p < P; p += num_threads) {
    const float x = points[p * 3 + 0];
    const float y = points[p * 3 + 1];
    const float z = points[p * 3 + 2];
    const float r = radius[p];
    // TODO: change to kEpsilon to match triangles?
    const bool skip = z < 0;
    bboxes[0 * P + p] = x - r;
    bboxes[1 * P + p] = x + r;
    bboxes[2 * P + p] = y - r;
    bboxes[3 * P + p] = y + r;
    skip_points[p] = skip;
  }
}

__global__ void RasterizeCoarseCudaKernel(
    const float* bboxes, // (4, E) (xmin, xmax, ymin, ymax)
    const bool* should_skip, // (E,)
    const int64_t* elem_first_idxs,
    const int64_t* elems_per_batch,
    const int N,
    const int E,
    const int H,
    const int W,
    const int bin_size,
    const int chunk_size,
    const int max_elem_per_bin,
    int* elems_per_bin,
    int* bin_elems) {
  extern __shared__ char sbuf[];
  const int M = max_elem_per_bin;
  // Integer divide round up
  const int num_bins_x = 1 + (W - 1) / bin_size;
  const int num_bins_y = 1 + (H - 1) / bin_size;

  // NDC range depends on the ratio of W/H
  // The shorter side from (H, W) is given an NDC range of 2.0 and
  // the other side is scaled by the ratio of H:W.
  const float NDC_x_half_range = NonSquareNdcRange(W, H) / 2.0f;
  const float NDC_y_half_range = NonSquareNdcRange(H, W) / 2.0f;

  // Size of half a pixel in NDC units is the NDC half range
  // divided by the corresponding image dimension
  const float half_pix_x = NDC_x_half_range / W;
  const float half_pix_y = NDC_y_half_range / H;

  // This is a boolean array of shape (num_bins_y, num_bins_x, chunk_size)
  // stored in shared memory that will track whether each elem in the chunk
  // falls into each bin of the image.
  BitMask binmask((unsigned int*)sbuf, num_bins_y, num_bins_x, chunk_size);

  // Have each block handle a chunk of elements
  const int chunks_per_batch = 1 + (E - 1) / chunk_size;
  const int num_chunks = N * chunks_per_batch;

  for (int chunk = blockIdx.x; chunk < num_chunks; chunk += gridDim.x) {
    const int batch_idx = chunk / chunks_per_batch; // batch index
    const int chunk_idx = chunk % chunks_per_batch;
    const int elem_chunk_start_idx = chunk_idx * chunk_size;

    binmask.block_clear();
    const int64_t elem_start_idx = elem_first_idxs[batch_idx];
    const int64_t elem_stop_idx = elem_start_idx + elems_per_batch[batch_idx];

    // Have each thread handle a different face within the chunk
    for (int e = threadIdx.x; e < chunk_size; e += blockDim.x) {
      const int e_idx = elem_chunk_start_idx + e;

      // Check that we are still within the same element of the batch
      if (e_idx >= elem_stop_idx || e_idx < elem_start_idx) {
        continue;
      }

      if (should_skip[e_idx]) {
        continue;
      }
      const float xmin = bboxes[0 * E + e_idx];
      const float xmax = bboxes[1 * E + e_idx];
      const float ymin = bboxes[2 * E + e_idx];
      const float ymax = bboxes[3 * E + e_idx];

      // Brute-force search over all bins; TODO(T54294966) something smarter.
      for (int by = 0; by < num_bins_y; ++by) {
        // Y coordinate of the top and bottom of the bin.
        // PixToNdc gives the location of the center of each pixel, so we
        // need to add/subtract a half pixel to get the true extent of the bin.
        // Reverse ordering of Y axis so that +Y is upwards in the image.
        const float bin_y_min =
            PixToNonSquareNdc(by * bin_size, H, W) - half_pix_y;
        const float bin_y_max =
            PixToNonSquareNdc((by + 1) * bin_size - 1, H, W) + half_pix_y;
        const bool y_overlap = (ymin <= bin_y_max) && (bin_y_min < ymax);

        for (int bx = 0; bx < num_bins_x; ++bx) {
          // X coordinate of the left and right of the bin.
          // Reverse ordering of x axis so that +X is left.
          const float bin_x_max =
              PixToNonSquareNdc((bx + 1) * bin_size - 1, W, H) + half_pix_x;
          const float bin_x_min =
              PixToNonSquareNdc(bx * bin_size, W, H) - half_pix_x;

          const bool x_overlap = (xmin <= bin_x_max) && (bin_x_min < xmax);
          if (y_overlap && x_overlap) {
            binmask.set(by, bx, e);
          }
        }
      }
    }
    __syncthreads();
    // Now we have processed every elem in the current chunk. We need to
    // count the number of elems in each bin so we can write the indices
    // out to global memory. We have each thread handle a different bin.
    for (int byx = threadIdx.x; byx < num_bins_y * num_bins_x;
         byx += blockDim.x) {
      const int by = byx / num_bins_x;
      const int bx = byx % num_bins_x;
      const int count = binmask.count(by, bx);
      const int elems_per_bin_idx =
          batch_idx * num_bins_y * num_bins_x + by * num_bins_x + bx;

      // This atomically increments the (global) number of elems found
      // in the current bin, and gets the previous value of the counter;
      // this effectively allocates space in the bin_faces array for the
      // elems in the current chunk that fall into this bin.
      const int start = atomicAdd(elems_per_bin + elems_per_bin_idx, count);
      if (start + count > M) {
        // The number of elems in this bin is so big that they won't fit.
        // We print a warning using CUDA's printf. This may be invisible
        // to notebook users, but apparent to others. It would be nice to
        // also have a Python-friendly warning, but it is not obvious
        // how to do this without slowing down the normal case.
        const char* warning =
            "Bin size was too small in the coarse rasterization phase. "
            "This caused an overflow, meaning output may be incomplete. "
            "To solve, "
            "try increasing max_faces_per_bin / max_points_per_bin, "
            "decreasing bin_size, "
            "or setting bin_size to 0 to use the naive rasterization.";
        printf(warning);
        continue;
      }

      // Now loop over the binmask and write the active bits for this bin
      // out to bin_faces.
      int next_idx = batch_idx * num_bins_y * num_bins_x * M +
          by * num_bins_x * M + bx * M + start;
      for (int e = 0; e < chunk_size; ++e) {
        if (binmask.get(by, bx, e)) {
          // TODO(T54296346) find the correct method for handling errors in
          // CUDA. Throw an error if num_faces_per_bin > max_faces_per_bin.
          // Either decrease bin size or increase max_faces_per_bin
          bin_elems[next_idx] = elem_chunk_start_idx + e;
          next_idx++;
        }
      }
    }
    __syncthreads();
  }
}

at::Tensor RasterizeCoarseCuda(
    const at::Tensor& bboxes,
    const at::Tensor& should_skip,
    const at::Tensor& elem_first_idxs,
    const at::Tensor& elems_per_batch,
    const std::tuple<int, int> image_size,
    const int bin_size,
    const int max_elems_per_bin) {
  // Set the device for the kernel launch based on the device of the input
  at::cuda::CUDAGuard device_guard(bboxes.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int H = std::get<0>(image_size);
  const int W = std::get<1>(image_size);

  const int E = bboxes.size(1);
  const int N = elems_per_batch.size(0);
  const int M = max_elems_per_bin;

  // Integer divide round up
  const int num_bins_y = 1 + (H - 1) / bin_size;
  const int num_bins_x = 1 + (W - 1) / bin_size;

  if (num_bins_y >= kMaxItemsPerBin || num_bins_x >= kMaxItemsPerBin) {
    std::stringstream ss;
    ss << "In RasterizeCoarseCuda got num_bins_y: " << num_bins_y
       << ", num_bins_x: " << num_bins_x << ", "
       << "; that's too many!";
    AT_ERROR(ss.str());
  }
  auto opts = elems_per_batch.options().dtype(at::kInt);
  at::Tensor elems_per_bin = at::zeros({N, num_bins_y, num_bins_x}, opts);
  at::Tensor bin_elems = at::full({N, num_bins_y, num_bins_x, M}, -1, opts);

  if (bin_elems.numel() == 0) {
    AT_CUDA_CHECK(cudaGetLastError());
    return bin_elems;
  }

  const int chunk_size = 512;
  const size_t shared_size = num_bins_y * num_bins_x * chunk_size / 8;
  const size_t blocks = 64;
  const size_t threads = 512;

  RasterizeCoarseCudaKernel<<<blocks, threads, shared_size, stream>>>(
      bboxes.contiguous().data_ptr<float>(),
      should_skip.contiguous().data_ptr<bool>(),
      elem_first_idxs.contiguous().data_ptr<int64_t>(),
      elems_per_batch.contiguous().data_ptr<int64_t>(),
      N,
      E,
      H,
      W,
      bin_size,
      chunk_size,
      M,
      elems_per_bin.data_ptr<int32_t>(),
      bin_elems.data_ptr<int32_t>());

  AT_CUDA_CHECK(cudaGetLastError());
  return bin_elems;
}

at::Tensor RasterizeMeshesCoarseCuda(
    const at::Tensor& face_verts,
    const at::Tensor& mesh_to_face_first_idx,
    const at::Tensor& num_faces_per_mesh,
    const std::tuple<int, int> image_size,
    const float blur_radius,
    const int bin_size,
    const int max_faces_per_bin) {
  TORCH_CHECK(
      face_verts.ndimension() == 3 && face_verts.size(1) == 3 &&
          face_verts.size(2) == 3,
      "face_verts must have dimensions (num_faces, 3, 3)");

  // Check inputs are on the same device
  at::TensorArg face_verts_t{face_verts, "face_verts", 1},
      mesh_to_face_first_idx_t{
          mesh_to_face_first_idx, "mesh_to_face_first_idx", 2},
      num_faces_per_mesh_t{num_faces_per_mesh, "num_faces_per_mesh", 3};
  at::CheckedFrom c = "RasterizeMeshesCoarseCuda";
  at::checkAllSameGPU(
      c, {face_verts_t, mesh_to_face_first_idx_t, num_faces_per_mesh_t});

  // Set the device for the kernel launch based on the device of the input
  at::cuda::CUDAGuard device_guard(face_verts.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // Allocate tensors for bboxes and should_skip
  const int F = face_verts.size(0);
  auto float_opts = face_verts.options().dtype(at::kFloat);
  auto bool_opts = face_verts.options().dtype(at::kBool);
  at::Tensor bboxes = at::empty({4, F}, float_opts);
  at::Tensor should_skip = at::empty({F}, bool_opts);

  // Launch kernel to compute triangle bboxes
  const size_t blocks = 128;
  const size_t threads = 256;
  TriangleBoundingBoxKernel<<<blocks, threads, 0, stream>>>(
      face_verts.contiguous().data_ptr<float>(),
      F,
      blur_radius,
      bboxes.contiguous().data_ptr<float>(),
      should_skip.contiguous().data_ptr<bool>());
  AT_CUDA_CHECK(cudaGetLastError());

  return RasterizeCoarseCuda(
      bboxes,
      should_skip,
      mesh_to_face_first_idx,
      num_faces_per_mesh,
      image_size,
      bin_size,
      max_faces_per_bin);
}

at::Tensor RasterizePointsCoarseCuda(
    const at::Tensor& points, // (P, 3)
    const at::Tensor& cloud_to_packed_first_idx, // (N,)
    const at::Tensor& num_points_per_cloud, // (N,)
    const std::tuple<int, int> image_size,
    const at::Tensor& radius,
    const int bin_size,
    const int max_points_per_bin) {
  TORCH_CHECK(
      points.ndimension() == 2 && points.size(1) == 3,
      "points must have dimensions (num_points, 3)");

  // Check inputs are on the same device
  at::TensorArg points_t{points, "points", 1},
      cloud_to_packed_first_idx_t{
          cloud_to_packed_first_idx, "cloud_to_packed_first_idx", 2},
      num_points_per_cloud_t{num_points_per_cloud, "num_points_per_cloud", 3};
  at::CheckedFrom c = "RasterizePointsCoarseCuda";
  at::checkAllSameGPU(
      c, {points_t, cloud_to_packed_first_idx_t, num_points_per_cloud_t});

  // Set the device for the kernel launch based on the device of the input
  at::cuda::CUDAGuard device_guard(points.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // Allocate tensors for bboxes and should_skip
  const int P = points.size(0);
  auto float_opts = points.options().dtype(at::kFloat);
  auto bool_opts = points.options().dtype(at::kBool);
  at::Tensor bboxes = at::empty({4, P}, float_opts);
  at::Tensor should_skip = at::empty({P}, bool_opts);

  // Launch kernel to compute point bboxes
  const size_t blocks = 128;
  const size_t threads = 256;
  PointBoundingBoxKernel<<<blocks, threads, 0, stream>>>(
      points.contiguous().data_ptr<float>(),
      radius.contiguous().data_ptr<float>(),
      P,
      bboxes.contiguous().data_ptr<float>(),
      should_skip.contiguous().data_ptr<bool>());
  AT_CUDA_CHECK(cudaGetLastError());

  return RasterizeCoarseCuda(
      bboxes,
      should_skip,
      cloud_to_packed_first_idx,
      num_points_per_cloud,
      image_size,
      bin_size,
      max_points_per_bin);
}
