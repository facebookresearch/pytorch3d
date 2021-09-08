/*
 * Copyright (c) Facebook, Inc. and its affiliates.
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

__global__ void RasterizePointsCoarseCudaKernel(
    const float* points, // (P, 3)
    const int64_t* cloud_to_packed_first_idx, // (N)
    const int64_t* num_points_per_cloud, // (N)
    const float* radius,
    const int N,
    const int P,
    const int H,
    const int W,
    const int bin_size,
    const int chunk_size,
    const int max_points_per_bin,
    int* points_per_bin,
    int* bin_points) {
  extern __shared__ char sbuf[];
  const int M = max_points_per_bin;

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
  // stored in shared memory that will track whether each point in the chunk
  // falls into each bin of the image.
  BitMask binmask((unsigned int*)sbuf, num_bins_y, num_bins_x, chunk_size);

  // Have each block handle a chunk of points and build a 3D bitmask in
  // shared memory to mark which points hit which bins.  In this first phase,
  // each thread processes one point at a time. After processing the chunk,
  // one thread is assigned per bin, and the thread counts and writes the
  // points for the bin out to global memory.
  const int chunks_per_batch = 1 + (P - 1) / chunk_size;
  const int num_chunks = N * chunks_per_batch;
  for (int chunk = blockIdx.x; chunk < num_chunks; chunk += gridDim.x) {
    const int batch_idx = chunk / chunks_per_batch;
    const int chunk_idx = chunk % chunks_per_batch;
    const int point_start_idx = chunk_idx * chunk_size;

    binmask.block_clear();

    // Using the batch index of the thread get the start and stop
    // indices for the points.
    const int64_t cloud_point_start_idx = cloud_to_packed_first_idx[batch_idx];
    const int64_t cloud_point_stop_idx =
        cloud_point_start_idx + num_points_per_cloud[batch_idx];

    // Have each thread handle a different point within the chunk
    for (int p = threadIdx.x; p < chunk_size; p += blockDim.x) {
      const int p_idx = point_start_idx + p;

      // Check if point index corresponds to the cloud in the batch given by
      // batch_idx.
      if (p_idx >= cloud_point_stop_idx || p_idx < cloud_point_start_idx) {
        continue;
      }

      const float px = points[p_idx * 3 + 0];
      const float py = points[p_idx * 3 + 1];
      const float pz = points[p_idx * 3 + 2];
      const float p_radius = radius[p_idx];
      if (pz < 0)
        continue; // Don't render points behind the camera.
      const float px0 = px - p_radius;
      const float px1 = px + p_radius;
      const float py0 = py - p_radius;
      const float py1 = py + p_radius;

      // Brute-force search over all bins; TODO something smarter?
      // For example we could compute the exact bin where the point falls,
      // then check neighboring bins. This way we wouldn't have to check
      // all bins (however then we might have more warp divergence?)
      for (int by = 0; by < num_bins_y; ++by) {
        // Get y extent for the bin. PixToNonSquareNdc gives us the location of
        // the center of each pixel, so we need to add/subtract a half
        // pixel to get the true extent of the bin.
        const float by0 = PixToNonSquareNdc(by * bin_size, H, W) - half_pix_y;
        const float by1 =
            PixToNonSquareNdc((by + 1) * bin_size - 1, H, W) + half_pix_y;
        const bool y_overlap = (py0 <= by1) && (by0 <= py1);

        if (!y_overlap) {
          continue;
        }
        for (int bx = 0; bx < num_bins_x; ++bx) {
          // Get x extent for the bin; again we need to adjust the
          // output of PixToNonSquareNdc by half a pixel.
          const float bx0 = PixToNonSquareNdc(bx * bin_size, W, H) - half_pix_x;
          const float bx1 =
              PixToNonSquareNdc((bx + 1) * bin_size - 1, W, H) + half_pix_x;
          const bool x_overlap = (px0 <= bx1) && (bx0 <= px1);

          if (x_overlap) {
            binmask.set(by, bx, p);
          }
        }
      }
    }
    __syncthreads();
    // Now we have processed every point in the current chunk. We need to
    // count the number of points in each bin so we can write the indices
    // out to global memory. We have each thread handle a different bin.
    for (int byx = threadIdx.x; byx < num_bins_y * num_bins_x;
         byx += blockDim.x) {
      const int by = byx / num_bins_x;
      const int bx = byx % num_bins_x;
      const int count = binmask.count(by, bx);
      const int points_per_bin_idx =
          batch_idx * num_bins_y * num_bins_x + by * num_bins_x + bx;

      // This atomically increments the (global) number of points found
      // in the current bin, and gets the previous value of the counter;
      // this effectively allocates space in the bin_points array for the
      // points in the current chunk that fall into this bin.
      const int start = atomicAdd(points_per_bin + points_per_bin_idx, count);

      // Now loop over the binmask and write the active bits for this bin
      // out to bin_points.
      int next_idx = batch_idx * num_bins_y * num_bins_x * M +
          by * num_bins_x * M + bx * M + start;
      for (int p = 0; p < chunk_size; ++p) {
        if (binmask.get(by, bx, p)) {
          // TODO: Throw an error if next_idx >= M -- this means that
          // we got more than max_points_per_bin in this bin
          // TODO: check if atomicAdd is needed in line 265.
          bin_points[next_idx] = point_start_idx + p;
          next_idx++;
        }
      }
    }
    __syncthreads();
  }
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
    const at::Tensor& cloud_to_packed_first_idx, // (N)
    const at::Tensor& num_points_per_cloud, // (N)
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

  const int H = std::get<0>(image_size);
  const int W = std::get<1>(image_size);

  const int P = points.size(0);
  const int N = num_points_per_cloud.size(0);
  const int M = max_points_per_bin;

  // Integer divide round up.
  const int num_bins_y = 1 + (H - 1) / bin_size;
  const int num_bins_x = 1 + (W - 1) / bin_size;

  if (num_bins_y >= kMaxItemsPerBin || num_bins_x >= kMaxItemsPerBin) {
    // Make sure we do not use too much shared memory.
    std::stringstream ss;
    ss << "In Coarse Rasterizer got num_bins_y: " << num_bins_y
       << ", num_bins_x: " << num_bins_x << ", "
       << "; that's too many!";
    AT_ERROR(ss.str());
  }
  auto opts = num_points_per_cloud.options().dtype(at::kInt);
  at::Tensor points_per_bin = at::zeros({N, num_bins_y, num_bins_x}, opts);
  at::Tensor bin_points = at::full({N, num_bins_y, num_bins_x, M}, -1, opts);

  if (bin_points.numel() == 0) {
    AT_CUDA_CHECK(cudaGetLastError());
    return bin_points;
  }

  const int chunk_size = 512;
  const size_t shared_size = num_bins_y * num_bins_x * chunk_size / 8;
  const size_t blocks = 64;
  const size_t threads = 512;

  RasterizePointsCoarseCudaKernel<<<blocks, threads, shared_size, stream>>>(
      points.contiguous().data_ptr<float>(),
      cloud_to_packed_first_idx.contiguous().data_ptr<int64_t>(),
      num_points_per_cloud.contiguous().data_ptr<int64_t>(),
      radius.contiguous().data_ptr<float>(),
      N,
      P,
      H,
      W,
      bin_size,
      chunk_size,
      M,
      points_per_bin.contiguous().data_ptr<int32_t>(),
      bin_points.contiguous().data_ptr<int32_t>());

  AT_CUDA_CHECK(cudaGetLastError());
  return bin_points;
}
