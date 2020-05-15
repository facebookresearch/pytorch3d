// Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <float.h>
#include <math.h>
#include <thrust/tuple.h>
#include <cstdio>
#include <tuple>
#include "rasterize_points/bitmask.cuh"
#include "rasterize_points/rasterization_utils.cuh"
#include "utils/float_math.cuh"
#include "utils/geometry_utils.cuh"

namespace {
// A structure for holding details about a pixel.
struct Pixel {
  float z;
  int64_t idx;
  float dist;
  float3 bary;
};

__device__ bool operator<(const Pixel& a, const Pixel& b) {
  return a.z < b.z;
}

__device__ float FloatMin3(const float p1, const float p2, const float p3) {
  return fminf(p1, fminf(p2, p3));
}

__device__ float FloatMax3(const float p1, const float p2, const float p3) {
  return fmaxf(p1, fmaxf(p2, p3));
}

// Get the xyz coordinates of the three vertices for the face given by the
// index face_idx into face_verts.
__device__ thrust::tuple<float3, float3, float3> GetSingleFaceVerts(
    const float* face_verts,
    int face_idx) {
  const float x0 = face_verts[face_idx * 9 + 0];
  const float y0 = face_verts[face_idx * 9 + 1];
  const float z0 = face_verts[face_idx * 9 + 2];
  const float x1 = face_verts[face_idx * 9 + 3];
  const float y1 = face_verts[face_idx * 9 + 4];
  const float z1 = face_verts[face_idx * 9 + 5];
  const float x2 = face_verts[face_idx * 9 + 6];
  const float y2 = face_verts[face_idx * 9 + 7];
  const float z2 = face_verts[face_idx * 9 + 8];

  const float3 v0xyz = make_float3(x0, y0, z0);
  const float3 v1xyz = make_float3(x1, y1, z1);
  const float3 v2xyz = make_float3(x2, y2, z2);

  return thrust::make_tuple(v0xyz, v1xyz, v2xyz);
}

// Get the min/max x/y/z values for the face given by vertices v0, v1, v2.
__device__ thrust::tuple<float2, float2, float2>
GetFaceBoundingBox(float3 v0, float3 v1, float3 v2) {
  const float xmin = FloatMin3(v0.x, v1.x, v2.x);
  const float ymin = FloatMin3(v0.y, v1.y, v2.y);
  const float zmin = FloatMin3(v0.z, v1.z, v2.z);
  const float xmax = FloatMax3(v0.x, v1.x, v2.x);
  const float ymax = FloatMax3(v0.y, v1.y, v2.y);
  const float zmax = FloatMax3(v0.z, v1.z, v2.z);

  return thrust::make_tuple(
      make_float2(xmin, xmax),
      make_float2(ymin, ymax),
      make_float2(zmin, zmax));
}

// Check if the point (px, py) lies outside the face bounding box face_bbox.
// Return true if the point is outside.
__device__ bool CheckPointOutsideBoundingBox(
    float3 v0,
    float3 v1,
    float3 v2,
    float blur_radius,
    float2 pxy) {
  const auto bbox = GetFaceBoundingBox(v0, v1, v2);
  const float2 xlims = thrust::get<0>(bbox);
  const float2 ylims = thrust::get<1>(bbox);
  const float2 zlims = thrust::get<2>(bbox);

  const float x_min = xlims.x - blur_radius;
  const float y_min = ylims.x - blur_radius;
  const float x_max = xlims.y + blur_radius;
  const float y_max = ylims.y + blur_radius;

  // Check if the current point is oustside the triangle bounding box.
  return (pxy.x > x_max || pxy.x < x_min || pxy.y > y_max || pxy.y < y_min);
}

// This function checks if a pixel given by xy location pxy lies within the
// face with index face_idx in face_verts. One of the inputs is a list (q)
// which contains Pixel structs with the indices of the faces which intersect
// with this pixel sorted by closest z distance. If the point pxy lies in the
// face, the list (q) is updated and re-orderered in place. In addition
// the auxillary variables q_size, q_max_z and q_max_idx are also modified.
// This code is shared between RasterizeMeshesNaiveCudaKernel and
// RasterizeMeshesFineCudaKernel.
template <typename FaceQ>
__device__ void CheckPixelInsideFace(
    const float* face_verts, // (F, 3, 3)
    const int face_idx,
    int& q_size,
    float& q_max_z,
    int& q_max_idx,
    FaceQ& q,
    const float blur_radius,
    const float2 pxy, // Coordinates of the pixel
    const int K,
    const bool perspective_correct,
    const bool cull_backfaces) {
  const auto v012 = GetSingleFaceVerts(face_verts, face_idx);
  const float3 v0 = thrust::get<0>(v012);
  const float3 v1 = thrust::get<1>(v012);
  const float3 v2 = thrust::get<2>(v012);

  // Only need xy for barycentric coordinates and distance calculations.
  const float2 v0xy = make_float2(v0.x, v0.y);
  const float2 v1xy = make_float2(v1.x, v1.y);
  const float2 v2xy = make_float2(v2.x, v2.y);

  // Perform checks and skip if:
  // 1. the face is behind the camera
  // 2. the face is facing away from the camera
  // 3. the face has very small face area
  // 4. the pixel is outside the face bbox
  const float zmax = FloatMax3(v0.z, v1.z, v2.z);
  const bool outside_bbox = CheckPointOutsideBoundingBox(
      v0, v1, v2, sqrt(blur_radius), pxy); // use sqrt of blur for bbox
  const float face_area = EdgeFunctionForward(v0xy, v1xy, v2xy);
  // Check if the face is visible to the camera.
  const bool back_face = face_area < 0.0;
  const bool zero_face_area =
      (face_area <= kEpsilon && face_area >= -1.0f * kEpsilon);

  if (zmax < 0 || cull_backfaces && back_face || outside_bbox ||
      zero_face_area) {
    return;
  }

  // Calculate barycentric coords and euclidean dist to triangle.
  const float3 p_bary0 = BarycentricCoordsForward(pxy, v0xy, v1xy, v2xy);
  const float3 p_bary = !perspective_correct
      ? p_bary0
      : BarycentricPerspectiveCorrectionForward(p_bary0, v0.z, v1.z, v2.z);

  const float pz = p_bary.x * v0.z + p_bary.y * v1.z + p_bary.z * v2.z;
  if (pz < 0) {
    return; // Face is behind the image plane.
  }

  // Get abs squared distance
  const float dist = PointTriangleDistanceForward(pxy, v0xy, v1xy, v2xy);

  // Use the bary coordinates to determine if the point is inside the face.
  const bool inside = p_bary.x > 0.0f && p_bary.y > 0.0f && p_bary.z > 0.0f;
  const float signed_dist = inside ? -dist : dist;

  // Check if pixel is outside blur region
  if (!inside && dist >= blur_radius) {
    return;
  }

  if (q_size < K) {
    // Just insert it.
    q[q_size] = {pz, face_idx, signed_dist, p_bary};
    if (pz > q_max_z) {
      q_max_z = pz;
      q_max_idx = q_size;
    }
    q_size++;
  } else if (pz < q_max_z) {
    // Overwrite the old max, and find the new max.
    q[q_max_idx] = {pz, face_idx, signed_dist, p_bary};
    q_max_z = pz;
    for (int i = 0; i < K; i++) {
      if (q[i].z > q_max_z) {
        q_max_z = q[i].z;
        q_max_idx = i;
      }
    }
  }
}
} // namespace

// ****************************************************************************
// *                          NAIVE RASTERIZATION                      *
// ****************************************************************************
__global__ void RasterizeMeshesNaiveCudaKernel(
    const float* face_verts,
    const int64_t* mesh_to_face_first_idx,
    const int64_t* num_faces_per_mesh,
    const float blur_radius,
    const bool perspective_correct,
    const bool cull_backfaces,
    const int N,
    const int H,
    const int W,
    const int K,
    int64_t* face_idxs,
    float* zbuf,
    float* pix_dists,
    float* bary) {
  // Simple version: One thread per output pixel
  int num_threads = gridDim.x * blockDim.x;
  int tid = blockDim.x * blockIdx.x + threadIdx.x;

  for (int i = tid; i < N * H * W; i += num_threads) {
    // Convert linear index to 3D index
    const int n = i / (H * W); // batch index.
    const int pix_idx = i % (H * W);

    // Reverse ordering of X and Y axes
    const int yi = H - 1 - pix_idx / W;
    const int xi = W - 1 - pix_idx % W;

    // screen coordinates to ndc coordiantes of pixel.
    const float xf = PixToNdc(xi, W);
    const float yf = PixToNdc(yi, H);
    const float2 pxy = make_float2(xf, yf);

    // For keeping track of the K closest points we want a data structure
    // that (1) gives O(1) access to the closest point for easy comparisons,
    // and (2) allows insertion of new elements. In the CPU version we use
    // std::priority_queue; then (2) is O(log K). We can't use STL
    // containers in CUDA; we could roll our own max heap in an array, but
    // that would likely have a lot of warp divergence so we do something
    // simpler instead: keep the elements in an unsorted array, but keep
    // track of the max value and the index of the max value. Then (1) is
    // still O(1) time, while (2) is O(K) with a clean loop. Since K <= 8
    // this should be fast enough for our purposes.
    Pixel q[kMaxPointsPerPixel];
    int q_size = 0;
    float q_max_z = -1000;
    int q_max_idx = -1;

    // Using the batch index of the thread get the start and stop
    // indices for the faces.
    const int64_t face_start_idx = mesh_to_face_first_idx[n];
    const int64_t face_stop_idx = face_start_idx + num_faces_per_mesh[n];

    // Loop through the faces in the mesh.
    for (int f = face_start_idx; f < face_stop_idx; ++f) {
      // Check if the pixel pxy is inside the face bounding box and if it is,
      // update q, q_size, q_max_z and q_max_idx in place.
      CheckPixelInsideFace(
          face_verts,
          f,
          q_size,
          q_max_z,
          q_max_idx,
          q,
          blur_radius,
          pxy,
          K,
          perspective_correct,
          cull_backfaces);
    }

    // TODO: make sorting an option as only top k is needed, not sorted values.
    BubbleSort(q, q_size);
    int idx = n * H * W * K + pix_idx * K;
    for (int k = 0; k < q_size; ++k) {
      face_idxs[idx + k] = q[k].idx;
      zbuf[idx + k] = q[k].z;
      pix_dists[idx + k] = q[k].dist;
      bary[(idx + k) * 3 + 0] = q[k].bary.x;
      bary[(idx + k) * 3 + 1] = q[k].bary.y;
      bary[(idx + k) * 3 + 2] = q[k].bary.z;
    }
  }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
RasterizeMeshesNaiveCuda(
    const at::Tensor& face_verts,
    const at::Tensor& mesh_to_faces_packed_first_idx,
    const at::Tensor& num_faces_per_mesh,
    const int image_size,
    const float blur_radius,
    const int num_closest,
    const bool perspective_correct,
    const bool cull_backfaces) {
  TORCH_CHECK(
      face_verts.ndimension() == 3 && face_verts.size(1) == 3 &&
          face_verts.size(2) == 3,
      "face_verts must have dimensions (num_faces, 3, 3)");

  TORCH_CHECK(
      num_faces_per_mesh.size(0) == mesh_to_faces_packed_first_idx.size(0),
      "num_faces_per_mesh must have save size first dimension as mesh_to_faces_packed_first_idx");

  if (num_closest > kMaxPointsPerPixel) {
    std::stringstream ss;
    ss << "Must have points_per_pixel <= " << kMaxPointsPerPixel;
    AT_ERROR(ss.str());
  }

  // Check inputs are on the same device
  at::TensorArg face_verts_t{face_verts, "face_verts", 1},
      mesh_to_faces_packed_first_idx_t{
          mesh_to_faces_packed_first_idx, "mesh_to_faces_packed_first_idx", 2},
      num_faces_per_mesh_t{num_faces_per_mesh, "num_faces_per_mesh", 3};
  at::CheckedFrom c = "RasterizeMeshesNaiveCuda";
  at::checkAllSameGPU(
      c,
      {face_verts_t, mesh_to_faces_packed_first_idx_t, num_faces_per_mesh_t});

  // Set the device for the kernel launch based on the device of the input
  at::cuda::CUDAGuard device_guard(face_verts.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int N = num_faces_per_mesh.size(0); // batch size.
  const int H = image_size; // Assume square images.
  const int W = image_size;
  const int K = num_closest;

  auto long_opts = num_faces_per_mesh.options().dtype(at::kLong);
  auto float_opts = face_verts.options().dtype(at::kFloat);

  at::Tensor face_idxs = at::full({N, H, W, K}, -1, long_opts);
  at::Tensor zbuf = at::full({N, H, W, K}, -1, float_opts);
  at::Tensor pix_dists = at::full({N, H, W, K}, -1, float_opts);
  at::Tensor bary = at::full({N, H, W, K, 3}, -1, float_opts);

  if (face_idxs.numel() == 0) {
    AT_CUDA_CHECK(cudaGetLastError());
    return std::make_tuple(face_idxs, zbuf, bary, pix_dists);
  }

  const size_t blocks = 1024;
  const size_t threads = 64;

  RasterizeMeshesNaiveCudaKernel<<<blocks, threads, 0, stream>>>(
      face_verts.contiguous().data_ptr<float>(),
      mesh_to_faces_packed_first_idx.contiguous().data_ptr<int64_t>(),
      num_faces_per_mesh.contiguous().data_ptr<int64_t>(),
      blur_radius,
      perspective_correct,
      cull_backfaces,
      N,
      H,
      W,
      K,
      face_idxs.data_ptr<int64_t>(),
      zbuf.data_ptr<float>(),
      pix_dists.data_ptr<float>(),
      bary.data_ptr<float>());

  AT_CUDA_CHECK(cudaGetLastError());
  return std::make_tuple(face_idxs, zbuf, bary, pix_dists);
}

// ****************************************************************************
// *                            BACKWARD PASS                                 *
// ****************************************************************************
// TODO: benchmark parallelizing over faces_verts instead of over pixels.
__global__ void RasterizeMeshesBackwardCudaKernel(
    const float* face_verts, // (F, 3, 3)
    const int64_t* pix_to_face, // (N, H, W, K)
    const bool perspective_correct,
    const int N,
    const int H,
    const int W,
    const int K,
    const float* grad_zbuf, // (N, H, W, K)
    const float* grad_bary, // (N, H, W, K, 3)
    const float* grad_dists, // (N, H, W, K)
    float* grad_face_verts) { // (F, 3, 3)

  // Parallelize over each pixel in images of
  // size H * W, for each image in the batch of size N.
  const int num_threads = gridDim.x * blockDim.x;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;

  for (int t_i = tid; t_i < N * H * W; t_i += num_threads) {
    // Convert linear index to 3D index
    const int n = t_i / (H * W); // batch index.
    const int pix_idx = t_i % (H * W);

    // Reverse ordering of X and Y axes.
    const int yi = H - 1 - pix_idx / W;
    const int xi = W - 1 - pix_idx % W;

    const float xf = PixToNdc(xi, W);
    const float yf = PixToNdc(yi, H);
    const float2 pxy = make_float2(xf, yf);

    // Loop over all the faces for this pixel.
    for (int k = 0; k < K; k++) {
      // Index into (N, H, W, K, :) grad tensors
      // pixel index + top k index
      int i = n * H * W * K + pix_idx * K + k;

      const int f = pix_to_face[i];
      if (f < 0) {
        continue; // padded face.
      }
      // Get xyz coordinates of the three face vertices.
      const auto v012 = GetSingleFaceVerts(face_verts, f);
      const float3 v0 = thrust::get<0>(v012);
      const float3 v1 = thrust::get<1>(v012);
      const float3 v2 = thrust::get<2>(v012);

      // Only neex xy for barycentric coordinate and distance calculations.
      const float2 v0xy = make_float2(v0.x, v0.y);
      const float2 v1xy = make_float2(v1.x, v1.y);
      const float2 v2xy = make_float2(v2.x, v2.y);

      // Get upstream gradients for the face.
      const float grad_dist_upstream = grad_dists[i];
      const float grad_zbuf_upstream = grad_zbuf[i];
      const float grad_bary_upstream_w0 = grad_bary[i * 3 + 0];
      const float grad_bary_upstream_w1 = grad_bary[i * 3 + 1];
      const float grad_bary_upstream_w2 = grad_bary[i * 3 + 2];
      const float3 grad_bary_upstream = make_float3(
          grad_bary_upstream_w0, grad_bary_upstream_w1, grad_bary_upstream_w2);

      const float3 bary0 = BarycentricCoordsForward(pxy, v0xy, v1xy, v2xy);
      const float3 bary = !perspective_correct
          ? bary0
          : BarycentricPerspectiveCorrectionForward(bary0, v0.z, v1.z, v2.z);
      const bool inside = bary.x > 0.0f && bary.y > 0.0f && bary.z > 0.0f;
      const float sign = inside ? -1.0f : 1.0f;

      // TODO(T52813608) Add support for non-square images.
      auto grad_dist_f = PointTriangleDistanceBackward(
          pxy, v0xy, v1xy, v2xy, sign * grad_dist_upstream);
      const float2 ddist_d_v0 = thrust::get<1>(grad_dist_f);
      const float2 ddist_d_v1 = thrust::get<2>(grad_dist_f);
      const float2 ddist_d_v2 = thrust::get<3>(grad_dist_f);

      // Upstream gradient for barycentric coords from zbuf calculation:
      // zbuf = bary_w0 * z0 + bary_w1 * z1 + bary_w2 * z2
      // Therefore
      // d_zbuf/d_bary_w0 = z0
      // d_zbuf/d_bary_w1 = z1
      // d_zbuf/d_bary_w2 = z2
      const float3 d_zbuf_d_bary = make_float3(v0.z, v1.z, v2.z);

      // Total upstream barycentric gradients are the sum of
      // external upstream gradients and contribution from zbuf.
      const float3 grad_bary_f_sum =
          (grad_bary_upstream + grad_zbuf_upstream * d_zbuf_d_bary);
      float3 grad_bary0 = grad_bary_f_sum;
      float dz0_persp = 0.0f, dz1_persp = 0.0f, dz2_persp = 0.0f;
      if (perspective_correct) {
        auto perspective_grads = BarycentricPerspectiveCorrectionBackward(
            bary0, v0.z, v1.z, v2.z, grad_bary_f_sum);
        grad_bary0 = thrust::get<0>(perspective_grads);
        dz0_persp = thrust::get<1>(perspective_grads);
        dz1_persp = thrust::get<2>(perspective_grads);
        dz2_persp = thrust::get<3>(perspective_grads);
      }
      auto grad_bary_f =
          BarycentricCoordsBackward(pxy, v0xy, v1xy, v2xy, grad_bary0);
      const float2 dbary_d_v0 = thrust::get<1>(grad_bary_f);
      const float2 dbary_d_v1 = thrust::get<2>(grad_bary_f);
      const float2 dbary_d_v2 = thrust::get<3>(grad_bary_f);

      atomicAdd(grad_face_verts + f * 9 + 0, dbary_d_v0.x + ddist_d_v0.x);
      atomicAdd(grad_face_verts + f * 9 + 1, dbary_d_v0.y + ddist_d_v0.y);
      atomicAdd(
          grad_face_verts + f * 9 + 2, grad_zbuf_upstream * bary.x + dz0_persp);
      atomicAdd(grad_face_verts + f * 9 + 3, dbary_d_v1.x + ddist_d_v1.x);
      atomicAdd(grad_face_verts + f * 9 + 4, dbary_d_v1.y + ddist_d_v1.y);
      atomicAdd(
          grad_face_verts + f * 9 + 5, grad_zbuf_upstream * bary.y + dz1_persp);
      atomicAdd(grad_face_verts + f * 9 + 6, dbary_d_v2.x + ddist_d_v2.x);
      atomicAdd(grad_face_verts + f * 9 + 7, dbary_d_v2.y + ddist_d_v2.y);
      atomicAdd(
          grad_face_verts + f * 9 + 8, grad_zbuf_upstream * bary.z + dz2_persp);
    }
  }
}

at::Tensor RasterizeMeshesBackwardCuda(
    const at::Tensor& face_verts, // (F, 3, 3)
    const at::Tensor& pix_to_face, // (N, H, W, K)
    const at::Tensor& grad_zbuf, // (N, H, W, K)
    const at::Tensor& grad_bary, // (N, H, W, K, 3)
    const at::Tensor& grad_dists, // (N, H, W, K)
    const bool perspective_correct) {
  // Check inputs are on the same device
  at::TensorArg face_verts_t{face_verts, "face_verts", 1},
      pix_to_face_t{pix_to_face, "pix_to_face", 2},
      grad_zbuf_t{grad_zbuf, "grad_zbuf", 3},
      grad_bary_t{grad_bary, "grad_bary", 4},
      grad_dists_t{grad_dists, "grad_dists", 5};
  at::CheckedFrom c = "RasterizeMeshesBackwardCuda";
  at::checkAllSameGPU(
      c, {face_verts_t, pix_to_face_t, grad_zbuf_t, grad_bary_t, grad_dists_t});
  at::checkAllSameType(
      c, {face_verts_t, grad_zbuf_t, grad_bary_t, grad_dists_t});

  // Set the device for the kernel launch based on the device of the input
  at::cuda::CUDAGuard device_guard(face_verts.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int F = face_verts.size(0);
  const int N = pix_to_face.size(0);
  const int H = pix_to_face.size(1);
  const int W = pix_to_face.size(2);
  const int K = pix_to_face.size(3);

  at::Tensor grad_face_verts = at::zeros({F, 3, 3}, face_verts.options());

  if (grad_face_verts.numel() == 0) {
    AT_CUDA_CHECK(cudaGetLastError());
    return grad_face_verts;
  }

  const size_t blocks = 1024;
  const size_t threads = 64;

  RasterizeMeshesBackwardCudaKernel<<<blocks, threads, 0, stream>>>(
      face_verts.contiguous().data_ptr<float>(),
      pix_to_face.contiguous().data_ptr<int64_t>(),
      perspective_correct,
      N,
      H,
      W,
      K,
      grad_zbuf.contiguous().data_ptr<float>(),
      grad_bary.contiguous().data_ptr<float>(),
      grad_dists.contiguous().data_ptr<float>(),
      grad_face_verts.data_ptr<float>());

  AT_CUDA_CHECK(cudaGetLastError());
  return grad_face_verts;
}

// ****************************************************************************
// *                          COARSE RASTERIZATION                            *
// ****************************************************************************

__global__ void RasterizeMeshesCoarseCudaKernel(
    const float* face_verts,
    const int64_t* mesh_to_face_first_idx,
    const int64_t* num_faces_per_mesh,
    const float blur_radius,
    const int N,
    const int F,
    const int H,
    const int W,
    const int bin_size,
    const int chunk_size,
    const int max_faces_per_bin,
    int* faces_per_bin,
    int* bin_faces) {
  extern __shared__ char sbuf[];
  const int M = max_faces_per_bin;
  const int num_bins = 1 + (W - 1) / bin_size; // Integer divide round up
  const float half_pix = 1.0f / W; // Size of half a pixel in NDC units
  // This is a boolean array of shape (num_bins, num_bins, chunk_size)
  // stored in shared memory that will track whether each point in the chunk
  // falls into each bin of the image.
  BitMask binmask((unsigned int*)sbuf, num_bins, num_bins, chunk_size);

  // Have each block handle a chunk of faces
  const int chunks_per_batch = 1 + (F - 1) / chunk_size;
  const int num_chunks = N * chunks_per_batch;

  for (int chunk = blockIdx.x; chunk < num_chunks; chunk += gridDim.x) {
    const int batch_idx = chunk / chunks_per_batch; // batch index
    const int chunk_idx = chunk % chunks_per_batch;
    const int face_start_idx = chunk_idx * chunk_size;

    binmask.block_clear();
    const int64_t mesh_face_start_idx = mesh_to_face_first_idx[batch_idx];
    const int64_t mesh_face_stop_idx =
        mesh_face_start_idx + num_faces_per_mesh[batch_idx];

    // Have each thread handle a different face within the chunk
    for (int f = threadIdx.x; f < chunk_size; f += blockDim.x) {
      const int f_idx = face_start_idx + f;

      // Check if face index corresponds to the mesh in the batch given by
      // batch_idx
      if (f_idx >= mesh_face_stop_idx || f_idx < mesh_face_start_idx) {
        continue;
      }

      // Get xyz coordinates of the three face vertices.
      const auto v012 = GetSingleFaceVerts(face_verts, f_idx);
      const float3 v0 = thrust::get<0>(v012);
      const float3 v1 = thrust::get<1>(v012);
      const float3 v2 = thrust::get<2>(v012);

      // Compute screen-space bbox for the triangle expanded by blur.
      float xmin = FloatMin3(v0.x, v1.x, v2.x) - sqrt(blur_radius);
      float ymin = FloatMin3(v0.y, v1.y, v2.y) - sqrt(blur_radius);
      float xmax = FloatMax3(v0.x, v1.x, v2.x) + sqrt(blur_radius);
      float ymax = FloatMax3(v0.y, v1.y, v2.y) + sqrt(blur_radius);
      float zmax = FloatMax3(v0.z, v1.z, v2.z);

      if (zmax < 0) {
        continue; // Face is behind the camera.
      }

      // Brute-force search over all bins; TODO(T54294966) something smarter.
      for (int by = 0; by < num_bins; ++by) {
        // Y coordinate of the top and bottom of the bin.
        // PixToNdc gives the location of the center of each pixel, so we
        // need to add/subtract a half pixel to get the true extent of the bin.
        // Reverse ordering of Y axis so that +Y is upwards in the image.
        const float bin_y_min = PixToNdc(by * bin_size, H) - half_pix;
        const float bin_y_max = PixToNdc((by + 1) * bin_size - 1, H) + half_pix;
        const bool y_overlap = (ymin <= bin_y_max) && (bin_y_min < ymax);

        for (int bx = 0; bx < num_bins; ++bx) {
          // X coordinate of the left and right of the bin.
          // Reverse ordering of x axis so that +X is left.
          const float bin_x_max =
              PixToNdc((bx + 1) * bin_size - 1, W) + half_pix;
          const float bin_x_min = PixToNdc(bx * bin_size, W) - half_pix;

          const bool x_overlap = (xmin <= bin_x_max) && (bin_x_min < xmax);
          if (y_overlap && x_overlap) {
            binmask.set(by, bx, f);
          }
        }
      }
    }
    __syncthreads();
    // Now we have processed every face in the current chunk. We need to
    // count the number of faces in each bin so we can write the indices
    // out to global memory. We have each thread handle a different bin.
    for (int byx = threadIdx.x; byx < num_bins * num_bins; byx += blockDim.x) {
      const int by = byx / num_bins;
      const int bx = byx % num_bins;
      const int count = binmask.count(by, bx);
      const int faces_per_bin_idx =
          batch_idx * num_bins * num_bins + by * num_bins + bx;

      // This atomically increments the (global) number of faces found
      // in the current bin, and gets the previous value of the counter;
      // this effectively allocates space in the bin_faces array for the
      // faces in the current chunk that fall into this bin.
      const int start = atomicAdd(faces_per_bin + faces_per_bin_idx, count);

      // Now loop over the binmask and write the active bits for this bin
      // out to bin_faces.
      int next_idx = batch_idx * num_bins * num_bins * M + by * num_bins * M +
          bx * M + start;
      for (int f = 0; f < chunk_size; ++f) {
        if (binmask.get(by, bx, f)) {
          // TODO(T54296346) find the correct method for handling errors in
          // CUDA. Throw an error if num_faces_per_bin > max_faces_per_bin.
          // Either decrease bin size or increase max_faces_per_bin
          bin_faces[next_idx] = face_start_idx + f;
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
    const int image_size,
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

  const int W = image_size;
  const int H = image_size;
  const int F = face_verts.size(0);
  const int N = num_faces_per_mesh.size(0);
  const int num_bins = 1 + (image_size - 1) / bin_size; // Divide round up.
  const int M = max_faces_per_bin;

  if (num_bins >= kMaxFacesPerBin) {
    std::stringstream ss;
    ss << "Got " << num_bins << "; that's too many!";
    AT_ERROR(ss.str());
  }
  auto opts = num_faces_per_mesh.options().dtype(at::kInt);
  at::Tensor faces_per_bin = at::zeros({N, num_bins, num_bins}, opts);
  at::Tensor bin_faces = at::full({N, num_bins, num_bins, M}, -1, opts);

  if (bin_faces.numel() == 0) {
    AT_CUDA_CHECK(cudaGetLastError());
    return bin_faces;
  }

  const int chunk_size = 512;
  const size_t shared_size = num_bins * num_bins * chunk_size / 8;
  const size_t blocks = 64;
  const size_t threads = 512;

  RasterizeMeshesCoarseCudaKernel<<<blocks, threads, shared_size, stream>>>(
      face_verts.contiguous().data_ptr<float>(),
      mesh_to_face_first_idx.contiguous().data_ptr<int64_t>(),
      num_faces_per_mesh.contiguous().data_ptr<int64_t>(),
      blur_radius,
      N,
      F,
      H,
      W,
      bin_size,
      chunk_size,
      M,
      faces_per_bin.data_ptr<int32_t>(),
      bin_faces.data_ptr<int32_t>());

  AT_CUDA_CHECK(cudaGetLastError());
  return bin_faces;
}

// ****************************************************************************
// *                            FINE RASTERIZATION                            *
// ****************************************************************************
__global__ void RasterizeMeshesFineCudaKernel(
    const float* face_verts, // (F, 3, 3)
    const int32_t* bin_faces, // (N, B, B, T)
    const float blur_radius,
    const int bin_size,
    const bool perspective_correct,
    const bool cull_backfaces,
    const int N,
    const int B,
    const int M,
    const int H,
    const int W,
    const int K,
    int64_t* face_idxs, // (N, S, S, K)
    float* zbuf, // (N, S, S, K)
    float* pix_dists, // (N, S, S, K)
    float* bary // (N, S, S, K, 3)
) {
  // This can be more than S^2 if S % bin_size != 0
  int num_pixels = N * B * B * bin_size * bin_size;
  int num_threads = gridDim.x * blockDim.x;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  for (int pid = tid; pid < num_pixels; pid += num_threads) {
    // Convert linear index into bin and pixel indices. We make the within
    // block pixel ids move the fastest, so that adjacent threads will fall
    // into the same bin; this should give them coalesced memory reads when
    // they read from faces and bin_faces.
    int i = pid;
    const int n = i / (B * B * bin_size * bin_size);
    i %= B * B * bin_size * bin_size;
    const int by = i / (B * bin_size * bin_size);
    i %= B * bin_size * bin_size;
    const int bx = i / (bin_size * bin_size);
    i %= bin_size * bin_size;
    const int yi = i / bin_size + by * bin_size;
    const int xi = i % bin_size + bx * bin_size;

    if (yi >= H || xi >= W)
      continue;

    const float xf = PixToNdc(xi, W);
    const float yf = PixToNdc(yi, H);
    const float2 pxy = make_float2(xf, yf);

    // This part looks like the naive rasterization kernel, except we use
    // bin_faces to only look at a subset of faces already known to fall
    // in this bin. TODO abstract out this logic into some data structure
    // that is shared by both kernels?
    Pixel q[kMaxPointsPerPixel];
    int q_size = 0;
    float q_max_z = -1000;
    int q_max_idx = -1;
    for (int m = 0; m < M; m++) {
      const int f = bin_faces[n * B * B * M + by * B * M + bx * M + m];
      if (f < 0) {
        continue; // bin_faces uses -1 as a sentinal value.
      }
      // Check if the pixel pxy is inside the face bounding box and if it is,
      // update q, q_size, q_max_z and q_max_idx in place.
      CheckPixelInsideFace(
          face_verts,
          f,
          q_size,
          q_max_z,
          q_max_idx,
          q,
          blur_radius,
          pxy,
          K,
          perspective_correct,
          cull_backfaces);
    }

    // Now we've looked at all the faces for this bin, so we can write
    // output for the current pixel.
    // TODO: make sorting an option as only top k is needed, not sorted values.
    BubbleSort(q, q_size);

    // Reverse ordering of the X and Y axis so that
    // in the image +Y is pointing up and +X is pointing left.
    const int yidx = H - 1 - yi;
    const int xidx = W - 1 - xi;
    const int pix_idx = n * H * W * K + yidx * H * K + xidx * K;
    for (int k = 0; k < q_size; k++) {
      face_idxs[pix_idx + k] = q[k].idx;
      zbuf[pix_idx + k] = q[k].z;
      pix_dists[pix_idx + k] = q[k].dist;
      bary[(pix_idx + k) * 3 + 0] = q[k].bary.x;
      bary[(pix_idx + k) * 3 + 1] = q[k].bary.y;
      bary[(pix_idx + k) * 3 + 2] = q[k].bary.z;
    }
  }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
RasterizeMeshesFineCuda(
    const at::Tensor& face_verts,
    const at::Tensor& bin_faces,
    const int image_size,
    const float blur_radius,
    const int bin_size,
    const int faces_per_pixel,
    const bool perspective_correct,
    const bool cull_backfaces) {
  TORCH_CHECK(
      face_verts.ndimension() == 3 && face_verts.size(1) == 3 &&
          face_verts.size(2) == 3,
      "face_verts must have dimensions (num_faces, 3, 3)");
  TORCH_CHECK(bin_faces.ndimension() == 4, "bin_faces must have 4 dimensions");

  // Check inputs are on the same device
  at::TensorArg face_verts_t{face_verts, "face_verts", 1},
      bin_faces_t{bin_faces, "bin_faces", 2};
  at::CheckedFrom c = "RasterizeMeshesFineCuda";
  at::checkAllSameGPU(c, {face_verts_t, bin_faces_t});

  // Set the device for the kernel launch based on the device of the input
  at::cuda::CUDAGuard device_guard(face_verts.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int N = bin_faces.size(0);
  const int B = bin_faces.size(1);
  const int M = bin_faces.size(3);
  const int K = faces_per_pixel;
  const int H = image_size; // Assume square images only.
  const int W = image_size;

  if (K > kMaxPointsPerPixel) {
    AT_ERROR("Must have num_closest <= 150");
  }
  auto long_opts = bin_faces.options().dtype(at::kLong);
  auto float_opts = face_verts.options().dtype(at::kFloat);

  at::Tensor face_idxs = at::full({N, H, W, K}, -1, long_opts);
  at::Tensor zbuf = at::full({N, H, W, K}, -1, float_opts);
  at::Tensor pix_dists = at::full({N, H, W, K}, -1, float_opts);
  at::Tensor bary = at::full({N, H, W, K, 3}, -1, float_opts);

  if (face_idxs.numel() == 0) {
    AT_CUDA_CHECK(cudaGetLastError());
    return std::make_tuple(face_idxs, zbuf, bary, pix_dists);
  }

  const size_t blocks = 1024;
  const size_t threads = 64;

  RasterizeMeshesFineCudaKernel<<<blocks, threads, 0, stream>>>(
      face_verts.contiguous().data_ptr<float>(),
      bin_faces.contiguous().data_ptr<int32_t>(),
      blur_radius,
      bin_size,
      perspective_correct,
      cull_backfaces,
      N,
      B,
      M,
      H,
      W,
      K,
      face_idxs.data_ptr<int64_t>(),
      zbuf.data_ptr<float>(),
      pix_dists.data_ptr<float>(),
      bary.data_ptr<float>());

  return std::make_tuple(face_idxs, zbuf, bary, pix_dists);
}
