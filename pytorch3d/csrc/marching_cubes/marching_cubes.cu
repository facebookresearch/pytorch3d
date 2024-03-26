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
#include <cstdio>
#include "marching_cubes/tables.h"

/*
Parallelized marching cubes for pytorch extension
referenced and adapted from CUDA-Samples:
(https://github.com/NVIDIA/cuda-samples/tree/master/Samples/5_Domain_Specific/marchingCubes)
We divide the algorithm into two forward-passes:
(1) The first forward-pass executes "ClassifyVoxelKernel" to
evaluate volume scalar field for each cube and pre-compute
two arrays -- number of vertices per cube (d_voxelVerts) and
occupied or not per cube (d_voxelOccupied).

Some prepration steps:
With d_voxelOccupied, an exclusive scan is performed to compute
the number of activeVoxels, which can be used to accelerate
computation. With d_voxelVerts, another exclusive scan
is performed to compute the accumulated sum of vertices in the 3d
grid and totalVerts.

(2) The second forward-pass calls "GenerateFacesKernel" to
generate interpolated vertex positions and face indices by "marching
through" each cube in the grid.

*/

// EPS: Used to indicate if two float values are close
__constant__ const float EPSILON = 1e-5;

// Linearly interpolate the position where an isosurface cuts an edge
// between two vertices, based on their scalar values
//
// Args:
//    isolevel: float value used as threshold
//    p1: position of point1
//    p2: position of point2
//    valp1: field value for p1
//    valp2: field value for p2
//
// Returns:
//    point: interpolated verte
//
__device__ float3
vertexInterp(float isolevel, float3 p1, float3 p2, float valp1, float valp2) {
  float ratio;
  float3 p;

  if (abs(isolevel - valp1) < EPSILON) {
    return p1;
  } else if (abs(isolevel - valp2) < EPSILON) {
    return p2;
  } else if (abs(valp1 - valp2) < EPSILON) {
    return p1;
  }

  ratio = (isolevel - valp1) / (valp2 - valp1);

  p.x = p1.x * (1 - ratio) + p2.x * ratio;
  p.y = p1.y * (1 - ratio) + p2.y * ratio;
  p.z = p1.z * (1 - ratio) + p2.z * ratio;

  return p;
}

// Determine if the triangle is degenerate
// A triangle is degenerate when at least two of the vertices
// share the same position.
//
// Args:
//    p1: position of vertex p1
//    p2: position of vertex p2
//    p3: position of vertex p3
//
// Returns:
//    boolean indicator if the triangle is degenerate
__device__ bool isDegenerate(float3 p1, float3 p2, float3 p3) {
  if ((abs(p1.x - p2.x) < EPSILON && abs(p1.y - p2.y) < EPSILON &&
       abs(p1.z - p2.z) < EPSILON) ||
      (abs(p2.x - p3.x) < EPSILON && abs(p2.y - p3.y) < EPSILON &&
       abs(p2.z - p3.z) < EPSILON) ||
      (abs(p3.x - p1.x) < EPSILON && abs(p3.y - p1.y) < EPSILON &&
       abs(p3.z - p1.z) < EPSILON)) {
    return true;
  } else {
    return false;
  }
}

// Convert from local vertex id to global vertex id, given position
// of the cube where the vertex resides. The function ensures vertices
// shared from adjacent cubes are mapped to the same global id.

// Args:
//     v: local vertex id
//     x: x position of the cube where the vertex belongs
//     y: y position of the cube where the vertex belongs
//     z: z position of the cube where the vertex belongs
//     W: width of x dimension
//     H: height of y dimension

// Returns:
//     global vertex id represented by its x/y/z offsets
__device__ uint localToGlobal(int v, int x, int y, int z, int W, int H) {
  const int dx = v & 1;
  const int dy = v >> 1 & 1;
  const int dz = v >> 2 & 1;
  return (x + dx) + (y + dy) * W + (z + dz) * W * H;
}

// Hash_combine a pair of global vertex id to a single integer.
//
// Args:
//    v1_id: global id of vertex 1
//    v2_id: global id of vertex 2
//    W: width of the 3d grid
//    H: height of the 3d grid
//    Z: depth of the 3d grid
//
// Returns:
//    hashing for a pair of vertex ids
//
__device__ int64_t hashVpair(uint v1_id, uint v2_id, int W, int H, int D) {
  return (int64_t)v1_id * (W + W * H + W * H * D) + (int64_t)v2_id;
}

// precompute number of vertices and occupancy
// for each voxel in the grid.
//
// Args:
//    voxelVerts: pointer to device array to store number
//          of verts per voxel
//    voxelOccupied: pointer to device array to store
//          occupancy state per voxel
//    vol: torch tensor stored with 3D scalar field
//    isolevel: threshold to determine isosurface intersection
//
__global__ void ClassifyVoxelKernel(
    at::PackedTensorAccessor32<int, 1, at::RestrictPtrTraits> voxelVerts,
    at::PackedTensorAccessor32<int, 1, at::RestrictPtrTraits> voxelOccupied,
    const at::PackedTensorAccessor32<float, 3, at::RestrictPtrTraits> vol,
    // const at::PackedTensorAccessor<int, 1, at::RestrictPtrTraits>
    // numVertsTable,
    float isolevel) {
  const int indexTable[8]{0, 1, 4, 5, 3, 2, 7, 6};
  const uint D = vol.size(0) - 1;
  const uint H = vol.size(1) - 1;
  const uint W = vol.size(2) - 1;

  // 1-d grid
  uint id = blockIdx.x * blockDim.x + threadIdx.x;
  uint num_threads = gridDim.x * blockDim.x;

  // Table mapping from cubeindex to number of vertices in the configuration
  const unsigned char numVertsTable[256] = {
      0,  3,  3,  6,  3,  6,  6,  9,  3,  6,  6,  9,  6,  9,  9,  6,  3,  6,
      6,  9,  6,  9,  9,  12, 6,  9,  9,  12, 9,  12, 12, 9,  3,  6,  6,  9,
      6,  9,  9,  12, 6,  9,  9,  12, 9,  12, 12, 9,  6,  9,  9,  6,  9,  12,
      12, 9,  9,  12, 12, 9,  12, 15, 15, 6,  3,  6,  6,  9,  6,  9,  9,  12,
      6,  9,  9,  12, 9,  12, 12, 9,  6,  9,  9,  12, 9,  12, 12, 15, 9,  12,
      12, 15, 12, 15, 15, 12, 6,  9,  9,  12, 9,  12, 6,  9,  9,  12, 12, 15,
      12, 15, 9,  6,  9,  12, 12, 9,  12, 15, 9,  6,  12, 15, 15, 12, 15, 6,
      12, 3,  3,  6,  6,  9,  6,  9,  9,  12, 6,  9,  9,  12, 9,  12, 12, 9,
      6,  9,  9,  12, 9,  12, 12, 15, 9,  6,  12, 9,  12, 9,  15, 6,  6,  9,
      9,  12, 9,  12, 12, 15, 9,  12, 12, 15, 12, 15, 15, 12, 9,  12, 12, 9,
      12, 15, 15, 12, 12, 9,  15, 6,  15, 12, 6,  3,  6,  9,  9,  12, 9,  12,
      12, 15, 9,  12, 12, 15, 6,  9,  9,  6,  9,  12, 12, 15, 12, 15, 15, 6,
      12, 9,  15, 12, 9,  6,  12, 3,  9,  12, 12, 15, 12, 15, 9,  12, 12, 15,
      15, 6,  9,  12, 6,  3,  6,  9,  9,  6,  9,  12, 6,  3,  9,  6,  12, 3,
      6,  3,  3,  0,
  };

  for (uint tid = id; tid < D * H * W; tid += num_threads) {
    // compute global location of the voxel
    const int gx = tid % W;
    const int gy = tid / W % H;
    const int gz = tid / (W * H);

    int cubeindex = 0;
    for (int i = 0; i < 8; i++) {
      const int dx = i & 1;
      const int dy = i >> 1 & 1;
      const int dz = i >> 2 & 1;

      const int x = gx + dx;
      const int y = gy + dy;
      const int z = gz + dz;

      if (vol[z][y][x] < isolevel) {
        cubeindex |= 1 << indexTable[i];
      }
    }
    // collect number of vertices for each voxel
    unsigned char numVerts = numVertsTable[cubeindex];
    voxelVerts[tid] = numVerts;
    voxelOccupied[tid] = (numVerts > 0);
  }
}

// extract compact voxel array for acceleration
//
// Args:
//    compactedVoxelArray: tensor of shape (activeVoxels,) which maps
//          from accumulated non-empty voxel index to original 3d grid index
//    voxelOccupied: tensor of shape (numVoxels,) which stores
//          the occupancy state per voxel
//    voxelOccupiedScan: tensor of shape (numVoxels,) which
//          stores the accumulated occupied voxel counts
//    numVoxels: number of total voxels in the grid
//
__global__ void CompactVoxelsKernel(
    at::PackedTensorAccessor32<int, 1, at::RestrictPtrTraits>
        compactedVoxelArray,
    const at::PackedTensorAccessor32<int, 1, at::RestrictPtrTraits>
        voxelOccupied,
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        voxelOccupiedScan,
    uint numVoxels) {
  uint id = blockIdx.x * blockDim.x + threadIdx.x;
  uint num_threads = gridDim.x * blockDim.x;
  for (uint tid = id; tid < numVoxels; tid += num_threads) {
    if (voxelOccupied[tid]) {
      compactedVoxelArray[voxelOccupiedScan[tid]] = tid;
    }
  }
}

// generate triangles for each voxel using marching cubes
//
// Args:
//    verts: torch tensor of shape (V, 3) to store interpolated mesh vertices
//    faces: torch tensor of shape (F, 3) to store indices for mesh faces
//    ids: torch tensor of shape (V) to store id of each vertex
//    compactedVoxelArray: tensor of shape (activeVoxels,) which stores
//          non-empty voxel index.
//    numVertsScanned: tensor of shape (numVoxels,) which stores accumulated
//          vertices count in the voxel
//    activeVoxels: number of active voxels used for acceleration
//    vol: torch tensor stored with 3D scalar field
//    isolevel: threshold to determine isosurface intersection
//
__global__ void GenerateFacesKernel(
    at::PackedTensorAccessor32<float, 2, at::RestrictPtrTraits> verts,
    at::PackedTensorAccessor<int64_t, 2, at::RestrictPtrTraits> faces,
    at::PackedTensorAccessor<int64_t, 1, at::RestrictPtrTraits> ids,
    at::PackedTensorAccessor32<int, 1, at::RestrictPtrTraits>
        compactedVoxelArray,
    at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        numVertsScanned,
    const uint activeVoxels,
    const at::PackedTensorAccessor32<float, 3, at::RestrictPtrTraits> vol,
    const at::PackedTensorAccessor32<int, 2, at::RestrictPtrTraits> faceTable,
    // const at::PackedTensorAccessor32<int, 1, at::RestrictPtrTraits>
    // numVertsTable,
    const float isolevel) {
  uint id = blockIdx.x * blockDim.x + threadIdx.x;
  uint num_threads = gridDim.x * blockDim.x;
  const int faces_size = faces.size(0);
  // Table mapping each edge to the corresponding cube vertices offsets
  const int edgeToVertsTable[12][2] = {
      {0, 1},
      {1, 5},
      {4, 5},
      {0, 4},
      {2, 3},
      {3, 7},
      {6, 7},
      {2, 6},
      {0, 2},
      {1, 3},
      {5, 7},
      {4, 6},
  };

  // Table mapping from cubeindex to number of vertices in the configuration
  const unsigned char numVertsTable[256] = {
      0,  3,  3,  6,  3,  6,  6,  9,  3,  6,  6,  9,  6,  9,  9,  6,  3,  6,
      6,  9,  6,  9,  9,  12, 6,  9,  9,  12, 9,  12, 12, 9,  3,  6,  6,  9,
      6,  9,  9,  12, 6,  9,  9,  12, 9,  12, 12, 9,  6,  9,  9,  6,  9,  12,
      12, 9,  9,  12, 12, 9,  12, 15, 15, 6,  3,  6,  6,  9,  6,  9,  9,  12,
      6,  9,  9,  12, 9,  12, 12, 9,  6,  9,  9,  12, 9,  12, 12, 15, 9,  12,
      12, 15, 12, 15, 15, 12, 6,  9,  9,  12, 9,  12, 6,  9,  9,  12, 12, 15,
      12, 15, 9,  6,  9,  12, 12, 9,  12, 15, 9,  6,  12, 15, 15, 12, 15, 6,
      12, 3,  3,  6,  6,  9,  6,  9,  9,  12, 6,  9,  9,  12, 9,  12, 12, 9,
      6,  9,  9,  12, 9,  12, 12, 15, 9,  6,  12, 9,  12, 9,  15, 6,  6,  9,
      9,  12, 9,  12, 12, 15, 9,  12, 12, 15, 12, 15, 15, 12, 9,  12, 12, 9,
      12, 15, 15, 12, 12, 9,  15, 6,  15, 12, 6,  3,  6,  9,  9,  12, 9,  12,
      12, 15, 9,  12, 12, 15, 6,  9,  9,  6,  9,  12, 12, 15, 12, 15, 15, 6,
      12, 9,  15, 12, 9,  6,  12, 3,  9,  12, 12, 15, 12, 15, 9,  12, 12, 15,
      15, 6,  9,  12, 6,  3,  6,  9,  9,  6,  9,  12, 6,  3,  9,  6,  12, 3,
      6,  3,  3,  0,
  };

  for (uint tid = id; tid < activeVoxels; tid += num_threads) {
    uint voxel = compactedVoxelArray[tid]; // maps from accumulated id to
                                           // original 3d voxel id
    // mapping from offsets to vi index
    int indexTable[8]{0, 1, 4, 5, 3, 2, 7, 6};
    // field value for each vertex
    float val[8];
    // position for each vertex
    float3 p[8];
    // 3d address
    const uint D = vol.size(0) - 1;
    const uint H = vol.size(1) - 1;
    const uint W = vol.size(2) - 1;

    const int gx = voxel % W;
    const int gy = voxel / W % H;
    const int gz = voxel / (W * H);

    // recalculate cubeindex;
    uint cubeindex = 0;
    for (int i = 0; i < 8; i++) {
      const int dx = i & 1;
      const int dy = i >> 1 & 1;
      const int dz = i >> 2 & 1;

      const int x = gx + dx;
      const int y = gy + dy;
      const int z = gz + dz;

      if (vol[z][y][x] < isolevel) {
        cubeindex |= 1 << indexTable[i];
      }
      val[indexTable[i]] = vol[z][y][x]; // maps from vi to volume
      p[indexTable[i]] = make_float3(x, y, z); // maps from vi to position
    }

    // Interpolate vertices where the surface intersects the cube
    float3 vertlist[12];
    vertlist[0] = vertexInterp(isolevel, p[0], p[1], val[0], val[1]);
    vertlist[1] = vertexInterp(isolevel, p[1], p[2], val[1], val[2]);
    vertlist[2] = vertexInterp(isolevel, p[3], p[2], val[3], val[2]);
    vertlist[3] = vertexInterp(isolevel, p[0], p[3], val[0], val[3]);

    vertlist[4] = vertexInterp(isolevel, p[4], p[5], val[4], val[5]);
    vertlist[5] = vertexInterp(isolevel, p[5], p[6], val[5], val[6]);
    vertlist[6] = vertexInterp(isolevel, p[7], p[6], val[7], val[6]);
    vertlist[7] = vertexInterp(isolevel, p[4], p[7], val[4], val[7]);

    vertlist[8] = vertexInterp(isolevel, p[0], p[4], val[0], val[4]);
    vertlist[9] = vertexInterp(isolevel, p[1], p[5], val[1], val[5]);
    vertlist[10] = vertexInterp(isolevel, p[2], p[6], val[2], val[6]);
    vertlist[11] = vertexInterp(isolevel, p[3], p[7], val[3], val[7]);

    // output triangle faces
    uint numVerts = numVertsTable[cubeindex];

    for (int i = 0; i < numVerts; i++) {
      int index = numVertsScanned[voxel] + i;
      unsigned char edge = faceTable[cubeindex][i];

      uint v1 = edgeToVertsTable[edge][0];
      uint v2 = edgeToVertsTable[edge][1];
      uint v1_id = localToGlobal(v1, gx, gy, gz, W + 1, H + 1);
      uint v2_id = localToGlobal(v2, gx, gy, gz, W + 1, H + 1);
      int64_t edge_id = hashVpair(v1_id, v2_id, W + 1, H + 1, D + 1);

      verts[index][0] = vertlist[edge].x;
      verts[index][1] = vertlist[edge].y;
      verts[index][2] = vertlist[edge].z;

      if (index < faces_size) {
        faces[index][0] = index * 3 + 0;
        faces[index][1] = index * 3 + 1;
        faces[index][2] = index * 3 + 2;
      }

      ids[index] = edge_id;
    }
  } // end for grid-strided kernel
}

// ATen/Torch does not have an exclusive-scan operator. Additionally, in the
// code below we need to get the "total number of items to work on" after
// a scan, which with an inclusive-scan would simply be the value of the last
// element in the tensor.
//
// This utility function hits two birds with one stone, by running
// an inclusive-scan into a right-shifted view of a tensor that's
// allocated to be one element bigger than the input tensor.
//
// Note; return tensor is `int64_t` per element, even if the input
// tensor is only 32-bit. Also, the return tensor is one element bigger
// than the input one.
//
// Secondary optional argument is an output argument that gets the
// value of the last element of the return tensor (because you almost
// always need this CPU-side right after this function anyway).
static at::Tensor ExclusiveScanAndTotal(
    const at::Tensor& inTensor,
    int64_t* optTotal = nullptr) {
  const auto inSize = inTensor.sizes()[0];
  auto retTensor = at::zeros({inSize + 1}, at::kLong).to(inTensor.device());

  using at::indexing::None;
  using at::indexing::Slice;
  auto rightShiftedView = retTensor.index({Slice(1, None)});

  // Do an (inclusive-scan) cumulative sum in to the view that's
  // shifted one element to the right...
  at::cumsum_out(rightShiftedView, inTensor, 0, at::kLong);

  if (optTotal) {
    *optTotal = retTensor[inSize].cpu().item<int64_t>();
  }

  // ...so that the not-shifted tensor holds the exclusive-scan
  return retTensor;
}

// Entrance for marching cubes cuda extension. Marching Cubes is an algorithm to
// create triangle meshes from an implicit function (one of the form f(x, y, z)
// = 0). It works by iteratively checking a grid of cubes superimposed over a
// region of the function. The number of faces and positions of the vertices in
// each cube are determined by the the isolevel as well as the volume values
// from the eight vertices of the cube.
//
// We implement this algorithm with two forward passes where the first pass
// checks the occupancy and collects number of vertices for each cube. The
// second pass will skip empty voxels and generate vertices as well as faces for
// each cube through table lookup. The vertex positions, faces and identifiers
// for each vertex will be returned.
//
//
// Args:
//    vol: torch tensor of shape (D, H, W) for volume scalar field
//    isolevel: threshold to determine isosurface intesection
//
// Returns:
//     tuple of <verts, faces, ids>: which stores vertex positions, face
//         indices and integer identifiers for each vertex.
//    verts: (N_verts, 3) FloatTensor for vertex positions
//    faces: (N_faces, 3) LongTensor of face indices
//    ids: (N_verts,) LongTensor used to identify each vertex. Vertices from
//         adjacent edges can share the same 3d position. To reduce memory
//         redudancy, we tag each vertex with a unique id for deduplication. In
//         contrast to deduping on vertices, this has the benefit to avoid
//         floating point precision issues.
//
std::tuple<at::Tensor, at::Tensor, at::Tensor> MarchingCubesCuda(
    const at::Tensor& vol,
    const float isolevel) {
  // Set the device for the kernel launch based on the device of vol
  at::cuda::CUDAGuard device_guard(vol.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // transfer _FACE_TABLE data to device
  at::Tensor face_table_tensor = at::zeros(
      {256, 16}, at::TensorOptions().dtype(at::kInt).device(at::kCPU));
  auto face_table_a = face_table_tensor.accessor<int, 2>();
  for (int i = 0; i < 256; i++) {
    for (int j = 0; j < 16; j++) {
      face_table_a[i][j] = _FACE_TABLE[i][j];
    }
  }
  at::Tensor faceTable = face_table_tensor.to(vol.device());

  // get numVoxels
  int threads = 128;
  const uint D = vol.size(0);
  const uint H = vol.size(1);
  const uint W = vol.size(2);
  const int numVoxels = (D - 1) * (H - 1) * (W - 1);
  dim3 grid((numVoxels + threads - 1) / threads, 1, 1);
  if (grid.x > 65535) {
    grid.x = 65535;
  }

  using at::indexing::None;
  using at::indexing::Slice;

  auto d_voxelVerts =
      at::zeros({numVoxels}, at::TensorOptions().dtype(at::kInt))
          .to(vol.device());
  auto d_voxelOccupied =
      at::zeros({numVoxels}, at::TensorOptions().dtype(at::kInt))
          .to(vol.device());

  // Execute "ClassifyVoxelKernel" kernel to precompute
  // two arrays - d_voxelOccupied and d_voxelVertices to global memory,
  // which stores the occupancy state and number of voxel vertices per voxel.
  ClassifyVoxelKernel<<<grid, threads, 0, stream>>>(
      d_voxelVerts.packed_accessor32<int, 1, at::RestrictPtrTraits>(),
      d_voxelOccupied.packed_accessor32<int, 1, at::RestrictPtrTraits>(),
      vol.packed_accessor32<float, 3, at::RestrictPtrTraits>(),
      isolevel);
  AT_CUDA_CHECK(cudaGetLastError());
  cudaDeviceSynchronize();

  // Scan "d_voxelOccupied" array to generate accumulated voxel occupancy
  // count for voxels in the grid and compute the number of active voxels.
  // If the number of active voxels is 0, return zero tensor for verts and
  // faces.
  int64_t activeVoxels = 0;
  auto d_voxelOccupiedScan =
      ExclusiveScanAndTotal(d_voxelOccupied, &activeVoxels);

  const int device_id = vol.device().index();
  auto opt = at::TensorOptions().dtype(at::kInt).device(at::kCUDA, device_id);
  auto opt_long =
      at::TensorOptions().dtype(at::kLong).device(at::kCUDA, device_id);

  if (activeVoxels == 0) {
    int ntris = 0;
    at::Tensor verts = at::zeros({ntris * 3, 3}, vol.options());
    at::Tensor faces = at::zeros({ntris, 3}, opt_long);
    at::Tensor ids = at::zeros({ntris}, opt_long);
    return std::make_tuple(verts, faces, ids);
  }

  // Execute "CompactVoxelsKernel" kernel to compress voxels for acceleration.
  // This allows us to run triangle generation on only the occupied voxels.
  auto d_compVoxelArray = at::zeros({activeVoxels}, opt);
  CompactVoxelsKernel<<<grid, threads, 0, stream>>>(
      d_compVoxelArray.packed_accessor32<int, 1, at::RestrictPtrTraits>(),
      d_voxelOccupied.packed_accessor32<int, 1, at::RestrictPtrTraits>(),
      d_voxelOccupiedScan
          .packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
      numVoxels);
  AT_CUDA_CHECK(cudaGetLastError());
  cudaDeviceSynchronize();

  // Scan d_voxelVerts array to generate offsets of vertices for each voxel
  int64_t totalVerts = 0;
  auto d_voxelVertsScan = ExclusiveScanAndTotal(d_voxelVerts, &totalVerts);

  // Execute "GenerateFacesKernel" kernel
  // This runs only on the occupied voxels.
  // It looks up the field values and generates the triangle data.
  at::Tensor verts = at::zeros({totalVerts, 3}, vol.options());
  at::Tensor faces = at::zeros({totalVerts / 3, 3}, opt_long);

  at::Tensor ids = at::zeros({totalVerts}, opt_long);

  dim3 grid2((activeVoxels + threads - 1) / threads, 1, 1);
  if (grid2.x > 65535) {
    grid2.x = 65535;
  }

  GenerateFacesKernel<<<grid2, threads, 0, stream>>>(
      verts.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
      faces.packed_accessor<int64_t, 2, at::RestrictPtrTraits>(),
      ids.packed_accessor<int64_t, 1, at::RestrictPtrTraits>(),
      d_compVoxelArray.packed_accessor32<int, 1, at::RestrictPtrTraits>(),
      d_voxelVertsScan.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
      activeVoxels,
      vol.packed_accessor32<float, 3, at::RestrictPtrTraits>(),
      faceTable.packed_accessor32<int, 2, at::RestrictPtrTraits>(),
      isolevel);
  AT_CUDA_CHECK(cudaGetLastError());

  return std::make_tuple(verts, faces, ids);
}
