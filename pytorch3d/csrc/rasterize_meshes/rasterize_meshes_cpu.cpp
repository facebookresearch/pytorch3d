/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <torch/extension.h>
#include <algorithm>
#include <list>
#include <thread>
#include <tuple>
#include "ATen/core/TensorAccessor.h"
#include "rasterize_points/rasterization_utils.h"
#include "utils/geometry_utils.h"
#include "utils/vec2.h"
#include "utils/vec3.h"

// Get (x, y, z) values for vertex from (3, 3) tensor face.
template <typename Face>
auto ExtractVerts(const Face& face, const int vertex_index) {
  return std::make_tuple(
      face[vertex_index][0], face[vertex_index][1], face[vertex_index][2]);
}

// Compute min/max x/y for each face.
auto ComputeFaceBoundingBoxes(const torch::Tensor& face_verts) {
  const int total_F = face_verts.size(0);
  auto float_opts = face_verts.options().dtype(torch::kFloat32);
  auto face_verts_a = face_verts.accessor<float, 3>();
  torch::Tensor face_bboxes = torch::full({total_F, 6}, -2.0, float_opts);

  // Loop through all the faces
  for (int f = 0; f < total_F; ++f) {
    const auto& face = face_verts_a[f];
    float x0, x1, x2, y0, y1, y2, z0, z1, z2;
    std::tie(x0, y0, z0) = ExtractVerts(face, 0);
    std::tie(x1, y1, z1) = ExtractVerts(face, 1);
    std::tie(x2, y2, z2) = ExtractVerts(face, 2);

    const float x_min = std::min(x0, std::min(x1, x2));
    const float y_min = std::min(y0, std::min(y1, y2));
    const float x_max = std::max(x0, std::max(x1, x2));
    const float y_max = std::max(y0, std::max(y1, y2));
    const float z_min = std::min(z0, std::min(z1, z2));
    const float z_max = std::max(z0, std::max(z1, z2));

    face_bboxes[f][0] = x_min;
    face_bboxes[f][1] = y_min;
    face_bboxes[f][2] = x_max;
    face_bboxes[f][3] = y_max;
    face_bboxes[f][4] = z_min;
    face_bboxes[f][5] = z_max;
  }

  return face_bboxes;
}

// Check if the point (px, py) lies inside the face bounding box face_bbox.
// Return true if the point is outside.
template <typename Face>
bool CheckPointOutsideBoundingBox(
    const Face& face_bbox,
    float blur_radius,
    float px,
    float py) {
  // Read triangle bbox coordinates and expand by blur radius.
  float x_min = face_bbox[0] - blur_radius;
  float y_min = face_bbox[1] - blur_radius;
  float x_max = face_bbox[2] + blur_radius;
  float y_max = face_bbox[3] + blur_radius;

  // Faces with at least one vertex behind the camera won't render correctly
  // and should be removed or clipped before calling the rasterizer
  const bool z_invalid = face_bbox[4] < kEpsilon;

  // Check if the current point is within the triangle bounding box.
  return (px > x_max || px < x_min || py > y_max || py < y_min || z_invalid);
}

// Calculate areas of all faces. Returns a tensor of shape (total_faces, 1)
// where faces with zero area have value -1.
auto ComputeFaceAreas(const torch::Tensor& face_verts) {
  const int total_F = face_verts.size(0);
  auto float_opts = face_verts.options().dtype(torch::kFloat32);
  auto face_verts_a = face_verts.accessor<float, 3>();
  torch::Tensor face_areas = torch::full({total_F}, -1, float_opts);

  // Loop through all the faces
  for (int f = 0; f < total_F; ++f) {
    const auto& face = face_verts_a[f];
    float x0, x1, x2, y0, y1, y2, z0, z1, z2;
    std::tie(x0, y0, z0) = ExtractVerts(face, 0);
    std::tie(x1, y1, z1) = ExtractVerts(face, 1);
    std::tie(x2, y2, z2) = ExtractVerts(face, 2);

    const vec2<float> v0(x0, y0);
    const vec2<float> v1(x1, y1);
    const vec2<float> v2(x2, y2);

    const float face_area = EdgeFunctionForward(v0, v1, v2);
    face_areas[f] = face_area;
  }

  return face_areas;
}

// Helper function to use with std::find_if to find the index of any
// values in the top k struct which match a given idx.
struct IsNeighbor {
  IsNeighbor(int neighbor_idx) {
    this->neighbor_idx = neighbor_idx;
  }
  bool operator()(std::tuple<float, int, float, float, float, float> elem) {
    return (std::get<1>(elem) == neighbor_idx);
  }
  int neighbor_idx;
};

namespace {
void RasterizeMeshesNaiveCpu_worker(
    const int start_yi,
    const int end_yi,
    const torch::Tensor& mesh_to_face_first_idx,
    const torch::Tensor& num_faces_per_mesh,
    const float blur_radius,
    const bool perspective_correct,
    const bool clip_barycentric_coords,
    const bool cull_backfaces,
    const int32_t N,
    const int H,
    const int W,
    const int K,
    at::TensorAccessor<float, 3>& face_verts_a,
    at::TensorAccessor<float, 1>& face_areas_a,
    at::TensorAccessor<float, 2>& face_bboxes_a,
    at::TensorAccessor<int64_t, 1>& neighbor_idx_a,
    at::TensorAccessor<float, 4>& zbuf_a,
    at::TensorAccessor<int64_t, 4>& face_idxs_a,
    at::TensorAccessor<float, 4>& pix_dists_a,
    at::TensorAccessor<float, 5>& barycentric_coords_a) {
  for (int n = 0; n < N; ++n) {
    // Loop through each mesh in the batch.
    // Get the start index of the faces in faces_packed and the num faces
    // in the mesh to avoid having to loop through all the faces.
    const int face_start_idx = mesh_to_face_first_idx[n].item().to<int32_t>();
    const int face_stop_idx =
        (face_start_idx + num_faces_per_mesh[n].item().to<int32_t>());

    // Iterate through the horizontal lines of the image from top to bottom.
    for (int yi = start_yi; yi < end_yi; ++yi) {
      // Reverse the order of yi so that +Y is pointing upwards in the image.
      const int yidx = H - 1 - yi;

      // Y coordinate of the top of the pixel.
      const float yf = PixToNonSquareNdc(yidx, H, W);
      // Iterate through pixels on this horizontal line, left to right.
      for (int xi = 0; xi < W; ++xi) {
        // Reverse the order of xi so that +X is pointing to the left in the
        // image.
        const int xidx = W - 1 - xi;

        // X coordinate of the left of the pixel.
        const float xf = PixToNonSquareNdc(xidx, W, H);

        // Use a deque to hold values:
        // (z, idx, r, bary.x, bary.y. bary.z)
        // Sort the deque as needed to mimic a priority queue.
        std::deque<std::tuple<float, int, float, float, float, float>> q;

        // Loop through the faces in the mesh.
        for (int f = face_start_idx; f < face_stop_idx; ++f) {
          // Get coordinates of three face vertices.
          const auto& face = face_verts_a[f];
          float x0, x1, x2, y0, y1, y2, z0, z1, z2;
          std::tie(x0, y0, z0) = ExtractVerts(face, 0);
          std::tie(x1, y1, z1) = ExtractVerts(face, 1);
          std::tie(x2, y2, z2) = ExtractVerts(face, 2);

          const vec2<float> v0(x0, y0);
          const vec2<float> v1(x1, y1);
          const vec2<float> v2(x2, y2);

          const float face_area = face_areas_a[f];
          const bool back_face = face_area < 0.0;
          // Check if the face is visible to the camera.
          if (cull_backfaces && back_face) {
            continue;
          }
          // Skip faces with zero area.
          if (face_area <= kEpsilon && face_area >= -1.0f * kEpsilon) {
            continue;
          }

          // Skip if point is outside the face bounding box.
          const auto face_bbox = face_bboxes_a[f];
          const bool outside_bbox = CheckPointOutsideBoundingBox(
              face_bbox, std::sqrt(blur_radius), xf, yf);
          if (outside_bbox) {
            continue;
          }

          // Compute barycentric coordinates and use this to get the
          // depth of the point on the triangle.
          const vec2<float> pxy(xf, yf);
          const vec3<float> bary0 =
              BarycentricCoordinatesForward(pxy, v0, v1, v2);
          const vec3<float> bary = !perspective_correct
              ? bary0
              : BarycentricPerspectiveCorrectionForward(bary0, z0, z1, z2);

          const vec3<float> bary_clip =
              !clip_barycentric_coords ? bary : BarycentricClipForward(bary);

          // Use barycentric coordinates to get the depth of the current pixel
          const float pz =
              (bary_clip.x * z0 + bary_clip.y * z1 + bary_clip.z * z2);

          if (pz < 0) {
            continue; // Point is behind the image plane so ignore.
          }

          // Compute squared distance of the point to the triangle.
          const float dist = PointTriangleDistanceForward(pxy, v0, v1, v2);

          // Use the bary coordinates to determine if the point is
          // inside the face.
          const bool inside = bary.x > 0.0f && bary.y > 0.0f && bary.z > 0.0f;

          // If the point is inside the triangle then signed_dist
          // is negative.
          const float signed_dist = inside ? -dist : dist;

          // Check if pixel is outside blur region
          if (!inside && dist >= blur_radius) {
            continue;
          }

          // Handle the case where a face (f) partially behind the image plane
          // is clipped to a quadrilateral and then split into two faces (t1,
          // t2). In this case we:
          // 1. Find the index of the neighbor (e.g. for t1 need index of t2)
          // 2. Check if the neighbor (t2) is already in the top K faces
          // 3. If yes, compare the distance of the pixel to t1 with the
          // distance to t2.
          // 4. If dist_t1 < dist_t2, overwrite the values for t2 in the top K
          // faces.
          const int neighbor_idx = neighbor_idx_a[f];
          int idx_top_k = -1;

          // Check if neighboring face is already in the top K.
          if (neighbor_idx != -1) {
            const auto it =
                std::find_if(q.begin(), q.end(), IsNeighbor(neighbor_idx));
            // Get the index of the element from the iterator
            idx_top_k = (it != q.end()) ? it - q.begin() : idx_top_k;
          }

          // If idx_top_k idx is not -1 then it is in the top K struct.
          if (idx_top_k != -1) {
            // If dist of current face is less than neighbor, overwrite
            // the neighbor face values in the top K struct.
            const auto neighbor = q[idx_top_k];
            const float dist_neighbor = std::abs(std::get<2>(neighbor));
            if (dist < dist_neighbor) {
              // Overwrite the neighbor face values.
              q[idx_top_k] = std::make_tuple(
                  pz, f, signed_dist, bary_clip.x, bary_clip.y, bary_clip.z);
            }
          } else {
            // Handle as a normal face.
            // The current pixel lies inside the current face.
            // Add at the end of the deque.
            q.emplace_back(
                pz, f, signed_dist, bary_clip.x, bary_clip.y, bary_clip.z);
          }

          // Sort the deque inplace based on the z distance
          // to mimic using a priority queue.
          std::sort(q.begin(), q.end());
          if (static_cast<int>(q.size()) > K) {
            // remove the last value
            q.pop_back();
          }
        }
        while (!q.empty()) {
          // Loop through and add values to the output tensors
          auto t = q.back();
          q.pop_back();
          const int i = q.size();
          zbuf_a[n][yi][xi][i] = std::get<0>(t);
          face_idxs_a[n][yi][xi][i] = std::get<1>(t);
          pix_dists_a[n][yi][xi][i] = std::get<2>(t);
          barycentric_coords_a[n][yi][xi][i][0] = std::get<3>(t);
          barycentric_coords_a[n][yi][xi][i][1] = std::get<4>(t);
          barycentric_coords_a[n][yi][xi][i][2] = std::get<5>(t);
        }
      }
    }
  }
}
} // namespace

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeMeshesNaiveCpu(
    const torch::Tensor& face_verts,
    const torch::Tensor& mesh_to_face_first_idx,
    const torch::Tensor& num_faces_per_mesh,
    const torch::Tensor& clipped_faces_neighbor_idx,
    const std::tuple<int, int> image_size,
    const float blur_radius,
    const int faces_per_pixel,
    const bool perspective_correct,
    const bool clip_barycentric_coords,
    const bool cull_backfaces) {
  if (face_verts.ndimension() != 3 || face_verts.size(1) != 3 ||
      face_verts.size(2) != 3) {
    AT_ERROR("face_verts must have dimensions (num_faces, 3, 3)");
  }
  if (num_faces_per_mesh.size(0) != mesh_to_face_first_idx.size(0)) {
    AT_ERROR(
        "num_faces_per_mesh must have save size first dimension as mesh_to_face_first_idx");
  }

  const int32_t N = mesh_to_face_first_idx.size(0); // batch_size.
  const int H = std::get<0>(image_size);
  const int W = std::get<1>(image_size);
  const int K = faces_per_pixel;

  auto long_opts = num_faces_per_mesh.options().dtype(torch::kInt64);
  auto float_opts = face_verts.options().dtype(torch::kFloat32);

  // Initialize output tensors.
  torch::Tensor face_idxs = torch::full({N, H, W, K}, -1, long_opts);
  torch::Tensor zbuf = torch::full({N, H, W, K}, -1, float_opts);
  torch::Tensor pix_dists = torch::full({N, H, W, K}, -1, float_opts);
  torch::Tensor barycentric_coords =
      torch::full({N, H, W, K, 3}, -1, float_opts);

  auto face_verts_a = face_verts.accessor<float, 3>();
  auto face_idxs_a = face_idxs.accessor<int64_t, 4>();
  auto zbuf_a = zbuf.accessor<float, 4>();
  auto pix_dists_a = pix_dists.accessor<float, 4>();
  auto barycentric_coords_a = barycentric_coords.accessor<float, 5>();
  auto neighbor_idx_a = clipped_faces_neighbor_idx.accessor<int64_t, 1>();

  auto face_bboxes = ComputeFaceBoundingBoxes(face_verts);
  auto face_bboxes_a = face_bboxes.accessor<float, 2>();
  auto face_areas = ComputeFaceAreas(face_verts);
  auto face_areas_a = face_areas.accessor<float, 1>();

  const int64_t n_threads = at::get_num_threads();
  std::vector<std::thread> threads;
  threads.reserve(n_threads);
  const int chunk_size = 1 + (H - 1) / n_threads;
  int start_yi = 0;
  for (int iThread = 0; iThread < n_threads; ++iThread) {
    const int64_t end_yi = std::min(start_yi + chunk_size, H);
    threads.emplace_back(
        RasterizeMeshesNaiveCpu_worker,
        start_yi,
        end_yi,
        mesh_to_face_first_idx,
        num_faces_per_mesh,
        blur_radius,
        perspective_correct,
        clip_barycentric_coords,
        cull_backfaces,
        N,
        H,
        W,
        K,
        std::ref(face_verts_a),
        std::ref(face_areas_a),
        std::ref(face_bboxes_a),
        std::ref(neighbor_idx_a),
        std::ref(zbuf_a),
        std::ref(face_idxs_a),
        std::ref(pix_dists_a),
        std::ref(barycentric_coords_a));
    start_yi += chunk_size;
  }
  for (auto&& thread : threads) {
    thread.join();
  }

  return std::make_tuple(face_idxs, zbuf, barycentric_coords, pix_dists);
}

torch::Tensor RasterizeMeshesBackwardCpu(
    const torch::Tensor& face_verts, // (F, 3, 3)
    const torch::Tensor& pix_to_face, // (N, H, W, K)
    const torch::Tensor& grad_zbuf, // (N, H, W, K)
    const torch::Tensor& grad_bary, // (N, H, W, K, 3)
    const torch::Tensor& grad_dists, // (N, H, W, K)
    const bool perspective_correct,
    const bool clip_barycentric_coords) {
  const int F = face_verts.size(0);
  const int N = pix_to_face.size(0);
  const int H = pix_to_face.size(1);
  const int W = pix_to_face.size(2);
  const int K = pix_to_face.size(3);

  torch::Tensor grad_face_verts = torch::zeros({F, 3, 3}, face_verts.options());
  auto face_verts_a = face_verts.accessor<float, 3>();
  auto pix_to_face_a = pix_to_face.accessor<int64_t, 4>();
  auto grad_dists_a = grad_dists.accessor<float, 4>();
  auto grad_zbuf_a = grad_zbuf.accessor<float, 4>();
  auto grad_bary_a = grad_bary.accessor<float, 5>();

  for (int n = 0; n < N; ++n) {
    // Iterate through the horizontal lines of the image from top to bottom.
    for (int y = 0; y < H; ++y) {
      // Reverse the order of yi so that +Y is pointing upwards in the image.
      const int yidx = H - 1 - y;

      // Y coordinate of the top of the pixel.
      const float yf = PixToNonSquareNdc(yidx, H, W);
      // Iterate through pixels on this horizontal line, left to right.
      for (int x = 0; x < W; ++x) {
        // Reverse the order of xi so that +X is pointing to the left in the
        // image.
        const int xidx = W - 1 - x;

        // X coordinate of the left of the pixel.
        const float xf = PixToNonSquareNdc(xidx, W, H);
        const vec2<float> pxy(xf, yf);

        // Iterate through the faces that hit this pixel.
        for (int k = 0; k < K; ++k) {
          // Get face index from forward pass output.
          const int f = pix_to_face_a[n][y][x][k];
          if (f < 0) {
            continue; // padded face.
          }
          // Get coordinates of the three face vertices.
          const auto face_verts_f = face_verts_a[f];
          const float x0 = face_verts_f[0][0];
          const float y0 = face_verts_f[0][1];
          const float z0 = face_verts_f[0][2];
          const float x1 = face_verts_f[1][0];
          const float y1 = face_verts_f[1][1];
          const float z1 = face_verts_f[1][2];
          const float x2 = face_verts_f[2][0];
          const float y2 = face_verts_f[2][1];
          const float z2 = face_verts_f[2][2];
          const vec2<float> v0xy(x0, y0);
          const vec2<float> v1xy(x1, y1);
          const vec2<float> v2xy(x2, y2);

          // Get upstream gradients for the face.
          const float grad_dist_upstream = grad_dists_a[n][y][x][k];
          const float grad_zbuf_upstream = grad_zbuf_a[n][y][x][k];
          const auto grad_bary_upstream_w012 = grad_bary_a[n][y][x][k];
          const float grad_bary_upstream_w0 = grad_bary_upstream_w012[0];
          const float grad_bary_upstream_w1 = grad_bary_upstream_w012[1];
          const float grad_bary_upstream_w2 = grad_bary_upstream_w012[2];
          const vec3<float> grad_bary_upstream(
              grad_bary_upstream_w0,
              grad_bary_upstream_w1,
              grad_bary_upstream_w2);

          const vec3<float> bary0 =
              BarycentricCoordinatesForward(pxy, v0xy, v1xy, v2xy);
          const vec3<float> bary = !perspective_correct
              ? bary0
              : BarycentricPerspectiveCorrectionForward(bary0, z0, z1, z2);
          const vec3<float> bary_clip =
              !clip_barycentric_coords ? bary : BarycentricClipForward(bary);

          // Distances inside the face are negative so get the
          // correct sign to apply to the upstream gradient.
          const bool inside = bary.x > 0.0f && bary.y > 0.0f && bary.z > 0.0f;
          const float sign = inside ? -1.0f : 1.0f;

          const auto grad_dist_f = PointTriangleDistanceBackward(
              pxy, v0xy, v1xy, v2xy, sign * grad_dist_upstream);
          const auto ddist_d_v0 = std::get<1>(grad_dist_f);
          const auto ddist_d_v1 = std::get<2>(grad_dist_f);
          const auto ddist_d_v2 = std::get<3>(grad_dist_f);

          // Upstream gradient for barycentric coords from zbuf calculation:
          // zbuf = bary_w0 * z0 + bary_w1 * z1 + bary_w2 * z2
          // Therefore
          // d_zbuf/d_bary_w0 = z0
          // d_zbuf/d_bary_w1 = z1
          // d_zbuf/d_bary_w2 = z2
          const vec3<float> d_zbuf_d_baryclip(z0, z1, z2);

          // Total upstream barycentric gradients are the sum of
          // external upstream gradients and contribution from zbuf.
          const vec3<float> grad_bary_f_sum =
              (grad_bary_upstream + grad_zbuf_upstream * d_zbuf_d_baryclip);

          vec3<float> grad_bary0 = grad_bary_f_sum;

          if (clip_barycentric_coords) {
            grad_bary0 = BarycentricClipBackward(bary, grad_bary0);
          }

          if (perspective_correct) {
            auto perspective_grads = BarycentricPerspectiveCorrectionBackward(
                bary0, z0, z1, z2, grad_bary0);
            grad_bary0 = std::get<0>(perspective_grads);
            grad_face_verts[f][0][2] += std::get<1>(perspective_grads);
            grad_face_verts[f][1][2] += std::get<2>(perspective_grads);
            grad_face_verts[f][2][2] += std::get<3>(perspective_grads);
          }

          auto grad_bary_f =
              BarycentricCoordsBackward(pxy, v0xy, v1xy, v2xy, grad_bary0);
          const vec2<float> dbary_d_v0 = std::get<1>(grad_bary_f);
          const vec2<float> dbary_d_v1 = std::get<2>(grad_bary_f);
          const vec2<float> dbary_d_v2 = std::get<3>(grad_bary_f);

          // Update output gradient buffer.
          grad_face_verts[f][0][0] += dbary_d_v0.x + ddist_d_v0.x;
          grad_face_verts[f][0][1] += dbary_d_v0.y + ddist_d_v0.y;
          grad_face_verts[f][0][2] += grad_zbuf_upstream * bary_clip.x;
          grad_face_verts[f][1][0] += dbary_d_v1.x + ddist_d_v1.x;
          grad_face_verts[f][1][1] += dbary_d_v1.y + ddist_d_v1.y;
          grad_face_verts[f][1][2] += grad_zbuf_upstream * bary_clip.y;
          grad_face_verts[f][2][0] += dbary_d_v2.x + ddist_d_v2.x;
          grad_face_verts[f][2][1] += dbary_d_v2.y + ddist_d_v2.y;
          grad_face_verts[f][2][2] += grad_zbuf_upstream * bary_clip.z;
        }
      }
    }
  }
  return grad_face_verts;
}

torch::Tensor RasterizeMeshesCoarseCpu(
    const torch::Tensor& face_verts,
    const torch::Tensor& mesh_to_face_first_idx,
    const torch::Tensor& num_faces_per_mesh,
    const std::tuple<int, int> image_size,
    const float blur_radius,
    const int bin_size,
    const int max_faces_per_bin) {
  if (face_verts.ndimension() != 3 || face_verts.size(1) != 3 ||
      face_verts.size(2) != 3) {
    AT_ERROR("face_verts must have dimensions (num_faces, 3, 3)");
  }
  if (num_faces_per_mesh.ndimension() != 1) {
    AT_ERROR("num_faces_per_mesh can only have one dimension");
  }

  const int N = num_faces_per_mesh.size(0); // batch size.
  const int M = max_faces_per_bin;

  const float H = std::get<0>(image_size);
  const float W = std::get<1>(image_size);

  // Integer division round up.
  const int BH = 1 + (H - 1) / bin_size;
  const int BW = 1 + (W - 1) / bin_size;

  auto opts = num_faces_per_mesh.options().dtype(torch::kInt32);
  torch::Tensor faces_per_bin = torch::zeros({N, BH, BW}, opts);
  torch::Tensor bin_faces = torch::full({N, BH, BW, M}, -1, opts);
  auto bin_faces_a = bin_faces.accessor<int32_t, 4>();

  // Precompute all face bounding boxes.
  auto face_bboxes = ComputeFaceBoundingBoxes(face_verts);
  auto face_bboxes_a = face_bboxes.accessor<float, 2>();

  const float ndc_x_range = NonSquareNdcRange(W, H);
  const float pixel_width_x = ndc_x_range / W;
  const float bin_width_x = pixel_width_x * bin_size;

  const float ndc_y_range = NonSquareNdcRange(H, W);
  const float pixel_width_y = ndc_y_range / H;
  const float bin_width_y = pixel_width_y * bin_size;

  // Iterate through the meshes in the batch.
  for (int n = 0; n < N; ++n) {
    const int face_start_idx = mesh_to_face_first_idx[n].item().to<int32_t>();
    const int face_stop_idx =
        (face_start_idx + num_faces_per_mesh[n].item().to<int32_t>());

    float bin_y_min = -1.0f;
    float bin_y_max = bin_y_min + bin_width_y;

    // Iterate through the horizontal bins from top to bottom.
    for (int by = 0; by < BH; ++by) {
      float bin_x_min = -1.0f;
      float bin_x_max = bin_x_min + bin_width_x;

      // Iterate through bins on this horizontal line, left to right.
      for (int bx = 0; bx < BW; ++bx) {
        int32_t faces_hit = 0;

        for (int32_t f = face_start_idx; f < face_stop_idx; ++f) {
          // Get bounding box and expand by blur radius.
          float face_x_min = face_bboxes_a[f][0] - std::sqrt(blur_radius);
          float face_y_min = face_bboxes_a[f][1] - std::sqrt(blur_radius);
          float face_x_max = face_bboxes_a[f][2] + std::sqrt(blur_radius);
          float face_y_max = face_bboxes_a[f][3] + std::sqrt(blur_radius);
          float face_z_min = face_bboxes_a[f][4];

          // Faces with at least one vertex behind the camera won't render
          // correctly and should be removed or clipped before calling the
          // rasterizer
          if (face_z_min < kEpsilon) {
            continue;
          }

          // Use a half-open interval so that faces exactly on the
          // boundary between bins will fall into exactly one bin.
          bool x_overlap =
              (face_x_min <= bin_x_max) && (bin_x_min < face_x_max);
          bool y_overlap =
              (face_y_min <= bin_y_max) && (bin_y_min < face_y_max);

          if (x_overlap && y_overlap) {
            // Got too many faces for this bin, so throw an error.
            if (faces_hit >= max_faces_per_bin) {
              AT_ERROR("Got too many faces per bin");
            }
            // The current point falls in the current bin, so
            // record it.
            bin_faces_a[n][by][bx][faces_hit] = f;
            faces_hit++;
          }
        }

        // Shift the bin to the right for the next loop iteration
        bin_x_min = bin_x_max;
        bin_x_max = bin_x_min + bin_width_x;
      }
      // Shift the bin down for the next loop iteration
      bin_y_min = bin_y_max;
      bin_y_max = bin_y_min + bin_width_y;
    }
  }
  return bin_faces;
}
