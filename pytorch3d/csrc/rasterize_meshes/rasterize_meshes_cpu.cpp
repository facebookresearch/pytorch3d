// Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

#include <torch/extension.h>
#include <algorithm>
#include <list>
#include <queue>
#include <tuple>
#include "utils/geometry_utils.h"
#include "utils/vec2.h"
#include "utils/vec3.h"

float PixToNdc(int i, int S) {
  // NDC x-offset + (i * pixel_width + half_pixel_width)
  return -1 + (2 * i + 1.0f) / S;
}

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

  // Check if the current point is within the triangle bounding box.
  return (px > x_max || px < x_min || py > y_max || py < y_min);
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

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeMeshesNaiveCpu(
    const torch::Tensor& face_verts,
    const torch::Tensor& mesh_to_face_first_idx,
    const torch::Tensor& num_faces_per_mesh,
    int image_size,
    const float blur_radius,
    const int faces_per_pixel,
    const bool perspective_correct,
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
  const int H = image_size;
  const int W = image_size;
  const int K = faces_per_pixel;

  auto long_opts = face_verts.options().dtype(torch::kInt64);
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

  auto face_bboxes = ComputeFaceBoundingBoxes(face_verts);
  auto face_bboxes_a = face_bboxes.accessor<float, 2>();
  auto face_areas = ComputeFaceAreas(face_verts);
  auto face_areas_a = face_areas.accessor<float, 1>();

  for (int n = 0; n < N; ++n) {
    // Loop through each mesh in the batch.
    // Get the start index of the faces in faces_packed and the num faces
    // in the mesh to avoid having to loop through all the faces.
    const int face_start_idx = mesh_to_face_first_idx[n].item().to<int32_t>();
    const int face_stop_idx =
        (face_start_idx + num_faces_per_mesh[n].item().to<int32_t>());

    // Iterate through the horizontal lines of the image from top to bottom.
    for (int yi = 0; yi < H; ++yi) {
      // Reverse the order of yi so that +Y is pointing upwards in the image.
      const int yidx = H - 1 - yi;

      // Y coordinate of the top of the pixel.
      const float yf = PixToNdc(yidx, H);
      // Iterate through pixels on this horizontal line, left to right.
      for (int xi = 0; xi < W; ++xi) {
        // Reverse the order of xi so that +X is pointing to the left in the
        // image.
        const int xidx = W - 1 - xi;

        // X coordinate of the left of the pixel.
        const float xf = PixToNdc(xidx, W);
        // Use a priority queue to hold values:
        // (z, idx, r, bary.x, bary.y. bary.z)
        std::priority_queue<std::tuple<float, int, float, float, float, float>>
            q;

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

          // Use barycentric coordinates to get the depth of the current pixel
          const float pz = (bary.x * z0 + bary.y * z1 + bary.z * z2);

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
          // The current pixel lies inside the current face.
          q.emplace(pz, f, signed_dist, bary.x, bary.y, bary.z);
          if (static_cast<int>(q.size()) > K) {
            q.pop();
          }
        }
        while (!q.empty()) {
          auto t = q.top();
          q.pop();
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
  return std::make_tuple(face_idxs, zbuf, barycentric_coords, pix_dists);
}

torch::Tensor RasterizeMeshesBackwardCpu(
    const torch::Tensor& face_verts, // (F, 3, 3)
    const torch::Tensor& pix_to_face, // (N, H, W, K)
    const torch::Tensor& grad_zbuf, // (N, H, W, K)
    const torch::Tensor& grad_bary, // (N, H, W, K, 3)
    const torch::Tensor& grad_dists, // (N, H, W, K)
    const bool perspective_correct) {
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
      const float yf = PixToNdc(yidx, H);
      // Iterate through pixels on this horizontal line, left to right.
      for (int x = 0; x < W; ++x) {
        // Reverse the order of xi so that +X is pointing to the left in the
        // image.
        const int xidx = W - 1 - x;

        // X coordinate of the left of the pixel.
        const float xf = PixToNdc(xidx, W);
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

          // Distances inside the face are negative so get the
          // correct sign to apply to the upstream gradient.
          const bool inside = bary.x > 0.0f && bary.y > 0.0f && bary.z > 0.0f;
          const float sign = inside ? -1.0f : 1.0f;

          // TODO(T52813608) Add support for non-square images.
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
          const vec3<float> d_zbuf_d_bary(z0, z1, z2);

          // Total upstream barycentric gradients are the sum of
          // external upstream gradients and contribution from zbuf.
          vec3<float> grad_bary_f_sum =
              (grad_bary_upstream + grad_zbuf_upstream * d_zbuf_d_bary);

          vec3<float> grad_bary0 = grad_bary_f_sum;
          if (perspective_correct) {
            auto perspective_grads = BarycentricPerspectiveCorrectionBackward(
                bary0, z0, z1, z2, grad_bary_f_sum);
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
          grad_face_verts[f][0][2] += grad_zbuf_upstream * bary.x;
          grad_face_verts[f][1][0] += dbary_d_v1.x + ddist_d_v1.x;
          grad_face_verts[f][1][1] += dbary_d_v1.y + ddist_d_v1.y;
          grad_face_verts[f][1][2] += grad_zbuf_upstream * bary.y;
          grad_face_verts[f][2][0] += dbary_d_v2.x + ddist_d_v2.x;
          grad_face_verts[f][2][1] += dbary_d_v2.y + ddist_d_v2.y;
          grad_face_verts[f][2][2] += grad_zbuf_upstream * bary.z;
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
    const int image_size,
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

  // Assume square images. TODO(T52813608) Support non square images.
  const float height = image_size;
  const float width = image_size;
  const int BH = 1 + (height - 1) / bin_size; // Integer division round up.
  const int BW = 1 + (width - 1) / bin_size; // Integer division round up.

  auto opts = face_verts.options().dtype(torch::kInt32);
  torch::Tensor faces_per_bin = torch::zeros({N, BH, BW}, opts);
  torch::Tensor bin_faces = torch::full({N, BH, BW, M}, -1, opts);
  auto bin_faces_a = bin_faces.accessor<int32_t, 4>();

  // Precompute all face bounding boxes.
  auto face_bboxes = ComputeFaceBoundingBoxes(face_verts);
  auto face_bboxes_a = face_bboxes.accessor<float, 2>();

  const float pixel_width = 2.0f / image_size;
  const float bin_width = pixel_width * bin_size;

  // Iterate through the meshes in the batch.
  for (int n = 0; n < N; ++n) {
    const int face_start_idx = mesh_to_face_first_idx[n].item().to<int32_t>();
    const int face_stop_idx =
        (face_start_idx + num_faces_per_mesh[n].item().to<int32_t>());

    float bin_y_min = -1.0f;
    float bin_y_max = bin_y_min + bin_width;

    // Iterate through the horizontal bins from top to bottom.
    for (int by = 0; by < BH; ++by) {
      float bin_x_min = -1.0f;
      float bin_x_max = bin_x_min + bin_width;

      // Iterate through bins on this horizontal line, left to right.
      for (int bx = 0; bx < BW; ++bx) {
        int32_t faces_hit = 0;

        for (int32_t f = face_start_idx; f < face_stop_idx; ++f) {
          // Get bounding box and expand by blur radius.
          float face_x_min = face_bboxes_a[f][0] - std::sqrt(blur_radius);
          float face_y_min = face_bboxes_a[f][1] - std::sqrt(blur_radius);
          float face_x_max = face_bboxes_a[f][2] + std::sqrt(blur_radius);
          float face_y_max = face_bboxes_a[f][3] + std::sqrt(blur_radius);
          float face_z_max = face_bboxes_a[f][5];

          if (face_z_max < 0) {
            continue; // Face is behind the camera.
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
        bin_x_max = bin_x_min + bin_width;
      }
      // Shift the bin down for the next loop iteration
      bin_y_min = bin_y_max;
      bin_y_max = bin_y_min + bin_width;
    }
  }
  return bin_faces;
}
