// Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

#pragma once
#include <torch/extension.h>
#include <cstdio>
#include <tuple>
#include "utils/pytorch3d_cutils.h"

// ****************************************************************************
// *                            FORWARD PASS                                 *
// ****************************************************************************

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeMeshesNaiveCpu(
    const torch::Tensor& face_verts,
    const torch::Tensor& mesh_to_face_first_idx,
    const torch::Tensor& num_faces_per_mesh,
    const int image_size,
    const float blur_radius,
    const int faces_per_pixel,
    const bool perspective_correct,
    const bool cull_backfaces);

#ifdef WITH_CUDA
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
RasterizeMeshesNaiveCuda(
    const at::Tensor& face_verts,
    const at::Tensor& mesh_to_face_first_idx,
    const at::Tensor& num_faces_per_mesh,
    const int image_size,
    const float blur_radius,
    const int num_closest,
    const bool perspective_correct,
    const bool cull_backfaces);
#endif
// Forward pass for rasterizing a batch of meshes.
//
// Args:
//    face_verts: Tensor of shape (F, 3, 3) giving (packed) vertex positions for
//                faces in all the meshes in the batch. Concretely,
//                face_verts[f, i] = [x, y, z] gives the coordinates for the
//                ith vertex of the fth face. These vertices are expected to be
//                in NDC coordinates in the range [-1, 1].
//    mesh_to_face_first_idx: LongTensor of shape (N) giving the index in
//                            faces_verts of the first face in each mesh in
//                            the batch where N is the batch size.
//    num_faces_per_mesh: LongTensor of shape (N) giving the number of faces
//                        for each mesh in the batch.
//    image_size: Size in pixels of the output image to be rasterized.
//                Assume square images only.
//    blur_radius: float distance in NDC coordinates uses to expand the face
//                 bounding boxes for the rasterization. Set to 0.0 if no blur
//                 is required.
//    faces_per_pixel: the number of closeset faces to rasterize per pixel.
//    perspective_correct: Whether to apply perspective correction when
//                         computing barycentric coordinates. If this is True,
//                         then this function returns world-space barycentric
//                         coordinates for each pixel; if this is False then
//                         this function instead returns screen-space
//                         barycentric coordinates for each pixel.
//    cull_backfaces: Bool, Whether to only rasterize mesh faces which are
//                    visible to the camera.  This assumes that vertices of
//                    front-facing triangles are ordered in an anti-clockwise
//                    fashion, and triangles that face away from the camera are
//                    in a clockwise order relative to the current view
//                    direction. NOTE: This will only work if the mesh faces are
//                    consistently defined with counter-clockwise ordering when
//                    viewed from the outside.
//
// Returns:
//    A 4 element tuple of:
//    pix_to_face: int64 tensor of shape (N, H, W, K) giving the face index of
//                 each of the closest faces to the pixel in the rasterized
//                 image, or -1 for pixels that are not covered by any face.
//    zbuf: float32 Tensor of shape (N, H, W, K) giving the depth of each of
//          the closest faces for each pixel.
//    barycentric_coords: float tensor of shape (N, H, W, K, 3) giving
//                        barycentric coordinates of the pixel with respect to
//                        each of the closest faces along the z axis, padded
//                        with -1 for pixels hit by fewer than
//                        faces_per_pixel faces.
//    dists: float tensor of shape (N, H, W, K) giving the euclidean distance
//           in the (NDC) x/y plane between each pixel and its K closest
//           faces along the z axis padded  with -1 for pixels hit by fewer than
//           faces_per_pixel faces.
inline std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeMeshesNaive(
    const torch::Tensor& face_verts,
    const torch::Tensor& mesh_to_face_first_idx,
    const torch::Tensor& num_faces_per_mesh,
    const int image_size,
    const float blur_radius,
    const int faces_per_pixel,
    const bool perspective_correct,
    const bool cull_backfaces) {
  // TODO: Better type checking.
  if (face_verts.is_cuda()) {
#ifdef WITH_CUDA
    CHECK_CONTIGUOUS_CUDA(face_verts);
    CHECK_CONTIGUOUS_CUDA(mesh_to_face_first_idx);
    CHECK_CONTIGUOUS_CUDA(num_faces_per_mesh);
    return RasterizeMeshesNaiveCuda(
        face_verts,
        mesh_to_face_first_idx,
        num_faces_per_mesh,
        image_size,
        blur_radius,
        faces_per_pixel,
        perspective_correct,
        cull_backfaces);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  } else {
    return RasterizeMeshesNaiveCpu(
        face_verts,
        mesh_to_face_first_idx,
        num_faces_per_mesh,
        image_size,
        blur_radius,
        faces_per_pixel,
        perspective_correct,
        cull_backfaces);
  }
}

// ****************************************************************************
// *                            BACKWARD PASS                                 *
// ****************************************************************************

torch::Tensor RasterizeMeshesBackwardCpu(
    const torch::Tensor& face_verts,
    const torch::Tensor& pix_to_face,
    const torch::Tensor& grad_bary,
    const torch::Tensor& grad_zbuf,
    const torch::Tensor& grad_dists,
    const bool perspective_correct);

#ifdef WITH_CUDA
torch::Tensor RasterizeMeshesBackwardCuda(
    const torch::Tensor& face_verts,
    const torch::Tensor& pix_to_face,
    const torch::Tensor& grad_bary,
    const torch::Tensor& grad_zbuf,
    const torch::Tensor& grad_dists,
    const bool perspective_correct);
#endif

// Args:
//    face_verts: float32 Tensor of shape (F, 3, 3) (from forward pass) giving
//                (packed) vertex positions for faces in all the meshes in
//                 the batch.
//    pix_to_face: int64 tensor of shape (N, H, W, K) giving the face index of
//                 each of the closest faces to the pixel in the rasterized
//                 image, or -1 for pixels that are not covered by any face.
//    grad_zbuf: Tensor of shape (N, H, W, K) giving upstream gradients
//               d(loss)/d(zbuf) of the zbuf tensor from the forward pass.
//    grad_bary: Tensor of shape (N, H, W, K, 3) giving upstream gradients
//               d(loss)/d(bary) of the barycentric_coords tensor returned by
//               the forward pass.
//    grad_dists: Tensor of shape (N, H, W, K) giving upstream gradients
//                d(loss)/d(dists) of the dists tensor from the forward pass.
//    perspective_correct: Whether to apply perspective correction when
//                         computing barycentric coordinates. If this is True,
//                         then this function returns world-space barycentric
//                         coordinates for each pixel; if this is False then
//                         this function instead returns screen-space
//                         barycentric coordinates for each pixel.
//
// Returns:
//    grad_face_verts: float32 Tensor of shape (F, 3, 3) giving downstream
//                     gradients for the face vertices.
torch::Tensor RasterizeMeshesBackward(
    const torch::Tensor& face_verts,
    const torch::Tensor& pix_to_face,
    const torch::Tensor& grad_zbuf,
    const torch::Tensor& grad_bary,
    const torch::Tensor& grad_dists,
    const bool perspective_correct) {
  if (face_verts.is_cuda()) {
#ifdef WITH_CUDA
    CHECK_CONTIGUOUS_CUDA(face_verts);
    CHECK_CONTIGUOUS_CUDA(pix_to_face);
    CHECK_CONTIGUOUS_CUDA(grad_zbuf);
    CHECK_CONTIGUOUS_CUDA(grad_bary);
    CHECK_CONTIGUOUS_CUDA(grad_dists);
    return RasterizeMeshesBackwardCuda(
        face_verts,
        pix_to_face,
        grad_zbuf,
        grad_bary,
        grad_dists,
        perspective_correct);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  } else {
    return RasterizeMeshesBackwardCpu(
        face_verts,
        pix_to_face,
        grad_zbuf,
        grad_bary,
        grad_dists,
        perspective_correct);
  }
}

// ****************************************************************************
// *                          COARSE RASTERIZATION                            *
// ****************************************************************************

torch::Tensor RasterizeMeshesCoarseCpu(
    const torch::Tensor& face_verts,
    const at::Tensor& mesh_to_face_first_idx,
    const at::Tensor& num_faces_per_mesh,
    const int image_size,
    const float blur_radius,
    const int bin_size,
    const int max_faces_per_bin);

#ifdef WITH_CUDA
torch::Tensor RasterizeMeshesCoarseCuda(
    const torch::Tensor& face_verts,
    const torch::Tensor& mesh_to_face_first_idx,
    const torch::Tensor& num_faces_per_mesh,
    const int image_size,
    const float blur_radius,
    const int bin_size,
    const int max_faces_per_bin);
#endif
// Args:
//    face_verts: Tensor of shape (F, 3, 3) giving (packed) vertex positions for
//                faces in all the meshes in the batch. Concretely,
//                face_verts[f, i] = [x, y, z] gives the coordinates for the
//                ith vertex of the fth face. These vertices are expected to be
//                in NDC coordinates in the range [-1, 1].
//    mesh_to_face_first_idx: LongTensor of shape (N) giving the index in
//                            faces_verts of the first face in each mesh in
//                            the batch where N is the batch size.
//    num_faces_per_mesh: LongTensor of shape (N) giving the number of faces
//                        for each mesh in the batch.
//    image_size: Size in pixels of the output image to be rasterized.
//    blur_radius: float distance in NDC coordinates uses to expand the face
//                 bounding boxes for the rasterization. Set to 0.0 if no blur
//                 is required.
//    bin_size: Size of each bin within the image (in pixels)
//    max_faces_per_bin: Maximum number of faces to count in each bin.
//
// Returns:
//   bin_face_idxs: Tensor of shape (N, num_bins, num_bins, K) giving the
//                  indices of faces that fall into each bin.

torch::Tensor RasterizeMeshesCoarse(
    const torch::Tensor& face_verts,
    const torch::Tensor& mesh_to_face_first_idx,
    const torch::Tensor& num_faces_per_mesh,
    const int image_size,
    const float blur_radius,
    const int bin_size,
    const int max_faces_per_bin) {
  if (face_verts.is_cuda()) {
#ifdef WITH_CUDA
    CHECK_CONTIGUOUS_CUDA(face_verts);
    CHECK_CONTIGUOUS_CUDA(mesh_to_face_first_idx);
    CHECK_CONTIGUOUS_CUDA(num_faces_per_mesh);
    return RasterizeMeshesCoarseCuda(
        face_verts,
        mesh_to_face_first_idx,
        num_faces_per_mesh,
        image_size,
        blur_radius,
        bin_size,
        max_faces_per_bin);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  } else {
    return RasterizeMeshesCoarseCpu(
        face_verts,
        mesh_to_face_first_idx,
        num_faces_per_mesh,
        image_size,
        blur_radius,
        bin_size,
        max_faces_per_bin);
  }
}

// ****************************************************************************
// *                            FINE RASTERIZATION                            *
// ****************************************************************************

#ifdef WITH_CUDA
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeMeshesFineCuda(
    const torch::Tensor& face_verts,
    const torch::Tensor& bin_faces,
    const int image_size,
    const float blur_radius,
    const int bin_size,
    const int faces_per_pixel,
    const bool perspective_correct,
    const bool cull_backfaces);
#endif
// Args:
//    face_verts: Tensor of shape (F, 3, 3) giving (packed) vertex positions for
//                faces in all the meshes in the batch. Concretely,
//                face_verts[f, i] = [x, y, z] gives the coordinates for the
//                ith vertex of the fth face. These vertices are expected to be
//                in NDC coordinates in the range [-1, 1].
//    bin_faces: int32 Tensor of shape (N, B, B, M) giving the indices of faces
//               that fall into each bin (output from coarse rasterization).
//    image_size: Size in pixels of the output image to be rasterized.
//    blur_radius: float distance in NDC coordinates uses to expand the face
//                 bounding boxes for the rasterization. Set to 0.0 if no blur
//                 is required.
//    bin_size: Size of each bin within the image (in pixels)
//    faces_per_pixel: the number of closeset faces to rasterize per pixel.
//    perspective_correct: Whether to apply perspective correction when
//                         computing barycentric coordinates. If this is True,
//                         then this function returns world-space barycentric
//                         coordinates for each pixel; if this is False then
//                         this function instead returns screen-space
//                         barycentric coordinates for each pixel.
//    cull_backfaces: Bool, Whether to only rasterize mesh faces which are
//                    visible to the camera.  This assumes that vertices of
//                    front-facing triangles are ordered in an anti-clockwise
//                    fashion, and triangles that face away from the camera are
//                    in a clockwise order relative to the current view
//                    direction. NOTE: This will only work if the mesh faces are
//                    consistently defined with counter-clockwise ordering when
//                    viewed from the outside.
//
// Returns (same as rasterize_meshes):
//    A 4 element tuple of:
//    pix_to_face: int64 tensor of shape (N, H, W, K) giving the face index of
//                 each of the closest faces to the pixel in the rasterized
//                 image, or -1 for pixels that are not covered by any face.
//    zbuf: float32 Tensor of shape (N, H, W, K) giving the depth of each of
//          the closest faces for each pixel.
//    barycentric_coords: float tensor of shape (N, H, W, K, 3) giving
//                        barycentric coordinates of the pixel with respect to
//                        each of the closest faces along the z axis, padded
//                        with -1 for pixels hit by fewer than
//                        faces_per_pixel faces.
//    dists: float tensor of shape (N, H, W, K) giving the euclidean distance
//           in the (NDC) x/y plane between each pixel and its K closest
//           faces along the z axis padded  with -1 for pixels hit by fewer than
//           faces_per_pixel faces.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeMeshesFine(
    const torch::Tensor& face_verts,
    const torch::Tensor& bin_faces,
    const int image_size,
    const float blur_radius,
    const int bin_size,
    const int faces_per_pixel,
    const bool perspective_correct,
    const bool cull_backfaces) {
  if (face_verts.is_cuda()) {
#ifdef WITH_CUDA
    CHECK_CONTIGUOUS_CUDA(face_verts);
    CHECK_CONTIGUOUS_CUDA(bin_faces);
    return RasterizeMeshesFineCuda(
        face_verts,
        bin_faces,
        image_size,
        blur_radius,
        bin_size,
        faces_per_pixel,
        perspective_correct,
        cull_backfaces);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  } else {
    AT_ERROR("NOT IMPLEMENTED");
  }
}

// ****************************************************************************
// *                         MAIN ENTRY POINT                                 *
// ****************************************************************************

// This is the main entry point for the forward pass of the mesh rasterizer;
// it uses either naive or coarse-to-fine rasterization based on bin_size.
//
// Args:
//    face_verts: Tensor of shape (F, 3, 3) giving (packed) vertex positions for
//                faces in all the meshes in the batch. Concretely,
//                face_verts[f, i] = [x, y, z] gives the coordinates for the
//                ith vertex of the fth face. These vertices are expected to be
//                in NDC coordinates in the range [-1, 1].
//    mesh_to_face_first_idx: LongTensor of shape (N) giving the index in
//                            faces_verts of the first face in each mesh in
//                            the batch where N is the batch size.
//    num_faces_per_mesh: LongTensor of shape (N) giving the number of faces
//                        for each mesh in the batch.
//    image_size: Size in pixels of the output image to be rasterized.
//    blur_radius: float distance in NDC coordinates uses to expand the face
//                 bounding boxes for the rasterization. Set to 0.0 if no blur
//                 is required.
//    bin_size: Bin size (in pixels) for coarse-to-fine rasterization. Setting
//              bin_size=0 uses naive rasterization instead.
//    max_faces_per_bin: The maximum number of faces allowed to fall into each
//                      bin when using coarse-to-fine rasterization.
//    perspective_correct: Whether to apply perspective correction when
//                         computing barycentric coordinates. If this is True,
//                         then this function returns world-space barycentric
//                         coordinates for each pixel; if this is False then
//                         this function instead returns screen-space
//                         barycentric coordinates for each pixel.
//    cull_backfaces: Bool, Whether to only rasterize mesh faces which are
//                    visible to the camera.  This assumes that vertices of
//                    front-facing triangles are ordered in an anti-clockwise
//                    fashion, and triangles that face away from the camera are
//                    in a clockwise order relative to the current view
//                    direction. NOTE: This will only work if the mesh faces are
//                    consistently defined with counter-clockwise ordering when
//                    viewed from the outside.
//
// Returns:
//    A 4 element tuple of:
//    pix_to_face: int64 tensor of shape (N, H, W, K) giving the face index of
//                 each of the closest faces to the pixel in the rasterized
//                 image, or -1 for pixels that are not covered by any face.
//    zbuf: float32 Tensor of shape (N, H, W, K) giving the depth of each of
//          the closest faces for each pixel.
//    barycentric_coords: float tensor of shape (N, H, W, K, 3) giving
//                        barycentric coordinates of the pixel with respect to
//                        each of the closest faces along the z axis, padded
//                        with -1 for pixels hit by fewer than
//                        faces_per_pixel faces.
//    dists: float tensor of shape (N, H, W, K) giving the euclidean distance
//           in the (NDC) x/y plane between each pixel and its K closest
//           faces along the z axis padded  with -1 for pixels hit by fewer than
//           faces_per_pixel faces.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeMeshes(
    const torch::Tensor& face_verts,
    const torch::Tensor& mesh_to_face_first_idx,
    const torch::Tensor& num_faces_per_mesh,
    const int image_size,
    const float blur_radius,
    const int faces_per_pixel,
    const int bin_size,
    const int max_faces_per_bin,
    const bool perspective_correct,
    const bool cull_backfaces) {
  if (bin_size > 0 && max_faces_per_bin > 0) {
    // Use coarse-to-fine rasterization
    auto bin_faces = RasterizeMeshesCoarse(
        face_verts,
        mesh_to_face_first_idx,
        num_faces_per_mesh,
        image_size,
        blur_radius,
        bin_size,
        max_faces_per_bin);
    return RasterizeMeshesFine(
        face_verts,
        bin_faces,
        image_size,
        blur_radius,
        bin_size,
        faces_per_pixel,
        perspective_correct,
        cull_backfaces);
  } else {
    // Use the naive per-pixel implementation
    return RasterizeMeshesNaive(
        face_verts,
        mesh_to_face_first_idx,
        num_faces_per_mesh,
        image_size,
        blur_radius,
        faces_per_pixel,
        perspective_correct,
        cull_backfaces);
  }
}
