/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "./renderer.h"
#include "../include/commands.h"
#include "./camera.h"
#include "./util.h"

#include <ATen/ATen.h>
#ifdef WITH_CUDA
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#endif

#ifndef TORCH_CHECK_ARG
// torch <= 1.10
#define TORCH_CHECK_ARG(cond, argN, ...) \
  TORCH_CHECK(cond, "invalid argument ", argN, ": ", __VA_ARGS__)
#endif

namespace PRE = ::pulsar::Renderer;

namespace pulsar {
namespace pytorch {

Renderer::Renderer(
    const unsigned int& width,
    const unsigned int& height,
    const unsigned int& max_n_balls,
    const bool& orthogonal_projection,
    const bool& right_handed_system,
    const float& background_normalization_depth,
    const uint& n_channels,
    const uint& n_track) {
  LOG_IF(INFO, PULSAR_LOG_INIT) << "Initializing renderer.";
  TORCH_CHECK_ARG(width > 0, 1, "image width must be > 0!");
  TORCH_CHECK_ARG(height > 0, 2, "image height must be > 0!");
  TORCH_CHECK_ARG(max_n_balls > 0, 3, "max_n_balls must be > 0!");
  TORCH_CHECK_ARG(
      background_normalization_depth > 0.f &&
          background_normalization_depth < 1.f,
      5,
      "background_normalization_depth must be in ]0., 1.[");
  TORCH_CHECK_ARG(n_channels > 0, 6, "n_channels must be > 0");
  TORCH_CHECK_ARG(
      n_track > 0 && n_track <= MAX_GRAD_SPHERES,
      7,
      ("n_track must be > 0 and <" + std::to_string(MAX_GRAD_SPHERES) +
       ". Is " + std::to_string(n_track) + ".")
          .c_str());
  LOG_IF(INFO, PULSAR_LOG_INIT)
      << "Image width: " << width << ", height: " << height;
  this->renderer_vec.emplace_back();
  this->device_type = c10::DeviceType::CPU;
  this->device_index = -1;
  PRE::construct<false>(
      this->renderer_vec.data(),
      max_n_balls,
      width,
      height,
      orthogonal_projection,
      right_handed_system,
      background_normalization_depth,
      n_channels,
      n_track);
  this->device_tracker = torch::zeros(1);
};

Renderer::~Renderer() {
  if (this->device_type == c10::DeviceType::CUDA) {
// Can't happen in the case that not compiled with CUDA.
#ifdef WITH_CUDA
    at::cuda::CUDAGuard device_guard(this->device_tracker.device());
    for (auto nrend : this->renderer_vec) {
      PRE::destruct<true>(&nrend);
    }
#endif
  } else {
    for (auto nrend : this->renderer_vec) {
      PRE::destruct<false>(&nrend);
    }
  }
}

bool Renderer::operator==(const Renderer& rhs) const {
  LOG_IF(INFO, PULSAR_LOG_INIT) << "Equality check.";
  bool renderer_agrees = (this->renderer_vec[0] == rhs.renderer_vec[0]);
  LOG_IF(INFO, PULSAR_LOG_INIT) << "  Renderer agrees: " << renderer_agrees;
  bool device_agrees =
      (this->device_tracker.device() == rhs.device_tracker.device());
  LOG_IF(INFO, PULSAR_LOG_INIT) << "  Device agrees: " << device_agrees;
  return (renderer_agrees && device_agrees);
};

void Renderer::ensure_on_device(torch::Device device, bool /*non_blocking*/) {
  TORCH_CHECK_ARG(
      device.type() == c10::DeviceType::CUDA ||
          device.type() == c10::DeviceType::CPU,
      1,
      "Only CPU and CUDA device types are supported.");
  if (device.type() != this->device_type ||
      device.index() != this->device_index) {
#ifdef WITH_CUDA
    LOG_IF(INFO, PULSAR_LOG_INIT)
        << "Transferring render buffers between devices.";
    int prev_active;
    cudaGetDevice(&prev_active);
    if (this->device_type == c10::DeviceType::CUDA) {
      LOG_IF(INFO, PULSAR_LOG_INIT) << "  Destructing on CUDA.";
      cudaSetDevice(this->device_index);
      for (auto& nrend : this->renderer_vec) {
        PRE::destruct<true>(&nrend);
      }
    } else {
      LOG_IF(INFO, PULSAR_LOG_INIT) << "  Destructing on CPU.";
      for (auto& nrend : this->renderer_vec) {
        PRE::destruct<false>(&nrend);
      }
    }
    if (device.type() == c10::DeviceType::CUDA) {
      LOG_IF(INFO, PULSAR_LOG_INIT) << "  Constructing on CUDA.";
      cudaSetDevice(device.index());
      for (auto& nrend : this->renderer_vec) {
        PRE::construct<true>(
            &nrend,
            this->renderer_vec[0].max_num_balls,
            this->renderer_vec[0].cam.film_width,
            this->renderer_vec[0].cam.film_height,
            this->renderer_vec[0].cam.orthogonal_projection,
            this->renderer_vec[0].cam.right_handed,
            this->renderer_vec[0].cam.background_normalization_depth,
            this->renderer_vec[0].cam.n_channels,
            this->n_track());
      }
    } else {
      LOG_IF(INFO, PULSAR_LOG_INIT) << "  Constructing on CPU.";
      for (auto& nrend : this->renderer_vec) {
        PRE::construct<false>(
            &nrend,
            this->renderer_vec[0].max_num_balls,
            this->renderer_vec[0].cam.film_width,
            this->renderer_vec[0].cam.film_height,
            this->renderer_vec[0].cam.orthogonal_projection,
            this->renderer_vec[0].cam.right_handed,
            this->renderer_vec[0].cam.background_normalization_depth,
            this->renderer_vec[0].cam.n_channels,
            this->n_track());
      }
    }
    cudaSetDevice(prev_active);
    this->device_type = device.type();
    this->device_index = device.index();
#else
    throw std::runtime_error(
        "pulsar was built without CUDA "
        "but a device move to a CUDA device was initiated.");
#endif
  }
};

void Renderer::ensure_n_renderers_gte(const size_t& batch_size) {
  if (this->renderer_vec.size() < batch_size) {
    ptrdiff_t diff = batch_size - this->renderer_vec.size();
    LOG_IF(INFO, PULSAR_LOG_INIT)
        << "Increasing render buffers by " << diff
        << " to account for batch size " << batch_size;
    for (ptrdiff_t i = 0; i < diff; ++i) {
      this->renderer_vec.emplace_back();
      if (this->device_type == c10::DeviceType::CUDA) {
#ifdef WITH_CUDA
        PRE::construct<true>(
            &this->renderer_vec[this->renderer_vec.size() - 1],
            this->max_num_balls(),
            this->width(),
            this->height(),
            this->renderer_vec[0].cam.orthogonal_projection,
            this->renderer_vec[0].cam.right_handed,
            this->renderer_vec[0].cam.background_normalization_depth,
            this->renderer_vec[0].cam.n_channels,
            this->n_track());
#endif
      } else {
        PRE::construct<false>(
            &this->renderer_vec[this->renderer_vec.size() - 1],
            this->max_num_balls(),
            this->width(),
            this->height(),
            this->renderer_vec[0].cam.orthogonal_projection,
            this->renderer_vec[0].cam.right_handed,
            this->renderer_vec[0].cam.background_normalization_depth,
            this->renderer_vec[0].cam.n_channels,
            this->n_track());
      }
    }
  }
}

std::tuple<size_t, size_t, bool, torch::Tensor> Renderer::arg_check(
    const torch::Tensor& vert_pos,
    const torch::Tensor& vert_col,
    const torch::Tensor& vert_radii,
    const torch::Tensor& cam_pos,
    const torch::Tensor& pixel_0_0_center,
    const torch::Tensor& pixel_vec_x,
    const torch::Tensor& pixel_vec_y,
    const torch::Tensor& focal_length,
    const torch::Tensor& principal_point_offsets,
    const float& gamma,
    const float& max_depth,
    float& min_depth,
    const std::optional<torch::Tensor>& bg_col,
    const std::optional<torch::Tensor>& opacity,
    const float& percent_allowed_difference,
    const uint& max_n_hits,
    const uint& mode) {
  LOG_IF(INFO, PULSAR_LOG_FORWARD || PULSAR_LOG_BACKWARD) << "Arg check.";
  size_t batch_size = 1;
  size_t n_points;
  bool batch_processing = false;
  if (vert_pos.ndimension() == 3) {
    // Check all parameters adhere batch size.
    batch_processing = true;
    batch_size = vert_pos.size(0);
    TORCH_CHECK_ARG(
        vert_col.ndimension() == 3 &&
            vert_col.size(0) == static_cast<int64_t>(batch_size),
        2,
        "vert_col needs to have batch size.");
    TORCH_CHECK_ARG(
        vert_radii.ndimension() == 2 &&
            vert_radii.size(0) == static_cast<int64_t>(batch_size),
        3,
        "vert_radii must be specified per batch.");
    TORCH_CHECK_ARG(
        cam_pos.ndimension() == 2 &&
            cam_pos.size(0) == static_cast<int64_t>(batch_size),
        4,
        "cam_pos must be specified per batch and have the correct batch size.");
    TORCH_CHECK_ARG(
        pixel_0_0_center.ndimension() == 2 &&
            pixel_0_0_center.size(0) == static_cast<int64_t>(batch_size),
        5,
        "pixel_0_0_center must be specified per batch.");
    TORCH_CHECK_ARG(
        pixel_vec_x.ndimension() == 2 &&
            pixel_vec_x.size(0) == static_cast<int64_t>(batch_size),
        6,
        "pixel_vec_x must be specified per batch.");
    TORCH_CHECK_ARG(
        pixel_vec_y.ndimension() == 2 &&
            pixel_vec_y.size(0) == static_cast<int64_t>(batch_size),
        7,
        "pixel_vec_y must be specified per batch.");
    TORCH_CHECK_ARG(
        focal_length.ndimension() == 1 &&
            focal_length.size(0) == static_cast<int64_t>(batch_size),
        8,
        "focal_length must be specified per batch.");
    TORCH_CHECK_ARG(
        principal_point_offsets.ndimension() == 2 &&
            principal_point_offsets.size(0) == static_cast<int64_t>(batch_size),
        9,
        "principal_point_offsets must be specified per batch.");
    if (opacity.has_value()) {
      TORCH_CHECK_ARG(
          opacity.value().ndimension() == 2 &&
              opacity.value().size(0) == static_cast<int64_t>(batch_size),
          13,
          "Opacity needs to be specified batch-wise.");
    }
    // Check all parameters are for a matching number of points.
    n_points = vert_pos.size(1);
    TORCH_CHECK_ARG(
        vert_col.size(1) == static_cast<int64_t>(n_points),
        2,
        ("The number of points for vertex positions (" +
         std::to_string(n_points) + ") and vertex colors (" +
         std::to_string(vert_col.size(1)) + ") doesn't agree.")
            .c_str());
    TORCH_CHECK_ARG(
        vert_radii.size(1) == static_cast<int64_t>(n_points),
        3,
        ("The number of points for vertex positions (" +
         std::to_string(n_points) + ") and vertex radii (" +
         std::to_string(vert_col.size(1)) + ") doesn't agree.")
            .c_str());
    if (opacity.has_value()) {
      TORCH_CHECK_ARG(
          opacity.value().size(1) == static_cast<int64_t>(n_points),
          13,
          "Opacity needs to be specified per point.");
    }
    // Check all parameters have the correct last dimension size.
    TORCH_CHECK_ARG(
        vert_pos.size(2) == 3,
        1,
        ("Vertex positions must be 3D (have shape " +
         std::to_string(vert_pos.size(2)) + ")!")
            .c_str());
    TORCH_CHECK_ARG(
        vert_col.size(2) == this->renderer_vec[0].cam.n_channels,
        2,
        ("Vertex colors must have the right number of channels (have shape " +
         std::to_string(vert_col.size(2)) + ", need " +
         std::to_string(this->renderer_vec[0].cam.n_channels) + ")!")
            .c_str());
    TORCH_CHECK_ARG(
        cam_pos.size(1) == 3,
        4,
        ("Camera position must be 3D (has shape " +
         std::to_string(cam_pos.size(1)) + ")!")
            .c_str());
    TORCH_CHECK_ARG(
        pixel_0_0_center.size(1) == 3,
        5,
        ("pixel_0_0_center must be 3D (has shape " +
         std::to_string(pixel_0_0_center.size(1)) + ")!")
            .c_str());
    TORCH_CHECK_ARG(
        pixel_vec_x.size(1) == 3,
        6,
        ("pixel_vec_x must be 3D (has shape " +
         std::to_string(pixel_vec_x.size(1)) + ")!")
            .c_str());
    TORCH_CHECK_ARG(
        pixel_vec_y.size(1) == 3,
        7,
        ("pixel_vec_y must be 3D (has shape " +
         std::to_string(pixel_vec_y.size(1)) + ")!")
            .c_str());
    TORCH_CHECK_ARG(
        principal_point_offsets.size(1) == 2,
        9,
        "principal_point_offsets must contain x and y offsets.");
    // Ensure enough renderers are available for the batch.
    ensure_n_renderers_gte(batch_size);
  } else {
    // Check all parameters are of correct dimension.
    TORCH_CHECK_ARG(
        vert_col.ndimension() == 2, 2, "vert_col needs to have dimension 2.");
    TORCH_CHECK_ARG(
        vert_radii.ndimension() == 1, 3, "vert_radii must have dimension 1.");
    TORCH_CHECK_ARG(
        cam_pos.ndimension() == 1, 4, "cam_pos must have dimension 1.");
    TORCH_CHECK_ARG(
        pixel_0_0_center.ndimension() == 1,
        5,
        "pixel_0_0_center must have dimension 1.");
    TORCH_CHECK_ARG(
        pixel_vec_x.ndimension() == 1, 6, "pixel_vec_x must have dimension 1.");
    TORCH_CHECK_ARG(
        pixel_vec_y.ndimension() == 1, 7, "pixel_vec_y must have dimension 1.");
    TORCH_CHECK_ARG(
        focal_length.ndimension() == 0,
        8,
        "focal_length must have dimension 0.");
    TORCH_CHECK_ARG(
        principal_point_offsets.ndimension() == 1,
        9,
        "principal_point_offsets must have dimension 1.");
    if (opacity.has_value()) {
      TORCH_CHECK_ARG(
          opacity.value().ndimension() == 1,
          13,
          "Opacity needs to be specified per sample.");
    }
    // Check each.
    n_points = vert_pos.size(0);
    TORCH_CHECK_ARG(
        vert_col.size(0) == static_cast<int64_t>(n_points),
        2,
        ("The number of points for vertex positions (" +
         std::to_string(n_points) + ") and vertex colors (" +
         std::to_string(vert_col.size(0)) + ") doesn't agree.")
            .c_str());
    TORCH_CHECK_ARG(
        vert_radii.size(0) == static_cast<int64_t>(n_points),
        3,
        ("The number of points for vertex positions (" +
         std::to_string(n_points) + ") and vertex radii (" +
         std::to_string(vert_col.size(0)) + ") doesn't agree.")
            .c_str());
    if (opacity.has_value()) {
      TORCH_CHECK_ARG(
          opacity.value().size(0) == static_cast<int64_t>(n_points),
          12,
          "Opacity needs to be specified per point.");
    }
    // Check all parameters have the correct last dimension size.
    TORCH_CHECK_ARG(
        vert_pos.size(1) == 3,
        1,
        ("Vertex positions must be 3D (have shape " +
         std::to_string(vert_pos.size(1)) + ")!")
            .c_str());
    TORCH_CHECK_ARG(
        vert_col.size(1) == this->renderer_vec[0].cam.n_channels,
        2,
        ("Vertex colors must have the right number of channels (have shape " +
         std::to_string(vert_col.size(1)) + ", need " +
         std::to_string(this->renderer_vec[0].cam.n_channels) + ")!")
            .c_str());
    TORCH_CHECK_ARG(
        cam_pos.size(0) == 3,
        4,
        ("Camera position must be 3D (has shape " +
         std::to_string(cam_pos.size(0)) + ")!")
            .c_str());
    TORCH_CHECK_ARG(
        pixel_0_0_center.size(0) == 3,
        5,
        ("pixel_0_0_center must be 3D (has shape " +
         std::to_string(pixel_0_0_center.size(0)) + ")!")
            .c_str());
    TORCH_CHECK_ARG(
        pixel_vec_x.size(0) == 3,
        6,
        ("pixel_vec_x must be 3D (has shape " +
         std::to_string(pixel_vec_x.size(0)) + ")!")
            .c_str());
    TORCH_CHECK_ARG(
        pixel_vec_y.size(0) == 3,
        7,
        ("pixel_vec_y must be 3D (has shape " +
         std::to_string(pixel_vec_y.size(0)) + ")!")
            .c_str());
    TORCH_CHECK_ARG(
        principal_point_offsets.size(0) == 2,
        9,
        "principal_point_offsets must have x and y component.");
  }
  // Check device placement.
  auto dev = torch::device_of(vert_pos).value();
  TORCH_CHECK_ARG(
      dev.type() == this->device_type && dev.index() == this->device_index,
      1,
      ("Vertex positions must be stored on device " +
       c10::DeviceTypeName(this->device_type) + ", index " +
       std::to_string(this->device_index) + "! Are stored on " +
       c10::DeviceTypeName(dev.type()) + ", index " +
       std::to_string(dev.index()) + ".")
          .c_str());
  dev = torch::device_of(vert_col).value();
  TORCH_CHECK_ARG(
      dev.type() == this->device_type && dev.index() == this->device_index,
      2,
      ("Vertex colors must be stored on device " +
       c10::DeviceTypeName(this->device_type) + ", index " +
       std::to_string(this->device_index) + "! Are stored on " +
       c10::DeviceTypeName(dev.type()) + ", index " +
       std::to_string(dev.index()) + ".")
          .c_str());
  dev = torch::device_of(vert_radii).value();
  TORCH_CHECK_ARG(
      dev.type() == this->device_type && dev.index() == this->device_index,
      3,
      ("Vertex radii must be stored on device " +
       c10::DeviceTypeName(this->device_type) + ", index " +
       std::to_string(this->device_index) + "! Are stored on " +
       c10::DeviceTypeName(dev.type()) + ", index " +
       std::to_string(dev.index()) + ".")
          .c_str());
  dev = torch::device_of(cam_pos).value();
  TORCH_CHECK_ARG(
      dev.type() == this->device_type && dev.index() == this->device_index,
      4,
      ("Camera position must be stored on device " +
       c10::DeviceTypeName(this->device_type) + ", index " +
       std::to_string(this->device_index) + "! Are stored on " +
       c10::DeviceTypeName(dev.type()) + ", index " +
       std::to_string(dev.index()) + ".")
          .c_str());
  dev = torch::device_of(pixel_0_0_center).value();
  TORCH_CHECK_ARG(
      dev.type() == this->device_type && dev.index() == this->device_index,
      5,
      ("pixel_0_0_center must be stored on device " +
       c10::DeviceTypeName(this->device_type) + ", index " +
       std::to_string(this->device_index) + "! Are stored on " +
       c10::DeviceTypeName(dev.type()) + ", index " +
       std::to_string(dev.index()) + ".")
          .c_str());
  dev = torch::device_of(pixel_vec_x).value();
  TORCH_CHECK_ARG(
      dev.type() == this->device_type && dev.index() == this->device_index,
      6,
      ("pixel_vec_x must be stored on device " +
       c10::DeviceTypeName(this->device_type) + ", index " +
       std::to_string(this->device_index) + "! Are stored on " +
       c10::DeviceTypeName(dev.type()) + ", index " +
       std::to_string(dev.index()) + ".")
          .c_str());
  dev = torch::device_of(pixel_vec_y).value();
  TORCH_CHECK_ARG(
      dev.type() == this->device_type && dev.index() == this->device_index,
      7,
      ("pixel_vec_y must be stored on device " +
       c10::DeviceTypeName(this->device_type) + ", index " +
       std::to_string(this->device_index) + "! Are stored on " +
       c10::DeviceTypeName(dev.type()) + ", index " +
       std::to_string(dev.index()) + ".")
          .c_str());
  dev = torch::device_of(principal_point_offsets).value();
  TORCH_CHECK_ARG(
      dev.type() == this->device_type && dev.index() == this->device_index,
      9,
      ("principal_point_offsets must be stored on device " +
       c10::DeviceTypeName(this->device_type) + ", index " +
       std::to_string(this->device_index) + "! Are stored on " +
       c10::DeviceTypeName(dev.type()) + ", index " +
       std::to_string(dev.index()) + ".")
          .c_str());
  if (opacity.has_value()) {
    dev = torch::device_of(opacity.value()).value();
    TORCH_CHECK_ARG(
        dev.type() == this->device_type && dev.index() == this->device_index,
        13,
        ("opacity must be stored on device " +
         c10::DeviceTypeName(this->device_type) + ", index " +
         std::to_string(this->device_index) + "! Is stored on " +
         c10::DeviceTypeName(dev.type()) + ", index " +
         std::to_string(dev.index()) + ".")
            .c_str());
  }
  // Type checks.
  TORCH_CHECK_ARG(
      vert_pos.scalar_type() == c10::kFloat, 1, "pulsar requires float types.");
  TORCH_CHECK_ARG(
      vert_col.scalar_type() == c10::kFloat, 2, "pulsar requires float types.");
  TORCH_CHECK_ARG(
      vert_radii.scalar_type() == c10::kFloat,
      3,
      "pulsar requires float types.");
  TORCH_CHECK_ARG(
      cam_pos.scalar_type() == c10::kFloat, 4, "pulsar requires float types.");
  TORCH_CHECK_ARG(
      pixel_0_0_center.scalar_type() == c10::kFloat,
      5,
      "pulsar requires float types.");
  TORCH_CHECK_ARG(
      pixel_vec_x.scalar_type() == c10::kFloat,
      6,
      "pulsar requires float types.");
  TORCH_CHECK_ARG(
      pixel_vec_y.scalar_type() == c10::kFloat,
      7,
      "pulsar requires float types.");
  TORCH_CHECK_ARG(
      focal_length.scalar_type() == c10::kFloat,
      8,
      "pulsar requires float types.");
  TORCH_CHECK_ARG(
      // Unfortunately, the PyTorch interface is inconsistent for
      // Int32: in Python, there exists an explicit int32 type, in
      // C++ this is currently `c10::kInt`.
      principal_point_offsets.scalar_type() == c10::kInt,
      9,
      "principal_point_offsets must be provided as int32.");
  if (opacity.has_value()) {
    TORCH_CHECK_ARG(
        opacity.value().scalar_type() == c10::kFloat,
        13,
        "opacity must be a float type.");
  }
  // Content checks.
  TORCH_CHECK_ARG(
      (vert_radii > FEPS).all().item<bool>(),
      3,
      ("Vertex radii must be > FEPS (min is " +
       std::to_string(vert_radii.min().item<float>()) + ").")
          .c_str());
  if (this->orthogonal()) {
    TORCH_CHECK_ARG(
        (focal_length == 0.f).all().item<bool>(),
        8,
        ("for an orthogonal projection focal length must be zero (abs max: " +
         std::to_string(focal_length.abs().max().item<float>()) + ").")
            .c_str());
  } else {
    TORCH_CHECK_ARG(
        (focal_length > FEPS).all().item<bool>(),
        8,
        ("for a perspective projection focal length must be > FEPS (min " +
         std::to_string(focal_length.min().item<float>()) + ").")
            .c_str());
  }
  TORCH_CHECK_ARG(
      gamma <= 1.f && gamma >= 1E-5f,
      10,
      ("gamma must be in [1E-5, 1] (" + std::to_string(gamma) + ").").c_str());
  if (min_depth == 0.f) {
    min_depth = focal_length.max().item<float>() + 2.f * FEPS;
  }
  TORCH_CHECK_ARG(
      min_depth > focal_length.max().item<float>(),
      12,
      ("min_depth must be > focal_length (" + std::to_string(min_depth) +
       " vs. " + std::to_string(focal_length.max().item<float>()) + ").")
          .c_str());
  TORCH_CHECK_ARG(
      max_depth > min_depth + FEPS,
      11,
      ("max_depth must be > min_depth + FEPS (" + std::to_string(max_depth) +
       " vs. " + std::to_string(min_depth + FEPS) + ").")
          .c_str());
  TORCH_CHECK_ARG(
      percent_allowed_difference >= 0.f && percent_allowed_difference < 1.f,
      14,
      ("percent_allowed_difference must be in [0., 1.[ (" +
       std::to_string(percent_allowed_difference) + ").")
          .c_str());
  TORCH_CHECK_ARG(max_n_hits > 0, 14, "max_n_hits must be > 0!");
  TORCH_CHECK_ARG(mode < 2, 15, "mode must be in {0, 1}.");
  torch::Tensor real_bg_col;
  if (bg_col.has_value()) {
    TORCH_CHECK_ARG(
        bg_col.value().device().type() == this->device_type &&
            bg_col.value().device().index() == this->device_index,
        13,
        "bg_col must be stored on the renderer device!");
    TORCH_CHECK_ARG(
        bg_col.value().ndimension() == 1 &&
            bg_col.value().size(0) == renderer_vec[0].cam.n_channels,
        13,
        "bg_col must have the same number of channels as the image,).");
    real_bg_col = bg_col.value();
  } else {
    real_bg_col = torch::ones(
                      {renderer_vec[0].cam.n_channels},
                      c10::Device(this->device_type, this->device_index))
                      .to(c10::kFloat);
  }
  if (opacity.has_value()) {
    TORCH_CHECK_ARG(
        (opacity.value() >= 0.f).all().item<bool>(),
        13,
        "opacity must be >= 0.");
    TORCH_CHECK_ARG(
        (opacity.value() <= 1.f).all().item<bool>(),
        13,
        "opacity must be <= 1.");
  }
  LOG_IF(INFO, PULSAR_LOG_FORWARD || PULSAR_LOG_BACKWARD)
      << "  batch_size: " << batch_size;
  LOG_IF(INFO, PULSAR_LOG_FORWARD || PULSAR_LOG_BACKWARD)
      << "  n_points: " << n_points;
  LOG_IF(INFO, PULSAR_LOG_FORWARD || PULSAR_LOG_BACKWARD)
      << "  batch_processing: " << batch_processing;
  return std::tuple<size_t, size_t, bool, torch::Tensor>(
      batch_size, n_points, batch_processing, real_bg_col);
}

std::tuple<torch::Tensor, torch::Tensor> Renderer::forward(
    const torch::Tensor& vert_pos,
    const torch::Tensor& vert_col,
    const torch::Tensor& vert_radii,
    const torch::Tensor& cam_pos,
    const torch::Tensor& pixel_0_0_center,
    const torch::Tensor& pixel_vec_x,
    const torch::Tensor& pixel_vec_y,
    const torch::Tensor& focal_length,
    const torch::Tensor& principal_point_offsets,
    const float& gamma,
    const float& max_depth,
    float min_depth,
    const std::optional<torch::Tensor>& bg_col,
    const std::optional<torch::Tensor>& opacity,
    const float& percent_allowed_difference,
    const uint& max_n_hits,
    const uint& mode) {
  // Parameter checks.
  this->ensure_on_device(this->device_tracker.device());
  size_t batch_size;
  size_t n_points;
  bool batch_processing;
  torch::Tensor real_bg_col;
  std::tie(batch_size, n_points, batch_processing, real_bg_col) =
      this->arg_check(
          vert_pos,
          vert_col,
          vert_radii,
          cam_pos,
          pixel_0_0_center,
          pixel_vec_x,
          pixel_vec_y,
          focal_length,
          principal_point_offsets,
          gamma,
          max_depth,
          min_depth,
          bg_col,
          opacity,
          percent_allowed_difference,
          max_n_hits,
          mode);
  LOG_IF(INFO, PULSAR_LOG_FORWARD) << "Extracting camera objects...";
  // Create the camera information.
  std::vector<CamInfo> cam_infos(batch_size);
  if (batch_processing) {
    for (size_t batch_i = 0; batch_i < batch_size; ++batch_i) {
      cam_infos[batch_i] = cam_info_from_params(
          cam_pos[batch_i],
          pixel_0_0_center[batch_i],
          pixel_vec_x[batch_i],
          pixel_vec_y[batch_i],
          principal_point_offsets[batch_i],
          focal_length[batch_i].item<float>(),
          this->renderer_vec[0].cam.film_width,
          this->renderer_vec[0].cam.film_height,
          min_depth,
          max_depth,
          this->renderer_vec[0].cam.right_handed);
    }
  } else {
    cam_infos[0] = cam_info_from_params(
        cam_pos,
        pixel_0_0_center,
        pixel_vec_x,
        pixel_vec_y,
        principal_point_offsets,
        focal_length.item<float>(),
        this->renderer_vec[0].cam.film_width,
        this->renderer_vec[0].cam.film_height,
        min_depth,
        max_depth,
        this->renderer_vec[0].cam.right_handed);
  }
  LOG_IF(INFO, PULSAR_LOG_FORWARD) << "Processing...";
  // Let's go!
  // Contiguous version of opacity, if available. We need to create this object
  // in scope to keep it alive.
  torch::Tensor opacity_contiguous;
  float const* opacity_ptr = nullptr;
  if (opacity.has_value()) {
    opacity_contiguous = opacity.value().contiguous();
    opacity_ptr = opacity_contiguous.data_ptr<float>();
  }
  if (this->device_type == c10::DeviceType::CUDA) {
// No else check necessary - if not compiled with CUDA
// we can't even reach this code (the renderer can't be
// moved to a CUDA device).
#ifdef WITH_CUDA
    int prev_active;
    cudaGetDevice(&prev_active);
    cudaSetDevice(this->device_index);
#ifdef PULSAR_TIMINGS_BATCHED_ENABLED
    START_TIME_CU(batch_forward);
#endif
    if (batch_processing) {
      for (size_t batch_i = 0; batch_i < batch_size; ++batch_i) {
        // These calls are non-blocking and just kick off the computations.
        PRE::forward<true>(
            &this->renderer_vec[batch_i],
            vert_pos[batch_i].contiguous().data_ptr<float>(),
            vert_col[batch_i].contiguous().data_ptr<float>(),
            vert_radii[batch_i].contiguous().data_ptr<float>(),
            cam_infos[batch_i],
            gamma,
            percent_allowed_difference,
            max_n_hits,
            real_bg_col.contiguous().data_ptr<float>(),
            opacity_ptr,
            n_points,
            mode,
            at::cuda::getCurrentCUDAStream());
      }
    } else {
      PRE::forward<true>(
          this->renderer_vec.data(),
          vert_pos.contiguous().data_ptr<float>(),
          vert_col.contiguous().data_ptr<float>(),
          vert_radii.contiguous().data_ptr<float>(),
          cam_infos[0],
          gamma,
          percent_allowed_difference,
          max_n_hits,
          real_bg_col.contiguous().data_ptr<float>(),
          opacity_ptr,
          n_points,
          mode,
          at::cuda::getCurrentCUDAStream());
    }
#ifdef PULSAR_TIMINGS_BATCHED_ENABLED
    STOP_TIME_CU(batch_forward);
    float time_ms;
    GET_TIME_CU(batch_forward, &time_ms);
    std::cout << "Forward render batched time per example: "
              << time_ms / static_cast<float>(batch_size) << "ms" << std::endl;
#endif
    cudaSetDevice(prev_active);
#endif
  } else {
#ifdef PULSAR_TIMINGS_BATCHED_ENABLED
    START_TIME(batch_forward);
#endif
    if (batch_processing) {
      for (size_t batch_i = 0; batch_i < batch_size; ++batch_i) {
        // These calls are non-blocking and just kick off the computations.
        PRE::forward<false>(
            &this->renderer_vec[batch_i],
            vert_pos[batch_i].contiguous().data_ptr<float>(),
            vert_col[batch_i].contiguous().data_ptr<float>(),
            vert_radii[batch_i].contiguous().data_ptr<float>(),
            cam_infos[batch_i],
            gamma,
            percent_allowed_difference,
            max_n_hits,
            real_bg_col.contiguous().data_ptr<float>(),
            opacity_ptr,
            n_points,
            mode,
            nullptr);
      }
    } else {
      PRE::forward<false>(
          this->renderer_vec.data(),
          vert_pos.contiguous().data_ptr<float>(),
          vert_col.contiguous().data_ptr<float>(),
          vert_radii.contiguous().data_ptr<float>(),
          cam_infos[0],
          gamma,
          percent_allowed_difference,
          max_n_hits,
          real_bg_col.contiguous().data_ptr<float>(),
          opacity_ptr,
          n_points,
          mode,
          nullptr);
    }
#ifdef PULSAR_TIMINGS_BATCHED_ENABLED
    STOP_TIME(batch_forward);
    float time_ms;
    GET_TIME(batch_forward, &time_ms);
    std::cout << "Forward render batched time per example: "
              << time_ms / static_cast<float>(batch_size) << "ms" << std::endl;
#endif
  }
  LOG_IF(INFO, PULSAR_LOG_FORWARD) << "Extracting results...";
  // Create the results.
  std::vector<torch::Tensor> results(batch_size);
  std::vector<torch::Tensor> forw_infos(batch_size);
  for (size_t batch_i = 0; batch_i < batch_size; ++batch_i) {
    results[batch_i] = from_blob(
        this->renderer_vec[batch_i].result_d,
        {this->renderer_vec[0].cam.film_height,
         this->renderer_vec[0].cam.film_width,
         this->renderer_vec[0].cam.n_channels},
        this->device_type,
        this->device_index,
        torch::kFloat,
        this->device_type == c10::DeviceType::CUDA
#ifdef WITH_CUDA
            ? at::cuda::getCurrentCUDAStream()
#else
            ? (cudaStream_t) nullptr
#endif
            : (cudaStream_t) nullptr);
    if (mode == 1)
      results[batch_i] = results[batch_i].slice(2, 0, 1, 1);
    forw_infos[batch_i] = from_blob(
        this->renderer_vec[batch_i].forw_info_d,
        {this->renderer_vec[0].cam.film_height,
         this->renderer_vec[0].cam.film_width,
         3 + 2 * this->n_track()},
        this->device_type,
        this->device_index,
        torch::kFloat,
        this->device_type == c10::DeviceType::CUDA
#ifdef WITH_CUDA
            ? at::cuda::getCurrentCUDAStream()
#else
            ? (cudaStream_t) nullptr
#endif
            : (cudaStream_t) nullptr);
  }
  LOG_IF(INFO, PULSAR_LOG_FORWARD) << "Forward render complete.";
  if (batch_processing) {
    return std::tuple<torch::Tensor, torch::Tensor>(
        torch::stack(results), torch::stack(forw_infos));
  } else {
    return std::tuple<torch::Tensor, torch::Tensor>(results[0], forw_infos[0]);
  }
};

std::tuple<
    std::optional<torch::Tensor>,
    std::optional<torch::Tensor>,
    std::optional<torch::Tensor>,
    std::optional<torch::Tensor>,
    std::optional<torch::Tensor>,
    std::optional<torch::Tensor>,
    std::optional<torch::Tensor>,
    std::optional<torch::Tensor>>
Renderer::backward(
    const torch::Tensor& grad_im,
    const torch::Tensor& image,
    const torch::Tensor& forw_info,
    const torch::Tensor& vert_pos,
    const torch::Tensor& vert_col,
    const torch::Tensor& vert_radii,
    const torch::Tensor& cam_pos,
    const torch::Tensor& pixel_0_0_center,
    const torch::Tensor& pixel_vec_x,
    const torch::Tensor& pixel_vec_y,
    const torch::Tensor& focal_length,
    const torch::Tensor& principal_point_offsets,
    const float& gamma,
    const float& max_depth,
    float min_depth,
    const std::optional<torch::Tensor>& bg_col,
    const std::optional<torch::Tensor>& opacity,
    const float& percent_allowed_difference,
    const uint& max_n_hits,
    const uint& mode,
    const bool& dif_pos,
    const bool& dif_col,
    const bool& dif_rad,
    const bool& dif_cam,
    const bool& dif_opy,
    const std::optional<std::pair<uint, uint>>& dbg_pos) {
  this->ensure_on_device(this->device_tracker.device());
  size_t batch_size;
  size_t n_points;
  bool batch_processing;
  torch::Tensor real_bg_col;
  std::tie(batch_size, n_points, batch_processing, real_bg_col) =
      this->arg_check(
          vert_pos,
          vert_col,
          vert_radii,
          cam_pos,
          pixel_0_0_center,
          pixel_vec_x,
          pixel_vec_y,
          focal_length,
          principal_point_offsets,
          gamma,
          max_depth,
          min_depth,
          bg_col,
          opacity,
          percent_allowed_difference,
          max_n_hits,
          mode);
  // Additional checks for the gradient computation.
  TORCH_CHECK_ARG(
      (grad_im.ndimension() == 3 + batch_processing &&
       static_cast<uint>(grad_im.size(0 + batch_processing)) ==
           this->height() &&
       static_cast<uint>(grad_im.size(1 + batch_processing)) == this->width() &&
       static_cast<uint>(grad_im.size(2 + batch_processing)) ==
           this->renderer_vec[0].cam.n_channels),
      1,
      "The gradient image size is not correct.");
  TORCH_CHECK_ARG(
      (image.ndimension() == 3 + batch_processing &&
       static_cast<uint>(image.size(0 + batch_processing)) == this->height() &&
       static_cast<uint>(image.size(1 + batch_processing)) == this->width() &&
       static_cast<uint>(image.size(2 + batch_processing)) ==
           this->renderer_vec[0].cam.n_channels),
      2,
      "The result image size is not correct.");
  TORCH_CHECK_ARG(
      grad_im.scalar_type() == c10::kFloat,
      1,
      "The gradient image must be of float type.");
  TORCH_CHECK_ARG(
      image.scalar_type() == c10::kFloat,
      2,
      "The image must be of float type.");
  if (dif_opy) {
    TORCH_CHECK_ARG(
        opacity.has_value(), 13, "dif_opy set requires opacity values.");
  }
  if (batch_processing) {
    TORCH_CHECK_ARG(
        grad_im.size(0) == static_cast<int64_t>(batch_size),
        1,
        "Gradient image batch size must agree.");
    TORCH_CHECK_ARG(
        image.size(0) == static_cast<int64_t>(batch_size),
        2,
        "Image batch size must agree.");
    TORCH_CHECK_ARG(
        forw_info.size(0) == static_cast<int64_t>(batch_size),
        3,
        "forward info must have batch size.");
  }
  TORCH_CHECK_ARG(
      (forw_info.ndimension() == 3 + batch_processing &&
       static_cast<uint>(forw_info.size(0 + batch_processing)) ==
           this->height() &&
       static_cast<uint>(forw_info.size(1 + batch_processing)) ==
           this->width() &&
       static_cast<uint>(forw_info.size(2 + batch_processing)) ==
           3 + 2 * this->n_track()),
      3,
      "The forward info image size is not correct.");
  TORCH_CHECK_ARG(
      forw_info.scalar_type() == c10::kFloat,
      3,
      "The forward info must be of float type.");
  // Check device.
  auto dev = torch::device_of(grad_im).value();
  TORCH_CHECK_ARG(
      dev.type() == this->device_type && dev.index() == this->device_index,
      1,
      ("grad_im must be stored on device " +
       c10::DeviceTypeName(this->device_type) + ", index " +
       std::to_string(this->device_index) + "! Are stored on " +
       c10::DeviceTypeName(dev.type()) + ", index " +
       std::to_string(dev.index()) + ".")
          .c_str());
  dev = torch::device_of(image).value();
  TORCH_CHECK_ARG(
      dev.type() == this->device_type && dev.index() == this->device_index,
      2,
      ("image must be stored on device " +
       c10::DeviceTypeName(this->device_type) + ", index " +
       std::to_string(this->device_index) + "! Are stored on " +
       c10::DeviceTypeName(dev.type()) + ", index " +
       std::to_string(dev.index()) + ".")
          .c_str());
  dev = torch::device_of(forw_info).value();
  TORCH_CHECK_ARG(
      dev.type() == this->device_type && dev.index() == this->device_index,
      3,
      ("forw_info must be stored on device " +
       c10::DeviceTypeName(this->device_type) + ", index " +
       std::to_string(this->device_index) + "! Are stored on " +
       c10::DeviceTypeName(dev.type()) + ", index " +
       std::to_string(dev.index()) + ".")
          .c_str());
  if (dbg_pos.has_value()) {
    TORCH_CHECK_ARG(
        dbg_pos.value().first < this->width() &&
            dbg_pos.value().second < this->height(),
        23,
        "The debug position must be within image bounds.");
  }
  // Prepare the return value.
  std::tuple<
      std::optional<torch::Tensor>,
      std::optional<torch::Tensor>,
      std::optional<torch::Tensor>,
      std::optional<torch::Tensor>,
      std::optional<torch::Tensor>,
      std::optional<torch::Tensor>,
      std::optional<torch::Tensor>,
      std::optional<torch::Tensor>>
      ret;
  if (mode == 1 || (!dif_pos && !dif_col && !dif_rad && !dif_cam && !dif_opy)) {
    return ret;
  }
  // Create the camera information.
  std::vector<CamInfo> cam_infos(batch_size);
  if (batch_processing) {
    for (size_t batch_i = 0; batch_i < batch_size; ++batch_i) {
      cam_infos[batch_i] = cam_info_from_params(
          cam_pos[batch_i],
          pixel_0_0_center[batch_i],
          pixel_vec_x[batch_i],
          pixel_vec_y[batch_i],
          principal_point_offsets[batch_i],
          focal_length[batch_i].item<float>(),
          this->renderer_vec[0].cam.film_width,
          this->renderer_vec[0].cam.film_height,
          min_depth,
          max_depth,
          this->renderer_vec[0].cam.right_handed);
    }
  } else {
    cam_infos[0] = cam_info_from_params(
        cam_pos,
        pixel_0_0_center,
        pixel_vec_x,
        pixel_vec_y,
        principal_point_offsets,
        focal_length.item<float>(),
        this->renderer_vec[0].cam.film_width,
        this->renderer_vec[0].cam.film_height,
        min_depth,
        max_depth,
        this->renderer_vec[0].cam.right_handed);
  }
  // Let's go!
  // Contiguous version of opacity, if available. We need to create this object
  // in scope to keep it alive.
  torch::Tensor opacity_contiguous;
  float const* opacity_ptr = nullptr;
  if (opacity.has_value()) {
    opacity_contiguous = opacity.value().contiguous();
    opacity_ptr = opacity_contiguous.data_ptr<float>();
  }
  if (this->device_type == c10::DeviceType::CUDA) {
// No else check necessary - it's not possible to move
// the renderer to a CUDA device if not built with CUDA.
#ifdef WITH_CUDA
    int prev_active;
    cudaGetDevice(&prev_active);
    cudaSetDevice(this->device_index);
#ifdef PULSAR_TIMINGS_BATCHED_ENABLED
    START_TIME_CU(batch_backward);
#endif
    if (batch_processing) {
      for (size_t batch_i = 0; batch_i < batch_size; ++batch_i) {
        // These calls are non-blocking and just kick off the computations.
        if (dbg_pos.has_value()) {
          PRE::backward_dbg<true>(
              &this->renderer_vec[batch_i],
              grad_im[batch_i].contiguous().data_ptr<float>(),
              image[batch_i].contiguous().data_ptr<float>(),
              forw_info[batch_i].contiguous().data_ptr<float>(),
              vert_pos[batch_i].contiguous().data_ptr<float>(),
              vert_col[batch_i].contiguous().data_ptr<float>(),
              vert_radii[batch_i].contiguous().data_ptr<float>(),
              cam_infos[batch_i],
              gamma,
              percent_allowed_difference,
              max_n_hits,
              opacity_ptr,
              n_points,
              mode,
              dif_pos,
              dif_col,
              dif_rad,
              dif_cam,
              dif_opy,
              dbg_pos.value().first,
              dbg_pos.value().second,
              at::cuda::getCurrentCUDAStream());
        } else {
          PRE::backward<true>(
              &this->renderer_vec[batch_i],
              grad_im[batch_i].contiguous().data_ptr<float>(),
              image[batch_i].contiguous().data_ptr<float>(),
              forw_info[batch_i].contiguous().data_ptr<float>(),
              vert_pos[batch_i].contiguous().data_ptr<float>(),
              vert_col[batch_i].contiguous().data_ptr<float>(),
              vert_radii[batch_i].contiguous().data_ptr<float>(),
              cam_infos[batch_i],
              gamma,
              percent_allowed_difference,
              max_n_hits,
              opacity_ptr,
              n_points,
              mode,
              dif_pos,
              dif_col,
              dif_rad,
              dif_cam,
              dif_opy,
              at::cuda::getCurrentCUDAStream());
        }
      }
    } else {
      if (dbg_pos.has_value()) {
        PRE::backward_dbg<true>(
            this->renderer_vec.data(),
            grad_im.contiguous().data_ptr<float>(),
            image.contiguous().data_ptr<float>(),
            forw_info.contiguous().data_ptr<float>(),
            vert_pos.contiguous().data_ptr<float>(),
            vert_col.contiguous().data_ptr<float>(),
            vert_radii.contiguous().data_ptr<float>(),
            cam_infos[0],
            gamma,
            percent_allowed_difference,
            max_n_hits,
            opacity_ptr,
            n_points,
            mode,
            dif_pos,
            dif_col,
            dif_rad,
            dif_cam,
            dif_opy,
            dbg_pos.value().first,
            dbg_pos.value().second,
            at::cuda::getCurrentCUDAStream());
      } else {
        PRE::backward<true>(
            this->renderer_vec.data(),
            grad_im.contiguous().data_ptr<float>(),
            image.contiguous().data_ptr<float>(),
            forw_info.contiguous().data_ptr<float>(),
            vert_pos.contiguous().data_ptr<float>(),
            vert_col.contiguous().data_ptr<float>(),
            vert_radii.contiguous().data_ptr<float>(),
            cam_infos[0],
            gamma,
            percent_allowed_difference,
            max_n_hits,
            opacity_ptr,
            n_points,
            mode,
            dif_pos,
            dif_col,
            dif_rad,
            dif_cam,
            dif_opy,
            at::cuda::getCurrentCUDAStream());
      }
    }
    cudaSetDevice(prev_active);
#ifdef PULSAR_TIMINGS_BATCHED_ENABLED
    STOP_TIME_CU(batch_backward);
    float time_ms;
    GET_TIME_CU(batch_backward, &time_ms);
    std::cout << "Backward render batched time per example: "
              << time_ms / static_cast<float>(batch_size) << "ms" << std::endl;
#endif
#endif // WITH_CUDA
  } else {
#ifdef PULSAR_TIMINGS_BATCHED_ENABLED
    START_TIME(batch_backward);
#endif
    if (batch_processing) {
      for (size_t batch_i = 0; batch_i < batch_size; ++batch_i) {
        // These calls are non-blocking and just kick off the computations.
        if (dbg_pos.has_value()) {
          PRE::backward_dbg<false>(
              &this->renderer_vec[batch_i],
              grad_im[batch_i].contiguous().data_ptr<float>(),
              image[batch_i].contiguous().data_ptr<float>(),
              forw_info[batch_i].contiguous().data_ptr<float>(),
              vert_pos[batch_i].contiguous().data_ptr<float>(),
              vert_col[batch_i].contiguous().data_ptr<float>(),
              vert_radii[batch_i].contiguous().data_ptr<float>(),
              cam_infos[batch_i],
              gamma,
              percent_allowed_difference,
              max_n_hits,
              opacity_ptr,
              n_points,
              mode,
              dif_pos,
              dif_col,
              dif_rad,
              dif_cam,
              dif_opy,
              dbg_pos.value().first,
              dbg_pos.value().second,
              nullptr);
        } else {
          PRE::backward<false>(
              &this->renderer_vec[batch_i],
              grad_im[batch_i].contiguous().data_ptr<float>(),
              image[batch_i].contiguous().data_ptr<float>(),
              forw_info[batch_i].contiguous().data_ptr<float>(),
              vert_pos[batch_i].contiguous().data_ptr<float>(),
              vert_col[batch_i].contiguous().data_ptr<float>(),
              vert_radii[batch_i].contiguous().data_ptr<float>(),
              cam_infos[batch_i],
              gamma,
              percent_allowed_difference,
              max_n_hits,
              opacity_ptr,
              n_points,
              mode,
              dif_pos,
              dif_col,
              dif_rad,
              dif_cam,
              dif_opy,
              nullptr);
        }
      }
    } else {
      if (dbg_pos.has_value()) {
        PRE::backward_dbg<false>(
            this->renderer_vec.data(),
            grad_im.contiguous().data_ptr<float>(),
            image.contiguous().data_ptr<float>(),
            forw_info.contiguous().data_ptr<float>(),
            vert_pos.contiguous().data_ptr<float>(),
            vert_col.contiguous().data_ptr<float>(),
            vert_radii.contiguous().data_ptr<float>(),
            cam_infos[0],
            gamma,
            percent_allowed_difference,
            max_n_hits,
            opacity_ptr,
            n_points,
            mode,
            dif_pos,
            dif_col,
            dif_rad,
            dif_cam,
            dif_opy,
            dbg_pos.value().first,
            dbg_pos.value().second,
            nullptr);
      } else {
        PRE::backward<false>(
            this->renderer_vec.data(),
            grad_im.contiguous().data_ptr<float>(),
            image.contiguous().data_ptr<float>(),
            forw_info.contiguous().data_ptr<float>(),
            vert_pos.contiguous().data_ptr<float>(),
            vert_col.contiguous().data_ptr<float>(),
            vert_radii.contiguous().data_ptr<float>(),
            cam_infos[0],
            gamma,
            percent_allowed_difference,
            max_n_hits,
            opacity_ptr,
            n_points,
            mode,
            dif_pos,
            dif_col,
            dif_rad,
            dif_cam,
            dif_opy,
            nullptr);
      }
    }
#ifdef PULSAR_TIMINGS_BATCHED_ENABLED
    STOP_TIME(batch_backward);
    float time_ms;
    GET_TIME(batch_backward, &time_ms);
    std::cout << "Backward render batched time per example: "
              << time_ms / static_cast<float>(batch_size) << "ms" << std::endl;
#endif
  }
  if (dif_pos) {
    if (batch_processing) {
      std::vector<torch::Tensor> results(batch_size);
      for (size_t batch_i = 0; batch_i < batch_size; ++batch_i) {
        results[batch_i] = from_blob(
            reinterpret_cast<float*>(this->renderer_vec[batch_i].grad_pos_d),
            {static_cast<ptrdiff_t>(n_points), 3},
            this->device_type,
            this->device_index,
            torch::kFloat,
            this->device_type == c10::DeviceType::CUDA
#ifdef WITH_CUDA
                ? at::cuda::getCurrentCUDAStream()
#else
                ? (cudaStream_t) nullptr
#endif
                : (cudaStream_t) nullptr);
      }
      std::get<0>(ret) = torch::stack(results);
    } else {
      std::get<0>(ret) = from_blob(
          reinterpret_cast<float*>(this->renderer_vec[0].grad_pos_d),
          {static_cast<ptrdiff_t>(n_points), 3},
          this->device_type,
          this->device_index,
          torch::kFloat,
          this->device_type == c10::DeviceType::CUDA
#ifdef WITH_CUDA
              ? at::cuda::getCurrentCUDAStream()
#else
              ? (cudaStream_t) nullptr
#endif
              : (cudaStream_t) nullptr);
    }
  }
  if (dif_col) {
    if (batch_processing) {
      std::vector<torch::Tensor> results(batch_size);
      for (size_t batch_i = 0; batch_i < batch_size; ++batch_i) {
        results[batch_i] = from_blob(
            reinterpret_cast<float*>(this->renderer_vec[batch_i].grad_col_d),
            {static_cast<ptrdiff_t>(n_points),
             this->renderer_vec[0].cam.n_channels},
            this->device_type,
            this->device_index,
            torch::kFloat,
            this->device_type == c10::DeviceType::CUDA
#ifdef WITH_CUDA
                ? at::cuda::getCurrentCUDAStream()
#else
                ? (cudaStream_t) nullptr
#endif
                : (cudaStream_t) nullptr);
      }
      std::get<1>(ret) = torch::stack(results);
    } else {
      std::get<1>(ret) = from_blob(
          reinterpret_cast<float*>(this->renderer_vec[0].grad_col_d),
          {static_cast<ptrdiff_t>(n_points),
           this->renderer_vec[0].cam.n_channels},
          this->device_type,
          this->device_index,
          torch::kFloat,
          this->device_type == c10::DeviceType::CUDA
#ifdef WITH_CUDA
              ? at::cuda::getCurrentCUDAStream()
#else
              ? (cudaStream_t) nullptr
#endif
              : (cudaStream_t) nullptr);
    }
  }
  if (dif_rad) {
    if (batch_processing) {
      std::vector<torch::Tensor> results(batch_size);
      for (size_t batch_i = 0; batch_i < batch_size; ++batch_i) {
        results[batch_i] = from_blob(
            reinterpret_cast<float*>(this->renderer_vec[batch_i].grad_rad_d),
            {static_cast<ptrdiff_t>(n_points)},
            this->device_type,
            this->device_index,
            torch::kFloat,
            this->device_type == c10::DeviceType::CUDA
#ifdef WITH_CUDA
                ? at::cuda::getCurrentCUDAStream()
#else
                ? (cudaStream_t) nullptr
#endif
                : (cudaStream_t) nullptr);
      }
      std::get<2>(ret) = torch::stack(results);
    } else {
      std::get<2>(ret) = from_blob(
          reinterpret_cast<float*>(this->renderer_vec[0].grad_rad_d),
          {static_cast<ptrdiff_t>(n_points)},
          this->device_type,
          this->device_index,
          torch::kFloat,
          this->device_type == c10::DeviceType::CUDA
#ifdef WITH_CUDA
              ? at::cuda::getCurrentCUDAStream()
#else
              ? (cudaStream_t) nullptr
#endif
              : (cudaStream_t) nullptr);
    }
  }
  if (dif_cam) {
    if (batch_processing) {
      std::vector<torch::Tensor> res_p1(batch_size);
      std::vector<torch::Tensor> res_p2(batch_size);
      std::vector<torch::Tensor> res_p3(batch_size);
      std::vector<torch::Tensor> res_p4(batch_size);
      for (size_t batch_i = 0; batch_i < batch_size; ++batch_i) {
        res_p1[batch_i] = from_blob(
            reinterpret_cast<float*>(this->renderer_vec[batch_i].grad_cam_d),
            {3},
            this->device_type,
            this->device_index,
            torch::kFloat,
            this->device_type == c10::DeviceType::CUDA
#ifdef WITH_CUDA
                ? at::cuda::getCurrentCUDAStream()
#else
                ? (cudaStream_t) nullptr
#endif
                : (cudaStream_t) nullptr);
        res_p2[batch_i] = from_blob(
            reinterpret_cast<float*>(
                this->renderer_vec[batch_i].grad_cam_d + 3),
            {3},
            this->device_type,
            this->device_index,
            torch::kFloat,
            this->device_type == c10::DeviceType::CUDA
#ifdef WITH_CUDA
                ? at::cuda::getCurrentCUDAStream()
#else
                ? (cudaStream_t) nullptr
#endif
                : (cudaStream_t) nullptr);
        res_p3[batch_i] = from_blob(
            reinterpret_cast<float*>(
                this->renderer_vec[batch_i].grad_cam_d + 6),
            {3},
            this->device_type,
            this->device_index,
            torch::kFloat,
            this->device_type == c10::DeviceType::CUDA
#ifdef WITH_CUDA
                ? at::cuda::getCurrentCUDAStream()
#else
                ? (cudaStream_t) nullptr
#endif
                : (cudaStream_t) nullptr);
        res_p4[batch_i] = from_blob(
            reinterpret_cast<float*>(
                this->renderer_vec[batch_i].grad_cam_d + 9),
            {3},
            this->device_type,
            this->device_index,
            torch::kFloat,
            this->device_type == c10::DeviceType::CUDA
#ifdef WITH_CUDA
                ? at::cuda::getCurrentCUDAStream()
#else
                ? (cudaStream_t) nullptr
#endif
                : (cudaStream_t) nullptr);
      }
      std::get<3>(ret) = torch::stack(res_p1);
      std::get<4>(ret) = torch::stack(res_p2);
      std::get<5>(ret) = torch::stack(res_p3);
      std::get<6>(ret) = torch::stack(res_p4);
    } else {
      std::get<3>(ret) = from_blob(
          reinterpret_cast<float*>(this->renderer_vec[0].grad_cam_d),
          {3},
          this->device_type,
          this->device_index,
          torch::kFloat,
          this->device_type == c10::DeviceType::CUDA
#ifdef WITH_CUDA
              ? at::cuda::getCurrentCUDAStream()
#else
              ? (cudaStream_t) nullptr
#endif
              : (cudaStream_t) nullptr);
      std::get<4>(ret) = from_blob(
          reinterpret_cast<float*>(this->renderer_vec[0].grad_cam_d + 3),
          {3},
          this->device_type,
          this->device_index,
          torch::kFloat,
          this->device_type == c10::DeviceType::CUDA
#ifdef WITH_CUDA
              ? at::cuda::getCurrentCUDAStream()
#else
              ? (cudaStream_t) nullptr
#endif
              : (cudaStream_t) nullptr);
      std::get<5>(ret) = from_blob(
          reinterpret_cast<float*>(this->renderer_vec[0].grad_cam_d + 6),
          {3},
          this->device_type,
          this->device_index,
          torch::kFloat,
          this->device_type == c10::DeviceType::CUDA
#ifdef WITH_CUDA
              ? at::cuda::getCurrentCUDAStream()
#else
              ? (cudaStream_t) nullptr
#endif
              : (cudaStream_t) nullptr);
      std::get<6>(ret) = from_blob(
          reinterpret_cast<float*>(this->renderer_vec[0].grad_cam_d + 9),
          {3},
          this->device_type,
          this->device_index,
          torch::kFloat,
          this->device_type == c10::DeviceType::CUDA
#ifdef WITH_CUDA
              ? at::cuda::getCurrentCUDAStream()
#else
              ? (cudaStream_t) nullptr
#endif
              : (cudaStream_t) nullptr);
    }
  }
  if (dif_opy) {
    if (batch_processing) {
      std::vector<torch::Tensor> results(batch_size);
      for (size_t batch_i = 0; batch_i < batch_size; ++batch_i) {
        results[batch_i] = from_blob(
            reinterpret_cast<float*>(this->renderer_vec[batch_i].grad_opy_d),
            {static_cast<ptrdiff_t>(n_points)},
            this->device_type,
            this->device_index,
            torch::kFloat,
            this->device_type == c10::DeviceType::CUDA
#ifdef WITH_CUDA
                ? at::cuda::getCurrentCUDAStream()
#else
                ? (cudaStream_t) nullptr
#endif
                : (cudaStream_t) nullptr);
      }
      std::get<7>(ret) = torch::stack(results);
    } else {
      std::get<7>(ret) = from_blob(
          reinterpret_cast<float*>(this->renderer_vec[0].grad_opy_d),
          {static_cast<ptrdiff_t>(n_points)},
          this->device_type,
          this->device_index,
          torch::kFloat,
          this->device_type == c10::DeviceType::CUDA
#ifdef WITH_CUDA
              ? at::cuda::getCurrentCUDAStream()
#else
              ? (cudaStream_t) nullptr
#endif
              : (cudaStream_t) nullptr);
    }
  }
  return ret;
};

} // namespace pytorch
} // namespace pulsar
