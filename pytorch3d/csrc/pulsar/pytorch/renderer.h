/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef PULSAR_NATIVE_PYTORCH_RENDERER_H_
#define PULSAR_NATIVE_PYTORCH_RENDERER_H_

#include "../global.h"
#include "../include/renderer.h"

namespace pulsar {
namespace pytorch {

struct Renderer {
 public:
  /**
   * Pytorch Pulsar differentiable rendering module.
   */
  explicit Renderer(
      const unsigned int& width,
      const unsigned int& height,
      const uint& max_n_balls,
      const bool& orthogonal_projection,
      const bool& right_handed_system,
      const float& background_normalization_depth,
      const uint& n_channels,
      const uint& n_track);
  ~Renderer();

  std::tuple<torch::Tensor, torch::Tensor> forward(
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
      const uint& mode);

  std::tuple<
      std::optional<torch::Tensor>,
      std::optional<torch::Tensor>,
      std::optional<torch::Tensor>,
      std::optional<torch::Tensor>,
      std::optional<torch::Tensor>,
      std::optional<torch::Tensor>,
      std::optional<torch::Tensor>,
      std::optional<torch::Tensor>>
  backward(
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
      const std::optional<std::pair<uint, uint>>& dbg_pos);

  // Infrastructure.
  /**
   * Ensure that the renderer is placed on this device.
   * Is nearly a no-op if the device is correct.
   */
  void ensure_on_device(torch::Device device, bool non_blocking = false);

  /**
   * Ensure that at least n renderers are available.
   */
  void ensure_n_renderers_gte(const size_t& batch_size);

  /**
   * Check the parameters.
   */
  std::tuple<size_t, size_t, bool, torch::Tensor> arg_check(
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
      const uint& mode);

  bool operator==(const Renderer& rhs) const;
  inline friend std::ostream& operator<<(
      std::ostream& stream,
      const Renderer& self) {
    stream << "pulsar::Renderer[";
    // Device info.
    stream << self.device_type;
    if (self.device_index != -1)
      stream << ", ID " << self.device_index;
    stream << "]";
    return stream;
  }

  inline uint width() const {
    return this->renderer_vec[0].cam.film_width;
  }
  inline uint height() const {
    return this->renderer_vec[0].cam.film_height;
  }
  inline int max_num_balls() const {
    return this->renderer_vec[0].max_num_balls;
  }
  inline bool orthogonal() const {
    return this->renderer_vec[0].cam.orthogonal_projection;
  }
  inline bool right_handed() const {
    return this->renderer_vec[0].cam.right_handed;
  }
  inline uint n_track() const {
    return static_cast<uint>(this->renderer_vec[0].n_track);
  }

  /** A tensor that is registered as a buffer with this Module to track its
   * device placement. Unfortunately, pytorch doesn't offer tracking Module
   * device placement in a better way as of now.
   */
  torch::Tensor device_tracker;

 protected:
  /** The device type for this renderer. */
  c10::DeviceType device_type;
  /** The device index for this renderer. */
  c10::DeviceIndex device_index;
  /** Pointer to the underlying pulsar renderers. */
  std::vector<pulsar::Renderer::Renderer> renderer_vec;
};

} // namespace pytorch
} // namespace pulsar

#endif
