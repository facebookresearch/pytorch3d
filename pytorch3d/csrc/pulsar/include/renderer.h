/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef PULSAR_NATIVE_INCLUDE_RENDERER_H_
#define PULSAR_NATIVE_INCLUDE_RENDERER_H_

#include <algorithm>

#include "../global.h"
#include "./camera.h"

namespace pulsar {
namespace Renderer {

//! Remember to order struct members from larger size to smaller size
//! to avoid padding (for more info, see for example here:
//! http://www.catb.org/esr/structure-packing/).

/**
 * This is the information that's needed to do a fast screen point
 * intersection with one of the balls.
 *
 * Aim to keep this below 8 bytes (256 bytes per cache-line / 32 threads in a
 * warp = 8 bytes per thread).
 */
struct IntersectInfo {
  ushort2 min; /** minimum x, y in pixel coordinates. */
  ushort2 max; /** maximum x, y in pixel coordinates. */
};
static_assert(
    sizeof(IntersectInfo) == 8,
    "The compiled size of `IntersectInfo` is wrong.");

/**
 * Reduction operation to find the limits of multiple IntersectInfo objects.
 */
struct IntersectInfoMinMax {
  IHD IntersectInfo
  operator()(const IntersectInfo& a, const IntersectInfo& b) const {
    // Treat the special case of an invalid intersect info object or one for
    // a ball out of bounds.
    if (b.max.x == MAX_USHORT && b.min.x == MAX_USHORT &&
        b.max.y == MAX_USHORT && b.min.y == MAX_USHORT) {
      return a;
    }
    if (a.max.x == MAX_USHORT && a.min.x == MAX_USHORT &&
        a.max.y == MAX_USHORT && a.min.y == MAX_USHORT) {
      return b;
    }
    IntersectInfo result;
    result.min.x = std::min<ushort>(a.min.x, b.min.x);
    result.min.y = std::min<ushort>(a.min.y, b.min.y);
    result.max.x = std::max<ushort>(a.max.x, b.max.x);
    result.max.y = std::max<ushort>(a.max.y, b.max.y);
    return result;
  }
};

/**
 * All information that's needed to draw a ball.
 *
 * It's necessary to keep this information in float (not half) format,
 * because the loss in accuracy would be too high and lead to artifacts.
 */
struct DrawInfo {
  float3 ray_center_norm; /** Ray to the ball center, normalized. */
  /** Ball color.
   *
   * This might be the full color in the case of n_channels <= 3. Otherwise,
   * a pointer to the original 'color' data is stored in the following union.
   */
  float first_color;
  union {
    float color[2];
    float* ptr;
  } color_union;
  float t_center; /** Distance from the camera to the ball center. */
  float radius; /** Ball radius. */
};
static_assert(
    sizeof(DrawInfo) == 8 * 4,
    "The compiled size of `DrawInfo` is wrong.");

/**
 * An object to collect all associated data with the renderer.
 *
 * The `_d` suffixed pointers point to memory 'on-device', potentially on the
 * GPU. All other variables are expected to point to CPU memory.
 */
struct Renderer {
  /** Dummy initializer to make sure all pointers are set to NULL to
   * be safe for the device-specific 'construct' and 'destruct' methods.
   */
  inline Renderer() {
    max_num_balls = 0;
    result_d = NULL;
    min_depth_d = NULL;
    min_depth_sorted_d = NULL;
    ii_d = NULL;
    ii_sorted_d = NULL;
    ids_d = NULL;
    ids_sorted_d = NULL;
    workspace_d = NULL;
    di_d = NULL;
    di_sorted_d = NULL;
    region_flags_d = NULL;
    num_selected_d = NULL;
    forw_info_d = NULL;
    grad_pos_d = NULL;
    grad_col_d = NULL;
    grad_rad_d = NULL;
    grad_cam_d = NULL;
    grad_opy_d = NULL;
    grad_cam_buf_d = NULL;
    n_grad_contributions_d = NULL;
  };
  /** The camera for this renderer. In world-coordinates. */
  CamInfo cam;
  /**
   * The maximum amount of balls the renderer can handle. Resources are
   * pre-allocated to account for this size. Less than this amount of balls
   * can be rendered, but not more.
   */
  int max_num_balls;
  /** The result buffer. */
  float* result_d;
  /** Closest possible intersection depth per sphere w.r.t. the camera. */
  float* min_depth_d;
  /** Closest possible intersection depth per sphere, ordered ascending. */
  float* min_depth_sorted_d;
  /** The intersect infos per sphere. */
  IntersectInfo* ii_d;
  /** The intersect infos per sphere, ordered by their closest possible
   * intersection depth (asc.). */
  IntersectInfo* ii_sorted_d;
  /** Original sphere IDs. */
  int* ids_d;
  /** Original sphere IDs, ordered by their closest possible intersection depth
   * (asc.). */
  int* ids_sorted_d;
  /** Workspace for CUB routines. */
  char* workspace_d;
  /** Workspace size for CUB routines. */
  size_t workspace_size;
  /** The draw information structures for each sphere. */
  DrawInfo* di_d;
  /** The draw information structures sorted by closest possible intersection
   * depth (asc.). */
  DrawInfo* di_sorted_d;
  /** Region association buffer. */
  char* region_flags_d;
  /** Num spheres in the current region. */
  size_t* num_selected_d;
  /** Pointer to information from the forward pass. */
  float* forw_info_d;
  /** Struct containing information about the min max pixels that contain
   * rendered information in the image. */
  IntersectInfo* min_max_pixels_d;
  /** Gradients w.r.t. position. */
  float3* grad_pos_d;
  /** Gradients w.r.t. color. */
  float* grad_col_d;
  /** Gradients w.r.t. radius. */
  float* grad_rad_d;
  /** Gradients w.r.t. camera parameters. */
  float* grad_cam_d;
  /** Gradients w.r.t. opacity. */
  float* grad_opy_d;
  /** Camera gradient information by sphere.
   *
   * Here, every sphere's contribution to the camera gradients is stored. It is
   * aggregated and written to grad_cam_d in a separate step. This avoids write
   * conflicts when processing the spheres.
   */
  CamGradInfo* grad_cam_buf_d;
  /** Total of all gradient contributions for this image. */
  int* n_grad_contributions_d;
  /** The number of spheres to track for backpropagation. */
  int n_track;
};

inline bool operator==(const Renderer& a, const Renderer& b) {
  return a.cam == b.cam && a.max_num_balls == b.max_num_balls;
}

/**
 * Construct a renderer.
 */
template <bool DEV>
void construct(
    Renderer* self,
    const size_t& max_num_balls,
    const int& width,
    const int& height,
    const bool& orthogonal_projection,
    const bool& right_handed_system,
    const float& background_normalization_depth,
    const uint& n_channels,
    const uint& n_track);

/**
 * Destruct the renderer and free the associated memory.
 */
template <bool DEV>
void destruct(Renderer* self);

/**
 * Create a selection of points inside a rectangle.
 *
 * This write boolen values into `region_flags_d', which can
 * for example be used by a CUB function to extract the selection.
 */
template <bool DEV>
GLOBAL void create_selector(
    IntersectInfo const* const RESTRICT ii_sorted_d,
    const uint num_balls,
    const int min_x,
    const int max_x,
    const int min_y,
    const int max_y,
    /* Out variables. */
    char* RESTRICT region_flags_d);

/**
 * Calculate a signature for a ball.
 *
 * Populate the `ids_d`, `ii_d`, `di_d` and `min_depth_d` fields of the
 * renderer. For spheres not visible in the image, sets the id field to -1,
 * min_depth_d to MAX_FLOAT and the ii_d.min.x fields to MAX_USHORT.
 */
template <bool DEV>
GLOBAL void calc_signature(
    Renderer renderer,
    float3 const* const RESTRICT vert_poss,
    float const* const RESTRICT vert_cols,
    float const* const RESTRICT vert_rads,
    const uint num_balls);

/**
 * The block size for rendering.
 *
 * This should be as large as possible, but is limited due to the amount
 * of variables we use and the memory required per thread.
 */
#define RENDER_BLOCK_SIZE 16
/**
 * The buffer size of spheres to be loaded and analyzed for relevance.
 *
 * This must be at least RENDER_BLOCK_SIZE * RENDER_BLOCK_SIZE so that
 * for every iteration through the loading loop every thread could add a
 * 'hit' to the buffer.
 */
#define RENDER_BUFFER_SIZE RENDER_BLOCK_SIZE* RENDER_BLOCK_SIZE * 2
/**
 * The threshold after which the spheres that are in the render buffer
 * are rendered and the buffer is flushed.
 *
 * Must be less than RENDER_BUFFER_SIZE.
 */
#define RENDER_BUFFER_LOAD_THRESH 16 * 4

/**
 * The render function.
 *
 * Assumptions:
 *   * the focal length is appropriately chosen,
 *   * ray_dir_norm.z is > EPS.
 *   * to be completed...
 */
template <bool DEV>
GLOBAL void render(
    size_t const* const RESTRICT
        num_balls, /** Number of balls relevant for this pass. */
    IntersectInfo const* const RESTRICT ii_d, /** Intersect information. */
    DrawInfo const* const RESTRICT di_d, /** Draw information. */
    float const* const RESTRICT min_depth_d, /** Minimum depth per sphere. */
    int const* const RESTRICT id_d, /** IDs. */
    float const* const RESTRICT op_d, /** Opacity. */
    const CamInfo cam_norm, /** Camera normalized with all vectors to be in the
                             * camera coordinate system.
                             */
    const float gamma, /** Transparency parameter. **/
    const float percent_allowed_difference, /** Maximum allowed
                                               error in color. */
    const uint max_n_hits,
    const float* bg_col_d,
    const uint mode,
    const int x_min,
    const int y_min,
    const int x_step,
    const int y_step,
    // Out variables.
    float* const RESTRICT result_d, /** The result image. */
    float* const RESTRICT forw_info_d, /** Additional information needed for the
                                           grad computation. */
    // Infrastructure.
    const int n_track /** The number of spheres to track. */
);

/**
 * Makes sure to paint background information.
 *
 * This is required as a separate post-processing step because certain
 * pixels may not be processed during the forward pass if there is no
 * possibility for a sphere to be present at their location.
 */
template <bool DEV>
GLOBAL void fill_bg(
    Renderer renderer,
    const CamInfo norm,
    float const* const bg_col_d,
    const float gamma,
    const uint mode);

/**
 * Rendering forward pass.
 *
 * Takes a renderer and sphere data as inputs and creates a rendering.
 */
template <bool DEV>
void forward(
    Renderer* self,
    const float* vert_pos,
    const float* vert_col,
    const float* vert_rad,
    const CamInfo& cam,
    const float& gamma,
    float percent_allowed_difference,
    const uint& max_n_hits,
    const float* bg_col_d,
    const float* opacity_d,
    const size_t& num_balls,
    const uint& mode,
    cudaStream_t stream);

/**
 * Normalize the camera gradients by the number of spheres that contributed.
 */
template <bool DEV>
GLOBAL void norm_cam_gradients(Renderer renderer);

/**
 * Normalize the sphere gradients.
 *
 * We're assuming that the samples originate from a Monte Carlo
 * sampling process and normalize by number and sphere area.
 */
template <bool DEV>
GLOBAL void norm_sphere_gradients(Renderer renderer, const int num_balls);

#define GRAD_BLOCK_SIZE 16
/** Calculate the gradients.
 */
template <bool DEV>
GLOBAL void calc_gradients(
    const CamInfo cam, /** Camera in world coordinates. */
    float const* const RESTRICT grad_im, /** The gradient image. */
    const float
        gamma, /** The transparency parameter used in the forward pass. */
    float3 const* const RESTRICT vert_poss, /** Vertex position vector. */
    float const* const RESTRICT vert_cols, /** Vertex color vector. */
    float const* const RESTRICT vert_rads, /** Vertex radius vector. */
    float const* const RESTRICT opacity, /** Vertex opacity. */
    const uint num_balls, /** Number of balls. */
    float const* const RESTRICT result_d, /** Result image. */
    float const* const RESTRICT forw_info_d, /** Forward pass info. */
    DrawInfo const* const RESTRICT di_d, /** Draw information. */
    IntersectInfo const* const RESTRICT ii_d, /** Intersect information. */
    // Mode switches.
    const bool calc_grad_pos,
    const bool calc_grad_col,
    const bool calc_grad_rad,
    const bool calc_grad_cam,
    const bool calc_grad_opy,
    // Out variables.
    float* const RESTRICT grad_rad_d, /** Radius gradients. */
    float* const RESTRICT grad_col_d, /** Color gradients. */
    float3* const RESTRICT grad_pos_d, /** Position gradients. */
    CamGradInfo* const RESTRICT grad_cam_buf_d, /** Camera gradient buffer. */
    float* const RESTRICT grad_opy_d, /** Opacity gradient buffer. */
    int* const RESTRICT
        grad_contributed_d, /** Gradient contribution counter. */
    // Infrastructure.
    const int n_track,
    const uint offs_x = 0,
    const uint offs_y = 0);

/**
 * A full backward pass.
 *
 * Creates the gradients for the given gradient_image and the spheres.
 */
template <bool DEV>
void backward(
    Renderer* self,
    const float* grad_im,
    const float* image,
    const float* forw_info,
    const float* vert_pos,
    const float* vert_col,
    const float* vert_rad,
    const CamInfo& cam,
    const float& gamma,
    float percent_allowed_difference,
    const uint& max_n_hits,
    const float* vert_opy,
    const size_t& num_balls,
    const uint& mode,
    const bool& dif_pos,
    const bool& dif_col,
    const bool& dif_rad,
    const bool& dif_cam,
    const bool& dif_opy,
    cudaStream_t stream);

/**
 * A debug backward pass.
 *
 * This is a function to debug the gradient calculation. It calculates the
 * gradients for exactly one pixel (set with pos_x and pos_y) without averaging.
 *
 * *Uses only the first sphere for camera gradient calculation!*
 */
template <bool DEV>
void backward_dbg(
    Renderer* self,
    const float* grad_im,
    const float* image,
    const float* forw_info,
    const float* vert_pos,
    const float* vert_col,
    const float* vert_rad,
    const CamInfo& cam,
    const float& gamma,
    float percent_allowed_difference,
    const uint& max_n_hits,
    const float* vert_opy,
    const size_t& num_balls,
    const uint& mode,
    const bool& dif_pos,
    const bool& dif_col,
    const bool& dif_rad,
    const bool& dif_cam,
    const bool& dif_opy,
    const uint& pos_x,
    const uint& pos_y,
    cudaStream_t stream);

template <bool DEV>
void nn(
    const float* ref_ptr,
    const float* tar_ptr,
    const uint& k,
    const uint& d,
    const uint& n,
    float* dist_ptr,
    int32_t* inds_ptr,
    cudaStream_t stream);

} // namespace Renderer
} // namespace pulsar

#endif
