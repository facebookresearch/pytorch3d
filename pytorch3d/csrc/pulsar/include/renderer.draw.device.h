/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef PULSAR_NATIVE_INCLUDE_RENDERER_CALC_SIGNATURE_DEVICE_H_
#define PULSAR_NATIVE_INCLUDE_RENDERER_CALC_SIGNATURE_DEVICE_H_

#include "../global.h"
#include "./camera.device.h"
#include "./commands.h"
#include "./math.h"
#include "./renderer.h"

namespace pulsar {
namespace Renderer {

/**
 * Draw a ball into the `result`.
 *
 * Returns whether a hit was noticed. See README for an explanation of sphere
 * points and variable notation.
 */
INLINE DEVICE bool draw(
    /* In variables. */
    const DrawInfo& draw_info, /** The draw information for this ball. */
    const float& opacity, /** The sphere opacity. */
    const CamInfo&
        cam, /** Camera information. Doesn't have to be normalized. */
    const float& gamma, /** 'Transparency' indicator (see paper for details). */
    const float3& ray_dir_norm, /** The direction of the ray, normalized. */
    const float2& projected_ray, /** The intersection of the ray with the image
                                    in pixel space. */
    /** Mode switches. */
    const bool& draw_only, /** Whether we are in draw vs. grad mode. */
    const bool& calc_grad_pos, /** Calculate position gradients. */
    const bool& calc_grad_col, /** Calculate color gradients. */
    const bool& calc_grad_rad, /** Calculate radius gradients. */
    const bool& calc_grad_cam, /** Calculate camera gradients. */
    const bool& calc_grad_opy, /** Calculate opacity gradients. */
    /** Position info. */
    const uint& coord_x, /** The pixel position x to draw at. */
    const uint& coord_y, /** The pixel position y to draw at. */
    const uint& idx, /** The id of the sphere to process. */
    /* Optional in variables. */
    IntersectInfo const* const RESTRICT
        intersect_info, /** The intersect information for this ball. */
    float3 const* const RESTRICT ray_dir, /** The ray direction (not normalized)
                             to draw at. Only used for grad computation. */
    float const* const RESTRICT norm_ray_dir, /** The length of the direction
                                 vector. Only used for grad computation. */
    float const* const RESTRICT grad_pix, /** The gradient for this pixel. Only
                              used for grad computation. */
    float const* const RESTRICT
        ln_pad_over_1minuspad, /** Allowed percentage indicator. */
    /* In or out variables, depending on mode. */
    float* const RESTRICT sm_d, /** Normalization denominator. */
    float* const RESTRICT
        sm_m, /** Maximum of normalization weight factors observed. */
    float* const RESTRICT
        result, /** Result pixel color. Must be zeros initially. */
    /* Optional out variables. */
    float* const RESTRICT depth_threshold, /** The depth threshold to use. Only
                                              used for rendering. */
    float* const RESTRICT intersection_depth_norm_out, /** The intersection
                                             depth. Only set when rendering. */
    float3* const RESTRICT grad_pos, /** Gradient w.r.t. position. */
    float* const RESTRICT grad_col, /** Gradient w.r.t. color. */
    float* const RESTRICT grad_rad, /** Gradient w.r.t. radius. */
    CamGradInfo* const RESTRICT grad_cam, /** Gradient w.r.t. camera. */
    float* const RESTRICT grad_opy /** Gradient w.r.t. opacity. */
) {
  // TODO: variable reuse?
  PASSERT(
      isfinite(draw_info.ray_center_norm.x) &&
      isfinite(draw_info.ray_center_norm.y) &&
      isfinite(draw_info.ray_center_norm.z));
  PASSERT(isfinite(draw_info.t_center) && draw_info.t_center >= 0.f);
  PASSERT(
      isfinite(draw_info.radius) && draw_info.radius >= 0.f &&
      draw_info.radius <= draw_info.t_center);
  PASSERT(isfinite(ray_dir_norm.x));
  PASSERT(isfinite(ray_dir_norm.y));
  PASSERT(isfinite(ray_dir_norm.z));
  PASSERT(isfinite(*sm_d));
  PASSERT(
      cam.orthogonal_projection && cam.focal_length == 0.f ||
      cam.focal_length > 0.f);
  PASSERT(gamma <= 1.f && gamma >= 1e-5f);
  /** The ball center in the camera coordinate system. */
  float3 center = draw_info.ray_center_norm * draw_info.t_center;
  /** The vector from the reference point to the ball center. */
  float3 raydiff;
  if (cam.orthogonal_projection) {
    center = rotate(
        center,
        cam.pixel_dir_x / length(cam.pixel_dir_x),
        cam.pixel_dir_y / length(cam.pixel_dir_y),
        cam.sensor_dir_z);
    raydiff =
        make_float3( // TODO: make offset consistent with `get_screen_area`.
            center.x -
                (projected_ray.x -
                 static_cast<float>(cam.aperture_width) * .5f) *
                    (2.f * cam.half_pixel_size),
            center.y -
                (projected_ray.y -
                 static_cast<float>(cam.aperture_height) * .5f) *
                    (2.f * cam.half_pixel_size),
            0.f);
  } else {
    /** The reference point on the ray; the point in the same distance
     * from the camera as the ball center, but along the ray.
     */
    const float3 rayref = ray_dir_norm * draw_info.t_center;
    raydiff = center - rayref;
  }
  /** The closeness of the reference point to ball center in world coords.
   *
   * In [0., radius].
   */
  const float closeness_world = length(raydiff);
  /** The reciprocal radius. */
  const float radius_rcp = FRCP(draw_info.radius);
  /** The closeness factor normalized with the ball radius.
   *
   * In [0., 1.].
   */
  float closeness = FSATURATE(FMA(-closeness_world, radius_rcp, 1.f));
  PULSAR_LOG_DEV_PIX(
      PULSAR_LOG_DRAW_PIX,
      "drawprep %u|center: %.9f, %.9f, %.9f. raydiff: %.9f, "
      "%.9f, %.9f. closeness_world: %.9f. closeness: %.9f\n",
      idx,
      center.x,
      center.y,
      center.z,
      raydiff.x,
      raydiff.y,
      raydiff.z,
      closeness_world,
      closeness);
  /** Whether this is the 'center pixel' for this ball, the pixel that
   * is closest to its projected center. This information is used to
   * make sure to draw 'tiny' spheres with less than one pixel in
   * projected size.
   */
  bool ray_through_center_pixel;
  float projected_radius, projected_x, projected_y;
  if (cam.orthogonal_projection) {
    projected_x = center.x / (2.f * cam.half_pixel_size) +
        (static_cast<float>(cam.aperture_width) - 1.f) / 2.f;
    projected_y = center.y / (2.f * cam.half_pixel_size) +
        (static_cast<float>(cam.aperture_height) - 1.f) / 2.f;
    projected_radius = draw_info.radius / (2.f * cam.half_pixel_size);
    ray_through_center_pixel =
        (FABS(FSUB(projected_x, projected_ray.x)) < 0.5f + FEPS &&
         FABS(FSUB(projected_y, projected_ray.y)) < 0.5f + FEPS);
    PULSAR_LOG_DEV_PIX(
        PULSAR_LOG_DRAW_PIX,
        "drawprep %u|closeness_world: %.9f. closeness: %.9f. "
        "projected (x, y): %.9f, %.9f. projected_ray (x, y): "
        "%.9f, %.9f. ray_through_center_pixel: %d.\n",
        idx,
        closeness_world,
        closeness,
        projected_x,
        projected_y,
        projected_ray.x,
        projected_ray.y,
        ray_through_center_pixel);
  } else {
    // Misusing this variable for half pixel size projected to the depth
    // at which the sphere resides. Leave some slack for numerical
    // inaccuracy (factor 1.5).
    projected_x = FMUL(cam.half_pixel_size * 1.5, draw_info.t_center) *
        FRCP(cam.focal_length);
    projected_radius = FMUL(draw_info.radius, cam.focal_length) *
        FRCP(draw_info.t_center) / (2.f * cam.half_pixel_size);
    ray_through_center_pixel = projected_x > closeness_world;
    PULSAR_LOG_DEV_PIX(
        PULSAR_LOG_DRAW_PIX,
        "drawprep %u|closeness_world: %.9f. closeness: %.9f. "
        "projected half pixel size: %.9f. "
        "ray_through_center_pixel: %d.\n",
        idx,
        closeness_world,
        closeness,
        projected_x,
        ray_through_center_pixel);
  }
  if (draw_only && draw_info.radius < closeness_world &&
      !ray_through_center_pixel) {
    PULSAR_LOG_DEV_PIX(
        PULSAR_LOG_DRAW_PIX,
        "drawprep %u|Abandoning since no hit has been detected.\n",
        idx);
    return false;
  } else {
    // This is always a hit since we are following the forward execution pass.
    // p2 is the closest intersection point with the sphere.
  }
  if (ray_through_center_pixel && projected_radius < 1.f) {
    // Make a tiny sphere visible.
    PULSAR_LOG_DEV_PIX(
        PULSAR_LOG_DRAW_PIX,
        "drawprep %u|Setting closeness to 1 (projected radius: %.9f).\n",
        idx,
        projected_radius);
    closeness = 1.;
  }
  PASSERT(closeness >= 0.f && closeness <= 1.f);
  /** Distance between the camera (`o`) and `p1`, the closest point to the
   * ball center along the casted ray.
   *
   * In [t_center - radius, t_center].
   */
  float o__p1_;
  /** The distance from ball center to p1.
   *
   * In [0., sqrt(t_center ^ 2 - (t_center - radius) ^ 2)].
   */
  float c__p1_;
  if (cam.orthogonal_projection) {
    o__p1_ = FABS(center.z);
    c__p1_ = length(raydiff);
  } else {
    o__p1_ = dot(center, ray_dir_norm);
    /**
     * This is being calculated as sqrt(t_center^2 - o__p1_^2) =
     * sqrt((t_center + o__p1_) * (t_center - o__p1_)) to avoid
     * catastrophic cancellation in floating point representations.
     */
    c__p1_ = FSQRT(
        (draw_info.t_center + o__p1_) * FMAX(draw_info.t_center - o__p1_, 0.f));
    // PASSERT(o__p1_ >= draw_info.t_center - draw_info.radius);
    // Numerical errors lead to too large values.
    o__p1_ = FMIN(o__p1_, draw_info.t_center);
    // PASSERT(o__p1_ <= draw_info.t_center);
  }
  /** The distance from the closest point to the sphere center (p1)
   * to the closest intersection point (p2).
   *
   * In [0., radius].
   */
  const float p1__p2_ =
      FSQRT((draw_info.radius + c__p1_) * FMAX(draw_info.radius - c__p1_, 0.f));
  PASSERT(p1__p2_ >= 0.f && p1__p2_ <= draw_info.radius);
  PULSAR_LOG_DEV_PIX(
      PULSAR_LOG_DRAW_PIX,
      "drawprep %u|o__p1_: %.9f, c__p1_: %.9f, p1__p2_: %.9f.\n",
      idx,
      o__p1_,
      c__p1_,
      p1__p2_);
  /** The intersection depth of the ray with this ball.
   *
   * In [t_center - radius, t_center].
   */
  const float intersection_depth = (o__p1_ - p1__p2_);
  PASSERT(
      cam.orthogonal_projection &&
          (intersection_depth >= center.z - draw_info.radius &&
           intersection_depth <= center.z) ||
      intersection_depth >= draw_info.t_center - draw_info.radius &&
          intersection_depth <= draw_info.t_center);
  /** Normalized distance of the closest intersection point; in [0., 1.]. */
  const float norm_dist =
      FMUL(FSUB(intersection_depth, cam.min_dist), cam.norm_fac);
  PASSERT(norm_dist >= 0.f && norm_dist <= 1.f);
  /** Scaled, normalized distance in [1., 0.] (closest, farthest). */
  const float norm_dist_scaled = FSUB(1.f, norm_dist) / gamma * opacity;
  PASSERT(norm_dist_scaled >= 0.f && norm_dist_scaled <= 1.f / gamma);
  PULSAR_LOG_DEV_PIX(
      PULSAR_LOG_DRAW_PIX,
      "drawprep %u|intersection_depth: %.9f, norm_dist: %.9f, "
      "norm_dist_scaled: %.9f.\n",
      idx,
      intersection_depth,
      norm_dist,
      norm_dist_scaled);
  float const* const col_ptr =
      cam.n_channels > 3 ? draw_info.color_union.ptr : &draw_info.first_color;
  // The implementation for the numerically stable weighted softmax is based
  // on https://arxiv.org/pdf/1805.02867.pdf .
  if (draw_only) {
    /** The old maximum observed value. */
    const float sm_m_old = *sm_m;
    *sm_m = FMAX(*sm_m, norm_dist_scaled);
    const float coeff_exp = FEXP(norm_dist_scaled - *sm_m);
    PASSERT(isfinite(coeff_exp));
    /** The color coefficient for the ball color; in [0., 1.]. */
    const float coeff = closeness * coeff_exp * opacity;
    PULSAR_LOG_DEV_PIX(
        PULSAR_LOG_DRAW_PIX,
        "draw %u|coeff: %.9f. closeness: %.9f. coeff_exp: %.9f. "
        "opacity: %.9f.\n",
        idx,
        coeff,
        closeness,
        coeff_exp,
        opacity);
    // Rendering.
    if (sm_m_old == *sm_m) {
      // Use the fact that exp(0) = 1 to avoid the exp calculation for
      // the case that the maximum remains the same (which it should
      // most of the time).
      *sm_d = FADD(*sm_d, coeff);
      for (uint c_id = 0; c_id < cam.n_channels; ++c_id) {
        PASSERT(isfinite(result[c_id]));
        result[c_id] = FMA(coeff, col_ptr[c_id], result[c_id]);
      }
    } else {
      const float exp_correction = FEXP(sm_m_old - *sm_m);
      *sm_d = FMA(*sm_d, exp_correction, coeff);
      for (uint c_id = 0; c_id < cam.n_channels; ++c_id) {
        PASSERT(isfinite(result[c_id]));
        result[c_id] =
            FMA(coeff, col_ptr[c_id], FMUL(result[c_id], exp_correction));
      }
    }
    PASSERT(isfinite(*sm_d));
    *intersection_depth_norm_out = intersection_depth;
    // Update the depth threshold.
    *depth_threshold =
        1.f - (FLN(*sm_d + FEPS) + *ln_pad_over_1minuspad + *sm_m) * gamma;
    *depth_threshold =
        FMA(*depth_threshold, FSUB(cam.max_dist, cam.min_dist), cam.min_dist);
  } else {
    // Gradient computation.
    const float coeff_exp = FEXP(norm_dist_scaled - *sm_m);
    const float gamma_rcp = FRCP(gamma);
    const float radius_sq = FMUL(draw_info.radius, draw_info.radius);
    const float coeff = FMAX(
        FMIN(closeness * coeff_exp * opacity, *sm_d - FEPS),
        0.f); // in [0., sm_d - FEPS].
    PASSERT(coeff >= 0.f && coeff <= *sm_d);
    const float otherw = *sm_d - coeff; // in [FEPS, sm_d].
    const float p1__p2_safe = FMAX(p1__p2_, FEPS); // in [eps, t_center].
    const float cam_range = FSUB(cam.max_dist, cam.min_dist); // in ]0, inf[
    PULSAR_LOG_DEV_PIX(
        PULSAR_LOG_GRAD,
        "grad %u|pos: %.9f, %.9f, %.9f. pixeldirx: %.9f, %.9f, %.9f. "
        "pixeldiry: %.9f, %.9f, %.9f. pixel00center: %.9f, %.9f, %.9f.\n",
        idx,
        draw_info.ray_center_norm.x * draw_info.t_center,
        draw_info.ray_center_norm.y * draw_info.t_center,
        draw_info.ray_center_norm.z * draw_info.t_center,
        cam.pixel_dir_x.x,
        cam.pixel_dir_x.y,
        cam.pixel_dir_x.z,
        cam.pixel_dir_y.x,
        cam.pixel_dir_y.y,
        cam.pixel_dir_y.z,
        cam.pixel_0_0_center.x,
        cam.pixel_0_0_center.y,
        cam.pixel_0_0_center.z);
    PULSAR_LOG_DEV_PIX(
        PULSAR_LOG_GRAD,
        "grad %u|ray_dir: %.9f, %.9f, %.9f. "
        "ray_dir_norm: %.9f, %.9f, %.9f. "
        "draw_info.ray_center_norm: %.9f, %.9f, %.9f.\n",
        idx,
        ray_dir->x,
        ray_dir->y,
        ray_dir->z,
        ray_dir_norm.x,
        ray_dir_norm.y,
        ray_dir_norm.z,
        draw_info.ray_center_norm.x,
        draw_info.ray_center_norm.y,
        draw_info.ray_center_norm.z);
    PULSAR_LOG_DEV_PIX(
        PULSAR_LOG_GRAD,
        "grad %u|coeff_exp: %.9f. "
        "norm_dist_scaled: %.9f. cam.norm_fac: %f.\n",
        idx,
        coeff_exp,
        norm_dist_scaled,
        cam.norm_fac);
    PULSAR_LOG_DEV_PIX(
        PULSAR_LOG_GRAD,
        "grad %u|p1__p2_: %.9f. p1__p2_safe: %.9f.\n",
        idx,
        p1__p2_,
        p1__p2_safe);
    PULSAR_LOG_DEV_PIX(
        PULSAR_LOG_GRAD,
        "grad %u|o__p1_: %.9f. c__p1_: %.9f.\n",
        idx,
        o__p1_,
        c__p1_);
    PULSAR_LOG_DEV_PIX(
        PULSAR_LOG_GRAD,
        "grad %u|intersection_depth: %f. norm_dist: %f. "
        "coeff: %.9f. closeness: %f. coeff_exp: %f. opacity: "
        "%f. color: %f, %f, %f.\n",
        idx,
        intersection_depth,
        norm_dist,
        coeff,
        closeness,
        coeff_exp,
        opacity,
        draw_info.first_color,
        draw_info.color_union.color[0],
        draw_info.color_union.color[1]);
    PULSAR_LOG_DEV_PIX(
        PULSAR_LOG_GRAD,
        "grad %u|t_center: %.9f. "
        "radius: %.9f. max_dist: %f. min_dist: %f. gamma: %f.\n",
        idx,
        draw_info.t_center,
        draw_info.radius,
        cam.max_dist,
        cam.min_dist,
        gamma);
    PULSAR_LOG_DEV_PIX(
        PULSAR_LOG_GRAD,
        "grad %u|sm_d: %f. sm_m: %f. grad_pix (first three): %f, %f, %f.\n",
        idx,
        *sm_d,
        *sm_m,
        grad_pix[0],
        grad_pix[1],
        grad_pix[2]);
    PULSAR_LOG_DEV_PIX(PULSAR_LOG_GRAD, "grad %u|otherw: %f.\n", idx, otherw);
    if (calc_grad_col) {
      const float sm_d_norm = FRCP(FMAX(*sm_d, FEPS));
      // First do the multiplication of coeff (in [0., sm_d]) and 1/sm_d. The
      // result is a factor in [0., 1.] to be multiplied with the incoming
      // gradient.
      for (uint c_id = 0; c_id < cam.n_channels; ++c_id) {
        ATOMICADD(grad_col + c_id, grad_pix[c_id] * FMUL(coeff, sm_d_norm));
      }
      PULSAR_LOG_DEV_PIX(
          PULSAR_LOG_GRAD,
          "grad %u|dimDdcol.x: %f. dresDdcol.x: %f.\n",
          idx,
          FMUL(coeff, sm_d_norm) * grad_pix[0],
          coeff * sm_d_norm);
    }
    // We disable the computation for too small spheres.
    // The comparison is made this way to avoid subtraction of unsigned types.
    if (calc_grad_cam || calc_grad_pos || calc_grad_rad || calc_grad_opy) {
      //! First find dimDdcoeff.
      const float n0 =
          otherw * FRCP(FMAX(*sm_d * *sm_d, FEPS)); // in [0., 1. / sm_d].
      PASSERT(isfinite(n0) && n0 >= 0. && n0 <= 1. / *sm_d + 1e2f * FEPS);
      // We'll aggergate dimDdcoeff over all the 'color' channels.
      float dimDdcoeff = 0.f;
      const float otherw_safe_rcp = FRCP(FMAX(otherw, FEPS));
      float othercol;
      for (uint c_id = 0; c_id < cam.n_channels; ++c_id) {
        othercol =
            (result[c_id] * *sm_d - col_ptr[c_id] * coeff) * otherw_safe_rcp;
        PULSAR_LOG_DEV_PIX(
            PULSAR_LOG_GRAD,
            "grad %u|othercol[%u]: %.9f.\n",
            idx,
            c_id,
            othercol);
        dimDdcoeff +=
            FMUL(FMUL(grad_pix[c_id], FSUB(col_ptr[c_id], othercol)), n0);
      }
      PASSERT(isfinite(dimDdcoeff));
      PULSAR_LOG_DEV_PIX(
          PULSAR_LOG_GRAD,
          "grad %u|dimDdcoeff: %.9f, n0: %f.\n",
          idx,
          dimDdcoeff,
          n0);
      if (calc_grad_opy) {
        //! dimDdopacity.
        *grad_opy += dimDdcoeff * coeff_exp * closeness *
            (1.f + opacity * (1.f - norm_dist) * gamma_rcp);
        PULSAR_LOG_DEV_PIX(
            PULSAR_LOG_GRAD,
            "grad %u|dcoeffDdopacity: %.9f, dimDdopacity: %.9f.\n",
            idx,
            coeff_exp * closeness,
            dimDdcoeff * coeff_exp * closeness);
      }
      if (intersect_info->max.x >= intersect_info->min.x + 3 &&
          intersect_info->max.y >= intersect_info->min.y + 3) {
        //! Now find dcoeffDdintersection_depth and dcoeffDdcloseness.
        const float dcoeffDdintersection_depth =
            -closeness * coeff_exp * opacity * opacity / (gamma * cam_range);
        const float dcoeffDdcloseness = coeff_exp * opacity;
        PULSAR_LOG_DEV_PIX(
            PULSAR_LOG_GRAD,
            "grad %u|dcoeffDdintersection_depth: %.9f. "
            "dimDdintersection_depth: %.9f. "
            "dcoeffDdcloseness: %.9f. dimDdcloseness: %.9f.\n",
            idx,
            dcoeffDdintersection_depth,
            dimDdcoeff * dcoeffDdintersection_depth,
            dcoeffDdcloseness,
            dimDdcoeff * dcoeffDdcloseness);
        //! Here, the execution paths for orthogonal and pinyhole camera split.
        if (cam.orthogonal_projection) {
          if (calc_grad_rad) {
            //! Find dcoeffDdrad.
            float dcoeffDdrad =
                dcoeffDdcloseness * (closeness_world / radius_sq) -
                dcoeffDdintersection_depth * draw_info.radius / p1__p2_safe;
            PASSERT(isfinite(dcoeffDdrad));
            *grad_rad += FMUL(dimDdcoeff, dcoeffDdrad);
            PULSAR_LOG_DEV_PIX(
                PULSAR_LOG_GRAD,
                "grad %u|dimDdrad: %.9f. dcoeffDdrad: %.9f.\n",
                idx,
                FMUL(dimDdcoeff, dcoeffDdrad),
                dcoeffDdrad);
          }
          if (calc_grad_pos || calc_grad_cam) {
            float3 dimDdcenter = raydiff /
                p1__p2_safe; /* making it dintersection_depthDdcenter. */
            dimDdcenter.z = sign_dir(center.z);
            PASSERT(FABS(center.z) >= cam.min_dist && cam.min_dist >= FEPS);
            dimDdcenter *= dcoeffDdintersection_depth; // dcoeffDdcenter
            dimDdcenter -= dcoeffDdcloseness * /* dclosenessDdcenter. */
                raydiff * FRCP(FMAX(length(raydiff) * draw_info.radius, FEPS));
            PULSAR_LOG_DEV_PIX(
                PULSAR_LOG_GRAD,
                "grad %u|dcoeffDdcenter: %.9f, %.9f, %.9f.\n",
                idx,
                dimDdcenter.x,
                dimDdcenter.y,
                dimDdcenter.z);
            // Now dcoeffDdcenter is stored in dimDdcenter.
            dimDdcenter *= dimDdcoeff;
            PULSAR_LOG_DEV_PIX(
                PULSAR_LOG_GRAD,
                "grad %u|dimDdcenter: %.9f, %.9f, %.9f.\n",
                idx,
                dimDdcenter.x,
                dimDdcenter.y,
                dimDdcenter.z);
            // Prepare for posglob and cam pos.
            const float pixel_size = length(cam.pixel_dir_x);
            // pixel_size is the same as length(pixeldiry)!
            const float pixel_size_rcp = FRCP(pixel_size);
            float3 dcenterDdposglob =
                (cam.pixel_dir_x + cam.pixel_dir_y) * pixel_size_rcp +
                cam.sensor_dir_z;
            PULSAR_LOG_DEV_PIX(
                PULSAR_LOG_GRAD,
                "grad %u|dcenterDdposglob: %.9f, %.9f, %.9f.\n",
                idx,
                dcenterDdposglob.x,
                dcenterDdposglob.y,
                dcenterDdposglob.z);
            if (calc_grad_pos) {
              //! dcenterDdposglob.
              *grad_pos += dimDdcenter * dcenterDdposglob;
              PULSAR_LOG_DEV_PIX(
                  PULSAR_LOG_GRAD,
                  "grad %u|dimDdpos: %.9f, %.9f, %.9f.\n",
                  idx,
                  dimDdcenter.x * dcenterDdposglob.x,
                  dimDdcenter.y * dcenterDdposglob.y,
                  dimDdcenter.z * dcenterDdposglob.z);
            }
            if (calc_grad_cam) {
              //! Camera.
              grad_cam->cam_pos -= dimDdcenter * dcenterDdposglob;
              PULSAR_LOG_DEV_PIX(
                  PULSAR_LOG_GRAD,
                  "grad %u|dimDdeye: %.9f, %.9f, %.9f.\n",
                  idx,
                  -dimDdcenter.x * dcenterDdposglob.x,
                  -dimDdcenter.y * dcenterDdposglob.y,
                  -dimDdcenter.z * dcenterDdposglob.z);
              // coord_world
              /*
              float3 dclosenessDdcoord_world =
                raydiff * FRCP(FMAX(draw_info.radius * length(raydiff), FEPS));
              float3 dintersection_depthDdcoord_world = -2.f * raydiff;
              */
              float3 dimDdcoord_world = /* dcoeffDdcoord_world */
                  dcoeffDdcloseness * raydiff *
                      FRCP(FMAX(draw_info.radius * length(raydiff), FEPS)) -
                  dcoeffDdintersection_depth * raydiff / p1__p2_safe;
              PULSAR_LOG_DEV_PIX(
                  PULSAR_LOG_GRAD,
                  "grad %u|dcoeffDdcoord_world: %.9f, %.9f, %.9f.\n",
                  idx,
                  dimDdcoord_world.x,
                  dimDdcoord_world.y,
                  dimDdcoord_world.z);
              dimDdcoord_world *= dimDdcoeff;
              PULSAR_LOG_DEV_PIX(
                  PULSAR_LOG_GRAD,
                  "grad %u|dimDdcoord_world: %.9f, %.9f, %.9f.\n",
                  idx,
                  dimDdcoord_world.x,
                  dimDdcoord_world.y,
                  dimDdcoord_world.z);
              // The third component of dimDdcoord_world is 0!
              PASSERT(dimDdcoord_world.z == 0.f);
              float3 coord_world = center - raydiff;
              coord_world.z = 0.f;
              PULSAR_LOG_DEV_PIX(
                  PULSAR_LOG_GRAD,
                  "grad %u|coord_world: %.9f, %.9f, %.9f.\n",
                  idx,
                  coord_world.x,
                  coord_world.y,
                  coord_world.z);
              // Do this component-wise to save unnecessary matmul steps.
              grad_cam->pixel_dir_x += dimDdcoord_world.x * cam.pixel_dir_x *
                  coord_world.x * pixel_size_rcp * pixel_size_rcp;
              grad_cam->pixel_dir_x += dimDdcoord_world.y * cam.pixel_dir_x *
                  coord_world.y * pixel_size_rcp * pixel_size_rcp;
              PULSAR_LOG_DEV_PIX(
                  PULSAR_LOG_GRAD,
                  "grad %u|dimDdpixel_dir_x|coord_world: %.9f, %.9f, %.9f.\n",
                  idx,
                  grad_cam->pixel_dir_x.x,
                  grad_cam->pixel_dir_x.y,
                  grad_cam->pixel_dir_x.z);
              // dcenterkDdpixel_dir_k.
              float3 center_in_pixels = draw_info.ray_center_norm *
                  draw_info.t_center * pixel_size_rcp;
              grad_cam->pixel_dir_x += dimDdcenter.x *
                  (center_in_pixels -
                   outer_product_sum(cam.pixel_dir_x) * center_in_pixels *
                       pixel_size_rcp * pixel_size_rcp);
              PULSAR_LOG_DEV_PIX(
                  PULSAR_LOG_GRAD,
                  "grad %u|dcenter0dpixel_dir_x: %.9f, %.9f, %.9f.\n",
                  idx,
                  (center_in_pixels -
                   outer_product_sum(cam.pixel_dir_x) * center_in_pixels *
                       pixel_size_rcp * pixel_size_rcp)
                      .x,
                  (center_in_pixels -
                   outer_product_sum(cam.pixel_dir_x) * center_in_pixels *
                       pixel_size_rcp * pixel_size_rcp)
                      .y,
                  (center_in_pixels -
                   outer_product_sum(cam.pixel_dir_x) * center_in_pixels *
                       pixel_size_rcp * pixel_size_rcp)
                      .z);
              grad_cam->pixel_dir_y += dimDdcenter.y *
                  (center_in_pixels -
                   outer_product_sum(cam.pixel_dir_y) * center_in_pixels *
                       pixel_size_rcp * pixel_size_rcp);
              PULSAR_LOG_DEV_PIX(
                  PULSAR_LOG_GRAD,
                  "grad %u|dcenter1dpixel_dir_y: %.9f, %.9f, %.9f.\n",
                  idx,
                  (center_in_pixels -
                   outer_product_sum(cam.pixel_dir_y) * center_in_pixels *
                       pixel_size_rcp * pixel_size_rcp)
                      .x,
                  (center_in_pixels -
                   outer_product_sum(cam.pixel_dir_y) * center_in_pixels *
                       pixel_size_rcp * pixel_size_rcp)
                      .y,
                  (center_in_pixels -
                   outer_product_sum(cam.pixel_dir_y) * center_in_pixels *
                       pixel_size_rcp * pixel_size_rcp)
                      .z);
              // dcenterzDdpixel_dir_k.
              float sensordirz_norm_rcp = FRCP(
                  FMAX(length(cross(cam.pixel_dir_y, cam.pixel_dir_x)), FEPS));
              grad_cam->pixel_dir_x += dimDdcenter.z *
                  (dot(center, cam.sensor_dir_z) *
                       cross(cam.pixel_dir_y, cam.sensor_dir_z) -
                   cross(cam.pixel_dir_y, center)) *
                  sensordirz_norm_rcp;
              PULSAR_LOG_DEV_PIX(
                  PULSAR_LOG_GRAD,
                  "grad %u|dcenterzDdpixel_dir_x: %.9f, %.9f, %.9f.\n",
                  idx,
                  ((dot(center, cam.sensor_dir_z) *
                        cross(cam.pixel_dir_y, cam.sensor_dir_z) -
                    cross(cam.pixel_dir_y, center)) *
                   sensordirz_norm_rcp)
                      .x,
                  ((dot(center, cam.sensor_dir_z) *
                        cross(cam.pixel_dir_y, cam.sensor_dir_z) -
                    cross(cam.pixel_dir_y, center)) *
                   sensordirz_norm_rcp)
                      .y,
                  ((dot(center, cam.sensor_dir_z) *
                        cross(cam.pixel_dir_y, cam.sensor_dir_z) -
                    cross(cam.pixel_dir_y, center)) *
                   sensordirz_norm_rcp)
                      .z);
              grad_cam->pixel_dir_y += dimDdcenter.z *
                  (dot(center, cam.sensor_dir_z) *
                       cross(cam.pixel_dir_x, cam.sensor_dir_z) -
                   cross(cam.pixel_dir_x, center)) *
                  sensordirz_norm_rcp;
              PULSAR_LOG_DEV_PIX(
                  PULSAR_LOG_GRAD,
                  "grad %u|dcenterzDdpixel_dir_y: %.9f, %.9f, %.9f.\n",
                  idx,
                  ((dot(center, cam.sensor_dir_z) *
                        cross(cam.pixel_dir_x, cam.sensor_dir_z) -
                    cross(cam.pixel_dir_x, center)) *
                   sensordirz_norm_rcp)
                      .x,
                  ((dot(center, cam.sensor_dir_z) *
                        cross(cam.pixel_dir_x, cam.sensor_dir_z) -
                    cross(cam.pixel_dir_x, center)) *
                   sensordirz_norm_rcp)
                      .y,
                  ((dot(center, cam.sensor_dir_z) *
                        cross(cam.pixel_dir_x, cam.sensor_dir_z) -
                    cross(cam.pixel_dir_x, center)) *
                   sensordirz_norm_rcp)
                      .z);
              PULSAR_LOG_DEV_PIX(
                  PULSAR_LOG_GRAD,
                  "grad %u|dimDdpixel_dir_x: %.9f, %.9f, %.9f.\n",
                  idx,
                  grad_cam->pixel_dir_x.x,
                  grad_cam->pixel_dir_x.y,
                  grad_cam->pixel_dir_x.z);
              PULSAR_LOG_DEV_PIX(
                  PULSAR_LOG_GRAD,
                  "grad %u|dimDdpixel_dir_y: %.9f, %.9f, %.9f.\n",
                  idx,
                  grad_cam->pixel_dir_y.x,
                  grad_cam->pixel_dir_y.y,
                  grad_cam->pixel_dir_y.z);
            }
          }
        } else {
          if (calc_grad_rad) {
            //! Find dcoeffDdrad.
            float dcoeffDdrad =
                dcoeffDdcloseness * (closeness_world / radius_sq) -
                dcoeffDdintersection_depth * draw_info.radius / p1__p2_safe;
            PASSERT(isfinite(dcoeffDdrad));
            *grad_rad += FMUL(dimDdcoeff, dcoeffDdrad);
            PULSAR_LOG_DEV_PIX(
                PULSAR_LOG_GRAD,
                "grad %u|dimDdrad: %.9f. dcoeffDdrad: %.9f.\n",
                idx,
                FMUL(dimDdcoeff, dcoeffDdrad),
                dcoeffDdrad);
          }
          if (calc_grad_pos || calc_grad_cam) {
            const float3 tmp1 = center - ray_dir_norm * o__p1_;
            const float3 tmp1n = tmp1 / p1__p2_safe;
            const float ray_dir_normDotRaydiff = dot(ray_dir_norm, raydiff);
            const float3 dcoeffDdray = dcoeffDdintersection_depth *
                    (tmp1 - o__p1_ * tmp1n) / *norm_ray_dir +
                dcoeffDdcloseness *
                    (ray_dir_norm * -ray_dir_normDotRaydiff + raydiff) /
                    (closeness_world * draw_info.radius) *
                    (draw_info.t_center / *norm_ray_dir);
            PULSAR_LOG_DEV_PIX(
                PULSAR_LOG_GRAD,
                "grad %u|dcoeffDdray: %.9f, %.9f, %.9f. dimDdray: "
                "%.9f, %.9f, %.9f.\n",
                idx,
                dcoeffDdray.x,
                dcoeffDdray.y,
                dcoeffDdray.z,
                dimDdcoeff * dcoeffDdray.x,
                dimDdcoeff * dcoeffDdray.y,
                dimDdcoeff * dcoeffDdray.z);
            const float3 dcoeffDdcenter =
                dcoeffDdintersection_depth * (ray_dir_norm + tmp1n) +
                dcoeffDdcloseness *
                    (draw_info.ray_center_norm * ray_dir_normDotRaydiff -
                     raydiff) /
                    (closeness_world * draw_info.radius);
            PULSAR_LOG_DEV_PIX(
                PULSAR_LOG_GRAD,
                "grad %u|dcoeffDdcenter: %.9f, %.9f, %.9f. "
                "dimDdcenter: %.9f, %.9f, %.9f.\n",
                idx,
                dcoeffDdcenter.x,
                dcoeffDdcenter.y,
                dcoeffDdcenter.z,
                dimDdcoeff * dcoeffDdcenter.x,
                dimDdcoeff * dcoeffDdcenter.y,
                dimDdcoeff * dcoeffDdcenter.z);
            if (calc_grad_pos) {
              *grad_pos += dimDdcoeff * dcoeffDdcenter;
              PULSAR_LOG_DEV_PIX(
                  PULSAR_LOG_GRAD,
                  "grad %u|dimDdposglob: %.9f, %.9f, %.9f.\n",
                  idx,
                  dimDdcoeff * dcoeffDdcenter.x,
                  dimDdcoeff * dcoeffDdcenter.y,
                  dimDdcoeff * dcoeffDdcenter.z);
            }
            if (calc_grad_cam) {
              PULSAR_LOG_DEV_PIX(
                  PULSAR_LOG_GRAD,
                  "grad %u|dimDdeye: %.9f, %.9f, %.9f.\n",
                  idx,
                  -dimDdcoeff * (dcoeffDdcenter.x + dcoeffDdray.x),
                  -dimDdcoeff * (dcoeffDdcenter.y + dcoeffDdray.y),
                  -dimDdcoeff * (dcoeffDdcenter.z + dcoeffDdray.z));
              grad_cam->cam_pos += -dimDdcoeff * (dcoeffDdcenter + dcoeffDdray);
              grad_cam->pixel_0_0_center += dimDdcoeff * dcoeffDdray;
              PULSAR_LOG_DEV_PIX(
                  PULSAR_LOG_GRAD,
                  "grad %u|dimDdpixel00centerglob: %.9f, %.9f, %.9f.\n",
                  idx,
                  dimDdcoeff * dcoeffDdray.x,
                  dimDdcoeff * dcoeffDdray.y,
                  dimDdcoeff * dcoeffDdray.z);
              grad_cam->pixel_dir_x +=
                  (dimDdcoeff * static_cast<float>(coord_x)) * dcoeffDdray;
              PULSAR_LOG_DEV_PIX(
                  PULSAR_LOG_GRAD,
                  "grad %u|dimDdpixel_dir_x: %.9f, %.9f, %.9f.\n",
                  idx,
                  (dimDdcoeff * static_cast<float>(coord_x)) * dcoeffDdray.x,
                  (dimDdcoeff * static_cast<float>(coord_x)) * dcoeffDdray.y,
                  (dimDdcoeff * static_cast<float>(coord_x)) * dcoeffDdray.z);
              grad_cam->pixel_dir_y +=
                  (dimDdcoeff * static_cast<float>(coord_y)) * dcoeffDdray;
              PULSAR_LOG_DEV_PIX(
                  PULSAR_LOG_GRAD,
                  "grad %u|dimDdpixel_dir_y: %.9f, %.9f, %.9f.\n",
                  idx,
                  (dimDdcoeff * static_cast<float>(coord_y)) * dcoeffDdray.x,
                  (dimDdcoeff * static_cast<float>(coord_y)) * dcoeffDdray.y,
                  (dimDdcoeff * static_cast<float>(coord_y)) * dcoeffDdray.z);
            }
          }
        }
      }
    }
  }
  return true;
};

} // namespace Renderer
} // namespace pulsar

#endif
