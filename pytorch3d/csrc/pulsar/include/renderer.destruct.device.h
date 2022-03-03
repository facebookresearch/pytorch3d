/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef PULSAR_NATIVE_INCLUDE_RENDERER_DESTRUCT_H_
#define PULSAR_NATIVE_INCLUDE_RENDERER_DESTRUCT_H_

#include "../global.h"
#include "./commands.h"
#include "./renderer.h"

namespace pulsar {
namespace Renderer {

template <bool DEV>
HOST void destruct(Renderer* self) {
  if (self->result_d != NULL)
    FREE(self->result_d);
  self->result_d = NULL;
  if (self->min_depth_d != NULL)
    FREE(self->min_depth_d);
  self->min_depth_d = NULL;
  if (self->min_depth_sorted_d != NULL)
    FREE(self->min_depth_sorted_d);
  self->min_depth_sorted_d = NULL;
  if (self->ii_d != NULL)
    FREE(self->ii_d);
  self->ii_d = NULL;
  if (self->ii_sorted_d != NULL)
    FREE(self->ii_sorted_d);
  self->ii_sorted_d = NULL;
  if (self->ids_d != NULL)
    FREE(self->ids_d);
  self->ids_d = NULL;
  if (self->ids_sorted_d != NULL)
    FREE(self->ids_sorted_d);
  self->ids_sorted_d = NULL;
  if (self->workspace_d != NULL)
    FREE(self->workspace_d);
  self->workspace_d = NULL;
  if (self->di_d != NULL)
    FREE(self->di_d);
  self->di_d = NULL;
  if (self->di_sorted_d != NULL)
    FREE(self->di_sorted_d);
  self->di_sorted_d = NULL;
  if (self->region_flags_d != NULL)
    FREE(self->region_flags_d);
  self->region_flags_d = NULL;
  if (self->num_selected_d != NULL)
    FREE(self->num_selected_d);
  self->num_selected_d = NULL;
  if (self->forw_info_d != NULL)
    FREE(self->forw_info_d);
  self->forw_info_d = NULL;
  if (self->min_max_pixels_d != NULL)
    FREE(self->min_max_pixels_d);
  self->min_max_pixels_d = NULL;
  if (self->grad_pos_d != NULL)
    FREE(self->grad_pos_d);
  self->grad_pos_d = NULL;
  if (self->grad_col_d != NULL)
    FREE(self->grad_col_d);
  self->grad_col_d = NULL;
  if (self->grad_rad_d != NULL)
    FREE(self->grad_rad_d);
  self->grad_rad_d = NULL;
  if (self->grad_cam_d != NULL)
    FREE(self->grad_cam_d);
  self->grad_cam_d = NULL;
  if (self->grad_cam_buf_d != NULL)
    FREE(self->grad_cam_buf_d);
  self->grad_cam_buf_d = NULL;
  if (self->grad_opy_d != NULL)
    FREE(self->grad_opy_d);
  self->grad_opy_d = NULL;
  if (self->n_grad_contributions_d != NULL)
    FREE(self->n_grad_contributions_d);
  self->n_grad_contributions_d = NULL;
}

} // namespace Renderer
} // namespace pulsar

#endif
