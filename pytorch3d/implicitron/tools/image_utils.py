# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Sequence, Union

import torch


def mask_background(
    image_rgb: torch.Tensor,
    mask_fg: torch.Tensor,
    dim_color: int = 1,
    bg_color: Union[torch.Tensor, Sequence, str, float] = 0.0,
) -> torch.Tensor:
    """
    Mask the background input image tensor `image_rgb` with `bg_color`.
    The background regions are obtained from the binary foreground segmentation
    mask `mask_fg`.
    """
    tgt_view = [1, 1, 1, 1]
    tgt_view[dim_color] = 3
    # obtain the background color tensor
    if isinstance(bg_color, torch.Tensor):
        bg_color_t = bg_color.view(1, 3, 1, 1).clone().to(image_rgb)
    elif isinstance(bg_color, (float, tuple, list)):
        if isinstance(bg_color, float):
            bg_color = [bg_color] * 3
        bg_color_t = torch.tensor(
            bg_color, device=image_rgb.device, dtype=image_rgb.dtype
        ).view(*tgt_view)
    elif isinstance(bg_color, str):
        if bg_color == "white":
            bg_color_t = image_rgb.new_ones(tgt_view)
        elif bg_color == "black":
            bg_color_t = image_rgb.new_zeros(tgt_view)
        else:
            raise ValueError(_invalid_color_error_msg(bg_color))
    else:
        raise ValueError(_invalid_color_error_msg(bg_color))
    # cast to the image_rgb's type
    mask_fg = mask_fg.type_as(image_rgb)
    # mask the bg
    image_masked = mask_fg * image_rgb + (1 - mask_fg) * bg_color_t
    return image_masked


def _invalid_color_error_msg(bg_color) -> str:
    return (
        f"Invalid bg_color={bg_color}. Plese set bg_color to a 3-element"
        + " tensor. or a string (white | black), or a float."
    )
