# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This file defines SplatterBlender, which is used for blending in SplatterPhongShader.

import itertools
from typing import Tuple

import torch
import torch.nn.functional as F
from pytorch3d.common.datatypes import Device
from pytorch3d.renderer import BlendParams
from pytorch3d.renderer.cameras import FoVPerspectiveCameras

from .blending import _get_background_color


def _precompute(
    input_shape: Tuple[int, int, int, int], device: Device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Precompute padding and offset constants that won't change for a given NHWK shape.

    Args:
        input_shape: Tuple indicating N (batch size), H, W (image size) and K (number of
            intersections) output by the rasterizer.
        device: Device to store the tensors on.

    returns:
        crop_ids_h: An (N, H, W+2, K, 9, 5) tensor, used during splatting to offset the
            p-pixels (splatting pixels) in one of the 9 splatting directions within a
            call to torch.gather. See comments and offset_splats for details.
        crop_ids_w: An (N, H, W, K, 9, 5) tensor, used similarly to crop_ids_h.
        offsets: A (1, 1, 1, 1, 9, 2) tensor (shaped so for broadcasting) containing va-
            lues [-1, -1], [-1, 0], [-1, 1], [0, -1], ..., [1, 1] which correspond to
            the nine splatting directions.
    """
    N, H, W, K = input_shape

    # (N, H, W+2, K, 9, 5) tensor, used to reduce a tensor from (N, H+2, W+2...) to
    # (N, H, W+2, ...) in torch.gather. If only torch.gather broadcasted, we wouldn't
    # need the tiling. But it doesn't.
    crop_ids_h = (
        torch.arange(0, H, device=device).view(1, H, 1, 1, 1, 1)
        + torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2], device=device).view(
            1, 1, 1, 1, 9, 1
        )
    ).expand(N, H, W + 2, K, 9, 5)

    # (N, H, W, K, 9, 5) tensor, used to reduce a tensor from (N, H, W+2, ...) to
    # (N, H, W, ...) in torch.gather.
    crop_ids_w = (
        torch.arange(0, W, device=device).view(1, 1, W, 1, 1, 1)
        + torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2], device=device).view(
            1, 1, 1, 1, 9, 1
        )
    ).expand(N, H, W, K, 9, 5)

    offsets = torch.tensor(
        list(itertools.product((-1, 0, 1), repeat=2)),
        dtype=torch.long,
        device=device,
    )

    return crop_ids_h, crop_ids_w, offsets


def _prepare_pixels_and_colors(
    pixel_coords_cameras: torch.Tensor,
    colors: torch.Tensor,
    cameras: FoVPerspectiveCameras,
    background_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Project pixel coords into the un-inverted screen frame of reference, and set
    background pixel z-values to 1.0 and alphas to 0.0.

    Args:
        pixel_coords_cameras: (N, H, W, K, 3) float tensor.
        colors: (N, H, W, K, 3) float tensor.
        cameras: PyTorch3D cameras, for now we assume FoVPerspectiveCameras.
        background_mask: (N, H, W, K) boolean tensor.

    Returns:
        pixel_coords_screen: (N, H, W, K, 3) float tensor. Background pixels have
            x=y=z=1.0.
        colors: (N, H, W, K, 4). Alpha is set to 1 for foreground pixels and 0 for back-
            ground pixels.
    """

    N, H, W, K, C = colors.shape
    # pixel_coords_screen will contain invalid values at background
    # intersections, and [H+0.5, W+0.5, z] at valid intersections. It is important
    # to not flip the xy axes, otherwise the gradients will be inverted when the
    # splatter works with a detached rasterizer.
    pixel_coords_screen = cameras.transform_points_screen(
        pixel_coords_cameras.view([N, -1, 3]), image_size=(H, W), with_xyflip=False
    ).reshape(pixel_coords_cameras.shape)

    # Set colors' alpha to 1 and background to 0.
    colors = torch.cat(
        [colors, torch.ones_like(colors[..., :1])], dim=-1
    )  # (N, H, W, K, 4)

    # The hw values of background don't matter because their alpha is set
    # to 0 in the next step (which means that no matter what their splatting kernel
    # value is, they will not splat as the kernel is multiplied by alpha). However,
    # their z-values need to be at max depth.  Otherwise, we could incorrectly compute
    # occlusion layer linkage.
    pixel_coords_screen[background_mask] = 1.0

    # Any background color value value with alpha=0 will do, as anything with
    # alpha=0 will have a zero-weight splatting power. Note that neighbors can still
    # splat on zero-alpha pixels: that's the way we get non-zero gradients at the
    # boundary with the background.
    colors[background_mask] = 0.0

    return pixel_coords_screen, colors


def _get_splat_kernel_normalization(
    offsets: torch.Tensor,
    sigma: float = 0.5,
):
    if sigma <= 0.0:
        raise ValueError("Only positive standard deviations make sense.")

    epsilon = 0.05
    normalization_constant = torch.exp(
        # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and `int`.
        -(offsets**2).sum(dim=1)
        / (2 * sigma**2)
    ).sum()

    # We add an epsilon to the normalization constant to ensure the gradient will travel
    # through non-boundary pixels' normalization factor, see Sec. 3.3.1 in "Differentia-
    # ble Surface Rendering via Non-Differentiable Sampling", Cole et al.
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    return (1 + epsilon) / normalization_constant


def _compute_occlusion_layers(
    q_depth: torch.Tensor,
) -> torch.Tensor:
    """
    For each splatting pixel, decide whether it splats from a background, surface, or
    foreground depth relative to the splatted pixel. See unit tests in
    test_splatter_blend for some enlightening examples.

    Args:
        q_depth: (N, H, W, K) tensor of z-values of the splatted pixels.

    Returns:
        occlusion_layers: (N, H, W, 9) long tensor. Each of the 9 values corresponds to
            one of the nine splatting directions ([-1, -1], [-1, 0], ..., [1,
            1]). The value at nhwd (where d is the splatting direction) is 0 if
            the splat in direction d is on the same surface level as the pixel at
            hw. The value is negative if the splat is in the background (occluded
            by another splat above it that is at the same surface level as the
            pixel splatted on), and the value is positive if the splat is in the
            foreground.
    """
    N, H, W, K = q_depth.shape

    # q are the "center pixels" and p the pixels splatting onto them. Use `unfold` to
    # create `p_depth`, a tensor with 9 layers, each of which corresponds to the
    # depth of a neighbor of q in one of the 9 directions. For example, p_depth[nk0hw]
    # is the depth of the pixel splatting onto pixel nhwk from the [-1, -1] direction,
    # and p_depth[nk4hw] the depth of q (self-splatting onto itself).
    # More concretely, imagine the pixel depths in a 2x2 image's k-th layer are
    #   .1 .2
    #   .3 .4
    # Then (remembering that we pad with zeros when a pixel has fewer than 9 neighbors):
    #
    # p_depth[n, k, :, 0, 0] = [ 0  0  0  0 .1 .2  0 .3 .4] - neighbors of .1
    # p_depth[n, k, :, 0, 1] = [ 0  0  0 .1 .2  0 .3 .4  0] - neighbors of .2
    # p_depth[n, k, :, 1, 0] = [ 0 .1 .2  0 .3 .4  0  0  0] - neighbors of .3
    # p_depth[n, k, :, 0, 1] = [.1 .2  0 .3 .4  0  0  0  0] - neighbors of .4
    q_depth = q_depth.permute(0, 3, 1, 2)  # (N, K, H, W)
    p_depth = F.unfold(q_depth, kernel_size=3, padding=1)  # (N, 3^2 * K, H * W)
    q_depth = q_depth.view(N, K, 1, H, W)
    p_depth = p_depth.view(N, K, 9, H, W)

    # Take the center pixel q's top rasterization layer. This is the "surface layer"
    # that we're splatting on. For each of the nine splatting directions p, find which
    # of the K splatting rasterization layers is closest in depth to the surface
    # splatted layer.
    qtop_to_p_zdist = torch.abs(p_depth - q_depth[:, 0:1])  # (N, K, 9, H, W)
    qtop_to_p_closest_zdist, qtop_to_p_closest_id = qtop_to_p_zdist.min(dim=1)

    # For each of the nine splatting directions p, take the top of the K rasterization
    # layers. Check which of the K q-layers (that the given direction is splatting on)
    # is closest in depth to the top splatting layer.
    ptop_to_q_zdist = torch.abs(p_depth[:, 0:1] - q_depth)  # (N, K, 9, H, W)
    ptop_to_q_closest_zdist, ptop_to_q_closest_id = ptop_to_q_zdist.min(dim=1)

    # Decide whether each p is on the same level, below, or above the q it is splatting
    # on. See Fig. 4 in [0] for an illustration. Briefly: say we're interested in pixel
    # p_{h, w} = [10, 32] splatting onto its neighbor q_{h, w} = [11, 33]. The splat is
    # coming from direction [-1, -1], which has index 0 in our enumeration of splatting
    # directions. Hence, we are interested in
    #
    # P = p_depth[n, :, d=0, 11, 33] - a vector of K depth values, and
    # Q = q_depth.squeeze()[n, :, 11, 33] - a vector of K depth values.
    #
    # If Q[0] is closest, say, to P[2], then we assume the 0th surface layer of Q is
    # the same surface as P[2] that's splatting onto it, and P[:2] are foreground splats
    # and P[3:] are background splats.
    #
    # If instead say Q[2] is closest to P[0], then all the splats are background splats,
    # because the top splatting layer is the same surface as a non-top splatted layer.
    #
    # Finally, if Q[0] is closest to P[0], then the top-level P is splatting onto top-
    # level Q, and P[1:] are all background splats.
    occlusion_offsets = torch.where(  # noqa
        ptop_to_q_closest_zdist < qtop_to_p_closest_zdist,
        -ptop_to_q_closest_id,
        qtop_to_p_closest_id,
    )  # (N, 9, H, W)

    occlusion_layers = occlusion_offsets.permute((0, 2, 3, 1))  # (N, H, W, 9)
    return occlusion_layers


def _compute_splatting_colors_and_weights(
    pixel_coords_screen: torch.Tensor,
    colors: torch.Tensor,
    sigma: float,
    offsets: torch.Tensor,
) -> torch.Tensor:
    """
    For each center pixel q, compute the splatting weights of its surrounding nine spla-
    tting pixels p, as well as their splatting colors (which are just their colors re-
    weighted by the splatting weights).

    Args:
        pixel_coords_screen: (N, H, W, K, 2) tensor of pixel screen coords.
        colors: (N, H, W, K, 4) RGBA tensor of pixel colors.
        sigma: splatting kernel variance.
        offsets: (9, 2) tensor computed by _precompute, indicating the nine
            splatting directions ([-1, -1], ..., [1, 1]).

    Returns:
        splat_colors_and_weights: (N, H, W, K, 9, 5) tensor.
            splat_colors_and_weights[..., :4] corresponds to the splatting colors, and
            splat_colors_and_weights[..., 4:5] to the splatting weights. The "9" di-
            mension corresponds to the nine splatting directions.
    """
    N, H, W, K, C = colors.shape
    splat_kernel_normalization = _get_splat_kernel_normalization(offsets, sigma)

    # Distance from each barycentric-interpolated triangle vertices' triplet from its
    # "ideal" pixel-center location. pixel_coords_screen are in screen coordinates, and
    # should be at the "ideal" locations on the forward pass -- e.g.
    # pixel_coords_screen[n, 24, 31, k] = [24.5, 31.5]. For this reason, q_to_px_center
    # should equal torch.zeros during the forward pass. On the backwards pass, these
    # coordinates will be adjusted and non-zero, allowing the gradients to flow back
    # to the mesh vertex coordinates.
    q_to_px_center = (
        torch.floor(pixel_coords_screen[..., :2]) - pixel_coords_screen[..., :2] + 0.5
    ).view((N, H, W, K, 1, 2))

    # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and `int`.
    dist2_p_q = torch.sum((q_to_px_center + offsets) ** 2, dim=5)  # (N, H, W, K, 9)
    splat_weights = torch.exp(-dist2_p_q / (2 * sigma**2))
    alpha = colors[..., 3:4]
    splat_weights = (alpha * splat_kernel_normalization * splat_weights).unsqueeze(
        5
    )  # (N, H, W, K, 9, 1)

    # splat_colors[n, h, w, direction, :] contains the splatting color (weighted by the
    # splatting weight) that pixel h, w will splat in one  of the nine possible
    # directions (e.g. nhw0 corresponds to splatting in [-1, 1] direciton, nhw4 is
    # self-splatting).
    splat_colors = splat_weights * colors.unsqueeze(4)  # (N, H, W, K, 9, 4)

    return torch.cat([splat_colors, splat_weights], dim=5)


def _offset_splats(
    splat_colors_and_weights: torch.Tensor,
    crop_ids_h: torch.Tensor,
    crop_ids_w: torch.Tensor,
) -> torch.Tensor:
    """
    Pad splatting colors and weights so that tensor locations/coordinates are aligned
    with the splatting directions. For example, say we have an example input Red channel
    splat_colors_and_weights[n, :, :, k, direction=0, channel=0] equal to
       .1  .2  .3
       .4  .5  .6
       .7  .8  .9
    the (h, w) entry indicates that pixel n, h, w, k splats the given color in direction
    equal to 0, which corresponds to offsets[0] = (-1, -1). Note that this is the x-y
    direction, not h-w. This function pads and crops this array to
        0   0   0
       .2  .3   0
       .5  .6   0
    which indicates, for example, that:
        * There is no pixel splatting in direction (-1, -1) whose splat lands on pixel
          h=w=0.
        * There is a pixel splatting in direction (-1, -1) whose splat lands on the pi-
          xel h=1, w=0, and that pixel's splatting color is .2.
        * There is a pixel splatting in direction (-1, -1) whose splat lands on the pi-
          xel h=2, w=1, and that pixel's splatting color is .6.

    Args:
        *splat_colors_and_weights*: (N, H, W, K, 9, 5) tensor of colors and weights,
        where dim=-2 corresponds to the splatting directions/offsets.
        *crop_ids_h*: (N, H, W+2, K, 9, 5) precomputed tensor used for padding within
            torch.gather. See _precompute for more info.
        *crop_ids_w*: (N, H, W, K, 9, 5) precomputed tensor used for padding within
            torch.gather. See _precompute for more info.


    Returns:
        *splat_colors_and_weights*: (N, H, W, K, 9, 5) tensor.
    """
    N, H, W, K, _, _ = splat_colors_and_weights.shape
    # Transform splat_colors such that each of the 9 layers (corresponding to
    # the 9 splat offsets) is padded with 1 and shifted in the appropriate
    # direction. E.g. splat_colors[n, :, :, 0] corresponds to the (-1, -1)
    # offset, so will be padded with one rows of 1 on the right and have a
    # single row clipped at the bottom, and splat_colors[n, :, :, 4] corrsponds
    # to offset (0, 0) and will remain unchanged.
    splat_colors_and_weights = F.pad(
        splat_colors_and_weights, [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]
    )  # N, H+2, W+2, 9, 5

    # (N, H+2, W+2, K, 9, 5) -> (N, H, W+2, K, 9, 5)
    splat_colors_and_weights = torch.gather(
        splat_colors_and_weights, dim=1, index=crop_ids_h
    )

    # (N, H, W+2, K, 9, 5) -> (N, H, W, K, 9, 5)
    splat_colors_and_weights = torch.gather(
        splat_colors_and_weights, dim=2, index=crop_ids_w
    )

    return splat_colors_and_weights


def _compute_splatted_colors_and_weights(
    occlusion_layers: torch.Tensor,  # (N, H, W, 9)
    splat_colors_and_weights: torch.Tensor,  # (N, H, W, K, 9, 5)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Accumulate splatted colors in background, surface and foreground occlusion buffers.

    Args:
        occlusion_layers: (N, H, W, 9) tensor. See _compute_occlusion_layers.
        splat_colors_and_weights: (N, H, W, K, 9, 5) tensor. See _offset_splats.

    Returns:
        splatted_colors: (N, H, W, 4, 3) tensor. Last dimension corresponds to back-
            ground, surface, and foreground splat colors.
        splatted_weights: (N, H, W, 1, 3) tensor. Last dimension corresponds to back-
            ground, surface, and foreground splat weights and is used for normalization.

    """
    N, H, W, K, _, _ = splat_colors_and_weights.shape

    # Create an occlusion mask, with the last dimension of length 3, corresponding to
    # background/surface/foreground splatting. E.g. occlusion_layer_mask[n,h,w,k,d,0] is
    # 1 if the pixel at hw is splatted from direction d such that the splatting pixel p
    # is below the splatted pixel q (in the background); otherwise, the value is 0.
    # occlusion_layer_mask[n,h,w,k,d,1] is 1 if the splatting pixel is at the same
    # surface level as the splatted pixel q, and occlusion_layer_mask[n,h,w,k,d,2] is
    # 1 only if the splatting pixel is in the foreground.
    layer_ids = torch.arange(K, device=splat_colors_and_weights.device).view(
        1, 1, 1, K, 1
    )
    occlusion_layers = occlusion_layers.view(N, H, W, 1, 9)
    occlusion_layer_mask = torch.stack(
        [
            occlusion_layers > layer_ids,  # (N, H, W, K, 9)
            occlusion_layers == layer_ids,  # (N, H, W, K, 9)
            occlusion_layers < layer_ids,  # (N, H, W, K, 9)
        ],
        dim=5,
    ).float()  # (N, H, W, K, 9, 3)

    # (N * H * W, 5, 9 * K) x (N * H * W, 9 * K, 3) -> (N * H * W, 5, 3)
    splatted_colors_and_weights = torch.bmm(
        splat_colors_and_weights.permute(0, 1, 2, 5, 3, 4).reshape(
            (N * H * W, 5, K * 9)
        ),
        occlusion_layer_mask.reshape((N * H * W, K * 9, 3)),
    ).reshape((N, H, W, 5, 3))

    return (
        splatted_colors_and_weights[..., :4, :],
        splatted_colors_and_weights[..., 4:5, :],
    )


def _normalize_and_compose_all_layers(
    background_color: torch.Tensor,
    splatted_colors_per_occlusion_layer: torch.Tensor,
    splatted_weights_per_occlusion_layer: torch.Tensor,
) -> torch.Tensor:
    """
    Normalize each bg/surface/fg buffer by its weight, and compose.

    Args:
        background_color: (3) RGB tensor.
        splatter_colors_per_occlusion_layer: (N, H, W, 4, 3) RGBA tensor, last dimension
            corresponds to foreground, surface, and background splatting.
        splatted_weights_per_occlusion_layer: (N, H, W, 1, 3) weight tensor.

    Returns:
        output_colors: (N, H, W, 4) RGBA tensor.
    """
    device = splatted_colors_per_occlusion_layer.device

    # Normalize each of bg/surface/fg splat layers separately.
    normalization_scales = 1.0 / (
        # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
        torch.maximum(
            splatted_weights_per_occlusion_layer,
            torch.tensor([1.0], device=device),
        )
    )  # (N, H, W, 1, 3)

    normalized_splatted_colors = (
        splatted_colors_per_occlusion_layer * normalization_scales
    )  # (N, H, W, 4, 3)

    # Use alpha-compositing to compose the splat layers.
    output_colors = torch.cat(
        [background_color, torch.tensor([0.0], device=device)]
    )  # (4), will broadcast to (N, H, W, 4) below.

    for occlusion_layer_id in (-1, -2, -3):
        # Over-compose the bg, surface, and fg occlusion layers. Note that we already
        # multiplied each pixel's RGBA by its own alpha as part of self-splatting in
        # _compute_splatting_colors_and_weights, so we don't re-multiply by alpha here.
        alpha = normalized_splatted_colors[..., 3:4, occlusion_layer_id]  # (N, H, W, 1)
        output_colors = (
            normalized_splatted_colors[..., occlusion_layer_id]
            + (1.0 - alpha) * output_colors
        )
    return output_colors


class SplatterBlender(torch.nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int, int, int, int],
        device,
    ):
        """
        A splatting blender. See `forward` docs for details of the splatting mechanism.

        Args:
            input_shape: Tuple (N, H, W, K) indicating the batch size, image height,
                image width, and number of rasterized layers. Used to precompute
                constant tensors that do not change as long as this tuple is unchanged.
        """
        super().__init__()
        self.crop_ids_h, self.crop_ids_w, self.offsets = _precompute(
            input_shape, device
        )

    def to(self, device):
        self.offsets = self.offsets.to(device)
        self.crop_ids_h = self.crop_ids_h.to(device)
        self.crop_ids_w = self.crop_ids_w.to(device)
        super().to(device)

    def forward(
        self,
        colors: torch.Tensor,
        pixel_coords_cameras: torch.Tensor,
        cameras: FoVPerspectiveCameras,
        background_mask: torch.Tensor,
        blend_params: BlendParams,
    ) -> torch.Tensor:
        """
        RGB blending using splatting, as proposed in [0].

        Args:
            colors: (N, H, W, K, 3) tensor of RGB colors at each h, w pixel location for
                K intersection layers.
            pixel_coords_cameras: (N, H, W, K, 3) tensor of pixel coordinates in the
                camera frame of reference. It is *crucial* that these are computed by
                interpolating triangle vertex positions using barycentric coordinates --
                this allows gradients to travel through pixel_coords_camera back to the
                vertex positions.
            cameras: Cameras object used to project pixel_coords_cameras screen coords.
            background_mask: (N, H, W, K, 3) boolean tensor, True for bg pixels. A pixel
                is considered "background" if no mesh triangle projects to it. This is
                typically computed by the rasterizer.
            blend_params: BlendParams, from which we use sigma (splatting kernel
                variance) and background_color.

        Returns:
            output_colors: (N, H, W, 4) tensor of RGBA values. The alpha layer is set to
                fully transparent in the background.

        [0] Cole, F. et al., "Differentiable Surface Rendering via Non-differentiable
            Sampling".
        """

        # Our implementation has 6 stages. In the description below, we will call each
        # pixel q and the 9 surrounding splatting pixels (including itself) p.
        #     1. Use barycentrics to compute the position of each pixel in screen
        # coordinates. These should exactly correspond to pixel centers during the
        # forward pass, but can be shifted on backwards. This step allows gradients to
        # travel to vertex coordinates, even if the rasterizer is non-differentiable.
        #     2a. For each center pixel q, take each splatting p and decide whether it
        # is on the same surface level as q, or in the background or foreground.
        #     2b. For each center pixel q, compute the splatting weight of surrounding
        # pixels p, and their splatting colors (which are just the original colors
        # weighted by the splatting weights).
        #     3. As a vectorization technicality, offset the tensors corresponding to
        # the splatting p values in the nine directions, by padding each of nine
        # splatting layers on the bottom/top, left/right.
        #     4. Do the actual splatting, by accumulating the splatting colors of the
        # surrounding p's for each pixel q. The weights get accumulated separately for
        # p's that got assigned to the background/surface/foreground in Step 2a.
        #     5. Normalize each the splatted bg/surface/fg colors for each q, and
        # compose the resulting color maps.
        #
        # Note that it is crucial that in Step 1 we compute the pixel coordinates by in-
        # terpolating triangle vertices using barycentric coords from the rasterizer. In
        # our case, these pixel_coords_camera are computed by the shader and passed to
        # this function to avoid re-computation.

        pixel_coords_screen, colors = _prepare_pixels_and_colors(
            pixel_coords_cameras, colors, cameras, background_mask
        )  # (N, H, W, K, 3) and (N, H, W, K, 4)

        occlusion_layers = _compute_occlusion_layers(
            pixel_coords_screen[..., 2:3].squeeze(dim=-1)
        )  # (N, H, W, 9)

        splat_colors_and_weights = _compute_splatting_colors_and_weights(
            pixel_coords_screen,
            colors,
            blend_params.sigma,
            self.offsets,
        )  # (N, H, W, K, 9, 5)

        splat_colors_and_weights = _offset_splats(
            splat_colors_and_weights,
            self.crop_ids_h,
            self.crop_ids_w,
        )  # (N, H, W, K, 9, 5)

        (
            splatted_colors_per_occlusion_layer,
            splatted_weights_per_occlusion_layer,
        ) = _compute_splatted_colors_and_weights(
            occlusion_layers, splat_colors_and_weights
        )  # (N, H, W, 4, 3) and (N, H, W, 1, 3)

        output_colors = _normalize_and_compose_all_layers(
            _get_background_color(blend_params, colors.device),
            splatted_colors_per_occlusion_layer,
            splatted_weights_per_occlusion_layer,
        )  # (N, H, W, 4)

        return output_colors
