# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""pulsar renderer PyTorch integration.

Proper Python support for pytorch requires creating a torch.autograd.function
(independent of whether this is being done within the C++ module). This is done
here and a torch.nn.Module is exposed for the use in more complex models.
"""

import logging
import warnings
from typing import Optional, Tuple, Union

import torch
from pytorch3d import _C
from pytorch3d.transforms import axis_angle_to_matrix, rotation_6d_to_matrix


LOGGER = logging.getLogger(__name__)
GAMMA_WARNING_EMITTED = False
AXANGLE_WARNING_EMITTED = False


class _Render(torch.autograd.Function):
    """
    Differentiable rendering function for the Pulsar renderer.

    Usually this will be used through the `Renderer` module, which takes care of
    setting up the buffers and putting them on the correct device. If you use
    the function directly, you will have to do this manually.

    The steps for this are two-fold: first, you need to create a native Renderer
    object to provide the required buffers. This is the `native_renderer` parameter
    for this function. You can create it by creating a `pytorch3d._C.PulsarRenderer`
    object (with parameters for width, height and maximum number of balls it should
    be able to render). This object by default resides on the CPU. If you want to
    shift the buffers to a different device, just assign an empty tensor on the target
    device to its property `device_tracker`.

    To convert camera parameters from a more convenient representation to the
    required vectors as in this function, you can use the static
    function `pytorch3d.renderer.points.pulsar.Renderer._transform_cam_params`.

    Args:
        * ctx: Pytorch context.
        * vert_pos: vertex positions. [Bx]Nx3 tensor of positions in 3D space.
        * vert_col: vertex colors. [Bx]NxK tensor of channels.
        * vert_rad: vertex radii. [Bx]N tensor of radiuses, >0.
        * cam_pos: camera position(s). [Bx]3 tensor in 3D coordinates.
        * pixel_0_0_center: [Bx]3 tensor center(s) of the upper left pixel(s) in
                            world coordinates.
        * pixel_vec_x: [Bx]3 tensor from one pixel center to the next in image x
                       direction in world coordinates.
        * pixel_vec_y: [Bx]3 tensor from one pixel center to the next in image y
                       direction in world coordinates.
        * focal_length: [Bx]1 tensor of focal lengths in world coordinates.
        * principal_point_offsets: [Bx]2 tensor of principal point offsets in pixels.
        * gamma: sphere transparency in [1.,1E-5], with 1 being mostly transparent.
                 [Bx]1.
        * max_depth: maximum depth for spheres to render. Set this as tighly
                     as possible to have good numerical accuracy for gradients.
        * native_renderer: a `pytorch3d._C.PulsarRenderer` object.
        * min_depth: a float with the minimum depth a sphere must have to be renderer.
                     Must be 0. or > max(focal_length).
        * bg_col: K tensor with a background color to use or None (uses all ones).
        * opacity: [Bx]N tensor of opacity values in [0., 1.] or None (uses all ones).
        * percent_allowed_difference: a float in [0., 1.[ with the maximum allowed
                     difference in color space. This is used to speed up the
                     computation. Default: 0.01.
        * max_n_hits: a hard limit on the number of hits per ray. Default: max int.
        * mode: render mode in {0, 1}. 0: render an image; 1: render the hit map.
        * return_forward_info: whether to return a second map. This second map contains
            13 channels: first channel contains sm_m (the maximum exponent factor
            observed), the second sm_d (the normalization denominator, the sum of all
            coefficients), the third the maximum closest possible intersection for a
            hit. The following channels alternate with the float encoded integer index
            of a sphere and its weight. They are the five spheres with the highest
            color contribution to this pixel color, ordered descending.

        Returns:
            * image: [Bx]HxWxK float tensor with the resulting image.
            * forw_info: [Bx]HxWx13 float forward information as described above,
                  if enabled.
    """

    @staticmethod
    # pyre-fixme[14]: `forward` overrides method defined in `Function` inconsistently.
    def forward(
        ctx,
        vert_pos,
        vert_col,
        vert_rad,
        cam_pos,
        pixel_0_0_center,
        pixel_vec_x,
        pixel_vec_y,
        focal_length,
        principal_point_offsets,
        gamma,
        max_depth,
        native_renderer,
        min_depth=0.0,
        bg_col=None,
        opacity=None,
        percent_allowed_difference=0.01,
        # pyre-fixme[16]: Module `_C` has no attribute `MAX_UINT`.
        max_n_hits=_C.MAX_UINT,
        mode=0,
        return_forward_info=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if mode != 0:
            assert not return_forward_info, (
                "You are using a non-standard rendering mode. This does "
                "not provide gradients, and also no `forward_info`. Please "
                "set `return_forward_info` to `False`."
            )
        ctx.gamma = gamma
        ctx.max_depth = max_depth
        ctx.min_depth = min_depth
        ctx.percent_allowed_difference = percent_allowed_difference
        ctx.max_n_hits = max_n_hits
        ctx.mode = mode
        ctx.native_renderer = native_renderer
        image, info = ctx.native_renderer.forward(
            vert_pos,
            vert_col,
            vert_rad,
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
            mode,
        )
        if mode != 0:
            # Backprop not possible!
            info = None
        # Prepare for backprop.
        ctx.save_for_backward(
            vert_pos,
            vert_col,
            vert_rad,
            cam_pos,
            pixel_0_0_center,
            pixel_vec_x,
            pixel_vec_y,
            focal_length,
            principal_point_offsets,
            bg_col,
            opacity,
            image,
            info,
        )
        if return_forward_info:
            return image, info
        else:
            return image

    @staticmethod
    def backward(ctx, grad_im, *args):
        global GAMMA_WARNING_EMITTED
        (
            vert_pos,
            vert_col,
            vert_rad,
            cam_pos,
            pixel_0_0_center,
            pixel_vec_x,
            pixel_vec_y,
            focal_length,
            principal_point_offsets,
            bg_col,
            opacity,
            image,
            info,
        ) = ctx.saved_tensors
        if (
            (
                ctx.needs_input_grad[0]
                or ctx.needs_input_grad[2]
                or ctx.needs_input_grad[3]
                or ctx.needs_input_grad[4]
                or ctx.needs_input_grad[5]
                or ctx.needs_input_grad[6]
                or ctx.needs_input_grad[7]
            )
            and ctx.gamma < 1e-3
            and not GAMMA_WARNING_EMITTED
        ):
            warnings.warn(
                "Optimizing for non-color parameters and having a gamma value < 1E-3! "
                "This is probably not going to produce usable gradients."
            )
            GAMMA_WARNING_EMITTED = True
        if ctx.mode == 0:
            (
                grad_pos,
                grad_col,
                grad_rad,
                grad_cam_pos,
                grad_pixel_0_0_center,
                grad_pixel_vec_x,
                grad_pixel_vec_y,
                grad_opacity,
            ) = ctx.native_renderer.backward(
                grad_im,
                image,
                info,
                vert_pos,
                vert_col,
                vert_rad,
                cam_pos,
                pixel_0_0_center,
                pixel_vec_x,
                pixel_vec_y,
                focal_length,
                principal_point_offsets,
                ctx.gamma,
                ctx.max_depth,
                ctx.min_depth,
                bg_col,
                opacity,
                ctx.percent_allowed_difference,
                ctx.max_n_hits,
                ctx.mode,
                ctx.needs_input_grad[0],
                ctx.needs_input_grad[1],
                ctx.needs_input_grad[2],
                ctx.needs_input_grad[3]
                or ctx.needs_input_grad[4]
                or ctx.needs_input_grad[5]
                or ctx.needs_input_grad[6]
                or ctx.needs_input_grad[7],
                ctx.needs_input_grad[14],
                None,  # No debug information provided.
            )
        else:
            raise ValueError(
                "Performing a backward pass for a "
                "rendering with `mode != 0`! This is not possible."
            )
        return (
            grad_pos,
            grad_col,
            grad_rad,
            grad_cam_pos,
            grad_pixel_0_0_center,
            grad_pixel_vec_x,
            grad_pixel_vec_y,
            None,  # focal_length
            None,  # principal_point_offsets
            None,  # gamma
            None,  # max_depth
            None,  # native_renderer
            None,  # min_depth
            None,  # bg_col
            grad_opacity,
            None,  # percent_allowed_difference
            None,  # max_n_hits
            None,  # mode
            None,  # return_forward_info
        )


class Renderer(torch.nn.Module):
    """
    Differentiable rendering module for the Pulsar renderer.

    Set the maximum number of balls to a reasonable value. It is used to determine
    several buffer sizes. It is no problem to render less balls than this number,
    but never more.

    When optimizing for sphere positions, sphere radiuses or camera parameters you
    have to use higher gamma values (closer to one) and larger sphere sizes: spheres
    can only 'move' to areas that they cover, and only with higher gamma values exists
    a gradient w.r.t. their color depending on their position.

    Args:
        * width: result image width in pixels.
        * height: result image height in pixels.
        * max_num_balls: the maximum number of balls this renderer will handle.
        * orthogonal_projection: use an orthogonal instead of perspective projection.
            Default: False.
        * right_handed_system: use a right-handed instead of a left-handed coordinate
            system. This is relevant for compatibility with other drawing or scanning
            systems. Pulsar by default assumes a left-handed world and camera coordinate
            system as known from mathematics with x-axis to the right, y axis up and z
            axis for increasing depth along the optical axis. In the image coordinate
            system, only the y axis is pointing down, leading still to a left-handed
            system. If you set this to True, it is assuming a right-handed world and
            camera coordinate system with x axis to the right, y axis to the top and
            z axis decreasing along the optical axis. Again, the image coordinate
            system has a flipped y axis, remaining a right-handed system.
            Default: False.
        * background_normalized_depth: the normalized depth the background is placed
            at.
            This is on a scale from 0. to 1. between the specified min and max depth
            (see the forward function). The value 0. is the most furthest depth whereas
            1. is the closest. Be careful when setting the background too far front - it
            may hide elements in your scene. Default: EPS.
        * n_channels: the number of image content channels to use. This is usually three
            for regular color representations, but can be a higher or lower number.
            Default: 3.
        * n_track: the number of spheres to track for gradient calculation per pixel.
            Only the closest n_track spheres will receive gradients. Default: 5.
    """

    def __init__(
        self,
        width: int,
        height: int,
        max_num_balls: int,
        orthogonal_projection: bool = False,
        right_handed_system: bool = False,
        # pyre-fixme[16]: Module `_C` has no attribute `EPS`.
        background_normalized_depth: float = _C.EPS,
        n_channels: int = 3,
        n_track: int = 5,
    ) -> None:
        super(Renderer, self).__init__()
        # pyre-fixme[16]: Module `pytorch3d` has no attribute `_C`.
        self._renderer = _C.PulsarRenderer(
            width,
            height,
            max_num_balls,
            orthogonal_projection,
            right_handed_system,
            background_normalized_depth,
            n_channels,
            n_track,
        )
        self.register_buffer("device_tracker", torch.zeros(1))

    @staticmethod
    def sphere_ids_from_result_info_nograd(result_info: torch.Tensor) -> torch.Tensor:
        """
        Get the sphere IDs from a result info tensor.
        """
        if result_info.ndim == 3:
            return Renderer.sphere_ids_from_result_info_nograd(result_info[None, ...])
        # pyre-fixme[16]: Module `pytorch3d` has no attribute `_C`.
        return _C.pulsar_sphere_ids_from_result_info_nograd(result_info)

    @staticmethod
    def depth_map_from_result_info_nograd(result_info: torch.Tensor) -> torch.Tensor:
        """
        Get the depth map from a result info tensor.

        This returns a map of the same size as the image with just one channel
        containing the closest intersection value at that position. Gradients
        are not available for this tensor, but do note that you can use
        `sphere_ids_from_result_info_nograd` to get the IDs of the spheres at
        each position and directly create a loss on their depth if required.

        The depth map contains -1. at positions where no intersection has
        been detected.
        """
        return result_info[..., 4]

    @staticmethod
    def _transform_cam_params(
        cam_params: torch.Tensor,
        width: int,
        height: int,
        orthogonal: bool,
        right_handed: bool,
        first_R_then_T: bool = False,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Transform 8 component camera parameter vector(s) to the internal camera
        representation.

        The input vectors consists of:
            * 3 components for camera position,
            * 3 components for camera rotation (three rotation angles) or
              6 components as described in "On the Continuity of Rotation
              Representations in Neural Networks" (Zhou et al.),
            * focal length,
            * the sensor width in world coordinates,
            * [optional] the principal point offset in x and y.

        The sensor height is inferred by pixel size and sensor width to obtain
        quadratic pixels.

        Args:
            * cam_params: [Bx]{8, 10, 11, 13}, input tensors as described above.
            * width: number of pixels in x direction.
            * height: number of pixels in y direction.
            * orthogonal: bool, whether an orthogonal projection is used
                  (does not use focal length).
            * right_handed: bool, whether to use a right handed system
                  (negative z in camera direction).
            * first_R_then_T: bool, whether to first rotate, then translate
                  the camera (PyTorch3D convention).

        Returns:
            * pos_vec: the position vector in 3D,
            * pixel_0_0_center: the center of the upper left pixel in world coordinates,
            * pixel_vec_x: the step to move one pixel on the image x axis
                   in world coordinates,
            * pixel_vec_y: the step to move one pixel on the image y axis
                   in world coordinates,
            * focal_length: the focal lengths,
            * principal_point_offsets: the principal point offsets in x, y.
        """
        global AXANGLE_WARNING_EMITTED
        # Set up all direction vectors, i.e., the sensor direction of all axes.
        assert width > 0
        assert height > 0
        batch_processing = True
        if cam_params.ndimension() == 1:
            batch_processing = False
            cam_params = cam_params[None, :]
        batch_size = cam_params.size(0)
        continuous_rep = True
        if cam_params.shape[1] in [8, 10]:
            if cam_params.requires_grad and not AXANGLE_WARNING_EMITTED:
                warnings.warn(
                    "Using an axis angle representation for camera rotations. "
                    "This has discontinuities and should not be used for optimization. "
                    "Alternatively, use a six-component representation as described in "
                    "'On the Continuity of Rotation Representations in Neural Networks'"
                    " (Zhou et al.). "
                    "The `pytorch3d.transforms` module provides "
                    "facilities for using this representation."
                )
                AXANGLE_WARNING_EMITTED = True
            continuous_rep = False
        else:
            assert cam_params.shape[1] in [11, 13]
        pos_vec: torch.Tensor = cam_params[:, :3]
        principal_point_offsets: torch.Tensor = torch.zeros(
            (cam_params.shape[0], 2), dtype=torch.int32, device=cam_params.device
        )
        if continuous_rep:
            rot_vec = cam_params[:, 3:9]
            focal_length: torch.Tensor = cam_params[:, 9:10]
            sensor_size_x = cam_params[:, 10:11]
            if cam_params.shape[1] == 13:
                principal_point_offsets: torch.Tensor = cam_params[:, 11:13].to(
                    torch.int32
                )
        else:
            rot_vec = cam_params[:, 3:6]
            focal_length: torch.Tensor = cam_params[:, 6:7]
            sensor_size_x = cam_params[:, 7:8]
            if cam_params.shape[1] == 10:
                principal_point_offsets: torch.Tensor = cam_params[:, 8:10].to(
                    torch.int32
                )
        # Always get quadratic pixels.
        pixel_size_x = sensor_size_x / float(width)
        sensor_size_y = height * pixel_size_x
        if continuous_rep:
            rot_mat = rotation_6d_to_matrix(rot_vec)
        else:
            rot_mat = axis_angle_to_matrix(rot_vec)
        if first_R_then_T:
            pos_vec = torch.matmul(rot_mat, pos_vec[..., None])[:, :, 0]
        sensor_dir_x = torch.matmul(
            rot_mat,
            torch.tensor(
                [1.0, 0.0, 0.0], dtype=torch.float32, device=rot_mat.device
            ).repeat(batch_size, 1)[:, :, None],
        )[:, :, 0]
        sensor_dir_y = torch.matmul(
            rot_mat,
            torch.tensor(
                [0.0, -1.0, 0.0], dtype=torch.float32, device=rot_mat.device
            ).repeat(batch_size, 1)[:, :, None],
        )[:, :, 0]
        sensor_dir_z = torch.matmul(
            rot_mat,
            torch.tensor(
                [0.0, 0.0, 1.0], dtype=torch.float32, device=rot_mat.device
            ).repeat(batch_size, 1)[:, :, None],
        )[:, :, 0]
        if right_handed:
            sensor_dir_z *= -1
        if orthogonal:
            sensor_center = pos_vec
        else:
            sensor_center = pos_vec + focal_length * sensor_dir_z
        sensor_luc = (  # Sensor left upper corner.
            sensor_center
            - sensor_dir_x * (sensor_size_x / 2.0)
            - sensor_dir_y * (sensor_size_y / 2.0)
        )
        pixel_size_x = sensor_size_x / float(width)
        pixel_size_y = sensor_size_y / float(height)
        pixel_vec_x: torch.Tensor = sensor_dir_x * pixel_size_x
        pixel_vec_y: torch.Tensor = sensor_dir_y * pixel_size_y
        pixel_0_0_center = sensor_luc + 0.5 * pixel_vec_x + 0.5 * pixel_vec_y
        # Reduce dimension.
        focal_length: torch.Tensor = focal_length[:, 0]
        if batch_processing:
            return (
                pos_vec,
                pixel_0_0_center,
                pixel_vec_x,
                pixel_vec_y,
                focal_length,
                principal_point_offsets,
            )
        else:
            return (
                pos_vec[0],
                pixel_0_0_center[0],
                pixel_vec_x[0],
                pixel_vec_y[0],
                focal_length[0],
                principal_point_offsets[0],
            )

    def forward(
        self,
        vert_pos: torch.Tensor,
        vert_col: torch.Tensor,
        vert_rad: torch.Tensor,
        cam_params: torch.Tensor,
        gamma: float,
        max_depth: float,
        min_depth: float = 0.0,
        bg_col: Optional[torch.Tensor] = None,
        opacity: Optional[torch.Tensor] = None,
        percent_allowed_difference: float = 0.01,
        # pyre-fixme[16]: Module `_C` has no attribute `MAX_UINT`.
        max_n_hits: int = _C.MAX_UINT,
        mode: int = 0,
        return_forward_info: bool = False,
        first_R_then_T: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """
        Rendering pass to create an image from the provided spheres and camera
        parameters.

        Args:
            * vert_pos: vertex positions. [Bx]Nx3 tensor of positions in 3D space.
            * vert_col: vertex colors. [Bx]NxK tensor of channels.
            * vert_rad: vertex radii. [Bx]N tensor of radiuses, >0.
            * cam_params: camera parameter(s). [Bx]8 tensor, consisting of:
                - 3 components for camera position,
                - 3 components for camera rotation (axis angle representation) or
                  6 components as described in "On the Continuity of Rotation
                  Representations in Neural Networks" (Zhou et al.),
                - focal length,
                - the sensor width in world coordinates,
                - [optional] an offset for the principal point in x, y (no gradients).
            * gamma: sphere transparency in [1.,1E-5], with 1 being mostly transparent.
                [Bx]1.
            * max_depth: maximum depth for spheres to render. Set this as tightly
                        as possible to have good numerical accuracy for gradients.
                        float > min_depth + eps.
            * min_depth: a float with the minimum depth a sphere must have to be
                        rendered. Must be 0. or > max(focal_length) + eps.
            * bg_col: K tensor with a background color to use or None (uses all ones).
            * opacity: [Bx]N tensor of opacity values in [0., 1.] or None (uses all
                    ones).
            * percent_allowed_difference: a float in [0., 1.[ with the maximum allowed
                        difference in color space. This is used to speed up the
                        computation. Default: 0.01.
            * max_n_hits: a hard limit on the number of hits per ray. Default: max int.
            * mode: render mode in {0, 1}. 0: render an image; 1: render the hit map.
            * return_forward_info: whether to return a second map. This second map
                contains 13 channels: first channel contains sm_m (the maximum
                exponent factor observed), the second sm_d (the normalization
                denominator, the sum of all coefficients), the third the maximum closest
                possible intersection for a hit. The following channels alternate with
                the float encoded integer index of a sphere and its weight. They are the
                five spheres with the highest color contribution to this pixel color,
                ordered descending. Default: False.
            * first_R_then_T: bool, whether to first apply rotation to the camera,
                then translation (PyTorch3D convention). Default: False.

        Returns:
            * image: [Bx]HxWx3 float tensor with the resulting image.
            * forw_info: [Bx]HxWx13 float forward information as described above, if
                    enabled.
        """
        # The device tracker is registered as buffer.
        self._renderer.device_tracker = self.device_tracker
        (
            pos_vec,
            pixel_0_0_center,
            pixel_vec_x,
            pixel_vec_y,
            focal_lengths,
            principal_point_offsets,
        ) = Renderer._transform_cam_params(
            cam_params,
            self._renderer.width,
            self._renderer.height,
            self._renderer.orthogonal,
            self._renderer.right_handed,
            first_R_then_T=first_R_then_T,
        )
        if (
            focal_lengths.min().item() > 0.0
            and max_depth > 10_000.0 * focal_lengths.min().item()
        ):
            warnings.warn(
                (
                    "Extreme ratio of `max_depth` vs. focal length detected "
                    "(%f vs. %f, ratio: %f). This will likely lead to "
                    "artifacts due to numerical instabilities."
                )
                % (
                    max_depth,
                    focal_lengths.min().item(),
                    max_depth / focal_lengths.min().item(),
                )
            )
        ret_res = _Render.apply(
            vert_pos,
            vert_col,
            vert_rad,
            pos_vec,
            pixel_0_0_center,
            pixel_vec_x,
            pixel_vec_y,
            # Focal length and sensor size don't need gradients other than through
            # `pixel_vec_x` and `pixel_vec_y`. The focal length is only used in the
            # renderer to determine the projection areas of the balls.
            focal_lengths,
            # principal_point_offsets does not receive gradients.
            principal_point_offsets,
            gamma,
            max_depth,
            self._renderer,
            min_depth,
            bg_col,
            opacity,
            percent_allowed_difference,
            max_n_hits,
            mode,
            (mode == 0) and return_forward_info,
        )
        if return_forward_info and mode != 0:
            return ret_res, None
        return ret_res

    def extra_repr(self) -> str:
        """Extra information to print in pytorch graphs."""
        return "width={}, height={}, max_num_balls={}".format(
            self._renderer.width, self._renderer.height, self._renderer.max_num_balls
        )
