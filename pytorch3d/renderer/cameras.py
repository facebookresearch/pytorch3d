# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from pytorch3d.common.datatypes import Device
from pytorch3d.transforms import Rotate, Transform3d, Translate

from .utils import convert_to_tensors_and_broadcast, TensorProperties


# Default values for rotation and translation matrices.
_R = torch.eye(3)[None]  # (1, 3, 3)
_T = torch.zeros(1, 3)  # (1, 3)

# An input which is a float per batch element
_BatchFloatType = Union[float, Sequence[float], torch.Tensor]

# one or two floats per batch element
_FocalLengthType = Union[
    float, Sequence[Tuple[float]], Sequence[Tuple[float, float]], torch.Tensor
]


class CamerasBase(TensorProperties):
    """
    `CamerasBase` implements a base class for all cameras.

    For cameras, there are four different coordinate systems (or spaces)
    - World coordinate system: This is the system the object lives - the world.
    - Camera view coordinate system: This is the system that has its origin on
        the camera and the Z-axis perpendicular to the image plane.
        In PyTorch3D, we assume that +X points left, and +Y points up and
        +Z points out from the image plane.
        The transformation from world --> view happens after applying a rotation (R)
        and translation (T)
    - NDC coordinate system: This is the normalized coordinate system that confines
        points in a volume the rendered part of the object or scene, also known as
        view volume. For square images, given the PyTorch3D convention, (+1, +1, znear)
        is the top left near corner, and (-1, -1, zfar) is the bottom right far
        corner of the volume.
        The transformation from view --> NDC happens after applying the camera
        projection matrix (P) if defined in NDC space.
        For non square images, we scale the points such that smallest side
        has range [-1, 1] and the largest side has range [-u, u], with u > 1.
    - Screen coordinate system: This is another representation of the view volume with
        the XY coordinates defined in image space instead of a normalized space.

    An illustration of the coordinate systems can be found in pytorch3d/docs/notes/cameras.md.

    CameraBase defines methods that are common to all camera models:
        - `get_camera_center` that returns the optical center of the camera in
            world coordinates
        - `get_world_to_view_transform` which returns a 3D transform from
            world coordinates to the camera view coordinates (R, T)
        - `get_full_projection_transform` which composes the projection
            transform (P) with the world-to-view transform (R, T)
        - `transform_points` which takes a set of input points in world coordinates and
            projects to the space the camera is defined in (NDC or screen)
        - `get_ndc_camera_transform` which defines the transform from screen/NDC to
            PyTorch3D's NDC space
        - `transform_points_ndc` which takes a set of points in world coordinates and
            projects them to PyTorch3D's NDC space
        - `transform_points_screen` which takes a set of points in world coordinates and
            projects them to screen space

    For each new camera, one should implement the `get_projection_transform`
    routine that returns the mapping from camera view coordinates to camera
    coordinates (NDC or screen).

    Another useful function that is specific to each camera model is
    `unproject_points` which sends points from camera coordinates (NDC or screen)
    back to camera view or world coordinates depending on the `world_coordinates`
    boolean argument of the function.
    """

    # Used in __getitem__ to index the relevant fields
    # When creating a new camera, this should be set in the __init__
    _FIELDS: Tuple[str, ...] = ()

    # Names of fields which are a constant property of the whole batch, rather
    # than themselves a batch of data.
    # When joining objects into a batch, they will have to agree.
    _SHARED_FIELDS: Tuple[str, ...] = ()

    def get_projection_transform(self, **kwargs):
        """
        Calculate the projective transformation matrix.

        Args:
            **kwargs: parameters for the projection can be passed in as keyword
                arguments to override the default values set in `__init__`.

        Return:
            a `Transform3d` object which represents a batch of projection
            matrices of shape (N, 3, 3)
        """
        raise NotImplementedError()

    def unproject_points(self, xy_depth: torch.Tensor, **kwargs):
        """
        Transform input points from camera coordinates (NDC or screen)
        to the world / camera coordinates.

        Each of the input points `xy_depth` of shape (..., 3) is
        a concatenation of the x, y location and its depth.

        For instance, for an input 2D tensor of shape `(num_points, 3)`
        `xy_depth` takes the following form:
            `xy_depth[i] = [x[i], y[i], depth[i]]`,
        for a each point at an index `i`.

        The following example demonstrates the relationship between
        `transform_points` and `unproject_points`:

        .. code-block:: python

            cameras = # camera object derived from CamerasBase
            xyz = # 3D points of shape (batch_size, num_points, 3)
            # transform xyz to the camera view coordinates
            xyz_cam = cameras.get_world_to_view_transform().transform_points(xyz)
            # extract the depth of each point as the 3rd coord of xyz_cam
            depth = xyz_cam[:, :, 2:]
            # project the points xyz to the camera
            xy = cameras.transform_points(xyz)[:, :, :2]
            # append depth to xy
            xy_depth = torch.cat((xy, depth), dim=2)
            # unproject to the world coordinates
            xyz_unproj_world = cameras.unproject_points(xy_depth, world_coordinates=True)
            print(torch.allclose(xyz, xyz_unproj_world)) # True
            # unproject to the camera coordinates
            xyz_unproj = cameras.unproject_points(xy_depth, world_coordinates=False)
            print(torch.allclose(xyz_cam, xyz_unproj)) # True

        Args:
            xy_depth: torch tensor of shape (..., 3).
            world_coordinates: If `True`, unprojects the points back to world
                coordinates using the camera extrinsics `R` and `T`.
                `False` ignores `R` and `T` and unprojects to
                the camera view coordinates.
            from_ndc: If `False` (default), assumes xy part of input is in
                NDC space if self.in_ndc(), otherwise in screen space. If
                `True`, assumes xy is in NDC space even if the camera
                is defined in screen space.

        Returns
            new_points: unprojected points with the same shape as `xy_depth`.
        """
        raise NotImplementedError()

    def get_camera_center(self, **kwargs) -> torch.Tensor:
        """
        Return the 3D location of the camera optical center
        in the world coordinates.

        Args:
            **kwargs: parameters for the camera extrinsics can be passed in
                as keyword arguments to override the default values
                set in __init__.

        Setting R or T here will update the values set in init as these
        values may be needed later on in the rendering pipeline e.g. for
        lighting calculations.

        Returns:
            C: a batch of 3D locations of shape (N, 3) denoting
            the locations of the center of each camera in the batch.
        """
        w2v_trans = self.get_world_to_view_transform(**kwargs)
        P = w2v_trans.inverse().get_matrix()
        # the camera center is the translation component (the first 3 elements
        # of the last row) of the inverted world-to-view
        # transform (4x4 RT matrix)
        C = P[:, 3, :3]
        return C

    def get_world_to_view_transform(self, **kwargs) -> Transform3d:
        """
        Return the world-to-view transform.

        Args:
            **kwargs: parameters for the camera extrinsics can be passed in
                as keyword arguments to override the default values
                set in __init__.

        Setting R and T here will update the values set in init as these
        values may be needed later on in the rendering pipeline e.g. for
        lighting calculations.

        Returns:
            A Transform3d object which represents a batch of transforms
            of shape (N, 3, 3)
        """
        R: torch.Tensor = kwargs.get("R", self.R)
        T: torch.Tensor = kwargs.get("T", self.T)
        self.R = R
        self.T = T
        world_to_view_transform = get_world_to_view_transform(R=R, T=T)
        return world_to_view_transform

    def get_full_projection_transform(self, **kwargs) -> Transform3d:
        """
        Return the full world-to-camera transform composing the
        world-to-view and view-to-camera transforms.
        If camera is defined in NDC space, the projected points are in NDC space.
        If camera is defined in screen space, the projected points are in screen space.

        Args:
            **kwargs: parameters for the projection transforms can be passed in
                as keyword arguments to override the default values
                set in __init__.

        Setting R and T here will update the values set in init as these
        values may be needed later on in the rendering pipeline e.g. for
        lighting calculations.

        Returns:
            a Transform3d object which represents a batch of transforms
            of shape (N, 3, 3)
        """
        self.R: torch.Tensor = kwargs.get("R", self.R)
        self.T: torch.Tensor = kwargs.get("T", self.T)
        world_to_view_transform = self.get_world_to_view_transform(R=self.R, T=self.T)
        view_to_proj_transform = self.get_projection_transform(**kwargs)
        return world_to_view_transform.compose(view_to_proj_transform)

    def transform_points(
        self, points, eps: Optional[float] = None, **kwargs
    ) -> torch.Tensor:
        """
        Transform input points from world to camera space.
        If camera is defined in NDC space, the projected points are in NDC space.
        If camera is defined in screen space, the projected points are in screen space.

        For `CamerasBase.transform_points`, setting `eps > 0`
        stabilizes gradients since it leads to avoiding division
        by excessively low numbers for points close to the camera plane.

        Args:
            points: torch tensor of shape (..., 3).
            eps: If eps!=None, the argument is used to clamp the
                divisor in the homogeneous normalization of the points
                transformed to the ndc space. Please see
                `transforms.Transform3d.transform_points` for details.

                For `CamerasBase.transform_points`, setting `eps > 0`
                stabilizes gradients since it leads to avoiding division
                by excessively low numbers for points close to the
                camera plane.

        Returns
            new_points: transformed points with the same shape as the input.
        """
        world_to_proj_transform = self.get_full_projection_transform(**kwargs)
        return world_to_proj_transform.transform_points(points, eps=eps)

    def get_ndc_camera_transform(self, **kwargs) -> Transform3d:
        """
        Returns the transform from camera projection space (screen or NDC) to NDC space.
        For cameras that can be specified in screen space, this transform
        allows points to be converted from screen to NDC space.
        The default transform scales the points from [0, W]x[0, H]
        to [-1, 1]x[-u, u] or [-u, u]x[-1, 1] where u > 1 is the aspect ratio of the image.
        This function should be modified per camera definitions if need be,
        e.g. for Perspective/Orthographic cameras we provide a custom implementation.
        This transform assumes PyTorch3D coordinate system conventions for
        both the NDC space and the input points.

        This transform interfaces with the PyTorch3D renderer which assumes
        input points to the renderer to be in NDC space.
        """
        if self.in_ndc():
            return Transform3d(device=self.device, dtype=torch.float32)
        else:
            # For custom cameras which can be defined in screen space,
            # users might might have to implement the screen to NDC transform based
            # on the definition of the camera parameters.
            # See PerspectiveCameras/OrthographicCameras for an example.
            # We don't flip xy because we assume that world points are in
            # PyTorch3D coordinates, and thus conversion from screen to ndc
            # is a mere scaling from image to [-1, 1] scale.
            image_size = kwargs.get("image_size", self.get_image_size())
            return get_screen_to_ndc_transform(
                self, with_xyflip=False, image_size=image_size
            )

    def transform_points_ndc(
        self, points, eps: Optional[float] = None, **kwargs
    ) -> torch.Tensor:
        """
        Transforms points from PyTorch3D world/camera space to NDC space.
        Input points follow the PyTorch3D coordinate system conventions: +X left, +Y up.
        Output points are in NDC space: +X left, +Y up, origin at image center.

        Args:
            points: torch tensor of shape (..., 3).
            eps: If eps!=None, the argument is used to clamp the
                divisor in the homogeneous normalization of the points
                transformed to the ndc space. Please see
                `transforms.Transform3d.transform_points` for details.

                For `CamerasBase.transform_points`, setting `eps > 0`
                stabilizes gradients since it leads to avoiding division
                by excessively low numbers for points close to the
                camera plane.

        Returns
            new_points: transformed points with the same shape as the input.
        """
        world_to_ndc_transform = self.get_full_projection_transform(**kwargs)
        if not self.in_ndc():
            to_ndc_transform = self.get_ndc_camera_transform(**kwargs)
            world_to_ndc_transform = world_to_ndc_transform.compose(to_ndc_transform)

        return world_to_ndc_transform.transform_points(points, eps=eps)

    def transform_points_screen(
        self, points, eps: Optional[float] = None, with_xyflip: bool = True, **kwargs
    ) -> torch.Tensor:
        """
        Transforms points from PyTorch3D world/camera space to screen space.
        Input points follow the PyTorch3D coordinate system conventions: +X left, +Y up.
        Output points are in screen space: +X right, +Y down, origin at top left corner.

        Args:
            points: torch tensor of shape (..., 3).
            eps: If eps!=None, the argument is used to clamp the
                divisor in the homogeneous normalization of the points
                transformed to the ndc space. Please see
                `transforms.Transform3d.transform_points` for details.

                For `CamerasBase.transform_points`, setting `eps > 0`
                stabilizes gradients since it leads to avoiding division
                by excessively low numbers for points close to the
                camera plane.
            with_xyflip: If True, flip x and y directions. In world/camera/ndc coords,
                +x points to the left and +y up. If with_xyflip is true, in screen
                coords +x points right, and +y down, following the usual RGB image
                convention. Warning: do not set to False unless you know what you're
                doing!

        Returns
            new_points: transformed points with the same shape as the input.
        """
        points_ndc = self.transform_points_ndc(points, eps=eps, **kwargs)
        image_size = kwargs.get("image_size", self.get_image_size())
        return get_ndc_to_screen_transform(
            self, with_xyflip=with_xyflip, image_size=image_size
        ).transform_points(points_ndc, eps=eps)

    def clone(self):
        """
        Returns a copy of `self`.
        """
        cam_type = type(self)
        other = cam_type(device=self.device)
        return super().clone(other)

    def is_perspective(self):
        raise NotImplementedError()

    def in_ndc(self):
        """
        Specifies whether the camera is defined in NDC space
        or in screen (image) space
        """
        raise NotImplementedError()

    def get_znear(self):
        return getattr(self, "znear", None)

    def get_image_size(self):
        """
        Returns the image size, if provided, expected in the form of (height, width)
        The image size is used for conversion of projected points to screen coordinates.
        """
        return getattr(self, "image_size", None)

    def __getitem__(
        self, index: Union[int, List[int], torch.BoolTensor, torch.LongTensor]
    ) -> "CamerasBase":
        """
        Override for the __getitem__ method in TensorProperties which needs to be
        refactored.

        Args:
            index: an integer index, list/tensor of integer indices, or tensor of boolean
                indicators used to filter all the fields in the cameras given by self._FIELDS.
        Returns:
            an instance of the current cameras class with only the values at the selected index.
        """

        kwargs = {}

        tensor_types = {
            # pyre-fixme[16]: Module `cuda` has no attribute `BoolTensor`.
            "bool": (torch.BoolTensor, torch.cuda.BoolTensor),
            # pyre-fixme[16]: Module `cuda` has no attribute `LongTensor`.
            "long": (torch.LongTensor, torch.cuda.LongTensor),
        }
        if not isinstance(
            index, (int, list, *tensor_types["bool"], *tensor_types["long"])
        ) or (
            isinstance(index, list)
            and not all(isinstance(i, int) and not isinstance(i, bool) for i in index)
        ):
            msg = (
                "Invalid index type, expected int, List[int] or Bool/LongTensor; got %r"
            )
            raise ValueError(msg % type(index))

        if isinstance(index, int):
            index = [index]

        if isinstance(index, tensor_types["bool"]):
            # pyre-fixme[16]: Item `List` of `Union[List[int], BoolTensor,
            #  LongTensor]` has no attribute `ndim`.
            # pyre-fixme[16]: Item `List` of `Union[List[int], BoolTensor,
            #  LongTensor]` has no attribute `shape`.
            if index.ndim != 1 or index.shape[0] != len(self):
                raise ValueError(
                    # pyre-fixme[16]: Item `List` of `Union[List[int], BoolTensor,
                    #  LongTensor]` has no attribute `shape`.
                    f"Boolean index of shape {index.shape} does not match cameras"
                )
        elif max(index) >= len(self):
            raise IndexError(f"Index {max(index)} is out of bounds for select cameras")

        for field in self._FIELDS:
            val = getattr(self, field, None)
            if val is None:
                continue

            # e.g. "in_ndc" is set as attribute "_in_ndc" on the class
            # but provided as "in_ndc" on initialization
            if field.startswith("_"):
                field = field[1:]

            if isinstance(val, (str, bool)):
                kwargs[field] = val
            elif isinstance(val, torch.Tensor):
                # In the init, all inputs will be converted to
                # tensors before setting as attributes
                kwargs[field] = val[index]
            else:
                raise ValueError(f"Field {field} type is not supported for indexing")

        kwargs["device"] = self.device
        return self.__class__(**kwargs)


############################################################
#             Field of View Camera Classes                 #
############################################################


def OpenGLPerspectiveCameras(
    znear: _BatchFloatType = 1.0,
    zfar: _BatchFloatType = 100.0,
    aspect_ratio: _BatchFloatType = 1.0,
    fov: _BatchFloatType = 60.0,
    degrees: bool = True,
    R: torch.Tensor = _R,
    T: torch.Tensor = _T,
    device: Device = "cpu",
) -> "FoVPerspectiveCameras":
    """
    OpenGLPerspectiveCameras has been DEPRECATED. Use FoVPerspectiveCameras instead.
    Preserving OpenGLPerspectiveCameras for backward compatibility.
    """

    warnings.warn(
        """OpenGLPerspectiveCameras is deprecated,
        Use FoVPerspectiveCameras instead.
        OpenGLPerspectiveCameras will be removed in future releases.""",
        PendingDeprecationWarning,
    )

    return FoVPerspectiveCameras(
        znear=znear,
        zfar=zfar,
        aspect_ratio=aspect_ratio,
        fov=fov,
        degrees=degrees,
        R=R,
        T=T,
        device=device,
    )


class FoVPerspectiveCameras(CamerasBase):
    """
    A class which stores a batch of parameters to generate a batch of
    projection matrices by specifying the field of view.
    The definitions of the parameters follow the OpenGL perspective camera.

    The extrinsics of the camera (R and T matrices) can also be set in the
    initializer or passed in to `get_full_projection_transform` to get
    the full transformation from world -> ndc.

    The `transform_points` method calculates the full world -> ndc transform
    and then applies it to the input points.

    The transforms can also be returned separately as Transform3d objects.

    * Setting the Aspect Ratio for Non Square Images *

    If the desired output image size is non square (i.e. a tuple of (H, W) where H != W)
    the aspect ratio needs special consideration: There are two aspect ratios
    to be aware of:
        - the aspect ratio of each pixel
        - the aspect ratio of the output image
    The `aspect_ratio` setting in the FoVPerspectiveCameras sets the
    pixel aspect ratio. When using this camera with the differentiable rasterizer
    be aware that in the rasterizer we assume square pixels, but allow
    variable image aspect ratio (i.e rectangle images).

    In most cases you will want to set the camera `aspect_ratio=1.0`
    (i.e. square pixels) and only vary the output image dimensions in pixels
    for rasterization.
    """

    # For __getitem__
    _FIELDS = (
        "K",
        "znear",
        "zfar",
        "aspect_ratio",
        "fov",
        "R",
        "T",
        "degrees",
    )

    _SHARED_FIELDS = ("degrees",)

    def __init__(
        self,
        znear: _BatchFloatType = 1.0,
        zfar: _BatchFloatType = 100.0,
        aspect_ratio: _BatchFloatType = 1.0,
        fov: _BatchFloatType = 60.0,
        degrees: bool = True,
        R: torch.Tensor = _R,
        T: torch.Tensor = _T,
        K: Optional[torch.Tensor] = None,
        device: Device = "cpu",
    ) -> None:
        """

        Args:
            znear: near clipping plane of the view frustrum.
            zfar: far clipping plane of the view frustrum.
            aspect_ratio: aspect ratio of the image pixels.
                1.0 indicates square pixels.
            fov: field of view angle of the camera.
            degrees: bool, set to True if fov is specified in degrees.
            R: Rotation matrix of shape (N, 3, 3)
            T: Translation matrix of shape (N, 3)
            K: (optional) A calibration matrix of shape (N, 4, 4)
                If provided, don't need znear, zfar, fov, aspect_ratio, degrees
            device: Device (as str or torch.device)
        """
        # The initializer formats all inputs to torch tensors and broadcasts
        # all the inputs to have the same batch dimension where necessary.
        super().__init__(
            device=device,
            znear=znear,
            zfar=zfar,
            aspect_ratio=aspect_ratio,
            fov=fov,
            R=R,
            T=T,
            K=K,
        )

        # No need to convert to tensor or broadcast.
        self.degrees = degrees

    def compute_projection_matrix(
        self, znear, zfar, fov, aspect_ratio, degrees: bool
    ) -> torch.Tensor:
        """
        Compute the calibration matrix K of shape (N, 4, 4)

        Args:
            znear: near clipping plane of the view frustrum.
            zfar: far clipping plane of the view frustrum.
            fov: field of view angle of the camera.
            aspect_ratio: aspect ratio of the image pixels.
                1.0 indicates square pixels.
            degrees: bool, set to True if fov is specified in degrees.

        Returns:
            torch.FloatTensor of the calibration matrix with shape (N, 4, 4)
        """
        K = torch.zeros((self._N, 4, 4), device=self.device, dtype=torch.float32)
        ones = torch.ones((self._N), dtype=torch.float32, device=self.device)
        if degrees:
            fov = (np.pi / 180) * fov

        if not torch.is_tensor(fov):
            fov = torch.tensor(fov, device=self.device)
        tanHalfFov = torch.tan((fov / 2))
        max_y = tanHalfFov * znear
        min_y = -max_y
        max_x = max_y * aspect_ratio
        min_x = -max_x

        # NOTE: In OpenGL the projection matrix changes the handedness of the
        # coordinate frame. i.e the NDC space positive z direction is the
        # camera space negative z direction. This is because the sign of the z
        # in the projection matrix is set to -1.0.
        # In pytorch3d we maintain a right handed coordinate system throughout
        # so the so the z sign is 1.0.
        z_sign = 1.0

        # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
        K[:, 0, 0] = 2.0 * znear / (max_x - min_x)
        # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
        K[:, 1, 1] = 2.0 * znear / (max_y - min_y)
        K[:, 0, 2] = (max_x + min_x) / (max_x - min_x)
        K[:, 1, 2] = (max_y + min_y) / (max_y - min_y)
        K[:, 3, 2] = z_sign * ones

        # NOTE: This maps the z coordinate from [0, 1] where z = 0 if the point
        # is at the near clipping plane and z = 1 when the point is at the far
        # clipping plane.
        K[:, 2, 2] = z_sign * zfar / (zfar - znear)
        K[:, 2, 3] = -(zfar * znear) / (zfar - znear)

        return K

    def get_projection_transform(self, **kwargs) -> Transform3d:
        """
        Calculate the perspective projection matrix with a symmetric
        viewing frustrum. Use column major order.
        The viewing frustrum will be projected into ndc, s.t.
        (max_x, max_y) -> (+1, +1)
        (min_x, min_y) -> (-1, -1)

        Args:
            **kwargs: parameters for the projection can be passed in as keyword
                arguments to override the default values set in `__init__`.

        Return:
            a Transform3d object which represents a batch of projection
            matrices of shape (N, 4, 4)

        .. code-block:: python

            h1 = (max_y + min_y)/(max_y - min_y)
            w1 = (max_x + min_x)/(max_x - min_x)
            tanhalffov = tan((fov/2))
            s1 = 1/tanhalffov
            s2 = 1/(tanhalffov * (aspect_ratio))

            # To map z to the range [0, 1] use:
            f1 =  far / (far - near)
            f2 = -(far * near) / (far - near)

            # Projection matrix
            K = [
                    [s1,   0,   w1,   0],
                    [0,   s2,   h1,   0],
                    [0,    0,   f1,  f2],
                    [0,    0,    1,   0],
            ]
        """
        K = kwargs.get("K", self.K)
        if K is not None:
            if K.shape != (self._N, 4, 4):
                msg = "Expected K to have shape of (%r, 4, 4)"
                raise ValueError(msg % (self._N))
        else:
            K = self.compute_projection_matrix(
                kwargs.get("znear", self.znear),
                kwargs.get("zfar", self.zfar),
                kwargs.get("fov", self.fov),
                kwargs.get("aspect_ratio", self.aspect_ratio),
                kwargs.get("degrees", self.degrees),
            )

        # Transpose the projection matrix as PyTorch3D transforms use row vectors.
        transform = Transform3d(
            matrix=K.transpose(1, 2).contiguous(), device=self.device
        )
        return transform

    def unproject_points(
        self,
        xy_depth: torch.Tensor,
        world_coordinates: bool = True,
        scaled_depth_input: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """>!
        FoV cameras further allow for passing depth in world units
        (`scaled_depth_input=False`) or in the [0, 1]-normalized units
        (`scaled_depth_input=True`)

        Args:
            scaled_depth_input: If `True`, assumes the input depth is in
                the [0, 1]-normalized units. If `False` the input depth is in
                the world units.
        """

        # obtain the relevant transformation to ndc
        if world_coordinates:
            to_ndc_transform = self.get_full_projection_transform()
        else:
            to_ndc_transform = self.get_projection_transform()

        if scaled_depth_input:
            # the input is scaled depth, so we don't have to do anything
            xy_sdepth = xy_depth
        else:
            # parse out important values from the projection matrix
            K_matrix = self.get_projection_transform(**kwargs.copy()).get_matrix()
            # parse out f1, f2 from K_matrix
            unsqueeze_shape = [1] * xy_depth.dim()
            unsqueeze_shape[0] = K_matrix.shape[0]
            f1 = K_matrix[:, 2, 2].reshape(unsqueeze_shape)
            f2 = K_matrix[:, 3, 2].reshape(unsqueeze_shape)
            # get the scaled depth
            sdepth = (f1 * xy_depth[..., 2:3] + f2) / xy_depth[..., 2:3]
            # concatenate xy + scaled depth
            xy_sdepth = torch.cat((xy_depth[..., 0:2], sdepth), dim=-1)

        # unproject with inverse of the projection
        unprojection_transform = to_ndc_transform.inverse()
        return unprojection_transform.transform_points(xy_sdepth)

    def is_perspective(self):
        return True

    def in_ndc(self):
        return True


def OpenGLOrthographicCameras(
    znear: _BatchFloatType = 1.0,
    zfar: _BatchFloatType = 100.0,
    top: _BatchFloatType = 1.0,
    bottom: _BatchFloatType = -1.0,
    left: _BatchFloatType = -1.0,
    right: _BatchFloatType = 1.0,
    scale_xyz=((1.0, 1.0, 1.0),),  # (1, 3)
    R: torch.Tensor = _R,
    T: torch.Tensor = _T,
    device: Device = "cpu",
) -> "FoVOrthographicCameras":
    """
    OpenGLOrthographicCameras has been DEPRECATED. Use FoVOrthographicCameras instead.
    Preserving OpenGLOrthographicCameras for backward compatibility.
    """

    warnings.warn(
        """OpenGLOrthographicCameras is deprecated,
        Use FoVOrthographicCameras instead.
        OpenGLOrthographicCameras will be removed in future releases.""",
        PendingDeprecationWarning,
    )

    return FoVOrthographicCameras(
        znear=znear,
        zfar=zfar,
        max_y=top,
        min_y=bottom,
        max_x=right,
        min_x=left,
        scale_xyz=scale_xyz,
        R=R,
        T=T,
        device=device,
    )


class FoVOrthographicCameras(CamerasBase):
    """
    A class which stores a batch of parameters to generate a batch of
    projection matrices by specifying the field of view.
    The definitions of the parameters follow the OpenGL orthographic camera.
    """

    # For __getitem__
    _FIELDS = (
        "K",
        "znear",
        "zfar",
        "R",
        "T",
        "max_y",
        "min_y",
        "max_x",
        "min_x",
        "scale_xyz",
    )

    def __init__(
        self,
        znear: _BatchFloatType = 1.0,
        zfar: _BatchFloatType = 100.0,
        max_y: _BatchFloatType = 1.0,
        min_y: _BatchFloatType = -1.0,
        max_x: _BatchFloatType = 1.0,
        min_x: _BatchFloatType = -1.0,
        scale_xyz=((1.0, 1.0, 1.0),),  # (1, 3)
        R: torch.Tensor = _R,
        T: torch.Tensor = _T,
        K: Optional[torch.Tensor] = None,
        device: Device = "cpu",
    ):
        """

        Args:
            znear: near clipping plane of the view frustrum.
            zfar: far clipping plane of the view frustrum.
            max_y: maximum y coordinate of the frustrum.
            min_y: minimum y coordinate of the frustrum.
            max_x: maximum x coordinate of the frustrum.
            min_x: minimum x coordinate of the frustrum
            scale_xyz: scale factors for each axis of shape (N, 3).
            R: Rotation matrix of shape (N, 3, 3).
            T: Translation of shape (N, 3).
            K: (optional) A calibration matrix of shape (N, 4, 4)
                If provided, don't need znear, zfar, max_y, min_y, max_x, min_x, scale_xyz
            device: torch.device or string.

        Only need to set min_x, max_x, min_y, max_y for viewing frustrums
        which are non symmetric about the origin.
        """
        # The initializer formats all inputs to torch tensors and broadcasts
        # all the inputs to have the same batch dimension where necessary.
        super().__init__(
            device=device,
            znear=znear,
            zfar=zfar,
            max_y=max_y,
            min_y=min_y,
            max_x=max_x,
            min_x=min_x,
            scale_xyz=scale_xyz,
            R=R,
            T=T,
            K=K,
        )

    def compute_projection_matrix(
        self, znear, zfar, max_x, min_x, max_y, min_y, scale_xyz
    ) -> torch.Tensor:
        """
        Compute the calibration matrix K of shape (N, 4, 4)

        Args:
            znear: near clipping plane of the view frustrum.
            zfar: far clipping plane of the view frustrum.
            max_x: maximum x coordinate of the frustrum.
            min_x: minimum x coordinate of the frustrum
            max_y: maximum y coordinate of the frustrum.
            min_y: minimum y coordinate of the frustrum.
            scale_xyz: scale factors for each axis of shape (N, 3).
        """
        K = torch.zeros((self._N, 4, 4), dtype=torch.float32, device=self.device)
        ones = torch.ones((self._N), dtype=torch.float32, device=self.device)
        # NOTE: OpenGL flips handedness of coordinate system between camera
        # space and NDC space so z sign is -ve. In PyTorch3D we maintain a
        # right handed coordinate system throughout.
        z_sign = +1.0

        K[:, 0, 0] = (2.0 / (max_x - min_x)) * scale_xyz[:, 0]
        K[:, 1, 1] = (2.0 / (max_y - min_y)) * scale_xyz[:, 1]
        K[:, 0, 3] = -(max_x + min_x) / (max_x - min_x)
        K[:, 1, 3] = -(max_y + min_y) / (max_y - min_y)
        K[:, 3, 3] = ones

        # NOTE: This maps the z coordinate to the range [0, 1] and replaces the
        # the OpenGL z normalization to [-1, 1]
        K[:, 2, 2] = z_sign * (1.0 / (zfar - znear)) * scale_xyz[:, 2]
        K[:, 2, 3] = -znear / (zfar - znear)

        return K

    def get_projection_transform(self, **kwargs) -> Transform3d:
        """
        Calculate the orthographic projection matrix.
        Use column major order.

        Args:
            **kwargs: parameters for the projection can be passed in to
                      override the default values set in __init__.
        Return:
            a Transform3d object which represents a batch of projection
               matrices of shape (N, 4, 4)

        .. code-block:: python

            scale_x = 2 / (max_x - min_x)
            scale_y = 2 / (max_y - min_y)
            scale_z = 2 / (far-near)
            mid_x = (max_x + min_x) / (max_x - min_x)
            mix_y = (max_y + min_y) / (max_y - min_y)
            mid_z = (far + near) / (far - near)

            K = [
                    [scale_x,        0,         0,  -mid_x],
                    [0,        scale_y,         0,  -mix_y],
                    [0,              0,  -scale_z,  -mid_z],
                    [0,              0,         0,       1],
            ]
        """
        K = kwargs.get("K", self.K)
        if K is not None:
            if K.shape != (self._N, 4, 4):
                msg = "Expected K to have shape of (%r, 4, 4)"
                raise ValueError(msg % (self._N))
        else:
            K = self.compute_projection_matrix(
                kwargs.get("znear", self.znear),
                kwargs.get("zfar", self.zfar),
                kwargs.get("max_x", self.max_x),
                kwargs.get("min_x", self.min_x),
                kwargs.get("max_y", self.max_y),
                kwargs.get("min_y", self.min_y),
                kwargs.get("scale_xyz", self.scale_xyz),
            )

        transform = Transform3d(
            matrix=K.transpose(1, 2).contiguous(), device=self.device
        )
        return transform

    def unproject_points(
        self,
        xy_depth: torch.Tensor,
        world_coordinates: bool = True,
        scaled_depth_input: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """>!
        FoV cameras further allow for passing depth in world units
        (`scaled_depth_input=False`) or in the [0, 1]-normalized units
        (`scaled_depth_input=True`)

        Args:
            scaled_depth_input: If `True`, assumes the input depth is in
                the [0, 1]-normalized units. If `False` the input depth is in
                the world units.
        """

        if world_coordinates:
            to_ndc_transform = self.get_full_projection_transform(**kwargs.copy())
        else:
            to_ndc_transform = self.get_projection_transform(**kwargs.copy())

        if scaled_depth_input:
            # the input depth is already scaled
            xy_sdepth = xy_depth
        else:
            # we have to obtain the scaled depth first
            K = self.get_projection_transform(**kwargs).get_matrix()
            unsqueeze_shape = [1] * K.dim()
            unsqueeze_shape[0] = K.shape[0]
            mid_z = K[:, 3, 2].reshape(unsqueeze_shape)
            scale_z = K[:, 2, 2].reshape(unsqueeze_shape)
            scaled_depth = scale_z * xy_depth[..., 2:3] + mid_z
            # cat xy and scaled depth
            xy_sdepth = torch.cat((xy_depth[..., :2], scaled_depth), dim=-1)
        # finally invert the transform
        unprojection_transform = to_ndc_transform.inverse()
        return unprojection_transform.transform_points(xy_sdepth)

    def is_perspective(self):
        return False

    def in_ndc(self):
        return True


############################################################
#             MultiView Camera Classes                     #
############################################################
"""
Note that the MultiView Cameras accept parameters in NDC space.
"""


def SfMPerspectiveCameras(
    focal_length: _FocalLengthType = 1.0,
    principal_point=((0.0, 0.0),),
    R: torch.Tensor = _R,
    T: torch.Tensor = _T,
    device: Device = "cpu",
) -> "PerspectiveCameras":
    """
    SfMPerspectiveCameras has been DEPRECATED. Use PerspectiveCameras instead.
    Preserving SfMPerspectiveCameras for backward compatibility.
    """

    warnings.warn(
        """SfMPerspectiveCameras is deprecated,
        Use PerspectiveCameras instead.
        SfMPerspectiveCameras will be removed in future releases.""",
        PendingDeprecationWarning,
    )

    return PerspectiveCameras(
        focal_length=focal_length,
        principal_point=principal_point,
        R=R,
        T=T,
        device=device,
    )


class PerspectiveCameras(CamerasBase):
    """
    A class which stores a batch of parameters to generate a batch of
    transformation matrices using the multi-view geometry convention for
    perspective camera.

    Parameters for this camera are specified in NDC if `in_ndc` is set to True.
    If parameters are specified in screen space, `in_ndc` must be set to False.
    """

    # For __getitem__
    _FIELDS = (
        "K",
        "R",
        "T",
        "focal_length",
        "principal_point",
        "_in_ndc",  # arg is in_ndc but attribute set as _in_ndc
        "image_size",
    )

    _SHARED_FIELDS = ("_in_ndc",)

    def __init__(
        self,
        focal_length: _FocalLengthType = 1.0,
        principal_point=((0.0, 0.0),),
        R: torch.Tensor = _R,
        T: torch.Tensor = _T,
        K: Optional[torch.Tensor] = None,
        device: Device = "cpu",
        in_ndc: bool = True,
        image_size: Optional[Union[List, Tuple, torch.Tensor]] = None,
    ) -> None:
        """

        Args:
            focal_length: Focal length of the camera in world units.
                A tensor of shape (N, 1) or (N, 2) for
                square and non-square pixels respectively.
            principal_point: xy coordinates of the center of
                the principal point of the camera in pixels.
                A tensor of shape (N, 2).
            in_ndc: True if camera parameters are specified in NDC.
                If camera parameters are in screen space, it must
                be set to False.
            R: Rotation matrix of shape (N, 3, 3)
            T: Translation matrix of shape (N, 3)
            K: (optional) A calibration matrix of shape (N, 4, 4)
                If provided, don't need focal_length, principal_point
            image_size: (height, width) of image size.
                A tensor of shape (N, 2) or a list/tuple. Required for screen cameras.
            device: torch.device or string
        """
        # The initializer formats all inputs to torch tensors and broadcasts
        # all the inputs to have the same batch dimension where necessary.
        kwargs = {"image_size": image_size} if image_size is not None else {}
        super().__init__(
            device=device,
            focal_length=focal_length,
            principal_point=principal_point,
            R=R,
            T=T,
            K=K,
            _in_ndc=in_ndc,
            **kwargs,  # pyre-ignore
        )
        if image_size is not None:
            if (self.image_size < 1).any():  # pyre-ignore
                raise ValueError("Image_size provided has invalid values")
        else:
            self.image_size = None

        # When focal length is provided as one value, expand to
        # create (N, 2) shape tensor
        if self.focal_length.ndim == 1:  # (N,)
            self.focal_length = self.focal_length[:, None]  # (N, 1)
        self.focal_length = self.focal_length.expand(-1, 2)  # (N, 2)

    def get_projection_transform(self, **kwargs) -> Transform3d:
        """
        Calculate the projection matrix using the
        multi-view geometry convention.

        Args:
            **kwargs: parameters for the projection can be passed in as keyword
                arguments to override the default values set in __init__.

        Returns:
            A `Transform3d` object with a batch of `N` projection transforms.

        .. code-block:: python

            fx = focal_length[:, 0]
            fy = focal_length[:, 1]
            px = principal_point[:, 0]
            py = principal_point[:, 1]

            K = [
                    [fx,   0,   px,   0],
                    [0,   fy,   py,   0],
                    [0,    0,    0,   1],
                    [0,    0,    1,   0],
            ]
        """
        K = kwargs.get("K", self.K)
        if K is not None:
            if K.shape != (self._N, 4, 4):
                msg = "Expected K to have shape of (%r, 4, 4)"
                raise ValueError(msg % (self._N))
        else:
            K = _get_sfm_calibration_matrix(
                self._N,
                self.device,
                kwargs.get("focal_length", self.focal_length),
                kwargs.get("principal_point", self.principal_point),
                orthographic=False,
            )

        transform = Transform3d(
            matrix=K.transpose(1, 2).contiguous(), device=self.device
        )
        return transform

    def unproject_points(
        self,
        xy_depth: torch.Tensor,
        world_coordinates: bool = True,
        from_ndc: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            from_ndc: If `False` (default), assumes xy part of input is in
                NDC space if self.in_ndc(), otherwise in screen space. If
                `True`, assumes xy is in NDC space even if the camera
                is defined in screen space.
        """
        if world_coordinates:
            to_camera_transform = self.get_full_projection_transform(**kwargs)
        else:
            to_camera_transform = self.get_projection_transform(**kwargs)
        if from_ndc:
            to_camera_transform = to_camera_transform.compose(
                self.get_ndc_camera_transform()
            )

        unprojection_transform = to_camera_transform.inverse()
        xy_inv_depth = torch.cat(
            (xy_depth[..., :2], 1.0 / xy_depth[..., 2:3]), dim=-1  # type: ignore
        )
        return unprojection_transform.transform_points(xy_inv_depth)

    def get_principal_point(self, **kwargs) -> torch.Tensor:
        """
        Return the camera's principal point

        Args:
            **kwargs: parameters for the camera extrinsics can be passed in
                as keyword arguments to override the default values
                set in __init__.
        """
        proj_mat = self.get_projection_transform(**kwargs).get_matrix()
        return proj_mat[:, 2, :2]

    def get_ndc_camera_transform(self, **kwargs) -> Transform3d:
        """
        Returns the transform from camera projection space (screen or NDC) to NDC space.
        If the camera is defined already in NDC space, the transform is identity.
        For cameras defined in screen space, we adjust the principal point computation
        which is defined in the image space (commonly) and scale the points to NDC space.

        This transform leaves the depth unchanged.

        Important: This transforms assumes PyTorch3D conventions for the input points,
        i.e. +X left, +Y up.
        """
        if self.in_ndc():
            ndc_transform = Transform3d(device=self.device, dtype=torch.float32)
        else:
            # when cameras are defined in screen/image space, the principal point is
            # provided in the (+X right, +Y down), aka image, coordinate system.
            # Since input points are defined in the PyTorch3D system (+X left, +Y up),
            # we need to adjust for the principal point transform.
            pr_point_fix = torch.zeros(
                (self._N, 4, 4), device=self.device, dtype=torch.float32
            )
            pr_point_fix[:, 0, 0] = 1.0
            pr_point_fix[:, 1, 1] = 1.0
            pr_point_fix[:, 2, 2] = 1.0
            pr_point_fix[:, 3, 3] = 1.0
            pr_point_fix[:, :2, 3] = -2.0 * self.get_principal_point(**kwargs)
            pr_point_fix_transform = Transform3d(
                matrix=pr_point_fix.transpose(1, 2).contiguous(), device=self.device
            )
            image_size = kwargs.get("image_size", self.get_image_size())
            screen_to_ndc_transform = get_screen_to_ndc_transform(
                self, with_xyflip=False, image_size=image_size
            )
            ndc_transform = pr_point_fix_transform.compose(screen_to_ndc_transform)

        return ndc_transform

    def is_perspective(self):
        return True

    def in_ndc(self):
        return self._in_ndc


def SfMOrthographicCameras(
    focal_length: _FocalLengthType = 1.0,
    principal_point=((0.0, 0.0),),
    R: torch.Tensor = _R,
    T: torch.Tensor = _T,
    device: Device = "cpu",
) -> "OrthographicCameras":
    """
    SfMOrthographicCameras has been DEPRECATED. Use OrthographicCameras instead.
    Preserving SfMOrthographicCameras for backward compatibility.
    """

    warnings.warn(
        """SfMOrthographicCameras is deprecated,
        Use OrthographicCameras instead.
        SfMOrthographicCameras will be removed in future releases.""",
        PendingDeprecationWarning,
    )

    return OrthographicCameras(
        focal_length=focal_length,
        principal_point=principal_point,
        R=R,
        T=T,
        device=device,
    )


class OrthographicCameras(CamerasBase):
    """
    A class which stores a batch of parameters to generate a batch of
    transformation matrices using the multi-view geometry convention for
    orthographic camera.

    Parameters for this camera are specified in NDC if `in_ndc` is set to True.
    If parameters are specified in screen space, `in_ndc` must be set to False.
    """

    # For __getitem__
    _FIELDS = (
        "K",
        "R",
        "T",
        "focal_length",
        "principal_point",
        "_in_ndc",
        "image_size",
    )

    _SHARED_FIELDS = ("_in_ndc",)

    def __init__(
        self,
        focal_length: _FocalLengthType = 1.0,
        principal_point=((0.0, 0.0),),
        R: torch.Tensor = _R,
        T: torch.Tensor = _T,
        K: Optional[torch.Tensor] = None,
        device: Device = "cpu",
        in_ndc: bool = True,
        image_size: Optional[Union[List, Tuple, torch.Tensor]] = None,
    ) -> None:
        """

        Args:
            focal_length: Focal length of the camera in world units.
                A tensor of shape (N, 1) or (N, 2) for
                square and non-square pixels respectively.
            principal_point: xy coordinates of the center of
                the principal point of the camera in pixels.
                A tensor of shape (N, 2).
            in_ndc: True if camera parameters are specified in NDC.
                If False, then camera parameters are in screen space.
            R: Rotation matrix of shape (N, 3, 3)
            T: Translation matrix of shape (N, 3)
            K: (optional) A calibration matrix of shape (N, 4, 4)
                If provided, don't need focal_length, principal_point, image_size
            image_size: (height, width) of image size.
                A tensor of shape (N, 2) or list/tuple. Required for screen cameras.
            device: torch.device or string
        """
        # The initializer formats all inputs to torch tensors and broadcasts
        # all the inputs to have the same batch dimension where necessary.
        kwargs = {"image_size": image_size} if image_size is not None else {}
        super().__init__(
            device=device,
            focal_length=focal_length,
            principal_point=principal_point,
            R=R,
            T=T,
            K=K,
            _in_ndc=in_ndc,
            **kwargs,  # pyre-ignore
        )
        if image_size is not None:
            if (self.image_size < 1).any():  # pyre-ignore
                raise ValueError("Image_size provided has invalid values")
        else:
            self.image_size = None

        # When focal length is provided as one value, expand to
        # create (N, 2) shape tensor
        if self.focal_length.ndim == 1:  # (N,)
            self.focal_length = self.focal_length[:, None]  # (N, 1)
        self.focal_length = self.focal_length.expand(-1, 2)  # (N, 2)

    def get_projection_transform(self, **kwargs) -> Transform3d:
        """
        Calculate the projection matrix using
        the multi-view geometry convention.

        Args:
            **kwargs: parameters for the projection can be passed in as keyword
                arguments to override the default values set in __init__.

        Returns:
            A `Transform3d` object with a batch of `N` projection transforms.

        .. code-block:: python

            fx = focal_length[:,0]
            fy = focal_length[:,1]
            px = principal_point[:,0]
            py = principal_point[:,1]

            K = [
                    [fx,   0,    0,  px],
                    [0,   fy,    0,  py],
                    [0,    0,    1,   0],
                    [0,    0,    0,   1],
            ]
        """
        K = kwargs.get("K", self.K)
        if K is not None:
            if K.shape != (self._N, 4, 4):
                msg = "Expected K to have shape of (%r, 4, 4)"
                raise ValueError(msg % (self._N))
        else:
            K = _get_sfm_calibration_matrix(
                self._N,
                self.device,
                kwargs.get("focal_length", self.focal_length),
                kwargs.get("principal_point", self.principal_point),
                orthographic=True,
            )

        transform = Transform3d(
            matrix=K.transpose(1, 2).contiguous(), device=self.device
        )
        return transform

    def unproject_points(
        self,
        xy_depth: torch.Tensor,
        world_coordinates: bool = True,
        from_ndc: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            from_ndc: If `False` (default), assumes xy part of input is in
                NDC space if self.in_ndc(), otherwise in screen space. If
                `True`, assumes xy is in NDC space even if the camera
                is defined in screen space.
        """
        if world_coordinates:
            to_camera_transform = self.get_full_projection_transform(**kwargs)
        else:
            to_camera_transform = self.get_projection_transform(**kwargs)
        if from_ndc:
            to_camera_transform = to_camera_transform.compose(
                self.get_ndc_camera_transform()
            )

        unprojection_transform = to_camera_transform.inverse()
        return unprojection_transform.transform_points(xy_depth)

    def get_principal_point(self, **kwargs) -> torch.Tensor:
        """
        Return the camera's principal point

        Args:
            **kwargs: parameters for the camera extrinsics can be passed in
                as keyword arguments to override the default values
                set in __init__.
        """
        proj_mat = self.get_projection_transform(**kwargs).get_matrix()
        return proj_mat[:, 3, :2]

    def get_ndc_camera_transform(self, **kwargs) -> Transform3d:
        """
        Returns the transform from camera projection space (screen or NDC) to NDC space.
        If the camera is defined already in NDC space, the transform is identity.
        For cameras defined in screen space, we adjust the principal point computation
        which is defined in the image space (commonly) and scale the points to NDC space.

        Important: This transforms assumes PyTorch3D conventions for the input points,
        i.e. +X left, +Y up.
        """
        if self.in_ndc():
            ndc_transform = Transform3d(device=self.device, dtype=torch.float32)
        else:
            # when cameras are defined in screen/image space, the principal point is
            # provided in the (+X right, +Y down), aka image, coordinate system.
            # Since input points are defined in the PyTorch3D system (+X left, +Y up),
            # we need to adjust for the principal point transform.
            pr_point_fix = torch.zeros(
                (self._N, 4, 4), device=self.device, dtype=torch.float32
            )
            pr_point_fix[:, 0, 0] = 1.0
            pr_point_fix[:, 1, 1] = 1.0
            pr_point_fix[:, 2, 2] = 1.0
            pr_point_fix[:, 3, 3] = 1.0
            pr_point_fix[:, :2, 3] = -2.0 * self.get_principal_point(**kwargs)
            pr_point_fix_transform = Transform3d(
                matrix=pr_point_fix.transpose(1, 2).contiguous(), device=self.device
            )
            image_size = kwargs.get("image_size", self.get_image_size())
            screen_to_ndc_transform = get_screen_to_ndc_transform(
                self, with_xyflip=False, image_size=image_size
            )
            ndc_transform = pr_point_fix_transform.compose(screen_to_ndc_transform)

        return ndc_transform

    def is_perspective(self):
        return False

    def in_ndc(self):
        return self._in_ndc


################################################
#       Helper functions for cameras           #
################################################


def _get_sfm_calibration_matrix(
    N: int,
    device: Device,
    focal_length,
    principal_point,
    orthographic: bool = False,
) -> torch.Tensor:
    """
    Returns a calibration matrix of a perspective/orthographic camera.

    Args:
        N: Number of cameras.
        focal_length: Focal length of the camera.
        principal_point: xy coordinates of the center of
            the principal point of the camera in pixels.
        orthographic: Boolean specifying if the camera is orthographic or not

        The calibration matrix `K` is set up as follows:

        .. code-block:: python

            fx = focal_length[:,0]
            fy = focal_length[:,1]
            px = principal_point[:,0]
            py = principal_point[:,1]

            for orthographic==True:
                K = [
                        [fx,   0,    0,  px],
                        [0,   fy,    0,  py],
                        [0,    0,    1,   0],
                        [0,    0,    0,   1],
                ]
            else:
                K = [
                        [fx,   0,   px,   0],
                        [0,   fy,   py,   0],
                        [0,    0,    0,   1],
                        [0,    0,    1,   0],
                ]

    Returns:
        A calibration matrix `K` of the SfM-conventioned camera
        of shape (N, 4, 4).
    """

    if not torch.is_tensor(focal_length):
        focal_length = torch.tensor(focal_length, device=device)

    if focal_length.ndim in (0, 1) or focal_length.shape[1] == 1:
        fx = fy = focal_length
    else:
        fx, fy = focal_length.unbind(1)

    if not torch.is_tensor(principal_point):
        principal_point = torch.tensor(principal_point, device=device)

    px, py = principal_point.unbind(1)

    K = fx.new_zeros(N, 4, 4)
    K[:, 0, 0] = fx
    K[:, 1, 1] = fy
    if orthographic:
        K[:, 0, 3] = px
        K[:, 1, 3] = py
        K[:, 2, 2] = 1.0
        K[:, 3, 3] = 1.0
    else:
        K[:, 0, 2] = px
        K[:, 1, 2] = py
        K[:, 3, 2] = 1.0
        K[:, 2, 3] = 1.0

    return K


################################################
# Helper functions for world to view transforms
################################################


def get_world_to_view_transform(
    R: torch.Tensor = _R, T: torch.Tensor = _T
) -> Transform3d:
    """
    This function returns a Transform3d representing the transformation
    matrix to go from world space to view space by applying a rotation and
    a translation.

    PyTorch3D uses the same convention as Hartley & Zisserman.
    I.e., for camera extrinsic parameters R (rotation) and T (translation),
    we map a 3D point `X_world` in world coordinates to
    a point `X_cam` in camera coordinates with:
    `X_cam = X_world R + T`

    Args:
        R: (N, 3, 3) matrix representing the rotation.
        T: (N, 3) matrix representing the translation.

    Returns:
        a Transform3d object which represents the composed RT transformation.

    """
    # TODO: also support the case where RT is specified as one matrix
    # of shape (N, 4, 4).

    if T.shape[0] != R.shape[0]:
        msg = "Expected R, T to have the same batch dimension; got %r, %r"
        raise ValueError(msg % (R.shape[0], T.shape[0]))
    if T.dim() != 2 or T.shape[1:] != (3,):
        msg = "Expected T to have shape (N, 3); got %r"
        raise ValueError(msg % repr(T.shape))
    if R.dim() != 3 or R.shape[1:] != (3, 3):
        msg = "Expected R to have shape (N, 3, 3); got %r"
        raise ValueError(msg % repr(R.shape))

    # Create a Transform3d object
    T_ = Translate(T, device=T.device)
    R_ = Rotate(R, device=R.device)
    return R_.compose(T_)


def camera_position_from_spherical_angles(
    distance: float,
    elevation: float,
    azimuth: float,
    degrees: bool = True,
    device: Device = "cpu",
) -> torch.Tensor:
    """
    Calculate the location of the camera based on the distance away from
    the target point, the elevation and azimuth angles.

    Args:
        distance: distance of the camera from the object.
        elevation, azimuth: angles.
            The inputs distance, elevation and azimuth can be one of the following
                - Python scalar
                - Torch scalar
                - Torch tensor of shape (N) or (1)
        degrees: bool, whether the angles are specified in degrees or radians.
        device: str or torch.device, device for new tensors to be placed on.

    The vectors are broadcast against each other so they all have shape (N, 1).

    Returns:
        camera_position: (N, 3) xyz location of the camera.
    """
    broadcasted_args = convert_to_tensors_and_broadcast(
        distance, elevation, azimuth, device=device
    )
    dist, elev, azim = broadcasted_args
    if degrees:
        elev = math.pi / 180.0 * elev
        azim = math.pi / 180.0 * azim
    x = dist * torch.cos(elev) * torch.sin(azim)
    y = dist * torch.sin(elev)
    z = dist * torch.cos(elev) * torch.cos(azim)
    camera_position = torch.stack([x, y, z], dim=1)
    if camera_position.dim() == 0:
        camera_position = camera_position.view(1, -1)  # add batch dim.
    return camera_position.view(-1, 3)


def look_at_rotation(
    camera_position, at=((0, 0, 0),), up=((0, 1, 0),), device: Device = "cpu"
) -> torch.Tensor:
    """
    This function takes a vector 'camera_position' which specifies the location
    of the camera in world coordinates and two vectors `at` and `up` which
    indicate the position of the object and the up directions of the world
    coordinate system respectively. The object is assumed to be centered at
    the origin.

    The output is a rotation matrix representing the transformation
    from world coordinates -> view coordinates.

    Args:
        camera_position: position of the camera in world coordinates
        at: position of the object in world coordinates
        up: vector specifying the up direction in the world coordinate frame.

    The inputs camera_position, at and up can each be a
        - 3 element tuple/list
        - torch tensor of shape (1, 3)
        - torch tensor of shape (N, 3)

    The vectors are broadcast against each other so they all have shape (N, 3).

    Returns:
        R: (N, 3, 3) batched rotation matrices
    """
    # Format input and broadcast
    broadcasted_args = convert_to_tensors_and_broadcast(
        camera_position, at, up, device=device
    )
    camera_position, at, up = broadcasted_args
    for t, n in zip([camera_position, at, up], ["camera_position", "at", "up"]):
        if t.shape[-1] != 3:
            msg = "Expected arg %s to have shape (N, 3); got %r"
            raise ValueError(msg % (n, t.shape))
    z_axis = F.normalize(at - camera_position, eps=1e-5)
    x_axis = F.normalize(torch.cross(up, z_axis, dim=1), eps=1e-5)
    y_axis = F.normalize(torch.cross(z_axis, x_axis, dim=1), eps=1e-5)
    is_close = torch.isclose(x_axis, torch.tensor(0.0), atol=5e-3).all(
        dim=1, keepdim=True
    )
    if is_close.any():
        replacement = F.normalize(torch.cross(y_axis, z_axis, dim=1), eps=1e-5)
        x_axis = torch.where(is_close, replacement, x_axis)
    R = torch.cat((x_axis[:, None, :], y_axis[:, None, :], z_axis[:, None, :]), dim=1)
    return R.transpose(1, 2)


def look_at_view_transform(
    dist: _BatchFloatType = 1.0,
    elev: _BatchFloatType = 0.0,
    azim: _BatchFloatType = 0.0,
    degrees: bool = True,
    eye: Optional[Union[Sequence, torch.Tensor]] = None,
    at=((0, 0, 0),),  # (1, 3)
    up=((0, 1, 0),),  # (1, 3)
    device: Device = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    This function returns a rotation and translation matrix
    to apply the 'Look At' transformation from world -> view coordinates [0].

    Args:
        dist: distance of the camera from the object
        elev: angle in degrees or radians. This is the angle between the
            vector from the object to the camera, and the horizontal plane y = 0 (xz-plane).
        azim: angle in degrees or radians. The vector from the object to
            the camera is projected onto a horizontal plane y = 0.
            azim is the angle between the projected vector and a
            reference vector at (0, 0, 1) on the reference plane (the horizontal plane).
        dist, elev and azim can be of shape (1), (N).
        degrees: boolean flag to indicate if the elevation and azimuth
            angles are specified in degrees or radians.
        eye: the position of the camera(s) in world coordinates. If eye is not
            None, it will override the camera position derived from dist, elev, azim.
        up: the direction of the x axis in the world coordinate system.
        at: the position of the object(s) in world coordinates.
        eye, up and at can be of shape (1, 3) or (N, 3).

    Returns:
        2-element tuple containing

        - **R**: the rotation to apply to the points to align with the camera.
        - **T**: the translation to apply to the points to align with the camera.

    References:
    [0] https://www.scratchapixel.com
    """

    if eye is not None:
        broadcasted_args = convert_to_tensors_and_broadcast(eye, at, up, device=device)
        eye, at, up = broadcasted_args
        C = eye
    else:
        broadcasted_args = convert_to_tensors_and_broadcast(
            dist, elev, azim, at, up, device=device
        )
        dist, elev, azim, at, up = broadcasted_args
        C = (
            camera_position_from_spherical_angles(
                dist, elev, azim, degrees=degrees, device=device
            )
            + at
        )

    R = look_at_rotation(C, at, up, device=device)
    T = -torch.bmm(R.transpose(1, 2), C[:, :, None])[:, :, 0]
    return R, T


def get_ndc_to_screen_transform(
    cameras,
    with_xyflip: bool = False,
    image_size: Optional[Union[List, Tuple, torch.Tensor]] = None,
) -> Transform3d:
    """
    PyTorch3D NDC to screen conversion.
    Conversion from PyTorch3D's NDC space (+X left, +Y up) to screen/image space
    (+X right, +Y down, origin top left).

    Args:
        cameras
        with_xyflip: flips x- and y-axis if set to True.
    Optional kwargs:
        image_size: ((height, width),) specifying the height, width
        of the image. If not provided, it reads it from cameras.

    We represent the NDC to screen conversion as a Transform3d
    with projection matrix

    K = [
            [s,   0,    0,  cx],
            [0,   s,    0,  cy],
            [0,   0,    1,   0],
            [0,   0,    0,   1],
    ]

    """
    # We require the image size, which is necessary for the transform
    if image_size is None:
        msg = "For NDC to screen conversion, image_size=(height, width) needs to be specified."
        raise ValueError(msg)

    K = torch.zeros((cameras._N, 4, 4), device=cameras.device, dtype=torch.float32)
    if not torch.is_tensor(image_size):
        image_size = torch.tensor(image_size, device=cameras.device)
    # pyre-fixme[16]: Item `List` of `Union[List[typing.Any], Tensor, Tuple[Any,
    #  ...]]` has no attribute `view`.
    image_size = image_size.view(-1, 2)  # of shape (1 or B)x2
    height, width = image_size.unbind(1)

    # For non square images, we scale the points such that smallest side
    # has range [-1, 1] and the largest side has range [-u, u], with u > 1.
    # This convention is consistent with the PyTorch3D renderer
    scale = (image_size.min(dim=1).values - 0.0) / 2.0

    K[:, 0, 0] = scale
    K[:, 1, 1] = scale
    K[:, 0, 3] = -1.0 * (width - 0.0) / 2.0
    K[:, 1, 3] = -1.0 * (height - 0.0) / 2.0
    K[:, 2, 2] = 1.0
    K[:, 3, 3] = 1.0

    # Transpose the projection matrix as PyTorch3D transforms use row vectors.
    transform = Transform3d(
        matrix=K.transpose(1, 2).contiguous(), device=cameras.device
    )

    if with_xyflip:
        # flip x, y axis
        xyflip = torch.eye(4, device=cameras.device, dtype=torch.float32)
        xyflip[0, 0] = -1.0
        xyflip[1, 1] = -1.0
        xyflip = xyflip.view(1, 4, 4).expand(cameras._N, -1, -1)
        xyflip_transform = Transform3d(
            matrix=xyflip.transpose(1, 2).contiguous(), device=cameras.device
        )
        transform = transform.compose(xyflip_transform)
    return transform


def get_screen_to_ndc_transform(
    cameras,
    with_xyflip: bool = False,
    image_size: Optional[Union[List, Tuple, torch.Tensor]] = None,
) -> Transform3d:
    """
    Screen to PyTorch3D NDC conversion.
    Conversion from screen/image space (+X right, +Y down, origin top left)
    to PyTorch3D's NDC space (+X left, +Y up).

    Args:
        cameras
        with_xyflip: flips x- and y-axis if set to True.
    Optional kwargs:
        image_size: ((height, width),) specifying the height, width
        of the image. If not provided, it reads it from cameras.

    We represent the screen to NDC conversion as a Transform3d
    with projection matrix

    K = [
            [1/s,    0,    0,  cx/s],
            [  0,  1/s,    0,  cy/s],
            [  0,    0,    1,     0],
            [  0,    0,    0,     1],
    ]

    """
    transform = get_ndc_to_screen_transform(
        cameras,
        with_xyflip=with_xyflip,
        image_size=image_size,
    ).inverse()
    return transform


def try_get_projection_transform(
    cameras: CamerasBase, cameras_kwargs: Dict[str, Any]
) -> Optional[Transform3d]:
    """
    Try block to get projection transform from cameras and cameras_kwargs.

    Args:
        cameras: cameras instance, can be linear cameras or nonliear cameras
        cameras_kwargs: camera parameters to be passed to cameras

    Returns:
        If the camera implemented projection_transform, return the
        projection transform; Otherwise, return None
    """

    transform = None
    try:
        transform = cameras.get_projection_transform(**cameras_kwargs)
    except NotImplementedError:
        pass
    return transform
