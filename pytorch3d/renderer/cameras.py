# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import math
import warnings
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from pytorch3d.transforms import Rotate, Transform3d, Translate

from .utils import TensorProperties, convert_to_tensors_and_broadcast


# Default values for rotation and translation matrices.
_R = torch.eye(3)[None]  # (1, 3, 3)
_T = torch.zeros(1, 3)  # (1, 3)


class CamerasBase(TensorProperties):
    """
    `CamerasBase` implements a base class for all cameras.

    For cameras, there are four different coordinate systems (or spaces)
    - World coordinate system: This is the system the object lives - the world.
    - Camera view coordinate system: This is the system that has its origin on the image plane
        and the and the Z-axis perpendicular to the image plane.
        In PyTorch3D, we assume that +X points left, and +Y points up and
        +Z points out from the image plane.
        The transformation from world -> view happens after applying a rotation (R)
        and translation (T)
    - NDC coordinate system: This is the normalized coordinate system that confines
        in a volume the rendered part of the object or scene. Also known as view volume.
        Given the PyTorch3D convention, (+1, +1, znear) is the top left near corner,
        and (-1, -1, zfar) is the bottom right far corner of the volume.
        The transformation from view -> NDC happens after applying the camera
        projection matrix (P).
    - Screen coordinate system: This is another representation of the view volume with
        the XY coordinates defined in pixel space instead of a normalized space.

    A better illustration of the coordinate systems can be found in pytorch3d/docs/notes/cameras.md.

    It defines methods that are common to all camera models:
        - `get_camera_center` that returns the optical center of the camera in
            world coordinates
        - `get_world_to_view_transform` which returns a 3D transform from
            world coordinates to the camera view coordinates (R, T)
        - `get_full_projection_transform` which composes the projection
            transform (P) with the world-to-view transform (R, T)
        - `transform_points` which takes a set of input points in world coordinates and
            projects to NDC coordinates ranging from [-1, -1, znear] to [+1, +1, zfar].
        - `transform_points_screen` which takes a set of input points in world coordinates and
            projects them to the screen coordinates ranging from
            [0, 0, znear] to [W-1, H-1, zfar]

    For each new camera, one should implement the `get_projection_transform`
    routine that returns the mapping from camera view coordinates to NDC coordinates.

    Another useful function that is specific to each camera model is
    `unproject_points` which sends points from NDC coordinates back to
    camera view or world coordinates depending on the `world_coordinates`
    boolean argument of the function.
    """

    def get_projection_transform(self):
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

    def unproject_points(self):
        """
        Transform input points from NDC coordinates
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

        Setting T here will update the values set in init as this
        value may be needed later on in the rendering pipeline e.g. for
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
        self.R = kwargs.get("R", self.R)  # pyre-ignore[16]
        self.T = kwargs.get("T", self.T)  # pyre-ignore[16]
        world_to_view_transform = get_world_to_view_transform(R=self.R, T=self.T)
        return world_to_view_transform

    def get_full_projection_transform(self, **kwargs) -> Transform3d:
        """
        Return the full world-to-NDC transform composing the
        world-to-view and view-to-NDC transforms.

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
        self.R = kwargs.get("R", self.R)  # pyre-ignore[16]
        self.T = kwargs.get("T", self.T)  # pyre-ignore[16]
        world_to_view_transform = self.get_world_to_view_transform(R=self.R, T=self.T)
        view_to_ndc_transform = self.get_projection_transform(**kwargs)
        return world_to_view_transform.compose(view_to_ndc_transform)

    def transform_points(
        self, points, eps: Optional[float] = None, **kwargs
    ) -> torch.Tensor:
        """
        Transform input points from world to NDC space.

        Args:
            points: torch tensor of shape (..., 3).
            eps: If eps!=None, the argument is used to clamp the
                divisor in the homogeneous normalization of the points
                transformed to the ndc space. Please see
                `transforms.Transform3D.transform_points` for details.

                For `CamerasBase.transform_points`, setting `eps > 0`
                stabilizes gradients since it leads to avoiding division
                by excessively low numbers for points close to the
                camera plane.

        Returns
            new_points: transformed points with the same shape as the input.
        """
        world_to_ndc_transform = self.get_full_projection_transform(**kwargs)
        return world_to_ndc_transform.transform_points(points, eps=eps)

    def transform_points_screen(
        self, points, image_size, eps: Optional[float] = None, **kwargs
    ) -> torch.Tensor:
        """
        Transform input points from world to screen space.

        Args:
            points: torch tensor of shape (N, V, 3).
            image_size: torch tensor of shape (N, 2)
            eps: If eps!=None, the argument is used to clamp the
                divisor in the homogeneous normalization of the points
                transformed to the ndc space. Please see
                `transforms.Transform3D.transform_points` for details.

                For `CamerasBase.transform_points`, setting `eps > 0`
                stabilizes gradients since it leads to avoiding division
                by excessively low numbers for points close to the
                camera plane.

        Returns
            new_points: transformed points with the same shape as the input.
        """

        ndc_points = self.transform_points(points, eps=eps, **kwargs)

        if not torch.is_tensor(image_size):
            image_size = torch.tensor(
                image_size, dtype=torch.int64, device=points.device
            )
        if (image_size < 1).any():
            raise ValueError("Provided image size is invalid.")

        image_width, image_height = image_size.unbind(1)
        image_width = image_width.view(-1, 1)  # (N, 1)
        image_height = image_height.view(-1, 1)  # (N, 1)

        ndc_z = ndc_points[..., 2]
        screen_x = (image_width - 1.0) / 2.0 * (1.0 - ndc_points[..., 0])
        screen_y = (image_height - 1.0) / 2.0 * (1.0 - ndc_points[..., 1])

        return torch.stack((screen_x, screen_y, ndc_z), dim=2)

    def clone(self):
        """
        Returns a copy of `self`.
        """
        cam_type = type(self)
        other = cam_type(device=self.device)
        return super().clone(other)

    def is_perspective(self):
        raise NotImplementedError()

    def get_znear(self):
        return self.znear if hasattr(self, "znear") else None


############################################################
#             Field of View Camera Classes                 #
############################################################


def OpenGLPerspectiveCameras(
    znear=1.0,
    zfar=100.0,
    aspect_ratio=1.0,
    fov=60.0,
    degrees: bool = True,
    R=_R,
    T=_T,
    device="cpu",
):
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
    The definition of the parameters follow the OpenGL perspective camera.

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

    def __init__(
        self,
        znear=1.0,
        zfar=100.0,
        aspect_ratio=1.0,
        fov=60.0,
        degrees: bool = True,
        R=_R,
        T=_T,
        K=None,
        device="cpu",
    ):
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
            device: torch.device or string
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
        self, znear, zfar, fov, aspect_ratio, degrees
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

        K[:, 0, 0] = 2.0 * znear / (max_x - min_x)
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
        transform = Transform3d(device=self.device)
        transform._matrix = K.transpose(1, 2).contiguous()
        return transform

    def unproject_points(
        self,
        xy_depth: torch.Tensor,
        world_coordinates: bool = True,
        scaled_depth_input: bool = False,
        **kwargs
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


def OpenGLOrthographicCameras(
    znear=1.0,
    zfar=100.0,
    top=1.0,
    bottom=-1.0,
    left=-1.0,
    right=1.0,
    scale_xyz=((1.0, 1.0, 1.0),),  # (1, 3)
    R=_R,
    T=_T,
    device="cpu",
):
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
    The definition of the parameters follow the OpenGL orthographic camera.
    """

    def __init__(
        self,
        znear=1.0,
        zfar=100.0,
        max_y=1.0,
        min_y=-1.0,
        max_x=1.0,
        min_x=-1.0,
        scale_xyz=((1.0, 1.0, 1.0),),  # (1, 3)
        R=_R,
        T=_T,
        K=None,
        device="cpu",
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

        transform = Transform3d(device=self.device)
        transform._matrix = K.transpose(1, 2).contiguous()
        return transform

    def unproject_points(
        self,
        xy_depth: torch.Tensor,
        world_coordinates: bool = True,
        scaled_depth_input: bool = False,
        **kwargs
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


############################################################
#             MultiView Camera Classes                     #
############################################################
"""
Note that the MultiView Cameras accept  parameters in both
screen and NDC space.
If the user specifies `image_size` at construction time then
we assume the parameters are in screen space.
"""


def SfMPerspectiveCameras(
    focal_length=1.0, principal_point=((0.0, 0.0),), R=_R, T=_T, device="cpu"
):
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

    Parameters for this camera can be specified in NDC or in screen space.
    If you wish to provide parameters in screen space, you NEED to provide
    the image_size = (imwidth, imheight).
    If you wish to provide parameters in NDC space, you should NOT provide
    image_size. Providing valid image_size will trigger a screen space to
    NDC space transformation in the camera.

    For example, here is how to define cameras on the two spaces.

    .. code-block:: python
        # camera defined in screen space
        cameras = PerspectiveCameras(
            focal_length=((22.0, 15.0),),  # (fx_screen, fy_screen)
            principal_point=((192.0, 128.0),),  # (px_screen, py_screen)
            image_size=((256, 256),),  # (imwidth, imheight)
        )

        # the equivalent camera defined in NDC space
        cameras = PerspectiveCameras(
            focal_length=((0.17875, 0.11718),),  # fx = fx_screen / half_imwidth,
                                                # fy = fy_screen / half_imheight
            principal_point=((-0.5, 0),),  # px = - (px_screen - half_imwidth) / half_imwidth,
                                           # py = - (py_screen - half_imheight) / half_imheight
        )
    """

    def __init__(
        self,
        focal_length=1.0,
        principal_point=((0.0, 0.0),),
        R=_R,
        T=_T,
        K=None,
        device="cpu",
        image_size=((-1, -1),),
    ):
        """

        Args:
            focal_length: Focal length of the camera in world units.
                A tensor of shape (N, 1) or (N, 2) for
                square and non-square pixels respectively.
            principal_point: xy coordinates of the center of
                the principal point of the camera in pixels.
                A tensor of shape (N, 2).
            R: Rotation matrix of shape (N, 3, 3)
            T: Translation matrix of shape (N, 3)
            K: (optional) A calibration matrix of shape (N, 4, 4)
                If provided, don't need focal_length, principal_point, image_size

            device: torch.device or string
            image_size: If image_size = (imwidth, imheight) with imwidth, imheight > 0
                is provided, the camera parameters are assumed to be in screen
                space. They will be converted to NDC space.
                If image_size is not provided, the parameters are assumed to
                be in NDC space.
        """
        # The initializer formats all inputs to torch tensors and broadcasts
        # all the inputs to have the same batch dimension where necessary.
        super().__init__(
            device=device,
            focal_length=focal_length,
            principal_point=principal_point,
            R=R,
            T=T,
            K=K,
            image_size=image_size,
        )

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
            image_size = kwargs.get("image_size", self.image_size)
            # if imwidth > 0, parameters are in screen space
            image_size = image_size if image_size[0][0] > 0 else None

            K = _get_sfm_calibration_matrix(
                self._N,
                self.device,
                kwargs.get("focal_length", self.focal_length),
                kwargs.get("principal_point", self.principal_point),
                orthographic=False,
                image_size=image_size,
            )

        transform = Transform3d(device=self.device)
        transform._matrix = K.transpose(1, 2).contiguous()
        return transform

    def unproject_points(
        self, xy_depth: torch.Tensor, world_coordinates: bool = True, **kwargs
    ) -> torch.Tensor:
        if world_coordinates:
            to_ndc_transform = self.get_full_projection_transform(**kwargs)
        else:
            to_ndc_transform = self.get_projection_transform(**kwargs)

        unprojection_transform = to_ndc_transform.inverse()
        xy_inv_depth = torch.cat(
            (xy_depth[..., :2], 1.0 / xy_depth[..., 2:3]), dim=-1  # type: ignore
        )
        return unprojection_transform.transform_points(xy_inv_depth)

    def is_perspective(self):
        return True


def SfMOrthographicCameras(
    focal_length=1.0, principal_point=((0.0, 0.0),), R=_R, T=_T, device="cpu"
):
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

    Parameters for this camera can be specified in NDC or in screen space.
    If you wish to provide parameters in screen space, you NEED to provide
    the image_size = (imwidth, imheight).
    If you wish to provide parameters in NDC space, you should NOT provide
    image_size. Providing valid image_size will trigger a screen space to
    NDC space transformation in the camera.

    For example, here is how to define cameras on the two spaces.

    .. code-block:: python
        # camera defined in screen space
        cameras = OrthographicCameras(
            focal_length=((22.0, 15.0),),  # (fx, fy)
            principal_point=((192.0, 128.0),),  # (px, py)
            image_size=((256, 256),),  # (imwidth, imheight)
        )

        # the equivalent camera defined in NDC space
        cameras = OrthographicCameras(
            focal_length=((0.17875, 0.11718),),  # := (fx / half_imwidth, fy / half_imheight)
            principal_point=((-0.5, 0),),  # := (- (px - half_imwidth) / half_imwidth,
                                                 - (py - half_imheight) / half_imheight)
        )
    """

    def __init__(
        self,
        focal_length=1.0,
        principal_point=((0.0, 0.0),),
        R=_R,
        T=_T,
        K=None,
        device="cpu",
        image_size=((-1, -1),),
    ):
        """

        Args:
            focal_length: Focal length of the camera in world units.
                A tensor of shape (N, 1) or (N, 2) for
                square and non-square pixels respectively.
            principal_point: xy coordinates of the center of
                the principal point of the camera in pixels.
                A tensor of shape (N, 2).
            R: Rotation matrix of shape (N, 3, 3)
            T: Translation matrix of shape (N, 3)
            K: (optional) A calibration matrix of shape (N, 4, 4)
                If provided, don't need focal_length, principal_point, image_size
            device: torch.device or string
            image_size: If image_size = (imwidth, imheight) with imwidth, imheight > 0
                is provided, the camera parameters are assumed to be in screen
                space. They will be converted to NDC space.
                If image_size is not provided, the parameters are assumed to
                be in NDC space.
        """
        # The initializer formats all inputs to torch tensors and broadcasts
        # all the inputs to have the same batch dimension where necessary.
        super().__init__(
            device=device,
            focal_length=focal_length,
            principal_point=principal_point,
            R=R,
            T=T,
            K=K,
            image_size=image_size,
        )

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
            image_size = kwargs.get("image_size", self.image_size)
            # if imwidth > 0, parameters are in screen space
            image_size = image_size if image_size[0][0] > 0 else None

            K = _get_sfm_calibration_matrix(
                self._N,
                self.device,
                kwargs.get("focal_length", self.focal_length),
                kwargs.get("principal_point", self.principal_point),
                orthographic=True,
                image_size=image_size,
            )

        transform = Transform3d(device=self.device)
        transform._matrix = K.transpose(1, 2).contiguous()
        return transform

    def unproject_points(
        self, xy_depth: torch.Tensor, world_coordinates: bool = True, **kwargs
    ) -> torch.Tensor:
        if world_coordinates:
            to_ndc_transform = self.get_full_projection_transform(**kwargs)
        else:
            to_ndc_transform = self.get_projection_transform(**kwargs)

        unprojection_transform = to_ndc_transform.inverse()
        return unprojection_transform.transform_points(xy_depth)

    def is_perspective(self):
        return False


################################################
#       Helper functions for cameras           #
################################################


def _get_sfm_calibration_matrix(
    N,
    device,
    focal_length,
    principal_point,
    orthographic: bool = False,
    image_size=None,
) -> torch.Tensor:
    """
    Returns a calibration matrix of a perspective/orthographic camera.

    Args:
        N: Number of cameras.
        focal_length: Focal length of the camera in world units.
        principal_point: xy coordinates of the center of
            the principal point of the camera in pixels.
        orthographic: Boolean specifying if the camera is orthographic or not
        image_size: (Optional) Specifying the image_size = (imwidth, imheight).
            If not None, the camera parameters are assumed to be in screen space
            and are transformed to NDC space.

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

    if image_size is not None:
        if not torch.is_tensor(image_size):
            image_size = torch.tensor(image_size, device=device)
        imwidth, imheight = image_size.unbind(1)
        # make sure imwidth, imheight are valid (>0)
        if (imwidth < 1).any() or (imheight < 1).any():
            raise ValueError(
                "Camera parameters provided in screen space. Image width or height invalid."
            )
        half_imwidth = imwidth / 2.0
        half_imheight = imheight / 2.0
        fx = fx / half_imwidth
        fy = fy / half_imheight
        px = -(px - half_imwidth) / half_imwidth
        py = -(py - half_imheight) / half_imheight

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


def get_world_to_view_transform(R=_R, T=_T) -> Transform3d:
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
    T = Translate(T, device=T.device)
    R = Rotate(R, device=R.device)
    return R.compose(T)


def camera_position_from_spherical_angles(
    distance, elevation, azimuth, degrees: bool = True, device: str = "cpu"
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
    camera_position, at=((0, 0, 0),), up=((0, 1, 0),), device: str = "cpu"
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
    dist=1.0,
    elev=0.0,
    azim=0.0,
    degrees: bool = True,
    eye: Optional[Sequence] = None,
    at=((0, 0, 0),),  # (1, 3)
    up=((0, 1, 0),),  # (1, 3)
    device="cpu",
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
        dist, elem and azim can be of shape (1), (N).
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
