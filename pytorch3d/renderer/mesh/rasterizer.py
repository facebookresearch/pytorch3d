# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import NamedTuple, Optional, Tuple, Union

import torch
import torch.nn as nn

from .rasterize_meshes import rasterize_meshes


# Class to store the outputs of mesh rasterization
class Fragments(NamedTuple):
    pix_to_face: torch.Tensor
    zbuf: torch.Tensor
    bary_coords: torch.Tensor
    dists: torch.Tensor


# Class to store the mesh rasterization params with defaults
class RasterizationSettings:
    __slots__ = [
        "image_size",
        "blur_radius",
        "faces_per_pixel",
        "bin_size",
        "max_faces_per_bin",
        "perspective_correct",
        "clip_barycentric_coords",
        "cull_backfaces",
        "z_clip_value",
        "cull_to_frustum",
    ]

    def __init__(
        self,
        image_size: Union[int, Tuple[int, int]] = 256,
        blur_radius: float = 0.0,
        faces_per_pixel: int = 1,
        bin_size: Optional[int] = None,
        max_faces_per_bin: Optional[int] = None,
        # set perspective_correct = None so that the
        # value can be inferred correctly from the Camera type
        perspective_correct: Optional[bool] = None,
        clip_barycentric_coords: Optional[bool] = None,
        cull_backfaces: bool = False,
        z_clip_value: Optional[float] = None,
        cull_to_frustum: bool = False,
    ) -> None:
        self.image_size = image_size
        self.blur_radius = blur_radius
        self.faces_per_pixel = faces_per_pixel
        self.bin_size = bin_size
        self.max_faces_per_bin = max_faces_per_bin
        self.perspective_correct = perspective_correct
        self.clip_barycentric_coords = clip_barycentric_coords
        self.cull_backfaces = cull_backfaces
        self.z_clip_value = z_clip_value
        self.cull_to_frustum = cull_to_frustum


class MeshRasterizer(nn.Module):
    """
    This class implements methods for rasterizing a batch of heterogeneous
    Meshes.
    """

    def __init__(self, cameras=None, raster_settings=None) -> None:
        """
        Args:
            cameras: A cameras object which has a  `transform_points` method
                which returns the transformed points after applying the
                world-to-view and view-to-ndc transformations.
            raster_settings: the parameters for rasterization. This should be a
                named tuple.

        All these initial settings can be overridden by passing keyword
        arguments to the forward function.
        """
        super().__init__()
        if raster_settings is None:
            raster_settings = RasterizationSettings()

        self.cameras = cameras
        self.raster_settings = raster_settings

    def to(self, device):
        # Manually move to device cameras as it is not a subclass of nn.Module
        self.cameras = self.cameras.to(device)
        return self

    def transform(self, meshes_world, **kwargs) -> torch.Tensor:
        """
        Args:
            meshes_world: a Meshes object representing a batch of meshes with
                vertex coordinates in world space.

        Returns:
            meshes_proj: a Meshes object with the vertex positions projected
            in NDC space

        NOTE: keeping this as a separate function for readability but it could
        be moved into forward.
        """
        cameras = kwargs.get("cameras", self.cameras)
        if cameras is None:
            msg = "Cameras must be specified either at initialization \
                or in the forward pass of MeshRasterizer"
            raise ValueError(msg)

        n_cameras = len(cameras)
        if n_cameras != 1 and n_cameras != len(meshes_world):
            msg = "Wrong number (%r) of cameras for %r meshes"
            raise ValueError(msg % (n_cameras, len(meshes_world)))

        verts_world = meshes_world.verts_padded()

        # NOTE: Retaining view space z coordinate for now.
        # TODO: Revisit whether or not to transform z coordinate to [-1, 1] or
        # [0, 1] range.
        eps = kwargs.get("eps", None)
        verts_view = cameras.get_world_to_view_transform(**kwargs).transform_points(
            verts_world, eps=eps
        )
        # view to NDC transform
        to_ndc_transform = cameras.get_ndc_camera_transform(**kwargs)
        projection_transform = cameras.get_projection_transform(**kwargs).compose(
            to_ndc_transform
        )
        verts_ndc = projection_transform.transform_points(verts_view, eps=eps)

        verts_ndc[..., 2] = verts_view[..., 2]
        meshes_ndc = meshes_world.update_padded(new_verts_padded=verts_ndc)
        return meshes_ndc

    def forward(self, meshes_world, **kwargs) -> Fragments:
        """
        Args:
            meshes_world: a Meshes object representing a batch of meshes with
                          coordinates in world space.
        Returns:
            Fragments: Rasterization outputs as a named tuple.
        """
        meshes_proj = self.transform(meshes_world, **kwargs)
        raster_settings = kwargs.get("raster_settings", self.raster_settings)

        # By default, turn on clip_barycentric_coords if blur_radius > 0.
        # When blur_radius > 0, a face can be matched to a pixel that is outside the
        # face, resulting in negative barycentric coordinates.
        clip_barycentric_coords = raster_settings.clip_barycentric_coords
        if clip_barycentric_coords is None:
            clip_barycentric_coords = raster_settings.blur_radius > 0.0

        # If not specified, infer perspective_correct and z_clip_value from the camera
        cameras = kwargs.get("cameras", self.cameras)
        if raster_settings.perspective_correct is not None:
            perspective_correct = raster_settings.perspective_correct
        else:
            perspective_correct = cameras.is_perspective()
        if raster_settings.z_clip_value is not None:
            z_clip = raster_settings.z_clip_value
        else:
            znear = cameras.get_znear()
            if isinstance(znear, torch.Tensor):
                znear = znear.min().item()
            z_clip = None if not perspective_correct or znear is None else znear / 2

        pix_to_face, zbuf, bary_coords, dists = rasterize_meshes(
            meshes_proj,
            image_size=raster_settings.image_size,
            blur_radius=raster_settings.blur_radius,
            faces_per_pixel=raster_settings.faces_per_pixel,
            bin_size=raster_settings.bin_size,
            max_faces_per_bin=raster_settings.max_faces_per_bin,
            clip_barycentric_coords=clip_barycentric_coords,
            perspective_correct=perspective_correct,
            cull_backfaces=raster_settings.cull_backfaces,
            z_clip_value=z_clip,
            cull_to_frustum=raster_settings.cull_to_frustum,
        )
        return Fragments(
            pix_to_face=pix_to_face, zbuf=zbuf, bary_coords=bary_coords, dists=dists
        )
