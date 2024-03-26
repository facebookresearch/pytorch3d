# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from pytorch3d.renderer.cameras import try_get_projection_transform

from .rasterize_meshes import rasterize_meshes


@dataclass(frozen=True)
class Fragments:
    """
    A dataclass representing the outputs of a rasterizer. Can be detached from the
    computational graph in order to stop the gradients from flowing through the
    rasterizer.

    Members:
        pix_to_face:
            LongTensor of shape (N, image_size, image_size, faces_per_pixel) giving
            the indices of the nearest faces at each pixel, sorted in ascending
            z-order. Concretely ``pix_to_face[n, y, x, k] = f`` means that
            ``faces_verts[f]`` is the kth closest face (in the z-direction) to pixel
            (y, x). Pixels that are hit by fewer than faces_per_pixel are padded with
            -1.

        zbuf:
            FloatTensor of shape (N, image_size, image_size, faces_per_pixel) giving
            the NDC z-coordinates of the nearest faces at each pixel, sorted in
            ascending z-order. Concretely, if ``pix_to_face[n, y, x, k] = f`` then
            ``zbuf[n, y, x, k] = face_verts[f, 2]``. Pixels hit by fewer than
            faces_per_pixel are padded with -1.

        bary_coords:
            FloatTensor of shape (N, image_size, image_size, faces_per_pixel, 3)
            giving the barycentric coordinates in NDC units of the nearest faces at
            each pixel, sorted in ascending z-order. Concretely, if ``pix_to_face[n,
            y, x, k] = f`` then ``[w0, w1, w2] = barycentric[n, y, x, k]`` gives the
            barycentric coords for pixel (y, x) relative to the face defined by
            ``face_verts[f]``. Pixels hit by fewer than faces_per_pixel are padded
            with -1.

        dists:
            FloatTensor of shape (N, image_size, image_size, faces_per_pixel) giving
            the signed Euclidean distance (in NDC units) in the x/y plane of each
            point closest to the pixel. Concretely if ``pix_to_face[n, y, x, k] = f``
            then ``pix_dists[n, y, x, k]`` is the squared distance between the pixel
            (y, x) and the face given by vertices ``face_verts[f]``. Pixels hit with
            fewer than ``faces_per_pixel`` are padded with -1.
    """

    pix_to_face: torch.Tensor
    zbuf: torch.Tensor
    bary_coords: torch.Tensor
    dists: Optional[torch.Tensor]

    def detach(self) -> "Fragments":
        return Fragments(
            pix_to_face=self.pix_to_face,
            zbuf=self.zbuf.detach(),
            bary_coords=self.bary_coords.detach(),
            dists=self.dists.detach() if self.dists is not None else self.dists,
        )


@dataclass
class RasterizationSettings:
    """
    Class to store the mesh rasterization params with defaults

    Members:
        image_size: Either common height and width or (height, width), in pixels.
        blur_radius: Float distance in the range [0, 2] used to expand the face
            bounding boxes for rasterization. Setting blur radius
            results in blurred edges around the shape instead of a
            hard boundary. Set to 0 for no blur.
        faces_per_pixel: (int) Number of faces to keep track of per pixel.
            We return the nearest faces_per_pixel faces along the z-axis.
        bin_size: Size of bins to use for coarse-to-fine rasterization. Setting
            bin_size=0 uses naive rasterization; setting bin_size=None attempts
            to set it heuristically based on the shape of the input. This should
            not affect the output, but can affect the speed of the forward pass.
        max_faces_opengl: Max number of faces in any mesh we will rasterize. Used only by
            MeshRasterizerOpenGL to pre-allocate OpenGL memory.
        max_faces_per_bin: Only applicable when using coarse-to-fine
            rasterization (bin_size != 0); this is the maximum number of faces
            allowed within each bin. This should not affect the output values,
            but can affect the memory usage in the forward pass.
            Setting max_faces_per_bin=None attempts to set with a heuristic.
        perspective_correct: Whether to apply perspective correction when
            computing barycentric coordinates for pixels.
            None (default) means make correction if the camera uses perspective.
        clip_barycentric_coords: Whether, after any perspective correction
            is applied but before the depth is calculated (e.g. for
            z clipping), to "correct" a location outside the face (i.e. with
            a negative barycentric coordinate) to a position on the edge of the
            face. None (default) means clip if blur_radius > 0, which is a condition
            under which such outside-face-points are likely.
        cull_backfaces: Whether to only rasterize mesh faces which are
            visible to the camera.  This assumes that vertices of
            front-facing triangles are ordered in an anti-clockwise
            fashion, and triangles that face away from the camera are
            in a clockwise order relative to the current view
            direction. NOTE: This will only work if the mesh faces are
            consistently defined with counter-clockwise ordering when
            viewed from the outside.
        z_clip_value: if not None, then triangles will be clipped (and possibly
            subdivided into smaller triangles) such that z >= z_clip_value.
            This avoids camera projections that go to infinity as z->0.
            Default is None as clipping affects rasterization speed and
            should only be turned on if explicitly needed.
            See clip.py for all the extra computation that is required.
        cull_to_frustum: Whether to cull triangles outside the view frustum.
            Culling involves removing all faces which fall outside view frustum.
            Default is False for performance as often not needed.
    """

    image_size: Union[int, Tuple[int, int]] = 256
    blur_radius: float = 0.0
    faces_per_pixel: int = 1
    bin_size: Optional[int] = None
    max_faces_opengl: int = 10_000_000
    max_faces_per_bin: Optional[int] = None
    perspective_correct: Optional[bool] = None
    clip_barycentric_coords: Optional[bool] = None
    cull_backfaces: bool = False
    z_clip_value: Optional[float] = None
    cull_to_frustum: bool = False


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
        if self.cameras is not None:
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
        to_ndc_transform = cameras.get_ndc_camera_transform(**kwargs)
        projection_transform = try_get_projection_transform(cameras, kwargs)
        if projection_transform is not None:
            projection_transform = projection_transform.compose(to_ndc_transform)
            verts_ndc = projection_transform.transform_points(verts_view, eps=eps)
        else:
            # Call transform_points instead of explicitly composing transforms to handle
            # the case, where camera class does not have a projection matrix form.
            verts_proj = cameras.transform_points(verts_world, eps=eps)
            verts_ndc = to_ndc_transform.transform_points(verts_proj, eps=eps)

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

        # By default, turn on clip_barycentric_coords if blur_radius > 0.
        # When blur_radius > 0, a face can be matched to a pixel that is outside the
        # face, resulting in negative barycentric coordinates.

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
            pix_to_face=pix_to_face,
            zbuf=zbuf,
            bary_coords=bary_coords,
            dists=dists,
        )
