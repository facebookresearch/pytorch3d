# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe


import torch
import torch.nn.functional as F

from ..common.datatypes import Device
from .utils import convert_to_tensors_and_broadcast, TensorProperties


def diffuse(normals, color, direction) -> torch.Tensor:
    """
    Calculate the diffuse component of light reflection using Lambert's
    cosine law.

    Args:
        normals: (N, ..., 3) xyz normal vectors. Normals and points are
            expected to have the same shape.
        color: (1, 3) or (N, 3) RGB color of the diffuse component of the light.
        direction: (x,y,z) direction of the light

    Returns:
        colors: (N, ..., 3), same shape as the input points.

    The normals and light direction should be in the same coordinate frame
    i.e. if the points have been transformed from world -> view space then
    the normals and direction should also be in view space.

    NOTE: to use with the packed vertices (i.e. no batch dimension) reformat the
    inputs in the following way.

    .. code-block:: python

        Args:
            normals: (P, 3)
            color: (N, 3)[batch_idx, :] -> (P, 3)
            direction: (N, 3)[batch_idx, :] -> (P, 3)

        Returns:
            colors: (P, 3)

        where batch_idx is of shape (P). For meshes, batch_idx can be:
        meshes.verts_packed_to_mesh_idx() or meshes.faces_packed_to_mesh_idx()
        depending on whether points refers to the vertex coordinates or
        average/interpolated face coordinates.
    """
    # TODO: handle multiple directional lights per batch element.
    # TODO: handle attenuation.

    # Ensure color and location have same batch dimension as normals
    normals, color, direction = convert_to_tensors_and_broadcast(
        normals, color, direction, device=normals.device
    )

    # Reshape direction and color so they have all the arbitrary intermediate
    # dimensions as normals. Assume first dim = batch dim and last dim = 3.
    points_dims = normals.shape[1:-1]
    expand_dims = (-1,) + (1,) * len(points_dims) + (3,)
    if direction.shape != normals.shape:
        direction = direction.view(expand_dims)
    if color.shape != normals.shape:
        color = color.view(expand_dims)

    # Renormalize the normals in case they have been interpolated.
    # We tried to replace the following with F.cosine_similarity, but it wasn't faster.
    normals = F.normalize(normals, p=2, dim=-1, eps=1e-6)
    direction = F.normalize(direction, p=2, dim=-1, eps=1e-6)
    angle = F.relu(torch.sum(normals * direction, dim=-1))
    return color * angle[..., None]


def specular(
    points, normals, direction, color, camera_position, shininess
) -> torch.Tensor:
    """
    Calculate the specular component of light reflection.

    Args:
        points: (N, ..., 3) xyz coordinates of the points.
        normals: (N, ..., 3) xyz normal vectors for each point.
        color: (N, 3) RGB color of the specular component of the light.
        direction: (N, 3) vector direction of the light.
        camera_position: (N, 3) The xyz position of the camera.
        shininess: (N)  The specular exponent of the material.

    Returns:
        colors: (N, ..., 3), same shape as the input points.

    The points, normals, camera_position, and direction should be in the same
    coordinate frame i.e. if the points have been transformed from
    world -> view space then the normals, camera_position, and light direction
    should also be in view space.

    To use with a batch of packed points reindex in the following way.
    .. code-block:: python::

        Args:
            points: (P, 3)
            normals: (P, 3)
            color: (N, 3)[batch_idx] -> (P, 3)
            direction: (N, 3)[batch_idx] -> (P, 3)
            camera_position: (N, 3)[batch_idx] -> (P, 3)
            shininess: (N)[batch_idx] -> (P)
        Returns:
            colors: (P, 3)

        where batch_idx is of shape (P). For meshes batch_idx can be:
        meshes.verts_packed_to_mesh_idx() or meshes.faces_packed_to_mesh_idx().
    """
    # TODO: handle multiple directional lights
    # TODO: attenuate based on inverse squared distance to the light source

    if points.shape != normals.shape:
        msg = "Expected points and normals to have the same shape: got %r, %r"
        raise ValueError(msg % (points.shape, normals.shape))

    # Ensure all inputs have same batch dimension as points
    matched_tensors = convert_to_tensors_and_broadcast(
        points, color, direction, camera_position, shininess, device=points.device
    )
    _, color, direction, camera_position, shininess = matched_tensors

    # Reshape direction and color so they have all the arbitrary intermediate
    # dimensions as points. Assume first dim = batch dim and last dim = 3.
    points_dims = points.shape[1:-1]
    expand_dims = (-1,) + (1,) * len(points_dims)
    if direction.shape != normals.shape:
        direction = direction.view(expand_dims + (3,))
    if color.shape != normals.shape:
        color = color.view(expand_dims + (3,))
    if camera_position.shape != normals.shape:
        camera_position = camera_position.view(expand_dims + (3,))
    if shininess.shape != normals.shape:
        shininess = shininess.view(expand_dims)

    # Renormalize the normals in case they have been interpolated.
    # We tried a version that uses F.cosine_similarity instead of renormalizing,
    # but it was slower.
    normals = F.normalize(normals, p=2, dim=-1, eps=1e-6)
    direction = F.normalize(direction, p=2, dim=-1, eps=1e-6)
    cos_angle = torch.sum(normals * direction, dim=-1)
    # No specular highlights if angle is less than 0.
    mask = (cos_angle > 0).to(torch.float32)

    # Calculate the specular reflection.
    view_direction = camera_position - points
    view_direction = F.normalize(view_direction, p=2, dim=-1, eps=1e-6)
    reflect_direction = -direction + 2 * (cos_angle[..., None] * normals)

    # Cosine of the angle between the reflected light ray and the viewer
    alpha = F.relu(torch.sum(view_direction * reflect_direction, dim=-1)) * mask
    return color * torch.pow(alpha, shininess)[..., None]


class DirectionalLights(TensorProperties):
    def __init__(
        self,
        ambient_color=((0.5, 0.5, 0.5),),
        diffuse_color=((0.3, 0.3, 0.3),),
        specular_color=((0.2, 0.2, 0.2),),
        direction=((0, 1, 0),),
        device: Device = "cpu",
    ) -> None:
        """
        Args:
            ambient_color: RGB color of the ambient component.
            diffuse_color: RGB color of the diffuse component.
            specular_color: RGB color of the specular component.
            direction: (x, y, z) direction vector of the light.
            device: Device (as str or torch.device) on which the tensors should be located

        The inputs can each be
            - 3 element tuple/list or list of lists
            - torch tensor of shape (1, 3)
            - torch tensor of shape (N, 3)
        The inputs are broadcast against each other so they all have batch
        dimension N.
        """
        super().__init__(
            device=device,
            ambient_color=ambient_color,
            diffuse_color=diffuse_color,
            specular_color=specular_color,
            direction=direction,
        )
        _validate_light_properties(self)
        if self.direction.shape[-1] != 3:
            msg = "Expected direction to have shape (N, 3); got %r"
            raise ValueError(msg % repr(self.direction.shape))

    def clone(self):
        other = self.__class__(device=self.device)
        return super().clone(other)

    def diffuse(self, normals, points=None) -> torch.Tensor:
        # NOTE: Points is not used but is kept in the args so that the API is
        # the same for directional and point lights. The call sites should not
        # need to know the light type.
        return diffuse(
            normals=normals,
            color=self.diffuse_color,
            direction=self.direction,
        )

    def specular(self, normals, points, camera_position, shininess) -> torch.Tensor:
        return specular(
            points=points,
            normals=normals,
            color=self.specular_color,
            direction=self.direction,
            camera_position=camera_position,
            shininess=shininess,
        )


class PointLights(TensorProperties):
    def __init__(
        self,
        ambient_color=((0.5, 0.5, 0.5),),
        diffuse_color=((0.3, 0.3, 0.3),),
        specular_color=((0.2, 0.2, 0.2),),
        location=((0, 1, 0),),
        device: Device = "cpu",
    ) -> None:
        """
        Args:
            ambient_color: RGB color of the ambient component
            diffuse_color: RGB color of the diffuse component
            specular_color: RGB color of the specular component
            location: xyz position of the light.
            device: Device (as str or torch.device) on which the tensors should be located

        The inputs can each be
            - 3 element tuple/list or list of lists
            - torch tensor of shape (1, 3)
            - torch tensor of shape (N, 3)
        The inputs are broadcast against each other so they all have batch
        dimension N.
        """
        super().__init__(
            device=device,
            ambient_color=ambient_color,
            diffuse_color=diffuse_color,
            specular_color=specular_color,
            location=location,
        )
        _validate_light_properties(self)
        if self.location.shape[-1] != 3:
            msg = "Expected location to have shape (N, 3); got %r"
            raise ValueError(msg % repr(self.location.shape))

    def clone(self):
        other = self.__class__(device=self.device)
        return super().clone(other)

    def reshape_location(self, points) -> torch.Tensor:
        """
        Reshape the location tensor to have dimensions
        compatible with the points which can either be of
        shape (P, 3) or (N, H, W, K, 3).
        """
        if self.location.ndim == points.ndim:
            # pyre-fixme[7]: Expected `Tensor` but got `Union[Tensor, Module]`.
            return self.location
        # pyre-fixme[29]: `Union[(self: TensorBase, indices: Union[None, slice[Any, A...
        return self.location[:, None, None, None, :]

    def diffuse(self, normals, points) -> torch.Tensor:
        location = self.reshape_location(points)
        direction = location - points
        return diffuse(normals=normals, color=self.diffuse_color, direction=direction)

    def specular(self, normals, points, camera_position, shininess) -> torch.Tensor:
        location = self.reshape_location(points)
        direction = location - points
        return specular(
            points=points,
            normals=normals,
            color=self.specular_color,
            direction=direction,
            camera_position=camera_position,
            shininess=shininess,
        )


class AmbientLights(TensorProperties):
    """
    A light object representing the same color of light everywhere.
    By default, this is white, which effectively means lighting is
    not used in rendering.

    Unlike other lights this supports an arbitrary number of channels, not just 3 for RGB.
    The ambient_color input determines the number of channels.
    """

    def __init__(self, *, ambient_color=None, device: Device = "cpu") -> None:
        """
        If ambient_color is provided, it should be a sequence of
        triples of floats.

        Args:
            ambient_color: RGB color
            device: Device (as str or torch.device) on which the tensors should be located

        The ambient_color if provided, should be
            - tuple/list of C-element tuples of floats
            - torch tensor of shape (1, C)
            - torch tensor of shape (N, C)
        where C is the number of channels and N is batch size.
        For RGB, C is 3.
        """
        if ambient_color is None:
            ambient_color = ((1.0, 1.0, 1.0),)
        super().__init__(ambient_color=ambient_color, device=device)

    def clone(self):
        other = self.__class__(device=self.device)
        return super().clone(other)

    def diffuse(self, normals, points) -> torch.Tensor:
        return self._zeros_channels(points)

    def specular(self, normals, points, camera_position, shininess) -> torch.Tensor:
        return self._zeros_channels(points)

    def _zeros_channels(self, points: torch.Tensor) -> torch.Tensor:
        ch = self.ambient_color.shape[-1]
        return torch.zeros(*points.shape[:-1], ch, device=points.device)


def _validate_light_properties(obj) -> None:
    props = ("ambient_color", "diffuse_color", "specular_color")
    for n in props:
        t = getattr(obj, n)
        if t.shape[-1] != 3:
            msg = "Expected %s to have shape (N, 3); got %r"
            raise ValueError(msg % (n, t.shape))
