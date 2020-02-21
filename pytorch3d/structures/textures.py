#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from typing import List, Optional, Union
import torch
import torchvision.transforms as T

from .utils import list_to_packed, padded_to_list


"""
This file has functions for interpolating textures after rasterization.
"""


def _pad_texture_maps(images: List[torch.Tensor]) -> torch.Tensor:
    """
    Pad all texture images so they have the same height and width.

    Args:
        images: list of N tensors of shape (H, W)

    Returns:
        tex_maps: Tensor of shape (N, max_H, max_W)
    """
    tex_maps = []
    max_H = 0
    max_W = 0
    for im in images:
        h, w, _3 = im.shape
        if h > max_H:
            max_H = h
        if w > max_W:
            max_W = w
        tex_maps.append(im)
    max_shape = (max_H, max_W)

    # If all texture images are not the same size then resize to the
    # largest size.
    resize = T.Compose([T.ToPILImage(), T.Resize(size=max_shape), T.ToTensor()])

    for i, image in enumerate(tex_maps):
        if image.shape != max_shape:
            # ToPIL takes and returns a C x H x W tensor
            image = resize(image.permute(2, 0, 1)).permute(1, 2, 0)
            tex_maps[i] = image
    tex_maps = torch.stack(tex_maps, dim=0)  # (num_tex_maps, max_H, max_W, 3)
    return tex_maps


def _extend_tensor(input_tensor: torch.Tensor, N: int) -> torch.Tensor:
    """
    Extend a tensor `input_tensor` with ndim > 2, `N` times along the batch
    dimension. This is done in the following sequence of steps (where `B` is
    the batch dimension):

    .. code-block:: python

        input_tensor (B, ...)
        -> add leading empty dimension (1, B, ...)
        -> expand (N, B, ...)
        -> reshape (N * B, ...)

    Args:
        input_tensor: torch.Tensor with ndim > 2 representing a batched input.
        N: number of times to extend each element of the batch.
    """
    if input_tensor.ndim < 2:
        raise ValueError("Input tensor must have ndimensions >= 2.")
    B = input_tensor.shape[0]
    non_batch_dims = tuple(input_tensor.shape[1:])
    constant_dims = (-1,) * input_tensor.ndim  # these dims are not expanded.
    return (
        input_tensor.clone()[None, ...]
        .expand(N, *constant_dims)
        .transpose(0, 1)
        .reshape(N * B, *non_batch_dims)
    )


class Textures(object):
    def __init__(
        self,
        maps: Union[List, torch.Tensor, None] = None,
        faces_uvs: Optional[torch.Tensor] = None,
        verts_uvs: Optional[torch.Tensor] = None,
        verts_rgb: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            maps: texture map per mesh. This can either be a list of maps
              [(H, W, 3)] or a padded tensor of shape (N, H, W, 3).
            faces_uvs: (N, F, 3) tensor giving the index into verts_uvs for each
                vertex in the face. Padding value is assumed to be -1.
            verts_uvs: (N, V, 2) tensor giving the uv coordinate per vertex.
            verts_rgb: (N, V, 3) tensor giving the rgb color per vertex.
        """
        if faces_uvs is not None and faces_uvs.ndim != 3:
            msg = "Expected faces_uvs to be of shape (N, F, 3); got %r"
            raise ValueError(msg % repr(faces_uvs.shape))
        if verts_uvs is not None and verts_uvs.ndim != 3:
            msg = "Expected verts_uvs to be of shape (N, V, 2); got %r"
            raise ValueError(msg % repr(faces_uvs.shape))
        if verts_rgb is not None and verts_rgb.ndim != 3:
            msg = "Expected verts_rgb to be of shape (N, V, 3); got %r"
            raise ValueError(msg % verts_rgb.shape)
        if maps is not None:
            if torch.is_tensor(maps) and maps.ndim != 4:
                msg = "Expected maps to be of shape (N, H, W, 3); got %r"
                raise ValueError(msg % repr(maps.shape))
            elif isinstance(maps, list):
                maps = _pad_texture_maps(maps)
        self._faces_uvs_padded = faces_uvs
        self._verts_uvs_padded = verts_uvs
        self._verts_rgb_padded = verts_rgb
        self._maps_padded = maps
        self._num_faces_per_mesh = None

        if self._faces_uvs_padded is not None:
            self._num_faces_per_mesh = faces_uvs.gt(-1).all(-1).sum(-1).tolist()

    def clone(self):
        other = Textures()
        for k in dir(self):
            v = getattr(self, k)
            if torch.is_tensor(v):
                setattr(other, k, v.clone())
        return other

    def to(self, device):
        for k in dir(self):
            v = getattr(self, k)
            if torch.is_tensor(v) and v.device != device:
                setattr(self, k, v.to(device))
        return self

    def faces_uvs_padded(self) -> torch.Tensor:
        return self._faces_uvs_padded

    def faces_uvs_list(self) -> List[torch.Tensor]:
        if self._faces_uvs_padded is not None:
            return padded_to_list(
                self._faces_uvs_padded, split_size=self._num_faces_per_mesh
            )

    def faces_uvs_packed(self) -> torch.Tensor:
        return list_to_packed(self.faces_uvs_list())[0]

    def verts_uvs_padded(self) -> torch.Tensor:
        return self._verts_uvs_padded

    def verts_uvs_list(self) -> List[torch.Tensor]:
        return padded_to_list(self._verts_uvs_padded)

    def verts_uvs_packed(self) -> torch.Tensor:
        return list_to_packed(self.verts_uvs_list())[0]

    def verts_rgb_padded(self) -> torch.Tensor:
        return self._verts_rgb_padded

    def verts_rgb_list(self) -> List[torch.Tensor]:
        return padded_to_list(self._verts_rgb_padded)

    def verts_rgb_packed(self) -> torch.Tensor:
        return list_to_packed(self.verts_rgb_list())[0]

    # Currently only the padded maps are used.
    def maps_padded(self) -> torch.Tensor:
        return self._maps_padded

    def extend(self, N: int) -> "Textures":
        """
        Create new Textures class which contains each input texture N times

        Args:
            N: number of new copies of each texture.

        Returns:
            new Textures object.
        """
        if not isinstance(N, int):
            raise ValueError("N must be an integer.")
        if N <= 0:
            raise ValueError("N must be > 0.")

        if all(
            v is not None
            for v in [
                self._faces_uvs_padded,
                self._verts_uvs_padded,
                self._maps_padded,
            ]
        ):
            new_verts_uvs = _extend_tensor(self._verts_uvs_padded, N)
            new_faces_uvs = _extend_tensor(self._faces_uvs_padded, N)
            new_maps = _extend_tensor(self._maps_padded, N)
            return Textures(
                verts_uvs=new_verts_uvs, faces_uvs=new_faces_uvs, maps=new_maps
            )
        elif self._verts_rgb_padded is not None:
            new_verts_rgb = _extend_tensor(self._verts_rgb_padded, N)
            return Textures(verts_rgb=new_verts_rgb)
        else:
            msg = "Either vertex colors or texture maps are required."
            raise ValueError(msg)
