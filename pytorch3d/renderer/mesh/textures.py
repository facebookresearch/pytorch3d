# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import warnings
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING, Union

import torch
import torch.nn.functional as F
from pytorch3d.ops import interpolate_face_attributes
from pytorch3d.structures.utils import list_to_packed, list_to_padded, padded_to_list
from torch.nn.functional import interpolate

from .utils import pack_unique_rectangles, PackedRectangle, Rectangle


# This file contains classes and helper functions for texturing.
# There are three types of textures: TexturesVertex, TexturesAtlas
# and TexturesUV which inherit from a base textures class TexturesBase.
#
# Each texture class has a method 'sample_textures' to sample a
# value given barycentric coordinates.
#
# All the textures accept either list or padded inputs. The values
# are stored as either per face values (TexturesAtlas, TexturesUV),
# or per face vertex features (TexturesVertex).


def _list_to_padded_wrapper(
    x: List[torch.Tensor],
    pad_size: Union[list, tuple, None] = None,
    pad_value: float = 0.0,
) -> torch.Tensor:
    r"""
    This is a wrapper function for
    pytorch3d.structures.utils.list_to_padded function which only accepts
    3-dimensional inputs.

    For this use case, the input x is of shape (F, 3, ...) where only F
    is different for each element in the list

    Transforms a list of N tensors each of shape (Mi, ...) into a single tensor
    of shape (N, pad_size, ...), or (N, max(Mi), ...)
    if pad_size is None.

    Args:
      x: list of Tensors
      pad_size: int specifying the size of the first dimension
        of the padded tensor
      pad_value: float value to be used to fill the padded tensor

    Returns:
      x_padded: tensor consisting of padded input tensors
    """
    N = len(x)
    dims = x[0].ndim
    reshape_dims = x[0].shape[1:]
    D = torch.prod(torch.tensor(reshape_dims)).item()
    x_reshaped = []
    for y in x:
        if y.ndim != dims and y.shape[1:] != reshape_dims:
            msg = (
                "list_to_padded requires tensors to have the same number of dimensions"
            )
            raise ValueError(msg)
        # pyre-fixme[6]: For 2nd param expected `int` but got `Union[bool, float, int]`.
        x_reshaped.append(y.reshape(-1, D))
    x_padded = list_to_padded(x_reshaped, pad_size=pad_size, pad_value=pad_value)
    # pyre-fixme[58]: `+` is not supported for operand types `Tuple[int, int]` and
    #  `Size`.
    return x_padded.reshape((N, -1) + reshape_dims)


def _padded_to_list_wrapper(
    x: torch.Tensor, split_size: Union[list, tuple, None] = None
) -> List[torch.Tensor]:
    r"""
    This is a wrapper function for pytorch3d.structures.utils.padded_to_list
    which only accepts 3-dimensional inputs.

    For this use case, the input x is of shape (N, F, ...) where F
    is the number of faces which is different for each tensor in the batch.

    This function transforms a padded tensor of shape (N, M, ...) into a
    list of N tensors of shape (Mi, ...) where (Mi) is specified in
    split_size(i), or of shape (M,) if split_size is None.

    Args:
      x: padded Tensor
      split_size: list of ints defining the number of items for each tensor
        in the output list.

    Returns:
      x_list: a list of tensors
    """
    N, M = x.shape[:2]
    reshape_dims = x.shape[2:]
    D = torch.prod(torch.tensor(reshape_dims)).item()
    # pyre-fixme[6]: For 3rd param expected `int` but got `Union[bool, float, int]`.
    x_reshaped = x.reshape(N, M, D)
    x_list = padded_to_list(x_reshaped, split_size=split_size)
    # pyre-fixme[58]: `+` is not supported for operand types `Tuple[typing.Any]` and
    #  `Size`.
    x_list = [xl.reshape((xl.shape[0],) + reshape_dims) for xl in x_list]
    return x_list


def _pad_texture_maps(
    images: Union[Tuple[torch.Tensor], List[torch.Tensor]], align_corners: bool
) -> torch.Tensor:
    """
    Pad all texture images so they have the same height and width.

    Args:
        images: list of N tensors of shape (H_i, W_i, C)
        align_corners: used for interpolation

    Returns:
        tex_maps: Tensor of shape (N, max_H, max_W, C)
    """
    tex_maps = []
    max_H = 0
    max_W = 0
    for im in images:
        h, w, _C = im.shape
        if h > max_H:
            max_H = h
        if w > max_W:
            max_W = w
        tex_maps.append(im)
    max_shape = (max_H, max_W)

    for i, image in enumerate(tex_maps):
        if image.shape[:2] != max_shape:
            image_BCHW = image.permute(2, 0, 1)[None]
            new_image_BCHW = interpolate(
                image_BCHW,
                size=max_shape,
                mode="bilinear",
                align_corners=align_corners,
            )
            tex_maps[i] = new_image_BCHW[0].permute(1, 2, 0)
    tex_maps = torch.stack(tex_maps, dim=0)  # (num_tex_maps, max_H, max_W, C)
    return tex_maps


# A base class for defining a batch of textures
# with helper methods.
# This is also useful to have so that inside `Meshes`
# we can allow the input textures to be any texture
# type which is an instance of the base class.
class TexturesBase:
    def isempty(self):
        if self._N is not None and self.valid is not None:
            return self._N == 0 or self.valid.eq(False).all()
        return False

    def to(self, device):
        for k in dir(self):
            v = getattr(self, k)
            if isinstance(v, (list, tuple)) and all(
                torch.is_tensor(elem) for elem in v
            ):
                v = [elem.to(device) for elem in v]
                setattr(self, k, v)
            if torch.is_tensor(v) and v.device != device:
                setattr(self, k, v.to(device))
        self.device = device
        return self

    def _extend(self, N: int, props: List[str]) -> Dict[str, Union[torch.Tensor, List]]:
        """
        Create a dict with the specified properties
        repeated N times per batch element.

        Args:
            N: number of new copies of each texture
                in the batch.
            props: a List of strings which refer to either
                class attributes or class methods which
                return tensors or lists.

        Returns:
            Dict with the same keys as props. The values are the
            extended properties.
        """
        if not isinstance(N, int):
            raise ValueError("N must be an integer.")
        if N <= 0:
            raise ValueError("N must be > 0.")

        new_props = {}
        for p in props:
            t = getattr(self, p)
            if callable(t):
                t = t()  # class method
            if isinstance(t, list):
                if not all(isinstance(elem, (int, float)) for elem in t):
                    raise ValueError("Extend only supports lists of scalars")
                t = [[ti] * N for ti in t]
                new_props[p] = list(itertools.chain(*t))
            elif torch.is_tensor(t):
                new_props[p] = t.repeat_interleave(N, dim=0)
        return new_props

    def _getitem(self, index: Union[int, slice], props: List[str]):
        """
        Helper function for __getitem__
        """
        new_props = {}
        if isinstance(index, (int, slice)):
            for p in props:
                t = getattr(self, p)
                if callable(t):
                    t = t()  # class method
                new_props[p] = t[index]
        elif isinstance(index, list):
            index = torch.tensor(index)
        if isinstance(index, torch.Tensor):
            if index.dtype == torch.bool:
                index = index.nonzero()
                index = index.squeeze(1) if index.numel() > 0 else index
                index = index.tolist()
            for p in props:
                t = getattr(self, p)
                if callable(t):
                    t = t()  # class method
                new_props[p] = [t[i] for i in index]

        return new_props

    def sample_textures(self) -> torch.Tensor:
        """
        Different texture classes sample textures in different ways
        e.g. for vertex textures, the values at each vertex
        are interpolated across the face using the barycentric
        coordinates.
        Each texture class should implement a sample_textures
        method to take the `fragments` from rasterization.
        Using `fragments.pix_to_face` and `fragments.bary_coords`
        this function should return the sampled texture values for
        each pixel in the output image.
        """
        raise NotImplementedError()

    def submeshes(
        self,
        vertex_ids_list: List[List[torch.LongTensor]],
        faces_ids_list: List[List[torch.LongTensor]],
    ) -> "TexturesBase":
        """
        Extract sub-textures used for submeshing.
        """
        raise NotImplementedError(f"{self.__class__} does not support submeshes")

    def faces_verts_textures_packed(self) -> torch.Tensor:
        """
        Returns the texture for each vertex for each face in the mesh.
        For N meshes, this function returns sum(Fi)x3xC where Fi is the
        number of faces in the i-th mesh and C is the dimensional of
        the feature (C = 3 for RGB textures).
        You can use the utils function in structures.utils to convert the
        packed representation to a list or padded.
        """
        raise NotImplementedError()

    def clone(self) -> "TexturesBase":
        """
        Each texture class should implement a method
        to clone all necessary internal tensors.
        """
        raise NotImplementedError()

    def detach(self) -> "TexturesBase":
        """
        Each texture class should implement a method
        to detach all necessary internal tensors.
        """
        raise NotImplementedError()

    def __getitem__(self, index) -> "TexturesBase":
        """
        Each texture class should implement a method
        to get the texture properties for the
        specified elements in the batch.
        The TexturesBase._getitem(i) method
        can be used as a helper function to retrieve the
        class attributes for item i. Then, a new
        instance of the child class can be created with
        the attributes.
        """
        raise NotImplementedError()


def Textures(
    maps: Optional[Union[List[torch.Tensor], torch.Tensor]] = None,
    faces_uvs: Optional[torch.Tensor] = None,
    verts_uvs: Optional[torch.Tensor] = None,
    verts_rgb: Optional[torch.Tensor] = None,
) -> TexturesBase:
    """
    Textures class has been DEPRECATED.
    Preserving Textures as a function for backwards compatibility.

    Args:
        maps: texture map per mesh. This can either be a list of maps
          [(H, W, C)] or a padded tensor of shape (N, H, W, C).
        faces_uvs: (N, F, 3) tensor giving the index into verts_uvs for each
            vertex in the face. Padding value is assumed to be -1.
        verts_uvs: (N, V, 2) tensor giving the uv coordinate per vertex.
        verts_rgb: (N, V, C) tensor giving the color per vertex. Padding
            value is assumed to be -1. (C=3 for RGB.)


    Returns:
        a Textures class which is an instance of TexturesBase e.g. TexturesUV,
        TexturesAtlas, TexturesVertex

    """

    warnings.warn(
        """Textures class is deprecated,
        use TexturesUV, TexturesAtlas, TexturesVertex instead.
        Textures class will be removed in future releases.""",
        PendingDeprecationWarning,
    )

    if faces_uvs is not None and verts_uvs is not None and maps is not None:
        return TexturesUV(maps=maps, faces_uvs=faces_uvs, verts_uvs=verts_uvs)

    if verts_rgb is not None:
        return TexturesVertex(verts_features=verts_rgb)

    raise ValueError(
        "Textures either requires all three of (faces uvs, verts uvs, maps) or verts rgb"
    )


class TexturesAtlas(TexturesBase):
    def __init__(self, atlas: Union[torch.Tensor, List[torch.Tensor]]) -> None:
        """
        A texture representation where each face has a square texture map.
        This is based on the implementation from SoftRasterizer [1].

        Args:
            atlas: (N, F, R, R, C) tensor giving the per face texture map.
                The atlas can be created during obj loading with the
                pytorch3d.io.load_obj function - in the input arguments
                set `create_texture_atlas=True`. The atlas will be
                returned in aux.texture_atlas.


        The padded and list representations of the textures are stored
        and the packed representations is computed on the fly and
        not cached.

        [1] Liu et al, 'Soft Rasterizer: A Differentiable Renderer for Image-based
            3D Reasoning', ICCV 2019
            See also https://github.com/ShichenLiu/SoftRas/issues/21
        """
        if isinstance(atlas, (list, tuple)):
            correct_format = all(
                (
                    torch.is_tensor(elem)
                    and elem.ndim == 4
                    and elem.shape[1] == elem.shape[2]
                    and elem.shape[1] == atlas[0].shape[1]
                )
                for elem in atlas
            )
            if not correct_format:
                msg = (
                    "Expected atlas to be a list of tensors of shape (F, R, R, C) "
                    "with the same value of R."
                )
                raise ValueError(msg)
            self._atlas_list = atlas
            self._atlas_padded = None
            self.device = torch.device("cpu")

            # These values may be overridden when textures is
            # passed into the Meshes constructor. For more details
            # refer to the __init__ of Meshes.
            self._N = len(atlas)
            self._num_faces_per_mesh = [len(a) for a in atlas]

            if self._N > 0:
                self.device = atlas[0].device

        elif torch.is_tensor(atlas):
            if atlas.ndim != 5:
                msg = "Expected atlas to be of shape (N, F, R, R, C); got %r"
                raise ValueError(msg % repr(atlas.ndim))
            self._atlas_padded = atlas
            self._atlas_list = None
            self.device = atlas.device

            # These values may be overridden when textures is
            # passed into the Meshes constructor. For more details
            # refer to the __init__ of Meshes.
            self._N = len(atlas)
            max_F = atlas.shape[1]
            self._num_faces_per_mesh = [max_F] * self._N
        else:
            raise ValueError("Expected atlas to be a tensor or list")

        # The num_faces_per_mesh, N and valid
        # are reset inside the Meshes object when textures is
        # passed into the Meshes constructor. For more details
        # refer to the __init__ of Meshes.
        self.valid = torch.ones((self._N,), dtype=torch.bool, device=self.device)

    def clone(self) -> "TexturesAtlas":
        tex = self.__class__(atlas=self.atlas_padded().clone())
        if self._atlas_list is not None:
            tex._atlas_list = [atlas.clone() for atlas in self._atlas_list]
        num_faces = (
            self._num_faces_per_mesh.clone()
            if torch.is_tensor(self._num_faces_per_mesh)
            else self._num_faces_per_mesh
        )
        tex.valid = self.valid.clone()
        tex._num_faces_per_mesh = num_faces
        return tex

    def detach(self) -> "TexturesAtlas":
        tex = self.__class__(atlas=self.atlas_padded().detach())
        if self._atlas_list is not None:
            tex._atlas_list = [atlas.detach() for atlas in self._atlas_list]
        num_faces = (
            self._num_faces_per_mesh.detach()
            if torch.is_tensor(self._num_faces_per_mesh)
            else self._num_faces_per_mesh
        )
        tex.valid = self.valid.detach()
        tex._num_faces_per_mesh = num_faces
        return tex

    def __getitem__(self, index) -> "TexturesAtlas":
        props = ["atlas_list", "_num_faces_per_mesh"]
        new_props = self._getitem(index, props=props)
        atlas = new_props["atlas_list"]
        if isinstance(atlas, list):
            # multiple batch elements
            new_tex = self.__class__(atlas=atlas)
        elif torch.is_tensor(atlas):
            # single element
            new_tex = self.__class__(atlas=[atlas])
        else:
            raise ValueError("Not all values are provided in the correct format")
        new_tex._num_faces_per_mesh = new_props["_num_faces_per_mesh"]
        return new_tex

    def atlas_padded(self) -> torch.Tensor:
        if self._atlas_padded is None:
            if self.isempty():
                self._atlas_padded = torch.zeros(
                    (self._N, 0, 0, 0, 3), dtype=torch.float32, device=self.device
                )
            else:
                self._atlas_padded = _list_to_padded_wrapper(
                    self._atlas_list, pad_value=0.0
                )
        return self._atlas_padded

    def atlas_list(self) -> List[torch.Tensor]:
        if self._atlas_list is None:
            if self.isempty():
                self._atlas_padded = [
                    torch.empty((0, 0, 0, 3), dtype=torch.float32, device=self.device)
                ] * self._N
            self._atlas_list = _padded_to_list_wrapper(
                self._atlas_padded, split_size=self._num_faces_per_mesh
            )
        return self._atlas_list

    def atlas_packed(self) -> torch.Tensor:
        if self.isempty():
            return torch.zeros(
                (self._N, 0, 0, 3), dtype=torch.float32, device=self.device
            )
        atlas_list = self.atlas_list()
        return list_to_packed(atlas_list)[0]

    def extend(self, N: int) -> "TexturesAtlas":
        new_props = self._extend(N, ["atlas_padded", "_num_faces_per_mesh"])
        new_tex = self.__class__(atlas=new_props["atlas_padded"])
        new_tex._num_faces_per_mesh = new_props["_num_faces_per_mesh"]
        return new_tex

    # pyre-fixme[14]: `sample_textures` overrides method defined in `TexturesBase`
    #  inconsistently.
    def sample_textures(self, fragments, **kwargs) -> torch.Tensor:
        """
        This is similar to a nearest neighbor sampling and involves a
        discretization step. The barycentric coordinates from
        rasterization are used to find the nearest grid cell in the texture
        atlas and the RGB is returned as the color.
        This means that this step is differentiable with respect to the RGB
        values of the texture atlas but not differentiable with respect to the
        barycentric coordinates.

        TODO: Add a different sampling mode which interpolates the barycentric
        coordinates to sample the texture and will be differentiable w.r.t
        the barycentric coordinates.

        Args:
            fragments:
                The outputs of rasterization. From this we use

                - pix_to_face: LongTensor of shape (N, H, W, K) specifying the indices
                of the faces (in the packed representation) which
                overlap each pixel in the image.
                - barycentric_coords: FloatTensor of shape (N, H, W, K, 3) specifying
                the barycentric coordinates of each pixel
                relative to the faces (in the packed
                representation) which overlap the pixel.

        Returns:
            texels: (N, H, W, K, C)
        """
        N, H, W, K = fragments.pix_to_face.shape
        atlas_packed = self.atlas_packed()
        R = atlas_packed.shape[1]
        bary = fragments.bary_coords
        pix_to_face = fragments.pix_to_face

        bary_w01 = bary[..., :2]
        # pyre-fixme[16]: `bool` has no attribute `__getitem__`.
        mask = (pix_to_face < 0)[..., None]
        bary_w01 = torch.where(mask, torch.zeros_like(bary_w01), bary_w01)
        # If barycentric coordinates are > 1.0 (in the case of
        # blur_radius > 0.0), wxy might be > R. We need to clamp this
        # index to R-1 to index into the texture atlas.
        w_xy = (bary_w01 * R).to(torch.int64).clamp(max=R - 1)  # (N, H, W, K, 2)

        below_diag = (
            bary_w01.sum(dim=-1) * R - w_xy.float().sum(dim=-1)
        ) <= 1.0  # (N, H, W, K)
        w_x, w_y = w_xy.unbind(-1)
        w_x = torch.where(below_diag, w_x, (R - 1 - w_x))
        w_y = torch.where(below_diag, w_y, (R - 1 - w_y))

        texels = atlas_packed[pix_to_face, w_y, w_x]
        texels = texels * (pix_to_face >= 0)[..., None].float()

        return texels

    def submeshes(
        self,
        vertex_ids_list: List[List[torch.LongTensor]],
        faces_ids_list: List[List[torch.LongTensor]],
    ) -> "TexturesAtlas":
        """
        Extract a sub-texture for use in a submesh.

        If the meshes batch corresponding to this TextureAtlas contains
        `n = len(faces_ids_list)` meshes, then self.atlas_list()
        will be of length n. After submeshing, we obtain a batch of
        `k = sum(len(v) for v in atlas_list` submeshes (see Meshes.submeshes). This
        function creates a corresponding TexturesAtlas object with `atlas_list`
        of length `k`.
        """
        if len(faces_ids_list) != len(self.atlas_list()):
            raise IndexError(
                "faces_ids_list must be of " "the same length as atlas_list."
            )

        sub_features = []
        for atlas, faces_ids in zip(self.atlas_list(), faces_ids_list):
            for faces_ids_submesh in faces_ids:
                sub_features.append(atlas[faces_ids_submesh])

        return self.__class__(sub_features)

    def faces_verts_textures_packed(self) -> torch.Tensor:
        """
        Samples texture from each vertex for each face in the mesh.
        For N meshes with {Fi} number of faces, it returns a
        tensor of shape sum(Fi)x3xC (C = 3 for RGB).
        You can use the utils function in structures.utils to convert the
        packed representation to a list or padded.
        """
        atlas_packed = self.atlas_packed()
        # assume each face consists of (v0, v1, v2).
        # to sample from the atlas we only need the first two barycentric coordinates.
        # for details on how this texture sample works refer to the sample_textures function.
        t0 = atlas_packed[:, 0, -1]  # corresponding to v0  with bary = (1, 0)
        t1 = atlas_packed[:, -1, 0]  # corresponding to v1 with bary = (0, 1)
        t2 = atlas_packed[:, 0, 0]  # corresponding to v2 with bary = (0, 0)
        return torch.stack((t0, t1, t2), dim=1)

    def join_batch(self, textures: List["TexturesAtlas"]) -> "TexturesAtlas":
        """
        Join the list of textures given by `textures` to
        self to create a batch of textures. Return a new
        TexturesAtlas object with the combined textures.

        Args:
            textures: List of TexturesAtlas objects

        Returns:
            new_tex: TexturesAtlas object with the combined
            textures from self and the list `textures`.
        """
        tex_types_same = all(isinstance(tex, TexturesAtlas) for tex in textures)
        if not tex_types_same:
            raise ValueError("All textures must be of type TexturesAtlas.")

        atlas_list = []
        atlas_list += self.atlas_list()
        num_faces_per_mesh = self._num_faces_per_mesh.copy()
        for tex in textures:
            atlas_list += tex.atlas_list()
            num_faces_per_mesh += tex._num_faces_per_mesh
        new_tex = self.__class__(atlas=atlas_list)
        new_tex._num_faces_per_mesh = num_faces_per_mesh
        return new_tex

    def join_scene(self) -> "TexturesAtlas":
        """
        Return a new TexturesAtlas amalgamating the batch.
        """
        return self.__class__(atlas=[torch.cat(self.atlas_list())])

    def check_shapes(
        self, batch_size: int, max_num_verts: int, max_num_faces: int
    ) -> bool:
        """
        Check if the dimensions of the atlas match that of the mesh faces
        """
        # (N, F) should be the same
        return self.atlas_padded().shape[0:2] == (batch_size, max_num_faces)


class TexturesUV(TexturesBase):
    def __init__(
        self,
        maps: Union[torch.Tensor, List[torch.Tensor]],
        faces_uvs: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor]],
        verts_uvs: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor]],
        padding_mode: str = "border",
        align_corners: bool = True,
        sampling_mode: str = "bilinear",
    ) -> None:
        """
        Textures are represented as a per mesh texture map and uv coordinates for each
        vertex in each face. NOTE: this class only supports one texture map per mesh.

        Args:
            maps: texture map per mesh. This can either be a list of maps
              [(H, W, C)] or a padded tensor of shape (N, H, W, C).
              For RGB, C = 3.
            faces_uvs: (N, F, 3) LongTensor giving the index into verts_uvs
                        for each face
            verts_uvs: (N, V, 2) tensor giving the uv coordinates per vertex
                        (a FloatTensor with values between 0 and 1).
            align_corners: If true, the extreme values 0 and 1 for verts_uvs
                            indicate the centers of the edge pixels in the maps.
            padding_mode: padding mode for outside grid values
                                ("zeros", "border" or "reflection").
            sampling_mode: type of interpolation used to sample the texture.
                            Corresponds to the mode parameter in PyTorch's
                            grid_sample ("nearest" or "bilinear").

        The align_corners and padding_mode arguments correspond to the arguments
        of the `grid_sample` torch function. There is an informative illustration of
        the two align_corners options at
        https://discuss.pytorch.org/t/22663/9 .

        An example of how the indexing into the maps, with align_corners=True,
        works is as follows.
        If maps[i] has shape [1001, 101] and the value of verts_uvs[i][j]
        is [0.4, 0.3], then a value of j in faces_uvs[i] means a vertex
        whose color is given by maps[i][700, 40]. padding_mode affects what
        happens if a value in verts_uvs is less than 0 or greater than 1.
        Note that increasing a value in verts_uvs[..., 0] increases an index
        in maps, whereas increasing a value in verts_uvs[..., 1] _decreases_
        an _earlier_ index in maps.

        If align_corners=False, an example would be as follows.
        If maps[i] has shape [1000, 100] and the value of verts_uvs[i][j]
        is [0.405, 0.2995], then a value of j in faces_uvs[i] means a vertex
        whose color is given by maps[i][700, 40].
        When align_corners=False, padding_mode even matters for values in
        verts_uvs slightly above 0 or slightly below 1. In this case, the
        padding_mode matters if the first value is outside the interval
        [0.0005, 0.9995] or if the second is outside the interval
        [0.005, 0.995].
        """
        self.padding_mode = padding_mode
        self.align_corners = align_corners
        self.sampling_mode = sampling_mode
        if isinstance(faces_uvs, (list, tuple)):
            for fv in faces_uvs:
                if fv.ndim != 2 or fv.shape[-1] != 3:
                    msg = "Expected faces_uvs to be of shape (F, 3); got %r"
                    raise ValueError(msg % repr(fv.shape))
            self._faces_uvs_list = faces_uvs
            self._faces_uvs_padded = None
            self.device = torch.device("cpu")

            # These values may be overridden when textures is
            # passed into the Meshes constructor. For more details
            # refer to the __init__ of Meshes.
            self._N = len(faces_uvs)
            self._num_faces_per_mesh = [len(fv) for fv in faces_uvs]

            if self._N > 0:
                self.device = faces_uvs[0].device

        elif torch.is_tensor(faces_uvs):
            if faces_uvs.ndim != 3 or faces_uvs.shape[-1] != 3:
                msg = "Expected faces_uvs to be of shape (N, F, 3); got %r"
                raise ValueError(msg % repr(faces_uvs.shape))
            self._faces_uvs_padded = faces_uvs
            self._faces_uvs_list = None
            self.device = faces_uvs.device

            # These values may be overridden when textures is
            # passed into the Meshes constructor. For more details
            # refer to the __init__ of Meshes.
            self._N = len(faces_uvs)
            max_F = faces_uvs.shape[1]
            self._num_faces_per_mesh = [max_F] * self._N
        else:
            raise ValueError("Expected faces_uvs to be a tensor or list")

        if isinstance(verts_uvs, (list, tuple)):
            for fv in verts_uvs:
                if fv.ndim != 2 or fv.shape[-1] != 2:
                    msg = "Expected verts_uvs to be of shape (V, 2); got %r"
                    raise ValueError(msg % repr(fv.shape))
            self._verts_uvs_list = verts_uvs
            self._verts_uvs_padded = None

            if len(verts_uvs) != self._N:
                raise ValueError(
                    "verts_uvs and faces_uvs must have the same batch dimension"
                )
            if not all(v.device == self.device for v in verts_uvs):
                raise ValueError("verts_uvs and faces_uvs must be on the same device")

        elif torch.is_tensor(verts_uvs):
            if (
                verts_uvs.ndim != 3
                or verts_uvs.shape[-1] != 2
                or verts_uvs.shape[0] != self._N
            ):
                msg = "Expected verts_uvs to be of shape (N, V, 2); got %r"
                raise ValueError(msg % repr(verts_uvs.shape))
            self._verts_uvs_padded = verts_uvs
            self._verts_uvs_list = None

            if verts_uvs.device != self.device:
                raise ValueError("verts_uvs and faces_uvs must be on the same device")
        else:
            raise ValueError("Expected verts_uvs to be a tensor or list")

        if isinstance(maps, (list, tuple)):
            self._maps_list = maps
        else:
            self._maps_list = None
        self._maps_padded = self._format_maps_padded(maps)

        if self._maps_padded.device != self.device:
            raise ValueError("maps must be on the same device as verts/faces uvs.")

        self.valid = torch.ones((self._N,), dtype=torch.bool, device=self.device)

    def _format_maps_padded(
        self, maps: Union[torch.Tensor, List[torch.Tensor]]
    ) -> torch.Tensor:
        if isinstance(maps, torch.Tensor):
            if maps.ndim != 4 or maps.shape[0] != self._N:
                msg = "Expected maps to be of shape (N, H, W, C); got %r"
                raise ValueError(msg % repr(maps.shape))
            return maps

        if isinstance(maps, (list, tuple)):
            if len(maps) != self._N:
                raise ValueError("Expected one texture map per mesh in the batch.")
            if self._N > 0:
                if not all(map.ndim == 3 for map in maps):
                    raise ValueError("Invalid number of dimensions in texture maps")
                if not all(map.shape[2] == maps[0].shape[2] for map in maps):
                    raise ValueError("Inconsistent number of channels in maps")
                maps_padded = _pad_texture_maps(maps, align_corners=self.align_corners)
            else:
                maps_padded = torch.empty(
                    (self._N, 0, 0, 3), dtype=torch.float32, device=self.device
                )
            return maps_padded

        raise ValueError("Expected maps to be a tensor or list of tensors.")

    def clone(self) -> "TexturesUV":
        tex = self.__class__(
            self.maps_padded().clone(),
            self.faces_uvs_padded().clone(),
            self.verts_uvs_padded().clone(),
            align_corners=self.align_corners,
            padding_mode=self.padding_mode,
            sampling_mode=self.sampling_mode,
        )
        if self._maps_list is not None:
            tex._maps_list = [m.clone() for m in self._maps_list]
        if self._verts_uvs_list is not None:
            tex._verts_uvs_list = [v.clone() for v in self._verts_uvs_list]
        if self._faces_uvs_list is not None:
            tex._faces_uvs_list = [f.clone() for f in self._faces_uvs_list]
        num_faces = (
            self._num_faces_per_mesh.clone()
            if torch.is_tensor(self._num_faces_per_mesh)
            else self._num_faces_per_mesh
        )
        tex._num_faces_per_mesh = num_faces
        tex.valid = self.valid.clone()
        return tex

    def detach(self) -> "TexturesUV":
        tex = self.__class__(
            self.maps_padded().detach(),
            self.faces_uvs_padded().detach(),
            self.verts_uvs_padded().detach(),
            align_corners=self.align_corners,
            padding_mode=self.padding_mode,
            sampling_mode=self.sampling_mode,
        )
        if self._maps_list is not None:
            tex._maps_list = [m.detach() for m in self._maps_list]
        if self._verts_uvs_list is not None:
            tex._verts_uvs_list = [v.detach() for v in self._verts_uvs_list]
        if self._faces_uvs_list is not None:
            tex._faces_uvs_list = [f.detach() for f in self._faces_uvs_list]
        num_faces = (
            self._num_faces_per_mesh.detach()
            if torch.is_tensor(self._num_faces_per_mesh)
            else self._num_faces_per_mesh
        )
        tex._num_faces_per_mesh = num_faces
        tex.valid = self.valid.detach()
        return tex

    def __getitem__(self, index) -> "TexturesUV":
        props = ["verts_uvs_list", "faces_uvs_list", "maps_list", "_num_faces_per_mesh"]
        new_props = self._getitem(index, props)
        faces_uvs = new_props["faces_uvs_list"]
        verts_uvs = new_props["verts_uvs_list"]
        maps = new_props["maps_list"]

        # if index has multiple values then faces/verts/maps may be a list of tensors
        if all(isinstance(f, (list, tuple)) for f in [faces_uvs, verts_uvs, maps]):
            new_tex = self.__class__(
                faces_uvs=faces_uvs,
                verts_uvs=verts_uvs,
                maps=maps,
                padding_mode=self.padding_mode,
                align_corners=self.align_corners,
                sampling_mode=self.sampling_mode,
            )
        elif all(torch.is_tensor(f) for f in [faces_uvs, verts_uvs, maps]):
            new_tex = self.__class__(
                faces_uvs=[faces_uvs],
                verts_uvs=[verts_uvs],
                maps=[maps],
                padding_mode=self.padding_mode,
                align_corners=self.align_corners,
                sampling_mode=self.sampling_mode,
            )
        else:
            raise ValueError("Not all values are provided in the correct format")
        new_tex._num_faces_per_mesh = new_props["_num_faces_per_mesh"]
        return new_tex

    def faces_uvs_padded(self) -> torch.Tensor:
        if self._faces_uvs_padded is None:
            if self.isempty():
                self._faces_uvs_padded = torch.zeros(
                    (self._N, 0, 3), dtype=torch.float32, device=self.device
                )
            else:
                self._faces_uvs_padded = list_to_padded(
                    self._faces_uvs_list, pad_value=0.0
                )
        return self._faces_uvs_padded

    def faces_uvs_list(self) -> List[torch.Tensor]:
        if self._faces_uvs_list is None:
            if self.isempty():
                self._faces_uvs_list = [
                    torch.empty((0, 3), dtype=torch.float32, device=self.device)
                ] * self._N
            else:
                self._faces_uvs_list = padded_to_list(
                    self._faces_uvs_padded, split_size=self._num_faces_per_mesh
                )
        return self._faces_uvs_list

    def verts_uvs_padded(self) -> torch.Tensor:
        if self._verts_uvs_padded is None:
            if self.isempty():
                self._verts_uvs_padded = torch.zeros(
                    (self._N, 0, 2), dtype=torch.float32, device=self.device
                )
            else:
                self._verts_uvs_padded = list_to_padded(
                    self._verts_uvs_list, pad_value=0.0
                )
        return self._verts_uvs_padded

    def verts_uvs_list(self) -> List[torch.Tensor]:
        if self._verts_uvs_list is None:
            if self.isempty():
                self._verts_uvs_list = [
                    torch.empty((0, 2), dtype=torch.float32, device=self.device)
                ] * self._N
            else:
                # The number of vertices in the mesh and in verts_uvs can differ
                # e.g. if a vertex is shared between 3 faces, it can
                # have up to 3 different uv coordinates.
                self._verts_uvs_list = list(self._verts_uvs_padded.unbind(0))
        return self._verts_uvs_list

    # Currently only the padded maps are used.
    def maps_padded(self) -> torch.Tensor:
        return self._maps_padded

    def maps_list(self) -> List[torch.Tensor]:
        if self._maps_list is not None:
            return self._maps_list
        return self._maps_padded.unbind(0)

    def extend(self, N: int) -> "TexturesUV":
        new_props = self._extend(
            N,
            [
                "maps_padded",
                "verts_uvs_padded",
                "faces_uvs_padded",
                "_num_faces_per_mesh",
            ],
        )
        new_tex = self.__class__(
            maps=new_props["maps_padded"],
            faces_uvs=new_props["faces_uvs_padded"],
            verts_uvs=new_props["verts_uvs_padded"],
            padding_mode=self.padding_mode,
            align_corners=self.align_corners,
            sampling_mode=self.sampling_mode,
        )

        new_tex._num_faces_per_mesh = new_props["_num_faces_per_mesh"]
        return new_tex

    # pyre-fixme[14]: `sample_textures` overrides method defined in `TexturesBase`
    #  inconsistently.
    def sample_textures(self, fragments, **kwargs) -> torch.Tensor:
        """
        Interpolate a 2D texture map using uv vertex texture coordinates for each
        face in the mesh. First interpolate the vertex uvs using barycentric coordinates
        for each pixel in the rasterized output. Then interpolate the texture map
        using the uv coordinate for each pixel.

        Args:
            fragments:
                The outputs of rasterization. From this we use

                - pix_to_face: LongTensor of shape (N, H, W, K) specifying the indices
                of the faces (in the packed representation) which
                overlap each pixel in the image.
                - barycentric_coords: FloatTensor of shape (N, H, W, K, 3) specifying
                the barycentric coordinates of each pixel
                relative to the faces (in the packed
                representation) which overlap the pixel.

        Returns:
            texels: tensor of shape (N, H, W, K, C) giving the interpolated
            texture for each pixel in the rasterized image.
        """
        if self.isempty():
            faces_verts_uvs = torch.zeros(
                (self._N, 3, 2), dtype=torch.float32, device=self.device
            )
        else:
            packing_list = [
                i[j] for i, j in zip(self.verts_uvs_list(), self.faces_uvs_list())
            ]
            faces_verts_uvs = torch.cat(packing_list)
        texture_maps = self.maps_padded()

        # pixel_uvs: (N, H, W, K, 2)
        pixel_uvs = interpolate_face_attributes(
            fragments.pix_to_face, fragments.bary_coords, faces_verts_uvs
        )

        N, H_out, W_out, K = fragments.pix_to_face.shape
        N, H_in, W_in, C = texture_maps.shape  # 3 for RGB

        # pixel_uvs: (N, H, W, K, 2) -> (N, K, H, W, 2) -> (NK, H, W, 2)
        pixel_uvs = pixel_uvs.permute(0, 3, 1, 2, 4).reshape(N * K, H_out, W_out, 2)

        # textures.map:
        #   (N, H, W, C) -> (N, C, H, W) -> (1, N, C, H, W)
        #   -> expand (K, N, C, H, W) -> reshape (N*K, C, H, W)
        texture_maps = (
            texture_maps.permute(0, 3, 1, 2)[None, ...]
            .expand(K, -1, -1, -1, -1)
            .transpose(0, 1)
            .reshape(N * K, C, H_in, W_in)
        )

        # Textures: (N*K, C, H, W), pixel_uvs: (N*K, H, W, 2)
        # Now need to format the pixel uvs and the texture map correctly!
        # From pytorch docs, grid_sample takes `grid` and `input`:
        #   grid specifies the sampling pixel locations normalized by
        #   the input spatial dimensions It should have most
        #   values in the range of [-1, 1]. Values x = -1, y = -1
        #   is the left-top pixel of input, and values x = 1, y = 1 is the
        #   right-bottom pixel of input.

        # map to a range of [-1, 1] and flip the y axis
        pixel_uvs = torch.lerp(
            pixel_uvs.new_tensor([-1.0, 1.0]),
            pixel_uvs.new_tensor([1.0, -1.0]),
            pixel_uvs,
        )

        if texture_maps.device != pixel_uvs.device:
            texture_maps = texture_maps.to(pixel_uvs.device)
        texels = F.grid_sample(
            texture_maps,
            pixel_uvs,
            mode=self.sampling_mode,
            align_corners=self.align_corners,
            padding_mode=self.padding_mode,
        )
        # texels now has shape (NK, C, H_out, W_out)
        texels = texels.reshape(N, K, C, H_out, W_out).permute(0, 3, 4, 1, 2)
        return texels

    def faces_verts_textures_packed(self) -> torch.Tensor:
        """
        Samples texture from each vertex and for each face in the mesh.
        For N meshes with {Fi} number of faces, it returns a
        tensor of shape sum(Fi)x3xC (C = 3 for RGB).
        You can use the utils function in structures.utils to convert the
        packed representation to a list or padded.
        """
        if self.isempty():
            return torch.zeros(
                (0, 3, self.maps_padded().shape[-1]),
                dtype=torch.float32,
                device=self.device,
            )
        else:
            packing_list = [
                i[j] for i, j in zip(self.verts_uvs_list(), self.faces_uvs_list())
            ]
            faces_verts_uvs = _list_to_padded_wrapper(
                packing_list, pad_value=0.0
            )  # Nxmax(Fi)x3x2
        texture_maps = self.maps_padded()  # NxHxWxC
        texture_maps = texture_maps.permute(0, 3, 1, 2)  # NxCxHxW

        # map to a range of [-1, 1] and flip the y axis
        faces_verts_uvs = torch.lerp(
            faces_verts_uvs.new_tensor([-1.0, 1.0]),
            faces_verts_uvs.new_tensor([1.0, -1.0]),
            faces_verts_uvs,
        )

        textures = F.grid_sample(
            texture_maps,
            faces_verts_uvs,
            mode=self.sampling_mode,
            align_corners=self.align_corners,
            padding_mode=self.padding_mode,
        )  # NxCxmax(Fi)x3

        textures = textures.permute(0, 2, 3, 1)  # Nxmax(Fi)x3xC
        textures = _padded_to_list_wrapper(
            textures, split_size=self._num_faces_per_mesh
        )  # list of N {Fix3xC} tensors
        return list_to_packed(textures)[0]

    def join_batch(self, textures: List["TexturesUV"]) -> "TexturesUV":
        """
        Join the list of textures given by `textures` to
        self to create a batch of textures. Return a new
        TexturesUV object with the combined textures.

        Args:
            textures: List of TexturesUV objects

        Returns:
            new_tex: TexturesUV object with the combined
            textures from self and the list `textures`.
        """
        tex_types_same = all(isinstance(tex, TexturesUV) for tex in textures)
        if not tex_types_same:
            raise ValueError("All textures must be of type TexturesUV.")

        padding_modes_same = all(
            tex.padding_mode == self.padding_mode for tex in textures
        )
        if not padding_modes_same:
            raise ValueError("All textures must have the same padding_mode.")
        align_corners_same = all(
            tex.align_corners == self.align_corners for tex in textures
        )
        if not align_corners_same:
            raise ValueError("All textures must have the same align_corners value.")
        sampling_mode_same = all(
            tex.sampling_mode == self.sampling_mode for tex in textures
        )
        if not sampling_mode_same:
            raise ValueError("All textures must have the same sampling_mode.")

        verts_uvs_list = []
        faces_uvs_list = []
        maps_list = []
        faces_uvs_list += self.faces_uvs_list()
        verts_uvs_list += self.verts_uvs_list()
        maps_list += self.maps_list()
        num_faces_per_mesh = self._num_faces_per_mesh.copy()
        for tex in textures:
            verts_uvs_list += tex.verts_uvs_list()
            faces_uvs_list += tex.faces_uvs_list()
            num_faces_per_mesh += tex._num_faces_per_mesh
            maps_list += tex.maps_list()

        new_tex = self.__class__(
            maps=maps_list,
            verts_uvs=verts_uvs_list,
            faces_uvs=faces_uvs_list,
            padding_mode=self.padding_mode,
            align_corners=self.align_corners,
            sampling_mode=self.sampling_mode,
        )
        new_tex._num_faces_per_mesh = num_faces_per_mesh
        return new_tex

    def _place_map_into_single_map(
        self, single_map: torch.Tensor, map_: torch.Tensor, location: PackedRectangle
    ) -> None:
        """
        Copy map into a larger tensor single_map at the destination specified by location.
        If align_corners is False, we add the needed border around the destination.

        Used by join_scene.

        Args:
            single_map: (total_H, total_W, C)
            map_: (H, W, C) source data
            location: where to place map
        """
        do_flip = location.flipped
        source = map_.transpose(0, 1) if do_flip else map_
        border_width = 0 if self.align_corners else 1
        lower_u = location.x + border_width
        lower_v = location.y + border_width
        upper_u = lower_u + source.shape[0]
        upper_v = lower_v + source.shape[1]
        single_map[lower_u:upper_u, lower_v:upper_v] = source

        if self.padding_mode != "zeros" and not self.align_corners:
            single_map[lower_u - 1, lower_v:upper_v] = single_map[
                lower_u, lower_v:upper_v
            ]
            single_map[upper_u, lower_v:upper_v] = single_map[
                upper_u - 1, lower_v:upper_v
            ]
            single_map[lower_u:upper_u, lower_v - 1] = single_map[
                lower_u:upper_u, lower_v
            ]
            single_map[lower_u:upper_u, upper_v] = single_map[
                lower_u:upper_u, upper_v - 1
            ]
            single_map[lower_u - 1, lower_v - 1] = single_map[lower_u, lower_v]
            single_map[lower_u - 1, upper_v] = single_map[lower_u, upper_v - 1]
            single_map[upper_u, lower_v - 1] = single_map[upper_u - 1, lower_v]
            single_map[upper_u, upper_v] = single_map[upper_u - 1, upper_v - 1]

    def join_scene(self) -> "TexturesUV":
        """
        Return a new TexturesUV amalgamating the batch.

        We calculate a large single map which contains the original maps,
        and find verts_uvs to point into it. This will not replicate
        behavior of padding for verts_uvs values outside [0,1].

        If align_corners=False, we need to add an artificial border around
        every map.

        We use the function `pack_unique_rectangles` to provide a layout for
        the single map. This means that if self was created with a list of maps,
        and to() has not been called, and there were two maps which were exactly
        the same tensor object, then they will become the same data in the unified map.
        _place_map_into_single_map is used to copy the maps into the single map.
        The merging of verts_uvs and faces_uvs is handled locally in this function.
        """
        maps = self.maps_list()
        heights_and_widths = []
        extra_border = 0 if self.align_corners else 2
        for map_ in maps:
            heights_and_widths.append(
                Rectangle(
                    map_.shape[0] + extra_border, map_.shape[1] + extra_border, id(map_)
                )
            )
        merging_plan = pack_unique_rectangles(heights_and_widths)
        C = maps[0].shape[-1]
        single_map = maps[0].new_zeros((*merging_plan.total_size, C))
        verts_uvs = self.verts_uvs_list()
        verts_uvs_merged = []

        for map_, loc, uvs in zip(maps, merging_plan.locations, verts_uvs):
            new_uvs = uvs.clone()
            if loc.is_first:
                self._place_map_into_single_map(single_map, map_, loc)
            do_flip = loc.flipped
            x_shape = map_.shape[1] if do_flip else map_.shape[0]
            y_shape = map_.shape[0] if do_flip else map_.shape[1]

            if do_flip:
                # Here we have flipped / transposed the map.
                # In uvs, the y values are decreasing from 1 to 0 and the x
                # values increase from 0 to 1. We subtract all values from 1
                # as the x's become y's and the y's become x's.
                new_uvs = 1.0 - new_uvs[:, [1, 0]]
                if TYPE_CHECKING:
                    new_uvs = torch.Tensor(new_uvs)

            # If align_corners is True, then an index of x (where x is in
            # the range 0 .. map_.shape[1]-1) in one of the input maps
            # was hit by a u of x/(map_.shape[1]-1).
            # That x is located at the index loc[1] + x in the single_map, and
            # to hit that we need u to equal (loc[1] + x) / (total_size[1]-1)
            # so the old u should be mapped to
            #   { u*(map_.shape[1]-1) + loc[1] } / (total_size[1]-1)

            # Also, an index of y (where y is in
            # the range 0 .. map_.shape[0]-1) in one of the input maps
            # was hit by a v of 1 - y/(map_.shape[0]-1).
            # That y is located at the index loc[0] + y in the single_map, and
            # to hit that we need v to equal 1 - (loc[0] + y) / (total_size[0]-1)
            # so the old v should be mapped to
            #   1 - { (1-v)*(map_.shape[0]-1) + loc[0] } / (total_size[0]-1)
            # =
            # { v*(map_.shape[0]-1) + total_size[0] - map.shape[0] - loc[0] }
            #        / (total_size[0]-1)

            # If align_corners is False, then an index of x (where x is in
            # the range 1 .. map_.shape[1]-2) in one of the input maps
            # was hit by a u of (x+0.5)/(map_.shape[1]).
            # That x is located at the index loc[1] + 1 + x in the single_map,
            # (where the 1 is for the border)
            # and to hit that we need u to equal (loc[1] + 1 + x + 0.5) / (total_size[1])
            # so the old u should be mapped to
            #   { loc[1] + 1 + u*map_.shape[1]-0.5 + 0.5 } / (total_size[1])
            #  = { loc[1] + 1 + u*map_.shape[1] } / (total_size[1])

            # Also, an index of y (where y is in
            # the range 1 .. map_.shape[0]-2) in one of the input maps
            # was hit by a v of 1 - (y+0.5)/(map_.shape[0]).
            # That y is located at the index loc[0] + 1 + y in the single_map,
            # (where the 1 is for the border)
            # and to hit that we need v to equal 1 - (loc[0] + 1 + y + 0.5) / (total_size[0])
            # so the old v should be mapped to
            #   1 - { loc[0] + 1 + (1-v)*map_.shape[0]-0.5 + 0.5 } / (total_size[0])
            #  = { total_size[0] - loc[0] -1 - (1-v)*map_.shape[0]  }
            #         / (total_size[0])
            #  = { total_size[0] - loc[0] - map.shape[0] - 1 + v*map_.shape[0] }
            #         / (total_size[0])

            # We change the y's in new_uvs for the scaling of height,
            # and the x's for the scaling of width.
            # That is why the 1's and 0's are mismatched in these lines.
            one_if_align = 1 if self.align_corners else 0
            one_if_not_align = 1 - one_if_align
            denom_x = merging_plan.total_size[0] - one_if_align
            scale_x = x_shape - one_if_align
            denom_y = merging_plan.total_size[1] - one_if_align
            scale_y = y_shape - one_if_align
            new_uvs[:, 1] *= scale_x / denom_x
            new_uvs[:, 1] += (
                merging_plan.total_size[0] - x_shape - loc.x - one_if_not_align
            ) / denom_x
            new_uvs[:, 0] *= scale_y / denom_y
            new_uvs[:, 0] += (loc.y + one_if_not_align) / denom_y

            verts_uvs_merged.append(new_uvs)

        faces_uvs_merged = []
        offset = 0
        for faces_uvs_, verts_uvs_ in zip(self.faces_uvs_list(), verts_uvs):
            faces_uvs_merged.append(offset + faces_uvs_)
            offset += verts_uvs_.shape[0]

        return self.__class__(
            maps=[single_map],
            verts_uvs=[torch.cat(verts_uvs_merged)],
            faces_uvs=[torch.cat(faces_uvs_merged)],
            align_corners=self.align_corners,
            padding_mode=self.padding_mode,
            sampling_mode=self.sampling_mode,
        )

    def centers_for_image(self, index: int) -> torch.Tensor:
        """
        Return the locations in the texture map which correspond to the given
        verts_uvs, for one of the meshes. This is potentially useful for
        visualizing the data. See the texturesuv_image_matplotlib and
        texturesuv_image_PIL functions.

        Args:
            index: batch index of the mesh whose centers to return.

        Returns:
            centers: coordinates of points in the texture image
                - a FloatTensor of shape (V,2)
        """
        if self._N != 1:
            raise ValueError(
                "This function only supports plotting textures for one mesh."
            )
        texture_image = self.maps_padded()
        verts_uvs = self.verts_uvs_list()[index][None]
        _, H, W, _3 = texture_image.shape
        coord1 = torch.arange(W).expand(H, W)
        coord2 = torch.arange(H)[:, None].expand(H, W)
        coords = torch.stack([coord1, coord2])[None]
        with torch.no_grad():
            # Get xy cartesian coordinates based on the uv coordinates
            centers = F.grid_sample(
                torch.flip(coords.to(texture_image), [2]),
                # Convert from [0, 1] -> [-1, 1] range expected by grid sample
                verts_uvs[:, None] * 2.0 - 1,
                mode=self.sampling_mode,
                align_corners=self.align_corners,
                padding_mode=self.padding_mode,
            ).cpu()
            centers = centers[0, :, 0].T
        return centers

    def check_shapes(
        self, batch_size: int, max_num_verts: int, max_num_faces: int
    ) -> bool:
        """
        Check if the dimensions of the verts/faces uvs match that of the mesh
        """
        # (N, F) should be the same
        # (N, V) is not guaranteed to be the same
        return (self.faces_uvs_padded().shape[0:2] == (batch_size, max_num_faces)) and (
            self.verts_uvs_padded().shape[0] == batch_size
        )

    def submeshes(
        self,
        vertex_ids_list: List[List[torch.LongTensor]],
        faces_ids_list: List[List[torch.LongTensor]],
    ) -> "TexturesUV":
        """
        Extract a sub-texture for use in a submesh.

        If the meshes batch corresponding to this  TexturesUV contains
        `n = len(faces_ids_list)` meshes, then self.faces_uvs_padded()
        will be of length n. After submeshing, we obtain a batch of
        `k = sum(len(f) for f in faces_ids_list` submeshes (see Meshes.submeshes). This
        function creates a corresponding  TexturesUV object with `faces_uvs_padded`
        of length `k`.

        Args:
            vertex_ids_list: Not used when submeshing TexturesUV.

            face_ids_list: A list of length equal to self.faces_uvs_padded. Each
                element is a LongTensor listing the face ids that the submesh keeps in
                each respective mesh.


        Returns:
            A  "TexturesUV in which faces_uvs_padded, verts_uvs_padded, and maps_padded
            have length sum(len(faces) for faces in faces_ids_list)
        """

        if len(faces_ids_list) != len(self.faces_uvs_padded()):
            raise IndexError(
                "faces_uvs_padded must be of " "the same length as face_ids_list."
            )

        sub_faces_uvs, sub_verts_uvs, sub_maps = [], [], []
        for faces_ids, faces_uvs, verts_uvs, map_ in zip(
            faces_ids_list,
            self.faces_uvs_padded(),
            self.verts_uvs_padded(),
            self.maps_padded(),
        ):
            for faces_ids_submesh in faces_ids:
                sub_faces_uvs.append(faces_uvs[faces_ids_submesh])
                sub_verts_uvs.append(verts_uvs)
                sub_maps.append(map_)

        return self.__class__(
            sub_maps,
            sub_faces_uvs,
            sub_verts_uvs,
            self.padding_mode,
            self.align_corners,
            self.sampling_mode,
        )


class TexturesVertex(TexturesBase):
    def __init__(
        self,
        verts_features: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor]],
    ) -> None:
        """
        Batched texture representation where each vertex in a mesh
        has a C dimensional feature vector.

        Args:
            verts_features: list of (Vi, C) or (N, V, C) tensor giving a feature
                vector with arbitrary dimensions for each vertex.
        """
        if isinstance(verts_features, (tuple, list)):
            correct_shape = all(
                (torch.is_tensor(v) and v.ndim == 2) for v in verts_features
            )
            if not correct_shape:
                raise ValueError(
                    "Expected verts_features to be a list of tensors of shape (V, C)."
                )

            self._verts_features_list = verts_features
            self._verts_features_padded = None
            self.device = torch.device("cpu")

            # These values may be overridden when textures is
            # passed into the Meshes constructor. For more details
            # refer to the __init__ of Meshes.
            self._N = len(verts_features)
            self._num_verts_per_mesh = [len(fv) for fv in verts_features]

            if self._N > 0:
                self.device = verts_features[0].device

        elif torch.is_tensor(verts_features):
            if verts_features.ndim != 3:
                msg = "Expected verts_features to be of shape (N, V, C); got %r"
                raise ValueError(msg % repr(verts_features.shape))
            self._verts_features_padded = verts_features
            self._verts_features_list = None
            self.device = verts_features.device

            # These values may be overridden when textures is
            # passed into the Meshes constructor. For more details
            # refer to the __init__ of Meshes.
            self._N = len(verts_features)
            max_F = verts_features.shape[1]
            self._num_verts_per_mesh = [max_F] * self._N
        else:
            raise ValueError("verts_features must be a tensor or list of tensors")

        # This is set inside the Meshes object when textures is
        # passed into the Meshes constructor. For more details
        # refer to the __init__ of Meshes.
        self.valid = torch.ones((self._N,), dtype=torch.bool, device=self.device)

    def clone(self) -> "TexturesVertex":
        tex = self.__class__(self.verts_features_padded().clone())
        if self._verts_features_list is not None:
            tex._verts_features_list = [f.clone() for f in self._verts_features_list]
        tex._num_verts_per_mesh = self._num_verts_per_mesh.copy()
        tex.valid = self.valid.clone()
        return tex

    def detach(self) -> "TexturesVertex":
        tex = self.__class__(self.verts_features_padded().detach())
        if self._verts_features_list is not None:
            tex._verts_features_list = [f.detach() for f in self._verts_features_list]
        tex._num_verts_per_mesh = self._num_verts_per_mesh.copy()
        tex.valid = self.valid.detach()
        return tex

    def __getitem__(self, index) -> "TexturesVertex":
        props = ["verts_features_list", "_num_verts_per_mesh"]
        new_props = self._getitem(index, props)
        verts_features = new_props["verts_features_list"]
        if isinstance(verts_features, list):
            # Handle the case of an empty list
            if len(verts_features) == 0:
                verts_features = torch.empty(
                    size=(0, 0, 3),
                    dtype=torch.float32,
                    device=self.verts_features_padded().device,
                )
            new_tex = self.__class__(verts_features=verts_features)
        elif torch.is_tensor(verts_features):
            new_tex = self.__class__(verts_features=[verts_features])
        else:
            raise ValueError("Not all values are provided in the correct format")
        new_tex._num_verts_per_mesh = new_props["_num_verts_per_mesh"]
        return new_tex

    def verts_features_padded(self) -> torch.Tensor:
        if self._verts_features_padded is None:
            if self.isempty():
                self._verts_features_padded = torch.zeros(
                    (self._N, 0, 3, 0), dtype=torch.float32, device=self.device
                )
            else:
                self._verts_features_padded = list_to_padded(
                    self._verts_features_list, pad_value=0.0
                )
        return self._verts_features_padded

    def verts_features_list(self) -> List[torch.Tensor]:
        if self._verts_features_list is None:
            if self.isempty():
                self._verts_features_list = [
                    torch.empty((0, 3), dtype=torch.float32, device=self.device)
                ] * self._N
            else:
                self._verts_features_list = padded_to_list(
                    self._verts_features_padded, split_size=self._num_verts_per_mesh
                )
        return self._verts_features_list

    def verts_features_packed(self) -> torch.Tensor:
        if self.isempty():
            return torch.zeros((self._N, 3, 0), dtype=torch.float32, device=self.device)
        verts_features_list = self.verts_features_list()
        return list_to_packed(verts_features_list)[0]

    def extend(self, N: int) -> "TexturesVertex":
        new_props = self._extend(N, ["verts_features_padded", "_num_verts_per_mesh"])
        new_tex = self.__class__(verts_features=new_props["verts_features_padded"])
        new_tex._num_verts_per_mesh = new_props["_num_verts_per_mesh"]
        return new_tex

    # pyre-fixme[14]: `sample_textures` overrides method defined in `TexturesBase`
    #  inconsistently.
    def sample_textures(self, fragments, faces_packed=None) -> torch.Tensor:
        """
        Determine the color for each rasterized face. Interpolate the colors for
        vertices which form the face using the barycentric coordinates.
        Args:
            fragments:
                The outputs of rasterization. From this we use

                - pix_to_face: LongTensor of shape (N, H, W, K) specifying the indices
                of the faces (in the packed representation) which
                overlap each pixel in the image.
                - barycentric_coords: FloatTensor of shape (N, H, W, K, 3) specifying
                the barycentric coordinates of each pixel
                relative to the faces (in the packed
                representation) which overlap the pixel.

        Returns:
            texels: An texture per pixel of shape (N, H, W, K, C).
            There will be one C dimensional value for each element in
            fragments.pix_to_face.
        """
        verts_features_packed = self.verts_features_packed()
        faces_verts_features = verts_features_packed[faces_packed]

        texels = interpolate_face_attributes(
            fragments.pix_to_face, fragments.bary_coords, faces_verts_features
        )
        return texels

    def submeshes(
        self,
        vertex_ids_list: List[List[torch.LongTensor]],
        faces_ids_list: List[List[torch.LongTensor]],
    ) -> "TexturesVertex":
        """
        Extract a sub-texture for use in a submesh.

        If the meshes batch corresponding to this TexturesVertex contains
        `n = len(vertex_ids_list)` meshes, then self.verts_features_list()
        will be of length n. After submeshing, we obtain a batch of
        `k = sum(len(v) for v in vertex_ids_list` submeshes (see Meshes.submeshes). This
        function creates a corresponding TexturesVertex object with `verts_features_list`
        of length `k`.

        Args:
            vertex_ids_list: A list of length equal to self.verts_features_list. Each
                element is a LongTensor listing the vertices that the submesh keeps in
                each respective mesh.

            face_ids_list: Not used when submeshing TexturesVertex.

        Returns:
            A TexturesVertex in which verts_features_list has length
            sum(len(vertices) for vertices in vertex_ids_list). Each element contains
            vertex features corresponding to the subset of vertices in that submesh.
        """
        if len(vertex_ids_list) != len(self.verts_features_list()):
            raise IndexError(
                "verts_features_list must be of " "the same length as vertex_ids_list."
            )

        sub_features = []
        for vertex_ids, features in zip(vertex_ids_list, self.verts_features_list()):
            for vertex_ids_submesh in vertex_ids:
                sub_features.append(features[vertex_ids_submesh])

        return self.__class__(sub_features)

    def faces_verts_textures_packed(self, faces_packed=None) -> torch.Tensor:
        """
        Samples texture from each vertex and for each face in the mesh.
        For N meshes with {Fi} number of faces, it returns a
        tensor of shape sum(Fi)x3xC (C = 3 for RGB).
        You can use the utils function in structures.utils to convert the
        packed representation to a list or padded.
        """
        verts_features_packed = self.verts_features_packed()
        faces_verts_features = verts_features_packed[faces_packed]
        return faces_verts_features

    def join_batch(self, textures: List["TexturesVertex"]) -> "TexturesVertex":
        """
        Join the list of textures given by `textures` to
        self to create a batch of textures. Return a new
        TexturesVertex object with the combined textures.

        Args:
            textures: List of TexturesVertex objects

        Returns:
            new_tex: TexturesVertex object with the combined
            textures from self and the list `textures`.
        """
        tex_types_same = all(isinstance(tex, TexturesVertex) for tex in textures)
        if not tex_types_same:
            raise ValueError("All textures must be of type TexturesVertex.")

        verts_features_list = []
        verts_features_list += self.verts_features_list()
        num_verts_per_mesh = self._num_verts_per_mesh.copy()
        for tex in textures:
            verts_features_list += tex.verts_features_list()
            num_verts_per_mesh += tex._num_verts_per_mesh

        new_tex = self.__class__(verts_features=verts_features_list)
        new_tex._num_verts_per_mesh = num_verts_per_mesh
        return new_tex

    def join_scene(self) -> "TexturesVertex":
        """
        Return a new TexturesVertex amalgamating the batch.
        """
        return self.__class__(verts_features=[torch.cat(self.verts_features_list())])

    def check_shapes(
        self, batch_size: int, max_num_verts: int, max_num_faces: int
    ) -> bool:
        """
        Check if the dimensions of the verts features match that of the mesh verts
        """
        # (N, V) should be the same
        return self.verts_features_padded().shape[:-1] == (batch_size, max_num_verts)
