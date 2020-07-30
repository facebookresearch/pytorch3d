# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import itertools
import warnings
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from pytorch3d.ops import interpolate_face_attributes
from pytorch3d.structures.utils import list_to_packed, list_to_padded, padded_to_list
from torch.nn.functional import interpolate


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
    # pyre-fixme[16]: `Tensor` has no attribute `ndim`.
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
        x_reshaped.append(y.reshape(-1, D))
    x_padded = list_to_padded(x_reshaped, pad_size=pad_size, pad_value=pad_value)
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
    x_reshaped = x.reshape(N, M, D)
    x_list = padded_to_list(x_reshaped, split_size=split_size)
    x_list = [xl.reshape((xl.shape[0],) + reshape_dims) for xl in x_list]
    return x_list


def _pad_texture_maps(
    images: Union[Tuple[torch.Tensor], List[torch.Tensor]]
) -> torch.Tensor:
    """
    Pad all texture images so they have the same height and width.

    Args:
        images: list of N tensors of shape (H, W, 3)

    Returns:
        tex_maps: Tensor of shape (N, max_H, max_W, 3)
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

    for i, image in enumerate(tex_maps):
        if image.shape[:2] != max_shape:
            image_BCHW = image.permute(2, 0, 1)[None]
            new_image_BCHW = interpolate(
                image_BCHW, size=max_shape, mode="bilinear", align_corners=False
            )
            tex_maps[i] = new_image_BCHW[0].permute(1, 2, 0)
    tex_maps = torch.stack(tex_maps, dim=0)  # (num_tex_maps, max_H, max_W, 3)
    return tex_maps


# A base class for defining a batch of textures
# with helper methods.
# This is also useful to have so that inside `Meshes`
# we can allow the input textures to be any texture
# type which is an instance of the base class.
class TexturesBase(object):
    def __init__(self):
        self._N = 0
        self.valid = None

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
                # pyre-fixme[16]: `Tensor` has no attribute `nonzero`.
                index = index.nonzero()
                index = index.squeeze(1) if index.numel() > 0 else index
                index = index.tolist()
            for p in props:
                t = getattr(self, p)
                if callable(t):
                    t = t()  # class method
                new_props[p] = [t[i] for i in index]

        return new_props

    def sample_textures(self):
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

    def clone(self):
        """
        Each texture class should implement a method
        to clone all necessary internal tensors.
        """
        raise NotImplementedError()

    def __getitem__(self, index):
        """
        Each texture class should implement a method
        to get the texture properites for the
        specified elements in the batch.
        The TexturesBase._getitem(i) method
        can be used as a helper funtion to retrieve the
        class attributes for item i. Then, a new
        instance of the child class can be created with
        the attributes.
        """
        raise NotImplementedError()

    def __repr__(self):
        return "TexturesBase"


def Textures(
    maps: Union[List, torch.Tensor, None] = None,
    faces_uvs: Optional[torch.Tensor] = None,
    verts_uvs: Optional[torch.Tensor] = None,
    verts_rgb: Optional[torch.Tensor] = None,
) -> TexturesBase:
    """
        Textures class has been DEPRECATED.
        Preserving Textures as a function for backwards compatibility.

        Args:
            maps: texture map per mesh. This can either be a list of maps
              [(H, W, 3)] or a padded tensor of shape (N, H, W, 3).
            faces_uvs: (N, F, 3) tensor giving the index into verts_uvs for each
                vertex in the face. Padding value is assumed to be -1.
            verts_uvs: (N, V, 2) tensor giving the uv coordinate per vertex.
            verts_rgb: (N, V, 3) tensor giving the rgb color per vertex. Padding
                value is assumed to be -1.


        Returns:
            a Textures class which is an instance of TexturesBase e.g. TexturesUV,
            TexturesAtlas, TexturesVerte

        """

    warnings.warn(
        """Textures class is deprecated,
        use TexturesUV, TexturesAtlas, TexturesVertex instead.
        Textures class will be removed in future releases.""",
        PendingDeprecationWarning,
    )

    if all(x is not None for x in [faces_uvs, verts_uvs, maps]):
        # pyre-fixme[6]: Expected `Union[List[torch.Tensor], torch.Tensor]` for 1st
        #  param but got `Union[None, List[typing.Any], torch.Tensor]`.
        return TexturesUV(maps=maps, faces_uvs=faces_uvs, verts_uvs=verts_uvs)
    elif verts_rgb is not None:
        return TexturesVertex(verts_features=verts_rgb)
    else:
        raise ValueError(
            "Textures either requires all three of (faces uvs, verts uvs, maps) or verts rgb"
        )


class TexturesAtlas(TexturesBase):
    def __init__(self, atlas: Union[torch.Tensor, List, None]):
        """
        A texture representation where each face has a square texture map.
        This is based on the implementation from SoftRasterizer [1].

        Args:
            atlas: (N, F, R, R, D) tensor giving the per face texture map.
                The atlas can be created during obj loading with the
                pytorch3d.io.load_obj function - in the input arguments
                set `create_texture_atlas=True`. The atlas will be
                returned in aux.texture_atlas.


        The padded and list representations of the textures are stored
        and the packed representations is computed on the fly and
        not cached.

        [1] Liu et al, 'Soft Rasterizer: A Differentiable Renderer for Image-based
            3D Reasoning', ICCV 2019
        """
        if isinstance(atlas, (list, tuple)):
            correct_format = all(
                (
                    torch.is_tensor(elem)
                    and elem.ndim == 4
                    and elem.shape[1] == elem.shape[2]
                )
                for elem in atlas
            )
            if not correct_format:
                msg = "Expected atlas to be a list of tensors of shape (F, R, R, D)"
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
            # pyre-fixme[16]: `Optional` has no attribute `ndim`.
            if atlas.ndim != 5:
                msg = "Expected atlas to be of shape (N, F, R, R, D); got %r"
                raise ValueError(msg % repr(atlas.ndim))
            self._atlas_padded = atlas
            self._atlas_list = None
            # pyre-fixme[16]: `Optional` has no attribute `device`.
            self.device = atlas.device

            # These values may be overridden when textures is
            # passed into the Meshes constructor. For more details
            # refer to the __init__ of Meshes.
            # pyre-fixme[6]: Expected `Sized` for 1st param but got
            #  `Optional[torch.Tensor]`.
            self._N = len(atlas)
            # pyre-fixme[16]: `Optional` has no attribute `shape`.
            max_F = atlas.shape[1]
            self._num_faces_per_mesh = [max_F] * self._N
        else:
            raise ValueError("Expected atlas to be a tensor or list")

        # The num_faces_per_mesh, N and valid
        # are reset inside the Meshes object when textures is
        # passed into the Meshes constructor. For more details
        # refer to the __init__ of Meshes.
        self.valid = torch.ones((self._N,), dtype=torch.bool, device=self.device)

    # This is a hack to allow the child classes to also have the same representation
    # as the parent. In meshes.py we check that the input textures have the correct
    # type. However due to circular imports issues, we can't import the texture
    # classes into any files in pytorch3d.structures. Instead we check
    # for repr(textures) == "TexturesBase".
    def __repr__(self):
        return super().__repr__()

    def clone(self):
        tex = self.__class__(atlas=self.atlas_padded().clone())
        num_faces = (
            self._num_faces_per_mesh.clone()
            if torch.is_tensor(self._num_faces_per_mesh)
            else self._num_faces_per_mesh
        )
        tex.valid = self.valid.clone()
        tex._num_faces_per_mesh = num_faces
        return tex

    def __getitem__(self, index):
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
        new_tex = TexturesAtlas(atlas=new_props["atlas_padded"])
        new_tex._num_faces_per_mesh = new_props["_num_faces_per_mesh"]
        return new_tex

    def sample_textures(self, fragments, **kwargs) -> torch.Tensor:
        """
        Args:
            fragments:
                The outputs of rasterization. From this we use

                - pix_to_face: LongTensor of shape (N, H, W, K) specifying the indices
                of the faces (in the packed representation) which
                overlap each pixel in the image.
                - barycentric_coords: FloatTensor of shape (N, H, W, K, 3) specifying
                the barycentric coordianates of each pixel
                relative to the faces (in the packed
                representation) which overlap the pixel.

        Returns:
            texels: (N, H, W, K, 3)
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
        w_xy = (bary_w01 * R).to(torch.int64)  # (N, H, W, K, 2)

        below_diag = (
            bary_w01.sum(dim=-1) * R - w_xy.float().sum(dim=-1)
        ) <= 1.0  # (N, H, W, K)
        w_x, w_y = w_xy.unbind(-1)
        w_x = torch.where(below_diag, w_x, (R - 1 - w_x))
        w_y = torch.where(below_diag, w_y, (R - 1 - w_y))

        texels = atlas_packed[pix_to_face, w_y, w_x]
        texels = texels * (pix_to_face >= 0)[..., None].float()

        return texels

    def join_batch(self, textures: List["TexturesAtlas"]) -> "TexturesAtlas":
        """
        Join the list of textures given by `textures` to
        self to create a batch of textures. Return a new
        TexturesAtlas object with the combined textures.

        Args:
            textures: List of TextureAtlas objects

        Returns:
            new_tex: TextureAtlas object with the combined
            textures from self and the list `textures`.
        """
        tex_types_same = all(isinstance(tex, TexturesAtlas) for tex in textures)
        if not tex_types_same:
            raise ValueError("All textures must be of type TexturesAtlas.")

        atlas_list = []
        atlas_list += self.atlas_list()
        num_faces_per_mesh = self._num_faces_per_mesh
        for tex in textures:
            atlas_list += tex.atlas_list()
            num_faces_per_mesh += tex._num_faces_per_mesh
        new_tex = self.__class__(atlas=atlas_list)
        new_tex._num_faces_per_mesh = num_faces_per_mesh
        return new_tex


class TexturesUV(TexturesBase):
    def __init__(
        self,
        maps: Union[torch.Tensor, List[torch.Tensor]],
        faces_uvs: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor]],
        verts_uvs: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor]],
    ):
        """
        Textures are represented as a per mesh texture map and uv coordinates for each
        vertex in each face. NOTE: this class only supports one texture map per mesh.

        Args:
            maps: texture map per mesh. This can either be a list of maps
              [(H, W, 3)] or a padded tensor of shape (N, H, W, 3)
            faces_uvs: (N, F, 3) tensor giving the index into verts_uvs for each face
            verts_uvs: (N, V, 2) tensor giving the uv coordinates per vertex

        Note: only the padded and list representation of the textures are stored
        and the packed representations is computed on the fly and
        not cached.
        """
        super().__init__()
        if isinstance(faces_uvs, (list, tuple)):
            for fv in faces_uvs:
                # pyre-fixme[16]: `Tensor` has no attribute `ndim`.
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
                import pdb

                pdb.set_trace()
                raise ValueError("verts_uvs and faces_uvs must be on the same device")

            # These values may be overridden when textures is
            # passed into the Meshes constructor. For more details
            # refer to the __init__ of Meshes.
            self._num_verts_per_mesh = [len(v) for v in verts_uvs]

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

            # These values may be overridden when textures is
            # passed into the Meshes constructor.
            max_V = verts_uvs.shape[1]
            self._num_verts_per_mesh = [max_V] * self._N
        else:
            raise ValueError("Expected verts_uvs to be a tensor or list")

        if torch.is_tensor(maps):
            # pyre-fixme[16]: `List` has no attribute `ndim`.
            # pyre-fixme[16]: `List` has no attribute `shape`.
            if maps.ndim != 4 or maps.shape[0] != self._N:
                msg = "Expected maps to be of shape (N, H, W, 3); got %r"
                raise ValueError(msg % repr(maps.shape))
            self._maps_padded = maps
            self._maps_list = None
        elif isinstance(maps, (list, tuple)):
            if len(maps) != self._N:
                raise ValueError("Expected one texture map per mesh in the batch.")
            self._maps_list = maps
            if self._N > 0:
                maps = _pad_texture_maps(maps)
            else:
                maps = torch.empty(
                    (self._N, 0, 0, 3), dtype=torch.float32, device=self.device
                )
            self._maps_padded = maps
        else:
            raise ValueError("Expected maps to be a tensor or list.")

        if self._maps_padded.device != self.device:
            raise ValueError("maps must be on the same device as verts/faces uvs.")

        self.valid = torch.ones((self._N,), dtype=torch.bool, device=self.device)

    def __repr__(self):
        return super().__repr__()

    def clone(self):
        tex = self.__class__(
            self.maps_padded().clone(),
            self.faces_uvs_padded().clone(),
            self.verts_uvs_padded().clone(),
        )
        num_faces = (
            self._num_faces_per_mesh.clone()
            if torch.is_tensor(self._num_faces_per_mesh)
            else self._num_faces_per_mesh
        )
        tex._num_faces_per_mesh = num_faces
        tex.valid = self.valid.clone()
        return tex

    def __getitem__(self, index):
        props = ["verts_uvs_list", "faces_uvs_list", "maps_list", "_num_faces_per_mesh"]
        new_props = self._getitem(index, props)
        faces_uvs = new_props["faces_uvs_list"]
        verts_uvs = new_props["verts_uvs_list"]
        maps = new_props["maps_list"]

        # if index has multiple values then faces/verts/maps may be a list of tensors
        if all(isinstance(f, (list, tuple)) for f in [faces_uvs, verts_uvs, maps]):
            new_tex = self.__class__(
                faces_uvs=faces_uvs, verts_uvs=verts_uvs, maps=maps
            )
        elif all(torch.is_tensor(f) for f in [faces_uvs, verts_uvs, maps]):
            new_tex = self.__class__(
                faces_uvs=[faces_uvs], verts_uvs=[verts_uvs], maps=[maps]
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

    def faces_uvs_packed(self) -> torch.Tensor:
        if self.isempty():
            return torch.zeros((self._N, 3), dtype=torch.float32, device=self.device)
        faces_uvs_list = self.faces_uvs_list()
        return list_to_packed(faces_uvs_list)[0]

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
                self._verts_uvs_list = padded_to_list(
                    self._verts_uvs_padded, split_size=self._num_verts_per_mesh
                )
        return self._verts_uvs_list

    def verts_uvs_packed(self) -> torch.Tensor:
        if self.isempty():
            return torch.zeros((self._N, 2), dtype=torch.float32, device=self.device)
        verts_uvs_list = self.verts_uvs_list()
        return list_to_packed(verts_uvs_list)[0]

    # Currently only the padded maps are used.
    def maps_padded(self) -> torch.Tensor:
        return self._maps_padded

    def maps_list(self) -> torch.Tensor:
        # maps_list is not used anywhere currently - maps
        # are padded to ensure the (H, W) of all maps is the
        # same across the batch and we don't store the
        # unpadded sizes of the maps. Therefore just return
        # the unbinded padded tensor.
        return self._maps_padded.unbind(0)

    def extend(self, N: int) -> "TexturesUV":
        new_props = self._extend(
            N,
            [
                "maps_padded",
                "verts_uvs_padded",
                "faces_uvs_padded",
                "_num_faces_per_mesh",
                "_num_verts_per_mesh",
            ],
        )
        new_tex = TexturesUV(
            maps=new_props["maps_padded"],
            faces_uvs=new_props["faces_uvs_padded"],
            verts_uvs=new_props["verts_uvs_padded"],
        )
        new_tex._num_faces_per_mesh = new_props["_num_faces_per_mesh"]
        new_tex._num_verts_per_mesh = new_props["_num_verts_per_mesh"]
        return new_tex

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
                the barycentric coordianates of each pixel
                relative to the faces (in the packed
                representation) which overlap the pixel.

        Returns:
            texels: tensor of shape (N, H, W, K, C) giving the interpolated
            texture for each pixel in the rasterized image.
        """
        verts_uvs = self.verts_uvs_packed()
        faces_uvs = self.faces_uvs_packed()
        faces_verts_uvs = verts_uvs[faces_uvs]
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

        pixel_uvs = pixel_uvs * 2.0 - 1.0
        texture_maps = torch.flip(texture_maps, [2])  # flip y axis of the texture map
        if texture_maps.device != pixel_uvs.device:
            texture_maps = texture_maps.to(pixel_uvs.device)
        texels = F.grid_sample(texture_maps, pixel_uvs, align_corners=False)
        texels = texels.reshape(N, K, C, H_out, W_out).permute(0, 3, 4, 1, 2)
        return texels

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

        verts_uvs_list = []
        faces_uvs_list = []
        maps_list = []
        faces_uvs_list += self.faces_uvs_list()
        verts_uvs_list += self.verts_uvs_list()
        maps_list += list(self.maps_padded().unbind(0))
        num_faces_per_mesh = self._num_faces_per_mesh
        for tex in textures:
            verts_uvs_list += tex.verts_uvs_list()
            faces_uvs_list += tex.faces_uvs_list()
            num_faces_per_mesh += tex._num_faces_per_mesh
            tex_map_list = list(tex.maps_padded().unbind(0))
            maps_list += tex_map_list

        new_tex = self.__class__(
            maps=maps_list, verts_uvs=verts_uvs_list, faces_uvs=faces_uvs_list
        )
        new_tex._num_faces_per_mesh = num_faces_per_mesh
        return new_tex


class TexturesVertex(TexturesBase):
    def __init__(
        self,
        verts_features: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor]],
    ):
        """
        Batched texture representation where each vertex in a mesh
        has a D dimensional feature vector.

        Args:
            verts_features: (N, V, D) tensor giving a feature vector with
                artbitrary dimensions for each vertex.
        """
        if isinstance(verts_features, (tuple, list)):
            correct_shape = all(
                # pyre-fixme[16]: `Tensor` has no attribute `ndim`.
                (torch.is_tensor(v) and v.ndim == 2)
                for v in verts_features
            )
            if not correct_shape:
                raise ValueError(
                    "Expected verts_features to be a list of tensors of shape (V, D)."
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
                msg = "Expected verts_features to be of shape (N, V, D); got %r"
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

    def __repr__(self):
        return super().__repr__()

    def clone(self):
        tex = self.__class__(self.verts_features_padded().clone())
        if self._verts_features_list is not None:
            tex._verts_features_list = [f.clone() for f in self._verts_features_list]
        num_faces = (
            self._num_verts_per_mesh.clone()
            if torch.is_tensor(self._num_verts_per_mesh)
            else self._num_verts_per_mesh
        )
        tex._num_verts_per_mesh = num_faces
        tex.valid = self.valid.clone()
        return tex

    def __getitem__(self, index):
        props = ["verts_features_list", "_num_verts_per_mesh"]
        new_props = self._getitem(index, props)
        verts_features = new_props["verts_features_list"]
        if isinstance(verts_features, list):
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
                    torch.empty((0, 3, 0), dtype=torch.float32, device=self.device)
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
        new_tex = TexturesVertex(verts_features=new_props["verts_features_padded"])
        new_tex._num_verts_per_mesh = new_props["_num_verts_per_mesh"]
        return new_tex

    def sample_textures(self, fragments, faces_packed=None) -> torch.Tensor:
        """
        Detemine the color for each rasterized face. Interpolate the colors for
        vertices which form the face using the barycentric coordinates.
        Args:
            fragments:
                The outputs of rasterization. From this we use

                - pix_to_face: LongTensor of shape (N, H, W, K) specifying the indices
                of the faces (in the packed representation) which
                overlap each pixel in the image.
                - barycentric_coords: FloatTensor of shape (N, H, W, K, 3) specifying
                the barycentric coordianates of each pixel
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
        num_faces_per_mesh = self._num_verts_per_mesh
        for tex in textures:
            verts_features_list += tex.verts_features_list()
            num_faces_per_mesh += tex._num_verts_per_mesh

        new_tex = self.__class__(verts_features=verts_features_list)
        new_tex._num_verts_per_mesh = num_faces_per_mesh
        return new_tex
