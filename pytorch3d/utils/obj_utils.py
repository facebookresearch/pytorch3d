# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pathlib
from typing import Optional, Union, Dict, List, Tuple
import torch

from ..common.datatypes import Device
from pytorch3d.structures import Meshes
from pytorch3d.renderer.mesh.textures import TexturesAtlas, TexturesUV

PathOrStr = Union[pathlib.Path, str]


def parse_obj_to_mesh_by_texture(
    verts: torch.Tensor,
    faces: torch.Tensor,
    verts_uvs: torch.Tensor,
    faces_uvs: torch.Tensor,
    device: Device,
    materials_idx: Optional[torch.Tensor],
    texture_images: Optional[Dict[str, torch.tensor]] = None,
    normals: Optional[torch.Tensor] = None,
    faces_normals_idx: Optional[torch.Tensor] = None,
    texture_atlas: Optional[torch.Tensor] = None,
    use_texture_atlas: Optional[bool] = False,
) -> List[Meshes]:
    """A utility function to parse an obj to a list of meshes object. Support
    for multiple textures is provided by creating mini-meshes by texture as
    a list of meshes that can be joined as a scene or batch. Parsing normals
    not currently supported.

    - Example Usage:
        ::
        from pytorch3d.utils import parse_obj_to_mesh_by_texture
        from pytorch3d.structures import join_meshes_as_batch, join_meshes_as_scene

        meshes_list = parse_obj_to_mesh_by_texture(
            verts=verts,
            faces=faces.verts_idx,
            verts_uvs=aux.verts_uvs,
            faces_uvs=faces.textures_idx,
            texture_images=aux.textu re_images,
            device=verts.device,
            texture_atlas=aux.texture_atlas,
            materials_idx=faces.materials_idx,
        )

        # return as a scene
        meshes_scene = join_meshes_as_scene(meshes_list)
        # return as a batch
        meshes_batch = join_meshes_as_batch(meshes_list)

    Args:
        verts: FloatTensor of shape (V, 3) giving vertex coordinates.
        faces: LongTensor of shape (F, 3) giving faces.
        verts_uvs: FloatTensor of shape (V, 2) giving the uv coordinate per vertex.
        faces_uvs: LongTensor of shape (F, 3) giving the index into verts_uvs for
            each vertex in the face.
        texture_images: Dictionary of str:FloatTensor of shape (H, W, 3) where
            where each key value pair, in order, represnts a material name and
            texture map; in objs, this value is often the aux.texture_images object.
            Each output mesh will use the textures_images as textures input unlesss
            texture_atlas is provided and use_texture_atlas is True.
        device: Device (as str or torch.device) on which to return the new tensors.
        materials_idx: IntTensor of shape (F, ) giving the material index that links
            each face in faces to a texture in texture_images. If loading multiple
            textures and providing a texture_images object, materials_idx must be
            provided. This value is often the aux.materials_idx value in an obj.
        normals: FloatTensor of shape (V, 3) giving normals for faces_normals_idx
            to index into.
        faces_normals_idx: LongTensor of shape (F, 3) giving the index into
            normals for each vertex in the face.
        texture_atlas: FloatTensor representing the RxR texture map for each face.
            This value must be provided if textures_images is not provided.
        use_texture_atlas: Default to False. If True and valid texture_atlas is provided,
            the obj's texture atlast is used as the input data for Meshes textures.

    Returns:
        - A List of Meshes where N meshes is equal to the number of input textures in the obj.
    """
    _validate_obj(
        verts=verts,
        faces=faces,
        faces_uvs=faces_uvs,
        verts_uvs=verts_uvs,
        texture_images=texture_images,
        materials_idx=materials_idx,
        normals=normals,
        faces_normals_idx=faces_normals_idx,
    )

    mesh = []

    for tex_mtl_idx, tex_mtl_name in enumerate(texture_images.keys()):
        # parse faces/verts from each texture in the obj into a single mesh
        faces_to_subset = materials_idx == tex_mtl_idx
        faces_to_subset = faces_to_subset.nonzero().squeeze().ravel()

        # skip any materials that are not referenced in the current mask
        if faces_to_subset.numel():
            _verts_idx, _faces = _reindex_verts_faces_by_index(faces, faces_to_subset)

            if verts_uvs is not None:
                # re-index vert uvs and face uvs based on current faces
                _verts_uvs, _faces_uvs = _reindex_verts_faces_uvs_by_index(
                    faces_uvs, faces_to_subset
                )

                # use faces_to_subset to slice either texture atlas or texture images
                if texture_atlas is not None and use_texture_atlas:
                    textures = TexturesAtlas(
                        atlas=[texture_atlas[faces_to_subset].to(device)]
                    )
                else:
                    textures = TexturesUV(
                        verts_uvs=verts_uvs[_verts_uvs][None].to(device),  # (V, 2)
                        faces_uvs=_faces_uvs[None].to(device),  # (F, 3)
                        maps=texture_images[tex_mtl_name][None].to(device),
                    )
            else:
                textures = None

            if normals is not None and faces_normals_idx is not None:
                _normals_idx_orig, _ = _reindex_face_normals_by_index(
                    faces_normals_idx, faces_to_subset
                )
                _normals = normals[_normals_idx_orig]
            else:
                _normals = None

            # create a list of meshes based on each obj's textures
            mesh.append(
                Meshes(
                    verts=[verts[_verts_idx].to(device)],
                    faces=[_faces.type(torch.int64).to(device)],
                    textures=textures,
                    verts_normals=[_normals.to(device)]
                    if _normals is not None
                    else None,
                )
            )

    return mesh


def _reindex_verts_faces_by_index(
    faces: torch.Tensor, faces_to_subset: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """A utility function to re-index verts_idx and corresponding faces
    by an array of faces_to_subset. This function enables subsetting operations
    for an obj by returning the inverse of faces with a given array of face
    indices.

    Args:
        faces: A torch.Tensor defining faces by verts index values, for objs
            this is often the namedTuple of faces.verts_idx or obj[1].verts_idx.
        faces_to_subset: A 1-dimentional tensor that represents the desired
            indices of the faces to keep in the subset.
    Returns:
        A 2-Tuple of:
        - _verts_idx: The unique values of faces as _verts_idx.
        - _faces: The reverse indices of unique faces as faces.
    """

    _verts_idx, _faces = torch.unique(faces[faces_to_subset], return_inverse=True)

    return _verts_idx, _faces


def _reindex_verts_faces_uvs_by_index(
    faces_uvs: torch.Tensor, faces_to_subset: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """A utility function to re-index uvs for verts and faces and
    corresponding to textures_idx by an array of face indices. This
    function enables subsetting operations for an obj by returning
    the inverse of faces with a given array of faces_to_subset.

    Args:
        faces_uvs: A torch.Tensor defining faces by verts index values;
            for objs, this is often the namedTuple of faces.textures_idx
            or obj[1].textures_idx.
        faces_to_subset: A 1-dimentional tensor that represents the desired
            indices of the faces to keep in the subset.
    Returns:
        A 2-Tuple of:
        - _verts_uvs: The unique values of faces_uvs as verts_uvs.
        - _faces_uvs: The reverse indices of unique faces_uvs as faces_uvs.
    """

    _verts_uvs, _faces_uvs = torch.unique(
        faces_uvs[faces_to_subset], return_inverse=True
    )

    return _verts_uvs, _faces_uvs


def _reindex_obj_materials_by_index(
    faces_materials: torch.Tensor, faces_to_subset: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """A utility function to re-index materials by faces
    corresponding to materials_idx by an array of face indices. This
    function enables subsetting operations for an obj by returning
    the inverse of faces with a given array of face indices.

    Args:
        faces_materials: A torch.Tensor defining materials by index
            for each face/verts; for objs this is often the namedTuple of
            faces.materials_idx or obj[1].materials_idx.
        faces_to_subset: A 1-dimentional tensor that represents the desired
            indices of the faces to keep in the subset.
    Returns:
        A 2-Tuple of:
        - _materials_idx_unique: The unique values of faces_materials as _materials_idx_unique.
        - _materials_idx: The reverse indices of unique faces_materials as _materials_idx.
    """

    _materials_idx_unique, _materials_idx = torch.unique(
        faces_materials[faces_to_subset], return_inverse=True
    )

    return _materials_idx_unique, _materials_idx


def _reindex_face_normals_by_index(
    normals_idx: torch.Tensor, faces_to_subset: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """A utility function to re-index face normals and by an array of
    face indices. This function enables subsetting operations for
    an obj by returning the inverse of faces with a given array of
    face indices.

    Args:
        normals_idx: A torch.Tensor defining face normals by face index
            values; for objs, this is often the namedTuple of faces.normals_idx
            or obj[1].normals_idx.
        faces_to_subset: A 1-dimentional tensor that represents the desired
            indices of the faces to keep in the subset.
    Returns:
        A 2-Tuple of:
        - normals_unique: The unique values of normals as normals_unique.
        - _normals_idx: The reverse indices of unique normals_idx as _normals_idx.
    """

    normals_unique, _normals_idx = torch.unique(
        normals_idx[faces_to_subset], return_inverse=True
    )

    return normals_unique, _normals_idx


def _validate_obj(
    verts: torch.Tensor,
    faces: torch.Tensor,
    faces_uvs: torch.Tensor = None,
    verts_uvs: torch.Tensor = None,
    texture_map: torch.Tensor = None,
    texture_images: Dict[str, torch.tensor] = None,
    materials_idx: torch.Tensor = None,
    normals: torch.Tensor = None,
    faces_normals_idx: torch.Tensor = None,
):
    """A helper function to validate an obj input."""
    if len(verts) and (verts.dim() != 2 or verts.size(1) != 3):
        message = "'verts' should either be empty or of shape (num_verts, 3)."
        raise ValueError(message)

    if len(faces) and (faces.dim() != 2 or faces.size(1) != 3):
        message = "'faces' should either be empty or of shape (num_faces, 3)."
        raise ValueError(message)

    if None not in [faces_uvs]:
        if faces_uvs.dim() != 2 or faces_uvs.size(1) != 3:
            message = "'faces_uvs' should either be empty or of shape (num_faces, 3)."
            raise ValueError(message)

    if None not in [verts_uvs]:
        if verts_uvs.dim() != 2 or verts_uvs.size(1) != 2:
            message = "'verts_uvs' should either be empty or of shape (num_verts, 2)."
            raise ValueError(message)

    if None not in [texture_map]:
        if texture_map.dim() != 3 or texture_map.size(2) != 3:
            message = (
                "'texture_map' should either be empty or of shape (H, W, 3); if multiple "
                "textures, try providing texture_images instead."
            )
            raise ValueError(message)

    if texture_images is not None and materials_idx is None:
        message = "If texture_images is not None, materials_idx must be provided"
        raise ValueError(message)

    if None not in [texture_images]:
        if not isinstance(texture_images, dict):
            message = "texture_images must be a dictionary"
            raise ValueError(message)

    if (normals is None) != (faces_normals_idx is None):
        message = "'normals' and 'faces_normals_idx' must both be None or neither."
        raise ValueError(message)

    if faces_normals_idx is not None and (
        faces_normals_idx.dim() != 2 or faces_normals_idx.size(1) != 3
    ):
        message = (
            "'faces_normals_idx' should either be empty or of shape (num_faces, 3)."
        )
        raise ValueError(message)

    if normals is not None and (normals.dim() != 2 or normals.size(1) != 3):
        message = "'normals' should either be empty or of shape (num_verts, 3)."
        raise ValueError(message)
