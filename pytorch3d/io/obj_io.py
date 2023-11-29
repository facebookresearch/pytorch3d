# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


"""This module implements utility functions for loading and saving meshes."""
import os
import warnings
from collections import namedtuple
from pathlib import Path
from typing import List, Optional, Dict, Union, Tuple

import numpy as np
import torch
from iopath.common.file_io import PathManager
from PIL import Image

from pytorch3d.utils.obj_utils import (
    parse_obj_to_mesh_by_texture,
    _reindex_face_normals_by_index,
    _reindex_obj_materials_by_index,
    _reindex_verts_faces_by_index,
    _reindex_verts_faces_uvs_by_index,
    _validate_obj,
)
from pytorch3d.common.datatypes import Device
from pytorch3d.io.mtl_io import load_mtl, make_mesh_texture_atlas
from pytorch3d.io.utils import _check_faces_indices, _make_tensor, _open_file, PathOrStr
from pytorch3d.renderer.mesh.textures import TexturesAtlas, TexturesUV
from pytorch3d.structures import join_meshes_as_batch, join_meshes_as_scene, Meshes

from .pluggable_formats import endswith, MeshFormatInterpreter

# Faces & Aux type returned from load_obj function.
_Faces = namedtuple("Faces", "verts_idx normals_idx textures_idx materials_idx")
_Aux = namedtuple(
    "Properties", "normals verts_uvs material_colors texture_images texture_atlas"
)


def _format_faces_indices(faces_indices, max_index: int, device, pad_value=None):
    """
    Format indices and check for invalid values. Indices can refer to
    values in one of the face properties: vertices, textures or normals.
    See comments of the load_obj function for more details.

    Args:
        faces_indices: List of ints of indices.
        max_index: Max index for the face property.
        pad_value: if any of the face_indices are padded, specify
            the value of the padding (e.g. -1). This is only used
            for texture indices indices where there might
            not be texture information for all the faces.

    Returns:
        faces_indices: List of ints of indices.

    Raises:
        ValueError if indices are not in a valid range.
    """
    faces_indices = _make_tensor(
        faces_indices, cols=3, dtype=torch.int64, device=device
    )

    if pad_value is not None:
        mask = faces_indices.eq(pad_value).all(dim=-1)

    # Change to 0 based indexing.
    faces_indices[(faces_indices > 0)] -= 1

    # Negative indexing counts from the end.
    faces_indices[(faces_indices < 0)] += max_index

    if pad_value is not None:
        # pyre-fixme[61]: `mask` is undefined, or not always defined.
        faces_indices[mask] = pad_value

    return _check_faces_indices(faces_indices, max_index, pad_value)


def load_obj(
    f,
    load_textures: bool = True,
    create_texture_atlas: bool = False,
    texture_atlas_size: int = 4,
    texture_wrap: Optional[str] = "repeat",
    device: Device = "cpu",
    path_manager: Optional[PathManager] = None,
):
    """
    Load a mesh from a .obj file and optionally textures from a .mtl file.
    Currently this handles verts, faces, vertex texture uv coordinates, normals,
    texture images and material reflectivity values.

    Note .obj files are 1-indexed. The tensors returned from this function
    are 0-indexed. OBJ spec reference: http://www.martinreddy.net/gfx/3d/OBJ.spec

    Example .obj file format:
    ::
        # this is a comment
        v 1.000000 -1.000000 -1.000000
        v 1.000000 -1.000000 1.000000
        v -1.000000 -1.000000 1.000000
        v -1.000000 -1.000000 -1.000000
        v 1.000000 1.000000 -1.000000
        vt 0.748573 0.750412
        vt 0.749279 0.501284
        vt 0.999110 0.501077
        vt 0.999455 0.750380
        vn 0.000000 0.000000 -1.000000
        vn -1.000000 -0.000000 -0.000000
        vn -0.000000 -0.000000 1.000000
        f 5/2/1 1/2/1 4/3/1
        f 5/1/1 4/3/1 2/4/1

    The first character of the line denotes the type of input:
    ::
        - v is a vertex
        - vt is the texture coordinate of one vertex
        - vn is the normal of one vertex
        - f is a face

    Faces are interpreted as follows:
    ::
        5/2/1 describes the first vertex of the first triangle
        - 5: index of vertex [1.000000 1.000000 -1.000000]
        - 2: index of texture coordinate [0.749279 0.501284]
        - 1: index of normal [0.000000 0.000000 -1.000000]

    If there are faces with more than 3 vertices
    they are subdivided into triangles. Polygonal faces are assumed to have
    vertices ordered counter-clockwise so the (right-handed) normal points
    out of the screen e.g. a proper rectangular face would be specified like this:
    ::
        0_________1
        |         |
        |         |
        3 ________2

    The face would be split into two triangles: (0, 2, 1) and (0, 3, 2),
    both of which are also oriented counter-clockwise and have normals
    pointing out of the screen.

    Args:
        f: A file-like object (with methods read, readline, tell, and seek),
           a pathlib path or a string containing a file name.
        load_textures: Boolean indicating whether material files are loaded
        create_texture_atlas: Bool, If True a per face texture map is created and
            a tensor `texture_atlas` is also returned in `aux`.
        texture_atlas_size: Int specifying the resolution of the texture map per face
            when `create_texture_atlas=True`. A (texture_size, texture_size, 3)
            map is created per face.
        texture_wrap: string, one of ["repeat", "clamp"]. This applies when computing
            the texture atlas.
            If `texture_mode="repeat"`, for uv values outside the range [0, 1] the integer part
            is ignored and a repeating pattern is formed.
            If `texture_mode="clamp"` the values are clamped to the range [0, 1].
            If None, then there is no transformation of the texture values.
        device: Device (as str or torch.device) on which to return the new tensors.
        path_manager: optionally a PathManager object to interpret paths.

    Returns:
        6-element tuple containing

        - **verts**: FloatTensor of shape (V, 3).
        - **faces**: NamedTuple with fields:
            - verts_idx: LongTensor of vertex indices, shape (F, 3).
            - normals_idx: (optional) LongTensor of normal indices, shape (F, 3).
            - textures_idx: (optional) LongTensor of texture indices, shape (F, 3).
              This can be used to index into verts_uvs.
            - materials_idx: (optional) List of indices indicating which
              material the texture is derived from for each face.
              If there is no material for a face, the index is -1.
              This can be used to retrieve the corresponding values
              in material_colors/texture_images after they have been
              converted to tensors or Materials/Textures data
              structures - see textures.py and materials.py for
              more info.
        - **aux**: NamedTuple with fields:
            - normals: FloatTensor of shape (N, 3)
            - verts_uvs: FloatTensor of shape (T, 2), giving the uv coordinate per
              vertex. If a vertex is shared between two faces, it can have
              a different uv value for each instance. Therefore it is
              possible that the number of verts_uvs is greater than
              num verts i.e. T > V.
              vertex.
            - material_colors: if `load_textures=True` and the material has associated
              properties this will be a dict of material names and properties of the form:

              .. code-block:: python

                  {
                      material_name_1:  {
                          "ambient_color": tensor of shape (1, 3),
                          "diffuse_color": tensor of shape (1, 3),
                          "specular_color": tensor of shape (1, 3),
                          "shininess": tensor of shape (1)
                      },
                      material_name_2: {},
                      ...
                  }

              If a material does not have any properties it will have an
              empty dict. If `load_textures=False`, `material_colors` will None.

            - texture_images: if `load_textures=True` and the material has a texture map,
              this will be a dict of the form:

              .. code-block:: python

                  {
                      material_name_1: (H, W, 3) image,
                      ...
                  }
              If `load_textures=False`, `texture_images` will None.
            - texture_atlas: if `load_textures=True` and `create_texture_atlas=True`,
              this will be a FloatTensor of the form: (F, texture_size, textures_size, 3)
              If the material does not have a texture map, then all faces
              will have a uniform white texture.  Otherwise `texture_atlas` will be
              None.
    """
    data_dir = "./"
    if isinstance(f, (str, bytes, Path)):
        # pyre-fixme[6]: For 1st argument expected `PathLike[Variable[AnyStr <:
        #  [str, bytes]]]` but got `Union[Path, bytes, str]`.
        data_dir = os.path.dirname(f)
    if path_manager is None:
        path_manager = PathManager()
    with _open_file(f, path_manager, "r") as f:
        return _load_obj(
            f,
            data_dir=data_dir,
            load_textures=load_textures,
            create_texture_atlas=create_texture_atlas,
            texture_atlas_size=texture_atlas_size,
            texture_wrap=texture_wrap,
            path_manager=path_manager,
            device=device,
        )


def subset_obj(
    obj: Tuple[torch.Tensor, _Faces, _Aux],
    faces_to_subset: torch.Tensor,
    device: Device,
) -> Tuple[torch.Tensor, _Faces, _Aux]:
    """A utility function to subset an obj by a faces_to_subset that represents
    the face indices to keep. Provides support for multitexture objs.

    Args:
        obj: An obj which is a 3-tuple of verts (torch.Tensor),
            faces (NamedTuple containing: verts_idx normals_idx textures_idx materials_idx),
            and aux (NamedTuple containing: normals verts_uvs material_colors texture_images texture_atlas).
        faces_to_subset: A 1-dimentional tensor that represents the desired
            indices of the faces to keep in the subset.
        device: Device (as str or torch.device) on which to return the new tensors.

    Returns:
        An subset of the input obj which effectively copies the input according to faces_to_subset.
        - obj: A subset copy of the input; 3-tuple of data structures of verts, faces, and aux.
    """
    # initialize default/empty values
    _texture_images, _texture_atlas, _material_colors = None, None, None
    _verts_uvs, _faces_uvs, _mtl_idx = None, None, None
    _normals, _normals_idx = None, None

    if len(obj) != 3:
        message = "obj must be 3-tuple"
        raise ValueError(message)

    if not isinstance(obj[1], _Faces):
        message = "obj[1] must be a _Faces NamedTuple object that defines obj faces"
        raise ValueError(message)

    if not isinstance(obj[2], _Aux):
        message = "obj[2] must be an _Aux NamedTuple object that defines obj properties"
        raise ValueError(message)

    _validate_obj(
        verts=obj[0],
        faces=obj[1].verts_idx,
        faces_uvs=obj[1].textures_idx,
        verts_uvs=obj[2].verts_uvs,
        texture_images=obj[2].texture_images,
        materials_idx=obj[1].materials_idx,
    )

    # raise errors for specific conditions
    if faces_to_subset.shape[0] == 0:
        message = "faces_to_subset is empty."
        raise ValueError(message)

    if not isinstance(faces_to_subset, torch.Tensor):
        message = "faces_to_subset must be a torch.Tensor"
        raise ValueError(message)

    unique_faces_to_subset = torch.unique(faces_to_subset)

    if unique_faces_to_subset.shape[0] != faces_to_subset.shape[0]:
        message = "Face indices are repeated in faces_to_subset."
        warnings.warn(message)

    if (
        unique_faces_to_subset.max() >= obj[1].verts_idx.shape[0]
        or unique_faces_to_subset.min() < 0
    ):
        message = "faces_to_subset contains invalid indices."
        raise ValueError(message)

    subset_device = faces_to_subset.device

    if not all(
        [
            obj[0].device == subset_device,
            obj[1].verts_idx.device == subset_device,
            obj[1].normals_idx.device == subset_device
            if obj[1].normals_idx is not None
            else True,
            obj[1].textures_idx.device == subset_device
            if obj[1].textures_idx is not None
            else True,
            obj[1].materials_idx.device == subset_device
            if obj[1].materials_idx is not None
            else True,
            obj[2].texture_atlas.device == subset_device
            if obj[2].texture_atlas is not None
            else True,
            obj[2].verts_uvs.device == subset_device
            if obj[2].verts_uvs is not None
            else True,
        ]
    ):
        message = "obj and faces_to_subset are not on the same device"
        raise ValueError(message)

    # reindex faces and verts according to faces_to_subset
    _verts_idx, _faces = _reindex_verts_faces_by_index(
        obj[1].verts_idx, faces_to_subset
    )

    # reindex face normals according to faces_to_subset, if present
    if obj[2].normals is not None and obj[1].normals_idx is not None:
        _normals_idx_orig, _normals_idx = _reindex_face_normals_by_index(
            obj[1].normals_idx, faces_to_subset
        )
        _normals = obj[2].normals[_normals_idx_orig]

    # slice texture_atlas by faces_to_subset, if present
    if obj[2].texture_atlas is not None:
        _texture_atlas = obj[2].texture_atlas[faces_to_subset].to(device)

    if obj[2].verts_uvs is not None:
        # if textures exist, reindex them in addition to faces and verts
        _verts_uvs, _faces_uvs = _reindex_verts_faces_uvs_by_index(
            obj[1].textures_idx, faces_to_subset
        )
        _mtl_idx_orig, _mtl_idx = _reindex_obj_materials_by_index(
            obj[1].materials_idx, faces_to_subset
        )

        # obtain a list of the texture's image keys based on the original texture order
        _tex_image_keys = list(obj[2].texture_images.keys())
        # for each index in _mtl_idx_orig, get the corresponding value in _tex_image_keys
        _tex_image_keys = list(map(_tex_image_keys.__getitem__, _mtl_idx_orig.tolist()))
        # reconstruct the texture images dictionary with the new texture image keys
        _texture_images = {
            k: obj[2].texture_images[k].to(device) for k in _tex_image_keys
        }

        if len(_texture_images) != len(torch.unique(_mtl_idx)):
            warnings.warn("Invalid textures.")

        # select material_colors, if present, according to filtered texture_images
        if obj[2].material_colors is not None:
            _material_colors = {
                k: {
                    k_i: k_j.to(device)
                    for k_i, k_j in obj[2].material_colors[k].items()
                }
                for k in list(_texture_images.keys())
                if k in obj[2].material_colors.keys()
            }

    # use new index tensors to slice, assemble, and return a subset obj
    verts = obj[0][_verts_idx].to(device)

    faces = _Faces(
        verts_idx=_faces.to(device),
        normals_idx=_normals_idx.to(device)
        if _normals_idx is not None
        else _normals_idx,
        textures_idx=_faces_uvs.to(device) if _faces_uvs is not None else _faces_uvs,
        materials_idx=_mtl_idx.to(device) if _mtl_idx is not None else _mtl_idx,
    )

    aux = _Aux(
        normals=_normals.to(device) if _normals is not None else None,
        verts_uvs=obj[2].verts_uvs[_verts_uvs].to(device)
        if _verts_uvs is not None
        else None,
        material_colors=_material_colors,
        texture_images=_texture_images,
        texture_atlas=_texture_atlas,
    )

    return verts, faces, aux


def load_objs_as_meshes(
    files: list,
    device: Optional[Device] = None,
    load_textures: bool = True,
    create_texture_atlas: bool = False,
    texture_atlas_size: int = 4,
    texture_wrap: Optional[str] = "repeat",
    path_manager: Optional[PathManager] = None,
):
    """
    Load meshes from a list of .obj files using the load_obj function, and
    return them as a Meshes object. Input .obj files with multiple texture
    images are supported. See the load_obj function for more details.
    normals are not stored.

    Args:
        files: A list of file-like objects (with methods read, readline, tell,
            and seek), pathlib paths or strings containing file names.
        device: Desired device of returned Meshes. Default:
            uses the current device for the default tensor type.
        load_textures: Boolean indicating whether material files are loaded
        create_texture_atlas, texture_atlas_size, texture_wrap: as for load_obj.
        path_manager: optionally a PathManager object to interpret paths.
    Returns:
        New Meshes object.
    """
    mesh_list = []
    for f_obj in files:
        verts, faces, aux = load_obj(
            f_obj,
            load_textures=load_textures,
            create_texture_atlas=create_texture_atlas,
            texture_atlas_size=texture_atlas_size,
            texture_wrap=texture_wrap,
            path_manager=path_manager,
        )

        if create_texture_atlas:
            # TexturesAtlas type
            mesh = Meshes(
                verts=[verts.to(device)],
                faces=[faces.verts_idx.to(device)],
                textures=TexturesAtlas(atlas=[aux.texture_atlas.to(device)]),
            )

        elif aux.texture_images is not None and len(aux.texture_images) > 0:
            # TexturesUV type with support for multiple texture inputs
            mesh = parse_obj_to_mesh_by_texture(
                verts=verts,
                faces=faces.verts_idx,
                verts_uvs=aux.verts_uvs,
                faces_uvs=faces.textures_idx,
                texture_images=aux.texture_images,
                device=device,
                materials_idx=faces.materials_idx,
                texture_atlas=aux.texture_atlas,
            )
            # combine partial meshes into a single scene
            mesh = join_meshes_as_scene(mesh)
        else:
            # if there are no valid textures
            mesh = Meshes(verts=[verts.to(device)], faces=[faces.verts_idx.to(device)])

        mesh_list.append(mesh)

    if len(mesh_list) == 1:
        return mesh_list[0]
    else:
        return join_meshes_as_batch(mesh_list)


class MeshObjFormat(MeshFormatInterpreter):
    def __init__(self) -> None:
        self.known_suffixes = (".obj",)

    def read(
        self,
        path: PathOrStr,
        include_textures: bool,
        device: Device,
        path_manager: PathManager,
        create_texture_atlas: bool = False,
        texture_atlas_size: int = 4,
        texture_wrap: Optional[str] = "repeat",
        **kwargs,
    ) -> Optional[Meshes]:
        if not endswith(path, self.known_suffixes):
            return None
        mesh = load_objs_as_meshes(
            files=[path],
            device=device,
            load_textures=include_textures,
            create_texture_atlas=create_texture_atlas,
            texture_atlas_size=texture_atlas_size,
            texture_wrap=texture_wrap,
            path_manager=path_manager,
        )
        return mesh

    def save(
        self,
        data: Meshes,
        path: PathOrStr,
        path_manager: PathManager,
        binary: Optional[bool],
        decimal_places: Optional[int] = None,
        **kwargs,
    ) -> bool:
        if not endswith(path, self.known_suffixes):
            return False

        verts = data.verts_list()[0]
        faces = data.faces_list()[0]

        verts_uvs: Optional[torch.Tensor] = None
        faces_uvs: Optional[torch.Tensor] = None
        texture_map: Optional[torch.Tensor] = None

        if isinstance(data.textures, TexturesUV):
            verts_uvs = data.textures.verts_uvs_padded()[0]
            faces_uvs = data.textures.faces_uvs_padded()[0]
            texture_map = data.textures.maps_padded()[0]

        save_obj(
            f=path,
            verts=verts,
            faces=faces,
            decimal_places=decimal_places,
            path_manager=path_manager,
            verts_uvs=verts_uvs,
            faces_uvs=faces_uvs,
            texture_map=texture_map,
        )
        return True


def _parse_face(
    line,
    tokens,
    material_idx,
    faces_verts_idx,
    faces_normals_idx,
    faces_textures_idx,
    faces_materials_idx,
) -> None:
    face = tokens[1:]
    face_list = [f.split("/") for f in face]
    face_verts = []
    face_normals = []
    face_textures = []

    for vert_props in face_list:
        # Vertex index.
        face_verts.append(int(vert_props[0]))
        if len(vert_props) > 1:
            if vert_props[1] != "":
                # Texture index is present e.g. f 4/1/1.
                face_textures.append(int(vert_props[1]))
            # if len(vert_props) > 2:
            if len(vert_props) > 2 and len(vert_props[2]) != 0:
                # do not parse normals if empty: ValueError: invalid literal for int() with base 10: ''
                # Normal index present e.g. 4/1/1 or 4//1.
                face_normals.append(int(vert_props[2]))
            if len(vert_props) > 3:
                raise ValueError(
                    "Face vertices can only have 3 properties. \
                                Face vert %s, Line: %s"
                    % (str(vert_props), str(line))
                )

    # Triplets must be consistent for all vertices in a face e.g.
    # legal statement: f 4/1/1 3/2/1 2/1/1.
    # illegal statement: f 4/1/1 3//1 2//1.
    # If the face does not have normals or textures indices
    # fill with pad value = -1. This will ensure that
    # all the face index tensors will have F values where
    # F is the number of faces.
    if len(face_normals) > 0:
        if not (len(face_verts) == len(face_normals)):
            raise ValueError(
                "Face %s is an illegal statement. \
                        Vertex properties are inconsistent. Line: %s"
                % (str(face), str(line))
            )
    else:
        face_normals = [-1] * len(face_verts)  # Fill with -1
    if len(face_textures) > 0:
        if not (len(face_verts) == len(face_textures)):
            raise ValueError(
                "Face %s is an illegal statement. \
                        Vertex properties are inconsistent. Line: %s"
                % (str(face), str(line))
            )
    else:
        face_textures = [-1] * len(face_verts)  # Fill with -1

    # Subdivide faces with more than 3 vertices.
    # See comments of the load_obj function for more details.
    for i in range(len(face_verts) - 2):
        faces_verts_idx.append((face_verts[0], face_verts[i + 1], face_verts[i + 2]))
        faces_normals_idx.append(
            (face_normals[0], face_normals[i + 1], face_normals[i + 2])
        )
        faces_textures_idx.append(
            (face_textures[0], face_textures[i + 1], face_textures[i + 2])
        )
        faces_materials_idx.append(material_idx)


def _parse_obj(f, data_dir: str):
    """
    Load a mesh from a file-like object. See load_obj function for more details
    about the return values.
    """
    verts, normals, verts_uvs = [], [], []
    faces_verts_idx, faces_normals_idx, faces_textures_idx = [], [], []
    faces_materials_idx = []
    material_names = []
    mtl_path = None

    lines = [line.strip() for line in f]

    # startswith expects each line to be a string. If the file is read in as
    # bytes then first decode to strings.
    if lines and isinstance(lines[0], bytes):
        lines = [el.decode("utf-8") for el in lines]

    materials_idx = -1

    for line in lines:
        tokens = line.strip().split()
        if line.startswith("mtllib"):
            if len(tokens) < 2:
                raise ValueError("material file name is not specified")
            # NOTE: only allow one .mtl file per .obj.
            # Definitions for multiple materials can be included
            # in this one .mtl file.
            mtl_path = line[len(tokens[0]) :].strip()  # Take the remainder of the line
            mtl_path = os.path.join(data_dir, mtl_path)
        elif len(tokens) and tokens[0] == "usemtl":
            material_name = tokens[1]
            # materials are often repeated for different parts
            # of a mesh.
            if material_name not in material_names:
                material_names.append(material_name)
                materials_idx = len(material_names) - 1
            else:
                materials_idx = material_names.index(material_name)
        elif line.startswith("v "):  # Line is a vertex.
            vert = [float(x) for x in tokens[1:4]]
            if len(vert) != 3:
                msg = "Vertex %s does not have 3 values. Line: %s"
                raise ValueError(msg % (str(vert), str(line)))
            verts.append(vert)
        elif line.startswith("vt "):  # Line is a texture.
            tx = [float(x) for x in tokens[1:3]]
            if len(tx) != 2:
                raise ValueError(
                    "Texture %s does not have 2 values. Line: %s" % (str(tx), str(line))
                )
            verts_uvs.append(tx)
        elif line.startswith("vn "):  # Line is a normal.
            norm = [float(x) for x in tokens[1:4]]
            if len(norm) != 3:
                msg = "Normal %s does not have 3 values. Line: %s"
                raise ValueError(msg % (str(norm), str(line)))
            normals.append(norm)
        elif line.startswith("f "):  # Line is a face.
            # Update face properties info.
            _parse_face(
                line,
                tokens,
                materials_idx,
                faces_verts_idx,
                faces_normals_idx,
                faces_textures_idx,
                faces_materials_idx,
            )

    return (
        verts,
        normals,
        verts_uvs,
        faces_verts_idx,
        faces_normals_idx,
        faces_textures_idx,
        faces_materials_idx,
        material_names,
        mtl_path,
    )


def _load_materials(
    material_names: List[str],
    f: Optional[str],
    *,
    data_dir: str,
    load_textures: bool,
    device: Device,
    path_manager: PathManager,
):
    """
    Load materials and optionally textures from the specified path.

    Args:
        material_names: a list of the material names found in the .obj file.
        f: path to the material information.
        data_dir: the directory where the material texture files are located.
        load_textures: whether textures should be loaded.
        device: Device (as str or torch.device) on which to return the new tensors.
        path_manager: PathManager object to interpret paths.

    Returns:
        material_colors: dict of properties for each material.
        texture_images: dict of material names and texture images.
    """
    if not load_textures:
        return None, None

    if f is None:
        warnings.warn("No mtl file provided")
        return None, None

    if not path_manager.exists(f):
        warnings.warn(f"Mtl file does not exist: {f}")
        return None, None

    # Texture mode uv wrap
    return load_mtl(
        f,
        material_names=material_names,
        data_dir=data_dir,
        path_manager=path_manager,
        device=device,
    )


def _load_obj(
    f_obj,
    *,
    data_dir: str,
    load_textures: bool = True,
    create_texture_atlas: bool = False,
    texture_atlas_size: int = 4,
    texture_wrap: Optional[str] = "repeat",
    path_manager: PathManager,
    device: Device = "cpu",
):
    """
    Load a mesh from a file-like object. See load_obj function more details.
    Any material files associated with the obj are expected to be in the
    directory given by data_dir.
    """

    if texture_wrap is not None and texture_wrap not in ["repeat", "clamp"]:
        msg = "texture_wrap must be one of ['repeat', 'clamp'] or None, got %s"
        raise ValueError(msg % texture_wrap)

    (
        verts,
        normals,
        verts_uvs,
        faces_verts_idx,
        faces_normals_idx,
        faces_textures_idx,
        faces_materials_idx,
        material_names,
        mtl_path,
    ) = _parse_obj(f_obj, data_dir)

    verts = _make_tensor(
        verts,
        cols=3,
        dtype=torch.float32,
        device=device,
    )  # (V, 3)
    normals = _make_tensor(
        normals,
        cols=3,
        dtype=torch.float32,
        device=device,
    )  # (N, 3)
    verts_uvs = _make_tensor(
        verts_uvs,
        cols=2,
        dtype=torch.float32,
        device=device,
    )  # (T, 2)

    faces_verts_idx = _format_faces_indices(
        faces_verts_idx, verts.shape[0], device=device
    )

    # Repeat for normals and textures if present.
    if len(faces_normals_idx):
        faces_normals_idx = _format_faces_indices(
            faces_normals_idx, normals.shape[0], device=device, pad_value=-1
        )
    if len(faces_textures_idx):
        faces_textures_idx = _format_faces_indices(
            faces_textures_idx, verts_uvs.shape[0], device=device, pad_value=-1
        )
    if len(faces_materials_idx):
        faces_materials_idx = torch.tensor(
            faces_materials_idx, dtype=torch.int64, device=device
        )

    texture_atlas = None
    material_colors, texture_images = _load_materials(
        material_names,
        mtl_path,
        data_dir=data_dir,
        load_textures=load_textures,
        path_manager=path_manager,
        device=device,
    )

    if material_colors and not material_names:
        # usemtl was not present but single material was present in the .mtl file
        material_names.append(next(iter(material_colors.keys())))
        # replace all -1 by 0 material idx
        if torch.is_tensor(faces_materials_idx):
            faces_materials_idx.clamp_(min=0)

    if create_texture_atlas:
        # Using the images and properties from the
        # material file make a per face texture map.

        # Create an array of strings of material names for each face.
        # If faces_materials_idx == -1 then that face doesn't have a material.
        idx = faces_materials_idx.cpu().numpy()
        face_material_names = np.array(material_names)[idx]  # (F,)
        face_material_names[idx == -1] = ""

        # Construct the atlas.
        texture_atlas = make_mesh_texture_atlas(
            material_colors,
            texture_images,
            face_material_names,
            faces_textures_idx,
            verts_uvs,
            texture_atlas_size,
            texture_wrap,
        )

    faces = _Faces(
        verts_idx=faces_verts_idx,
        normals_idx=faces_normals_idx,
        textures_idx=faces_textures_idx,
        materials_idx=faces_materials_idx,
    )
    aux = _Aux(
        normals=normals if len(normals) else None,
        verts_uvs=verts_uvs if len(verts_uvs) else None,
        material_colors=material_colors,
        texture_images=texture_images,
        texture_atlas=texture_atlas,
    )
    return verts, faces, aux


def save_obj(
    f: PathOrStr,
    verts,
    faces,
    decimal_places: Optional[int] = None,
    path_manager: Optional[PathManager] = None,
    *,
    normals: Optional[torch.Tensor] = None,
    faces_normals_idx: Optional[torch.Tensor] = None,
    verts_uvs: Optional[torch.Tensor] = None,
    faces_uvs: Optional[torch.Tensor] = None,
    texture_map: Optional[torch.Tensor] = None,
    texture_images: Optional[Dict[str, torch.tensor]] = None,
    materials_idx: Optional[torch.Tensor] = None,
    image_quality: Optional[int] = 75,
    image_subsampling: Optional[Union[str, int]] = 0,
    image_format: Optional[str] = "png",
    **image_name_kwargs,
) -> None:
    """
    Save a mesh to an .obj file.

    Args:
        f: File (str or path) to which the mesh should be written.
        verts: FloatTensor of shape (V, 3) giving vertex coordinates.
        faces: LongTensor of shape (F, 3) giving faces.
        decimal_places: Number of decimal places for saving.
        path_manager: Optional PathManager for interpreting f if
            it is a str.
        normals: FloatTensor of shape (V, 3) giving normals for faces_normals_idx
            to index into.
        faces_normals_idx: LongTensor of shape (F, 3) giving the index into
            normals for each vertex in the face.
        verts_uvs: FloatTensor of shape (V, 2) giving the uv coordinate per vertex.
        faces_uvs: LongTensor of shape (F, 3) giving the index into verts_uvs for
            each vertex in the face.
        texture_map: FloatTensor of shape (H, W, 3) representing the texture map
            for the mesh which will be saved as an image. The values are expected
            to be in the range [0, 1]. Supports saving an obj with a single texture.
            If providing texture_map, texture_images must be None.
        texture_images: Dictionary of str:FloatTensor of shape (H, W, 3) where
            where each key value pair, in order, represnts a material name and
            texture map; in objs, this value is often the aux.texture_images object.
            Providing texture_images enables saving an obj with multiple textures
            and texture files. If providing texture_images, texture_map must be None
            and materials_idx must be provided.
        materials_idx: IntTensor of shape (F, ) giving the material index that links
            each face in faces to a texture in texture_images. If saving multiple
            textures and providing a texture_images object, materials_idx must be
            provided. This value is often the aux.materials_idx value in an obj.
        image_quality: An optional integer value between 0 and 95 to pass through
            to PIL that sets texture image quality. See PIL docs for details:
            https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html.
        image_subsampling: An optional string or integer value to pass through to
            PIL that sets subsampling. See PIL docs for details:
            https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html.
        image_format: An optional string value that can be either 'png' or 'jpeg'
            to pass through to PIL that sets the image type. See PIL docs for details:
            https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html.
        **image_name_kwargs: Optional kwargs to determine image file names.
            For an obj with n textures, the default behavior saves an image name with
            the obj basename appended with the index position of the texture. Example:
            output.obj with 2 texture files writes output_0.png and output_1.png as
            image files.

            Available image_name_kwargs include:
            * material_name_as_file_name: If True, the image file is saved with the
            material name as the filename. Example: output.obj with material_1 and
            material_2 as textures writes image files of material_1.png and
            material_2.png, respectively.
            * reuse_material_files: If True, material names are used as filenames.
            In addition, image files are reused (without rewriting) for all objs within the
            same directory that reference the same material name. This option may be
            convenient if the same materials are re-used across obj files or if writing
            subsets of the input obj to the origin directory. In the latter case the original
            material files are preserverd and only new mtl and obj files of the subsets are written.
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

    if texture_map is not None and texture_images is not None:
        message = "texture_map is not None and texture_images is not None; only one can be provided"
        raise ValueError(message)

    if not any([image_format == i for i in ["png", "jpeg"]]):
        message = "'image_format' must be either 'png' or 'jpeg'"
        raise ValueError(message)

    if image_quality < 1 or image_quality > 95:
        message = "'image_quality is recommended to be set between 0 and 95 according to PIL documentation"
        warnings.warn(message)

    if path_manager is None:
        path_manager = PathManager()

    save_texture = all([t is not None for t in [faces_uvs, verts_uvs]]) and (
        texture_map is not None or texture_images is not None
    )
    output_path = Path(f)

    if save_texture and texture_images is None:
        # if a single texture map is given, treat it like a texture_images with a single image
        texture_images = dict(mesh=texture_map)
        materials_idx = torch.zeros(faces.shape[0], dtype=torch.int64)

    # configure how image names are written to disk
    enumerate_material_filename_by_index = True  # the default setting
    reuse_material_files = False
    material_name_as_file_name = False

    if image_name_kwargs is not None:
        image_name_options = image_name_kwargs.get("image_name_kwargs", None)
    if image_name_options is not None:
        reuse_material_files = image_name_options.get(
            "reuse_material_files", reuse_material_files
        )
        material_name_as_file_name = image_name_options.get(
            "material_name_as_file_name", material_name_as_file_name
        )

        if reuse_material_files or material_name_as_file_name:
            enumerate_material_filename_by_index = False

    # Save the .obj file
    with _open_file(f, path_manager, "w") as f:
        if save_texture:
            obj_header = "\nmtllib {0}.mtl\nusemtl mesh\n\n".format(output_path.stem)
            f.write(obj_header)

            # init offsets and a mtl path
            face_offset, uvs_offset, normals_offset = 0, 0, 0
            mtl_path = output_path.with_suffix(".mtl")
            # init an mtl list to accumulate material strings
            mtl_lines = []

            if isinstance(f, str):
                # Back to str for iopath interpretation.
                mtl_path = str(mtl_path)

            # split the obj into sections by material
            for tex_idx, (tex_name, tex_value) in enumerate(texture_images.items()):
                faces_to_subset = materials_idx == tex_idx
                faces_to_subset = faces_to_subset.nonzero().squeeze().ravel()

                # parse obj by associated faces and verts for each material, if not None
                if faces_to_subset.numel():
                    # Add the header required for the texture info to be loaded correctly
                    obj_header = (
                        f"\nmtllib {output_path.stem}.mtl\nusemtl {tex_name}\n\n"
                    )

                    f.write(obj_header)

                    # re-index verts and faces based on the current faces
                    _verts_idx, _faces = _reindex_verts_faces_by_index(
                        faces, faces_to_subset
                    )

                    # re-index vert uvs and face uvs  based on current faces
                    _verts_uvs, _faces_uvs = _reindex_verts_faces_uvs_by_index(
                        faces_uvs, faces_to_subset
                    )

                    if normals is not None and faces_normals_idx is not None:
                        (
                            _normals_idx_orig,
                            _normals_idx,
                        ) = _reindex_face_normals_by_index(
                            faces_normals_idx, faces_to_subset
                        )
                        _normals = normals[_normals_idx_orig]
                    else:
                        _normals, _normals_idx = None, None

                    # in multitexture obj output, the current faces may reference verts that reference other materials
                    if not torch.any(
                        _faces + face_offset >= verts.shape[0]
                    ) and not torch.any(_faces_uvs + uvs_offset < 0):
                        # supress where the current faces do not exceed the overall verts and do not violate other conditions
                        supress_face_warning = True
                    else:
                        # if exceptional conditions are not met, disable supression
                        supress_face_warning = False

                    # write current lines to obj and appy offset to faces and uvs, subsequent iterations append to the open file
                    _save(
                        f,
                        verts[_verts_idx],
                        _faces + face_offset,
                        decimal_places,
                        normals=_normals,
                        faces_normals_idx=_normals_idx + normals_offset
                        if _normals is not None
                        else _normals_idx,
                        verts_uvs=verts_uvs[_verts_uvs],
                        faces_uvs=_faces_uvs + uvs_offset,
                        save_texture=save_texture,
                        save_normals=_normals is not None,
                        supress_face_warning=supress_face_warning,
                    )

                    # increment the offset by the size of verts and uvs for each material
                    face_offset += verts[_verts_idx].shape[0]
                    uvs_offset += verts_uvs[_verts_uvs].shape[0]
                    if _normals is not None:
                        normals_offset += _normals.shape[0]

                    # set a path for the current texture and save the image
                    if enumerate_material_filename_by_index:
                        # if multiple images, append the index
                        if len(texture_images) > 1:
                            _image_name = f"{output_path.stem}_{tex_idx}.{image_format}"
                        else:
                            # if only one material, use only the stem
                            _image_name = f"{output_path.stem}.{image_format}"

                        image_path = output_path.parent / _image_name
                    else:
                        image_path = output_path.parent / f"{tex_name}.{image_format}"

                    if isinstance(f, str):
                        # Back to str for iopath interpretation.
                        image_path = str(image_path)

                    if os.path.exists(image_path) and reuse_material_files:
                        skip_save_texture_image = True
                    else:
                        skip_save_texture_image = False

                    if not skip_save_texture_image:
                        _save_texture_image(
                            tex_value,
                            image_path,
                            path_manager,
                            image_format,
                            image_subsampling,
                            image_quality,
                        )

                    # accumulate materials as a list of strings
                    mtl_lines.append(
                        f"newmtl {tex_name}\n"
                        f"map_Kd {image_path.stem}.{image_format}\n"
                    )

            # write accumlated materials to file after iterating by textures for obj lines and images
            _save_material_lines(mtl_path, path_manager, mtl_lines)

        else:
            # if there are no given or valid textures, write only an obj
            _save(
                f,
                verts,
                faces,
                decimal_places,
                normals=normals,
                faces_normals_idx=faces_normals_idx,
                verts_uvs=verts_uvs,
                faces_uvs=faces_uvs,
                save_texture=save_texture,
                save_normals=normals is not None,
            )


def _save_texture_image(
    texture: torch.Tensor,
    image_path: str,
    path_manager: PathManager,
    image_format: str,
    image_subsampling: Union[str, int],
    image_quality: int,
):
    """A utility function that writes image data as textures to file for objs."""

    # Save texture map to output folder
    # pyre-fixme[16] # undefined attribute cpu
    texture = texture.detach().cpu() * 255.0
    image = Image.fromarray(texture.numpy().astype(np.uint8))
    with _open_file(image_path, path_manager, "wb") as im_f:
        image.save(
            im_f,
            format=image_format.upper(),
            subsampling=image_subsampling,
            quality=image_quality,
        )


def _save_material_lines(
    mtl_path: str, path_manager: PathManager, mtl_lines: List[str]
):
    """A utility function that writes material information to file for objs."""
    # Create .mtl file with the material name and texture map filename
    # TODO: enable material properties to also be saved.
    with _open_file(mtl_path, path_manager, "w") as f_mtl:
        for mtl_line in mtl_lines:
            f_mtl.write(mtl_line)


def _save(
    f,
    verts,
    faces,
    decimal_places: Optional[int] = None,
    *,
    normals: Optional[torch.Tensor] = None,
    faces_normals_idx: Optional[torch.Tensor] = None,
    verts_uvs: Optional[torch.Tensor] = None,
    faces_uvs: Optional[torch.Tensor] = None,
    save_texture: bool = False,
    save_normals: bool = False,
    supress_face_warning: Optional[bool] = False,
) -> None:
    if len(verts) and (verts.dim() != 2 or verts.size(1) != 3):
        message = "'verts' should either be empty or of shape (num_verts, 3)."
        raise ValueError(message)

    if len(faces) and (faces.dim() != 2 or faces.size(1) != 3):
        message = "'faces' should either be empty or of shape (num_faces, 3)."
        raise ValueError(message)

    if not (len(verts) or len(faces)):
        warnings.warn("Empty 'verts' and 'faces' arguments provided")
        return

    verts, faces = verts.cpu(), faces.cpu()

    lines = ""

    if decimal_places is None:
        float_str = "%f"
    else:
        float_str = "%" + ".%df" % decimal_places

    if len(verts):
        V, D = verts.shape
        for i in range(V):
            vert = [float_str % verts[i, j] for j in range(D)]
            lines += "v %s\n" % " ".join(vert)

    if save_normals:
        assert normals is not None
        assert faces_normals_idx is not None
        lines += _write_normals(normals, faces_normals_idx, float_str)

    if save_texture:
        assert faces_uvs is not None
        assert verts_uvs is not None

        if faces_uvs is not None and (faces_uvs.dim() != 2 or faces_uvs.size(1) != 3):
            message = "'faces_uvs' should either be empty or of shape (num_faces, 3)."
            raise ValueError(message)

        if verts_uvs is not None and (verts_uvs.dim() != 2 or verts_uvs.size(1) != 2):
            message = "'verts_uvs' should either be empty or of shape (num_verts, 2)."
            raise ValueError(message)

        verts_uvs, faces_uvs = verts_uvs.cpu(), faces_uvs.cpu()

        # Save verts uvs after verts
        if len(verts_uvs):
            uV, uD = verts_uvs.shape
            for i in range(uV):
                uv = [float_str % verts_uvs[i, j] for j in range(uD)]
                lines += "vt %s\n" % " ".join(uv)

    f.write(lines)

    if not supress_face_warning:
        # this warning may not be valid if writing multi-texture obj
        if torch.any(faces >= verts.shape[0]) or torch.any(faces < 0):
            warnings.warn("Faces have invalid indices")

    if len(faces):
        _write_faces(
            f,
            faces,
            faces_uvs if save_texture else None,
            faces_normals_idx if save_normals else None,
        )


def _write_normals(
    normals: torch.Tensor, faces_normals_idx: torch.Tensor, float_str: str
) -> str:
    if faces_normals_idx.dim() != 2 or faces_normals_idx.size(1) != 3:
        message = (
            "'faces_normals_idx' should either be empty or of shape (num_faces, 3)."
        )
        raise ValueError(message)

    if normals.dim() != 2 or normals.size(1) != 3:
        message = "'normals' should either be empty or of shape (num_verts, 3)."
        raise ValueError(message)

    normals, faces_normals_idx = normals.cpu(), faces_normals_idx.cpu()

    lines = []
    V, D = normals.shape
    for i in range(V):
        normal = [float_str % normals[i, j] for j in range(D)]
        lines.append("vn %s\n" % " ".join(normal))
    return "".join(lines)


def _write_faces(
    f,
    faces: torch.Tensor,
    faces_uvs: Optional[torch.Tensor],
    faces_normals_idx: Optional[torch.Tensor],
) -> None:
    F, P = faces.shape
    for i in range(F):
        if faces_normals_idx is not None:
            if faces_uvs is not None:
                # Format faces as {verts_idx}/{verts_uvs_idx}/{verts_normals_idx}
                face = [
                    "%d/%d/%d"
                    % (
                        faces[i, j] + 1,
                        faces_uvs[i, j] + 1,
                        faces_normals_idx[i, j] + 1,
                    )
                    for j in range(P)
                ]
            else:
                # Format faces as {verts_idx}//{verts_normals_idx}
                face = [
                    "%d//%d" % (faces[i, j] + 1, faces_normals_idx[i, j] + 1)
                    for j in range(P)
                ]
        elif faces_uvs is not None:
            # Format faces as {verts_idx}/{verts_uvs_idx}
            face = ["%d/%d" % (faces[i, j] + 1, faces_uvs[i, j] + 1) for j in range(P)]
        else:
            face = ["%d" % (faces[i, j] + 1) for j in range(P)]

        if i + 1 < F:
            f.write("f %s\n" % " ".join(face))
        else:
            # No newline at the end of the file.
            f.write("f %s" % " ".join(face))
