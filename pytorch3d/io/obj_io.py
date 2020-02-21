#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.


"""This module implements utility functions for loading and saving meshes."""
import numpy as np
import os
import pathlib
import warnings
from collections import namedtuple
from typing import List, Optional
import torch
from fvcore.common.file_io import PathManager
from PIL import Image

from pytorch3d.structures import Meshes, Textures, join_meshes


def _read_image(file_name: str, format=None):
    """
    Read an image from a file using Pillow.
    Args:
        file_name: image file path.
        format: one of ["RGB", "BGR"]
    Returns:
        image: an image of shape (H, W, C).
    """
    if format not in ["RGB", "BGR"]:
        raise ValueError("format can only be one of [RGB, BGR]; got %s", format)
    with PathManager.open(file_name, "rb") as f:
        image = Image.open(f)
        if format is not None:
            # PIL only supports RGB. First convert to RGB and flip channels
            # below for BGR.
            image = image.convert("RGB")
        image = np.asarray(image).astype(np.float32)
        if format == "BGR":
            image = image[:, :, ::-1]
        return image


# Faces & Aux type returned from load_obj function.
_Faces = namedtuple("Faces", "verts_idx normals_idx textures_idx materials_idx")
_Aux = namedtuple(
    "Properties", "normals verts_uvs material_colors texture_images"
)


def _format_faces_indices(faces_indices, max_index):
    """
    Format indices and check for invalid values. Indices can refer to
    values in one of the face properties: vertices, textures or normals.
    See comments of the load_obj function for more details.

    Args:
        faces_indices: List of ints of indices.
        max_index: Max index for the face property.

    Returns:
        faces_indices: List of ints of indices.

    Raises:
        ValueError if indices are not in a valid range.
    """
    faces_indices = torch.tensor(faces_indices, dtype=torch.int64)

    # Change to 0 based indexing.
    faces_indices[(faces_indices > 0)] -= 1

    # Negative indexing counts from the end.
    faces_indices[(faces_indices < 0)] += max_index

    # Check indices are valid.
    if not (
        torch.all(faces_indices < max_index) and torch.all(faces_indices >= 0)
    ):
        raise ValueError("Faces have invalid indices.")

    return faces_indices


def _open_file(f):
    new_f = False
    if isinstance(f, str):
        new_f = True
        f = open(f, "r")
    elif isinstance(f, pathlib.Path):
        new_f = True
        f = f.open("r")
    return f, new_f


def load_obj(f_obj, load_textures=True):
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
        5/2/1 describes the first vertex of the first triange
        - 5: index of vertex [1.000000 1.000000 -1.000000]
        - 2: index of texture coordinate [0.749279 0.501284]
        - 1: index of normal [0.000000 0.000000 -1.000000]

    If there are faces with more than 3 vertices
    they are subdivided into triangles. Polygonal faces are assummed to have
    vertices ordered counter-clockwise so the (right-handed) normal points
    into the screen e.g. a proper rectangular face would be specified like this:
    ::
        0_________1
        |         |
        |         |
        3 ________2

    The face would be split into two triangles: (0, 1, 2) and (0, 2, 3),
    both of which are also oriented clockwise and have normals
    pointing into the screen.

    Args:
        f: A file-like object (with methods read, readline, tell, and seek),
           a pathlib path or a string containing a file name.
        load_textures: Boolean indicating whether material files are loaded

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
            - material_colors: dict of material names and associated properties.
              If a material does not have any properties it will have an
              empty dict.

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
            - texture_images: dict of material names and texture images.
              .. code-block:: python

                  {
                      material_name_1: (H, W, 3) image,
                      ...
                  }
    """
    data_dir = "./"
    if isinstance(f_obj, (str, bytes, os.PathLike)):
        data_dir = os.path.dirname(f_obj)
    f_obj, new_f = _open_file(f_obj)
    try:
        return _load(f_obj, data_dir, load_textures=load_textures)
    finally:
        if new_f:
            f_obj.close()


def load_objs_as_meshes(files: list, device=None, load_textures: bool = True):
    """
    Load meshes from a list of .obj files using the load_obj function, and
    return them as a Meshes object. This only works for meshes which have a
    single texture image for the whole mesh. See the load_obj function for more
    details. material_colors and normals are not stored.

    Args:
        f: A list of file-like objects (with methods read, readline, tell,
        and seek), pathlib paths or strings containing file names.
        device: Desired device of returned Meshes. Default:
            uses the current device for the default tensor type.
        load_textures: Boolean indicating whether material files are loaded

    Returns:
        New Meshes object.
    """
    mesh_list = []
    for f_obj in files:
        verts, faces, aux = load_obj(f_obj, load_textures=load_textures)
        verts = verts.to(device)
        tex = None
        tex_maps = aux.texture_images
        if tex_maps is not None and len(tex_maps) > 0:
            verts_uvs = aux.verts_uvs[None, ...].to(device)  # (1, V, 2)
            faces_uvs = faces.textures_idx[None, ...].to(device)  # (1, F, 3)
            image = list(tex_maps.values())[0].to(device)[None]
            tex = Textures(verts_uvs=verts_uvs, faces_uvs=faces_uvs, maps=image)

        mesh = Meshes(
            verts=[verts], faces=[faces.verts_idx.to(device)], textures=tex
        )
        mesh_list.append(mesh)
    if len(mesh_list) == 1:
        return mesh_list[0]
    return join_meshes(mesh_list)


def _parse_face(
    line,
    material_idx,
    faces_verts_idx,
    faces_normals_idx,
    faces_textures_idx,
    faces_materials_idx,
):
    face = line.split()[1:]
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
            if len(vert_props) > 2:
                # Normal index present e.g. 4/1/1 or 4//1.
                face_normals.append(int(vert_props[2]))
            if len(vert_props) > 3:
                raise ValueError(
                    "Face vertices can ony have 3 properties. \
                                Face vert %s, Line: %s"
                    % (str(vert_props), str(line))
                )

    # Triplets must be consistent for all vertices in a face e.g.
    # legal statement: f 4/1/1 3/2/1 2/1/1.
    # illegal statement: f 4/1/1 3//1 2//1.
    if len(face_normals) > 0:
        if not (len(face_verts) == len(face_normals)):
            raise ValueError(
                "Face %s is an illegal statement. \
                        Vertex properties are inconsistent. Line: %s"
                % (str(face), str(line))
            )
    if len(face_textures) > 0:
        if not (len(face_verts) == len(face_textures)):
            raise ValueError(
                "Face %s is an illegal statement. \
                        Vertex properties are inconsistent. Line: %s"
                % (str(face), str(line))
            )

    # Subdivide faces with more than 3 vertices. See comments of the
    # load_obj function for more details.
    for i in range(len(face_verts) - 2):
        faces_verts_idx.append(
            (face_verts[0], face_verts[i + 1], face_verts[i + 2])
        )
        if len(face_normals) > 0:
            faces_normals_idx.append(
                (face_normals[0], face_normals[i + 1], face_normals[i + 2])
            )
        if len(face_textures) > 0:
            faces_textures_idx.append(
                (face_textures[0], face_textures[i + 1], face_textures[i + 2])
            )
        faces_materials_idx.append(material_idx)


def _load(f_obj, data_dir, load_textures=True):
    """
    Load a mesh from a file-like object. See load_obj function more details.
    Any material files associated with the obj are expected to be in the
    directory given by data_dir.
    """
    lines = [line.strip() for line in f_obj]
    verts = []
    normals = []
    verts_uvs = []
    faces_verts_idx = []
    faces_normals_idx = []
    faces_textures_idx = []
    material_names = []
    faces_materials_idx = []
    f_mtl = None
    materials_idx = -1

    # startswith expects each line to be a string. If the file is read in as
    # bytes then first decode to strings.
    if isinstance(lines[0], bytes):
        lines = [l.decode("utf-8") for l in lines]

    for line in lines:
        if line.startswith("mtllib"):
            if len(line.split()) < 2:
                raise ValueError("material file name is not specified")
            # NOTE: this assumes only one mtl file per .obj.
            f_mtl = os.path.join(data_dir, line.split()[1])
        elif len(line.split()) != 0 and line.split()[0] == "usemtl":
            material_name = line.split()[1]
            material_names.append(material_name)
            materials_idx = len(material_names) - 1
        elif line.startswith("v "):
            # Line is a vertex.
            vert = [float(x) for x in line.split()[1:4]]
            if len(vert) != 3:
                msg = "Vertex %s does not have 3 values. Line: %s"
                raise ValueError(msg % (str(vert), str(line)))
            verts.append(vert)
        elif line.startswith("vt "):
            # Line is a texture.
            tx = [float(x) for x in line.split()[1:3]]
            if len(tx) != 2:
                raise ValueError(
                    "Texture %s does not have 2 values. Line: %s"
                    % (str(tx), str(line))
                )
            verts_uvs.append(tx)
        elif line.startswith("vn "):
            # Line is a normal.
            norm = [float(x) for x in line.split()[1:4]]
            if len(norm) != 3:
                msg = "Normal %s does not have 3 values. Line: %s"
                raise ValueError(msg % (str(norm), str(line)))
            normals.append(norm)
        elif line.startswith("f "):
            # Line is a face.
            _parse_face(
                line,
                materials_idx,
                faces_verts_idx,
                faces_normals_idx,
                faces_textures_idx,
                faces_materials_idx,
            )

    verts = torch.tensor(verts)  # (V, 3)
    normals = torch.tensor(normals)  # (N, 3)
    verts_uvs = torch.tensor(verts_uvs)  # (T, 3)

    faces_verts_idx = _format_faces_indices(faces_verts_idx, verts.shape[0])

    # Repeat for normals and textures if present.
    if len(faces_normals_idx) > 0:
        faces_normals_idx = _format_faces_indices(
            faces_normals_idx, normals.shape[0]
        )
    if len(faces_textures_idx) > 0:
        faces_textures_idx = _format_faces_indices(
            faces_textures_idx, verts_uvs.shape[0]
        )
    if len(faces_materials_idx) > 0:
        faces_materials_idx = torch.tensor(
            faces_materials_idx, dtype=torch.int64
        )

    # Load materials
    material_colors, texture_images = None, None
    if load_textures:
        if (len(material_names) > 0) and (f_mtl is not None):
            if os.path.isfile(f_mtl):
                material_colors, texture_images = load_mtl(
                    f_mtl, material_names, data_dir
                )
            else:
                warnings.warn(f"Mtl file does not exist: {f_mtl}")
        elif len(material_names) > 0:
            warnings.warn("No mtl file provided")

    faces = _Faces(
        verts_idx=faces_verts_idx,
        normals_idx=faces_normals_idx,
        textures_idx=faces_textures_idx,
        materials_idx=faces_materials_idx,
    )

    aux = _Aux(
        normals=normals if len(normals) > 0 else None,
        verts_uvs=verts_uvs if len(verts_uvs) > 0 else None,
        material_colors=material_colors,
        texture_images=texture_images,
    )
    return verts, faces, aux


def load_mtl(f_mtl, material_names: List, data_dir: str):
    """
    Load texture images and material reflectivity values for ambient, diffuse
    and specular light (Ka, Kd, Ks, Ns).

    Args:
        f_mtl: a file like object of the material information.
        material_names: a list of the material names found in the .obj file.
        data_dir: the directory where the material texture files are located.

    Returns:
        material_colors: dict of properties for each material. If a material
                does not have any properties it will have an emtpy dict.
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
        texture_images: dict of material names and texture images
                {
                    material_name_1: (H, W, 3) image,
                    ...
                }
    """
    texture_files = {}
    material_colors = {}
    material_properties = {}
    texture_images = {}
    material_name = ""

    f_mtl, new_f = _open_file(f_mtl)
    lines = [line.strip() for line in f_mtl]
    for line in lines:
        if len(line.split()) != 0:
            if line.split()[0] == "newmtl":
                material_name = line.split()[1]
                material_colors[material_name] = {}
            if line.split()[0] == "map_Kd":
                # Texture map.
                texture_files[material_name] = line.split()[1]
            if line.split()[0] == "Kd":
                # RGB diffuse reflectivity
                kd = np.array(list(line.split()[1:4])).astype(np.float32)
                kd = torch.from_numpy(kd)
                material_colors[material_name]["diffuse_color"] = kd
            if line.split()[0] == "Ka":
                # RGB ambient reflectivity
                ka = np.array(list(line.split()[1:4])).astype(np.float32)
                ka = torch.from_numpy(ka)
                material_colors[material_name]["ambient_color"] = ka
            if line.split()[0] == "Ks":
                # RGB specular reflectivity
                ks = np.array(list(line.split()[1:4])).astype(np.float32)
                ks = torch.from_numpy(ks)
                material_colors[material_name]["specular_color"] = ks
            if line.split()[0] == "Ns":
                # Specular exponent
                ns = np.array(list(line.split()[1:4])).astype(np.float32)
                ns = torch.from_numpy(ns)
                material_colors[material_name]["shininess"] = ns

    if new_f:
        f_mtl.close()

    # Only keep the materials referenced in the obj.
    for name in material_names:
        if name in texture_files:
            # Load the texture image.
            filename = texture_files[name]
            filename_texture = os.path.join(data_dir, filename)
            if os.path.isfile(filename_texture):
                image = _read_image(filename_texture, format="RGB") / 255.0
                image = torch.from_numpy(image)
                texture_images[name] = image
            else:
                msg = f"Texture file does not exist: {filename_texture}"
                warnings.warn(msg)

        if name in material_colors:
            material_properties[name] = material_colors[name]

    return material_properties, texture_images


def save_obj(f, verts, faces, decimal_places: Optional[int] = None):
    """
    Save a mesh to an .obj file.

    Args:
        f: File (or path) to which the mesh should be written.
        verts: FloatTensor of shape (V, 3) giving vertex coordinates.
        faces: LongTensor of shape (F, 3) giving faces.
        decimal_places: Number of decimal places for saving.
    """
    new_f = False
    if isinstance(f, str):
        new_f = True
        f = open(f, "w")
    elif isinstance(f, pathlib.Path):
        new_f = True
        f = f.open("w")
    try:
        return _save(f, verts, faces, decimal_places)
    finally:
        if new_f:
            f.close()


# TODO (nikhilar) Speed up this function.
def _save(f, verts, faces, decimal_places: Optional[int] = None):
    if verts.dim() != 2 or verts.size(1) != 3:
        raise ValueError("Argument 'verts' should be of shape (num_verts, 3).")
    if faces.dim() != 2 or faces.size(1) != 3:
        raise ValueError("Argument 'faces' should be of shape (num_faces, 3).")
    verts, faces = verts.cpu(), faces.cpu()

    if decimal_places is None:
        float_str = "%f"
    else:
        float_str = "%" + ".%df" % decimal_places

    lines = ""
    V, D = verts.shape
    for i in range(V):
        vert = [float_str % verts[i, j] for j in range(D)]
        lines += "v %s\n" % " ".join(vert)

    F, P = faces.shape
    for i in range(F):
        face = ["%d" % (faces[i, j] + 1) for j in range(P)]
        if i + 1 < F:
            lines += "f %s\n" % " ".join(face)
        elif i + 1 == F:
            # No newline at the end of the file.
            lines += "f %s" % " ".join(face)

    f.write(lines)
