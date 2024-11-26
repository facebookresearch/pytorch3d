# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe


"""
This module implements utility functions for loading and saving
meshes as .off files.

This format is introduced, for example, at
http://www.geomview.org/docs/html/OFF.html .
"""

import warnings
from typing import cast, Optional, Tuple, Union

import numpy as np
import torch
from iopath.common.file_io import PathManager
from pytorch3d.io.utils import _check_faces_indices, _open_file, PathOrStr
from pytorch3d.renderer import TexturesAtlas, TexturesVertex
from pytorch3d.structures import Meshes

from .pluggable_formats import endswith, MeshFormatInterpreter


def _is_line_empty(line: Union[str, bytes]) -> bool:
    """
    Returns whether line is not relevant in an OFF file.
    """
    line = line.strip()
    return len(line) == 0 or line[:1] == b"#"


def _count_next_line_periods(file) -> int:
    """
    Returns the number of . characters before any # on the next
    meaningful line.
    """
    old_offset = file.tell()
    line = file.readline()
    while _is_line_empty(line):
        line = file.readline()
        if len(line) == 0:
            raise ValueError("Premature end of file")

    contents = line.split(b"#")[0]
    count = contents.count(b".")
    file.seek(old_offset)
    return count


def _read_faces_lump(
    file, n_faces: int, n_colors: Optional[int]
) -> Optional[Tuple[np.ndarray, int, Optional[np.ndarray]]]:
    """
    Parse n_faces faces and faces_colors from the file,
    if they all have the same number of vertices.
    This is used in two ways.
    1) To try to read all faces.
    2) To read faces one-by-one if that failed.

    Args:
        file: file-like object being read.
        n_faces: The known number of faces yet to read.
        n_colors: The number of colors if known already.

    Returns:
        - 2D numpy array of faces
        - number of colors found
        - 2D numpy array of face colors if found.
        of None if there are faces with different numbers of vertices.
    """
    if n_faces == 0:
        return np.array([[]]), 0, None
    old_offset = file.tell()
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=".* Empty input file.*", category=UserWarning
            )
            data = np.loadtxt(file, dtype=np.float32, ndmin=2, max_rows=n_faces)
    except ValueError as e:
        if n_faces > 1 and "number of columns" in e.args[0]:
            file.seek(old_offset)
            return None
        raise ValueError("Not enough face data.") from None

    if len(data) != n_faces:
        raise ValueError("Not enough face data.")
    face_size = int(data[0, 0])
    if (data[:, 0] != face_size).any():
        msg = "A line of face data did not have the specified length."
        raise ValueError(msg)
    if face_size < 3:
        raise ValueError("Faces must have at least 3 vertices.")

    n_colors_found = data.shape[1] - 1 - face_size
    if n_colors is not None and n_colors_found != n_colors:
        raise ValueError("Number of colors differs between faces.")
    n_colors = n_colors_found
    if n_colors not in [0, 3, 4]:
        raise ValueError("Unexpected number of colors.")

    face_raw_data = data[:, 1 : 1 + face_size].astype("int64")
    if face_size == 3:
        face_data = face_raw_data
    else:
        face_arrays = [
            face_raw_data[:, [0, i + 1, i + 2]] for i in range(face_size - 2)
        ]
        face_data = np.vstack(face_arrays)

    if n_colors == 0:
        return face_data, 0, None
    colors = data[:, 1 + face_size :]
    if face_size == 3:
        return face_data, n_colors, colors
    return face_data, n_colors, np.tile(colors, (face_size - 2, 1))


def _read_faces(
    file, n_faces: int
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Returns faces and face colors from the file.

    Args:
        file: file-like object being read.
        n_faces: The known number of faces.

    Returns:
        2D numpy arrays of faces and face colors, or None for each if
            they are not present.
    """
    if n_faces == 0:
        return None, None

    color_is_int = 0 == _count_next_line_periods(file)
    color_scale = 1 / 255.0 if color_is_int else 1

    faces_ncolors_colors = _read_faces_lump(file, n_faces=n_faces, n_colors=None)
    if faces_ncolors_colors is not None:
        faces, _, colors = faces_ncolors_colors
        if colors is None:
            return faces, None
        return faces, colors * color_scale

    faces_list, colors_list = [], []
    n_colors = None
    for _ in range(n_faces):
        faces_ncolors_colors = _read_faces_lump(file, n_faces=1, n_colors=n_colors)
        faces_found, n_colors, colors_found = cast(
            Tuple[np.ndarray, int, Optional[np.ndarray]], faces_ncolors_colors
        )
        faces_list.append(faces_found)
        colors_list.append(colors_found)
    faces = np.vstack(faces_list)
    if n_colors == 0:
        colors = None
    else:
        colors = np.vstack(colors_list) * color_scale
    return faces, colors


def _read_verts(file, n_verts: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Returns verts and vertex colors from the file.

    Args:
        file: file-like object being read.
        n_verts: The known number of faces.

    Returns:
        2D numpy arrays of verts and (if present)
        vertex colors.
    """

    color_is_int = 3 == _count_next_line_periods(file)
    color_scale = 1 / 255.0 if color_is_int else 1

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=".* Empty input file.*", category=UserWarning
        )
        data = np.loadtxt(file, dtype=np.float32, ndmin=2, max_rows=n_verts)
    if data.shape[0] != n_verts:
        raise ValueError("Not enough vertex data.")
    if data.shape[1] not in [3, 6, 7]:
        raise ValueError("Bad vertex data.")

    if data.shape[1] == 3:
        return data, None
    return data[:, :3], data[:, 3:] * color_scale  # []


def _load_off_stream(file) -> dict:
    """
    Load the data from a stream of an .off file.

    Example .off file format:

    off
    8 6 1927                   { number of vertices, faces, and (not used) edges }
    # comment                  { comments with # sign }
    0 0 0                      { start of vertex list }
    0 0 1
    0 1 1
    0 1 0
    1 0 0
    1 0 1
    1 1 1
    1 1 0
    4 0 1 2 3                  { start of face list }
    4 7 6 5 4
    4 0 4 5 1
    4 1 5 6 2
    4 2 6 7 3
    4 3 7 4 0

    Args:
        file:  A binary file-like object (with methods read, readline,
            tell and seek).

    Returns dictionary possibly containing:
        verts: (always present) FloatTensor of shape (V, 3).
        verts_colors: FloatTensor of shape (V, C) where C is 3 or 4.
        faces: LongTensor of vertex indices, split into triangles, shape (F, 3).
        faces_colors: FloatTensor of shape (F, C), where C is 3 or 4.
    """
    header = file.readline()

    while _is_line_empty(header):
        header = file.readline()

    if header[:3].lower() == b"off":
        header = header[3:]

    while _is_line_empty(header):
        header = file.readline()

    items = header.split()
    if len(items) < 3:
        raise ValueError("Invalid counts line: %s" % header)

    try:
        n_verts = int(items[0])
    except ValueError:
        raise ValueError("Invalid counts line: %s" % header) from None
    try:
        n_faces = int(items[1])
    except ValueError:
        raise ValueError("Invalid counts line: %s" % header) from None

    if (len(items) > 3 and not items[3].startswith(b"#")) or n_verts < 0 or n_faces < 0:
        raise ValueError("Invalid counts line: %s" % header)

    verts, verts_colors = _read_verts(file, n_verts)
    faces, faces_colors = _read_faces(file, n_faces)

    end = file.read().strip()
    if len(end) != 0:
        raise ValueError("Extra data at end of file: " + str(end[:20]))

    out = {"verts": verts}
    if verts_colors is not None:
        out["verts_colors"] = verts_colors
    if faces is not None:
        out["faces"] = faces
    if faces_colors is not None:
        out["faces_colors"] = faces_colors
    return out


def _write_off_data(
    file,
    verts: torch.Tensor,
    verts_colors: Optional[torch.Tensor] = None,
    faces: Optional[torch.LongTensor] = None,
    faces_colors: Optional[torch.Tensor] = None,
    decimal_places: Optional[int] = None,
) -> None:
    """
    Internal implementation for saving 3D data to a .off file.

    Args:
        file: Binary file object to which the 3D data should be written.
        verts: FloatTensor of shape (V, 3) giving vertex coordinates.
        verts_colors: FloatTensor of shape (V, C) giving vertex colors where C is 3 or 4.
        faces: LongTensor of shape (F, 3) giving faces.
        faces_colors: FloatTensor of shape (V, C) giving face colors where C is 3 or 4.
        decimal_places: Number of decimal places for saving.
    """
    nfaces = 0 if faces is None else faces.shape[0]
    file.write(f"off\n{verts.shape[0]} {nfaces} 0\n".encode("ascii"))

    if verts_colors is not None:
        verts = torch.cat((verts, verts_colors), dim=1)
    if decimal_places is None:
        float_str = "%f"
    else:
        float_str = "%" + ".%df" % decimal_places
    np.savetxt(file, verts.cpu().detach().numpy(), float_str)

    if faces is not None:
        _check_faces_indices(faces, max_index=verts.shape[0])

    if faces_colors is not None:
        face_data = torch.cat(
            [
                cast(torch.Tensor, faces).cpu().to(torch.float64),
                faces_colors.detach().cpu().to(torch.float64),
            ],
            dim=1,
        )
        format = "3 %d %d %d" + " %f" * faces_colors.shape[1]
        np.savetxt(file, face_data.numpy(), format)
    elif faces is not None:
        np.savetxt(file, faces.cpu().detach().numpy(), "3 %d %d %d")


def _save_off(
    file,
    *,
    verts: torch.Tensor,
    verts_colors: Optional[torch.Tensor] = None,
    faces: Optional[torch.LongTensor] = None,
    faces_colors: Optional[torch.Tensor] = None,
    decimal_places: Optional[int] = None,
    path_manager: PathManager,
) -> None:
    """
    Save a mesh to an ascii .off file.

    Args:
        file: File (or path) to which the mesh should be written.
        verts: FloatTensor of shape (V, 3) giving vertex coordinates.
        verts_colors: FloatTensor of shape (V, C) giving vertex colors where C is 3 or 4.
        faces: LongTensor of shape (F, 3) giving faces.
        faces_colors: FloatTensor of shape (V, C) giving face colors where C is 3 or 4.
        decimal_places: Number of decimal places for saving.
    """
    if len(verts) and not (verts.dim() == 2 and verts.size(1) == 3):
        message = "Argument 'verts' should either be empty or of shape (num_verts, 3)."
        raise ValueError(message)

    if verts_colors is not None and 0 == len(verts_colors):
        verts_colors = None
    if faces_colors is not None and 0 == len(faces_colors):
        faces_colors = None
    if faces is not None and 0 == len(faces):
        faces = None

    if verts_colors is not None:
        if not (verts_colors.dim() == 2 and verts_colors.size(1) in [3, 4]):
            message = "verts_colors should have shape (num_faces, C)."
            raise ValueError(message)
        if verts_colors.shape[0] != verts.shape[0]:
            message = "verts_colors should have the same length as verts."
            raise ValueError(message)

    if faces is not None and not (faces.dim() == 2 and faces.size(1) == 3):
        message = "Argument 'faces' if present should have shape (num_faces, 3)."
        raise ValueError(message)
    if faces_colors is not None and faces is None:
        message = "Cannot have face colors without faces"
        raise ValueError(message)

    if faces_colors is not None:
        if not (faces_colors.dim() == 2 and faces_colors.size(1) in [3, 4]):
            message = "faces_colors should have shape (num_faces, C)."
            raise ValueError(message)
        if faces_colors.shape[0] != cast(torch.LongTensor, faces).shape[0]:
            message = "faces_colors should have the same length as faces."
            raise ValueError(message)

    with _open_file(file, path_manager, "wb") as f:
        _write_off_data(f, verts, verts_colors, faces, faces_colors, decimal_places)


class MeshOffFormat(MeshFormatInterpreter):
    """
    Loads and saves meshes in the ascii OFF format. This is a simple
    format which can only deal with the following texture types:

    - TexturesVertex, i.e. one color for each vertex
    - TexturesAtlas with R=1, i.e. one color for each face.

    There are some possible features of OFF files which we do not support
    and which appear to be rare:

    - Four dimensional data.
    - Binary data.
    - Vertex Normals.
    - Texture coordinates.
    - "COFF" header.

    Example .off file format:

    off
    8 6 1927                   { number of vertices, faces, and (not used) edges }
    # comment                  { comments with # sign }
    0 0 0                      { start of vertex list }
    0 0 1
    0 1 1
    0 1 0
    1 0 0
    1 0 1
    1 1 1
    1 1 0
    4 0 1 2 3                  { start of face list }
    4 7 6 5 4
    4 0 4 5 1
    4 1 5 6 2
    4 2 6 7 3
    4 3 7 4 0

    """

    def __init__(self) -> None:
        self.known_suffixes = (".off",)

    def read(
        self,
        path: PathOrStr,
        include_textures: bool,
        device,
        path_manager: PathManager,
        **kwargs,
    ) -> Optional[Meshes]:
        if not endswith(path, self.known_suffixes):
            return None

        with _open_file(path, path_manager, "rb") as f:
            data = _load_off_stream(f)
        verts = torch.from_numpy(data["verts"]).to(device)
        if "faces" in data:
            faces = torch.from_numpy(data["faces"]).to(dtype=torch.int64, device=device)
        else:
            faces = torch.zeros((0, 3), dtype=torch.int64, device=device)

        textures = None
        if "verts_colors" in data:
            if "faces_colors" in data:
                msg = "Faces colors ignored because vertex colors provided too."
                warnings.warn(msg)
            verts_colors = torch.from_numpy(data["verts_colors"]).to(device)
            textures = TexturesVertex([verts_colors])
        elif "faces_colors" in data:
            faces_colors = torch.from_numpy(data["faces_colors"]).to(device)
            textures = TexturesAtlas([faces_colors[:, None, None, :]])

        mesh = Meshes(
            verts=[verts.to(device)], faces=[faces.to(device)], textures=textures
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
        if isinstance(data.textures, TexturesVertex):
            [verts_colors] = data.textures.verts_features_list()
        else:
            verts_colors = None

        faces_colors = None
        if isinstance(data.textures, TexturesAtlas):
            [atlas] = data.textures.atlas_list()
            F, R, _, D = atlas.shape
            if R == 1:
                faces_colors = atlas[:, 0, 0, :]

        _save_off(
            file=path,
            verts=verts,
            faces=faces,
            verts_colors=verts_colors,
            faces_colors=faces_colors,
            decimal_places=decimal_places,
            path_manager=path_manager,
        )
        return True
