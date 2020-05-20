# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


"""This module implements utility functions for loading and saving meshes."""
import pathlib
import struct
import sys
import warnings
from collections import namedtuple
from typing import Optional, Tuple

import numpy as np
import torch


_PlyTypeData = namedtuple("_PlyTypeData", "size struct_char np_type")

_PLY_TYPES = {
    "char": _PlyTypeData(1, "b", np.byte),
    "uchar": _PlyTypeData(1, "B", np.ubyte),
    "short": _PlyTypeData(2, "h", np.short),
    "ushort": _PlyTypeData(2, "H", np.ushort),
    "int": _PlyTypeData(4, "i", np.int32),
    "uint": _PlyTypeData(4, "I", np.uint32),
    "float": _PlyTypeData(4, "f", np.float32),
    "double": _PlyTypeData(8, "d", np.float64),
    "int8": _PlyTypeData(1, "b", np.byte),
    "uint8": _PlyTypeData(1, "B", np.ubyte),
    "int16": _PlyTypeData(2, "h", np.short),
    "uint16": _PlyTypeData(2, "H", np.ushort),
    "int32": _PlyTypeData(4, "i", np.int32),
    "uint32": _PlyTypeData(4, "I", np.uint32),
    "float32": _PlyTypeData(4, "f", np.float32),
    "float64": _PlyTypeData(8, "d", np.float64),
}

_Property = namedtuple("_Property", "name data_type list_size_type")


class _PlyElementType:
    """
    Description of an element of a Ply file.
    Members:
        self.properties: (List[_Property]) description of all the properties.
                            Each one contains a name and data type.
        self.count:      (int) number of such elements in the file
        self.name:       (str) name of the element
    """

    def __init__(self, name: str, count: int):
        self.name = name
        self.count = count
        self.properties = []

    def add_property(
        self, name: str, data_type: str, list_size_type: Optional[str] = None
    ):
        """Adds a new property.

        Args:
            name:           (str) name of the property.
            data_type:      (str) PLY data type.
            list_size_type: (str) PLY data type of the list size, or None if not
                            a list.
        """
        for property in self.properties:
            if property.name == name:
                msg = "Cannot have two properties called %s in %s."
                raise ValueError(msg % (name, self.name))
        self.properties.append(_Property(name, data_type, list_size_type))

    def is_fixed_size(self) -> bool:
        """Return whether the Element has no list properties

        Returns:
            True if none of the properties are lists.
        """
        for property in self.properties:
            if property.list_size_type is not None:
                return False
        return True

    def is_constant_type_fixed_size(self) -> bool:
        """Return whether the Element has all properties of the same non-list
        type.

        Returns:
            True if none of the properties are lists and all the properties
            share a type.
        """
        if not self.is_fixed_size():
            return False
        first_type = _PLY_TYPES[self.properties[0].data_type]
        for property in self.properties:
            if _PLY_TYPES[property.data_type] != first_type:
                return False
        return True

    def try_constant_list(self) -> bool:
        """Whether the element is just a single list, which might have a
        constant size, and therefore we could try to parse quickly with numpy.

        Returns:
            True if the only property is a list.
        """
        if len(self.properties) != 1:
            return False
        if self.properties[0].list_size_type is None:
            return False
        return True


class _PlyHeader:
    def __init__(self, f):
        """
        Load a header of a Ply file from a file-like object.
        Members:
            self.elements:   (List[_PlyElementType]) element description
            self.ascii:      (bool) Whether in ascii format
            self.big_endian: (bool) (if not ascii) whether big endian
            self.obj_info:   (dict) arbitrary extra data

        Args:
            f: file-like object.
        """
        if f.readline() not in [b"ply\n", b"ply\r\n", "ply\n"]:
            raise ValueError("Invalid file header.")
        seen_format = False
        self.elements = []
        self.obj_info = {}
        while True:
            line = f.readline()
            if isinstance(line, bytes):
                line = line.decode("ascii")
            line = line.strip()
            if line == "end_header":
                if not self.elements:
                    raise ValueError("No elements found.")
                if not self.elements[-1].properties:
                    raise ValueError("Found an element with no properties.")
                if not seen_format:
                    raise ValueError("No format line found.")
                break
            if not seen_format:
                if line == "format ascii 1.0":
                    seen_format = True
                    self.ascii = True
                    continue
                if line == "format binary_little_endian 1.0":
                    seen_format = True
                    self.ascii = False
                    self.big_endian = False
                    continue
                if line == "format binary_big_endian 1.0":
                    seen_format = True
                    self.ascii = False
                    self.big_endian = True
                    continue
            if line.startswith("format"):
                raise ValueError("Invalid format line.")
            if line.startswith("comment") or len(line) == 0:
                continue
            if line.startswith("element"):
                self._parse_element(line)
                continue
            if line.startswith("obj_info"):
                items = line.split(" ")
                if len(items) != 3:
                    raise ValueError("Invalid line: %s" % line)
                self.obj_info[items[1]] = items[2]
                continue
            if line.startswith("property"):
                self._parse_property(line)
                continue
            raise ValueError("Invalid line: %s." % line)

    def _parse_property(self, line: str):
        """
        Decode a ply file header property line.

        Args:
            line: (str) the ply file's line.
        """
        if not self.elements:
            raise ValueError("Encountered property before any element.")
        items = line.split(" ")
        if len(items) not in [3, 5]:
            raise ValueError("Invalid line: %s" % line)
        datatype = items[1]
        name = items[-1]
        if datatype == "list":
            datatype = items[3]
            list_size_type = items[2]
            if list_size_type not in _PLY_TYPES:
                raise ValueError("Invalid datatype: %s" % list_size_type)
        else:
            list_size_type = None
        if datatype not in _PLY_TYPES:
            raise ValueError("Invalid datatype: %s" % datatype)
        self.elements[-1].add_property(name, datatype, list_size_type)

    def _parse_element(self, line: str):
        """
        Decode a ply file header element line.

        Args:
            line: (str) the ply file's line.
        """
        if self.elements and not self.elements[-1].properties:
            raise ValueError("Found an element with no properties.")
        items = line.split(" ")
        if len(items) != 3:
            raise ValueError("Invalid line: %s" % line)
        try:
            count = int(items[2])
        except ValueError:
            msg = "Number of items for %s was not a number."
            raise ValueError(msg % items[1])
        self.elements.append(_PlyElementType(items[1], count))


def _make_tensor(data, cols: int, dtype: torch.dtype) -> torch.Tensor:
    """
    Return a 2D tensor with the specified cols and dtype filled with data,
    even when data is empty.
    """
    if not len(data):
        return torch.zeros((0, cols), dtype=dtype)

    return torch.tensor(data, dtype=dtype)


def _read_ply_fixed_size_element_ascii(f, definition: _PlyElementType):
    """
    Given an element which has no lists and one type, read the
    corresponding data.

    Args:
        f: file-like object being read.
        definition: The element object which describes what we are reading.

    Returns:
        2D numpy array corresponding to the data. The rows are the different
        values. There is one column for each property.
    """
    np_type = _PLY_TYPES[definition.properties[0].data_type].np_type
    old_offset = f.tell()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=".* Empty input file.*", category=UserWarning
        )
        data = np.loadtxt(
            f, dtype=np_type, comments=None, ndmin=2, max_rows=definition.count
        )
    if not len(data):  # np.loadtxt() seeks even on empty data
        f.seek(old_offset)
    if definition.count and data.shape[1] != len(definition.properties):
        raise ValueError("Inconsistent data for %s." % definition.name)
    if data.shape[0] != definition.count:
        raise ValueError("Not enough data for %s." % definition.name)
    return data


def _try_read_ply_constant_list_ascii(f, definition: _PlyElementType):
    """
    If definition is an element which is a single list, attempt to read the
    corresponding data assuming every value has the same length.
    If the data is ragged, return None and leave f undisturbed.

    Args:
        f: file-like object being read.
        definition: The element object which describes what we are reading.

    Returns:
        If every element has the same size, 2D numpy array corresponding to the
        data. The rows are the different values. Otherwise None.
    """
    np_type = _PLY_TYPES[definition.properties[0].data_type].np_type
    old_offset = f.tell()
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=".* Empty input file.*", category=UserWarning
            )
            data = np.loadtxt(
                f, dtype=np_type, comments=None, ndmin=2, max_rows=definition.count
            )
    except ValueError:
        f.seek(old_offset)
        return None
    if not len(data):  # np.loadtxt() seeks even on empty data
        f.seek(old_offset)
    if (data.shape[1] - 1 != data[:, 0]).any():
        msg = "A line of %s data did not have the specified length."
        raise ValueError(msg % definition.name)
    if data.shape[0] != definition.count:
        raise ValueError("Not enough data for %s." % definition.name)
    return data[:, 1:]


def _parse_heterogenous_property_ascii(datum, line_iter, property: _Property):
    """
    Read a general data property from an ascii .ply file.

    Args:
        datum: list to append the single value to. That value will be a numpy
                array if the property is a list property, otherwise an int or
                float.
        line_iter: iterator to words on the line from which we read.
        property: the property object describing the property we are reading.
    """
    value = next(line_iter, None)
    if value is None:
        raise ValueError("Too little data for an element.")
    if property.list_size_type is None:
        try:
            if property.data_type in ["double", "float"]:
                datum.append(float(value))
            else:
                datum.append(int(value))
        except ValueError:
            raise ValueError("Bad numerical data.")
    else:
        try:
            length = int(value)
        except ValueError:
            raise ValueError("A list length was not a number.")
        list_value = np.zeros(length, dtype=_PLY_TYPES[property.data_type].np_type)
        for i in range(length):
            inner_value = next(line_iter, None)
            if inner_value is None:
                raise ValueError("Too little data for an element.")
            try:
                list_value[i] = float(inner_value)
            except ValueError:
                raise ValueError("Bad numerical data.")
        datum.append(list_value)


def _read_ply_element_ascii(f, definition: _PlyElementType):
    """
    Decode all instances of a single element from an ascii .ply file.

    Args:
        f: file-like object being read.
        definition: The element object which describes what we are reading.

    Returns:
        In simple cases where every element has the same size, 2D numpy array
        corresponding to the data. The rows are the different values.
        Otherwise a list of lists of values, where the outer list is
        each occurence of the element, and the inner lists have one value per
        property.
    """
    if definition.is_constant_type_fixed_size():
        return _read_ply_fixed_size_element_ascii(f, definition)
    if definition.try_constant_list():
        data = _try_read_ply_constant_list_ascii(f, definition)
        if data is not None:
            return data

    # We failed to read the element as a lump, must process each line manually.
    data = []
    for _i in range(definition.count):
        line_string = f.readline()
        if line_string == "":
            raise ValueError("Not enough data for %s." % definition.name)
        datum = []
        line_iter = iter(line_string.strip().split())
        for property in definition.properties:
            _parse_heterogenous_property_ascii(datum, line_iter, property)
        data.append(datum)
        if next(line_iter, None) is not None:
            raise ValueError("Too much data for an element.")
    return data


def _read_ply_fixed_size_element_binary(
    f, definition: _PlyElementType, big_endian: bool
):
    """
    Given an element which has no lists and one type, read the
    corresponding data.

    Args:
        f: file-like object being read.
        definition: The element object which describes what we are reading.
        big_endian: (bool) whether the document is encoded as big endian.

    Returns:
        2D numpy array corresponding to the data. The rows are the different
        values. There is one column for each property.
    """
    ply_type = _PLY_TYPES[definition.properties[0].data_type]
    np_type = ply_type.np_type
    type_size = ply_type.size
    needed_length = definition.count * len(definition.properties)
    needed_bytes = needed_length * type_size
    bytes_data = f.read(needed_bytes)
    if len(bytes_data) != needed_bytes:
        raise ValueError("Not enough data for %s." % definition.name)
    data = np.frombuffer(bytes_data, dtype=np_type)

    if (sys.byteorder == "big") != big_endian:
        data = data.byteswap()
    return data.reshape(definition.count, len(definition.properties))


def _read_ply_element_struct(f, definition: _PlyElementType, endian_str: str):
    """
    Given an element which has no lists, read the corresponding data. Uses the
    struct library.

    Note: It looks like struct would also support lists where
    type=size_type=char, but it is hard to know how much data to read in that
    case.

    Args:
        f: file-like object being read.
        definition: The element object which describes what we are reading.
        endian_str: ">" or "<" according to whether the document is big or
                    little endian.

    Returns:
        2D numpy array corresponding to the data. The rows are the different
        values. There is one column for each property.
    """
    format = "".join(
        _PLY_TYPES[property.data_type].struct_char for property in definition.properties
    )
    format = endian_str + format
    pattern = struct.Struct(format)
    size = pattern.size
    needed_bytes = size * definition.count
    bytes_data = f.read(needed_bytes)
    if len(bytes_data) != needed_bytes:
        raise ValueError("Not enough data for %s." % definition.name)
    data = [pattern.unpack_from(bytes_data, i * size) for i in range(definition.count)]
    return data


def _try_read_ply_constant_list_binary(
    f, definition: _PlyElementType, big_endian: bool
):
    """
    If definition is an element which is a single list, attempt to read the
    corresponding data assuming every value has the same length.
    If the data is ragged, return None and leave f undisturbed.

    Args:
        f: file-like object being read.
        definition: The element object which describes what we are reading.
        big_endian: (bool) whether the document is encoded as big endian.

    Returns:
        If every element has the same size, 2D numpy array corresponding to the
        data. The rows are the different values. Otherwise None.
    """
    property = definition.properties[0]
    endian_str = ">" if big_endian else "<"
    length_format = endian_str + _PLY_TYPES[property.list_size_type].struct_char
    length_struct = struct.Struct(length_format)

    def get_length():
        bytes_data = f.read(length_struct.size)
        if len(bytes_data) != length_struct.size:
            raise ValueError("Not enough data for %s." % definition.name)
        [length] = length_struct.unpack(bytes_data)
        return length

    old_offset = f.tell()

    length = get_length()
    np_type = _PLY_TYPES[definition.properties[0].data_type].np_type
    type_size = _PLY_TYPES[definition.properties[0].data_type].size
    data_size = type_size * length

    output = np.zeros((definition.count, length), dtype=np_type)

    for i in range(definition.count):
        bytes_data = f.read(data_size)
        if len(bytes_data) != data_size:
            raise ValueError("Not enough data for %s" % definition.name)
        output[i] = np.frombuffer(bytes_data, dtype=np_type)
        if i + 1 == definition.count:
            break
        if length != get_length():
            f.seek(old_offset)
            return None
    if (sys.byteorder == "big") != big_endian:
        output = output.byteswap()

    return output


def _read_ply_element_binary(f, definition: _PlyElementType, big_endian: bool) -> list:
    """
    Decode all instances of a single element from a binary .ply file.

    Args:
        f: file-like object being read.
        definition: The element object which describes what we are reading.
        big_endian: (bool) whether the document is encoded as big endian.

    Returns:
        In simple cases where every element has the same size, 2D numpy array
        corresponding to the data. The rows are the different values.
        Otherwise a list of lists/tuples of values, where the outer list is
        each occurence of the element, and the inner lists have one value per
        property.
    """
    endian_str = ">" if big_endian else "<"

    if definition.is_constant_type_fixed_size():
        return _read_ply_fixed_size_element_binary(f, definition, big_endian)
    if definition.is_fixed_size():
        return _read_ply_element_struct(f, definition, endian_str)
    if definition.try_constant_list():
        data = _try_read_ply_constant_list_binary(f, definition, big_endian)
        if data is not None:
            return data

    # We failed to read the element as a lump, must process each line manually.
    property_structs = []
    for property in definition.properties:
        initial_type = property.list_size_type or property.data_type
        property_structs.append(
            struct.Struct(endian_str + _PLY_TYPES[initial_type].struct_char)
        )

    data = []
    for _i in range(definition.count):
        datum = []
        for property, property_struct in zip(definition.properties, property_structs):
            size = property_struct.size
            initial_data = f.read(size)
            if len(initial_data) != size:
                raise ValueError("Not enough data for %s" % definition.name)
            [initial] = property_struct.unpack(initial_data)
            if property.list_size_type is None:
                datum.append(initial)
            else:
                type_size = _PLY_TYPES[property.data_type].size
                needed_bytes = type_size * initial
                list_data = f.read(needed_bytes)
                if len(list_data) != needed_bytes:
                    raise ValueError("Not enough data for %s" % definition.name)
                np_type = _PLY_TYPES[property.data_type].np_type
                list_np = np.frombuffer(list_data, dtype=np_type)
                if (sys.byteorder == "big") != big_endian:
                    list_np = list_np.byteswap()
                datum.append(list_np)
        data.append(datum)
    return data


def _load_ply_raw_stream(f) -> Tuple[_PlyHeader, dict]:
    """
    Implementation for _load_ply_raw which takes a stream.

    Args:
        f:  A binary or text file-like object.

    Returns:
        header: A _PlyHeader object describing the metadata in the ply file.
        elements: A dictionary of element names to values. If an element is regular, in
        the sense of having no lists or being one uniformly-sized list, then the
        value will be a 2D numpy array. If not, it is a list of the relevant
        property values.
    """

    header = _PlyHeader(f)
    elements = {}
    if header.ascii:
        for element in header.elements:
            elements[element.name] = _read_ply_element_ascii(f, element)
    else:
        big = header.big_endian
        for element in header.elements:
            elements[element.name] = _read_ply_element_binary(f, element, big)
    end = f.read().strip()
    if len(end) != 0:
        raise ValueError("Extra data at end of file: " + str(end[:20]))
    return header, elements


def _load_ply_raw(f) -> Tuple[_PlyHeader, dict]:
    """
    Load the data from a .ply file.

    Args:
        f:  A binary or text file-like object (with methods read, readline,
            tell and seek), a pathlib path or a string containing a file name.
            If the ply file is binary, a text stream is not supported.
            It is recommended to use a binary stream.

    Returns:
        header: A _PlyHeader object describing the metadata in the ply file.
        elements: A dictionary of element names to values. If an element is
                  regular, in the sense of having no lists or being one
                  uniformly-sized list, then the value will be a 2D numpy array.
                  If not, it is a list of the relevant property values.
    """
    new_f = False
    if isinstance(f, str):
        new_f = True
        f = open(f, "rb")
    elif isinstance(f, pathlib.Path):
        new_f = True
        f = f.open("rb")
    try:
        header, elements = _load_ply_raw_stream(f)
    finally:
        if new_f:
            f.close()

    return header, elements


def load_ply(f):
    """
    Load the data from a .ply file.

    Example .ply file format:

    ply
    format ascii 1.0           { ascii/binary, format version number }
    comment made by Greg Turk  { comments keyword specified, like all lines }
    comment this file is a cube
    element vertex 8           { define "vertex" element, 8 of them in file }
    property float x           { vertex contains float "x" coordinate }
    property float y           { y coordinate is also a vertex property }
    property float z           { z coordinate, too }
    element face 6             { there are 6 "face" elements in the file }
    property list uchar int vertex_index { "vertex_indices" is a list of ints }
    end_header                 { delimits the end of the header }
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
        f:  A binary or text file-like object (with methods read, readline,
            tell and seek), a pathlib path or a string containing a file name.
            If the ply file is in the binary ply format rather than the text
            ply format, then a text stream is not supported.
            It is easiest to use a binary stream in all cases.

    Returns:
        verts: FloatTensor of shape (V, 3).
        faces: LongTensor of vertex indices, shape (F, 3).
    """
    header, elements = _load_ply_raw(f)

    vertex = elements.get("vertex", None)
    if vertex is None:
        raise ValueError("The ply file has no vertex element.")

    face = elements.get("face", None)
    if face is None:
        raise ValueError("The ply file has no face element.")

    if len(vertex) and (
        not isinstance(vertex, np.ndarray) or vertex.ndim != 2 or vertex.shape[1] != 3
    ):
        raise ValueError("Invalid vertices in file.")
    verts = _make_tensor(vertex, cols=3, dtype=torch.float32)

    face_head = next(head for head in header.elements if head.name == "face")
    if len(face_head.properties) != 1 or face_head.properties[0].list_size_type is None:
        raise ValueError("Unexpected form of faces data.")
    # face_head.properties[0].name is usually "vertex_index" or "vertex_indices"
    # but we don't need to enforce this.

    if not len(face):
        faces = torch.zeros(size=(0, 3), dtype=torch.int64)
    elif isinstance(face, np.ndarray) and face.ndim == 2:  # Homogeneous elements
        if face.shape[1] < 3:
            raise ValueError("Faces must have at least 3 vertices.")
        face_arrays = [face[:, [0, i + 1, i + 2]] for i in range(face.shape[1] - 2)]
        faces = torch.LongTensor(np.vstack(face_arrays))
    else:
        face_list = []
        for face_item in face:
            if face_item.ndim != 1:
                raise ValueError("Bad face data.")
            if face_item.shape[0] < 3:
                raise ValueError("Faces must have at least 3 vertices.")
            for i in range(face_item.shape[0] - 2):
                face_list.append([face_item[0], face_item[i + 1], face_item[i + 2]])
        faces = _make_tensor(face_list, cols=3, dtype=torch.int64)

    if torch.any(faces >= verts.shape[0]) or torch.any(faces < 0):
        warnings.warn("Faces have invalid indices")

    return verts, faces


def _save_ply(
    f,
    verts: torch.Tensor,
    faces: torch.LongTensor,
    verts_normals: torch.Tensor,
    decimal_places: Optional[int] = None,
) -> None:
    """
    Internal implementation for saving 3D data to a .ply file.

    Args:
        f: File object to which the 3D data should be written.
        verts: FloatTensor of shape (V, 3) giving vertex coordinates.
        faces: LongTensor of shsape (F, 3) giving faces.
        verts_normals: FloatTensor of shape (V, 3) giving vertex normals.
        decimal_places: Number of decimal places for saving.
    """
    assert not len(verts) or (verts.dim() == 2 and verts.size(1) == 3)
    assert not len(faces) or (faces.dim() == 2 and faces.size(1) == 3)
    assert not len(verts_normals) or (
        verts_normals.dim() == 2 and verts_normals.size(1) == 3
    )

    print("ply\nformat ascii 1.0", file=f)
    print(f"element vertex {verts.shape[0]}", file=f)
    print("property float x", file=f)
    print("property float y", file=f)
    print("property float z", file=f)
    if verts_normals.numel() > 0:
        print("property float nx", file=f)
        print("property float ny", file=f)
        print("property float nz", file=f)
    print(f"element face {faces.shape[0]}", file=f)
    print("property list uchar int vertex_index", file=f)
    print("end_header", file=f)

    if not (len(verts) or len(faces)):
        warnings.warn("Empty 'verts' and 'faces' arguments provided")
        return

    if decimal_places is None:
        float_str = "%f"
    else:
        float_str = "%" + ".%df" % decimal_places

    vert_data = torch.cat((verts, verts_normals), dim=1)
    np.savetxt(f, vert_data.detach().numpy(), float_str)

    faces_array = faces.detach().numpy()

    if torch.any(faces >= verts.shape[0]) or torch.any(faces < 0):
        warnings.warn("Faces have invalid indices")

    if len(faces_array):
        np.savetxt(f, faces_array, "3 %d %d %d")


def save_ply(
    f,
    verts: torch.Tensor,
    faces: Optional[torch.LongTensor] = None,
    verts_normals: Optional[torch.Tensor] = None,
    decimal_places: Optional[int] = None,
) -> None:
    """
    Save a mesh to a .ply file.

    Args:
        f: File (or path) to which the mesh should be written.
        verts: FloatTensor of shape (V, 3) giving vertex coordinates.
        faces: LongTensor of shape (F, 3) giving faces.
        verts_normals: FloatTensor of shape (V, 3) giving vertex normals.
        decimal_places: Number of decimal places for saving.
    """

    verts_normals = torch.FloatTensor([]) if verts_normals is None else verts_normals
    faces = torch.LongTensor([]) if faces is None else faces

    if len(verts) and not (verts.dim() == 2 and verts.size(1) == 3):
        message = "Argument 'verts' should either be empty or of shape (num_verts, 3)."
        raise ValueError(message)

    if len(faces) and not (faces.dim() == 2 and faces.size(1) == 3):
        message = "Argument 'faces' should either be empty or of shape (num_faces, 3)."
        raise ValueError(message)

    if len(verts_normals) and not (
        verts_normals.dim() == 2
        and verts_normals.size(1) == 3
        and verts_normals.size(0) == verts.size(0)
    ):
        message = "Argument 'verts_normals' should either be empty or of shape (num_verts, 3)."
        raise ValueError(message)

    new_f = False
    if isinstance(f, str):
        new_f = True
        f = open(f, "w")
    elif isinstance(f, pathlib.Path):
        new_f = True
        f = f.open("w")
    try:
        _save_ply(f, verts, faces, verts_normals, decimal_places)
    finally:
        if new_f:
            f.close()
