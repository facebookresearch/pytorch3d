# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe


"""
This module implements loading meshes from glTF 2 assets stored in a
GLB container file or a glTF JSON file with embedded binary data.
It is experimental.

The module provides a MeshFormatInterpreter called
MeshGlbFormat which must be used explicitly.
e.g.

.. code-block:: python

    from pytorch3d.io import IO
    from pytorch3d.io.experimental_gltf_io import MeshGlbFormat

    io = IO()
    io.register_meshes_format(MeshGlbFormat())
    io.load_mesh(...)

This implementation is quite restricted in what it supports.

    - It does not try to validate the input against the standard.
    - It loads the default scene only.
    - Only triangulated geometry is supported.
    - The geometry of all meshes of the entire scene is aggregated into a single mesh.
      Use `load_meshes()` instead to get un-aggregated (but transformed) ones.
    - All material properties are ignored except for either vertex color, baseColorTexture
      or baseColorFactor. If available, one of these (in this order) is exclusively
      used which does not match the semantics of the standard.
"""

import json
import struct
import warnings
from base64 import b64decode
from collections import defaultdict, deque
from enum import IntEnum
from io import BytesIO
from typing import Any, BinaryIO, cast, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from iopath.common.file_io import PathManager
from PIL import Image
from pytorch3d.io.utils import _open_file, PathOrStr
from pytorch3d.renderer.mesh import TexturesBase, TexturesUV, TexturesVertex
from pytorch3d.structures import join_meshes_as_scene, Meshes
from pytorch3d.transforms import quaternion_to_matrix, Transform3d

from .pluggable_formats import endswith, MeshFormatInterpreter


_GLTF_MAGIC = 0x46546C67
_JSON_CHUNK_TYPE = 0x4E4F534A
_BINARY_CHUNK_TYPE = 0x004E4942
_DATA_URI_PREFIX = "data:application/octet-stream;base64,"


class _PrimitiveMode(IntEnum):
    POINTS = 0
    LINES = 1
    LINE_LOOP = 2
    LINE_STRIP = 3
    TRIANGLES = 4
    TRIANGLE_STRIP = 5
    TRIANGLE_FAN = 6


class _ComponentType(IntEnum):
    BYTE = 5120
    UNSIGNED_BYTE = 5121
    SHORT = 5122
    UNSIGNED_SHORT = 5123
    UNSIGNED_INT = 5125
    FLOAT = 5126


_ITEM_TYPES: Dict[int, Any] = {
    5120: np.int8,
    5121: np.uint8,
    5122: np.int16,
    5123: np.uint16,
    5125: np.uint32,
    5126: np.float32,
}


_ElementShape = Union[Tuple[int], Tuple[int, int]]
_ELEMENT_SHAPES: Dict[str, _ElementShape] = {
    "SCALAR": (1,),
    "VEC2": (2,),
    "VEC3": (3,),
    "VEC4": (4,),
    "MAT2": (2, 2),
    "MAT3": (3, 3),
    "MAT4": (4, 4),
}

_DTYPE_BYTES: Dict[Any, int] = {
    np.int8: 1,
    np.uint8: 1,
    np.int16: 2,
    np.uint16: 2,
    np.uint32: 4,
    np.float32: 4,
}


class _TargetType(IntEnum):
    ARRAY_BUFFER = 34962
    ELEMENT_ARRAY_BUFFER = 34963


class OurEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.int64):
            return str(obj)
        return super(OurEncoder, self).default(obj)


def _read_header(stream: BinaryIO) -> Optional[Tuple[int, int]]:
    header = stream.read(12)
    magic, version, length = struct.unpack("<III", header)

    if magic != _GLTF_MAGIC:
        return None
    return version, length


def _read_chunks(
    stream: BinaryIO, length: int
) -> Optional[Tuple[Dict[str, Any], np.ndarray]]:
    """
    Get the json header and the binary data from a
    GLB file.
    """
    json_data = None
    binary_data = None

    while stream.tell() < length:
        chunk_header = stream.read(8)
        chunk_length, chunk_type = struct.unpack("<II", chunk_header)
        chunk_data = stream.read(chunk_length)
        if chunk_type == _JSON_CHUNK_TYPE:
            json_data = json.loads(chunk_data)
        elif chunk_type == _BINARY_CHUNK_TYPE:
            binary_data = chunk_data
        else:
            warnings.warn("Unsupported chunk type")
            return None

    if json_data is None:
        raise ValueError("Missing json header")

    if binary_data is not None:
        binary_data = np.frombuffer(binary_data, dtype=np.uint8)

    assert binary_data is not None

    return json_data, binary_data


def _make_node_transform(node: Dict[str, Any]) -> Transform3d:
    """
    Convert a transform from the json data in to a PyTorch3D
    Transform3d format.
    """
    array = node.get("matrix")
    if array is not None:  # Stored in column-major order
        M = np.array(array, dtype=np.float32).reshape(4, 4, order="F")
        return Transform3d(matrix=torch.from_numpy(M))

    out = Transform3d()

    # Given some of (scale/rotation/translation), we do them in that order to
    # get points in to the world space.
    # See https://github.com/KhronosGroup/glTF/issues/743 .

    array = node.get("scale", None)
    if array is not None:
        scale_vector = torch.FloatTensor(array)
        out = out.scale(scale_vector[None])

    # Rotation quaternion (x, y, z, w) where w is the scalar
    array = node.get("rotation", None)
    if array is not None:
        x, y, z, w = array
        # We negate w. This is equivalent to inverting the rotation.
        # This is needed as quaternion_to_matrix makes a matrix which
        # operates on column vectors, whereas Transform3d wants a
        # matrix which operates on row vectors.
        rotation_quaternion = torch.FloatTensor([-w, x, y, z])
        rotation_matrix = quaternion_to_matrix(rotation_quaternion)
        out = out.rotate(R=rotation_matrix)

    array = node.get("translation", None)
    if array is not None:
        translation_vector = torch.FloatTensor(array)
        out = out.translate(x=translation_vector[None])

    return out


class _GLTFLoader:
    def __init__(self, stream: BinaryIO) -> None:
        self._json_data = None
        # Map from buffer index to (decoded) binary data
        self._binary_data = {}

        version_and_length = _read_header(stream)
        if version_and_length is None:  # GLTF
            stream.seek(0)
            json_data = json.load(stream)
        else:  # GLB
            version, length = version_and_length
            if version != 2:
                warnings.warn("Unsupported version")
                return
            json_and_binary_data = _read_chunks(stream, length)
            if json_and_binary_data is None:
                raise ValueError("Data not found")
            json_data, binary_data = json_and_binary_data
            self._binary_data[0] = binary_data

        self._json_data = json_data
        self._accessors = json_data.get("accessors", [])
        self._buffer_views = json_data.get("bufferViews", [])
        self._buffers = json_data.get("buffers", [])
        self._texture_map_images = {}

    def _access_image(self, image_index: int) -> np.ndarray:
        """
        Get the data for an image from the file. This is only called
        by _get_texture_map_image which caches it.
        """

        image_json = self._json_data["images"][image_index]
        buffer_view = self._buffer_views[image_json["bufferView"]]
        if "byteStride" in buffer_view:
            raise NotImplementedError("strided buffer views")

        length = buffer_view["byteLength"]
        offset = buffer_view.get("byteOffset", 0)

        binary_data = self.get_binary_data(buffer_view["buffer"])
        bytesio = BytesIO(binary_data[offset : offset + length].tobytes())
        with Image.open(bytesio) as f:
            array = np.array(f)
            if array.dtype == np.uint8:
                return array.astype(np.float32) / 255.0
            else:
                return array

    def _get_texture_map_image(self, image_index: int) -> torch.Tensor:
        """
        Return a texture map image as a torch tensor.
        Calling this function repeatedly with the same arguments returns
        the very same tensor, this allows a memory optimization to happen
        later in TexturesUV.join_scene.
        Any alpha channel is ignored.
        """
        im = self._texture_map_images.get(image_index)
        if im is not None:
            return im

        im = torch.from_numpy(self._access_image(image_index))[:, :, :3]
        self._texture_map_images[image_index] = im
        return im

    def _access_data(self, accessor_index: int) -> np.ndarray:
        """
        Get the raw data from an accessor as a numpy array.
        """
        accessor = self._accessors[accessor_index]

        buffer_view_index = accessor.get("bufferView")
        # Undefined buffer view (all zeros) are not (yet) supported
        if buffer_view_index is None:
            raise NotImplementedError("Undefined buffer view")

        accessor_byte_offset = accessor.get("byteOffset", 0)
        component_type = accessor["componentType"]
        element_count = accessor["count"]
        element_type = accessor["type"]

        # Sparse accessors are not (yet) supported
        if accessor.get("sparse") is not None:
            raise NotImplementedError("Sparse Accessors")

        buffer_view = self._buffer_views[buffer_view_index]
        buffer_index = buffer_view["buffer"]
        buffer_byte_length = buffer_view["byteLength"]
        element_byte_offset = buffer_view.get("byteOffset", 0)
        element_byte_stride = buffer_view.get("byteStride", 0)
        if element_byte_stride != 0 and element_byte_stride < 4:
            raise ValueError("Stride is too small.")
        if element_byte_stride > 252:
            raise ValueError("Stride is too big.")

        element_shape = _ELEMENT_SHAPES[element_type]
        item_type = _ITEM_TYPES[component_type]
        item_dtype = np.dtype(item_type)
        item_count = np.prod(element_shape)
        item_size = item_dtype.itemsize
        size = element_count * item_count * item_size
        if size > buffer_byte_length:
            raise ValueError("Buffer did not have enough data for the accessor")

        buffer_ = self._buffers[buffer_index]
        binary_data = self.get_binary_data(buffer_index)
        if len(binary_data) < buffer_["byteLength"]:
            raise ValueError("Not enough binary data for the buffer")

        if element_byte_stride == 0:
            element_byte_stride = item_size * item_count
        # The same buffer can store interleaved elements
        if element_byte_stride < item_size * item_count:
            raise ValueError("Items should not overlap")

        dtype = np.dtype(
            {
                "names": ["element"],
                "formats": [str(element_shape) + item_dtype.str],
                "offsets": [0],
                "itemsize": element_byte_stride,
            }
        )

        byte_offset = accessor_byte_offset + element_byte_offset
        if byte_offset % item_size != 0:
            raise ValueError("Misaligned data")
        byte_length = element_count * element_byte_stride
        buffer_view = binary_data[byte_offset : byte_offset + byte_length].view(dtype)[
            "element"
        ]

        # Convert matrix data from column-major (OpenGL) to row-major order
        if element_type in ("MAT2", "MAT3", "MAT4"):
            buffer_view = np.transpose(buffer_view, (0, 2, 1))

        return buffer_view

    def _get_primitive_attribute(
        self, primitive_attributes: Dict[str, Any], key: str, dtype
    ) -> Optional[np.ndarray]:
        accessor_index = primitive_attributes.get(key)
        if accessor_index is None:
            return None
        primitive_attribute = self._access_data(accessor_index)
        if key == "JOINTS_0":
            pass
        elif dtype == np.uint8:
            primitive_attribute /= 255.0
        elif dtype == np.uint16:
            primitive_attribute /= 65535.0
        else:
            if dtype != np.float32:
                raise ValueError("Unexpected data type")
        primitive_attribute = primitive_attribute.astype(dtype)
        return primitive_attribute

    def get_binary_data(self, buffer_index: int):
        """
        Get the binary data from a buffer as a 1D numpy array of bytes.
        This is implemented for explicit uri data buffers or the main GLB data
        segment.
        """
        buffer_ = self._buffers[buffer_index]
        binary_data = self._binary_data.get(buffer_index)
        if binary_data is None:  # Lazily decode binary data
            uri = buffer_.get("uri")
            if not uri.startswith(_DATA_URI_PREFIX):
                raise NotImplementedError("Unexpected URI type")
            binary_data = b64decode(uri[len(_DATA_URI_PREFIX) :])
            binary_data = np.frombuffer(binary_data, dtype=np.uint8)
            self._binary_data[buffer_index] = binary_data
        return binary_data

    def get_texture_for_mesh(
        self, primitive: Dict[str, Any], indices: torch.Tensor
    ) -> Optional[TexturesBase]:
        """
        Get the texture object representing the given mesh primitive.

        Args:
            primitive: the mesh primitive being loaded.
            indices: the face indices of the mesh
        """
        attributes = primitive["attributes"]
        vertex_colors = self._get_primitive_attribute(attributes, "COLOR_0", np.float32)
        if vertex_colors is not None:
            return TexturesVertex([torch.from_numpy(vertex_colors)])

        vertex_texcoords_0 = self._get_primitive_attribute(
            attributes, "TEXCOORD_0", np.float32
        )
        if vertex_texcoords_0 is not None:
            verts_uvs = torch.from_numpy(vertex_texcoords_0)
            verts_uvs[:, 1] = 1 - verts_uvs[:, -1]
            faces_uvs = indices
            material_index = primitive.get("material", 0)
            material = self._json_data["materials"][material_index]
            material_roughness = material["pbrMetallicRoughness"]
            if "baseColorTexture" in material_roughness:
                texture_index = material_roughness["baseColorTexture"]["index"]
                texture_json = self._json_data["textures"][texture_index]
                # Todo - include baseColorFactor when also given
                # Todo - look at the sampler
                image_index = texture_json["source"]
                map = self._get_texture_map_image(image_index)
            elif "baseColorFactor" in material_roughness:
                # Constant color?
                map = torch.FloatTensor(material_roughness["baseColorFactor"])[
                    None, None, :3
                ]
            texture = TexturesUV(
                # pyre-fixme[61]: `map` may not be initialized here.
                maps=[map],  # alpha channel ignored
                faces_uvs=[faces_uvs],
                verts_uvs=[verts_uvs],
            )
            return texture

        return None

    def load(self, include_textures: bool) -> List[Tuple[Optional[str], Meshes]]:
        """
        Attempt to load all the meshes making up the default scene from
        the file as a list of possibly-named Meshes objects.

        Args:
            include_textures: Whether to try loading textures.

        Returns:
            Meshes object containing one mesh.
        """
        if self._json_data is None:
            raise ValueError("Initialization problem")

        # This loads the default scene from the file.
        # This is usually the only one.
        # It is possible to have multiple scenes, in which case
        # you could choose another here instead of taking the default.
        scene_index = self._json_data.get("scene")

        if scene_index is None:
            raise ValueError("Default scene is not specified.")

        scene = self._json_data["scenes"][scene_index]
        nodes = self._json_data.get("nodes", [])
        meshes = self._json_data.get("meshes", [])
        root_node_indices = scene["nodes"]

        mesh_transform = Transform3d()
        names_meshes_list: List[Tuple[Optional[str], Meshes]] = []

        # Keep track and apply the transform of the scene node to mesh vertices
        Q = deque([(Transform3d(), node_index) for node_index in root_node_indices])

        while Q:
            parent_transform, current_node_index = Q.popleft()

            current_node = nodes[current_node_index]

            transform = _make_node_transform(current_node)
            current_transform = transform.compose(parent_transform)

            if "mesh" in current_node:
                mesh_index = current_node["mesh"]
                mesh = meshes[mesh_index]
                mesh_name = mesh.get("name", None)
                mesh_transform = current_transform

                for primitive in mesh["primitives"]:
                    attributes = primitive["attributes"]
                    accessor_index = attributes["POSITION"]
                    positions = torch.from_numpy(
                        self._access_data(accessor_index).copy()
                    )
                    positions = mesh_transform.transform_points(positions)

                    mode = primitive.get("mode", _PrimitiveMode.TRIANGLES)
                    if mode != _PrimitiveMode.TRIANGLES:
                        raise NotImplementedError("Non triangular meshes")

                    if "indices" in primitive:
                        accessor_index = primitive["indices"]
                        indices = self._access_data(accessor_index).astype(np.int64)
                    else:
                        indices = np.arange(0, len(positions), dtype=np.int64)
                    indices = torch.from_numpy(indices.reshape(-1, 3))

                    texture = None
                    if include_textures:
                        texture = self.get_texture_for_mesh(primitive, indices)

                    mesh_obj = Meshes(
                        verts=[positions], faces=[indices], textures=texture
                    )
                    names_meshes_list.append((mesh_name, mesh_obj))

            if "children" in current_node:
                children_node_indices = current_node["children"]
                Q.extend(
                    [
                        (current_transform, node_index)
                        for node_index in children_node_indices
                    ]
                )

        return names_meshes_list


def load_meshes(
    path: PathOrStr,
    path_manager: PathManager,
    include_textures: bool = True,
) -> List[Tuple[Optional[str], Meshes]]:
    """
    Loads all the meshes from the default scene in the given GLB file.
    and returns them separately.

    Args:
        path: path to read from
        path_manager: PathManager object for interpreting the path
        include_textures: whether to load textures

    Returns:
        List of (name, mesh) pairs, where the name is the optional name property
            from the GLB file, or None if it is absent, and the mesh is a Meshes
            object containing one mesh.
    """
    with _open_file(path, path_manager, "rb") as f:
        loader = _GLTFLoader(cast(BinaryIO, f))
    names_meshes_list = loader.load(include_textures=include_textures)
    return names_meshes_list


class _GLTFWriter:
    def __init__(self, data: Meshes, buffer_stream: BinaryIO) -> None:
        self._json_data = defaultdict(list)
        self.mesh = data
        self.buffer_stream = buffer_stream

        # initialize json with one scene and one node
        scene_index = 0
        # pyre-fixme[6]: Incompatible parameter type
        self._json_data["scene"] = scene_index
        self._json_data["scenes"].append({"nodes": [scene_index]})
        self._json_data["asset"] = {"version": "2.0"}
        node = {"name": "Node", "mesh": 0}
        self._json_data["nodes"].append(node)

        # mesh primitives
        meshes = defaultdict(list)
        # pyre-fixme[6]: Incompatible parameter type
        meshes["name"] = "Node-Mesh"
        if isinstance(self.mesh.textures, TexturesVertex):
            primitives = {
                "attributes": {"POSITION": 0, "COLOR_0": 2},
                "indices": 1,
                "mode": _PrimitiveMode.TRIANGLES,
            }
        elif isinstance(self.mesh.textures, TexturesUV):
            primitives = {
                "attributes": {"POSITION": 0, "TEXCOORD_0": 2},
                "indices": 1,
                "mode": _PrimitiveMode.TRIANGLES,
                "material": 0,
            }
        else:
            primitives = {
                "attributes": {"POSITION": 0},
                "indices": 1,
                "mode": _PrimitiveMode.TRIANGLES,
            }

        meshes["primitives"].append(primitives)
        self._json_data["meshes"].append(meshes)

        # default material
        material = {
            "name": "material_1",
            "pbrMetallicRoughness": {
                "baseColorTexture": {"index": 0},
                "baseColorFactor": [1, 1, 1, 1],
                "metallicFactor": 0,
                "roughnessFactor": 0.99,
            },
            "emissiveFactor": [0, 0, 0],
            "alphaMode": "OPAQUE",
        }
        self._json_data["materials"].append(material)

        # default sampler
        sampler = {"magFilter": 9729, "minFilter": 9986, "wrapS": 10497, "wrapT": 10497}
        self._json_data["samplers"].append(sampler)

        # default textures
        texture = {"sampler": 0, "source": 0}
        self._json_data["textures"].append(texture)

    def _write_accessor_json(self, key: str) -> Tuple[int, np.ndarray]:
        name = "Node-Mesh_%s" % key
        byte_offset = 0
        if key == "positions":
            data = self.mesh.verts_packed().cpu().numpy()
            component_type = _ComponentType.FLOAT
            element_type = "VEC3"
            buffer_view = 0
            element_min = list(map(float, np.min(data, axis=0)))
            element_max = list(map(float, np.max(data, axis=0)))
            byte_per_element = 3 * _DTYPE_BYTES[_ITEM_TYPES[_ComponentType.FLOAT]]
        elif key == "texcoords":
            component_type = _ComponentType.FLOAT
            data = self.mesh.textures.verts_uvs_list()[0].cpu().numpy()
            data[:, 1] = 1 - data[:, -1]  # flip y tex-coordinate
            element_type = "VEC2"
            buffer_view = 2
            element_min = list(map(float, np.min(data, axis=0)))
            element_max = list(map(float, np.max(data, axis=0)))
            byte_per_element = 2 * _DTYPE_BYTES[_ITEM_TYPES[_ComponentType.FLOAT]]
        elif key == "texvertices":
            component_type = _ComponentType.FLOAT
            data = self.mesh.textures.verts_features_list()[0].cpu().numpy()
            element_type = "VEC3"
            buffer_view = 2
            element_min = list(map(float, np.min(data, axis=0)))
            element_max = list(map(float, np.max(data, axis=0)))
            byte_per_element = 3 * _DTYPE_BYTES[_ITEM_TYPES[_ComponentType.FLOAT]]
        elif key == "indices":
            component_type = _ComponentType.UNSIGNED_SHORT
            data = (
                self.mesh.faces_packed()
                .cpu()
                .numpy()
                .astype(_ITEM_TYPES[component_type])
            )
            element_type = "SCALAR"
            buffer_view = 1
            element_min = list(map(int, np.min(data, keepdims=True)))
            element_max = list(map(int, np.max(data, keepdims=True)))
            byte_per_element = (
                3 * _DTYPE_BYTES[_ITEM_TYPES[_ComponentType.UNSIGNED_SHORT]]
            )
        else:
            raise NotImplementedError(
                "invalid key accessor, should be one of positions, indices or texcoords"
            )

        count = int(data.shape[0])
        byte_length = count * byte_per_element
        accessor_json = {
            "name": name,
            "componentType": component_type,
            "type": element_type,
            "bufferView": buffer_view,
            "byteOffset": byte_offset,
            "min": element_min,
            "max": element_max,
            "count": count * 3 if key == "indices" else count,
        }
        self._json_data["accessors"].append(accessor_json)
        return (byte_length, data)

    def _write_bufferview(self, key: str, **kwargs):
        if key not in ["positions", "texcoords", "texvertices", "indices"]:
            raise ValueError(
                "key must be one of positions, texcoords, texvertices or indices"
            )

        bufferview = {
            "name": "bufferView_%s" % key,
            "buffer": 0,
        }
        target = _TargetType.ARRAY_BUFFER
        if key == "positions":
            byte_per_element = 3 * _DTYPE_BYTES[_ITEM_TYPES[_ComponentType.FLOAT]]
            bufferview["byteStride"] = int(byte_per_element)
        elif key == "texcoords":
            byte_per_element = 2 * _DTYPE_BYTES[_ITEM_TYPES[_ComponentType.FLOAT]]
            target = _TargetType.ARRAY_BUFFER
            bufferview["byteStride"] = int(byte_per_element)
        elif key == "texvertices":
            byte_per_element = 3 * _DTYPE_BYTES[_ITEM_TYPES[_ComponentType.FLOAT]]
            target = _TargetType.ELEMENT_ARRAY_BUFFER
            bufferview["byteStride"] = int(byte_per_element)
        elif key == "indices":
            byte_per_element = (
                3 * _DTYPE_BYTES[_ITEM_TYPES[_ComponentType.UNSIGNED_SHORT]]
            )
            target = _TargetType.ELEMENT_ARRAY_BUFFER

        bufferview["target"] = target
        bufferview["byteOffset"] = kwargs.get("offset")
        bufferview["byteLength"] = kwargs.get("byte_length")
        self._json_data["bufferViews"].append(bufferview)

    def _write_image_buffer(self, **kwargs) -> Tuple[int, bytes]:
        image_np = self.mesh.textures.maps_list()[0].cpu().numpy()
        image_array = (image_np * 255.0).astype(np.uint8)
        im = Image.fromarray(image_array)
        with BytesIO() as f:
            im.save(f, format="PNG")
            image_data = f.getvalue()

        image_data_byte_length = len(image_data)
        bufferview_image = {
            "buffer": 0,
        }
        bufferview_image["byteOffset"] = kwargs.get("offset")
        bufferview_image["byteLength"] = image_data_byte_length
        self._json_data["bufferViews"].append(bufferview_image)

        image = {"name": "texture", "mimeType": "image/png", "bufferView": 3}
        self._json_data["images"].append(image)
        return (image_data_byte_length, image_data)

    def save(self):
        # check validity of mesh
        if self.mesh.verts_packed() is None or self.mesh.faces_packed() is None:
            raise ValueError("invalid mesh to save, verts or face indices are empty")

        # accessors for positions, texture uvs and face indices
        pos_byte, pos_data = self._write_accessor_json("positions")
        idx_byte, idx_data = self._write_accessor_json("indices")
        include_textures = False
        if self.mesh.textures is not None:
            if hasattr(self.mesh.textures, "verts_features_list"):
                tex_byte, tex_data = self._write_accessor_json("texvertices")
                include_textures = True
                texcoords = False
            elif self.mesh.textures.verts_uvs_list()[0] is not None:
                tex_byte, tex_data = self._write_accessor_json("texcoords")
                include_textures = True
                texcoords = True

        # bufferViews for positions, texture coords and indices
        byte_offset = 0
        self._write_bufferview("positions", byte_length=pos_byte, offset=byte_offset)
        byte_offset += pos_byte

        self._write_bufferview("indices", byte_length=idx_byte, offset=byte_offset)
        byte_offset += idx_byte

        if include_textures:
            if texcoords:
                self._write_bufferview(
                    "texcoords", byte_length=tex_byte, offset=byte_offset
                )
            else:
                self._write_bufferview(
                    "texvertices", byte_length=tex_byte, offset=byte_offset
                )
            byte_offset += tex_byte

        # image bufferView
        include_image = False
        if self.mesh.textures is not None and hasattr(self.mesh.textures, "maps_list"):
            include_image = True
            image_byte, image_data = self._write_image_buffer(offset=byte_offset)
            byte_offset += image_byte

        # buffers
        self._json_data["buffers"].append({"byteLength": int(byte_offset)})

        # organize into a glb
        json_bytes = bytes(json.dumps(self._json_data, cls=OurEncoder), "utf-8")
        json_length = len(json_bytes)

        # write header
        version = 2
        total_header_length = 28  # (file header = 12) + 2 * (chunk header = 8)
        file_length = json_length + byte_offset + total_header_length
        header = struct.pack("<III", _GLTF_MAGIC, version, file_length)
        self.buffer_stream.write(header)

        # write json
        self.buffer_stream.write(struct.pack("<II", json_length, _JSON_CHUNK_TYPE))
        self.buffer_stream.write(json_bytes)

        # write binary data
        self.buffer_stream.write(struct.pack("<II", byte_offset, _BINARY_CHUNK_TYPE))
        self.buffer_stream.write(pos_data)
        self.buffer_stream.write(idx_data)
        if include_textures:
            self.buffer_stream.write(tex_data)
        if include_image:
            self.buffer_stream.write(image_data)


class MeshGlbFormat(MeshFormatInterpreter):
    """
    Implements loading meshes from glTF 2 assets stored in a
    GLB container file or a glTF JSON file with embedded binary data.

    This implementation is quite restricted in what it supports.

        - It does not try to validate the input against the standard.
        - It loads the default scene only.
        - Only triangulated geometry is supported.
        - The geometry of all meshes of the entire scene is aggregated into a single mesh.
        Use `load_meshes()` instead to get un-aggregated (but transformed) ones.
        - All material properties are ignored except for either vertex color, baseColorTexture
        or baseColorFactor. If available, one of these (in this order) is exclusively
        used which does not match the semantics of the standard.
    """

    def __init__(self) -> None:
        self.known_suffixes = (".glb",)

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

        names_meshes_list = load_meshes(
            path=path,
            path_manager=path_manager,
            include_textures=include_textures,
        )

        meshes_list = [mesh for name, mesh in names_meshes_list]
        mesh = join_meshes_as_scene(meshes_list)
        return mesh.to(device)

    def save(
        self,
        data: Meshes,
        path: PathOrStr,
        path_manager: PathManager,
        binary: Optional[bool],
        **kwargs,
    ) -> bool:
        """
        Writes all the meshes from the default scene to GLB file.

        Args:
            data: meshes to save
            path: path of the GLB file to write into
            path_manager: PathManager object for interpreting the path

        Return True if saving succeeds and False otherwise
        """

        if not endswith(path, self.known_suffixes):
            return False

        with _open_file(path, path_manager, "wb") as f:
            writer = _GLTFWriter(data, cast(BinaryIO, f))
            writer.save()
        return True
