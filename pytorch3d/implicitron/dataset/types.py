# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe


import dataclasses
import gzip
import json
from dataclasses import dataclass, Field, MISSING
from typing import (
    Any,
    cast,
    Dict,
    get_args,
    get_origin,
    IO,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import numpy as np


_X = TypeVar("_X")

TF3 = Tuple[float, float, float]


@dataclass
class ImageAnnotation:
    # path to jpg file, relative w.r.t. dataset_root
    path: str
    # H x W
    size: Tuple[int, int]  # TODO: rename size_hw?


@dataclass
class DepthAnnotation:
    # path to png file, relative w.r.t. dataset_root, storing `depth / scale_adjustment`
    path: str
    # a factor to convert png values to actual depth: `depth = png * scale_adjustment`
    scale_adjustment: float
    # path to png file, relative w.r.t. dataset_root, storing binary `depth` mask
    mask_path: Optional[str]


@dataclass
class MaskAnnotation:
    # path to png file storing (Prob(fg | pixel) * 255)
    path: str
    # (soft) number of pixels in the mask; sum(Prob(fg | pixel))
    mass: Optional[float] = None
    # tight bounding box around the foreground mask
    bounding_box_xywh: Optional[Tuple[float, float, float, float]] = None


@dataclass
class ViewpointAnnotation:
    # In right-multiply (PyTorch3D) format. X_cam = X_world @ R + T
    R: Tuple[TF3, TF3, TF3]
    T: TF3

    focal_length: Tuple[float, float]
    principal_point: Tuple[float, float]

    intrinsics_format: str = "ndc_norm_image_bounds"
    # Defines the co-ordinate system where focal_length and principal_point live.
    # Possible values: ndc_isotropic | ndc_norm_image_bounds (default)
    # ndc_norm_image_bounds: legacy PyTorch3D NDC format, where image boundaries
    #     correspond to [-1, 1] x [-1, 1], and the scale along x and y may differ
    # ndc_isotropic: PyTorch3D 0.5+ NDC convention where the shorter side has
    #     the range [-1, 1], and the longer one has the range [-s, s]; s >= 1,
    #     where s is the aspect ratio. The scale is same along x and y.


@dataclass
class FrameAnnotation:
    """A dataclass used to load annotations from json."""

    # can be used to join with `SequenceAnnotation`
    sequence_name: str
    # 0-based, continuous frame number within sequence
    frame_number: int
    # timestamp in seconds from the video start
    frame_timestamp: float

    image: ImageAnnotation
    depth: Optional[DepthAnnotation] = None
    mask: Optional[MaskAnnotation] = None
    viewpoint: Optional[ViewpointAnnotation] = None
    meta: Optional[Dict[str, Any]] = None


@dataclass
class PointCloudAnnotation:
    # path to ply file with points only, relative w.r.t. dataset_root
    path: str
    # the bigger the better
    quality_score: float
    n_points: Optional[int]


@dataclass
class VideoAnnotation:
    # path to the original video file, relative w.r.t. dataset_root
    path: str
    # length of the video in seconds
    length: float


@dataclass
class SequenceAnnotation:
    sequence_name: str
    category: str
    video: Optional[VideoAnnotation] = None
    point_cloud: Optional[PointCloudAnnotation] = None
    # the bigger the better
    viewpoint_quality_score: Optional[float] = None


def dump_dataclass(obj: Any, f: IO, binary: bool = False) -> None:
    """
    Args:
        f: Either a path to a file, or a file opened for writing.
        obj: A @dataclass or collection hierarchy including dataclasses.
        binary: Set to True if `f` is a file handle, else False.
    """
    if binary:
        f.write(json.dumps(_asdict_rec(obj)).encode("utf8"))
    else:
        json.dump(_asdict_rec(obj), f)


def load_dataclass(f: IO, cls: Type[_X], binary: bool = False) -> _X:
    """
    Loads to a @dataclass or collection hierarchy including dataclasses
    from a json recursively.
    Call it like load_dataclass(f, typing.List[FrameAnnotationAnnotation]).
    raises KeyError if json has keys not mapping to the dataclass fields.

    Args:
        f: Either a path to a file, or a file opened for writing.
        cls: The class of the loaded dataclass.
        binary: Set to True if `f` is a file handle, else False.
    """
    if binary:
        asdict = json.loads(f.read().decode("utf8"))
    else:
        asdict = json.load(f)

    if isinstance(asdict, list):
        # in the list case, run a faster "vectorized" version
        cls = get_args(cls)[0]
        res = list(_dataclass_list_from_dict_list(asdict, cls))
    else:
        res = _dataclass_from_dict(asdict, cls)

    return res


def _dataclass_list_from_dict_list(dlist, typeannot):
    """
    Vectorised version of `_dataclass_from_dict`.
    The output should be equivalent to
    `[_dataclass_from_dict(d, typeannot) for d in dlist]`.

    Args:
        dlist: list of objects to convert.
        typeannot: type of each of those objects.
    Returns:
        iterator or list over converted objects of the same length as `dlist`.

    Raises:
        ValueError: it assumes the objects have None's in consistent places across
            objects, otherwise it would ignore some values. This generally holds for
            auto-generated annotations, but otherwise use `_dataclass_from_dict`.
    """

    cls = get_origin(typeannot) or typeannot

    if typeannot is Any:
        return dlist
    if all(obj is None for obj in dlist):  # 1st recursion base: all None nodes
        return dlist
    if any(obj is None for obj in dlist):
        # filter out Nones and recurse on the resulting list
        idx_notnone = [(i, obj) for i, obj in enumerate(dlist) if obj is not None]
        idx, notnone = zip(*idx_notnone)
        converted = _dataclass_list_from_dict_list(notnone, typeannot)
        res = [None] * len(dlist)
        for i, obj in zip(idx, converted):
            res[i] = obj
        return res

    is_optional, contained_type = _resolve_optional(typeannot)
    if is_optional:
        return _dataclass_list_from_dict_list(dlist, contained_type)

    # otherwise, we dispatch by the type of the provided annotation to convert to
    if issubclass(cls, tuple) and hasattr(cls, "_fields"):  # namedtuple
        # For namedtuple, call the function recursively on the lists of corresponding keys
        types = cls.__annotations__.values()
        dlist_T = zip(*dlist)
        res_T = [
            _dataclass_list_from_dict_list(key_list, tp)
            for key_list, tp in zip(dlist_T, types)
        ]
        return [cls(*converted_as_tuple) for converted_as_tuple in zip(*res_T)]
    elif issubclass(cls, (list, tuple)):
        # For list/tuple, call the function recursively on the lists of corresponding positions
        types = get_args(typeannot)
        if len(types) == 1:  # probably List; replicate for all items
            types = types * len(dlist[0])
        dlist_T = zip(*dlist)
        res_T = (
            _dataclass_list_from_dict_list(pos_list, tp)
            for pos_list, tp in zip(dlist_T, types)
        )
        if issubclass(cls, tuple):
            return list(zip(*res_T))
        else:
            return [cls(converted_as_tuple) for converted_as_tuple in zip(*res_T)]
    elif issubclass(cls, dict):
        # For the dictionary, call the function recursively on concatenated keys and vertices
        key_t, val_t = get_args(typeannot)
        all_keys_res = _dataclass_list_from_dict_list(
            [k for obj in dlist for k in obj.keys()], key_t
        )
        all_vals_res = _dataclass_list_from_dict_list(
            [k for obj in dlist for k in obj.values()], val_t
        )
        indices = np.cumsum([len(obj) for obj in dlist])
        assert indices[-1] == len(all_keys_res)

        keys = np.split(list(all_keys_res), indices[:-1])
        all_vals_res_iter = iter(all_vals_res)
        return [cls(zip(k, all_vals_res_iter)) for k in keys]
    elif not dataclasses.is_dataclass(typeannot):
        return dlist

    # dataclass node: 2nd recursion base; call the function recursively on the lists
    # of the corresponding fields
    assert dataclasses.is_dataclass(cls)
    fieldtypes = {
        f.name: (_unwrap_type(f.type), _get_dataclass_field_default(f))
        for f in dataclasses.fields(typeannot)
    }

    # NOTE the default object is shared here
    key_lists = (
        _dataclass_list_from_dict_list([obj.get(k, default) for obj in dlist], type_)
        for k, (type_, default) in fieldtypes.items()
    )
    transposed = zip(*key_lists)
    return [cls(*vals_as_tuple) for vals_as_tuple in transposed]


def _dataclass_from_dict(d, typeannot):
    if d is None or typeannot is Any:
        return d
    is_optional, contained_type = _resolve_optional(typeannot)
    if is_optional:
        # an Optional not set to None, just use the contents of the Optional.
        return _dataclass_from_dict(d, contained_type)

    cls = get_origin(typeannot) or typeannot
    if issubclass(cls, tuple) and hasattr(cls, "_fields"):  # namedtuple
        types = cls.__annotations__.values()
        return cls(*[_dataclass_from_dict(v, tp) for v, tp in zip(d, types)])
    elif issubclass(cls, (list, tuple)):
        types = get_args(typeannot)
        if len(types) == 1:  # probably List; replicate for all items
            types = types * len(d)
        return cls(_dataclass_from_dict(v, tp) for v, tp in zip(d, types))
    elif issubclass(cls, dict):
        key_t, val_t = get_args(typeannot)
        return cls(
            (_dataclass_from_dict(k, key_t), _dataclass_from_dict(v, val_t))
            for k, v in d.items()
        )
    elif not dataclasses.is_dataclass(typeannot):
        return d

    assert dataclasses.is_dataclass(cls)
    fieldtypes = {f.name: _unwrap_type(f.type) for f in dataclasses.fields(typeannot)}
    return cls(**{k: _dataclass_from_dict(v, fieldtypes[k]) for k, v in d.items()})


def _unwrap_type(tp):
    # strips Optional wrapper, if any
    if get_origin(tp) is Union:
        args = get_args(tp)
        if len(args) == 2 and any(a is type(None) for a in args):  # noqa: E721
            # this is typing.Optional
            return args[0] if args[1] is type(None) else args[1]  # noqa: E721
    return tp


def _get_dataclass_field_default(field: Field) -> Any:
    if field.default_factory is not MISSING:
        # pyre-fixme[29]: `Union[dataclasses._MISSING_TYPE,
        #  dataclasses._DefaultFactory[typing.Any]]` is not a function.
        return field.default_factory()
    elif field.default is not MISSING:
        return field.default
    else:
        return None


def _asdict_rec(obj):
    return dataclasses._asdict_inner(obj, dict)


def dump_dataclass_jgzip(outfile: str, obj: Any) -> None:
    """
    Dumps obj to a gzipped json outfile.

    Args:
        obj: A @dataclass or collection hiererchy including dataclasses.
        outfile: The path to the output file.
    """
    with gzip.GzipFile(outfile, "wb") as f:
        dump_dataclass(obj, cast(IO, f), binary=True)


def load_dataclass_jgzip(outfile, cls):
    """
    Loads a dataclass from a gzipped json outfile.

    Args:
        outfile: The path to the loaded file.
        cls: The type annotation of the loaded dataclass.

    Returns:
        loaded_dataclass: The loaded dataclass.
    """
    with gzip.GzipFile(outfile, "rb") as f:
        return load_dataclass(cast(IO, f), cls, binary=True)


def _resolve_optional(type_: Any) -> Tuple[bool, Any]:
    """Check whether `type_` is equivalent to `typing.Optional[T]` for some T."""
    if get_origin(type_) is Union:
        args = get_args(type_)
        if len(args) == 2 and args[1] == type(None):  # noqa E721
            return True, args[0]
    if type_ is Any:
        return True, Any

    return False, type_
