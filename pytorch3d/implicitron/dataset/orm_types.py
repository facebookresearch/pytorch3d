# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

# This functionality requires SQLAlchemy 2.0 or later.

import math
import struct
from typing import Optional, Tuple

import numpy as np

from pytorch3d.implicitron.dataset.types import (
    DepthAnnotation,
    ImageAnnotation,
    MaskAnnotation,
    PointCloudAnnotation,
    VideoAnnotation,
    ViewpointAnnotation,
)

from sqlalchemy import LargeBinary
from sqlalchemy.orm import (
    composite,
    DeclarativeBase,
    Mapped,
    mapped_column,
    MappedAsDataclass,
)
from sqlalchemy.types import TypeDecorator


# these produce policies to serialize structured types to blobs
def ArrayTypeFactory(shape=None):
    if shape is None:

        class VariableShapeNumpyArrayType(TypeDecorator):
            impl = LargeBinary

            def process_bind_param(self, value, dialect):
                if value is None:
                    return None

                ndim_bytes = np.int32(value.ndim).tobytes()
                shape_bytes = np.array(value.shape, dtype=np.int64).tobytes()
                value_bytes = value.astype(np.float32).tobytes()
                return ndim_bytes + shape_bytes + value_bytes

            def process_result_value(self, value, dialect):
                if value is None:
                    return None

                ndim = np.frombuffer(value[:4], dtype=np.int32)[0]
                value_start = 4 + 8 * ndim
                shape = np.frombuffer(value[4:value_start], dtype=np.int64)
                assert shape.shape == (ndim,)
                return np.frombuffer(value[value_start:], dtype=np.float32).reshape(
                    shape
                )

        return VariableShapeNumpyArrayType

    class NumpyArrayType(TypeDecorator):
        impl = LargeBinary

        def process_bind_param(self, value, dialect):
            if value is not None:
                if value.shape != shape:
                    raise ValueError(f"Passed an array of wrong shape: {value.shape}")
                return value.astype(np.float32).tobytes()
            return None

        def process_result_value(self, value, dialect):
            if value is not None:
                return np.frombuffer(value, dtype=np.float32).reshape(shape)
            return None

    return NumpyArrayType


def TupleTypeFactory(dtype=float, shape: Tuple[int, ...] = (2,)):
    format_symbol = {
        float: "f",  # float32
        int: "i",  # int32
    }[dtype]

    class TupleType(TypeDecorator):
        impl = LargeBinary
        _format = format_symbol * math.prod(shape)

        def process_bind_param(self, value, _):
            if value is None:
                return None

            if len(shape) > 1:
                value = np.array(value, dtype=dtype).reshape(-1)

            return struct.pack(TupleType._format, *value)

        def process_result_value(self, value, _):
            if value is None:
                return None

            loaded = struct.unpack(TupleType._format, value)
            if len(shape) > 1:
                loaded = _rec_totuple(
                    np.array(loaded, dtype=dtype).reshape(shape).tolist()
                )

            return loaded

    return TupleType


def _rec_totuple(t):
    if isinstance(t, list):
        return tuple(_rec_totuple(x) for x in t)

    return t


class Base(MappedAsDataclass, DeclarativeBase):
    """subclasses will be converted to dataclasses"""


class SqlFrameAnnotation(Base):
    __tablename__ = "frame_annots"

    sequence_name: Mapped[str] = mapped_column(primary_key=True)
    frame_number: Mapped[int] = mapped_column(primary_key=True)
    frame_timestamp: Mapped[float] = mapped_column(index=True)

    image: Mapped[ImageAnnotation] = composite(
        mapped_column("_image_path"),
        mapped_column("_image_size", TupleTypeFactory(int)),
    )

    depth: Mapped[DepthAnnotation] = composite(
        mapped_column("_depth_path", nullable=True),
        mapped_column("_depth_scale_adjustment", nullable=True),
        mapped_column("_depth_mask_path", nullable=True),
    )

    mask: Mapped[MaskAnnotation] = composite(
        mapped_column("_mask_path", nullable=True),
        mapped_column("_mask_mass", index=True, nullable=True),
        mapped_column(
            "_mask_bounding_box_xywh",
            TupleTypeFactory(float, shape=(4,)),
            nullable=True,
        ),
    )

    viewpoint: Mapped[ViewpointAnnotation] = composite(
        mapped_column(
            "_viewpoint_R", TupleTypeFactory(float, shape=(3, 3)), nullable=True
        ),
        mapped_column(
            "_viewpoint_T", TupleTypeFactory(float, shape=(3,)), nullable=True
        ),
        mapped_column(
            "_viewpoint_focal_length", TupleTypeFactory(float), nullable=True
        ),
        mapped_column(
            "_viewpoint_principal_point", TupleTypeFactory(float), nullable=True
        ),
        mapped_column("_viewpoint_intrinsics_format", nullable=True),
    )


class SqlSequenceAnnotation(Base):
    __tablename__ = "sequence_annots"

    sequence_name: Mapped[str] = mapped_column(primary_key=True)
    category: Mapped[str] = mapped_column(index=True)

    video: Mapped[VideoAnnotation] = composite(
        mapped_column("_video_path", nullable=True),
        mapped_column("_video_length", nullable=True),
    )
    point_cloud: Mapped[PointCloudAnnotation] = composite(
        mapped_column("_point_cloud_path", nullable=True),
        mapped_column("_point_cloud_quality_score", nullable=True),
        mapped_column("_point_cloud_n_points", nullable=True),
    )
    # the bigger the better
    viewpoint_quality_score: Mapped[Optional[float]] = mapped_column()
