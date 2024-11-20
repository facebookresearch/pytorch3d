# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import logging
import os
import tempfile
import unittest
from typing import ClassVar, Optional, Type

import pandas as pd
import pkg_resources
import sqlalchemy as sa

from pytorch3d.implicitron.dataset import types
from pytorch3d.implicitron.dataset.frame_data import FrameData, GenericFrameDataBuilder
from pytorch3d.implicitron.dataset.orm_types import (
    SqlFrameAnnotation,
    SqlSequenceAnnotation,
)
from pytorch3d.implicitron.dataset.sql_dataset import SqlIndexDataset
from pytorch3d.implicitron.dataset.utils import GenericWorkaround
from pytorch3d.implicitron.tools.config import registry
from sqlalchemy.orm import composite, Mapped, mapped_column, Session

NO_BLOBS_KWARGS = {
    "dataset_root": "",
    "load_images": False,
    "load_depths": False,
    "load_masks": False,
    "load_depth_masks": False,
    "box_crop": False,
}

DATASET_ROOT = pkg_resources.resource_filename(__name__, "data/sql_dataset")
METADATA_FILE = os.path.join(DATASET_ROOT, "sql_dataset_100.sqlite")

logger = logging.getLogger("pytorch3d.implicitron.dataset.sql_dataset")
sh = logging.StreamHandler()
logger.addHandler(sh)
logger.setLevel(logging.DEBUG)


@dataclasses.dataclass
class MagneticFieldAnnotation:
    path: str
    average_flux_density: Optional[float] = None


class ExtendedSqlFrameAnnotation(SqlFrameAnnotation):
    num_dogs: Mapped[Optional[int]] = mapped_column(default=None)

    magnetic_field: Mapped[MagneticFieldAnnotation] = composite(
        mapped_column("_magnetic_field_path", nullable=True),
        mapped_column("_magnetic_field_average_flux_density", nullable=True),
        default_factory=lambda: None,
    )


class ExtendedSqlIndexDataset(SqlIndexDataset):
    frame_annotations_type: ClassVar[Type[SqlFrameAnnotation]] = (
        ExtendedSqlFrameAnnotation
    )


class CanineFrameData(FrameData):
    num_dogs: Optional[int] = None
    magnetic_field_average_flux_density: Optional[float] = None


@registry.register
class CanineFrameDataBuilder(
    GenericWorkaround, GenericFrameDataBuilder[CanineFrameData]
):
    """
    A concrete class to build an extended FrameData object
    """

    frame_data_type: ClassVar[Type[FrameData]] = CanineFrameData

    def build(
        self,
        frame_annotation: ExtendedSqlFrameAnnotation,
        sequence_annotation: types.SequenceAnnotation,
        load_blobs: bool = True,
    ) -> CanineFrameData:
        frame_data = super().build(
            frame_annotation, sequence_annotation, load_blobs=load_blobs
        )
        frame_data.num_dogs = frame_annotation.num_dogs or 101
        frame_data.magnetic_field_average_flux_density = (
            frame_annotation.magnetic_field.average_flux_density
        )
        return frame_data


class CanineSqlIndexDataset(SqlIndexDataset):
    frame_annotations_type: ClassVar[Type[SqlFrameAnnotation]] = (
        ExtendedSqlFrameAnnotation
    )

    frame_data_builder_class_type: str = "CanineFrameDataBuilder"


class TestExtendingOrmTypes(unittest.TestCase):
    def setUp(self):
        # create a temporary copy of the DB with an extended schema
        engine = sa.create_engine(f"sqlite:///{METADATA_FILE}")
        with Session(engine) as session:
            extended_annots = [
                ExtendedSqlFrameAnnotation(
                    **{
                        k: v
                        for k, v in frame_annot.__dict__.items()
                        if not k.startswith("_")  # remove mapped fields and SA metadata
                    }
                )
                for frame_annot in session.scalars(sa.select(SqlFrameAnnotation))
            ]
            seq_annots = session.scalars(
                sa.select(SqlSequenceAnnotation),
                execution_options={"prebuffer_rows": True},
            )
            session.expunge_all()

        self._temp_db = tempfile.NamedTemporaryFile(delete=False)
        engine_ext = sa.create_engine(f"sqlite:///{self._temp_db.name}")
        ExtendedSqlFrameAnnotation.metadata.create_all(engine_ext, checkfirst=True)
        with Session(engine_ext, expire_on_commit=False) as session_ext:
            session_ext.add_all(extended_annots)
            for instance in seq_annots:
                session_ext.merge(instance)
            session_ext.commit()

        # check the setup is correct
        with engine_ext.connect() as connection_ext:
            df = pd.read_sql_query(
                sa.select(ExtendedSqlFrameAnnotation), connection_ext
            )
            self.assertEqual(len(df), 100)
            self.assertIn("_magnetic_field_average_flux_density", df.columns)

            df_seq = pd.read_sql_query(sa.select(SqlSequenceAnnotation), connection_ext)
            self.assertEqual(len(df_seq), 10)

    def tearDown(self):
        self._temp_db.close()
        os.remove(self._temp_db.name)

    def test_basic(self, sequence="cat1_seq2", frame_number=4):
        dataset = ExtendedSqlIndexDataset(
            sqlite_metadata_file=self._temp_db.name,
            remove_empty_masks=False,
            frame_data_builder_FrameDataBuilder_args=NO_BLOBS_KWARGS,
        )

        self.assertEqual(len(dataset), 100)

        # check the items are consecutive
        past_sequences = set()
        last_frame_number = -1
        last_sequence = ""
        for i in range(len(dataset)):
            item = dataset[i]

            if item.frame_number == 0:
                self.assertNotIn(item.sequence_name, past_sequences)
                past_sequences.add(item.sequence_name)
                last_sequence = item.sequence_name
            else:
                self.assertEqual(item.sequence_name, last_sequence)
                self.assertEqual(item.frame_number, last_frame_number + 1)

            last_frame_number = item.frame_number

        # test indexing
        with self.assertRaises(IndexError):
            dataset[len(dataset) + 1]

        # test sequence-frame indexing
        item = dataset[sequence, frame_number]
        self.assertEqual(item.sequence_name, sequence)
        self.assertEqual(item.frame_number, frame_number)

        with self.assertRaises(IndexError):
            dataset[sequence, 13]

    def test_extending_frame_data(self, sequence="cat1_seq2", frame_number=4):
        dataset = CanineSqlIndexDataset(
            sqlite_metadata_file=self._temp_db.name,
            remove_empty_masks=False,
            frame_data_builder_CanineFrameDataBuilder_args=NO_BLOBS_KWARGS,
        )

        self.assertEqual(len(dataset), 100)

        # check the items are consecutive
        past_sequences = set()
        last_frame_number = -1
        last_sequence = ""
        for i in range(len(dataset)):
            item = dataset[i]
            self.assertIsInstance(item, CanineFrameData)
            self.assertEqual(item.num_dogs, 101)
            self.assertIsNone(item.magnetic_field_average_flux_density)

            if item.frame_number == 0:
                self.assertNotIn(item.sequence_name, past_sequences)
                past_sequences.add(item.sequence_name)
                last_sequence = item.sequence_name
            else:
                self.assertEqual(item.sequence_name, last_sequence)
                self.assertEqual(item.frame_number, last_frame_number + 1)

            last_frame_number = item.frame_number

        # test indexing
        with self.assertRaises(IndexError):
            dataset[len(dataset) + 1]

        # test sequence-frame indexing
        item = dataset[sequence, frame_number]
        self.assertIsInstance(item, CanineFrameData)
        self.assertEqual(item.sequence_name, sequence)
        self.assertEqual(item.frame_number, frame_number)
        self.assertEqual(item.num_dogs, 101)

        with self.assertRaises(IndexError):
            dataset[sequence, 13]
