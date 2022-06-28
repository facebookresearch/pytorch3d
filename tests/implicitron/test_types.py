# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import dataclasses
import unittest
from typing import Dict, List, NamedTuple, Tuple

from pytorch3d.implicitron.dataset import types
from pytorch3d.implicitron.dataset.types import FrameAnnotation


class _NT(NamedTuple):
    annot: FrameAnnotation


class TestDatasetTypes(unittest.TestCase):
    def setUp(self):
        self.entry = FrameAnnotation(
            frame_number=23,
            sequence_name="1",
            frame_timestamp=1.2,
            image=types.ImageAnnotation(path="/tmp/1.jpg", size=(224, 224)),
            mask=types.MaskAnnotation(path="/tmp/1.png", mass=42.0),
            viewpoint=types.ViewpointAnnotation(
                R=(
                    (1, 0, 0),
                    (1, 0, 0),
                    (1, 0, 0),
                ),
                T=(0, 0, 0),
                principal_point=(100, 100),
                focal_length=(200, 200),
            ),
        )

    def test_asdict_rec(self):
        first = [dataclasses.asdict(self.entry)]
        second = types._asdict_rec([self.entry])
        self.assertEqual(first, second)

    def test_parsing(self):
        """Test that we handle collections enclosing dataclasses."""

        dct = dataclasses.asdict(self.entry)

        parsed = types._dataclass_from_dict(dct, FrameAnnotation)
        self.assertEqual(parsed, self.entry)

        # namedtuple
        parsed = types._dataclass_from_dict(_NT(dct), _NT)
        self.assertEqual(parsed.annot, self.entry)

        # tuple
        parsed = types._dataclass_from_dict((dct,), Tuple[FrameAnnotation])
        self.assertEqual(parsed, (self.entry,))

        # list
        parsed = types._dataclass_from_dict(
            [
                dct,
            ],
            List[FrameAnnotation],
        )
        self.assertEqual(
            parsed,
            [
                self.entry,
            ],
        )

        # dict
        parsed = types._dataclass_from_dict({"key": dct}, Dict[str, FrameAnnotation])
        self.assertEqual(parsed, {"key": self.entry})

    def test_parsing_vectorized(self):
        dct = dataclasses.asdict(self.entry)

        self._compare_with_scalar(dct, FrameAnnotation)
        self._compare_with_scalar(_NT(dct), _NT)
        self._compare_with_scalar((dct,), Tuple[FrameAnnotation])
        self._compare_with_scalar([dct], List[FrameAnnotation])
        self._compare_with_scalar({"key": dct}, Dict[str, FrameAnnotation])

        dct2 = dct.copy()
        dct2["meta"] = {"aux": 76}
        self._compare_with_scalar(dct2, FrameAnnotation)

    def _compare_with_scalar(self, obj, typeannot, repeat=3):
        input = [obj] * 3
        vect_output = types._dataclass_list_from_dict_list(input, typeannot)
        self.assertEqual(len(input), repeat)
        gt = types._dataclass_from_dict(obj, typeannot)
        self.assertTrue(all(res == gt for res in vect_output))
