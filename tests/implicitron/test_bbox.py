# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import numpy as np

import torch
from pytorch3d.implicitron.dataset.json_index_dataset import (
    _bbox_xywh_to_xyxy,
    _bbox_xyxy_to_xywh,
    _get_bbox_from_mask,
)
from tests.common_testing import TestCaseMixin


class TestBBox(TestCaseMixin, unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_bbox_conversion(self):
        bbox_xywh_list = torch.LongTensor(
            [
                [0, 0, 10, 20],
                [10, 20, 5, 1],
                [10, 20, 1, 1],
                [5, 4, 0, 1],
            ]
        )
        for bbox_xywh in bbox_xywh_list:
            bbox_xyxy = _bbox_xywh_to_xyxy(bbox_xywh)
            bbox_xywh_ = _bbox_xyxy_to_xywh(bbox_xyxy)
            bbox_xyxy_ = _bbox_xywh_to_xyxy(bbox_xywh_)
            self.assertClose(bbox_xywh_, bbox_xywh)
            self.assertClose(bbox_xyxy, bbox_xyxy_)

    def test_compare_to_expected(self):
        bbox_xywh_to_xyxy_expected = torch.LongTensor(
            [
                [[0, 0, 10, 20], [0, 0, 10, 20]],
                [[10, 20, 5, 1], [10, 20, 15, 21]],
                [[10, 20, 1, 1], [10, 20, 11, 21]],
                [[5, 4, 0, 1], [5, 4, 5, 5]],
            ]
        )
        for bbox_xywh, bbox_xyxy_expected in bbox_xywh_to_xyxy_expected:
            self.assertClose(_bbox_xywh_to_xyxy(bbox_xywh), bbox_xyxy_expected)
            self.assertClose(_bbox_xyxy_to_xywh(bbox_xyxy_expected), bbox_xywh)

        clamp_amnt = 3
        bbox_xywh_to_xyxy_clamped_expected = torch.LongTensor(
            [
                [[0, 0, 10, 20], [0, 0, 10, 20]],
                [[10, 20, 5, 1], [10, 20, 15, 20 + clamp_amnt]],
                [[10, 20, 1, 1], [10, 20, 10 + clamp_amnt, 20 + clamp_amnt]],
                [[5, 4, 0, 1], [5, 4, 5 + clamp_amnt, 4 + clamp_amnt]],
            ]
        )
        for bbox_xywh, bbox_xyxy_expected in bbox_xywh_to_xyxy_clamped_expected:
            self.assertClose(
                _bbox_xywh_to_xyxy(bbox_xywh, clamp_size=clamp_amnt),
                bbox_xyxy_expected,
            )

    def test_mask_to_bbox(self):
        mask = np.array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        ).astype(np.float32)
        expected_bbox_xywh = [2, 1, 2, 1]
        bbox_xywh = _get_bbox_from_mask(mask, 0.5)
        self.assertClose(bbox_xywh, expected_bbox_xywh)
