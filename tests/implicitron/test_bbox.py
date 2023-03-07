# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import numpy as np

import torch
from pytorch3d.implicitron.dataset.blob_loader import (
    _bbox_xywh_to_xyxy,
    _bbox_xyxy_to_xywh,
    _get_bbox_from_mask,
    _crop_around_box,
    _clamp_box_to_image_bounds_and_round,
    _get_clamp_bbox,
    _rescale_bbox,
    _get_1d_bounds,
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

    def test_crop_around_box(self):
        bbox = (0, 1, 2, 2) # (x_min, y_min, x_max, y_max)
        image = torch.LongTensor(
            [
                [0, 0, 10, 20],
                [10, 20, 5, 1],
                [10, 20, 1, 1],
                [5, 4, 0, 1],
            ]
        )
        cropped = _crop_around_box(image, bbox)
        self.assertClose(cropped, image[0:2, 1:2])

    def test_clamp_box_to_image_bounds_and_round(self):
        bbox = torch.LongTensor([0, 1, 10, 12])
        image_size = (5, 6)
        clamped_bbox = _clamp_box_to_image_bounds_and_round(bbox)
        self.assertClose(clamped_bbox == [0, 1, 5, 6])

    def test_get_clamp_bbox(self):
        bbox_xywh = torch.LongTensor([1, 1, 4, 5])
        clamped_bbox_xyxy = _get_clamp_bbox(bbox, box_crop_context=2)
        # size multiplied by 2 and added coordinates
        self.assertClose(clamped_bbox_xyxy == torch.LongTensor([0, 1, 9, 11]))

    def test_rescale_bbox(self):
        bbox = torch.LongTensor([0, 1, 3, 4])
        original_resolution = (4, 4) #
        new_resolution = (8, 8)
        rescaled_bbox = _rescale_bbox(bbox, original_resolution, new_resolution)
        self.assertClose(bbox * 2 == rescaled_bbox)

    def test_get_1d_bounds(self):
        array = [0, 1, 2]
        bounds = _get_1d_bounds(array)
        # make nonzero 1d bounds of image
        assert bounds == [1, 2]
