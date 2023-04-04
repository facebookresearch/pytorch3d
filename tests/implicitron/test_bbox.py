# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import numpy as np

import torch

from pytorch3d.implicitron.dataset.utils import (
    bbox_xywh_to_xyxy,
    bbox_xyxy_to_xywh,
    clamp_box_to_image_bounds_and_round,
    crop_around_box,
    get_1d_bounds,
    get_bbox_from_mask,
    get_clamp_bbox,
    rescale_bbox,
    resize_image,
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
            bbox_xyxy = bbox_xywh_to_xyxy(bbox_xywh)
            bbox_xywh_ = bbox_xyxy_to_xywh(bbox_xyxy)
            bbox_xyxy_ = bbox_xywh_to_xyxy(bbox_xywh_)
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
            self.assertClose(bbox_xywh_to_xyxy(bbox_xywh), bbox_xyxy_expected)
            self.assertClose(bbox_xyxy_to_xywh(bbox_xyxy_expected), bbox_xywh)

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
                bbox_xywh_to_xyxy(bbox_xywh, clamp_size=clamp_amnt),
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
        bbox_xywh = get_bbox_from_mask(mask, 0.5)
        self.assertClose(bbox_xywh, expected_bbox_xywh)

    def test_crop_around_box(self):
        bbox = torch.LongTensor([0, 1, 2, 3])  # (x_min, y_min, x_max, y_max)
        image = torch.LongTensor(
            [
                [0, 0, 10, 20],
                [10, 20, 5, 1],
                [10, 20, 1, 1],
                [5, 4, 0, 1],
            ]
        )
        cropped = crop_around_box(image, bbox)
        self.assertClose(cropped, image[1:3, 0:2])

    def test_clamp_box_to_image_bounds_and_round(self):
        bbox = torch.LongTensor([0, 1, 10, 12])
        image_size = (5, 6)
        expected_clamped_bbox = torch.LongTensor([0, 1, image_size[1], image_size[0]])
        clamped_bbox = clamp_box_to_image_bounds_and_round(bbox, image_size)
        self.assertClose(clamped_bbox, expected_clamped_bbox)

    def test_get_clamp_bbox(self):
        bbox_xywh = torch.LongTensor([1, 1, 4, 5])
        clamped_bbox_xyxy = get_clamp_bbox(bbox_xywh, box_crop_context=2)
        # size multiplied by 2 and added coordinates
        self.assertClose(clamped_bbox_xyxy, torch.Tensor([-3, -4, 9, 11]))

    def test_rescale_bbox(self):
        bbox = torch.Tensor([0.0, 1.0, 3.0, 4.0])
        original_resolution = (4, 4)
        new_resolution = (8, 8)  # twice bigger
        rescaled_bbox = rescale_bbox(bbox, original_resolution, new_resolution)
        self.assertClose(bbox * 2, rescaled_bbox)

    def test_get_1d_bounds(self):
        array = [0, 1, 2]
        bounds = get_1d_bounds(array)
        # make nonzero 1d bounds of image
        self.assertClose(bounds, [1, 3])

    def test_resize_image(self):
        image = np.random.rand(3, 300, 500)  # rgb image 300x500
        expected_shape = (150, 250)

        resized_image, scale, mask_crop = resize_image(
            image, image_height=expected_shape[0], image_width=expected_shape[1]
        )

        original_shape = image.shape[-2:]
        expected_scale = min(
            expected_shape[0] / original_shape[0], expected_shape[1] / original_shape[1]
        )

        self.assertEqual(scale, expected_scale)
        self.assertEqual(resized_image.shape[-2:], expected_shape)
        self.assertEqual(mask_crop.shape[-2:], expected_shape)
