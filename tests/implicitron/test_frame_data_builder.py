# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import gzip
import os
import unittest
from typing import List

import numpy as np
import torch

from pytorch3d.implicitron.dataset import types
from pytorch3d.implicitron.dataset.dataset_base import FrameData
from pytorch3d.implicitron.dataset.frame_data import FrameDataBuilder
from pytorch3d.implicitron.dataset.utils import (
    get_bbox_from_mask,
    load_16big_png_depth,
    load_1bit_png_mask,
    load_depth,
    load_depth_mask,
    load_image,
    load_mask,
    safe_as_tensor,
    transpose_normalize_image,
)
from pytorch3d.implicitron.tools.config import get_default_args
from pytorch3d.renderer.cameras import PerspectiveCameras

from tests.common_testing import TestCaseMixin
from tests.implicitron.common_resources import get_skateboard_data


class TestFrameDataBuilder(TestCaseMixin, unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

        category = "skateboard"
        stack = contextlib.ExitStack()
        self.dataset_root, self.path_manager = stack.enter_context(
            get_skateboard_data()
        )
        self.addCleanup(stack.close)
        self.image_height = 768
        self.image_width = 512

        self.frame_data_builder = FrameDataBuilder(
            image_height=self.image_height,
            image_width=self.image_width,
            dataset_root=self.dataset_root,
            path_manager=self.path_manager,
        )

        # loading single frame annotation of dataset (see JsonIndexDataset._load_frames())
        frame_file = os.path.join(self.dataset_root, category, "frame_annotations.jgz")
        local_file = self.path_manager.get_local_path(frame_file)
        with gzip.open(local_file, "rt", encoding="utf8") as zipfile:
            frame_annots_list = types.load_dataclass(
                zipfile, List[types.FrameAnnotation]
            )
            self.frame_annotation = frame_annots_list[0]

        sequence_annotations_file = os.path.join(
            self.dataset_root, category, "sequence_annotations.jgz"
        )
        local_file = self.path_manager.get_local_path(sequence_annotations_file)
        with gzip.open(local_file, "rt", encoding="utf8") as zipfile:
            seq_annots_list = types.load_dataclass(
                zipfile, List[types.SequenceAnnotation]
            )
            seq_annots = {entry.sequence_name: entry for entry in seq_annots_list}
            self.seq_annotation = seq_annots[self.frame_annotation.sequence_name]

        point_cloud = self.seq_annotation.point_cloud
        self.frame_data = FrameData(
            frame_number=safe_as_tensor(self.frame_annotation.frame_number, torch.long),
            frame_timestamp=safe_as_tensor(
                self.frame_annotation.frame_timestamp, torch.float
            ),
            sequence_name=self.frame_annotation.sequence_name,
            sequence_category=self.seq_annotation.category,
            camera_quality_score=safe_as_tensor(
                self.seq_annotation.viewpoint_quality_score, torch.float
            ),
            point_cloud_quality_score=(
                safe_as_tensor(point_cloud.quality_score, torch.float)
                if point_cloud is not None
                else None
            ),
        )

    def test_frame_data_builder_args(self):
        # test that FrameDataBuilder works with get_default_args
        get_default_args(FrameDataBuilder)

    def test_load_and_adjust_frame_data(self):
        self.frame_data.image_size_hw = safe_as_tensor(
            self.frame_annotation.image.size, torch.long
        )
        self.frame_data.effective_image_size_hw = self.frame_data.image_size_hw

        mask_path = os.path.join(self.dataset_root, self.frame_annotation.mask.path)
        fg_mask_np = self.frame_data_builder._load_fg_probability(
            self.frame_annotation, mask_path
        )
        self.frame_data.mask_path = mask_path
        self.frame_data.fg_probability = safe_as_tensor(fg_mask_np, torch.float)
        mask_thr = self.frame_data_builder.box_crop_mask_thr
        bbox_xywh = get_bbox_from_mask(fg_mask_np, mask_thr)
        self.frame_data.bbox_xywh = safe_as_tensor(bbox_xywh, torch.long)

        self.assertTrue(torch.is_tensor(self.frame_data.fg_probability))
        self.assertTrue(torch.is_tensor(self.frame_data.bbox_xywh))
        # assert bboxes shape
        self.assertEqual(self.frame_data.bbox_xywh.shape, torch.Size([4]))

        image_path = os.path.join(
            self.frame_data_builder.dataset_root, self.frame_annotation.image.path
        )
        image_np = load_image(self.frame_data_builder._local_path(image_path))
        self.assertIsInstance(image_np, np.ndarray)
        self.frame_data.image_rgb = self.frame_data_builder._postprocess_image(
            image_np, self.frame_annotation.image.size, self.frame_data.fg_probability
        )
        self.assertIsInstance(self.frame_data.image_rgb, torch.Tensor)

        depth_path = os.path.join(self.dataset_root, self.frame_annotation.depth.path)
        (
            self.frame_data.depth_map,
            self.frame_data.depth_mask,
        ) = self.frame_data_builder._load_mask_depth(
            self.frame_annotation,
            depth_path,
            self.frame_data.fg_probability,
        )
        self.assertTrue(torch.is_tensor(self.frame_data.depth_map))
        self.assertTrue(torch.is_tensor(self.frame_data.depth_mask))

        new_size = (self.image_height, self.image_width)

        if self.frame_data_builder.box_crop:
            self.frame_data.crop_by_metadata_bbox_(
                self.frame_data_builder.box_crop_context,
            )

        # assert image and mask shapes after resize
        self.frame_data.resize_frame_(
            new_size_hw=torch.tensor(new_size, dtype=torch.long),
        )
        self.assertEqual(
            self.frame_data.mask_crop.shape,
            torch.Size([1, self.image_height, self.image_width]),
        )
        self.assertEqual(
            self.frame_data.image_rgb.shape,
            torch.Size([3, self.image_height, self.image_width]),
        )
        self.assertEqual(
            self.frame_data.mask_crop.shape,
            torch.Size([1, self.image_height, self.image_width]),
        )
        self.assertEqual(
            self.frame_data.fg_probability.shape,
            torch.Size([1, self.image_height, self.image_width]),
        )
        self.assertEqual(
            self.frame_data.depth_map.shape,
            torch.Size([1, self.image_height, self.image_width]),
        )
        self.assertEqual(
            self.frame_data.depth_mask.shape,
            torch.Size([1, self.image_height, self.image_width]),
        )
        self.frame_data.camera = self.frame_data_builder._get_pytorch3d_camera(
            self.frame_annotation,
        )
        self.assertEqual(type(self.frame_data.camera), PerspectiveCameras)

    def test_transpose_normalize_image(self):
        def inverse_transpose_normalize_image(image: np.ndarray) -> np.ndarray:
            im = image * 255.0
            return im.transpose((1, 2, 0)).astype(np.uint8)

        # Test 2D input
        input_image = np.array(
            [[10, 20, 30], [40, 50, 60], [70, 80, 90]], dtype=np.uint8
        )
        expected_input = inverse_transpose_normalize_image(
            transpose_normalize_image(input_image)
        )
        self.assertClose(input_image[..., None], expected_input)

        # Test 3D input
        input_image = np.array(
            [
                [[10, 20, 30], [40, 50, 60], [70, 80, 90]],
                [[100, 110, 120], [130, 140, 150], [160, 170, 180]],
                [[190, 200, 210], [220, 230, 240], [250, 255, 255]],
            ],
            dtype=np.uint8,
        )
        expected_input = inverse_transpose_normalize_image(
            transpose_normalize_image(input_image)
        )
        self.assertClose(input_image, expected_input)

    def test_load_image(self):
        path = os.path.join(self.dataset_root, self.frame_annotation.image.path)
        local_path = self.path_manager.get_local_path(path)
        image = load_image(local_path)
        self.assertEqual(image.dtype, np.float32)
        self.assertLessEqual(np.max(image), 1.0)
        self.assertGreaterEqual(np.min(image), 0.0)

    def test_load_mask(self):
        path = os.path.join(self.dataset_root, self.frame_annotation.mask.path)
        path = self.path_manager.get_local_path(path)
        mask = load_mask(path)
        self.assertEqual(mask.dtype, np.float32)
        self.assertLessEqual(np.max(mask), 1.0)
        self.assertGreaterEqual(np.min(mask), 0.0)

    def test_load_depth(self):
        path = os.path.join(self.dataset_root, self.frame_annotation.depth.path)
        path = self.path_manager.get_local_path(path)
        depth_map = load_depth(path, self.frame_annotation.depth.scale_adjustment)
        self.assertEqual(depth_map.dtype, np.float32)
        self.assertEqual(len(depth_map.shape), 3)

    def test_load_16big_png_depth(self):
        path = os.path.join(self.dataset_root, self.frame_annotation.depth.path)
        path = self.path_manager.get_local_path(path)
        depth_map = load_16big_png_depth(path)
        self.assertEqual(depth_map.dtype, np.float32)
        self.assertEqual(len(depth_map.shape), 2)

    def test_load_1bit_png_mask(self):
        mask_path = os.path.join(
            self.dataset_root, self.frame_annotation.depth.mask_path
        )
        mask_path = self.path_manager.get_local_path(mask_path)
        mask = load_1bit_png_mask(mask_path)
        self.assertEqual(mask.dtype, np.float32)
        self.assertEqual(len(mask.shape), 2)

    def test_load_depth_mask(self):
        mask_path = os.path.join(
            self.dataset_root, self.frame_annotation.depth.mask_path
        )
        mask_path = self.path_manager.get_local_path(mask_path)
        mask = load_depth_mask(mask_path)
        self.assertEqual(mask.dtype, np.float32)
        self.assertEqual(len(mask.shape), 3)
