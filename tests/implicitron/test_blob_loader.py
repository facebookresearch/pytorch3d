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
from pytorch3d.implicitron.dataset.blob_loader import (
    _load_16big_png_depth,
    _load_1bit_png_mask,
    _load_depth,
    _load_depth_mask,
    _load_image,
    _load_mask,
    _safe_as_tensor,
    BlobLoader,
)
from pytorch3d.implicitron.dataset.dataset_base import FrameData
from pytorch3d.implicitron.tools.config import get_default_args
from pytorch3d.renderer.cameras import PerspectiveCameras

from tests.common_testing import TestCaseMixin
from tests.implicitron.common_resources import get_skateboard_data


class TestBlobLoader(TestCaseMixin, unittest.TestCase):
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

        self.blob_loader = BlobLoader(
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
            frame_number=_safe_as_tensor(
                self.frame_annotation.frame_number, torch.long
            ),
            frame_timestamp=_safe_as_tensor(
                self.frame_annotation.frame_timestamp, torch.float
            ),
            sequence_name=self.frame_annotation.sequence_name,
            sequence_category=self.seq_annotation.category,
            camera_quality_score=_safe_as_tensor(
                self.seq_annotation.viewpoint_quality_score, torch.float
            ),
            point_cloud_quality_score=_safe_as_tensor(
                point_cloud.quality_score, torch.float
            )
            if point_cloud is not None
            else None,
        )

    def test_BlobLoader_args(self):
        # test that BlobLoader works with get_default_args
        get_default_args(BlobLoader)

    def test_fix_point_cloud_path(self):
        """Some files in Co3Dv2 have an accidental absolute path stored."""
        original_path = "some_file_path"
        modified_path = self.blob_loader._fix_point_cloud_path(original_path)
        assert original_path in modified_path
        assert self.blob_loader.dataset_root in modified_path

    def test_load_(self):
        bbox_xywh = None
        self.frame_data.image_size_hw = _safe_as_tensor(
            self.frame_annotation.image.size, torch.long
        )
        (
            self.frame_data.fg_probability,
            self.frame_data.mask_path,
            self.frame_data.bbox_xywh,
        ) = self.blob_loader._load_fg_probability(self.frame_annotation, bbox_xywh)

        assert self.frame_data.mask_path
        assert torch.is_tensor(self.frame_data.fg_probability)
        assert torch.is_tensor(self.frame_data.bbox_xywh)
        # assert bboxes shape
        self.assertEqual(self.frame_data.bbox_xywh.shape, torch.Size([4]))
        (
            self.frame_data.image_rgb,
            self.frame_data.image_path,
        ) = self.blob_loader._load_images(
            self.frame_annotation, self.frame_data.fg_probability
        )
        self.assertEqual(type(self.frame_data.image_rgb), np.ndarray)
        assert self.frame_data.image_path

        (
            self.frame_data.depth_map,
            depth_path,
            self.frame_data.depth_mask,
        ) = self.blob_loader._load_mask_depth(
            self.frame_annotation,
            self.frame_data.fg_probability,
        )
        assert torch.is_tensor(self.frame_data.depth_map)
        assert depth_path
        assert torch.is_tensor(self.frame_data.depth_mask)

        clamp_bbox_xyxy = None
        if self.blob_loader.box_crop:
            clamp_bbox_xyxy = self.frame_data.crop_by_bbox_(
                self.blob_loader.box_crop_context
            )

        # assert image and mask shapes after resize
        scale = self.frame_data.resize_frame_(self.image_height, self.image_width)
        assert scale
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

        self.frame_data.camera = self.blob_loader._get_pytorch3d_camera(
            self.frame_annotation,
            scale,
            clamp_bbox_xyxy,
        )
        self.assertEqual(type(self.frame_data.camera), PerspectiveCameras)

    def test_load_image(self):
        path = os.path.join(self.dataset_root, self.frame_annotation.image.path)
        local_path = self.path_manager.get_local_path(path)
        image = _load_image(local_path)
        self.assertEqual(image.dtype, np.float32)
        assert np.max(image) <= 1.0
        assert np.min(image) >= 0.0

    def test_load_mask(self):
        path = os.path.join(self.dataset_root, self.frame_annotation.mask.path)
        mask = _load_mask(path)
        self.assertEqual(mask.dtype, np.float32)
        assert np.max(mask) <= 1.0
        assert np.min(mask) >= 0.0

    def test_load_depth(self):
        path = os.path.join(self.dataset_root, self.frame_annotation.depth.path)
        depth_map = _load_depth(path, self.frame_annotation.depth.scale_adjustment)
        self.assertEqual(depth_map.dtype, np.float32)
        self.assertEqual(len(depth_map.shape), 3)

    def test_load_16big_png_depth(self):
        path = os.path.join(self.dataset_root, self.frame_annotation.depth.path)
        depth_map = _load_16big_png_depth(path)
        self.assertEqual(depth_map.dtype, np.float32)
        self.assertEqual(len(depth_map.shape), 2)

    def test_load_1bit_png_mask(self):
        mask_path = os.path.join(
            self.dataset_root, self.frame_annotation.depth.mask_path
        )
        mask = _load_1bit_png_mask(mask_path)
        self.assertEqual(mask.dtype, np.float32)
        self.assertEqual(len(mask.shape), 2)

    def test_load_depth_mask(self):
        mask_path = os.path.join(
            self.dataset_root, self.frame_annotation.depth.mask_path
        )
        mask = _load_depth_mask(mask_path)
        self.assertEqual(mask.dtype, np.float32)
        self.assertEqual(len(mask.shape), 3)
