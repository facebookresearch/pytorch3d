import os
import math
import contextlib
import unittest

import numpy as np

import torch
from pytorch3d.implicitron.dataset.blob_loader import (
    _load_image,
    _load_mask,
    _load_depth,
    _load_16big_png_depth,
    _load_1bit_png_mask,
    _load_depth_mask,
)
from pytorch3d.implicitron.dataset.json_index_dataset import JsonIndexDataset
from pytorch3d.implicitron.dataset.blob_loader import BlobLoader
from pytorch3d.renderer.cameras import PerspectiveCameras
from tests.common_testing import TestCaseMixin
from pytorch3d.implicitron.tools.config import expand_args_fields
from pytorch3d.implicitron.tools.config import get_default_args

from tests.implicitron.common_resources import get_skateboard_data


class TestBlobLoader(TestCaseMixin, unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

        category = "skateboard"
        stack = contextlib.ExitStack()
        self.dataset_root, self.path_manager = stack.enter_context(get_skateboard_data())
        self.addCleanup(stack.close)
        frame_file = os.path.join(dataset_root, category, "frame_annotations.jgz")
        sequence_file = os.path.join(dataset_root, category, "sequence_annotations.jgz")
        self.image_size = 256

        expand_args_fields(JsonIndexDataset)

        self.dataset = JsonIndexDataset(
                frame_annotations_file=frame_file,
                sequence_annotations_file=sequence_file,
                dataset_root=self.dataset_root,
                image_height=self.image_size,
                image_width=self.image_size,
                box_crop=True,
                load_point_clouds=True,
                path_manager=self.path_manager,
        )
        index = 7000
        self.entry = self.dataset.frame_annots[index]["frame_annotation"]

    def test_BlobLoader_args(self):
        # test that BlobLoader works with get_default_args
        get_default_args(BlobLoader)

    def test_load_pipeline(self):
        (
            fg_probability,
            mask_path,
            bbox_xywh,
            clamp_bbox_xyxy,
            crop_bbox_xywh,
        ) = self.dataset.blob_loader._load_crop_fg_probability(self.entry)

        assert torch.is_tensor(fg_probability)
        assert mask_path
        assert bbox_xywh
        assert torch.is_tensor(clamp_bbox_xyxy)
        assert torch.is_tensor(crop_bbox_xywh)
        (
            image_rgb,
            image_path,
            mask_crop,
            scale,
        ) = self.dataset.blob_loader._load_crop_images(
            self.entry, fg_probability, clamp_bbox_xyxy,
        )
        assert torch.is_tensor(image_rgb)
        assert image_path
        assert torch.is_tensor(mask_crop)
        assert scale
        (
            depth_map,
            depth_path,
            depth_mask,
        ) = self.dataset.blob_loader._load_mask_depth(
            self.entry, clamp_bbox_xyxy, fg_probability,
        )
        assert torch.is_tensor(depth_map)
        assert torch.is_tensor(depth_path)
        assert torch.is_tensor(depth_mask)

        camera = self.dataset.blob_loader._get_pytorch3d_camera(
                self.entry, scale, clamp_bbox_xyxy,
            )
        assert type(camera) == PerspectiveCameras

    def test_fix_point_cloud_path(self):
        """Some files in Co3Dv2 have an accidental absolute path stored."""
        original_path = 'some_file_path'
        modified_path = self.dataset.blob_loader._fix_point_cloud_path(original_path)
        assert original_path in modified_path
        assert self.dataset.blob_loader.dataset_root in modified_path

    def test_resize_image(self):
        path = os.path.join(self.dataset_root, self.entry.image.path)
        local_path = self.path_manager.get_local_path(path)
        image = _load_image(local_path)
        image_rgb, scale, mask_crop = self.dataset.blob_loader._resize_image(image)

        original_shape = image.shape[-2:]
        expected_shape = (self.dataset.blob_loader.image_width, self.dataset.blob_loader.image_height)
        expected_scale = expected_shape[0] / original_shape[0]

        assert scale == expected_scale
        assert image_rgb.shape[-2:] == expected_shape
        assert mask_crop.shape[-2:] == expected_shape

    def test_load_image(self):
        path = os.path.join(self.dataset_root, self.entry.image.path)
        local_path = self.path_manager.get_local_path(path)
        image = _load_image(local_path)
        assert image.dtype == np.float32
        assert torch.max(image) <= 1.0
        assert torch.min(image) >= 0.0

    def test_load_mask(self):
        mask = _load_mask(self.entry.mask.path)
        assert mask.dtype == np.float32
        assert torch.max(mask) <= 1.0
        assert torch.min(mask) >= 0.0

    def test_load_depth(self):
        path = os.path.join(self.dataset_root, entry_depth.path)
        path = entry_depth.path
        depth_map = _load_depth(path, entry_depth.scale_adjustment)
        assert depth_map.dtype == np.float32
        assert depth_map.shape

    def test_load_16big_png_depth(self):
        path = os.path.join(self.dataset_root, self.entry.depth.path)
        depth_map = _load_16big_png_depth(path)
        assert depth_map.dtype == np.float32
        assert depth_map.shape

    def test_load_1bit_png_mask(self):
        mask_path = os.path.join(self.dataset_root, self.entry.depth.mask_path)
        mask_path = entry_depth.mask_path
        mask = _load_16big_png_depth(mask_path)
        assert mask.dtype == np.float32
        assert mask.shape

    def test_load_depth_mask(self):
        mask_path = os.path.join(self.dataset_root, self.entry.depth.mask_path)
        mask_path = entry_depth.mask_path
        mask = _load_depth_mask(mask_path)
        assert mask.dtype == np.float32
        assert mask.shape
