import contextlib
import os
import unittest

import numpy as np

import torch
from pytorch3d.implicitron.dataset.blob_loader import (
    _load_16big_png_depth,
    _load_1bit_png_mask,
    _load_depth,
    _load_depth_mask,
    _load_image,
    _load_mask,
    BlobLoader,
)
from pytorch3d.implicitron.dataset.json_index_dataset import JsonIndexDataset
from pytorch3d.implicitron.tools.config import expand_args_fields, get_default_args
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
        frame_file = os.path.join(self.dataset_root, category, "frame_annotations.jgz")
        sequence_file = os.path.join(
            self.dataset_root, category, "sequence_annotations.jgz"
        )
        self.image_height = 768
        self.image_width = 512

        expand_args_fields(JsonIndexDataset)

        self.dataset = JsonIndexDataset(
            frame_annotations_file=frame_file,
            sequence_annotations_file=sequence_file,
            dataset_root=self.dataset_root,
            image_height=self.image_height,
            image_width=self.image_width,
            box_crop=True,
            load_point_clouds=True,
            path_manager=self.path_manager,
        )

    def test_BlobLoader_args(self):
        # test that BlobLoader works with get_default_args
        get_default_args(BlobLoader)

    def test_fix_point_cloud_path(self):
        """Some files in Co3Dv2 have an accidental absolute path stored."""
        original_path = "some_file_path"
        modified_path = self.dataset.blob_loader._fix_point_cloud_path(original_path)
        assert original_path in modified_path
        assert self.dataset.blob_loader.dataset_root in modified_path

    def test_entry_loading_functions(self):
        for index in range(len(self.dataset.frame_annots)):
            entry = self.dataset.frame_annots[index]["frame_annotation"]
            self.load_test(entry)
            self._resize_image_test(entry)
            self._load_image_test(entry)
            self._load_mask_test(entry)
            self._load_depth_test(entry)
            self._load_16big_png_depth_test(entry)
            self._load_1bit_png_mask_test(entry)
            self._load_depth_mask_test(entry)

    def load_test(self, entry):
        (
            fg_probability,
            mask_path,
            bbox_xywh,
            clamp_bbox_xyxy,
            crop_bbox_xywh,
        ) = self.dataset.blob_loader._load_crop_fg_probability(entry)

        assert mask_path
        assert torch.is_tensor(fg_probability)
        assert torch.is_tensor(bbox_xywh)
        assert torch.is_tensor(clamp_bbox_xyxy)
        assert torch.is_tensor(crop_bbox_xywh)
        # assert bboxes shape
        self.assertEqual(fg_probability.shape, torch.Size([1, self.image_height, self.image_width]))
        self.assertEqual(bbox_xywh.shape, torch.Size([4]))
        self.assertEqual(clamp_bbox_xyxy.shape, torch.Size([4]))
        self.assertEqual(crop_bbox_xywh.shape, torch.Size([4]))
        (
            image_rgb,
            image_path,
            mask_crop,
            scale,
        ) = self.dataset.blob_loader._load_crop_images(
            entry,
            fg_probability,
            clamp_bbox_xyxy,
        )
        assert torch.is_tensor(image_rgb)
        assert image_path
        assert torch.is_tensor(mask_crop)
        assert scale
        # assert image and mask shapes
        self.assertEqual(image_rgb.shape, torch.Size([3, self.image_height, self.image_width]))
        self.assertEqual(mask_crop.shape, torch.Size([1, self.image_height, self.image_width])

        (
            depth_map,
            depth_path,
            depth_mask,
        ) = self.dataset.blob_loader._load_mask_depth(
            entry,
            clamp_bbox_xyxy,
            fg_probability,
        )
        assert torch.is_tensor(depth_map)
        assert depth_path
        assert torch.is_tensor(depth_mask)
        # assert image and mask shapes
        self.assertEqual(depth_map.shape, torch.Size([1, self.image_height, self.image_width]))
        self.assertEqual(depth_mask.shape, torch.Size([1, self.image_height, self.image_width]))

        camera = self.dataset.blob_loader._get_pytorch3d_camera(
            entry,
            scale,
            clamp_bbox_xyxy,
        )
        self.assertEqual(type(camera), PerspectiveCameras)

    def _resize_image_test(self, entry):
        path = os.path.join(self.dataset_root, entry.image.path)
        local_path = self.path_manager.get_local_path(path)
        image = _load_image(local_path)
        image_rgb, scale, mask_crop = self.dataset.blob_loader._resize_image(image)

        original_shape = image.shape[-2:]
        expected_shape = (
            self.image_height,
            self.image_width,
        )
        expected_scale = min(
            expected_shape[0] / original_shape[0], expected_shape[1] / original_shape[1]
        )

        self.assertEqual(scale, expected_scale)
        self.assertEqual(image_rgb.shape[-2:], expected_shape)
        self.assertEqual(mask_crop.shape[-2:], expected_shape)

    def _load_image_test(self, entry):
        path = os.path.join(self.dataset_root, entry.image.path)
        local_path = self.path_manager.get_local_path(path)
        image = _load_image(local_path)
        self.assertEqual(image.dtype, np.float32)
        assert np.max(image) <= 1.0
        assert np.min(image) >= 0.0

    def _load_mask_test(self, entry):
        path = os.path.join(self.dataset_root, entry.mask.path)
        mask = _load_mask(path)
        self.assertEqual(mask.dtype, np.float32)
        assert np.max(mask) <= 1.0
        assert np.min(mask) >= 0.0

    def _load_depth_test(self, entry):
        path = os.path.join(self.dataset_root, entry.depth.path)
        depth_map = _load_depth(path, entry.depth.scale_adjustment)
        self.assertEqual(depth_map.dtype, np.float32)
        self.assertEqual(len(depth_map.shape), 2)

    def _load_16big_png_depth_test(self, entry):
        path = os.path.join(self.dataset_root, entry.depth.path)
        depth_map = _load_16big_png_depth(path)
        self.assertEqual(depth_map.dtype, np.float32)
        self.assertEqual(len(depth_map.shape), 2)

    def _load_1bit_png_mask_test(self, entry):
        mask_path = os.path.join(self.dataset_root, entry.depth.mask_path)
        mask = _load_1bit_png_mask(mask_path)
        self.assertEqual(mask.dtype, np.float32)
        self.assertEqual(len(mask.shape), 2)

    def _load_depth_mask_test(self, entry):
        mask_path = os.path.join(self.dataset_root, entry.depth.mask_path)
        mask = _load_depth_mask(mask_path)
        self.assertEqual(mask.dtype, np.float32)
        self.assertEqual(len(mask.shape), 3)
