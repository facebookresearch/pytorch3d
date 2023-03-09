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
from tests.common_testing import TestCaseMixin
from pytorch3d.implicitron.tools.config import expand_args_fields
from pytorch3d.implicitron.tools.config import get_default_args

from tests.implicitron.common_resources import get_skateboard_data


class TestBlobLoader(TestCaseMixin, unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

        category = "skateboard"
        stack = contextlib.ExitStack()
        dataset_root, path_manager = stack.enter_context(get_skateboard_data())
        self.addCleanup(stack.close)
        frame_file = os.path.join(dataset_root, category, "frame_annotations.jgz")
        sequence_file = os.path.join(dataset_root, category, "sequence_annotations.jgz")
        self.image_size = 256

        expand_args_fields(JsonIndexDataset)

        self.datasets = JsonIndexDataset(
                frame_annotations_file=frame_file,
                sequence_annotations_file=sequence_file,
                dataset_root=dataset_root,
                image_height=self.image_size,
                image_width=self.image_size,
                box_crop=True,
                load_point_clouds=True,
                path_manager=path_manager,
        )
        self.entry = self.datasets.frame_annots[index]["frame_annotation"]

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
        ) = self.datasets.loader._load_crop_fg_probability(entry)

        assert fg_probability
        assert mask_path
        assert bbox_xywh
        assert clamp_bbox_xyxy
        assert crop_bbox_xywh
        (
            image_rgb,
            image_path,
            mask_crop,
            scale,
        ) = self.dataset.loader._load_crop_images(
            self.entry, fg_probability, clamp_bbox_xyxy,
        )
        assert image_rgb
        assert image_path
        assert mask_crop
        assert scale
        (
            depth_map,
            depth_path,
            depth_mask,
        ) = self.dataset.loader._load_mask_depth(
            self.entry, clamp_bbox_xyxy, fg_probability,
        )
        assert depth_map
        assert depth_path
        assert depth_mask

        camera = self.dataset.loader._get_pytorch3d_camera(
                self.entry, scale, clamp_bbox_xyxy,
            )
        assert camera

    def test_fix_point_cloud_path(self):
        """Some files in Co3Dv2 have an accidental absolute path stored."""
        original_path = 'some_file_path'
        modified_path = self.dataset.loader._fix_point_cloud_path(original_path)
        assert original_path in modified_path
        assert self.dataset.loader.dataset_root in modified_path

    def test_resize_image(self):
        image = None
        image_rgb, scale, mask_crop = self.dataset.loader._resize_image(image)
        assert image_rgb.shape == (self.dataset.loader.width, self.dataset.loader.height)
        assert scale == 1
        assert masc_crop.shape == (self.dataset.loader.width, self.dataset.loader.height)

    def test_load_image(self):
        image = _load_image(self.entry.image.path)
        assert image.dtype == np.float32
        assert torch.max(image) <= 1.0
        assert torch.min(image) >= 0.0

    def test_load_mask(self):
        mask = _load_mask(self.entry.mask.path)
        assert mask.dtype == np.float32
        assert torch.max(mask) <= 1.0
        assert torch.min(mask) >= 0.0

    def test_load_depth(self):
        entry_depth = self.entry.depth
        # path = os.path.join(self.dataset_root, entry_depth.path)
        path = entry_depth.path
        depth_map = _load_depth(path, entry_depth.scale_adjustment)
        assert depth_map.dtype == np.float32
        assert depth_map.shape

    def test_load_16big_png_depth(self):
        entry_depth = self.entry.depth
        # path = os.path.join(self.dataset_root, entry_depth.path)
        path = entry_depth.path
        depth_map = _load_16big_png_depth(path)
        assert depth_map.dtype == np.float32
        assert depth_map.shape

    def test_load_1bit_png_mask(self):
        entry_depth = self.entry.depth
        # mask_path = os.path.join(self.dataset_root, entry_depth.mask_path)
        mask_path = entry_depth.mask_path
        mask = _load_16big_png_depth(mask_path)
        assert mask.dtype == np.float32
        assert mask.shape

    def test_load_depth_mask(self):
        entry_depth = self.entry.depth
        # mask_path = os.path.join(self.dataset_root, entry_depth.mask_path)
        mask_path = entry_depth.mask_path
        mask = _load_depth_mask(mask_path)
        assert mask.dtype == np.float32
        assert mask.shape
