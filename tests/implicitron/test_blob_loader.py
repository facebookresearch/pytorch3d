import contextlib
import unittest

import numpy as np

import torch
from pytorch3d.implicitron.dataset.blob_loader import (
    _bbox_xywh_to_xyxy,
    _bbox_xyxy_to_xywh,
    _get_bbox_from_mask,
)
from pytorch3d.implicitron.dataset.json_index_dataset import JsonIndexDataset
from pytorch3d.implicitron.dataset.blob_loader import BlobLoader
from tests.common_testing import TestCaseMixin
from pytorch3d.implicitron.tools.config import expand_args_fields
from pytorch3d.implicitron.tools.config import get_default_args


class TestBlobLoader(TestCaseMixin, unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.blob_loader = BlobLoader()

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

    def test_BlobLoader_args(self):
        # test that BlobLoader works with get_default_args
        get_default_args(BlobLoader)

    def test_load_crop_fg_probability(self):
        pass

    def test_load_crop_images(self):
        pass

    def test_load_mask_depth(self):
        pass

    def test_fix_point_cloud_path(self):
        pass

    def test_resize_image(self):
        pass

    def test_crop_around_box(self):
        pass

    def test_clamp_box_to_image_bounds_and_round(self):
        pass

    def test_get_clamp_bbox(self):
        pass

    def test_load_depth(self):
        pass

    def test_load_16big_png_depth(self):
        pass

    def test_rescale_bbox(self):
        pass

    def test_load_1bit_png_mask(self):
        pass

    def test_load_depth_mask(self):
        pass

    def test_get_1d_bounds(self):
        pass
