# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import unittest

import torch
from pytorch3d.implicitron.dataset.frame_data import FrameData
from pytorch3d.implicitron.dataset.rendered_mesh_dataset_map_provider import (
    RenderedMeshDatasetMapProvider,
)
from pytorch3d.implicitron.tools.config import expand_args_fields
from pytorch3d.renderer import FoVPerspectiveCameras
from tests.common_testing import TestCaseMixin


inside_re_worker = os.environ.get("INSIDE_RE_WORKER", False)


class TestDataCow(TestCaseMixin, unittest.TestCase):
    def test_simple(self):
        if inside_re_worker:
            return
        expand_args_fields(RenderedMeshDatasetMapProvider)
        self._runtest(use_point_light=True, num_views=4)
        self._runtest(use_point_light=False, num_views=4)

    def _runtest(self, **kwargs):
        provider = RenderedMeshDatasetMapProvider(**kwargs)
        dataset_map = provider.get_dataset_map()
        known_matrix = torch.zeros(1, 4, 4)
        known_matrix[0, 0, 0] = 1.7321
        known_matrix[0, 1, 1] = 1.7321
        known_matrix[0, 2, 2] = 1.0101
        known_matrix[0, 3, 2] = -1.0101
        known_matrix[0, 2, 3] = 1

        self.assertIsNone(dataset_map.val)
        self.assertIsNone(dataset_map.test)
        self.assertEqual(len(dataset_map.train), provider.num_views)

        value = dataset_map.train[0]
        self.assertIsInstance(value, FrameData)

        self.assertEqual(value.image_rgb.shape, (3, 128, 128))
        self.assertEqual(value.fg_probability.shape, (1, 128, 128))
        # corner of image is background
        self.assertEqual(value.fg_probability[0, 0, 0], 0)
        self.assertEqual(value.fg_probability.max(), 1.0)
        self.assertIsInstance(value.camera, FoVPerspectiveCameras)
        self.assertEqual(len(value.camera), 1)
        self.assertIsNone(value.camera.K)
        matrix = value.camera.get_projection_transform().get_matrix()
        self.assertClose(matrix, known_matrix, atol=1e-4)
