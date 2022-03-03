# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest

import torch
from pytorch3d.renderer.mesh.utils import _clip_barycentric_coordinates


class TestMeshRenderingUtils(unittest.TestCase):
    def test_bary_clip(self):
        N = 10
        bary = torch.randn(size=(N, 3))
        # randomly make some values negative
        bary[bary < 0.3] *= -1.0
        # randomly make some values be greater than 1
        bary[bary > 0.8] *= 2.0
        negative_mask = bary < 0.0
        positive_mask = bary > 1.0
        clipped = _clip_barycentric_coordinates(bary)
        self.assertTrue(clipped[negative_mask].sum() == 0)
        self.assertTrue(clipped[positive_mask].gt(1.0).sum() == 0)
        self.assertTrue(torch.allclose(clipped.sum(dim=-1), torch.ones(N)))
