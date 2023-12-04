# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest

import torch

from pytorch3d.implicitron.models.utils import preprocess_input, weighted_sum_losses


class TestUtils(unittest.TestCase):
    def test_prepare_inputs_wrong_num_dim(self):
        img = torch.randn(3, 3, 3)
        text = (
            "Model received unbatched inputs. "
            + "Perhaps they came from a FrameData which had not been collated."
        )
        with self.assertRaisesRegex(ValueError, text):
            img, fg_prob, depth_map = preprocess_input(
                img, None, None, True, True, 0.5, (0.0, 0.0, 0.0)
            )

    def test_prepare_inputs_mask_image_true(self):
        batch, channels, height, width = 2, 3, 10, 10
        img = torch.ones(batch, channels, height, width)
        # Create a mask on the lower triangular matrix
        fg_prob = torch.tril(torch.ones(batch, 1, height, width)) * 0.3

        out_img, out_fg_prob, out_depth_map = preprocess_input(
            img, fg_prob, None, True, False, 0.3, (0.0, 0.0, 0.0)
        )

        self.assertTrue(torch.equal(out_img, torch.tril(img)))
        self.assertTrue(torch.equal(out_fg_prob, fg_prob >= 0.3))
        self.assertIsNone(out_depth_map)

    def test_prepare_inputs_mask_depth_true(self):
        batch, channels, height, width = 2, 3, 10, 10
        img = torch.ones(batch, channels, height, width)
        depth_map = torch.randn(batch, channels, height, width)
        # Create a mask on the lower triangular matrix
        fg_prob = torch.tril(torch.ones(batch, 1, height, width)) * 0.3

        out_img, out_fg_prob, out_depth_map = preprocess_input(
            img, fg_prob, depth_map, False, True, 0.3, (0.0, 0.0, 0.0)
        )

        self.assertTrue(torch.equal(out_img, img))
        self.assertTrue(torch.equal(out_fg_prob, fg_prob >= 0.3))
        self.assertTrue(torch.equal(out_depth_map, torch.tril(depth_map)))

    def test_weighted_sum_losses(self):
        preds = {"a": torch.tensor(2), "b": torch.tensor(2)}
        weights = {"a": 2.0, "b": 0.0}
        loss = weighted_sum_losses(preds, weights)
        self.assertEqual(loss, 4.0)

    def test_weighted_sum_losses_raise_warning(self):
        preds = {"a": torch.tensor(2), "b": torch.tensor(2)}
        weights = {"c": 2.0, "d": 2.0}
        self.assertIsNone(weighted_sum_losses(preds, weights))
