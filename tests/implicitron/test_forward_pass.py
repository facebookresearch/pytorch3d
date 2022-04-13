# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from pytorch3d.implicitron.models.base import GenericModel
from pytorch3d.implicitron.models.renderer.base import EvaluationMode
from pytorch3d.implicitron.tools.config import expand_args_fields
from pytorch3d.renderer.cameras import look_at_view_transform, PerspectiveCameras


class TestGenericModel(unittest.TestCase):
    def test_gm(self):
        # Simple test of a forward pass of the default GenericModel.
        device = torch.device("cuda:1")
        expand_args_fields(GenericModel)
        model = GenericModel()
        model.to(device)

        n_train_cameras = 2
        R, T = look_at_view_transform(azim=torch.rand(n_train_cameras) * 360)
        cameras = PerspectiveCameras(R=R, T=T, device=device)

        # TODO: make these default to None?
        defaulted_args = {
            "fg_probability": None,
            "depth_map": None,
            "mask_crop": None,
            "sequence_name": None,
        }

        with self.assertWarnsRegex(UserWarning, "No main objective found"):
            model(
                camera=cameras,
                evaluation_mode=EvaluationMode.TRAINING,
                **defaulted_args,
                image_rgb=None,
            )
        target_image_rgb = torch.rand(
            (n_train_cameras, 3, model.render_image_height, model.render_image_width),
            device=device,
        )
        train_preds = model(
            camera=cameras,
            evaluation_mode=EvaluationMode.TRAINING,
            image_rgb=target_image_rgb,
            **defaulted_args,
        )
        self.assertGreater(train_preds["objective"].item(), 0)

        model.eval()
        with torch.no_grad():
            # TODO: perhaps this warning should be skipped in eval mode?
            with self.assertWarnsRegex(UserWarning, "No main objective found"):
                eval_preds = model(
                    camera=cameras[0],
                    **defaulted_args,
                    image_rgb=None,
                )
        self.assertEqual(
            eval_preds["images_render"].shape,
            (1, 3, model.render_image_height, model.render_image_width),
        )
