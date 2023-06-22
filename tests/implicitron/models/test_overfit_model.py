# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import Any, Dict
from unittest.mock import patch

import torch
from pytorch3d.implicitron.models.generic_model import GenericModel
from pytorch3d.implicitron.models.overfit_model import OverfitModel
from pytorch3d.implicitron.models.renderer.base import EvaluationMode
from pytorch3d.implicitron.tools.config import expand_args_fields
from pytorch3d.renderer.cameras import look_at_view_transform, PerspectiveCameras

DEVICE = torch.device("cuda:0")


def _generate_fake_inputs(N: int, H: int, W: int) -> Dict[str, Any]:
    R, T = look_at_view_transform(azim=torch.rand(N) * 360)
    return {
        "camera": PerspectiveCameras(R=R, T=T, device=DEVICE),
        "fg_probability": torch.randint(
            high=2, size=(N, 1, H, W), device=DEVICE
        ).float(),
        "depth_map": torch.rand((N, 1, H, W), device=DEVICE) + 0.1,
        "mask_crop": torch.randint(high=2, size=(N, 1, H, W), device=DEVICE).float(),
        "sequence_name": ["sequence"] * N,
        "image_rgb": torch.rand((N, 1, H, W), device=DEVICE),
    }


def mock_safe_multinomial(input: torch.Tensor, num_samples: int) -> torch.Tensor:
    """Return non deterministic indexes to mock safe_multinomial

    Args:
        input: tensor of shape [B, n] containing non-negative values;
                rows are interpreted as unnormalized event probabilities
                in categorical distributions.
        num_samples: number of samples to take.

    Returns:
        Tensor of shape [B, num_samples]
    """
    batch_size = input.shape[0]
    return torch.arange(num_samples).repeat(batch_size, 1).to(DEVICE)


class TestOverfitModel(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_overfit_model_vs_generic_model_with_batch_size_one(self):
        """In this test we compare OverfitModel to GenericModel behavior.

        We use a Nerf setup (2 rendering passes).

        OverfitModel is a specific case of GenericModel. Hence, with the same inputs,
        they should provide the exact same results.
        """
        expand_args_fields(OverfitModel)
        expand_args_fields(GenericModel)
        batch_size, image_height, image_width = 1, 80, 80
        assert batch_size == 1
        overfit_model = OverfitModel(
            render_image_height=image_height,
            render_image_width=image_width,
            coarse_implicit_function_class_type="NeuralRadianceFieldImplicitFunction",
            # To avoid randomization to compare the outputs of our model
            # we deactivate the stratified_point_sampling_training
            raysampler_AdaptiveRaySampler_args={
                "stratified_point_sampling_training": False
            },
            global_encoder_class_type="SequenceAutodecoder",
            global_encoder_SequenceAutodecoder_args={
                "autodecoder_args": {
                    "n_instances": 1000,
                    "init_scale": 1.0,
                    "encoding_dim": 64,
                }
            },
        )
        generic_model = GenericModel(
            render_image_height=image_height,
            render_image_width=image_width,
            n_train_target_views=batch_size,
            num_passes=2,
            # To avoid randomization to compare the outputs of our model
            # we deactivate the stratified_point_sampling_training
            raysampler_AdaptiveRaySampler_args={
                "stratified_point_sampling_training": False
            },
            global_encoder_class_type="SequenceAutodecoder",
            global_encoder_SequenceAutodecoder_args={
                "autodecoder_args": {
                    "n_instances": 1000,
                    "init_scale": 1.0,
                    "encoding_dim": 64,
                }
            },
        )

        # Check if they do share the number of parameters
        num_params_mvm = sum(p.numel() for p in overfit_model.parameters())
        num_params_gm = sum(p.numel() for p in generic_model.parameters())
        self.assertEqual(num_params_mvm, num_params_gm)

        # Adapt the mapping from generic model to overfit model
        mapping_om_from_gm = {
            key.replace(
                "_implicit_functions.0._fn", "coarse_implicit_function"
            ).replace("_implicit_functions.1._fn", "implicit_function"): val
            for key, val in generic_model.state_dict().items()
        }
        # Copy parameters from generic_model to overfit_model
        overfit_model.load_state_dict(mapping_om_from_gm)

        overfit_model.to(DEVICE)
        generic_model.to(DEVICE)
        inputs_ = _generate_fake_inputs(batch_size, image_height, image_width)

        # training forward pass
        overfit_model.train()
        generic_model.train()

        with patch(
            "pytorch3d.renderer.implicit.raysampling._safe_multinomial",
            side_effect=mock_safe_multinomial,
        ):
            train_preds_om = overfit_model(
                **inputs_,
                evaluation_mode=EvaluationMode.TRAINING,
            )
            train_preds_gm = generic_model(
                **inputs_,
                evaluation_mode=EvaluationMode.TRAINING,
            )

        self.assertTrue(len(train_preds_om) == len(train_preds_gm))

        self.assertTrue(train_preds_om["objective"].isfinite().item())
        # We avoid all the randomization and the weights are the same
        # The objective should be the same
        self.assertTrue(
            torch.allclose(train_preds_om["objective"], train_preds_gm["objective"])
        )

        # Test if the evaluation works
        overfit_model.eval()
        generic_model.eval()
        with torch.no_grad():
            eval_preds_om = overfit_model(
                **inputs_,
                evaluation_mode=EvaluationMode.EVALUATION,
            )
            eval_preds_gm = generic_model(
                **inputs_,
                evaluation_mode=EvaluationMode.EVALUATION,
            )

        self.assertEqual(
            eval_preds_om["images_render"].shape,
            (batch_size, 3, image_height, image_width),
        )
        self.assertTrue(
            torch.allclose(eval_preds_om["objective"], eval_preds_gm["objective"])
        )
        self.assertTrue(
            torch.allclose(
                eval_preds_om["images_render"], eval_preds_gm["images_render"]
            )
        )

    def test_overfit_model_check_share_weights(self):
        model = OverfitModel(share_implicit_function_across_passes=True)
        for p1, p2 in zip(
            model.implicit_function.parameters(),
            model.coarse_implicit_function.parameters(),
        ):
            self.assertEqual(id(p1), id(p2))

        model.to(DEVICE)
        inputs_ = _generate_fake_inputs(2, 80, 80)
        model(**inputs_, evaluation_mode=EvaluationMode.TRAINING)

    def test_overfit_model_check_no_share_weights(self):
        model = OverfitModel(
            share_implicit_function_across_passes=False,
            coarse_implicit_function_class_type="NeuralRadianceFieldImplicitFunction",
            coarse_implicit_function_NeuralRadianceFieldImplicitFunction_args={
                "transformer_dim_down_factor": 1.0,
                "n_hidden_neurons_xyz": 256,
                "n_layers_xyz": 8,
                "append_xyz": (5,),
            },
        )
        for p1, p2 in zip(
            model.implicit_function.parameters(),
            model.coarse_implicit_function.parameters(),
        ):
            self.assertNotEqual(id(p1), id(p2))

        model.to(DEVICE)
        inputs_ = _generate_fake_inputs(2, 80, 80)
        model(**inputs_, evaluation_mode=EvaluationMode.TRAINING)

    def test_overfit_model_coarse_implicit_function_is_none(self):
        model = OverfitModel(
            share_implicit_function_across_passes=False,
            coarse_implicit_function_NeuralRadianceFieldImplicitFunction_args=None,
        )
        self.assertIsNone(model.coarse_implicit_function)
        model.to(DEVICE)
        inputs_ = _generate_fake_inputs(2, 80, 80)
        model(**inputs_, evaluation_mode=EvaluationMode.TRAINING)
