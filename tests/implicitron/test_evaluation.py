# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import contextlib
import dataclasses
import itertools
import math
import os
import unittest

import lpips
import numpy as np
import torch

from pytorch3d.implicitron.dataset.frame_data import FrameData
from pytorch3d.implicitron.dataset.json_index_dataset import JsonIndexDataset
from pytorch3d.implicitron.evaluation.evaluate_new_view_synthesis import eval_batch
from pytorch3d.implicitron.models.base_model import ImplicitronModelBase
from pytorch3d.implicitron.models.generic_model import GenericModel  # noqa
from pytorch3d.implicitron.models.model_dbir import ModelDBIR  # noqa
from pytorch3d.implicitron.tools.config import expand_args_fields, registry
from pytorch3d.implicitron.tools.metric_utils import calc_psnr, eval_depth
from pytorch3d.implicitron.tools.utils import dataclass_to_cuda_

from .common_resources import get_skateboard_data, provide_lpips_vgg


class TestEvaluation(unittest.TestCase):
    def setUp(self):
        # initialize evaluation dataset/dataloader
        torch.manual_seed(42)

        stack = contextlib.ExitStack()
        dataset_root, path_manager = stack.enter_context(get_skateboard_data())
        self.addCleanup(stack.close)

        category = "skateboard"
        frame_file = os.path.join(dataset_root, category, "frame_annotations.jgz")
        sequence_file = os.path.join(dataset_root, category, "sequence_annotations.jgz")
        self.image_size = 64
        expand_args_fields(JsonIndexDataset)
        self.dataset = JsonIndexDataset(
            frame_annotations_file=frame_file,
            sequence_annotations_file=sequence_file,
            dataset_root=dataset_root,
            image_height=self.image_size,
            image_width=self.image_size,
            box_crop=True,
            remove_empty_masks=False,
            path_manager=path_manager,
        )
        self.bg_color = (0.0, 0.0, 0.0)

        # init the lpips model for eval
        provide_lpips_vgg()
        self.lpips_model = lpips.LPIPS(net="vgg").cuda()

    def test_eval_depth(self):
        """
        Check that eval_depth correctly masks errors and that, for get_best_scale=True,
        the error with scaled prediction equals the error without scaling the
        predicted depth. Finally, test that the error values are as expected
        for prediction and gt differing by a constant offset.
        """
        gt = (torch.randn(10, 1, 300, 400, device="cuda") * 5.0).clamp(0.0)
        mask = (torch.rand_like(gt) > 0.5).type_as(gt)

        for diff in 10 ** torch.linspace(-5, 0, 6):
            for crop in (0, 5):
                pred = gt + (torch.rand_like(gt) - 0.5) * 2 * diff

                # scaled prediction test
                mse_depth, abs_depth = eval_depth(
                    pred,
                    gt,
                    crop=crop,
                    mask=mask,
                    get_best_scale=True,
                )
                mse_depth_scale, abs_depth_scale = eval_depth(
                    pred * 10.0,
                    gt,
                    crop=crop,
                    mask=mask,
                    get_best_scale=True,
                )
                self.assertAlmostEqual(
                    float(mse_depth.sum()), float(mse_depth_scale.sum()), delta=1e-4
                )
                self.assertAlmostEqual(
                    float(abs_depth.sum()), float(abs_depth_scale.sum()), delta=1e-4
                )

                # error masking test
                pred_masked_err = gt + (torch.rand_like(gt) + diff) * (1 - mask)
                mse_depth_masked, abs_depth_masked = eval_depth(
                    pred_masked_err,
                    gt,
                    crop=crop,
                    mask=mask,
                    get_best_scale=True,
                )
                self.assertAlmostEqual(
                    float(mse_depth_masked.sum()), float(0.0), delta=1e-4
                )
                self.assertAlmostEqual(
                    float(abs_depth_masked.sum()), float(0.0), delta=1e-4
                )
                mse_depth_unmasked, abs_depth_unmasked = eval_depth(
                    pred_masked_err,
                    gt,
                    crop=crop,
                    mask=1 - mask,
                    get_best_scale=True,
                )
                self.assertGreater(
                    float(mse_depth_unmasked.sum()),
                    float(diff**2),
                )
                self.assertGreater(
                    float(abs_depth_unmasked.sum()),
                    float(diff),
                )

                # tests with constant error
                pred_fix_diff = gt + diff * mask
                for _mask_gt in (mask, None):
                    mse_depth_fix_diff, abs_depth_fix_diff = eval_depth(
                        pred_fix_diff,
                        gt,
                        crop=crop,
                        mask=_mask_gt,
                        get_best_scale=False,
                    )
                    if _mask_gt is not None:
                        expected_err_abs = diff
                        expected_err_mse = diff**2
                    else:
                        err_mask = (gt > 0.0).float() * mask
                        if crop > 0:
                            err_mask = err_mask[:, :, crop:-crop, crop:-crop]
                            gt_cropped = gt[:, :, crop:-crop, crop:-crop]
                        else:
                            gt_cropped = gt
                        gt_mass = (gt_cropped > 0.0).float().sum(dim=(1, 2, 3))
                        expected_err_abs = (
                            diff * err_mask.sum(dim=(1, 2, 3)) / (gt_mass)
                        )
                        expected_err_mse = diff * expected_err_abs
                    self.assertTrue(
                        torch.allclose(
                            abs_depth_fix_diff,
                            expected_err_abs * torch.ones_like(abs_depth_fix_diff),
                            atol=1e-4,
                        )
                    )
                    self.assertTrue(
                        torch.allclose(
                            mse_depth_fix_diff,
                            expected_err_mse * torch.ones_like(mse_depth_fix_diff),
                            atol=1e-4,
                        )
                    )

    def test_psnr(self):
        """
        Compare against opencv and check that the psnr is above
        the minimum possible value.
        """
        import cv2

        im1 = torch.rand(100, 3, 256, 256).cuda()
        im1_uint8 = (im1 * 255).to(torch.uint8)
        im1_rounded = im1_uint8.float() / 255
        for max_diff in 10 ** torch.linspace(-5, 0, 6):
            im2 = im1 + (torch.rand_like(im1) - 0.5) * 2 * max_diff
            im2 = im2.clamp(0.0, 1.0)
            im2_uint8 = (im2 * 255).to(torch.uint8)
            im2_rounded = im2_uint8.float() / 255
            # check that our psnr matches the output of opencv
            psnr = calc_psnr(im1_rounded, im2_rounded)
            # some versions of cv2 can only take uint8 input
            psnr_cv2 = cv2.PSNR(
                im1_uint8.cpu().numpy(),
                im2_uint8.cpu().numpy(),
            )
            self.assertAlmostEqual(float(psnr), float(psnr_cv2), delta=1e-4)
            # check that all PSNRs are bigger than the minimum possible PSNR
            max_mse = max_diff**2
            min_psnr = 10 * math.log10(1.0 / max_mse)
            for _im1, _im2 in zip(im1, im2):
                _psnr = calc_psnr(_im1, _im2)
                self.assertGreaterEqual(float(_psnr) + 1e-6, min_psnr)

    def _one_sequence_test(
        self,
        seq_dataset,
        model,
        batch_indices,
        check_metrics=False,
    ):
        loader = torch.utils.data.DataLoader(
            seq_dataset,
            shuffle=False,
            batch_sampler=batch_indices,
            collate_fn=FrameData.collate,
        )

        for frame_data in loader:
            self.assertIsNone(frame_data.frame_type)
            self.assertIsNotNone(frame_data.image_rgb)
            # override the frame_type
            frame_data.frame_type = [
                "train_unseen",
                *(["train_known"] * (len(frame_data.image_rgb) - 1)),
            ]

            frame_data = dataclass_to_cuda_(frame_data)
            preds = model(**dataclasses.asdict(frame_data))

            eval_result = eval_batch(
                frame_data,
                preds["implicitron_render"],
                bg_color=self.bg_color,
                lpips_model=self.lpips_model,
            )

            if check_metrics:
                self._check_metrics(
                    frame_data, preds["implicitron_render"], eval_result
                )

    def _check_metrics(self, frame_data, implicitron_render, eval_result):
        # Make a terribly bad NVS prediction and check that this is worse
        # than the DBIR prediction.
        implicitron_render_bad = implicitron_render.clone()
        implicitron_render_bad.depth_render += (
            torch.randn_like(implicitron_render_bad.depth_render) * 100.0
        )
        implicitron_render_bad.image_render += (
            torch.randn_like(implicitron_render_bad.image_render) * 100.0
        )
        implicitron_render_bad.mask_render = (
            torch.randn_like(implicitron_render_bad.mask_render) > 0.0
        ).float()
        eval_result_bad = eval_batch(
            frame_data,
            implicitron_render_bad,
            bg_color=self.bg_color,
            lpips_model=self.lpips_model,
        )

        lower_better = {
            "psnr_masked": False,
            "psnr_fg": False,
            "psnr_full_image": False,
            "depth_abs_fg": True,
            "iou": False,
            "rgb_l1_masked": True,
            "rgb_l1_fg": True,
            "lpips_masked": True,
            "lpips_full_image": True,
        }

        for metric in lower_better:
            m_better = eval_result[metric]
            m_worse = eval_result_bad[metric]
            if np.isnan(m_better) or np.isnan(m_worse):
                continue  # metric is missing, i.e. NaN
            _assert = (
                self.assertLessEqual
                if lower_better[metric]
                else self.assertGreaterEqual
            )
            _assert(m_better, m_worse)

    def _get_random_batch_indices(
        self, seq_dataset, n_batches=2, min_batch_size=5, max_batch_size=10
    ):
        batch_indices = []
        for _ in range(n_batches):
            batch_size = torch.randint(
                low=min_batch_size, high=max_batch_size, size=(1,)
            )
            batch_indices.append(torch.randperm(len(seq_dataset))[:batch_size])

        return batch_indices

    def test_full_eval(self, n_sequences=5):
        """Test evaluation."""

        # caching batch indices first to preserve RNG state
        seq_datasets = {}
        batch_indices = {}
        for seq in itertools.islice(self.dataset.sequence_names(), n_sequences):
            idx = list(self.dataset.sequence_indices_in_order(seq))
            seq_dataset = torch.utils.data.Subset(self.dataset, idx)
            seq_datasets[seq] = seq_dataset
            batch_indices[seq] = self._get_random_batch_indices(seq_dataset)

        for model_class_type in ["ModelDBIR", "GenericModel"]:
            ModelClass = registry.get(ImplicitronModelBase, model_class_type)
            expand_args_fields(ModelClass)
            model = ModelClass(
                render_image_width=self.image_size,
                render_image_height=self.image_size,
                bg_color=self.bg_color,
            )
            model.eval()
            model.cuda()

            for seq in itertools.islice(self.dataset.sequence_names(), n_sequences):
                self._one_sequence_test(
                    seq_datasets[seq],
                    model,
                    batch_indices[seq],
                    check_metrics=(model_class_type == "ModelDBIR"),
                )
