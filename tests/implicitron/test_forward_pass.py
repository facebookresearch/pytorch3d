# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import unittest

import torch
from omegaconf import DictConfig, OmegaConf
from pytorch3d.implicitron.models.generic_model import GenericModel
from pytorch3d.implicitron.models.renderer.base import EvaluationMode
from pytorch3d.implicitron.tools.config import expand_args_fields, get_default_args
from pytorch3d.renderer.cameras import look_at_view_transform, PerspectiveCameras
from tests.common_testing import get_pytorch3d_dir

from .common_resources import provide_resnet34

IMPLICITRON_CONFIGS_DIR = (
    get_pytorch3d_dir() / "projects" / "implicitron_trainer" / "configs"
)


class TestGenericModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        provide_resnet34()

    def setUp(self):
        torch.manual_seed(42)

    def test_gm(self):
        # Simple test of a forward and backward pass of the default GenericModel.
        device = torch.device("cuda:0")
        expand_args_fields(GenericModel)
        model = GenericModel(render_image_height=80, render_image_width=80)
        model.to(device)
        self._one_model_test(model, device)

    def test_all_gm_configs(self):
        # Tests all model settings in the implicitron_trainer config folder.
        device = torch.device("cuda:0")
        config_files = []

        for pattern in ("repro_singleseq*.yaml", "repro_multiseq*.yaml"):
            config_files.extend(
                [
                    f
                    for f in IMPLICITRON_CONFIGS_DIR.glob(pattern)
                    if not f.name.endswith("_base.yaml")
                ]
            )

        for config_file in config_files:
            with self.subTest(name=config_file.stem):
                cfg = _load_model_config_from_yaml(str(config_file))
                cfg.render_image_height = 80
                cfg.render_image_width = 80
                model = GenericModel(**cfg)
                model.to(device)
                self._one_model_test(
                    model,
                    device,
                    eval_test=True,
                    bw_test=True,
                )

    def _one_model_test(
        self,
        model,
        device,
        n_train_cameras: int = 5,
        eval_test: bool = True,
        bw_test: bool = True,
    ):
        R, T = look_at_view_transform(azim=torch.rand(n_train_cameras) * 360)
        cameras = PerspectiveCameras(R=R, T=T, device=device)

        N, H, W = n_train_cameras, model.render_image_height, model.render_image_width

        random_args = {
            "camera": cameras,
            "fg_probability": _random_input_tensor(N, 1, H, W, True, device),
            "depth_map": _random_input_tensor(N, 1, H, W, False, device) + 0.1,
            "mask_crop": _random_input_tensor(N, 1, H, W, True, device),
            "sequence_name": ["sequence"] * N,
            "image_rgb": _random_input_tensor(N, 3, H, W, False, device),
        }

        # training foward pass
        model.train()
        train_preds = model(
            **random_args,
            evaluation_mode=EvaluationMode.TRAINING,
        )
        self.assertTrue(
            train_preds["objective"].isfinite().item()
        )  # check finiteness of the objective

        if bw_test:
            train_preds["objective"].backward()

        if eval_test:
            model.eval()
            with torch.no_grad():
                eval_preds = model(
                    **random_args,
                    evaluation_mode=EvaluationMode.EVALUATION,
                )
                self.assertEqual(
                    eval_preds["images_render"].shape,
                    (1, 3, model.render_image_height, model.render_image_width),
                )

    def test_idr(self):
        # Forward pass of GenericModel with IDR.
        device = torch.device("cuda:0")
        args = get_default_args(GenericModel)
        args.renderer_class_type = "SignedDistanceFunctionRenderer"
        args.implicit_function_class_type = "IdrFeatureField"
        args.implicit_function_IdrFeatureField_args.n_harmonic_functions_xyz = 6

        model = GenericModel(**args)
        model.to(device)

        n_train_cameras = 2
        R, T = look_at_view_transform(azim=torch.rand(n_train_cameras) * 360)
        cameras = PerspectiveCameras(R=R, T=T, device=device)

        defaulted_args = {
            "depth_map": None,
            "mask_crop": None,
            "sequence_name": None,
        }

        target_image_rgb = torch.rand(
            (n_train_cameras, 3, model.render_image_height, model.render_image_width),
            device=device,
        )
        fg_probability = torch.rand(
            (n_train_cameras, 1, model.render_image_height, model.render_image_width),
            device=device,
        )
        train_preds = model(
            camera=cameras,
            evaluation_mode=EvaluationMode.TRAINING,
            image_rgb=target_image_rgb,
            fg_probability=fg_probability,
            **defaulted_args,
        )
        self.assertGreater(train_preds["objective"].item(), 0)

    def test_viewpool(self):
        device = torch.device("cuda:0")
        args = get_default_args(GenericModel)
        args.view_pooler_enabled = True
        args.image_feature_extractor_class_type = "ResNetFeatureExtractor"
        args.image_feature_extractor_ResNetFeatureExtractor_args.add_masks = False
        model = GenericModel(**args)
        model.to(device)

        n_train_cameras = 2
        R, T = look_at_view_transform(azim=torch.rand(n_train_cameras) * 360)
        cameras = PerspectiveCameras(R=R, T=T, device=device)

        defaulted_args = {
            "fg_probability": None,
            "depth_map": None,
            "mask_crop": None,
        }

        target_image_rgb = torch.rand(
            (n_train_cameras, 3, model.render_image_height, model.render_image_width),
            device=device,
        )
        train_preds = model(
            camera=cameras,
            evaluation_mode=EvaluationMode.TRAINING,
            image_rgb=target_image_rgb,
            sequence_name=["a"] * n_train_cameras,
            **defaulted_args,
        )
        self.assertGreater(train_preds["objective"].item(), 0)


def _random_input_tensor(
    N: int,
    C: int,
    H: int,
    W: int,
    is_binary: bool,
    device: torch.device,
) -> torch.Tensor:
    T = torch.rand(N, C, H, W, device=device)
    if is_binary:
        T = (T > 0.5).float()
    return T


def _load_model_config_from_yaml(config_path, strict=True) -> DictConfig:
    default_cfg = get_default_args(GenericModel)
    cfg = _load_model_config_from_yaml_rec(default_cfg, config_path)
    return cfg


def _load_model_config_from_yaml_rec(cfg: DictConfig, config_path: str) -> DictConfig:
    cfg_loaded = OmegaConf.load(config_path)
    cfg_model_loaded = None
    if "model_factory_ImplicitronModelFactory_args" in cfg_loaded:
        factory_args = cfg_loaded.model_factory_ImplicitronModelFactory_args
        if "model_GenericModel_args" in factory_args:
            cfg_model_loaded = factory_args.model_GenericModel_args
    defaults = cfg_loaded.pop("defaults", None)
    if defaults is not None:
        for default_name in defaults:
            if default_name in ("_self_", "default_config"):
                continue
            default_name = os.path.splitext(default_name)[0]
            defpath = os.path.join(os.path.dirname(config_path), default_name + ".yaml")
            cfg = _load_model_config_from_yaml_rec(cfg, defpath)
            if cfg_model_loaded is not None:
                cfg = OmegaConf.merge(cfg, cfg_model_loaded)
    elif cfg_model_loaded is not None:
        cfg = OmegaConf.merge(cfg, cfg_model_loaded)
    return cfg
