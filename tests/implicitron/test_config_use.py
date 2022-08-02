# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from omegaconf import OmegaConf
from pytorch3d.implicitron.models.feature_extractor.resnet_feature_extractor import (
    ResNetFeatureExtractor,
)
from pytorch3d.implicitron.models.generic_model import GenericModel
from pytorch3d.implicitron.models.global_encoder.global_encoder import (
    SequenceAutodecoder,
)
from pytorch3d.implicitron.models.implicit_function.idr_feature_field import (
    IdrFeatureField,
)
from pytorch3d.implicitron.models.implicit_function.neural_radiance_field import (
    NeuralRadianceFieldImplicitFunction,
)
from pytorch3d.implicitron.models.renderer.lstm_renderer import LSTMRenderer
from pytorch3d.implicitron.models.renderer.multipass_ea import (
    MultiPassEmissionAbsorptionRenderer,
)
from pytorch3d.implicitron.models.view_pooler.feature_aggregator import (
    AngleWeightedIdentityFeatureAggregator,
)
from pytorch3d.implicitron.tools.config import (
    get_default_args,
    remove_unused_components,
)
from tests.common_testing import get_tests_dir

from .common_resources import provide_resnet34

DATA_DIR = get_tests_dir() / "implicitron/data"
DEBUG: bool = False

# Tests the use of the config system in implicitron


class TestGenericModel(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None

    def test_create_gm(self):
        args = get_default_args(GenericModel)
        gm = GenericModel(**args)
        self.assertIsInstance(gm.renderer, MultiPassEmissionAbsorptionRenderer)
        self.assertIsInstance(
            gm._implicit_functions[0]._fn, NeuralRadianceFieldImplicitFunction
        )
        self.assertIsNone(gm.global_encoder)
        self.assertFalse(hasattr(gm, "implicit_function"))
        self.assertIsNone(gm.view_pooler)
        self.assertIsNone(gm.image_feature_extractor)

    def test_create_gm_overrides(self):
        provide_resnet34()
        args = get_default_args(GenericModel)
        args.view_pooler_enabled = True
        args.view_pooler_args.feature_aggregator_class_type = (
            "AngleWeightedIdentityFeatureAggregator"
        )
        args.image_feature_extractor_class_type = "ResNetFeatureExtractor"
        args.implicit_function_class_type = "IdrFeatureField"
        args.global_encoder_class_type = "SequenceAutodecoder"
        idr_args = args.implicit_function_IdrFeatureField_args
        idr_args.n_harmonic_functions_xyz = 1729

        args.renderer_class_type = "LSTMRenderer"
        gm = GenericModel(**args)
        self.assertIsInstance(gm.renderer, LSTMRenderer)
        self.assertIsInstance(
            gm.view_pooler.feature_aggregator,
            AngleWeightedIdentityFeatureAggregator,
        )
        self.assertIsInstance(gm._implicit_functions[0]._fn, IdrFeatureField)
        self.assertEqual(gm._implicit_functions[0]._fn.n_harmonic_functions_xyz, 1729)
        self.assertIsInstance(gm.global_encoder, SequenceAutodecoder)
        self.assertIsInstance(gm.image_feature_extractor, ResNetFeatureExtractor)
        self.assertFalse(hasattr(gm, "implicit_function"))

        instance_args = OmegaConf.structured(gm)
        if DEBUG:
            full_yaml = OmegaConf.to_yaml(instance_args, sort_keys=False)
            (DATA_DIR / "overrides_full.yaml").write_text(full_yaml)
        remove_unused_components(instance_args)
        yaml = OmegaConf.to_yaml(instance_args, sort_keys=False)
        if DEBUG:
            (DATA_DIR / "overrides_.yaml").write_text(yaml)
        self.assertEqual(yaml, (DATA_DIR / "overrides.yaml").read_text())
