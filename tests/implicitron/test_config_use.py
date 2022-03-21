# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import unittest

from omegaconf import OmegaConf
from pytorch3d.implicitron.models.autodecoder import Autodecoder
from pytorch3d.implicitron.models.base import GenericModel
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
from pytorch3d.implicitron.models.view_pooling.feature_aggregation import (
    AngleWeightedIdentityFeatureAggregator,
    AngleWeightedReductionFeatureAggregator,
)
from pytorch3d.implicitron.tools.config import (
    get_default_args,
    remove_unused_components,
)


if os.environ.get("FB_TEST", False):
    from common_testing import get_tests_dir
else:
    from tests.common_testing import get_tests_dir

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
            gm.feature_aggregator, AngleWeightedReductionFeatureAggregator
        )
        self.assertIsInstance(
            gm._implicit_functions[0]._fn, NeuralRadianceFieldImplicitFunction
        )
        self.assertIsInstance(gm.sequence_autodecoder, Autodecoder)
        self.assertFalse(hasattr(gm, "implicit_function"))
        self.assertFalse(hasattr(gm, "image_feature_extractor"))

    def test_create_gm_overrides(self):
        args = get_default_args(GenericModel)
        args.feature_aggregator_class_type = "AngleWeightedIdentityFeatureAggregator"
        args.implicit_function_class_type = "IdrFeatureField"
        args.renderer_class_type = "LSTMRenderer"
        gm = GenericModel(**args)
        self.assertIsInstance(gm.renderer, LSTMRenderer)
        self.assertIsInstance(
            gm.feature_aggregator, AngleWeightedIdentityFeatureAggregator
        )
        self.assertIsInstance(gm._implicit_functions[0]._fn, IdrFeatureField)
        self.assertIsInstance(gm.sequence_autodecoder, Autodecoder)
        self.assertFalse(hasattr(gm, "implicit_function"))

        instance_args = OmegaConf.structured(gm)
        remove_unused_components(instance_args)
        yaml = OmegaConf.to_yaml(instance_args, sort_keys=False)
        if DEBUG:
            (DATA_DIR / "overrides.yaml_").write_text(yaml)
        self.assertEqual(yaml, (DATA_DIR / "overrides.yaml").read_text())
