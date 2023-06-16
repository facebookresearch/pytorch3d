# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import os
import sys
import unittest
import unittest.mock

from tests.common_testing import get_pytorch3d_dir


# This file groups together tests which look at the code without running it.
class TestBuild(unittest.TestCase):
    def test_no_import_cycles(self):
        # Check each module of pytorch3d imports cleanly,
        # which may fail if there are import cycles.

        with unittest.mock.patch.dict(sys.modules):
            for module in list(sys.modules):
                # If any of pytorch3d is already imported,
                # the test would be pointless.
                if module.startswith("pytorch3d"):
                    sys.modules.pop(module, None)

            # torchvision seems to cause problems if re-imported,
            # so make sure it has been imported here.
            import torchvision.utils  # noqa

            root_dir = get_pytorch3d_dir() / "pytorch3d"
            # Exclude opengl-related files, as Implicitron is decoupled from opengl
            # components which will not work without adding a dep on pytorch3d_opengl.
            ignored_modules = (
                "__init__",
                "plotly_vis",
                "opengl_utils",
                "rasterizer_opengl",
            )
            if os.environ.get("FB_TEST", False):
                ignored_modules += ("orm_types", "sql_dataset", "sql_dataset_provider")
            for module_file in root_dir.glob("**/*.py"):
                if module_file.stem in ignored_modules:
                    continue
                relative_module = str(module_file.relative_to(root_dir))[:-3]
                module = "pytorch3d." + relative_module.replace("/", ".")
                with self.subTest(name=module):
                    with unittest.mock.patch.dict(sys.modules):
                        importlib.import_module(module)
