# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import builtins
import importlib.util
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from . import common_testing


TEST_MODULES = (
    "test_opengl_utils",
    "test_rasterizer",
    "test_rasterize_rectangle_images",
    "test_render_meshes",
    "test_render_multigpu",
)


def _import_test_module(name):
    path = Path(__file__).with_name(f"{name}.py")
    module_name = f"tests._opengl_import_probe_{name}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
    return module


class OpenGLImportPolicyTest(unittest.TestCase):
    def test_disabled_backend_retains_cuda_tests(self):
        original_import = builtins.__import__
        blocked_imports = []

        def import_without_opengl(name, *args, **kwargs):
            if name.startswith(("OpenGL", "pycuda", "pytorch3d.renderer.opengl")):
                blocked_imports.append(name)
                raise ModuleNotFoundError(name, name=name)
            return original_import(name, *args, **kwargs)

        with (
            patch.dict(os.environ, {"PYTORCH3D_NO_TEST_OPENGL": "1"}),
            patch.object(common_testing, "usesOpengl", unittest.skip("uses opengl")),
            patch("builtins.__import__", new=import_without_opengl),
        ):
            utils_module = _import_test_module("test_opengl_utils")
            utils_suite = unittest.TestLoader().loadTestsFromModule(utils_module)
            utils_count = utils_suite.countTestCases()
            self.assertGreater(utils_count, 0)
            utils_result = unittest.TestResult()
            utils_suite.run(utils_result)
            self.assertEqual(len(utils_result.skipped), utils_count)
            modules = {name: _import_test_module(name) for name in TEST_MODULES[1:]}
            for name, module in modules.items():
                with self.subTest(module=name):
                    self.assertIsNone(module.MeshRasterizerOpenGL)
                    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
                    self.assertGreater(suite.countTestCases(), 0)
            rasterizer_tests = modules["test_rasterizer"].TestMeshRasterizer
            renderer_tests = modules["test_render_meshes"].TestRenderMeshes
            self.assertFalse(
                getattr(rasterizer_tests.test_simple_sphere, "__unittest_skip__", False)
            )
            self.assertFalse(
                getattr(renderer_tests.test_simple_sphere, "__unittest_skip__", False)
            )
            self.assertTrue(
                rasterizer_tests.test_simple_sphere_opengl.__unittest_skip__
            )
            self.assertTrue(renderer_tests.test_texture_map_opengl.__unittest_skip__)
        self.assertEqual(blocked_imports, [])

    def test_enabled_backend_requires_dependencies(self):
        original_import = builtins.__import__

        def import_without_opengl(name, *args, **kwargs):
            if name.startswith(("OpenGL", "pycuda", "pytorch3d.renderer.opengl")):
                raise ModuleNotFoundError(name, name=name)
            return original_import(name, *args, **kwargs)

        with (
            patch.dict(os.environ, {"PYTORCH3D_NO_TEST_OPENGL": "0"}),
            patch("builtins.__import__", new=import_without_opengl),
        ):
            for name in TEST_MODULES:
                with self.subTest(module=name), self.assertRaises(ModuleNotFoundError):
                    _import_test_module(name)
