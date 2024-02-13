# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import ctypes
import os
import sys
import threading
import unittest

import torch

os.environ["PYOPENGL_PLATFORM"] = "egl"
import pycuda._driver  # noqa
from OpenGL import GL as gl  # noqa
from OpenGL.raw.EGL._errors import EGLError  # noqa
from pytorch3d.renderer.opengl import _can_import_egl_and_pycuda  # noqa
from pytorch3d.renderer.opengl.opengl_utils import (  # noqa
    _define_egl_extension,
    _egl_convert_to_int_array,
    _get_cuda_device,
    egl,
    EGLContext,
    global_device_context_store,
)

from .common_testing import TestCaseMixin, usesOpengl  # noqa

MAX_EGL_HEIGHT = global_device_context_store.max_egl_height
MAX_EGL_WIDTH = global_device_context_store.max_egl_width


def _draw_square(r=1.0, g=0.0, b=1.0, **kwargs) -> torch.Tensor:
    gl.glClear(gl.GL_COLOR_BUFFER_BIT)
    gl.glColor3f(r, g, b)
    x1, x2 = -0.5, 0.5
    y1, y2 = -0.5, 0.5
    gl.glRectf(x1, y1, x2, y2)
    out_buffer = gl.glReadPixels(
        0, 0, MAX_EGL_WIDTH, MAX_EGL_HEIGHT, gl.GL_RGB, gl.GL_UNSIGNED_BYTE
    )
    image = torch.frombuffer(out_buffer, dtype=torch.uint8).reshape(
        MAX_EGL_HEIGHT, MAX_EGL_WIDTH, 3
    )
    return image


def _draw_squares_with_context(
    cuda_device_id=0, result=None, thread_id=None, **kwargs
) -> None:
    context = EGLContext(MAX_EGL_WIDTH, MAX_EGL_HEIGHT, cuda_device_id)
    with context.active_and_locked():
        images = []
        for _ in range(3):
            images.append(_draw_square(**kwargs).float())
        if result is not None and thread_id is not None:
            egl_info = context.get_context_info()
            data = {"egl": egl_info, "images": images}
            result[thread_id] = data


def _draw_squares_with_context_store(
    cuda_device_id=0,
    result=None,
    thread_id=None,
    verbose=False,
    **kwargs,
) -> None:
    device = torch.device(f"cuda:{cuda_device_id}")
    context = global_device_context_store.get_egl_context(device)
    if verbose:
        print(f"In thread {thread_id}, device {cuda_device_id}.")
    with context.active_and_locked():
        images = []
        for _ in range(3):
            images.append(_draw_square(**kwargs).float())
        if result is not None and thread_id is not None:
            egl_info = context.get_context_info()
            data = {"egl": egl_info, "images": images}
            result[thread_id] = data


@usesOpengl
class TestDeviceContextStore(TestCaseMixin, unittest.TestCase):
    def test_cuda_context(self):
        cuda_context_1 = global_device_context_store.get_cuda_context(
            device=torch.device("cuda:0")
        )
        cuda_context_2 = global_device_context_store.get_cuda_context(
            device=torch.device("cuda:0")
        )
        cuda_context_3 = global_device_context_store.get_cuda_context(
            device=torch.device("cuda:1")
        )
        cuda_context_4 = global_device_context_store.get_cuda_context(
            device=torch.device("cuda:1")
        )
        self.assertIs(cuda_context_1, cuda_context_2)
        self.assertIs(cuda_context_3, cuda_context_4)
        self.assertIsNot(cuda_context_1, cuda_context_3)

    def test_egl_context(self):
        egl_context_1 = global_device_context_store.get_egl_context(
            torch.device("cuda:0")
        )
        egl_context_2 = global_device_context_store.get_egl_context(
            torch.device("cuda:0")
        )
        egl_context_3 = global_device_context_store.get_egl_context(
            torch.device("cuda:1")
        )
        egl_context_4 = global_device_context_store.get_egl_context(
            torch.device("cuda:1")
        )
        self.assertIs(egl_context_1, egl_context_2)
        self.assertIs(egl_context_3, egl_context_4)
        self.assertIsNot(egl_context_1, egl_context_3)


@usesOpengl
class TestUtils(TestCaseMixin, unittest.TestCase):
    def test_load_extensions(self):
        # This should work
        _define_egl_extension("eglGetPlatformDisplayEXT", egl.EGLDisplay)

        # And this shouldn't (wrong extension)
        with self.assertRaisesRegex(RuntimeError, "Cannot find EGL extension"):
            _define_egl_extension("eglFakeExtensionEXT", egl.EGLBoolean)

    def test_get_cuda_device(self):
        # This should work
        device = _get_cuda_device(0)
        self.assertIsNotNone(device)

        with self.assertRaisesRegex(ValueError, "Device 10000 not available"):
            _get_cuda_device(10000)

    def test_egl_convert_to_int_array(self):
        egl_attributes = {egl.EGL_RED_SIZE: 8}
        attribute_array = _egl_convert_to_int_array(egl_attributes)
        self.assertEqual(attribute_array._type_, ctypes.c_int)
        self.assertEqual(attribute_array._length_, 3)
        self.assertEqual(attribute_array[0], egl.EGL_RED_SIZE)
        self.assertEqual(attribute_array[1], 8)
        self.assertEqual(attribute_array[2], egl.EGL_NONE)


@usesOpengl
class TestOpenGLSingleThreaded(TestCaseMixin, unittest.TestCase):
    def test_draw_square(self):
        context = EGLContext(width=MAX_EGL_WIDTH, height=MAX_EGL_HEIGHT)
        with context.active_and_locked():
            rendering_result = _draw_square().float()
            expected_result = torch.zeros(
                (MAX_EGL_WIDTH, MAX_EGL_HEIGHT, 3), dtype=torch.float
            )
            start_px = int(MAX_EGL_WIDTH / 4)
            end_px = int(MAX_EGL_WIDTH * 3 / 4)
            expected_result[start_px:end_px, start_px:end_px, 0] = 255.0
            expected_result[start_px:end_px, start_px:end_px, 2] = 255.0

        self.assertTrue(torch.all(expected_result == rendering_result))

    def test_render_two_squares(self):
        # Check that drawing twice doesn't overwrite the initial buffer.
        context = EGLContext(width=MAX_EGL_WIDTH, height=MAX_EGL_HEIGHT)
        with context.active_and_locked():
            red_square = _draw_square(r=1.0, g=0.0, b=0.0)
            blue_square = _draw_square(r=0.0, g=0.0, b=1.0)

        start_px = int(MAX_EGL_WIDTH / 4)
        end_px = int(MAX_EGL_WIDTH * 3 / 4)

        self.assertTrue(
            torch.all(
                red_square[start_px:end_px, start_px:end_px]
                == torch.tensor([255, 0, 0])
            )
        )
        self.assertTrue(
            torch.all(
                blue_square[start_px:end_px, start_px:end_px]
                == torch.tensor([0, 0, 255])
            )
        )


@usesOpengl
class TestOpenGLMultiThreaded(TestCaseMixin, unittest.TestCase):
    def test_multiple_renders_single_gpu_single_context(self):
        _draw_squares_with_context()

    def test_multiple_renders_single_gpu_context_store(self):
        _draw_squares_with_context_store()

    def test_render_two_threads_single_gpu(self):
        self._render_two_threads_single_gpu(_draw_squares_with_context)

    def test_render_two_threads_single_gpu_context_store(self):
        self._render_two_threads_single_gpu(_draw_squares_with_context_store)

    def test_render_two_threads_two_gpus(self):
        self._render_two_threads_two_gpus(_draw_squares_with_context)

    def test_render_two_threads_two_gpus_context_store(self):
        self._render_two_threads_two_gpus(_draw_squares_with_context_store)

    def _render_two_threads_single_gpu(self, draw_fn):
        result = [None] * 2
        thread1 = threading.Thread(
            target=draw_fn,
            kwargs={
                "cuda_device_id": 0,
                "result": result,
                "thread_id": 0,
                "r": 1.0,
                "g": 0.0,
                "b": 0.0,
            },
        )
        thread2 = threading.Thread(
            target=draw_fn,
            kwargs={
                "cuda_device_id": 0,
                "result": result,
                "thread_id": 1,
                "r": 0.0,
                "g": 1.0,
                "b": 0.0,
            },
        )

        thread1.start()
        thread2.start()
        thread1.join()
        thread2.join()

        start_px = int(MAX_EGL_WIDTH / 4)
        end_px = int(MAX_EGL_WIDTH * 3 / 4)
        red_squares = torch.stack(result[0]["images"], dim=0)[
            :, start_px:end_px, start_px:end_px
        ]
        green_squares = torch.stack(result[1]["images"], dim=0)[
            :, start_px:end_px, start_px:end_px
        ]
        self.assertTrue(torch.all(red_squares == torch.tensor([255.0, 0.0, 0.0])))
        self.assertTrue(torch.all(green_squares == torch.tensor([0.0, 255.0, 0.0])))

    def _render_two_threads_two_gpus(self, draw_fn):
        # Contrary to _render_two_threads_two_gpus, this renders in two separate threads
        # but on a different GPU each. This means using different EGL contexts and is a
        # much less risky endeavour.
        result = [None] * 2
        thread1 = threading.Thread(
            target=draw_fn,
            kwargs={
                "cuda_device_id": 0,
                "result": result,
                "thread_id": 0,
                "r": 1.0,
                "g": 0.0,
                "b": 0.0,
            },
        )
        thread2 = threading.Thread(
            target=draw_fn,
            kwargs={
                "cuda_device_id": 1,
                "result": result,
                "thread_id": 1,
                "r": 0.0,
                "g": 1.0,
                "b": 0.0,
            },
        )
        thread1.start()
        thread2.start()
        thread1.join()
        thread2.join()
        self.assertNotEqual(
            result[0]["egl"]["context"].address, result[1]["egl"]["context"].address
        )

        start_px = int(MAX_EGL_WIDTH / 4)
        end_px = int(MAX_EGL_WIDTH * 3 / 4)
        red_squares = torch.stack(result[0]["images"], dim=0)[
            :, start_px:end_px, start_px:end_px
        ]
        green_squares = torch.stack(result[1]["images"], dim=0)[
            :, start_px:end_px, start_px:end_px
        ]
        self.assertTrue(torch.all(red_squares == torch.tensor([255.0, 0.0, 0.0])))
        self.assertTrue(torch.all(green_squares == torch.tensor([0.0, 255.0, 0.0])))

    def test_render_multi_thread_multi_gpu(self):
        # Multiple threads using up multiple GPUs; more threads than GPUs.
        # This is certainly not encouraged in practice, but shouldn't fail. Note that
        # the context store will only allow one rendering at a time to occur on a
        # single GPU, even across threads.
        n_gpus = torch.cuda.device_count()
        n_threads = 10
        kwargs = {
            "r": 1.0,
            "g": 0.0,
            "b": 0.0,
            "verbose": True,
        }

        threads = []
        for thread_id in range(n_threads):
            kwargs.update(
                {"cuda_device_id": thread_id % n_gpus, "thread_id": thread_id}
            )
            threads.append(
                threading.Thread(
                    target=_draw_squares_with_context_store, kwargs=dict(kwargs)
                )
            )

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()


@usesOpengl
class TestOpenGLUtils(TestCaseMixin, unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        global_device_context_store.set_context_data(torch.device("cuda:0"), None)

    def test_device_context_store(self):
        # Most of DCS's functionality is tested in the tests above, test the remainder.
        device = torch.device("cuda:0")
        global_device_context_store.set_context_data(device, 123)

        self.assertEqual(global_device_context_store.get_context_data(device), 123)

        self.assertEqual(
            global_device_context_store.get_context_data(torch.device("cuda:1")), None
        )

        # Check that contexts in store can be manually released (although that's a very
        # bad idea! Don't do it manually!)
        egl_ctx = global_device_context_store.get_egl_context(device)
        cuda_ctx = global_device_context_store.get_cuda_context(device)
        egl_ctx.release()
        cuda_ctx.detach()

        # Reset the contexts (just for testing! never do this manually!). Then, check
        # that first running DeviceContextStore.release() will cause subsequent releases
        # to fail (because we already released all the contexts).
        global_device_context_store._cuda_contexts = {}
        global_device_context_store._egl_contexts = {}

        egl_ctx = global_device_context_store.get_egl_context(device)
        cuda_ctx = global_device_context_store.get_cuda_context(device)
        global_device_context_store.release()
        with self.assertRaisesRegex(EGLError, "EGL_NOT_INITIALIZED"):
            egl_ctx.release()
        with self.assertRaisesRegex(pycuda._driver.LogicError, "cannot detach"):
            cuda_ctx.detach()

    def test_no_egl_error(self):
        # Remove EGL, import OpenGL with the wrong backend. This should make it
        # impossible to import OpenGL.EGL.
        del os.environ["PYOPENGL_PLATFORM"]
        modules = list(sys.modules)
        for m in modules:
            if "OpenGL" in m:
                del sys.modules[m]
        import OpenGL.GL  # noqa

        self.assertFalse(_can_import_egl_and_pycuda())

        # Import OpenGL back with the right backend. This should get things on track.
        modules = list(sys.modules)
        for m in modules:
            if "OpenGL" in m:
                del sys.modules[m]

        os.environ["PYOPENGL_PLATFORM"] = "egl"
        self.assertTrue(_can_import_egl_and_pycuda())

    def test_egl_release_error(self):
        # Creating two contexts on the same device will lead to trouble (that's one of
        # the reasons behind DeviceContextStore). You can release one of them,
        # but you cannot release the same EGL resources twice!
        ctx1 = EGLContext(width=100, height=100)
        ctx2 = EGLContext(width=100, height=100)

        ctx1.release()
        with self.assertRaisesRegex(EGLError, "EGL_NOT_INITIALIZED"):
            ctx2.release()
