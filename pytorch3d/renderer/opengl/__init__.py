# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe


# If we can access EGL, import MeshRasterizerOpenGL.
def _can_import_egl_and_pycuda():
    import os
    import warnings

    try:
        os.environ["PYOPENGL_PLATFORM"] = "egl"
        import OpenGL.EGL
    except (AttributeError, ImportError, ModuleNotFoundError):
        warnings.warn(
            "Can't import EGL, not importing MeshRasterizerOpenGL. This might happen if"
            " your Python application imported OpenGL with a non-EGL backend before"
            " importing PyTorch3D, or if you don't have pyopengl installed as part"
            " of your Python distribution."
        )
        return False

    try:
        import pycuda.gl
    except (ImportError, ImportError, ModuleNotFoundError):
        warnings.warn("Can't import pycuda.gl, not importing MeshRasterizerOpenGL.")
        return False

    return True


if _can_import_egl_and_pycuda():
    from .opengl_utils import EGLContext, global_device_context_store
    from .rasterizer_opengl import MeshRasterizerOpenGL

__all__ = [k for k in globals().keys() if not k.startswith("_")]
