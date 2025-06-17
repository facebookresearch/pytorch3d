# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

# Utilities useful for OpenGL rendering.
#
# NOTE: This module MUST be imported before any other OpenGL modules in this Python
# session, unless you set PYOPENGL_PLATFORM to egl *before* importing other modules.
# Otherwise, the imports below will throw an error.
#
# This module (as well as rasterizer_opengl) will not be imported into pytorch3d if
# you do not have pycuda.gl and pyopengl installed.

import contextlib
import ctypes
import os
import threading
from typing import Any, Dict


os.environ["PYOPENGL_PLATFORM"] = "egl"
import OpenGL.EGL as egl  # noqa

import pycuda.driver as cuda  # noqa
from OpenGL._opaque import opaque_pointer_cls  # noqa
from OpenGL.raw.EGL._errors import EGLError  # noqa

# A few constants necessary to use EGL extensions, see links for details.

# https://www.khronos.org/registry/EGL/extensions/EXT/EGL_EXT_platform_device.txt
EGL_PLATFORM_DEVICE_EXT = 0x313F
# https://www.khronos.org/registry/EGL/extensions/NV/EGL_NV_device_cuda.txt
EGL_CUDA_DEVICE_NV = 0x323A


# To use EGL extensions, we need to tell OpenGL about them. For details, see
# https://developer.nvidia.com/blog/egl-eye-opengl-visualization-without-x-server/.
# To avoid garbage collection of the protos, we'll store them in a module-global list.
def _define_egl_extension(name: str, type):
    if hasattr(egl, name):
        return
    addr = egl.eglGetProcAddress(name)
    if addr is None:
        raise RuntimeError(f"Cannot find EGL extension {name}.")
    else:
        proto = ctypes.CFUNCTYPE(type)
        func = proto(addr)
        setattr(egl, name, func)
    return proto


_protos = []
_protos.append(_define_egl_extension("eglGetPlatformDisplayEXT", egl.EGLDisplay))
_protos.append(_define_egl_extension("eglQueryDevicesEXT", egl.EGLBoolean))
_protos.append(_define_egl_extension("eglQueryDeviceAttribEXT", egl.EGLBoolean))
_protos.append(_define_egl_extension("eglQueryDisplayAttribEXT", egl.EGLBoolean))
_protos.append(_define_egl_extension("eglQueryDeviceStringEXT", ctypes.c_char_p))

if not hasattr(egl, "EGLDeviceEXT"):
    egl.EGLDeviceEXT = opaque_pointer_cls("EGLDeviceEXT")


def _egl_convert_to_int_array(egl_attributes):
    """
    Convert a Python dict of EGL attributes into an array of ints (some of which are
    special EGL ints.

    Args:
        egl_attributes: A dict where keys are EGL attributes, and values are their vals.

    Returns:
        A c-list of length 2 * len(egl_attributes) + 1, of the form [key1, val1, ...,
        keyN, valN, EGL_NONE]
    """
    attributes_list = sum(([k, v] for k, v in egl_attributes.items()), []) + [
        egl.EGL_NONE
    ]
    return (egl.EGLint * len(attributes_list))(*attributes_list)


def _get_cuda_device(requested_device_id: int):
    """
    Find an EGL device with a given CUDA device ID.

    Args:
        requested_device_id: The desired CUDA device ID, e.g. "1" for "cuda:1".

    Returns:
        EGL device with the desired CUDA ID.
    """
    num_devices = egl.EGLint()
    if (
        # pyre-ignore Undefined attribute [16]
        not egl.eglQueryDevicesEXT(0, None, ctypes.pointer(num_devices))
        or num_devices.value < 1
    ):
        raise RuntimeError("EGL requires a system that supports at least one device.")
    devices = (egl.EGLDeviceEXT * num_devices.value)()  # array of size num_devices
    if (
        # pyre-ignore Undefined attribute [16]
        not egl.eglQueryDevicesEXT(
            num_devices.value, devices, ctypes.pointer(num_devices)
        )
        or num_devices.value < 1
    ):
        raise RuntimeError("EGL sees no available devices.")
    if len(devices) < requested_device_id + 1:
        raise ValueError(
            f"Device {requested_device_id} not available. Found only {len(devices)} devices."
        )

    # Iterate over all the EGL devices, and check if their CUDA ID matches the request.
    for device in devices:
        available_device_id = egl.EGLAttrib(ctypes.c_int(-1))
        # pyre-ignore Undefined attribute [16]
        egl.eglQueryDeviceAttribEXT(device, EGL_CUDA_DEVICE_NV, available_device_id)
        if available_device_id.contents.value == requested_device_id:
            return device
    raise ValueError(
        f"Found {len(devices)} CUDA devices, but none with CUDA id {requested_device_id}."
    )


def _get_egl_config(egl_dpy, surface_type):
    """
    Get an EGL config with reasonable settings (for use with MeshRasterizerOpenGL).

    Args:
        egl_dpy: An EGL display constant (int).
        surface_type: An EGL surface_type int.

    Returns:
        An EGL config object.

    Throws:
        ValueError if the desired config is not available or invalid.
    """
    egl_config_dict = {
        egl.EGL_RED_SIZE: 8,
        egl.EGL_GREEN_SIZE: 8,
        egl.EGL_BLUE_SIZE: 8,
        egl.EGL_ALPHA_SIZE: 8,
        egl.EGL_DEPTH_SIZE: 24,
        egl.EGL_STENCIL_SIZE: egl.EGL_DONT_CARE,
        egl.EGL_RENDERABLE_TYPE: egl.EGL_OPENGL_BIT,
        egl.EGL_SURFACE_TYPE: surface_type,
    }
    egl_config_array = _egl_convert_to_int_array(egl_config_dict)
    egl_config = egl.EGLConfig()
    num_configs = egl.EGLint()
    if (
        not egl.eglChooseConfig(
            egl_dpy,
            egl_config_array,
            ctypes.pointer(egl_config),
            1,
            ctypes.pointer(num_configs),
        )
        or num_configs.value == 0
    ):
        raise ValueError("Invalid EGL config.")
    return egl_config


class EGLContext:
    """
    A class representing an EGL context. In short, EGL allows us to render OpenGL con-
    tent in a headless mode, that is without an actual display to render to. This capa-
    bility enables MeshRasterizerOpenGL to render on the GPU and then transfer the re-
    sults to PyTorch3D.
    """

    def __init__(self, width: int, height: int, cuda_device_id: int = 0) -> None:
        """
        Args:
            width: Width of the "display" to render to.
            height: Height of the "display" to render to.
            cuda_device_id: Device ID to render to, in the CUDA convention (note that
                this might be different than EGL's device numbering).
        """
        # Lock used to prevent multiple threads from rendering on the same device
        # at the same time, creating/destroying contexts at the same time, etc.
        self.lock = threading.RLock()
        self.cuda_device_id = cuda_device_id
        self.device = _get_cuda_device(self.cuda_device_id)
        self.width = width
        self.height = height
        self.dpy = egl.eglGetPlatformDisplayEXT(
            EGL_PLATFORM_DEVICE_EXT, self.device, None
        )
        major, minor = egl.EGLint(), egl.EGLint()

        # Initialize EGL components: the display, surface, and context
        egl.eglInitialize(self.dpy, ctypes.pointer(major), ctypes.pointer(minor))

        config = _get_egl_config(self.dpy, egl.EGL_PBUFFER_BIT)
        pb_surf_attribs = _egl_convert_to_int_array(
            {
                egl.EGL_WIDTH: width,
                egl.EGL_HEIGHT: height,
            }
        )
        self.surface = egl.eglCreatePbufferSurface(self.dpy, config, pb_surf_attribs)
        if self.surface == egl.EGL_NO_SURFACE:
            raise RuntimeError("Failed to create an EGL surface.")

        if not egl.eglBindAPI(egl.EGL_OPENGL_API):
            raise RuntimeError("Failed to bind EGL to the OpenGL API.")
        self.context = egl.eglCreateContext(self.dpy, config, egl.EGL_NO_CONTEXT, None)
        if self.context == egl.EGL_NO_CONTEXT:
            raise RuntimeError("Failed to create an EGL context.")

    @contextlib.contextmanager
    def active_and_locked(self):
        """
        A context manager used to make sure a given EGL context is only current in
        a single thread at a single time. It is recommended to ALWAYS use EGL within
        a `with context.active_and_locked():` context.

        Throws:
            EGLError when the context cannot be made current or make non-current.
        """
        with self.lock:
            egl.eglMakeCurrent(self.dpy, self.surface, self.surface, self.context)
            try:
                yield
            finally:
                egl.eglMakeCurrent(
                    self.dpy, egl.EGL_NO_SURFACE, egl.EGL_NO_SURFACE, egl.EGL_NO_CONTEXT
                )

    def get_context_info(self) -> Dict[str, Any]:
        """
        Return context info. Useful for debugging.

        Returns:
            A dict of keys and ints, representing the context's display, surface,
            the context itself, and the current thread.
        """
        return {
            "dpy": self.dpy,
            "surface": self.surface,
            "context": self.context,
            "thread": threading.get_ident(),
        }

    def release(self):
        """
        Release the context's resources.
        """
        self.lock.acquire()
        try:
            if self.surface:
                egl.eglDestroySurface(self.dpy, self.surface)
            if self.context and self.dpy:
                egl.eglDestroyContext(self.dpy, self.context)
            egl.eglMakeCurrent(
                self.dpy, egl.EGL_NO_SURFACE, egl.EGL_NO_SURFACE, egl.EGL_NO_CONTEXT
            )
            if self.dpy:
                egl.eglTerminate(self.dpy)
        except EGLError as err:
            print(
                f"EGL could not release context on device cuda:{self.cuda_device_id}."
                " This can happen if you created two contexts on the same device."
                " Instead, you can use DeviceContextStore to use a single context"
                " per device, and EGLContext.make_(in)active_in_current_thread to"
                " (in)activate the context as needed."
            )
            raise err

        egl.eglReleaseThread()
        self.lock.release()


class _DeviceContextStore:
    """
    DeviceContextStore provides thread-safe storage for EGL and pycuda contexts. It
    should not be used directly. opengl_utils instantiates a module-global variable
    called opengl_utils.global_device_context_store. MeshRasterizerOpenGL uses this
    store to avoid unnecessary context creation and destruction.

    The EGL/CUDA contexts are not meant to be created and destroyed all the time,
    and having multiple on a single device can be troublesome. Intended use is entirely
    transparent to the user::

        rasterizer1 = MeshRasterizerOpenGL(...some args...)
        mesh1 = load_mesh_on_cuda_0()

        # Now rasterizer1 will request EGL/CUDA contexts from
        # global_device_context_store on cuda:0, and since there aren't any, the
        # store will create new ones.
        rasterizer1.rasterize(mesh1)

        # rasterizer2 also needs EGL & CUDA contexts. But global_context_store
        # already has them for cuda:0. Instead of creating new contexts, the store
        # will tell rasterizer2 to use them.
        rasterizer2 = MeshRasterizerOpenGL(dcs)
        rasterize2.rasterize(mesh1)

        # When rasterizer1 needs to render on cuda:1, the store will create new contexts.
        mesh2 = load_mesh_on_cuda_1()
        rasterizer1.rasterize(mesh2)

    """

    def __init__(self):
        cuda.init()
        # pycuda contexts, at most one per device.
        self._cuda_contexts = {}
        # EGL contexts, at most one per device.
        self._egl_contexts = {}
        # Any extra per-device data (e.g. precompiled GL objects).
        self._context_data = {}
        # Lock for DeviceContextStore used in multithreaded multidevice scenarios.
        self._lock = threading.Lock()
        # All EGL contexts created by this store will have this resolution.
        self.max_egl_width = 2048
        self.max_egl_height = 2048

    def get_cuda_context(self, device):
        """
        Return a pycuda's CUDA context on a given CUDA device. If we have not created
        such a context yet, create a new one and store it in a dict. The context is
        popped (you need to call context.push() to start using it). This function
        is thread-safe.

        Args:
            device: A torch.device.

        Returns: A pycuda context corresponding to the given device.
        """
        cuda_device_id = device.index
        with self._lock:
            if cuda_device_id not in self._cuda_contexts:
                self._cuda_contexts[cuda_device_id] = _init_cuda_context(cuda_device_id)
                self._cuda_contexts[cuda_device_id].pop()
            return self._cuda_contexts[cuda_device_id]

    def get_egl_context(self, device):
        """
        Return an EGL context on a given CUDA device. If we have not created such a
        context yet, create a new one and store it in a dict. The context if not current
        (you should use the `with egl_context.active_and_locked:` context manager when
        you need it to be current). This function is thread-safe.

        Args:
            device: A torch.device.

        Returns: An EGLContext on the requested device. The context will have size
            self.max_egl_width and self.max_egl_height.
        """
        cuda_device_id = device.index
        with self._lock:
            egl_context = self._egl_contexts.get(cuda_device_id, None)
            if egl_context is None:
                self._egl_contexts[cuda_device_id] = EGLContext(
                    self.max_egl_width, self.max_egl_height, cuda_device_id
                )
            return self._egl_contexts[cuda_device_id]

    def set_context_data(self, device, value):
        """
        Set arbitrary data in a per-device dict.

        This function is intended for storing precompiled OpenGL objects separately for
        EGL contexts on different devices. Each such context needs a separate compiled
        OpenGL program, but (in case e.g. of MeshRasterizerOpenGL) there's no need to
        re-compile it each time we move the rasterizer to the same device repeatedly,
        as it happens when using DataParallel.

        Args:
            device: A torch.device
            value: An arbitrary Python object.
        """

        cuda_device_id = device.index
        self._context_data[cuda_device_id] = value

    def get_context_data(self, device):
        """
        Get arbitrary data in a per-device dict. See set_context_data for more detail.

        Args:
            device: A torch.device

        Returns:
            The most recent object stored using set_context_data.
        """
        cuda_device_id = device.index
        return self._context_data.get(cuda_device_id, None)

    def release(self):
        """
        Release all CUDA and EGL contexts.
        """
        for context in self._cuda_contexts.values():
            context.detach()

        for context in self._egl_contexts.values():
            context.release()


def _init_cuda_context(device_id: int = 0):
    """
    Initialize a pycuda context on a chosen device.

    Args:
        device_id: int, specifies which GPU to use.

    Returns:
        A pycuda Context.
    """
    # pyre-ignore Undefined attribute [16]
    device = cuda.Device(device_id)
    cuda_context = device.make_context()
    return cuda_context


def _torch_to_opengl(torch_tensor, cuda_context, cuda_buffer):
    # CUDA access to the OpenGL buffer is only allowed within a map-unmap block.
    cuda_context.push()
    mapping_obj = cuda_buffer.map()

    # data_ptr points to the OpenGL shader storage buffer memory.
    data_ptr, sz = mapping_obj.device_ptr_and_size()

    # Copy the torch tensor to the OpenGL buffer directly on device.
    cuda_copy = cuda.Memcpy2D()
    cuda_copy.set_src_device(torch_tensor.data_ptr())
    cuda_copy.set_dst_device(data_ptr)
    cuda_copy.width_in_bytes = cuda_copy.src_pitch = cuda_copy.dst_ptch = (
        torch_tensor.shape[1] * 4
    )
    cuda_copy.height = torch_tensor.shape[0]
    cuda_copy(False)

    # Unmap and pop the cuda context to make sure OpenGL won't interfere with
    # PyTorch ops down the line.
    mapping_obj.unmap()
    cuda_context.pop()


# Initialize a global _DeviceContextStore. Almost always we will only need a single one.
global_device_context_store = _DeviceContextStore()
