# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

# NOTE: This module (as well as rasterizer_opengl) will not be imported into pytorch3d
# if you do not have pycuda.gl and pyopengl installed. In addition, please make sure
# your Python application *does not* import OpenGL before importing PyTorch3D, unless
# you are using the EGL backend.
import warnings
from typing import Optional, Tuple, Union

import numpy as np
import OpenGL.GL as gl
import pycuda.gl
import torch

import torch.nn as nn

from pytorch3d.structures.meshes import Meshes

from ..cameras import FoVOrthographicCameras, FoVPerspectiveCameras
from ..mesh.rasterizer import Fragments, RasterizationSettings
from ..utils import parse_image_size

from .opengl_utils import _torch_to_opengl, global_device_context_store

# Shader strings, used below to compile an OpenGL program.
vertex_shader = """
// The vertex shader does nothing.
#version 430

void main() { }
"""

geometry_shader = """
#version 430

layout (points) in;
layout (triangle_strip, max_vertices = 3) out;

out layout (location = 0) vec2 bary_coords;
out layout (location = 1) float depth;
out layout (location = 2) float p2f;

layout(binding=0) buffer triangular_mesh { float mesh_buffer[]; };

uniform mat4 perspective_projection;

vec3 get_vertex_position(int vertex_index) {
    int offset = gl_PrimitiveIDIn * 9 + vertex_index * 3;
    return vec3(
        mesh_buffer[offset],
        mesh_buffer[offset + 1],
        mesh_buffer[offset + 2]
    );
}

void main() {
    vec3 positions[3] = {
        get_vertex_position(0),
        get_vertex_position(1),
        get_vertex_position(2)
    };
    vec4 projected_vertices[3] = {
        perspective_projection * vec4(positions[0], 1.0),
        perspective_projection * vec4(positions[1], 1.0),
        perspective_projection * vec4(positions[2], 1.0)
    };

    for (int i = 0; i < 3; ++i) {
        gl_Position = projected_vertices[i];
        bary_coords = vec2(i==0 ? 1.0 : 0.0, i==1 ? 1.0 : 0.0);
        // At the moment, we output depth as the distance from the image plane in
        // view coordinates -- NOT distance along the camera ray.
        depth = positions[i][2];
        p2f = gl_PrimitiveIDIn;
        EmitVertex();
    }
    EndPrimitive();
}
"""

fragment_shader = """
#version 430

in layout(location = 0) vec2 bary_coords;
in layout(location = 1) float depth;
in layout(location = 2) float p2f;


out vec4 bary_depth_p2f;

void main() {
    bary_depth_p2f = vec4(bary_coords, depth, round(p2f));
}
"""


def _parse_and_verify_image_size(
    image_size: Union[Tuple[int, int], int],
) -> Tuple[int, int]:
    """
    Parse image_size as a tuple of ints. Throw ValueError if the size is incompatible
    with the maximum renderable size as set in global_device_context_store.
    """
    height, width = parse_image_size(image_size)
    max_h = global_device_context_store.max_egl_height
    max_w = global_device_context_store.max_egl_width
    if height > max_h or width > max_w:
        raise ValueError(
            f"Max rasterization size is height={max_h}, width={max_w}. "
            f"Cannot raster an image of size {height}, {width}. You can change max "
            "allowed rasterization size by modifying the MAX_EGL_HEIGHT and "
            "MAX_EGL_WIDTH environment variables."
        )
    return height, width


class MeshRasterizerOpenGL(nn.Module):
    """
    EXPERIMENTAL, USE WITH CAUTION

    This class implements methods for rasterizing a batch of heterogeneous
    Meshes using OpenGL. This rasterizer, as opposed to MeshRasterizer, is
    *not differentiable* and needs to be used with shading methods such as
    SplatterPhongShader, which do not require differentiable rasterizerization.
    It is, however, faster: on a 2M-faced mesh, about 20x so.

    Fragments output by MeshRasterizerOpenGL and MeshRasterizer should have near
    identical pix_to_face, bary_coords and zbuf. However, MeshRasterizerOpenGL does not
    return Fragments.dists which is only relevant to SoftPhongShader and
    SoftSilhouetteShader. These do not work with MeshRasterizerOpenGL (because it is
    not differentiable).
    """

    def __init__(
        self,
        cameras: Optional[Union[FoVOrthographicCameras, FoVPerspectiveCameras]] = None,
        raster_settings=None,
    ) -> None:
        """
        Args:
            cameras: A cameras object which has a `transform_points` method
                which returns the transformed points after applying the
                world-to-view and view-to-ndc transformations. Currently, only FoV
                cameras are supported.
            raster_settings: the parameters for rasterization. This should be a
                named tuple.
        """
        super().__init__()
        if raster_settings is None:
            raster_settings = RasterizationSettings()
        self.raster_settings = raster_settings
        _check_raster_settings(self.raster_settings)
        self.cameras = cameras
        self.image_size = _parse_and_verify_image_size(self.raster_settings.image_size)

        self.opengl_machinery = _OpenGLMachinery(
            max_faces=self.raster_settings.max_faces_opengl,
        )

    def forward(self, meshes_world: Meshes, **kwargs) -> Fragments:
        """
        Args:
            meshes_world: a Meshes object representing a batch of meshes with
                coordinates in world space. The batch must live on a GPU.

        Returns:
            Fragments: Rasterization outputs as a named tuple. These are different than
                Fragments returned by MeshRasterizer in two ways. First, we return no
                `dist` which is only relevant to SoftPhongShader which doesn't work
                with MeshRasterizerOpenGL (because it is not differentiable). Second,
                the zbuf uses the opengl zbuf convention, where the z-vals are between 0
                (at projection plane) and 1 (at clipping distance), and are a non-linear
                function of the depth values of the camera ray intersections. In
                contrast, MeshRasterizer's zbuf values are simply the distance of each
                ray intersection from the camera.

        Throws:
            ValueError if meshes_world lives on the CPU.
        """
        if meshes_world.device == torch.device("cpu"):
            raise ValueError("MeshRasterizerOpenGL works only on CUDA devices.")

        raster_settings = kwargs.get("raster_settings", self.raster_settings)
        _check_raster_settings(raster_settings)

        image_size = (
            _parse_and_verify_image_size(raster_settings.image_size) or self.image_size
        )

        # OpenGL needs vertices in NDC coordinates with un-flipped xy directions.
        cameras_unpacked = kwargs.get("cameras", self.cameras)
        _check_cameras(cameras_unpacked)
        meshes_gl_ndc = _convert_meshes_to_gl_ndc(
            meshes_world, image_size, cameras_unpacked, **kwargs
        )

        # Perspective projection will happen within the OpenGL rasterizer.
        projection_matrix = cameras_unpacked.get_projection_transform(**kwargs)._matrix

        # Run OpenGL rasterization machinery.
        pix_to_face, bary_coords, zbuf = self.opengl_machinery(
            meshes_gl_ndc, projection_matrix, image_size
        )

        # Return the Fragments and detach, because gradients don't go through OpenGL.
        return Fragments(
            pix_to_face=pix_to_face,
            zbuf=zbuf,
            bary_coords=bary_coords,
            dists=None,
        ).detach()

    def to(self, device):
        # Manually move to device cameras as it is not a subclass of nn.Module
        if self.cameras is not None:
            self.cameras = self.cameras.to(device)

        # Create a new OpenGLMachinery, as its member variables can be tied to a GPU.
        self.opengl_machinery = _OpenGLMachinery(
            max_faces=self.raster_settings.max_faces_opengl,
        )


class _OpenGLMachinery:
    """
    A class holding OpenGL machinery used by MeshRasterizerOpenGL.
    """

    def __init__(
        self,
        max_faces: int = 10_000_000,
    ) -> None:
        self.max_faces = max_faces
        self.program = None

        # These will be created on an appropriate GPU each time we render a new mesh on
        # that GPU for the first time.
        self.egl_context = None
        self.cuda_context = None
        self.perspective_projection_uniform = None
        self.mesh_buffer_object = None
        self.vao = None
        self.fbo = None
        self.cuda_buffer = None

    def __call__(
        self,
        meshes_gl_ndc: Meshes,
        projection_matrix: torch.Tensor,
        image_size: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Rasterize a batch of meshes, using a given batch of projection matrices and
        image size.

        Args:
            meshes_gl_ndc: A Meshes object, with vertices in the OpenGL NDC convention.
            projection_matrix: A 3x3 camera projection matrix, or a tensor of projection
                matrices equal in length to the number of meshes in meshes_gl_ndc.
            image_size: Image size to rasterize. Must be smaller than the max height and
                width stored in global_device_context_store.

        Returns:
            pix_to_faces: A BHW1 tensor of ints, filled with -1 where no face projects
                to a given pixel.
            bary_coords: A BHW3 float tensor, filled with -1 where no face projects to
                a given pixel.
            zbuf: A BHW1 float tensor, filled with 1 where no face projects to a given
                pixel. NOTE: this zbuf uses the opengl zbuf convention, where the z-vals
                are between 0 (at projection plane) and 1 (at clipping distance), and
                are a non-linear function of the depth values of the camera ray inter-
                sections.
        """

        self.initialize_device_data(meshes_gl_ndc.device)
        with self.egl_context.active_and_locked():
            # Perspective projection happens in OpenGL. Move the matrix over if there's only
            # a single camera shared by all the meshes.
            if projection_matrix.shape[0] == 1:
                self._projection_matrix_to_opengl(projection_matrix)

            pix_to_faces = []
            bary_coords = []
            zbufs = []

            # pyre-ignore Incompatible parameter type [6]
            for mesh_id, mesh in enumerate(meshes_gl_ndc):
                pix_to_face, bary_coord, zbuf = self._rasterize_mesh(
                    mesh,
                    image_size,
                    projection_matrix=(
                        projection_matrix[mesh_id]
                        if projection_matrix.shape[0] > 1
                        else None
                    ),
                )
                pix_to_faces.append(pix_to_face)
                bary_coords.append(bary_coord)
                zbufs.append(zbuf)

        return (
            torch.cat(pix_to_faces, dim=0),
            torch.cat(bary_coords, dim=0),
            torch.cat(zbufs, dim=0),
        )

    def initialize_device_data(self, device) -> None:
        """
        Initialize data specific to a GPU device: the EGL and CUDA contexts, the OpenGL
        program, as well as various buffer and array objects used to communicate with
        OpenGL.

        Args:
            device: A torch.device.
        """
        self.egl_context = global_device_context_store.get_egl_context(device)
        self.cuda_context = global_device_context_store.get_cuda_context(device)

        # self.program represents the OpenGL program we use for rasterization.
        if global_device_context_store.get_context_data(device) is None:
            with self.egl_context.active_and_locked():
                self.program = self._compile_and_link_gl_program()
                self._set_up_gl_program_properties(self.program)

                # Create objects used to transfer data into and out of the program.
                (
                    self.perspective_projection_uniform,
                    self.mesh_buffer_object,
                    self.vao,
                    self.fbo,
                ) = self._prepare_persistent_opengl_objects(
                    self.program,
                    self.max_faces,
                )

                # Register the input buffer with pycuda, to transfer data directly into it.
                self.cuda_context.push()
                self.cuda_buffer = pycuda.gl.RegisteredBuffer(
                    int(self.mesh_buffer_object),
                    pycuda.gl.graphics_map_flags.WRITE_DISCARD,
                )
                self.cuda_context.pop()

            global_device_context_store.set_context_data(
                device,
                (
                    self.program,
                    self.perspective_projection_uniform,
                    self.mesh_buffer_object,
                    self.vao,
                    self.fbo,
                    self.cuda_buffer,
                ),
            )
        (
            self.program,
            self.perspective_projection_uniform,
            self.mesh_buffer_object,
            self.vao,
            self.fbo,
            self.cuda_buffer,
        ) = global_device_context_store.get_context_data(device)

    def release(self) -> None:
        """
        Release CUDA and OpenGL resources.
        """
        # Finish all current operations.
        torch.cuda.synchronize()
        self.cuda_context.synchronize()

        # Free pycuda resources.
        self.cuda_context.push()
        self.cuda_buffer.unregister()
        self.cuda_context.pop()

        # Free GL resources.
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.fbo)
        # pyre-fixme[16]: Module `GL_3_0` has no attribute `glDeleteFramebuffers`.
        gl.glDeleteFramebuffers(1, [self.fbo])
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        del self.fbo

        gl.glBindBufferBase(gl.GL_SHADER_STORAGE_BUFFER, 0, self.mesh_buffer_object)
        # pyre-fixme[16]: Module `GL_1_5` has no attribute `glDeleteBuffers`.
        gl.glDeleteBuffers(1, [self.mesh_buffer_object])
        gl.glBindBufferBase(gl.GL_SHADER_STORAGE_BUFFER, 0, 0)
        del self.mesh_buffer_object

        gl.glDeleteProgram(self.program)
        self.egl_context.release()

    def _projection_matrix_to_opengl(self, projection_matrix: torch.Tensor) -> None:
        """
        Transfer a torch projection matrix to OpenGL.

        Args:
            projection matrix: A 3x3 float tensor.
        """
        gl.glUseProgram(self.program)
        # pyre-fixme[16]: Module `GL_2_0` has no attribute `glUniformMatrix4fv`.
        gl.glUniformMatrix4fv(
            self.perspective_projection_uniform,
            1,
            gl.GL_FALSE,
            projection_matrix.detach().flatten().cpu().numpy().astype(np.float32),
        )
        gl.glUseProgram(0)

    def _rasterize_mesh(
        self,
        mesh: Meshes,
        image_size: Tuple[int, int],
        projection_matrix: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Rasterize a single mesh using OpenGL.

        Args:
            mesh: A Meshes object, containing a single mesh only.
            projection_matrix: A 3x3 camera projection matrix, or a tensor of projection
                matrices equal in length to the number of meshes in meshes_gl_ndc.
            image_size: Image size to rasterize. Must be smaller than the max height and
                width stored in global_device_context_store.

        Returns:
            pix_to_faces: A 1HW1 tensor of ints, filled with -1 where no face projects
                to a given pixel.
            bary_coords: A 1HW3 float tensor, filled with -1 where no face projects to
                a given pixel.
            zbuf: A 1HW1 float tensor, filled with 1 where no face projects to a given
                pixel. NOTE: this zbuf uses the opengl zbuf convention, where the z-vals
                are between 0 (at projection plane) and 1 (at clipping distance), and
                are a non-linear function of the depth values of the camera ray inter-
                sections.
        """
        height, width = image_size
        # Extract face_verts and move them to OpenGL as well. We use pycuda to
        # directly move the vertices on the GPU, to avoid a costly torch/GPU -> CPU
        # -> openGL/GPU trip.
        verts_packed = mesh.verts_packed().detach()
        faces_packed = mesh.faces_packed().detach()
        face_verts = verts_packed[faces_packed].reshape(-1, 9)
        _torch_to_opengl(face_verts, self.cuda_context, self.cuda_buffer)

        if projection_matrix is not None:
            self._projection_matrix_to_opengl(projection_matrix)

        # Start OpenGL operations.
        gl.glUseProgram(self.program)

        # Render an image of size (width, height).
        gl.glViewport(0, 0, width, height)

        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.fbo)
        # Clear the output framebuffer. The "background" value for both pix_to_face
        # as well as bary_coords is -1 (background = pixels which the rasterizer
        # projected no triangle to).
        gl.glClearColor(-1.0, -1.0, -1.0, -1.0)
        gl.glClearDepth(1.0)
        # pyre-ignore Unsupported operand [58]
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        # Run the actual rendering. The face_verts were transported to the OpenGL
        # program into a shader storage buffer which is used directly in the geometry
        # shader. Here, we only pass the number of these vertices to the vertex shader
        # (which doesn't do anything and passes directly to the geometry shader).
        gl.glBindVertexArray(self.vao)
        gl.glDrawArrays(gl.GL_POINTS, 0, len(face_verts))
        gl.glBindVertexArray(0)

        # Read out the result. We ignore the depth buffer. The RGBA color buffer stores
        # barycentrics in the RGB component and pix_to_face in the A component.
        bary_depth_p2f_gl = gl.glReadPixels(
            0,
            0,
            width,
            height,
            gl.GL_RGBA,
            gl.GL_FLOAT,
        )

        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        gl.glUseProgram(0)

        # Create torch tensors containing the results.
        bary_depth_p2f = (
            torch.frombuffer(bary_depth_p2f_gl, dtype=torch.float)
            .reshape(1, height, width, 1, -1)
            .to(verts_packed.device)
        )

        # Read out barycentrics. GL only outputs the first two, so we need to compute
        # the third one and make sure we still leave no-intersection pixels with -1.
        barycentric_coords = torch.cat(
            [
                bary_depth_p2f[..., :2],
                1.0 - bary_depth_p2f[..., 0:1] - bary_depth_p2f[..., 1:2],
            ],
            dim=-1,
        )
        barycentric_coords = torch.where(
            barycentric_coords == 3, -1, barycentric_coords
        )
        depth = bary_depth_p2f[..., 2:3].squeeze(-1)
        pix_to_face = bary_depth_p2f[..., -1].long()

        return pix_to_face, barycentric_coords, depth

    @staticmethod
    def _compile_and_link_gl_program():
        """
        Compile the vertex, geometry, and fragment shaders and link them into an OpenGL
        program. The shader sources are strongly inspired by https://github.com/tensorflow/
        graphics/blob/master/tensorflow_graphics/rendering/opengl/rasterization_backend.py.

        Returns:
            An OpenGL program for mesh rasterization.
        """
        program = gl.glCreateProgram()
        shader_objects = []

        for shader_string, shader_type in zip(
            [vertex_shader, geometry_shader, fragment_shader],
            [gl.GL_VERTEX_SHADER, gl.GL_GEOMETRY_SHADER, gl.GL_FRAGMENT_SHADER],
        ):
            shader_objects.append(gl.glCreateShader(shader_type))
            gl.glShaderSource(shader_objects[-1], shader_string)

            gl.glCompileShader(shader_objects[-1])
            status = gl.glGetShaderiv(shader_objects[-1], gl.GL_COMPILE_STATUS)
            if status == gl.GL_FALSE:
                gl.glDeleteShader(shader_objects[-1])
                gl.glDeleteProgram(program)
                error_msg = gl.glGetShaderInfoLog(shader_objects[-1]).decode("utf-8")
                raise RuntimeError(f"Compilation failure:\n {error_msg}")

            gl.glAttachShader(program, shader_objects[-1])
            gl.glDeleteShader(shader_objects[-1])

        gl.glLinkProgram(program)
        status = gl.glGetProgramiv(program, gl.GL_LINK_STATUS)

        if status == gl.GL_FALSE:
            gl.glDeleteProgram(program)
            error_msg = gl.glGetProgramInfoLog(program)
            raise RuntimeError(f"Link failure:\n {error_msg}")

        return program

    @staticmethod
    def _set_up_gl_program_properties(program) -> None:
        """
        Set basic OpenGL program properties: disable blending, enable depth testing,
        and disable face culling.
        """
        gl.glUseProgram(program)
        gl.glDisable(gl.GL_BLEND)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDisable(gl.GL_CULL_FACE)
        gl.glUseProgram(0)

    @staticmethod
    def _prepare_persistent_opengl_objects(program, max_faces: int):
        """
        Prepare OpenGL objects that we want to persist between rasterizations.

        Args:
            program: The OpenGL program the resources will be tied to.
            max_faces: Max number of faces of any mesh we will rasterize.

        Returns:
            perspective_projection_uniform: An OpenGL object pointing to a location of
                the perspective projection matrix in OpenGL memory.
            mesh_buffer_object: An OpenGL object pointing to the location of the mesh
                buffer object in OpenGL memory.
            vao: The OpenGL input array object.
            fbo: The OpenGL output framebuffer.

        """
        gl.glUseProgram(program)
        # Get location of the "uniform" (that is, an internal OpenGL variable available
        # to the shaders) that we'll load the projection matrices to.
        perspective_projection_uniform = gl.glGetUniformLocation(
            program, "perspective_projection"
        )

        # Mesh buffer object -- our main input point. We'll copy the mesh here
        # from pytorch/cuda. The buffer needs enough space to store the three vertices
        # of each face, that is its size in bytes is
        # max_faces * 3 (vertices) * 3 (coordinates) * 4 (bytes)
        # pyre-fixme[16]: Module `GL_1_5` has no attribute `glGenBuffers`.
        mesh_buffer_object = gl.glGenBuffers(1)
        gl.glBindBufferBase(gl.GL_SHADER_STORAGE_BUFFER, 0, mesh_buffer_object)

        gl.glBufferData(
            gl.GL_SHADER_STORAGE_BUFFER,
            max_faces * 9 * 4,
            np.zeros((max_faces, 9), dtype=np.float32),
            gl.GL_DYNAMIC_COPY,
        )

        # Input vertex array object. We will only use it implicitly for indexing the
        # vertices, but the actual input data is passed in the shader storage buffer.
        # pyre-fixme[16]: Module `GL_3_0` has no attribute `glGenVertexArrays`.
        vao = gl.glGenVertexArrays(1)

        # Create the framebuffer object (fbo) where we'll store output data.
        MAX_EGL_WIDTH = global_device_context_store.max_egl_width
        MAX_EGL_HEIGHT = global_device_context_store.max_egl_height
        # pyre-fixme[16]: Module `GL_3_0` has no attribute `glGenRenderbuffers`.
        color_buffer = gl.glGenRenderbuffers(1)
        gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, color_buffer)
        gl.glRenderbufferStorage(
            gl.GL_RENDERBUFFER, gl.GL_RGBA32F, MAX_EGL_WIDTH, MAX_EGL_HEIGHT
        )
        gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, 0)

        # pyre-fixme[16]: Module `GL_3_0` has no attribute `glGenRenderbuffers`.
        depth_buffer = gl.glGenRenderbuffers(1)
        gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, depth_buffer)
        gl.glRenderbufferStorage(
            gl.GL_RENDERBUFFER, gl.GL_DEPTH_COMPONENT, MAX_EGL_WIDTH, MAX_EGL_HEIGHT
        )
        gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, 0)

        # pyre-fixme[16]: Module `GL_3_0` has no attribute `glGenFramebuffers`.
        fbo = gl.glGenFramebuffers(1)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, fbo)
        gl.glFramebufferRenderbuffer(
            gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_RENDERBUFFER, color_buffer
        )
        gl.glFramebufferRenderbuffer(
            gl.GL_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT, gl.GL_RENDERBUFFER, depth_buffer
        )
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)

        gl.glUseProgram(0)
        return perspective_projection_uniform, mesh_buffer_object, vao, fbo


def _check_cameras(cameras) -> None:
    # Check that the cameras are non-None and compatible with MeshRasterizerOpenGL.
    if cameras is None:
        msg = "Cameras must be specified either at initialization \
            or in the forward pass of MeshRasterizer"
        raise ValueError(msg)
    if type(cameras).__name__ in {"PerspectiveCameras", "OrthographicCameras"}:
        raise ValueError(
            "MeshRasterizerOpenGL only works with FoVPerspectiveCameras and "
            "FoVOrthographicCameras, which are OpenGL compatible."
        )


def _check_raster_settings(raster_settings) -> None:
    # Check that the rasterizer's settings are compatible with MeshRasterizerOpenGL.
    if raster_settings.faces_per_pixel > 1:
        warnings.warn(
            "MeshRasterizerOpenGL currently works only with one face per pixel."
        )
    if raster_settings.cull_backfaces:
        warnings.warn(
            "MeshRasterizerOpenGL cannot cull backfaces yet, rasterizing without culling."
        )
    if raster_settings.cull_to_frustum:
        warnings.warn(
            "MeshRasterizerOpenGL cannot cull to frustum yet, rasterizing without culling."
        )
    if raster_settings.z_clip_value is not None:
        raise NotImplementedError("MeshRasterizerOpenGL cannot do z-clipping yet.")
    if raster_settings.perspective_correct is False:
        raise ValueError(
            "MeshRasterizerOpenGL always uses perspective-correct interpolation."
        )


def _convert_meshes_to_gl_ndc(
    meshes_world: Meshes, image_size: Tuple[int, int], camera, **kwargs
) -> Meshes:
    """
    Convert a batch of world-coordinate meshes to GL NDC coordinates.

    Args:
        meshes_world: Meshes in the world coordinate system.
        image_size: Image height and width, used to modify mesh coords for rendering in
            non-rectangular images. OpenGL will expand anything within the [-1, 1] NDC
            range to fit the width and height of the screen, so we will squeeze the NDCs
            appropriately if rendering a rectangular image.
        camera: FoV cameras.
        kwargs['R'], kwargs['T']: If present, used to define the world-view transform.
    """
    height, width = image_size
    verts_ndc = (
        camera.get_world_to_view_transform(**kwargs)
        .compose(camera.get_ndc_camera_transform(**kwargs))
        .transform_points(meshes_world.verts_padded(), eps=None)
    )
    verts_ndc[..., 0] = -verts_ndc[..., 0]
    verts_ndc[..., 1] = -verts_ndc[..., 1]

    # In case of a non-square viewport, transform the vertices. OpenGL will expand
    # the anything within the [-1, 1] NDC range to fit the width and height of the
    # screen. So to work with PyTorch3D cameras, we need to squeeze the NDCs
    # appropriately.
    dtype, device = verts_ndc.dtype, verts_ndc.device
    if height > width:
        verts_ndc = verts_ndc * torch.tensor(
            [1, width / height, 1], dtype=dtype, device=device
        )
    elif width > height:
        verts_ndc = verts_ndc * torch.tensor(
            [height / width, 1, 1], dtype=dtype, device=device
        )

    meshes_gl_ndc = meshes_world.update_padded(new_verts_padded=verts_ndc)

    return meshes_gl_ndc
