# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import typing

import torch
from pytorch3d.common.datatypes import Device
from pytorch3d.renderer.cameras import (
    CamerasBase,
    FoVOrthographicCameras,
    FoVPerspectiveCameras,
    OpenGLOrthographicCameras,
    OpenGLPerspectiveCameras,
    OrthographicCameras,
    PerspectiveCameras,
    SfMOrthographicCameras,
    SfMPerspectiveCameras,
)
from pytorch3d.renderer.fisheyecameras import FishEyeCameras
from pytorch3d.transforms.so3 import so3_exp_map


def init_random_cameras(
    cam_type: typing.Type[CamerasBase],
    batch_size: int,
    random_z: bool = False,
    device: Device = "cpu",
):
    cam_params = {}
    T = torch.randn(batch_size, 3) * 0.03
    if not random_z:
        T[:, 2] = 4
    R = so3_exp_map(torch.randn(batch_size, 3) * 3.0)
    cam_params = {"R": R, "T": T, "device": device}
    if cam_type in (OpenGLPerspectiveCameras, OpenGLOrthographicCameras):
        cam_params["znear"] = torch.rand(batch_size) * 10 + 0.1
        cam_params["zfar"] = torch.rand(batch_size) * 4 + 1 + cam_params["znear"]
        if cam_type == OpenGLPerspectiveCameras:
            cam_params["fov"] = torch.rand(batch_size) * 60 + 30
            cam_params["aspect_ratio"] = torch.rand(batch_size) * 0.5 + 0.5
        else:
            cam_params["top"] = torch.rand(batch_size) * 0.2 + 0.9
            cam_params["bottom"] = -(torch.rand(batch_size)) * 0.2 - 0.9
            cam_params["left"] = -(torch.rand(batch_size)) * 0.2 - 0.9
            cam_params["right"] = torch.rand(batch_size) * 0.2 + 0.9
    elif cam_type in (FoVPerspectiveCameras, FoVOrthographicCameras):
        cam_params["znear"] = torch.rand(batch_size) * 10 + 0.1
        cam_params["zfar"] = torch.rand(batch_size) * 4 + 1 + cam_params["znear"]
        if cam_type == FoVPerspectiveCameras:
            cam_params["fov"] = torch.rand(batch_size) * 60 + 30
            cam_params["aspect_ratio"] = torch.rand(batch_size) * 0.5 + 0.5
        else:
            cam_params["max_y"] = torch.rand(batch_size) * 0.2 + 0.9
            cam_params["min_y"] = -(torch.rand(batch_size)) * 0.2 - 0.9
            cam_params["min_x"] = -(torch.rand(batch_size)) * 0.2 - 0.9
            cam_params["max_x"] = torch.rand(batch_size) * 0.2 + 0.9
    elif cam_type in (
        SfMOrthographicCameras,
        SfMPerspectiveCameras,
        OrthographicCameras,
        PerspectiveCameras,
    ):
        cam_params["focal_length"] = torch.rand(batch_size) * 10 + 0.1
        cam_params["principal_point"] = torch.randn((batch_size, 2))
    elif cam_type == FishEyeCameras:
        cam_params["focal_length"] = torch.rand(batch_size, 1) * 10 + 0.1
        cam_params["principal_point"] = torch.randn((batch_size, 2))
        cam_params["radial_params"] = torch.randn((batch_size, 6))
        cam_params["tangential_params"] = torch.randn((batch_size, 2))
        cam_params["thin_prism_params"] = torch.randn((batch_size, 4))

    else:
        raise ValueError(str(cam_type))
    return cam_type(**cam_params)
