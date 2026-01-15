# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Test render speed."""

import logging
import sys
from os import path

import torch
from fvcore.common.benchmark import benchmark
from pytorch3d.renderer.points.pulsar import Renderer
from torch.autograd import Variable


# Making sure you can run this, even if pulsar hasn't been installed yet.
sys.path.insert(0, path.join(path.dirname(__file__), ".."))
LOGGER = logging.getLogger(__name__)


"""Measure the execution speed of the rendering.

This measures a very pessimistic upper bound on speed, because synchronization
points have to be introduced in Python. On a pure PyTorch execution pipeline,
results should be significantly faster. You can get pure CUDA timings through
C++ by activating `PULSAR_TIMINGS_BATCHED_ENABLED` in the file
`pytorch3d/csrc/pulsar/logging.h` or defining it for your compiler.
"""


def _bm_pulsar():
    n_points = 1_000_000
    width = 1_000
    height = 1_000
    renderer = Renderer(width, height, n_points)
    # Generate sample data.
    torch.manual_seed(1)
    vert_pos = torch.rand(n_points, 3, dtype=torch.float32) * 10.0
    vert_pos[:, 2] += 25.0
    vert_pos[:, :2] -= 5.0
    vert_col = torch.rand(n_points, 3, dtype=torch.float32)
    vert_rad = torch.rand(n_points, dtype=torch.float32)
    cam_params = torch.tensor(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 2.0], dtype=torch.float32
    )
    device = torch.device("cuda")
    vert_pos = vert_pos.to(device)
    vert_col = vert_col.to(device)
    vert_rad = vert_rad.to(device)
    cam_params = cam_params.to(device)
    renderer = renderer.to(device)
    vert_pos_var = Variable(vert_pos, requires_grad=False)
    vert_col_var = Variable(vert_col, requires_grad=False)
    vert_rad_var = Variable(vert_rad, requires_grad=False)
    cam_params_var = Variable(cam_params, requires_grad=False)

    def bm_closure():
        renderer.forward(
            vert_pos_var,
            vert_col_var,
            vert_rad_var,
            cam_params_var,
            1.0e-1,
            45.0,
            percent_allowed_difference=0.01,
        )
        torch.cuda.synchronize()

    return bm_closure


def _bm_pulsar_backward():
    n_points = 1_000_000
    width = 1_000
    height = 1_000
    renderer = Renderer(width, height, n_points)
    # Generate sample data.
    torch.manual_seed(1)
    vert_pos = torch.rand(n_points, 3, dtype=torch.float32) * 10.0
    vert_pos[:, 2] += 25.0
    vert_pos[:, :2] -= 5.0
    vert_col = torch.rand(n_points, 3, dtype=torch.float32)
    vert_rad = torch.rand(n_points, dtype=torch.float32)
    cam_params = torch.tensor(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 2.0], dtype=torch.float32
    )
    device = torch.device("cuda")
    vert_pos = vert_pos.to(device)
    vert_col = vert_col.to(device)
    vert_rad = vert_rad.to(device)
    cam_params = cam_params.to(device)
    renderer = renderer.to(device)
    vert_pos_var = Variable(vert_pos, requires_grad=True)
    vert_col_var = Variable(vert_col, requires_grad=True)
    vert_rad_var = Variable(vert_rad, requires_grad=True)
    cam_params_var = Variable(cam_params, requires_grad=True)
    res = renderer.forward(
        vert_pos_var,
        vert_col_var,
        vert_rad_var,
        cam_params_var,
        1.0e-1,
        45.0,
        percent_allowed_difference=0.01,
    )
    loss = res.sum()

    def bm_closure():
        loss.backward(retain_graph=True)
        torch.cuda.synchronize()

    return bm_closure


def bm_pulsar() -> None:
    if not torch.cuda.is_available():
        return

    benchmark(_bm_pulsar, "PULSAR_FORWARD", [{}], warmup_iters=3)
    benchmark(_bm_pulsar_backward, "PULSAR_BACKWARD", [{}], warmup_iters=3)


if __name__ == "__main__":
    bm_pulsar()
