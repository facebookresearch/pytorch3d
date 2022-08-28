# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools

from fvcore.common.benchmark import benchmark
from tests.test_cameras import TestCamerasCommon


def _setUp():
    case_grid = {
        "cam_type": [
            "OpenGLOrthographicCameras",
            "OpenGLPerspectiveCameras",
            "SfMOrthographicCameras",
            "SfMPerspectiveCameras",
            "FoVOrthographicCameras",
            "FoVPerspectiveCameras",
            "OrthographicCameras",
            "PerspectiveCameras",
            "FishEyeCameras",
        ],
        "batch_size": [1, 10],
        "num_points": [10, 100],
        "device": ["cpu", "cuda:0"],
    }
    test_cases = itertools.product(*case_grid.values())
    kwargs_list = [dict(zip(case_grid.keys(), case)) for case in test_cases]
    return kwargs_list


def _bm_cameras_project() -> None:
    kwargs_list = _setUp()
    benchmark(
        TestCamerasCommon.transform_points,
        "TEST_TRANSFORM_POINTS",
        kwargs_list,
    )


def _bm_cameras_unproject() -> None:
    kwargs_list = _setUp()
    benchmark(
        TestCamerasCommon.unproject_points,
        "TEST_UNPROJECT_POINTS",
        kwargs_list,
    )


def bm_cameras() -> None:
    _bm_cameras_project()
    _bm_cameras_unproject()


if __name__ == "__main__":
    bm_cameras()
