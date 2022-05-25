# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools

from fvcore.common.benchmark import benchmark
from pytorch3d.renderer import (
    FoVOrthographicCameras,
    FoVPerspectiveCameras,
    MonteCarloRaysampler,
    MultinomialRaysampler,
    NDCMultinomialRaysampler,
    OrthographicCameras,
    PerspectiveCameras,
)
from tests.test_raysampling import TestRaysampling


def bm_raysampling() -> None:
    case_grid = {
        "raysampler_type": [
            MultinomialRaysampler,
            NDCMultinomialRaysampler,
            MonteCarloRaysampler,
        ],
        "camera_type": [
            PerspectiveCameras,
            OrthographicCameras,
            FoVPerspectiveCameras,
            FoVOrthographicCameras,
        ],
        "batch_size": [1, 10],
        "n_pts_per_ray": [10, 1000, 10000],
        "image_width": [10, 300],
        "image_height": [10, 300],
    }
    test_cases = itertools.product(*case_grid.values())
    kwargs_list = [dict(zip(case_grid.keys(), case)) for case in test_cases]

    benchmark(TestRaysampling.raysampler, "RAYSAMPLER", kwargs_list, warmup_iters=1)


if __name__ == "__main__":
    bm_raysampling()
