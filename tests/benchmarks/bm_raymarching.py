# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools

from fvcore.common.benchmark import benchmark
from pytorch3d.renderer import AbsorptionOnlyRaymarcher, EmissionAbsorptionRaymarcher
from tests.test_raymarching import TestRaymarching


def bm_raymarching() -> None:
    case_grid = {
        "raymarcher_type": [EmissionAbsorptionRaymarcher, AbsorptionOnlyRaymarcher],
        "n_rays": [10, 1000, 10000],
        "n_pts_per_ray": [10, 1000, 10000],
    }
    test_cases = itertools.product(*case_grid.values())
    kwargs_list = [dict(zip(case_grid.keys(), case)) for case in test_cases]

    benchmark(TestRaymarching.raymarcher, "RAYMARCHER", kwargs_list, warmup_iters=1)
