# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools

from fvcore.common.benchmark import benchmark
from pytorch3d.renderer import AbsorptionOnlyRaymarcher, EmissionAbsorptionRaymarcher
from tests.test_render_implicit import TestRenderImplicit


def bm_render_volumes() -> None:
    case_grid = {
        "batch_size": [1, 5],
        "raymarcher_type": [EmissionAbsorptionRaymarcher, AbsorptionOnlyRaymarcher],
        "n_rays_per_image": [64**2, 256**2],
        "n_pts_per_ray": [16, 128],
    }
    test_cases = itertools.product(*case_grid.values())
    kwargs_list = [dict(zip(case_grid.keys(), case)) for case in test_cases]

    benchmark(
        TestRenderImplicit.renderer, "IMPLICIT_RENDERER", kwargs_list, warmup_iters=1
    )
