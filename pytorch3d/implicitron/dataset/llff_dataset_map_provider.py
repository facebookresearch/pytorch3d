# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe


import numpy as np
import torch
from pytorch3d.implicitron.tools.config import registry

from .load_llff import load_llff_data

from .single_sequence_dataset import (
    _interpret_blender_cameras,
    SingleSceneDatasetMapProviderBase,
)


@registry.register
class LlffDatasetMapProvider(SingleSceneDatasetMapProviderBase):
    """
    Provides data for one scene from the LLFF dataset.

    Members:
        base_dir: directory holding the data for the scene.
        object_name: The name of the scene (e.g. "fern"). This is just used as a label.
            It will typically be equal to the name of the directory self.base_dir.
        path_manager_factory: Creates path manager which may be used for
            interpreting paths.
        n_known_frames_for_test: If set, training frames are included in the val
            and test datasets, and this many random training frames are added to
            each test batch. If not set, test batches each contain just a single
            testing frame.
        downscale_factor: determines image sizes.
    """

    downscale_factor: int = 4

    def _load_data(self) -> None:
        path_manager = self.path_manager_factory.get()
        images, poses, _ = load_llff_data(
            self.base_dir, factor=self.downscale_factor, path_manager=path_manager
        )
        hwf = poses[0, :3, -1]
        poses = poses[:, :3, :4]

        llffhold = 8
        i_test = np.arange(images.shape[0])[::llffhold]
        i_test_index = set(i_test.tolist())
        i_train = np.array(
            [i for i in np.arange(images.shape[0]) if i not in i_test_index]
        )
        i_split = (i_train, i_test, i_test)
        H, W, focal = hwf
        focal_ndc = 2 * focal / min(H, W)
        images = torch.from_numpy(images).permute(0, 3, 1, 2)
        poses = torch.from_numpy(poses)

        # pyre-ignore[16]
        self.poses = _interpret_blender_cameras(poses, focal_ndc)
        # pyre-ignore[16]
        self.images = images
        # pyre-ignore[16]
        self.fg_probabilities = None
        # pyre-ignore[16]
        self.i_split = i_split
