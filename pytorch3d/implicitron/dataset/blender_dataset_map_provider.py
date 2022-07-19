# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch
from pytorch3d.implicitron.tools.config import registry

from .load_blender import load_blender_data
from .single_sequence_dataset import (
    _interpret_blender_cameras,
    SingleSceneDatasetMapProviderBase,
)


@registry.register
class BlenderDatasetMapProvider(SingleSceneDatasetMapProviderBase):
    """
    Provides data for one scene from Blender synthetic dataset.
    Uses the code in load_blender.py

    Members:
        base_dir: directory holding the data for the scene.
        object_name: The name of the scene (e.g. "lego"). This is just used as a label.
            It will typically be equal to the name of the directory self.base_dir.
        path_manager_factory: Creates path manager which may be used for
            interpreting paths.
        n_known_frames_for_test: If set, training frames are included in the val
            and test datasets, and this many random training frames are added to
            each test batch. If not set, test batches each contain just a single
            testing frame.
    """

    def _load_data(self) -> None:
        path_manager = self.path_manager_factory.get()
        images, poses, _, hwf, i_split = load_blender_data(
            self.base_dir,
            testskip=1,
            path_manager=path_manager,
        )
        H, W, focal = hwf
        images_masks = torch.from_numpy(images).permute(0, 3, 1, 2)

        # pyre-ignore[16]
        self.poses = _interpret_blender_cameras(poses, focal)
        # pyre-ignore[16]
        self.images = images_masks[:, :3]
        # pyre-ignore[16]
        self.fg_probabilities = images_masks[:, 3:4]
        # pyre-ignore[16]
        self.i_split = i_split
