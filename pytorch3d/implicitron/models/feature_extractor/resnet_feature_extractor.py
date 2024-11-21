# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import logging
import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as Fu
import torchvision
from pytorch3d.implicitron.tools.config import registry

from . import FeatureExtractorBase


logger = logging.getLogger(__name__)

MASK_FEATURE_NAME = "mask"
IMAGE_FEATURE_NAME = "image"

_FEAT_DIMS = {
    "resnet18": (64, 128, 256, 512),
    "resnet34": (64, 128, 256, 512),
    "resnet50": (256, 512, 1024, 2048),
    "resnet101": (256, 512, 1024, 2048),
    "resnet152": (256, 512, 1024, 2048),
}

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


@registry.register
class ResNetFeatureExtractor(FeatureExtractorBase):
    """
    Implements an image feature extractor. Depending on the settings allows
    to extract:
        - deep features: A CNN ResNet backbone from torchvision (with/without
            pretrained weights) which extracts deep features.
        - masks: Segmentation masks.
        - images: Raw input RGB images.

    Settings:
        name: name of the resnet backbone (from torchvision)
        pretrained: If true, will load the pretrained weights
        stages: List of stages from which to extract features.
            Features from each stage are returned as key value
            pairs in the forward function
        normalize_image: If set will normalize the RGB values of
            the image based on the Resnet mean/std
        image_rescale: If not 1.0, this rescale factor will be
            used to resize the image
        first_max_pool: If set, a max pool layer is added after the first
            convolutional layer
        proj_dim: The number of output channels for the convolutional layers
        l2_norm: If set, l2 normalization is applied to the extracted features
        add_masks: If set, the masks will be saved in the output dictionary
        add_images: If set, the images will be saved in the output dictionary
        global_average_pool: If set, global average pooling step is performed
        feature_rescale: If not 1.0, this rescale factor will be used to
            rescale the output features
    """

    name: str = "resnet34"
    pretrained: bool = True
    stages: Tuple[int, ...] = (1, 2, 3, 4)
    normalize_image: bool = True
    image_rescale: float = 128 / 800.0
    first_max_pool: bool = True
    proj_dim: int = 32
    l2_norm: bool = True
    add_masks: bool = True
    add_images: bool = True
    global_average_pool: bool = False  # this can simulate global/non-spacial features
    feature_rescale: float = 1.0

    def __post_init__(self):
        if self.normalize_image:
            # register buffers needed to normalize the image
            for k, v in (("_resnet_mean", _RESNET_MEAN), ("_resnet_std", _RESNET_STD)):
                self.register_buffer(
                    k,
                    torch.FloatTensor(v).view(1, 3, 1, 1),
                    persistent=False,
                )

        self._feat_dim = {}

        if len(self.stages) == 0:
            # do not extract any resnet features
            pass
        else:
            net = getattr(torchvision.models, self.name)(pretrained=self.pretrained)
            if self.first_max_pool:
                self.stem = torch.nn.Sequential(
                    net.conv1, net.bn1, net.relu, net.maxpool
                )
            else:
                self.stem = torch.nn.Sequential(net.conv1, net.bn1, net.relu)
            self.max_stage = max(self.stages)
            self.layers = torch.nn.ModuleList()
            self.proj_layers = torch.nn.ModuleList()
            for stage in range(self.max_stage):
                stage_name = f"layer{stage + 1}"
                feature_name = self._get_resnet_stage_feature_name(stage)
                if (stage + 1) in self.stages:
                    if (
                        self.proj_dim > 0
                        and _FEAT_DIMS[self.name][stage] > self.proj_dim
                    ):
                        proj = torch.nn.Conv2d(
                            _FEAT_DIMS[self.name][stage],
                            self.proj_dim,
                            1,
                            1,
                            bias=True,
                        )
                        self._feat_dim[feature_name] = self.proj_dim
                    else:
                        proj = torch.nn.Identity()
                        self._feat_dim[feature_name] = _FEAT_DIMS[self.name][stage]
                else:
                    proj = torch.nn.Identity()
                self.proj_layers.append(proj)
                self.layers.append(getattr(net, stage_name))

        if self.add_masks:
            self._feat_dim[MASK_FEATURE_NAME] = 1

        if self.add_images:
            self._feat_dim[IMAGE_FEATURE_NAME] = 3

        logger.info(f"Feat extractor total dim = {self.get_feat_dims()}")
        self.stages = set(self.stages)  # convert to set for faster "in"

    def _get_resnet_stage_feature_name(self, stage) -> str:
        return f"res_layer_{stage + 1}"

    def _resnet_normalize_image(self, img: torch.Tensor) -> torch.Tensor:
        # pyre-fixme[58]: `-` is not supported for operand types `Tensor` and
        #  `Union[Tensor, Module]`.
        # pyre-fixme[58]: `/` is not supported for operand types `Tensor` and
        #  `Union[Tensor, Module]`.
        return (img - self._resnet_mean) / self._resnet_std

    def get_feat_dims(self) -> int:
        # pyre-fixme[29]: `Union[(self: TensorBase) -> Tensor, Tensor, Module]` is
        #  not a function.
        return sum(self._feat_dim.values())

    def forward(
        self,
        imgs: Optional[torch.Tensor],
        masks: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[Any, torch.Tensor]:
        """
        Args:
            imgs: A batch of input images of shape `(B, 3, H, W)`.
            masks: A batch of input masks of shape `(B, 3, H, W)`.

        Returns:
            out_feats: A dict `{f_i: t_i}` keyed by predicted feature names `f_i`
                and their corresponding tensors `t_i` of shape `(B, dim_i, H_i, W_i)`.
        """

        out_feats = {}

        imgs_input = imgs
        if self.image_rescale != 1.0 and imgs_input is not None:
            imgs_resized = Fu.interpolate(
                imgs_input,
                scale_factor=self.image_rescale,
                mode="bilinear",
            )
        else:
            imgs_resized = imgs_input

        if len(self.stages) > 0:
            assert imgs_resized is not None

            if self.normalize_image:
                imgs_normed = self._resnet_normalize_image(imgs_resized)
            else:
                imgs_normed = imgs_resized
            #  is not a function.
            # pyre-fixme[29]: `Union[Tensor, Module]` is not a function.
            feats = self.stem(imgs_normed)
            # pyre-fixme[6]: For 1st argument expected `Iterable[_T1]` but got
            #  `Union[Tensor, Module]`.
            # pyre-fixme[6]: For 2nd argument expected `Iterable[_T2]` but got
            #  `Union[Tensor, Module]`.
            for stage, (layer, proj) in enumerate(zip(self.layers, self.proj_layers)):
                feats = layer(feats)
                # just a sanity check below
                assert feats.shape[1] == _FEAT_DIMS[self.name][stage]
                if (stage + 1) in self.stages:
                    f = proj(feats)
                    if self.global_average_pool:
                        f = f.mean(dims=(2, 3))
                    if self.l2_norm:
                        normfac = 1.0 / math.sqrt(len(self.stages))
                        f = Fu.normalize(f, dim=1) * normfac
                    feature_name = self._get_resnet_stage_feature_name(stage)
                    out_feats[feature_name] = f

        if self.add_masks:
            assert masks is not None
            out_feats[MASK_FEATURE_NAME] = masks

        if self.add_images:
            assert imgs_resized is not None
            out_feats[IMAGE_FEATURE_NAME] = imgs_resized

        if self.feature_rescale != 1.0:
            out_feats = {k: self.feature_rescale * f for k, f in out_feats.items()}

        # pyre-fixme[7]: Incompatible return type, expected `Dict[typing.Any, Tensor]`
        # but got `Dict[typing.Any, float]`
        return out_feats
