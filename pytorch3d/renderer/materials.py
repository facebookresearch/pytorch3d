# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.


import torch

from .utils import TensorProperties


class Materials(TensorProperties):
    """
    A class for storing a batch of material properties. Currently only one
    material per batch element is supported.
    """

    def __init__(
        self,
        ambient_color=((1, 1, 1),),
        diffuse_color=((1, 1, 1),),
        specular_color=((1, 1, 1),),
        shininess=64,
        device="cpu",
    ):
        """
        Args:
            ambient_color: RGB ambient reflectivity of the material
            diffuse_color: RGB diffuse reflectivity of the material
            specular_color: RGB specular reflectivity of the material
            shininess: The specular exponent for the material. This defines
                the focus of the specular highlight with a high value
                resulting in a concentrated highlight. Shininess values
                can range from 0-1000.
            device: torch.device or string

        ambient_color, diffuse_color and specular_color can be of shape
        (1, 3) or (N, 3). shininess can be of shape (1) or (N).

        The colors and shininess are broadcast against each other so need to
        have either the same batch dimension or batch dimension = 1.
        """
        super().__init__(
            device=device,
            diffuse_color=diffuse_color,
            ambient_color=ambient_color,
            specular_color=specular_color,
            shininess=shininess,
        )
        for n in ["ambient_color", "diffuse_color", "specular_color"]:
            t = getattr(self, n)
            if t.shape[-1] != 3:
                msg = "Expected %s to have shape (N, 3); got %r"
                raise ValueError(msg % (n, t.shape))
        if self.shininess.shape != torch.Size([self._N]):
            msg = "shininess should have shape (N); got %r"
            raise ValueError(msg % repr(self.shininess.shape))

    def clone(self):
        other = Materials(device=self.device)
        return super().clone(other)
