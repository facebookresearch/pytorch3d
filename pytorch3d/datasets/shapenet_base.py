# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from typing import Dict

import torch
from pytorch3d.renderer import (
    HardPhongShader,
    MeshRasterizer,
    MeshRenderer,
    OpenGLPerspectiveCameras,
    PointLights,
    RasterizationSettings,
)
from pytorch3d.structures import Meshes, Textures


class ShapeNetBase(torch.utils.data.Dataset):
    """
    'ShapeNetBase' implements a base Dataset for ShapeNet and R2N2 with helper methods.
    It is not intended to be used on its own as a Dataset for a Dataloader. Both __init__
    and __getitem__ need to be implemented.
    """

    def __init__(self):
        """
        Set up lists of synset_ids and model_ids.
        """
        self.synset_ids = []
        self.model_ids = []

    def __len__(self):
        """
        Return number of total models in the loaded dataset.
        """
        return len(self.model_ids)

    def __getitem__(self, idx) -> Dict:
        """
        Read a model by the given index. Need to be implemented for every child class
        of ShapeNetBase.

        Args:
            idx: The idx of the model to be retrieved in the dataset.

        Returns:
            dictionary containing information about the model.
        """
        raise NotImplementedError(
            "__getitem__ should be implemented in the child class of ShapeNetBase"
        )

    def _get_item_ids(self, idx) -> Dict:
        """
        Read a model by the given index.

        Args:
            idx: The idx of the model to be retrieved in the dataset.

        Returns:
            dictionary with following keys:
            - synset_id (str): synset id
            - model_id (str): model id
        """
        model = {}
        model["synset_id"] = self.synset_ids[idx]
        model["model_id"] = self.model_ids[idx]
        return model

    def render(
        self, idx: int = 0, shader_type=HardPhongShader, device="cpu", **kwargs
    ) -> torch.Tensor:
        """
        Renders a model by the given index.

        Args:
            idx: The index of model to be rendered in the dataset.
            shader_type: select shading. Valid options include HardPhongShader (default),
                SoftPhongShader, HardGouraudShader, SoftGouraudShader, HardFlatShader,
                SoftSilhouetteShader.
            device: torch.device on which the tensors should be located.
            **kwargs: Accepts any of the kwargs that the renderer supports.

        Returns:
            Rendered image of shape (1, H, W, 3).
        """

        model = self.__getitem__(idx)
        verts, faces = model["verts"], model["faces"]
        verts_rgb = torch.ones_like(verts, device=device)[None]
        mesh = Meshes(
            verts=[verts.to(device)],
            faces=[faces.to(device)],
            textures=Textures(verts_rgb=verts_rgb.to(device)),
        )
        cameras = kwargs.get("cameras", OpenGLPerspectiveCameras()).to(device)
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=kwargs.get("raster_settings", RasterizationSettings()),
            ),
            shader=shader_type(
                device=device,
                cameras=cameras,
                lights=kwargs.get("lights", PointLights()).to(device),
            ),
        )
        return renderer(mesh)
