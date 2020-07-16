# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.


from pathlib import Path
import unittest

import numpy as np
import torch
import torch.nn as nn
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    OpenGLPerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRasterizer, MeshRenderer
)
from pytorch3d.renderer.mesh.shader import NormalShader, UVsCorrespondenceShader
from PIL import Image


class Test(unittest.TestCase):
    def test_correpondence_mapping(self):
        device = torch.device('cuda:0')

        torch.cuda.set_device(device)
        # Set paths
        data_dir = Path(__file__).parent  / 'data'
        data_dir.mkdir(exist_ok=True)
        obj_dir = Path(__file__).resolve().parent.parent  / "docs/tutorials/data"
        obj_filename = obj_dir / "cow_mesh/cow.obj"

        # Load obj file
        mesh = load_objs_as_meshes([obj_filename], device=device)

        try:
            texture_image = mesh.textures.maps_padded()
        except:
            pass

        R, T = look_at_view_transform(2.55, 10, 180)

        cameras = OpenGLPerspectiveCameras(device=device, R=R, T=T)

        # Define the settings for rasterization and shading. Here we set the output image to be of size
        # 512x512. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
        # and blur_radius=0.0. We also set bin_size and max_faces_per_bin to None which ensure that
        # the faster coarse-to-fine rasterization method is used. Refer to rasterize_meshes.py for
        # explanations of these parameters. Refer to docs/notes/renderer.md for an explanation of
        # the difference between naive and coarse-to-fine rasterization.
        raster_settings = RasterizationSettings(
            image_size=512,
            blur_radius=0.0,
            faces_per_pixel=1,
            perspective_correct=True,
        )

        # create a colormap to be displayed on the object.

        ##generating some  data
        x, y = np.meshgrid(
            np.linspace(1, 0, 100),
            np.linspace(0, 1, 100),
        )
        directions = (np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y) + 1) * np.pi
        magnitude = np.exp(-(x * x + y * y))

        ##normalize data:
        def normalize(M):
            return (M - np.min(M)) / (np.max(M) - np.min(M))

        d_norm = normalize(directions)
        m_norm = normalize(magnitude)
        colors = np.dstack((x, y, np.zeros_like(x)))
        colors = (torch.from_numpy(np.array(colors))).unsqueeze(0).float()


        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=raster_settings
            ),

            shader=UVsCorrespondenceShader(
                device=device,
                cameras=cameras,
                colormap=colors
            )
        )
        images = renderer(mesh)
        # cv2.imshow('render_correspondence_texture.png',
        #            ((255 * images[0, ..., :3]).squeeze().cpu().numpy().astype(np.uint8))[..., ::-1])

        # cv2.imwrite(str(data_dir / 'render_correspondence_texture.png'),
        #             ((255 * images[0, ..., :3]).squeeze().cpu().numpy().astype(np.uint8))[..., ::-1])
        Image.fromarray(((255 * images[0, ..., :3]).squeeze().cpu().numpy().astype(np.uint8))).save(str(data_dir / 'render_correspondence_texture.png')
        )
        # cv2.waitKey(0)
        self.assertTrue((data_dir / 'render_correspondence_texture.png').exists())


if __name__ == '__main__':
    # begin the unittest.main()
    unittest.main()
