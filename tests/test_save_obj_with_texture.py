# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from pathlib import Path
import unittest
import torch
from pytorch3d.io import load_objs_as_meshes, save_obj


class Test(unittest.TestCase):
    def test_save_obj_with_texture(self):
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)
        # Set paths
        data_dir = Path(__file__).parent / 'data'
        data_dir.mkdir(exist_ok=True)
        obj_dir = Path(__file__).resolve().parent.parent / "docs/tutorials/data"
        obj_filename = obj_dir / "cow_mesh/cow.obj"
        final_obj = data_dir / "cow_exported.obj"
        # Load obj file
        mesh = load_objs_as_meshes([obj_filename], device=device)

        try:
            texture_image = mesh.textures.maps_padded()
            save_obj(final_obj, mesh.verts_packed(), mesh.faces_packed(),
                     verts_uvs=mesh.textures.verts_uvs_packed(), texture_map=texture_image,
                     faces_uvs=mesh.textures.faces_uvs_packed())
        except:
            pass
if __name__ == '__main__':
    # begin the unittest.main()
    unittest.main()
