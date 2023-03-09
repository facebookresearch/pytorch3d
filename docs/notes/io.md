---
hide_title: true
sidebar_label: File IO
---

# File IO
There is a flexible interface for loading and saving point clouds and meshes from different formats.

The main usage is via the `pytorch3d.io.IO` object, and its methods
`load_mesh`, `save_mesh`, `load_pointcloud` and `save_pointcloud`.

For example, to load a mesh you might do
```
from pytorch3d.io import IO

device=torch.device("cuda:0")
mesh = IO().load_mesh("mymesh.obj", device=device)
```

and to save a pointcloud you might do
```
pcl = Pointclouds(...)
IO().save_pointcloud(pcl, "output_pointcloud.ply")
```

For meshes, this supports OBJ, PLY and OFF files.

For pointclouds, this supports PLY files.

In addition, there is experimental support for loading meshes from
[glTF 2 assets](https://github.com/KhronosGroup/glTF/tree/master/specification/2.0)
stored either in a GLB container file or a glTF JSON file with embedded binary data.
This must be enabled explicitly, as described in
`pytorch3d/io/experimental_gltf_io.py`.
