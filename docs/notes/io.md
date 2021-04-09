---
hide_title: true
sidebar_label: File IO
---

# File IO
There is a flexible interface for loading and saving point clouds and meshes from different formats.

The main usage is via the `pytorch3d.io.IO` object, and its methods
`load_mesh`, `save_mesh`, `load_point_cloud` and `save_point_cloud`.

For example, to load a mesh you might do
```
from pytorch3d.io import IO

device=torch.device("cuda:0")
mesh = IO().load_mesh("mymesh.ply", device=device)
```

and to save a pointcloud you might do
```
pcl = Pointclouds(...)
IO().save_point_cloud(pcl, "output_pointcloud.obj")
```
