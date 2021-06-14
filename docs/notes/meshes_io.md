---
sidebar_label: Loading from file
hide_title: true
---

# Meshes and IO

The Meshes object represents a batch of triangulated meshes, and is central to
much of the functionality of PyTorch3D. There is no insistence that each mesh in
the batch has the same number of vertices or faces. When available, it can store
other data which pertains to the mesh, for example face normals, face areas
and textures.

Two common file formats for storing single meshes are ".obj" and ".ply" files,
and PyTorch3D has functions for reading these.

## OBJ

Obj files have a standard way to store extra information about a mesh. Given an
obj file, it can be read with

```
  verts, faces, aux = load_obj(filename)
```

which sets `verts` to be a (V,3)-tensor of vertices and `faces.verts_idx` to be
an (F,3)- tensor of the vertex-indices of each of the corners of the faces.
Faces which are not triangles will be split into triangles. `aux` is an object
which may contain normals, uv coordinates, material colors and textures if they
are present, and `faces` may additionally contain indices into these normals,
textures and materials in its NamedTuple structure. A Meshes object containing a
single mesh can be created from just the vertices and faces using
```
    meshes = Meshes(verts=[verts], faces=[faces.verts_idx])
```

If there is texture information in the `.obj` it can be used to initialize a
`Textures` class which is passed into the `Meshes` constructor.  Currently we
support loading of texture maps for meshes which have one texture map for the
entire mesh e.g.

```
verts_uvs = aux.verts_uvs[None, ...]  # (1, V, 2)
faces_uvs = faces.textures_idx[None, ...]  # (1, F, 3)
tex_maps = aux.texture_images

# tex_maps is a dictionary of {material name: texture image}.
# Take the first image:
texture_image = list(tex_maps.values())[0]
texture_image = texture_image[None, ...]  # (1, H, W, 3)

# Create a textures object
tex = Textures(verts_uvs=verts_uvs, faces_uvs=faces_uvs, maps=texture_image)

# Initialise the mesh with textures
meshes = Meshes(verts=[verts], faces=[faces.verts_idx], textures=tex)
```

The `load_objs_as_meshes` function provides this procedure.

## PLY

Ply files are flexible in the way they store additional information. PyTorch3D
provides a function just to read the vertices and faces from a ply file.
The call
```
    verts, faces = load_ply(filename)
```
sets `verts` to be a (V,3)-tensor of vertices and `faces` to be an (F,3)-
tensor of the vertex-indices of each of the corners of the faces. Faces which
are not triangles will be split into triangles. A Meshes object containing a
single mesh can be created from this data using
```
    meshes = Meshes(verts=[verts], faces=[faces])
```
