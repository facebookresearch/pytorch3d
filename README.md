<img src="https://raw.githubusercontent.com/facebookresearch/pytorch3d/main/.github/pytorch3dlogo.png" width="900"/>

[![CircleCI](https://circleci.com/gh/facebookresearch/pytorch3d.svg?style=svg)](https://circleci.com/gh/facebookresearch/pytorch3d)
[![Anaconda-Server Badge](https://anaconda.org/pytorch3d/pytorch3d/badges/version.svg)](https://anaconda.org/pytorch3d/pytorch3d)

# Introduction

PyTorch3D provides efficient, reusable components for 3D Computer Vision research with [PyTorch](https://pytorch.org).

Key features include:

- Data structure for storing and manipulating triangle meshes
- Efficient operations on triangle meshes (projective transformations, graph convolution, sampling, loss functions)
- A differentiable mesh renderer
- Implicitron, see [its README](projects/implicitron_trainer), a framework for new-view synthesis via implicit representations. ([blog post](https://ai.facebook.com/blog/implicitron-a-new-modular-extensible-framework-for-neural-implicit-representations-in-pytorch3d/))

PyTorch3D is designed to integrate smoothly with deep learning methods for predicting and manipulating 3D data.
For this reason, all operators in PyTorch3D:

- Are implemented using PyTorch tensors
- Can handle minibatches of hetereogenous data
- Can be differentiated
- Can utilize GPUs for acceleration

Within FAIR, PyTorch3D has been used to power research projects such as [Mesh R-CNN](https://arxiv.org/abs/1906.02739).

See our [blog post](https://ai.facebook.com/blog/-introducing-pytorch3d-an-open-source-library-for-3d-deep-learning/) to see more demos and learn about PyTorch3D.

## Installation

For detailed instructions refer to [INSTALL.md](INSTALL.md).

## License

PyTorch3D is released under the [BSD License](LICENSE).

## Tutorials

Get started with PyTorch3D by trying one of the tutorial notebooks.

|<img src="https://raw.githubusercontent.com/facebookresearch/pytorch3d/main/.github/dolphin_deform.gif" width="310"/>|<img src="https://raw.githubusercontent.com/facebookresearch/pytorch3d/main/.github/bundle_adjust.gif" width="310"/>|
|:-----------------------------------------------------------------------------------------------------------:|:--------------------------------------------------:|
| [Deform a sphere mesh to dolphin](https://github.com/facebookresearch/pytorch3d/blob/main/docs/tutorials/deform_source_mesh_to_target_mesh.ipynb)| [Bundle adjustment](https://github.com/facebookresearch/pytorch3d/blob/main/docs/tutorials/bundle_adjustment.ipynb) |

| <img src="https://raw.githubusercontent.com/facebookresearch/pytorch3d/main/.github/render_textured_mesh.gif" width="310"/> | <img src="https://raw.githubusercontent.com/facebookresearch/pytorch3d/main/.github/camera_position_teapot.gif" width="310" height="310"/>
|:------------------------------------------------------------:|:--------------------------------------------------:|
| [Render textured meshes](https://github.com/facebookresearch/pytorch3d/blob/main/docs/tutorials/render_textured_meshes.ipynb)| [Camera position optimization](https://github.com/facebookresearch/pytorch3d/blob/main/docs/tutorials/camera_position_optimization_with_differentiable_rendering.ipynb)|

| <img src="https://raw.githubusercontent.com/facebookresearch/pytorch3d/main/.github/pointcloud_render.png" width="310"/> | <img src="https://raw.githubusercontent.com/facebookresearch/pytorch3d/main/.github/cow_deform.gif" width="310" height="310"/>
|:------------------------------------------------------------:|:--------------------------------------------------:|
| [Render textured pointclouds](https://github.com/facebookresearch/pytorch3d/blob/main/docs/tutorials/render_colored_points.ipynb)| [Fit a mesh with texture](https://github.com/facebookresearch/pytorch3d/blob/main/docs/tutorials/fit_textured_mesh.ipynb)|

| <img src="https://raw.githubusercontent.com/facebookresearch/pytorch3d/main/.github/densepose_render.png" width="310"/> | <img src="https://raw.githubusercontent.com/facebookresearch/pytorch3d/main/.github/shapenet_render.png" width="310" height="310"/>
|:------------------------------------------------------------:|:--------------------------------------------------:|
| [Render DensePose data](https://github.com/facebookresearch/pytorch3d/blob/main/docs/tutorials/render_densepose.ipynb)| [Load & Render ShapeNet data](https://github.com/facebookresearch/pytorch3d/blob/main/docs/tutorials/dataloaders_ShapeNetCore_R2N2.ipynb)|

| <img src="https://raw.githubusercontent.com/facebookresearch/pytorch3d/main/.github/fit_textured_volume.gif" width="310"/> | <img src="https://raw.githubusercontent.com/facebookresearch/pytorch3d/main/.github/fit_nerf.gif" width="310" height="310"/>
|:------------------------------------------------------------:|:--------------------------------------------------:|
| [Fit Textured Volume](https://github.com/facebookresearch/pytorch3d/blob/main/docs/tutorials/fit_textured_volume.ipynb)| [Fit A Simple Neural Radiance Field](https://github.com/facebookresearch/pytorch3d/blob/main/docs/tutorials/fit_simple_neural_radiance_field.ipynb)|

| <img src="https://raw.githubusercontent.com/facebookresearch/pytorch3d/main/.github/fit_textured_volume.gif" width="310"/> | <img src="https://raw.githubusercontent.com/facebookresearch/pytorch3d/main/.github/implicitron_config.gif" width="310" height="310"/>
|:------------------------------------------------------------:|:--------------------------------------------------:|
| [Fit Textured Volume in Implicitron](https://github.com/facebookresearch/pytorch3d/blob/main/docs/tutorials/implicitron_volumes.ipynb)| [Implicitron Config System](https://github.com/facebookresearch/pytorch3d/blob/main/docs/tutorials/implicitron_config_system.ipynb)|





## Documentation

Learn more about the API by reading the PyTorch3D [documentation](https://pytorch3d.readthedocs.org/).

We also have deep dive notes on several API components:

- [Heterogeneous Batching](https://github.com/facebookresearch/pytorch3d/tree/main/docs/notes/batching.md)
- [Mesh IO](https://github.com/facebookresearch/pytorch3d/tree/main/docs/notes/meshes_io.md)
- [Differentiable Rendering](https://github.com/facebookresearch/pytorch3d/tree/main/docs/notes/renderer_getting_started.md)

### Overview Video

We have created a short (~14 min) video tutorial providing an overview of the PyTorch3D codebase including several code examples. Click on the image below to watch the video on YouTube:

<a href="http://www.youtube.com/watch?v=Pph1r-x9nyY"><img src="http://img.youtube.com/vi/Pph1r-x9nyY/0.jpg" height="225" ></a>

## Development

We welcome new contributions to PyTorch3D and we will be actively maintaining this library! Please refer to [CONTRIBUTING.md](./.github/CONTRIBUTING.md) for full instructions on how to run the code, tests and linter, and submit your pull requests.

## Development and Compatibility

- `main` branch: actively developed, without any guarantee, Anything can be broken at any time
  - REMARK: this includes nightly builds which are built from `main`
  - HINT: the commit history can help locate regressions or changes
- backward-compatibility between releases: no guarantee. Best efforts to communicate breaking changes and facilitate migration of code or data (incl. models).

## Contributors

PyTorch3D is written and maintained by the Facebook AI Research Computer Vision Team.

In alphabetical order:

* Amitav Baruah
* Steve Branson
* Krzysztof Chalupka
* Jiali Duan
* Luya Gao
* Georgia Gkioxari
* Taylor Gordon
* Justin Johnson
* Patrick Labatut
* Christoph Lassner
* Wan-Yen Lo
* David Novotny
* Nikhila Ravi
* Jeremy Reizenstein
* Dave Schnizlein
* Roman Shapovalov
* Olivia Wiles

## Citation

If you find PyTorch3D useful in your research, please cite our tech report:

```bibtex
@article{ravi2020pytorch3d,
    author = {Nikhila Ravi and Jeremy Reizenstein and David Novotny and Taylor Gordon
                  and Wan-Yen Lo and Justin Johnson and Georgia Gkioxari},
    title = {Accelerating 3D Deep Learning with PyTorch3D},
    journal = {arXiv:2007.08501},
    year = {2020},
}
```

If you are using the pulsar backend for sphere-rendering (the `PulsarPointRenderer` or `pytorch3d.renderer.points.pulsar.Renderer`), please cite the tech report:

```bibtex
@article{lassner2020pulsar,
    author = {Christoph Lassner and Michael Zollh\"ofer},
    title = {Pulsar: Efficient Sphere-based Neural Rendering},
    journal = {arXiv:2004.07484},
    year = {2020},
}
```

## News

Please see below for a timeline of the codebase updates in reverse chronological order. We are sharing updates on the releases as well as research projects which are built with PyTorch3D. The changelogs for the releases are available under [`Releases`](https://github.com/facebookresearch/pytorch3d/releases),  and the builds can be installed using `conda` as per the instructions in [INSTALL.md](INSTALL.md).

**[Dec 19th 2022]:**   PyTorch3D [v0.7.2](https://github.com/facebookresearch/pytorch3d/releases/tag/v0.7.2) released.

**[Oct 23rd 2022]:**   PyTorch3D [v0.7.1](https://github.com/facebookresearch/pytorch3d/releases/tag/v0.7.1) released.

**[Aug 10th 2022]:**   PyTorch3D [v0.7.0](https://github.com/facebookresearch/pytorch3d/releases/tag/v0.7.0) released with Implicitron and MeshRasterizerOpenGL.

**[Apr 28th 2022]:**   PyTorch3D [v0.6.2](https://github.com/facebookresearch/pytorch3d/releases/tag/v0.6.2) released

**[Dec 16th 2021]:**   PyTorch3D [v0.6.1](https://github.com/facebookresearch/pytorch3d/releases/tag/v0.6.1) released

**[Oct 6th 2021]:**   PyTorch3D [v0.6.0](https://github.com/facebookresearch/pytorch3d/releases/tag/v0.6.0) released

**[Aug 5th 2021]:**   PyTorch3D [v0.5.0](https://github.com/facebookresearch/pytorch3d/releases/tag/v0.5.0) released

**[Feb 9th 2021]:** PyTorch3D [v0.4.0](https://github.com/facebookresearch/pytorch3d/releases/tag/v0.4.0) released with support for implicit functions, volume rendering and a [reimplementation of NeRF](https://github.com/facebookresearch/pytorch3d/tree/main/projects/nerf).

**[November 2nd 2020]:** PyTorch3D [v0.3.0](https://github.com/facebookresearch/pytorch3d/releases/tag/v0.3.0) released, integrating the pulsar backend.

**[Aug 28th 2020]:**   PyTorch3D [v0.2.5](https://github.com/facebookresearch/pytorch3d/releases/tag/v0.2.5) released

**[July 17th 2020]:**   PyTorch3D tech report published on ArXiv: https://arxiv.org/abs/2007.08501

**[April 24th 2020]:**   PyTorch3D [v0.2.0](https://github.com/facebookresearch/pytorch3d/releases/tag/v0.2.0) released

**[March 25th 2020]:**   [SynSin](https://arxiv.org/abs/1912.08804) codebase released using PyTorch3D: https://github.com/facebookresearch/synsin

**[March 8th 2020]:**   PyTorch3D [v0.1.1](https://github.com/facebookresearch/pytorch3d/releases/tag/v0.1.1) bug fix release

**[Jan 23rd 2020]:**   PyTorch3D [v0.1.0](https://github.com/facebookresearch/pytorch3d/releases/tag/v0.1.0) released. [Mesh R-CNN](https://arxiv.org/abs/1906.02739) codebase released: https://github.com/facebookresearch/meshrcnn
