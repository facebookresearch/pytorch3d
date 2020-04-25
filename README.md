<img src="https://raw.githubusercontent.com/facebookresearch/pytorch3d/master/.github/pytorch3dlogo.png" width="900"/>

[![CircleCI](https://circleci.com/gh/facebookresearch/pytorch3d.svg?style=svg)](https://circleci.com/gh/facebookresearch/pytorch3d)
[![Anaconda-Server Badge](https://anaconda.org/pytorch3d/pytorch3d/badges/version.svg)](https://anaconda.org/pytorch3d/pytorch3d)

# Introduction

PyTorch3D provides efficient, reusable components for 3D Computer Vision research with [PyTorch](https://pytorch.org).

Key features include:

- Data structure for storing and manipulating triangle meshes
- Efficient operations on triangle meshes (projective transformations, graph convolution, sampling, loss functions)
- A differentiable mesh renderer

PyTorch3D is designed to integrate smoothly with deep learning methods for predicting and manipulating 3D data.
For this reason, all operators in PyTorch3D:

- Are implemented using PyTorch tensors
- Can handle minibatches of hetereogenous data
- Can be differentiated
- Can utilize GPUs for acceleration

Within FAIR, PyTorch3D has been used to power research projects such as [Mesh R-CNN](https://arxiv.org/abs/1906.02739).

## Installation

For detailed instructions refer to [INSTALL.md](INSTALL.md).

## License

PyTorch3D is released under the [BSD-3-Clause License](LICENSE).

## Tutorials

Get started with PyTorch3D by trying one of the tutorial notebooks.

|<img src="https://raw.githubusercontent.com/facebookresearch/pytorch3d/master/.github/dolphin_deform.gif" width="310"/>|<img src="https://raw.githubusercontent.com/facebookresearch/pytorch3d/master/.github/bundle_adjust.gif" width="310"/>|
|:-----------------------------------------------------------------------------------------------------------:|:--------------------------------------------------:|
| [Deform a sphere mesh to dolphin](https://github.com/facebookresearch/pytorch3d/blob/master/docs/tutorials/deform_source_mesh_to_target_mesh.ipynb)| [Bundle adjustment](https://github.com/facebookresearch/pytorch3d/blob/master/docs/tutorials/bundle_adjustment.ipynb) |

| <img src="https://raw.githubusercontent.com/facebookresearch/pytorch3d/master/.github/render_textured_mesh.gif" width="310"/> | <img src="https://raw.githubusercontent.com/facebookresearch/pytorch3d/master/.github/camera_position_teapot.gif" width="310" height="310"/>
|:------------------------------------------------------------:|:--------------------------------------------------:|
| [Render textured meshes](https://github.com/facebookresearch/pytorch3d/blob/master/docs/tutorials/render_textured_meshes.ipynb)| [Camera position optimization](https://github.com/facebookresearch/pytorch3d/blob/master/docs/tutorials/camera_position_optimization_with_differentiable_rendering.ipynb)|

## Documentation

Learn more about the API by reading the PyTorch3D [documentation](https://pytorch3d.readthedocs.org/).

We also have deep dive notes on several API components:

- [Heterogeneous Batching](https://github.com/facebookresearch/pytorch3d/tree/master/docs/notes/batching.md)
- [Mesh IO](https://github.com/facebookresearch/pytorch3d/tree/master/docs/notes/meshes_io.md)
- [Differentiable Rendering](https://github.com/facebookresearch/pytorch3d/tree/master/docs/notes/renderer_getting_started.md)

## Development

We welcome new contributions to PyTorch3D and we will be actively maintaining this library! Please refer to [CONTRIBUTING.md](./.github/CONTRIBUTING.md) for full instructions on how to run the code, tests and linter, and submit your pull requests.

## Contributors

PyTorch3D is written and maintained by the Facebook AI Research Computer Vision Team.

## Citation

If you find PyTorch3D useful in your research, please cite:

```bibtex
@misc{ravi2020pytorch3d,
  author =       {Nikhila Ravi and Jeremy Reizenstein and David Novotny and Taylor Gordon
                  and Wan-Yen Lo and Justin Johnson and Georgia Gkioxari},
  title =        {PyTorch3D},
  howpublished = {\url{https://github.com/facebookresearch/pytorch3d}},
  year =         {2020}
}
```

## News

Please see below for a timeline of the codebase updates in reverse chronological order. We are sharing updates on the releases as well as research projects which are built with PyTorch3D. The changelogs for the releases are available under [`Releases`](https://github.com/facebookresearch/pytorch3d/releases),  and the builds can be installed using `conda` as per the instructions in [INSTALL.md](INSTALL.md).

**[April 24th 2020]:**   PyTorch3D v0.2 released

**[March 25th 2020]:**   [SynSin](https://arxiv.org/abs/1912.08804) codebase released using PyTorch3D: https://github.com/facebookresearch/synsin

**[March 8th 2020]:**   PyTorch3D v0.1.1 bug fix release

**[Jan 23rd 2020]:**   PyTorch3D v0.1 released. [Mesh R-CNN](https://arxiv.org/abs/1906.02739) codebase released: https://github.com/facebookresearch/meshrcnn
