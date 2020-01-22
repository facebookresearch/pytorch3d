<img src=".github/pytorch3dlogo.png" width="900"/>

PyTorch3d provides efficient, reusable components for 3D Computer Vision research with [PyTorch](https://pytorch.org).

Key features include:
- Data structure for storing and manipulating triangle meshes
- Efficient operations on triangle meshes (projective transformations, graph convolution, sampling, loss functions)
- A differentiable mesh renderer

PyTorch3d is designed to integrate smoothly with deep learning methods for predicting and manipulating 3D data.
For this reason, all operators in PyTorch3d:
- Are implemented using PyTorch tensors
- Can handle minibatches of hetereogenous data
- Can be differentiated
- Can utilize GPUs for acceleration

Within FAIR, PyTorch3d has been used to power research projects such as [Mesh R-CNN](https://arxiv.org/abs/1906.02739).

## Installation

See [INSTALL.md](INSTALL.md).

## License

PyTorch3d is released under the [BSD-3-Clause License](LICENSE).

## Tutorials

Get started with PyTorch3d by trying one of the tutorial notebooks:

| <img src=".github/dolphin_deform.gif" width="310"/> | <img src=".github/bundle_adjust.gif" width="310"/> |
|:---:|:---:|
| [Deform a sphere mesh to dolphin](https://github.com/fairinternal/pytorch3d/blob/master/docs/tutorials/deform_source_mesh_to_target_mesh.ipynb)| [Bundle adjustment](https://github.com/fairinternal/pytorch3d/blob/master/docs/tutorials/bundle_adjustment.ipynb) |

| <img src=".github/render_textured_mesh.gif" width="310"/> | <img src=".github/camera_position_teapot.gif" width="310" height="310"/>
|:---:|:---:|
| [Render textured meshes](https://github.com/fairinternal/pytorch3d/blob/master/docs/tutorials/render_textured_meshes.ipynb)| [Camera position optimization](https://github.com/fairinternal/pytorch3d/blob/master/docs/tutorials/camera_position_optimization_with_differentiable_rendering.ipynb)|

## Documentation

Learn more about the API by reading the PyTorch3d [documentation](https://pytorch3d.readthedocs.org/).

## Development

We welcome new contributions to Pytorch3d and we will be actively maintaining this library! Please refer to [CONTRIBUTING.md](CONTRIBUTING.md) for full instructions on how to run the code, test and linter and submit your pull requests.

## Contributors

PyTorch3d is written and maintained by the Facebook AI Research Computer Vision Team.

## Citation

If you find PyTorch3d useful in your research, please cite:

```bibtex
@misc{ravi2020pytorch3d,
  author =       {Nikhila Ravi and Jeremy Reizenstein and David Novotny and Taylor Gordon
                  and Wan-Yen Lo and Justin Johnson and Georgia Gkioxari},
  title =        {PyTorch3D},
  howpublished = {\url{https://github.com/facebookresearch/pytorch3d}},
  year =         {2020}
}
```
