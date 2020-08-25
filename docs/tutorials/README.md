# Tutorial notebooks

For current versions of the tutorials, which correspond to the latest release,
please look at this directory at the `stable` tag, namely at
https://github.com/facebookresearch/pytorch3d/tree/stable/docs/tutorials .

There are links at the project homepage for opening these directly in colab.

They install torch, torchvision and PyTorch3D from pip, which should work
with the CUDA 10.1 inside a GPU colab notebook. If you need to install
pytorch3d from source inside colab, you can use
`!pip install 'git+https://github.com/facebookresearch/pytorch3d.git@stable'`
instead.

The versions of these tutorials on the main branch may need to use the latest
pytorch3d from the main branch. You may be able to install this from source
with
`!pip install 'git+https://github.com/facebookresearch/pytorch3d.git'`.
