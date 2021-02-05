# Tutorial notebooks

For current versions of the tutorials, which correspond to the latest release,
please look at this directory at the `stable` tag, namely at
https://github.com/facebookresearch/pytorch3d/tree/stable/docs/tutorials .

There are links at the project homepage for opening these directly in colab.

They install PyTorch3D from pip, which should work inside a GPU colab notebook.
If you need to install PyTorch3D from source inside colab, you can use
```
import os
!curl -LO https://github.com/NVIDIA/cub/archive/1.10.0.tar.gz
!tar xzf 1.10.0.tar.gz
os.environ["CUB_HOME"] = os.getcwd() + "/cub-1.10.0"
!pip install 'git+https://github.com/facebookresearch/pytorch3d.git@stable'`
```
instead.

The versions of these tutorials on the main branch may need to use the latest
PyTorch3D from the main branch. You may be able to install this from source
with the same commands as above, but replacing the last line with
`!pip install 'git+https://github.com/facebookresearch/pytorch3d.git'`.
