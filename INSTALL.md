# Installation


## Requirements

### Core library

The core library is written in PyTorch. Several components have underlying implementation in CUDA for improved performance. A subset of these components have CPU implementations in C++/PyTorch. It is advised to use PyTorch3D with GPU support in order to use all the features.

- Linux or macOS or Windows
- Python 3.6, 3.7, 3.8 or 3.9
- PyTorch 1.7.0, 1.7.1, 1.8.0, 1.8.1, 1.9.0, 1.9.1, 1.10.0, 1.10.1, 1.10.2 or 1.11.0.
- torchvision that matches the PyTorch installation. You can install them together as explained at pytorch.org to make sure of this.
- gcc & g++ ≥ 4.9
- [fvcore](https://github.com/facebookresearch/fvcore)
- [ioPath](https://github.com/facebookresearch/iopath)
- If CUDA is to be used, use a version which is supported by the corresponding pytorch version and at least version 9.2.
- If CUDA is to be used and you are building from source, the CUB library must be available. We recommend version 1.10.0.

The runtime dependencies can be installed by running:
```
conda create -n pytorch3d python=3.9
conda activate pytorch3d
conda install -c pytorch pytorch=1.9.1 torchvision cudatoolkit=10.2
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
```

For the CUB build time dependency, if you are using conda, you can continue with
```
conda install -c bottler nvidiacub
```
Otherwise download the CUB library from https://github.com/NVIDIA/cub/releases and unpack it to a folder of your choice.
Define the environment variable CUB_HOME before building and point it to the directory that contains `CMakeLists.txt` for CUB.
For example on Linux/Mac,
```
curl -LO https://github.com/NVIDIA/cub/archive/1.10.0.tar.gz
tar xzf 1.10.0.tar.gz
export CUB_HOME=$PWD/cub-1.10.0
```

### Tests/Linting and Demos

For developing on top of PyTorch3D or contributing, you will need to run the linter and tests. If you want to run any of the notebook tutorials as `docs/tutorials` or the examples in `docs/examples` you will also need matplotlib and OpenCV.
- scikit-image
- black
- usort
- flake8
- matplotlib
- tdqm
- jupyter
- imageio
- plotly
- opencv-python

These can be installed by running:
```
# Demos and examples
conda install jupyter
pip install scikit-image matplotlib imageio plotly opencv-python

# Tests/Linting
pip install black usort flake8 flake8-bugbear flake8-comprehensions
```

## Installing prebuilt binaries for PyTorch3D
After installing the above dependencies, run one of the following commands:

### 1. Install with CUDA support from Anaconda Cloud, on Linux only

```
# Anaconda Cloud
conda install pytorch3d -c pytorch3d
```

Or, to install a nightly (non-official, alpha) build:
```
# Anaconda Cloud
conda install pytorch3d -c pytorch3d-nightly
```
### 2. Install from PyPI, on Mac only.
This works with pytorch 1.9.0 only. The build is CPU only.
```
pip install pytorch3d
```

### 3. Install wheels for Linux
We have prebuilt wheels with CUDA for Linux for PyTorch 1.11.0, for each of the supported CUDA versions,
for Python 3.7, 3.8 and 3.9. This is for ease of use on Google Colab.
These are installed in a special way.
For example, to install for Python 3.8, PyTorch 1.11.0 and CUDA 10.2
```
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu102_pyt1110/download.html
```

In general, from inside IPython, or in Google Colab or a jupyter notebook, you can install with
```
import sys
import torch
pyt_version_str=torch.__version__.split("+")[0].replace(".", "")
version_str="".join([
    f"py3{sys.version_info.minor}_cu",
    torch.version.cuda.replace(".",""),
    f"_pyt{pyt_version_str}"
])
!pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/{version_str}/download.html
```

## Building / installing from source.
CUDA support will be included if CUDA is available in pytorch or if the environment variable
`FORCE_CUDA` is set to `1`.

### 1. Install from GitHub
```
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```
To install using the code of the released version instead of from the main branch, use the following instead.
```
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
```

For CUDA builds with versions earlier than CUDA 11, set `CUB_HOME` before building as described above.

**Install from Github on macOS:**
Some environment variables should be provided, like this.
```
MACOSX_DEPLOYMENT_TARGET=10.14 CC=clang CXX=clang++ pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```

### 2. Install from a local clone
```
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d && pip install -e .
```
To rebuild after installing from a local clone run, `rm -rf build/ **/*.so` then `pip install -e .`. You often need to rebuild pytorch3d after reinstalling PyTorch. For CUDA builds with versions earlier than CUDA 11, set `CUB_HOME` before building as described above.

**Install from local clone on macOS:**
```
MACOSX_DEPLOYMENT_TARGET=10.14 CC=clang CXX=clang++ pip install -e .
```

**Install from local clone on Windows:**

Depending on the version of PyTorch, changes to some PyTorch headers may be needed before compilation. These are often discussed in issues in this repository.

After any necessary patching, you can go to "x64 Native Tools Command Prompt for VS 2019" to compile and install
```
cd pytorch3d
python3 setup.py install
```
After installing, verify whether all unit tests have passed
```
cd tests
python3 -m unittest discover -p *.py
```

# FAQ

### Can I use Docker?

We don't provide a docker file but see [#113](https://github.com/facebookresearch/pytorch3d/issues/113) for a docker file shared by a user (NOTE: this has not been tested by the PyTorch3D team).
