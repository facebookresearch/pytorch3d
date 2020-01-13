# Installation


## Requirements

### Core library

The core library is written in PyTorch. Several components have underlying implementation in CUDA for improved performance. A subset of these components have CPU implementations in C++/Pytorch. It is advised to use PyTorch3d with GPU support in order to use all the features.

- Linux or macOS
- Python ≥ 3.6
- PyTorch ≥ 1.3
- torchvision that matches the PyTorch installation. You can install them together at pytorch.org to make sure of this.
- gcc & g++ ≥ 4.9
- CUDA 10.0

These can be installed by running:
```
conda create -n pytorch3d python=3.7
conda activate pytorch3d
conda install -c pytorch pytorch torchvision cudatoolkit=10.0
```

### Tests/Demos/Linting

For developing on top of PyTorch3d or contributing, you will need to run the linter and tests. If you want to run any of the notebook tutorials as `docs/tutorials` you will also need matplotlib.
- [fvcore](https://github.com/facebookresearch/fvcore)
- Pillow
- Matplotlib
- black
- isort
- flake8

These can be installed by running:
```
conda install 'pillow<7'
pip install 'git+https://github.com/facebookresearch/fvcore'
pip install scikit-image black isort flake8
```

## Build/Install Pytorch3d
After installing the above dependencies, run one of the following commands:

### 1. Install from PyPi/Anaconda Cloud

```
# PyPi
pip install pytorch3d

# Anaconda Cloud
conda install pytorch3d
```

### 2. Install from GitHub
```
pip install 'git+https://github.com/facebookresearch/detectron2.git'
# (add --user if you don't have permission)
```

### 3. Install from a local clone
```
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d && pip install -e .
```
To rebuild after installing from a local clone run, `rm -rf build/ **/*.so` then `pip install -e` .. You often need to rebuild pytorch3d after reinstalling PyTorch.

**Install from local clone on macOS:**
```
MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ pip install -e .
```
