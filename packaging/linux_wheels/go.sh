#!/usr/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Some directory to persist downloaded conda packages
conda_cache=/raid/$USER/building_conda_cache

mkdir -p "$conda_cache"

sudo docker run --rm -v "$conda_cache:/conda_cache" -v "$PWD/../../:/inside" -e SELECTED_CUDA=cu113 pytorch/conda-builder:cuda113 bash inside/packaging/linux_wheels/inside.sh
sudo docker run --rm -v "$conda_cache:/conda_cache" -v "$PWD/../../:/inside" -e SELECTED_CUDA=cu115 pytorch/conda-builder:cuda115 bash inside/packaging/linux_wheels/inside.sh
sudo docker run --rm -v "$conda_cache:/conda_cache" -v "$PWD/../../:/inside" -e SELECTED_CUDA=cu116 pytorch/conda-builder:cuda116 bash inside/packaging/linux_wheels/inside.sh
sudo docker run --rm -v "$conda_cache:/conda_cache" -v "$PWD/../../:/inside" -e SELECTED_CUDA=cu117 pytorch/conda-builder:cuda117 bash inside/packaging/linux_wheels/inside.sh
sudo docker run --rm -v "$conda_cache:/conda_cache" -v "$PWD/../../:/inside" -e SELECTED_CUDA=cu118 pytorch/conda-builder:cuda118 bash inside/packaging/linux_wheels/inside.sh
