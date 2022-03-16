#!/usr/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

sudo docker run --rm  -v "$PWD/../../:/inside" pytorch/conda-cuda bash inside/packaging/linux_wheels/inside.sh
sudo docker run --rm  -v "$PWD/../../:/inside" -e SELECTED_CUDA=cu113 pytorch/conda-builder:cuda113 bash inside/packaging/linux_wheels/inside.sh
sudo docker run --rm  -v "$PWD/../../:/inside" -e SELECTED_CUDA=cu115 pytorch/conda-builder:cuda115 bash inside/packaging/linux_wheels/inside.sh
