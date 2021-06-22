#!/usr/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

sudo docker run --rm  -v "$PWD/../../:/inside" pytorch/conda-cuda bash inside/packaging/linux_wheels/inside.sh
