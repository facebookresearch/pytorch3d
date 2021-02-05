#!/usr/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
sudo docker run --rm  -v "$PWD/../../:/inside" pytorch/conda-cuda bash inside/packaging/linux_wheels/inside.sh
