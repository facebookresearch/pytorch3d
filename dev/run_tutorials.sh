#!/usr/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This script is for running some of the tutorials using the nightly build in
# an isolated environment. It is designed to be run in docker.

# If you run this script in this directory with
#   sudo docker run --runtime=nvidia -it --rm -v $PWD/../docs/tutorials:/notebooks -v $PWD:/loc pytorch/conda-cuda bash /loc/run_tutorials.sh | tee log.txt
# it should execute some tutorials with the nightly build and resave them, and
# save a log in the current directory.

# We use nbconvert. runipy would be an alternative but it currently doesn't
# work well with plotly.

set -e

conda init bash
# shellcheck source=/dev/null
source ~/.bashrc
conda create -y -n myenv python=3.8 matplotlib ipython ipywidgets nbconvert
conda activate myenv
conda install -y -c iopath iopath
conda install -y -c pytorch pytorch=1.6.0 cudatoolkit=10.1 torchvision
conda install -y -c pytorch3d-nightly pytorch3d
pip install plotly scikit-image

for notebook in /notebooks/*.ipynb
do
    name=$(basename "$notebook")

    if [[ "$name" == "dataloaders_ShapeNetCore_R2N2.ipynb" ]]
    then
        #skip as data not easily available
        continue
    fi
    if [[ "$name" == "render_densepose.ipynb" ]]
    then
        #skip as data not easily available
        continue
    fi

    #comment the lines which install torch, torchvision and pytorch3d
    sed -Ei '/(torchvision)|(pytorch3d)/ s/!pip/!#pip/' "$notebook"
    #Don't let tqdm use widgets
    sed -i 's/from tqdm.notebook import tqdm/from tqdm import tqdm/' "$notebook"

    echo
    echo "###   ###   ###"
    echo "starting $name"
    time jupyter nbconvert --to notebook --inplace --ExecutePreprocessor.kernel_name=python3 --execute "$notebook" || true
    echo "ending $name"
done
