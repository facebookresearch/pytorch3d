#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


# Instructions, assuming you are on a fresh pytorch3d checkout on a local
# drive.

# (1) Have a separate checkout of pytorch3d at the head of the gh-pages branch
# on a local drive. Set the variable GHP to its full path.
# Any uncommitted changes there will be obliterated.
# For example
#   GHP=/path/to/pytorch3d-gh-pages
#   git clone -b gh-pages https://github.com/facebookresearch/pytorch3d $GHP

# (2) Run this script in this directory with
#   sudo docker run -it --rm -v $PWD/..:/loc -v $GHP:/ghp continuumio/miniconda3 bash --login /loc/scripts/publish_website.sh

# (3) Choose a commit message, commit and push:
#   cd $GHP && git add .
#   git commit -m 'Update latest version of site'
#   git push

set -e

conda create -y -n myenv python=3.7 nodejs

# Note: Using bash --login together with the continuumio/miniconda3 image
# is what lets conda activate work so smoothly.

conda activate myenv
pip install nbformat==4.4.0 nbconvert==5.3.1 ipywidgets==7.5.1 tornado==4.2 bs4 notebook==5.7.12 'mistune<2'
npm install --global yarn

cd /loc
bash scripts/build_website.sh -b

rm -rf /ghp/*
echo "pytorch3d.org" > /ghp/CNAME
mv /loc/website/build/pytorch3d/* /ghp/
