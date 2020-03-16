#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

usage() {
  echo "Usage: $0 [-b]"
  echo ""
  echo "Build and push updated PyTorch3D site."
  echo ""
  exit 1
}

# Current directory (needed for cleanup later)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Make temporary directory
WORK_DIR=$(mktemp -d)
cd "${WORK_DIR}" || exit

# Clone both master & gh-pages branches
git clone git@github.com:facebookresearch/pytorch3d.git pytorch3d-master
git clone --branch gh-pages git@github.com:facebookresearch/pytorch3d.git pytorch3d-gh-pages

cd pytorch3d-master/website || exit

# Build site, tagged with "latest" version; baseUrl set to /versions/latest/
yarn
yarn run build

cd .. || exit
./scripts/build_website.sh -b

cd "${WORK_DIR}" || exit
rm -rf pytorch3d-gh-pages/*
touch pytorch3d-gh-pages/CNAME
echo "pytorch3d.org" > pytorch3d-gh-pages/CNAME
mv pytorch3d-master/website/build/pytorch3d/* pytorch3d-gh-pages/

cd pytorch3d-gh-pages || exit
git add .
git commit -m 'Update latest version of site'
git push

# Clean up
cd "${SCRIPT_DIR}" || exit
rm -rf "${WORK_DIR}"
