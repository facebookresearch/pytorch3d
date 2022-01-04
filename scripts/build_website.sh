#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# run this script from the project root using `./scripts/build_docs.sh`

set -e

usage() {
  echo "Usage: $0 [-b]"
  echo ""
  echo "Build PyTorch3D documentation."
  echo ""
  echo "  -b   Build static version of documentation (otherwise start server)"
  echo ""
  exit 1
}

BUILD_STATIC=false

while getopts 'hb' flag; do
  case "${flag}" in
    h)
      usage
      ;;
    b)
      BUILD_STATIC=true
      ;;
    *)
      usage
      ;;
  esac
done


echo "-----------------------------------"
echo "Building PyTorch3D Docusaurus site"
echo "-----------------------------------"
cd website
yarn
cd ..

echo "-----------------------------------"
echo "Generating tutorials"
echo "-----------------------------------"
cwd=$(pwd)
mkdir -p "website/_tutorials"
mkdir -p "website/static/files"
python scripts/parse_tutorials.py --repo_dir "${cwd}"

cd website

if [[ $BUILD_STATIC == true ]]; then
  echo "-----------------------------------"
  echo "Building static site"
  echo "-----------------------------------"
  yarn build
else
  echo "-----------------------------------"
  echo "Starting local server"
  echo "-----------------------------------"
  yarn start
fi
