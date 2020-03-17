#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# run this script from the project root using `./scripts/build_docs.sh`

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
cd website || exit
yarn
cd ..

echo "-----------------------------------"
echo "Generating tutorials"
echo "-----------------------------------"
cwd=$(pwd)
mkdir -p "website/_tutorials"
mkdir -p "website/static/files"
python scripts/parse_tutorials.py --repo_dir "${cwd}"

cd website || exit

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
