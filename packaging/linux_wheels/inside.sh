#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

conda init bash
# shellcheck source=/dev/null
source ~/.bashrc

cd /inside
VERSION=$(python -c "exec(open('pytorch3d/__init__.py').read()); print(__version__)")

export BUILD_VERSION=$VERSION
export FORCE_CUDA=1

wget --no-verbose https://github.com/NVIDIA/cub/archive/1.10.0.tar.gz
tar xzf 1.10.0.tar.gz
CUB_HOME=$(realpath ./cub-1.10.0)
export CUB_HOME
echo "CUB_HOME is now $CUB_HOME"

# As a rule, we want to build for any combination of dependencies which is supported by
# PyTorch3D and not older than the current Google Colab set up.

PYTHON_VERSIONS="3.7 3.8 3.9 3.10"
# the keys are pytorch versions
declare -A CONDA_CUDA_VERSIONS=(
    ["1.10.1"]="cu111 cu113"
    ["1.10.2"]="cu111 cu113"
    ["1.10.0"]="cu111 cu113"
    ["1.11.0"]="cu111 cu113 cu115"
)



for python_version in $PYTHON_VERSIONS
do
    for pytorch_version in "${!CONDA_CUDA_VERSIONS[@]}"
    do
        if [[ "3.7 3.8" != *$python_version* ]] && [[ "1.7.0" == *$pytorch_version* ]]
        then
            #python 3.9 and later not supported by pytorch 1.7.0 and before
            continue
        fi
        if [[ "3.7 3.8 3.9" != *$python_version* ]] && [[ "1.7.0 1.7.1 1.8.0 1.8.1 1.9.0 1.9.1 1.10.0 1.10.1 1.10.2" == *$pytorch_version* ]]
        then
            #python 3.10 and later not supported by pytorch 1.10.2 and before
            continue
        fi

        extra_channel="-c conda-forge"
        if [[ "1.11.0" == "$pytorch_version" ]]
        then
            extra_channel=""
        fi

        for cu_version in ${CONDA_CUDA_VERSIONS[$pytorch_version]}
        do
            if [[ "cu113 cu115" == *$cu_version* ]]
            #       ^^^ CUDA versions listed here have to be built
            # in their own containers.
            then
            if [[ $SELECTED_CUDA != "$cu_version" ]]
                then
                    continue
                fi
            elif [[ $SELECTED_CUDA != "" ]]
            then
                continue
            fi

            case "$cu_version" in
                cu115)
                    export CUDA_HOME=/usr/local/cuda-11.5/
                    export CUDA_TAG=11.5
                    export NVCC_FLAGS="-gencode=arch=compute_35,code=sm_35 -gencode=arch=compute_50,code=sm_50 -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_86,code=sm_86 -gencode=arch=compute_50,code=compute_50"
                ;;
                cu113)
                    export CUDA_HOME=/usr/local/cuda-11.3/
                    export CUDA_TAG=11.3
                    export NVCC_FLAGS="-gencode=arch=compute_35,code=sm_35 -gencode=arch=compute_50,code=sm_50 -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_86,code=sm_86 -gencode=arch=compute_50,code=compute_50"
                ;;
                cu112)
                    export CUDA_HOME=/usr/local/cuda-11.2/
                    export CUDA_TAG=11.2
                    export NVCC_FLAGS="-gencode=arch=compute_35,code=sm_35 -gencode=arch=compute_50,code=sm_50 -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_86,code=sm_86 -gencode=arch=compute_50,code=compute_50"
                ;;
                cu111)
                    export CUDA_HOME=/usr/local/cuda-11.1/
                    export CUDA_TAG=11.1
                    export NVCC_FLAGS="-gencode=arch=compute_35,code=sm_35 -gencode=arch=compute_50,code=sm_50 -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_86,code=sm_86 -gencode=arch=compute_50,code=compute_50"
                ;;
                cu110)
                    export CUDA_HOME=/usr/local/cuda-11.0/
                    export CUDA_TAG=11.0
                    export NVCC_FLAGS="-gencode=arch=compute_35,code=sm_35 -gencode=arch=compute_50,code=sm_50 -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_50,code=compute_50"
                ;;
                cu102)
                    export CUDA_HOME=/usr/local/cuda-10.2/
                    export CUDA_TAG=10.2
                    export NVCC_FLAGS="-gencode=arch=compute_35,code=sm_35 -gencode=arch=compute_50,code=sm_50 -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_50,code=compute_50"
                ;;
                cu101)
                    export CUDA_HOME=/usr/local/cuda-10.1/
                    export CUDA_TAG=10.1
                    export NVCC_FLAGS="-gencode=arch=compute_35,code=sm_35 -gencode=arch=compute_50,code=sm_50 -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_50,code=compute_50"
                ;;
                *)
                    echo "Unrecognized cu_version=$cu_version"
                    exit 1
                ;;
            esac
            tag=py"${python_version//./}"_"${cu_version}"_pyt"${pytorch_version//./}"

            outdir="/inside/packaging/linux_wheels/output/$tag"
            if [[ -d "$outdir" ]]
            then
                continue
            fi

            conda create -y -n "$tag" "python=$python_version"
            conda activate "$tag"
            conda install -y -c pytorch $extra_channel "pytorch=$pytorch_version" "cudatoolkit=$CUDA_TAG" torchvision
            pip install fvcore iopath
            echo "python version" "$python_version" "pytorch version" "$pytorch_version" "cuda version" "$cu_version" "tag" "$tag"

            rm -rf dist

            python setup.py clean
            python setup.py bdist_wheel

            rm -rf "$outdir"
            mkdir -p "$outdir"
            cp dist/*whl "$outdir"

            conda deactivate
        done
    done
done
echo "DONE"
