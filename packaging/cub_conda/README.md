## For building conda package for NVIDIA CUB

CUB is required for building PyTorch3D so it makes sense
to provide a conda package to make its header files available.
This directory is used to do that, it is independent of the rest
of this repo.

Make sure you are in a conda environment with
anaconda-client and conda-build installed.

From this directory, build the package with the following.
```
mkdir -p ./out
conda build --no-anaconda-upload --output-folder ./out cub
```

You can then upload the package with the following.
```
retry () {
    # run a command, and try again if it fails
    $*  || (echo && sleep 8 && echo retrying && $*)
}

file=out/linux-64/nvidiacub-1.10.0-0.tar.bz2
retry anaconda --verbose -t ${TOKEN} upload -u pytorch3d --force ${file} --no-progress
```
