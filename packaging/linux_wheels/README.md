## Building Linux pip Packages

1. Make sure this directory is on a filesystem which docker can
use - e.g. not NFS. If you are using a local hard drive there is
nothing to do here.

2. You may want to `docker pull pytorch/conda-cuda:latest`.

3. Run `bash go.sh` in this directory. This takes ages
and writes packages to `inside/output`.

4. You can upload the packages to s3, along with basic html files
which enable them to be used, with `bash after.sh`.


In particular, if you are in a jupyter/colab notebook you can
then install using these wheels with the following series of
commands.

```
import sys
import torch
pyt_version_str=torch.__version__.split("+")[0].replace(".", "")
version_str="".join([
    f"py3{sys.version_info.minor}_cu",
    torch.version.cuda.replace(".",""),
    f"_pyt{pyt_version_str}"
])
!pip install iopath
!pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/{version_str}/download.html
```
