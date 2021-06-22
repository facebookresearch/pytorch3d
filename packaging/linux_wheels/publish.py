# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import subprocess
from pathlib import Path
from typing import List


dest = "s3://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/"

output = Path("output")


def fs3cmd(args, allow_failure: bool = False) -> List[str]:
    """
    This function returns the args for subprocess to mimic the bash command
    fs3cmd available in the fairusers_aws module on the FAIR cluster.
    """
    os.environ["FAIR_CLUSTER_NAME"] = os.environ["FAIR_ENV_CLUSTER"].lower()
    cmd_args = ["/public/apps/fairusers_aws/bin/fs3cmd"] + args
    return cmd_args


def fs3_exists(path) -> bool:
    """
    Returns True if the path exists inside dest on S3.
    In fact, will also return True if there is a file which has the given
    path as a prefix, but we are careful about this.
    """
    out = subprocess.check_output(fs3cmd(["ls", path]))
    return len(out) != 0


def get_html_wrappers() -> None:
    for directory in sorted(output.iterdir()):
        output_wrapper = directory / "download.html"
        assert not output_wrapper.exists()
        dest_wrapper = dest + directory.name + "/download.html"
        if fs3_exists(dest_wrapper):
            subprocess.check_call(fs3cmd(["get", dest_wrapper, str(output_wrapper)]))


def write_html_wrappers() -> None:
    html = """
    <a href="$">$</a><br>
    """

    for directory in sorted(output.iterdir()):
        files = list(directory.glob("*.whl"))
        assert len(files) == 1, files
        [wheel] = files

        this_html = html.replace("$", wheel.name)
        output_wrapper = directory / "download.html"
        if output_wrapper.exists():
            contents = output_wrapper.read_text()
            if this_html not in contents:
                with open(output_wrapper, "a") as f:
                    f.write(this_html)
        else:
            output_wrapper.write_text(this_html)


def to_aws() -> None:
    for directory in output.iterdir():
        for file in directory.iterdir():
            print(file)
            subprocess.check_call(
                fs3cmd(["put", str(file), dest + str(file.relative_to(output))])
            )


if __name__ == "__main__":
    # Uncomment this for subsequent releases.
    # get_html_wrappers()
    write_html_wrappers()
    to_aws()
