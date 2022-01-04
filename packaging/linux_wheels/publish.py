# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import subprocess
from pathlib import Path
from typing import List


dest = "s3://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/"

output = Path("output")


def aws_s3_cmd(args) -> List[str]:
    """
    This function returns the full args for subprocess to do a command
    with aws.
    """
    cmd_args = ["aws", "s3", "--profile", "saml"] + args
    return cmd_args


def fs3_exists(path) -> bool:
    """
    Returns True if the path exists inside dest on S3.
    In fact, will also return True if there is a file which has the given
    path as a prefix, but we are careful about this.
    """
    out = subprocess.check_output(aws_s3_cmd(["ls", path]))
    return len(out) != 0


def get_html_wrappers() -> None:
    for directory in sorted(output.iterdir()):
        output_wrapper = directory / "download.html"
        assert not output_wrapper.exists()
        dest_wrapper = dest + directory.name + "/download.html"
        if fs3_exists(dest_wrapper):
            subprocess.check_call(aws_s3_cmd(["cp", dest_wrapper, str(output_wrapper)]))


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
                aws_s3_cmd(["cp", str(file), dest + str(file.relative_to(output))])
            )


if __name__ == "__main__":
    # Uncomment this for subsequent releases.
    # get_html_wrappers()
    write_html_wrappers()
    to_aws()


# see all files with
#  aws s3 --profile saml ls --recursive s3://dl.fbaipublicfiles.com/pytorch3d/

# empty current with
#  aws s3 --profile saml rm --recursive
#                 s3://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/
