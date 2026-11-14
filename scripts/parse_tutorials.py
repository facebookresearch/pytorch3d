#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import os

import nbformat
from bs4 import BeautifulSoup
from nbconvert import HTMLExporter, ScriptExporter


TEMPLATE = """const CWD = process.cwd();

const React = require('react');
const Tutorial = require(`${{CWD}}/core/Tutorial.js`);

class TutorialPage extends React.Component {{
  render() {{
      const {{config: siteConfig}} = this.props;
      const {{baseUrl}} = siteConfig;
      return <Tutorial baseUrl={{baseUrl}} tutorialID="{}"/>;
  }}
}}

module.exports = TutorialPage;

"""

JS_SCRIPTS = """
<script
  src="https://cdnjs.cloudflare.com/ajax/libs/require.js/2.1.10/require.min.js">
</script>
<script
  src="https://cdnjs.cloudflare.com/ajax/libs/jquery/2.0.3/jquery.min.js">
</script>
"""  # noqa: E501


def gen_tutorials(repo_dir: str) -> None:
    """Generate HTML tutorials for PyTorch3D Docusaurus site from Jupyter notebooks.

    Also create ipynb and py versions of tutorial in Docusaurus site for
    download.
    """
    with open(os.path.join(repo_dir, "website", "tutorials.json"), "r") as infile:
        tutorial_config = json.loads(infile.read())

    tutorial_ids = {x["id"] for v in tutorial_config.values() for x in v}

    for tid in tutorial_ids:
        print("Generating {} tutorial".format(tid))

        # convert notebook to HTML
        ipynb_in_path = os.path.join(
            repo_dir, "docs", "tutorials", "{}.ipynb".format(tid)
        )
        with open(ipynb_in_path, "r") as infile:
            nb_str = infile.read()
            nb = nbformat.reads(nb_str, nbformat.NO_CONVERT)

        # displayname is absent from notebook metadata
        nb["metadata"]["kernelspec"]["display_name"] = "python3"

        exporter = HTMLExporter()
        html, meta = exporter.from_notebook_node(nb)

        # pull out html div for notebook
        soup = BeautifulSoup(html, "html.parser")
        nb_meat = soup.find("div", {"id": "notebook-container"})
        del nb_meat.attrs["id"]
        nb_meat.attrs["class"] = ["notebook"]
        html_out = JS_SCRIPTS + str(nb_meat)

        # generate html file
        html_out_path = os.path.join(
            repo_dir, "website", "_tutorials", "{}.html".format(tid)
        )
        with open(html_out_path, "w") as html_outfile:
            html_outfile.write(html_out)

        # generate JS file
        script = TEMPLATE.format(tid)
        js_out_path = os.path.join(
            repo_dir, "website", "pages", "tutorials", "{}.js".format(tid)
        )
        with open(js_out_path, "w") as js_outfile:
            js_outfile.write(script)

        # output tutorial in both ipynb & py form
        ipynb_out_path = os.path.join(
            repo_dir, "website", "static", "files", "{}.ipynb".format(tid)
        )
        with open(ipynb_out_path, "w") as ipynb_outfile:
            ipynb_outfile.write(nb_str)
        exporter = ScriptExporter()
        script, meta = exporter.from_notebook_node(nb)
        py_out_path = os.path.join(
            repo_dir, "website", "static", "files", "{}.py".format(tid)
        )
        with open(py_out_path, "w") as py_outfile:
            py_outfile.write(script)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate JS, HTML, ipynb, and py files for tutorials."
    )
    parser.add_argument(
        "--repo_dir", metavar="path", required=True, help="PyTorch3D repo directory."
    )
    args = parser.parse_args()
    gen_tutorials(args.repo_dir)
