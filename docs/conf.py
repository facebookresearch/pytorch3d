# -*- coding: utf-8 -*-
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# flake8: noqa

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

import unittest.mock as mock

from recommonmark.parser import CommonMarkParser
from recommonmark.states import DummyStateMachine
from sphinx.builders.html import StandaloneHTMLBuilder
from sphinx.ext.autodoc import between


# Monkey patch to fix recommonmark 0.4 doc reference issues.
orig_run_role = DummyStateMachine.run_role


def run_role(self, name, options=None, content=None):
    if name == "doc":
        name = "any"
    return orig_run_role(self, name, options, content)


DummyStateMachine.run_role = run_role


StandaloneHTMLBuilder.supported_image_types = [
    "image/svg+xml",
    "image/gif",
    "image/png",
    "image/jpeg",
]

# -- Path setup --------------------------------------------------------------


sys.path.insert(0, os.path.abspath("../"))
sys.path.insert(0, os.path.abspath("../pytorch3d"))
sys.path.insert(0, os.path.abspath("../../"))

DEPLOY = os.environ.get("READTHEDOCS") == "True"
needs_sphinx = "1.7"


try:
    import torch  # noqa
except ImportError:
    for m in [
        "torch",
        "torchvision",
        "torch.nn",
        "torch.autograd",
        "torch.autograd.function",
        "torch.nn.modules",
        "torch.nn.modules.utils",
        "torch.utils",
        "torch.utils.data",
        "torchvision",
        "torchvision.ops",
    ]:
        sys.modules[m] = mock.Mock(name=m)

for m in ["cv2", "scipy", "numpy", "pytorch3d._C", "np.eye", "np.zeros"]:
    sys.modules[m] = mock.Mock(name=m)

# -- Project information -----------------------------------------------------

project = "PyTorch3D"
copyright = "Meta Platforms, Inc"
author = "facebookresearch"

# The short X.Y version
version = ""

# The full version, including alpha/beta/rc tags
release = version

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

extensions = [
    "sphinx_markdown_tables",
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
]

# -- Configurations for plugins ------------
napoleon_google_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_special_with_doc = True
napoleon_numpy_docstring = False
# napoleon_use_param = False
napoleon_use_rtype = False
autodoc_inherit_docstrings = False
autodoc_member_order = "bysource"

source_parsers = {".md": CommonMarkParser}


# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
source_suffix = [".rst", ".md"]

# The master toctree document.
master_doc = "index"

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "build", "README.md"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_theme_options = {"collapse_navigation": True}


def url_resolver(url):
    if ".html" not in url:
        url = url.replace("../", "")
        return "https://github.com/facebookresearch/pytorch3d/blob/main/" + url
    else:
        if DEPLOY:
            return "http://pytorch3d.readthedocs.io/" + url
        else:
            return "/" + url


def setup(app):
    # Add symlink to root README
    if DEPLOY:
        import subprocess

        subprocess.call(["ln", "-s", "../README.md", "overview.md"])

    from recommonmark.transform import AutoStructify

    app.add_config_value(
        "recommonmark_config",
        {
            "url_resolver": url_resolver,
            "auto_toc_tree_section": "Contents",
            "enable_math": True,
            "enable_inline_math": True,
            "enable_eval_rst": True,
            "enable_auto_toc_tree": True,
        },
        True,
    )

    # Register a sphinx.ext.autodoc.between listener to ignore everything
    # between lines that contain the word IGNORE
    app.connect("autodoc-process-docstring", between("^.*IGNORE.*$", exclude=True))
    app.add_transform(AutoStructify)

    return app
