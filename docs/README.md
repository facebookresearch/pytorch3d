## Setup

### Install dependencies

```
pip install -U recommonmark sphinx sphinx_rtd_theme sphinx_markdown_tables
```

### Add symlink to the root README.md

We want to include the root readme as an overview. Before generating the docs create a symlink to the root readme.

```
cd docs
ln -s ../README.md  overview.md
```

In `conf.py` for deployment this is done using `subprocess.call`.

### Add a new file

Add a new `.md` or `.rst` file and add the name to the doc tree in `index.rst` e.g

```
.. toctree::
   :maxdepth: 1
   :caption: Intro Documentation

   overview
```

To autogenerate docs from docstrings in the source code, add the import path for the function e.g.

```
Chamfer Loss
--------------------

.. autoclass:: loss.chamfer.chamfer_distance
    :members:
    :undoc-members:

    .. automethod:: __init__

````

### Build

From `pytorch3d/docs` run:

```
> make html
```

The website is generated in `_build/html`.

### Common Issues

Sphinx can be fussy, and sometimes about things you weren’t expecting. For example, you might encounter something like:

WARNING: toctree contains reference to nonexisting document u'overview'
...
checking consistency...
<pytorch3d>/docs/overview.rst::
WARNING: document isn't included in any toctree

You might have indented overview in the .. toctree:: in index.rst with four spaces, when Sphinx is expecting three.


### View

Start a python simple server:

```
> python -m http.server
```

Navigate to: `http://0.0.0.0:8000/`
