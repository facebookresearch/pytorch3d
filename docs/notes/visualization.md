---
hide_title: true
sidebar_label: Plotly Visualization
---

# Overview

PyTorch3D provides a modular differentiable renderer, but for instances where we want interactive plots or are not concerned with the differentiability of the rendering process, we provide [functions to render meshes and pointclouds in plotly](https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/vis/plotly_vis.py). These plotly figures allow you to rotate and zoom the rendered images and support plotting batched data as multiple traces in a singular plot or divided into individual subplots.


# Examples

These rendering functions accept plotly x,y, and z axis arguments as `kwargs`, allowing us to customize the plots. Here are two plots with colored axes, a [Pointclouds plot](assets/plotly_pointclouds.png), a [batched Meshes plot in subplots](assets/plotly_meshes_batch.png), and a [batched Meshes plot with multiple traces](assets/plotly_meshes_trace.png). Refer to the [render textured meshes](https://pytorch3d.org/tutorials/render_textured_meshes) and [render colored pointclouds](https://pytorch3d.org/tutorials/render_colored_points) tutorials for code examples.

# Saving plots to images

If you want to save these plotly plots, you will need to install a separate library such as [Kaleido](https://plotly.com/python/static-image-export/).

Install Kaleido
```
$ pip install Kaleido
```
Export a figure as a .png image. The image will be saved in the current working directory.
```
fig = ...
fig.write_image("image_name.png")
```
