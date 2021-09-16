Neural Radiance Fields in PyTorch3D
===================================

This project implements the Neural Radiance Fields (NeRF) from [1].

<img src="https://raw.githubusercontent.com/facebookresearch/pytorch3d/main/.github/nerf_project_logo.gif" width="600" height="338"/>


Installation
------------
1) [Install PyTorch3D](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md)

2) Install other dependencies:
    - [`visdom`](https://github.com/facebookresearch/visdom)
    - [`hydra`](https://github.com/facebookresearch/hydra)
    - [`Pillow`](https://python-pillow.org/)
    - [`requests`](https://pypi.org/project/requests/)

    E.g. using `pip`:
    ```
    pip install visdom
    pip install hydra-core --upgrade
    pip install Pillow
    pip install requests
    ```

    Exporting videos further requires a working `ffmpeg`.

Training NeRF
-------------
```
python ./train_nerf.py --config-name lego
```
will train the model from [1] on the Lego dataset.

Note that the script outputs visualizations to `Visdom`. In order to enable this, make sure to start the visdom server (before launching the training) with the following command:
```
python -m visdom.server
```
Note that training on the "lego" scene takes roughly 24 hours on a single Tesla V100.

#### Training data
Note that the `train_nerf.py` script will automatically download the relevant dataset in case it is missing.

Testing NeRF
------------
```
python ./test_nerf.py --config-name lego
```
Will load a trained model from the `./checkpoints` directory and evaluate it on the test split of the corresponding dataset (Lego in the case above).

### Exporting multi-view video of the radiance field
Furthermore, the codebase supports generating videos of the neural radiance field.
The following generates a turntable video of the Lego scene:
```
python ./test_nerf.py --config-name=lego test.mode='export_video'
```
Note that this requires a working `ffmpeg` for generating the video from exported frames.

Additionally, note that generation of the video in the original resolution is quite slow. In order to speed up the process, one can decrease the resolution of the output video by setting the `data.image_size` flag:
```
python ./test_nerf.py --config-name=lego test.mode='export_video' data.image_size="[128,128]"
```
This will generate the video in a lower `128 x 128` resolution.


Training & testing on other datasets
------------------------------------
Currently we support the following datasets:
- lego `python ./train_nerf.py --config-name lego`
- fern `python ./train_nerf.py --config-name fern`
- pt3logo `python ./train_nerf.py --config-name pt3logo`

The dataset files are located in the following public S3 bucket:
https://dl.fbaipublicfiles.com/pytorch3d_nerf_data

Attribution: `lego` and `fern` are data from the original code release of [1] in https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1, which are hosted under the CC-BY license (https://creativecommons.org/licenses/by/4.0/) The S3 bucket files contains the same images while the camera matrices have been adjusted to follow the PyTorch3D convention.

#### Quantitative results
Below are the comparisons between our implementation and the official [`TensorFlow code`](https://github.com/bmild/nerf). The speed is measured on NVidia Quadro GP100.
```
+----------------+------------------+------------------+-----------------+
| Implementation |  Lego: test PSNR |  Fern: test PSNR |  training speed |
+----------------+------------------+------------------+-----------------+
| TF (official)  |             31.0 |             27.5 |  0.24 sec/it    |
| PyTorch3D      |             32.7 |             27.9 |  0.18 sec/it    |
+----------------+------------------+------------------+-----------------+
```

#### References
[1] Ben Mildenhall and Pratul P. Srinivasan and Matthew Tancik and Jonathan T. Barron and Ravi Ramamoorthi and Ren Ng, NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis, ECCV2020
