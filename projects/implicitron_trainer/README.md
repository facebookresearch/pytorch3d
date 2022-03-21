# Introduction

Implicitron is a PyTorch3D-based framework for new-view synthesis via modeling the neural-network based representations.

# License

Implicitron is distributed as part of PyTorch3D under the [BSD license](https://github.com/facebookresearch/pytorch3d/blob/main/LICENSE).
It includes code from [SRN](http://github.com/vsitzmann/scene-representation-networks) and [IDR](http://github.com/lioryariv/idr) repos.
See [LICENSE-3RD-PARTY](https://github.com/facebookresearch/pytorch3d/blob/main/LICENSE-3RD-PARTY) for their licenses.


# Installation

There are three ways to set up Implicitron, depending on the flexibility level required.
If you only want to train or evaluate models as they are implemented changing only the parameters, you can just install the package.
Implicitron also provides a flexible API that supports user-defined plug-ins;
if you want to re-implement some of the components without changing the high-level pipeline, you need to create a custom launcher script.
The most flexible option, though, is cloning PyTorch3D repo and building it from sources, which allows changing the code in arbitrary ways.
Below, we descibe all three options in more details.


## [Option 1] Running an executable from the package

This option allows you to use the code as is without changing the implementations.
Only configuration can be changed (see [Configuration system](#configuration-system)).

For this setup, install the dependencies and PyTorch3D from conda following [the guide](https://github.com/facebookresearch/pytorch3d/blob/master/INSTALL.md#1-install-with-cuda-support-from-anaconda-cloud-on-linux-only). Then, install implicitron-specific dependencies:

```shell
pip install "hydra-core>=1.1" visdom lpips matplotlib
```

Runner executable is available as `pytorch3d_implicitron_runner` shell command.
See [Running](#running) section below for examples of training and evaluation commands.

## [Option 2] Supporting custom implementations

To plug in custom implementations, for example, of renderer or implicit-function protocols, you need to create your own runner script and import the plug-in implementations there.
First, install PyTorch3D and Implicitron dependencies as described in the previous section.
Then, implement the custom script; copying `pytorch3d/projects/implicitron_trainer/experiment.py` is a good place to start.
See [Custom plugins](#custom-plugins) for more information on how to import implementations and enable them in the configs.


## [Option 3] Cloning PyTorch3D repo

This is the most flexible way to set up Implicitron as it allows changing the code directly.
It allows modifying the high-level rendering pipeline or implementing yet-unsupported loss functions.
Please follow the instructions to [install PyTorch3D from a local clone](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md#2-install-from-a-local-clone).
Then, install Implicitron-specific dependencies:

```shell
pip install "hydra-core>=1.1" visdom lpips matplotlib
```

You are still encouraged to implement custom plugins as above where possible as it makes reusing the code easier.
The executable is located in `pytorch3d/projects/implicitron_trainer`.


# Running

This section assumes that you use the executable provided by the installed package.
If you have a custom `experiment.py` script (as in the Option 2 above), replace the executable with the path to your script.

## Training

To run training, pass a yaml config file, followed by a list of overridden arguments.
For example, to train NeRF on the first skateboard sequence from CO3D dataset, you can run:
```shell
pytorch3d_implicitron_runner --config-path ./configs/ --config-name repro_singleseq_nerf dataset_args.dataset_root=<DATASET_ROOT> dataset_args.category='skateboard' dataset_args.test_restrict_sequence_id=0 test_when_finished=True exp_dir=<CHECKPOINT_DIR>
```

Here, `--config-path` points to the config path relative to `pytorch3d_implicitron_runner` location;
`--config-name` picks the config (in this case, `repro_singleseq_nerf.yaml`);
`test_when_finished` will launch evaluation script once training is finished.
Replace `<DATASET_ROOT>` with the location where the dataset in Implicitron format is stored
and `<CHECKPOINT_DIR>` with a directory where checkpoints will be dumped during training.
Other configuration parameters can be overridden in the same way.
See [Configuration system](#configuration-system) section for more information on this.


## Evaluation

To run evaluation on the latest checkpoint after (or during) training, simply add `eval_only=True` to your training command.

E.g. for executing the evaluation on the NeRF skateboard sequence, you can run:
```shell
pytorch3d_implicitron_runner --config-path ./configs/ --config-name repro_singleseq_nerf dataset_args.dataset_root=<CO3D_DATASET_ROOT> dataset_args.category='skateboard' dataset_args.test_restrict_sequence_id=0 exp_dir=<CHECKPOINT_DIR> eval_only=True
```
Evaluation prints the metrics to `stdout` and dumps them to a json file in `exp_dir`.

## Visualisation

The script produces a video of renders by a trained model assuming a pre-defined camera trajectory.
In order for it to work, `ffmpeg` needs to be installed:

```shell
conda install ffmpeg
```

Here is an example of calling the script:
```shell
projects/implicitron_trainer/visualize_reconstruction.py exp_dir=<CHECKPOINT_DIR> visdom_show_preds=True n_eval_cameras=40 render_size="[64,64]" video_size="[256,256]"
```

The argument `n_eval_cameras` sets the number of renderring viewpoints sampled on a trajectory, which defaults to a circular fly-around;
`render_size` sets the size of a render passed to the model, which can be resized to `video_size` before writing.

Rendered videos of images, masks, and depth maps will be saved to `<CHECKPOINT_DIR>/vis`.


# Configuration system

We use hydra and OmegaConf to parse the configs.
The config schema and default values are defined by the dataclasses implementing the modules.
More specifically, if a class derives from `Configurable`, its fields can be set in config yaml files or overridden in CLI.
For example, `GenericModel` has a field `render_image_width` with the default value 400.
If it is specified in the yaml config file or in CLI command, the new value will be used.

Configurables can form hierarchies.
For example, `GenericModel` has a field `raysampler: RaySampler`, which is also Configurable.
In the config, inner parameters can be propagated using `_args` postfix, e.g. to change `raysampler.n_pts_per_ray_training` (the number of sampled points per ray), the node `raysampler_args.n_pts_per_ray_training` should be specified.

The root of the hierarchy is defined by `ExperimentConfig` dataclass.
It has top-level fields like `eval_only` which was used above for running evaluation by adding a CLI override.
Additionally, it has non-leaf nodes like `generic_model_args`, which dispatches the config parameters to `GenericModel`. Thus, changing the model parameters may be achieved in two ways: either by editing the config file, e.g.
```yaml
generic_model_args:
    render_image_width: 800
    raysampler_args:
        n_pts_per_ray_training: 128
```

or, equivalently, by adding the following to `pytorch3d_implicitron_runner` arguments:

```shell
generic_model_args.render_image_width=800 generic_model_args.raysampler_args.n_pts_per_ray_training=128
```

See the documentation in `pytorch3d/implicitron/tools/config.py` for more details.

## Replaceable implementations

Sometimes changing the model parameters does not provide enough flexibility, and you want to provide a new implementation for a building block.
The configuration system also supports it!
Abstract classes like `BaseRenderer` derive from `ReplaceableBase` instead of `Configurable`.
This means that other Configurables can refer to them using the base type, while the specific implementation is chosen in the config using `_class_type`-postfixed node.
In that case, `_args` node name has to include the implementation type.
More specifically, to change renderer settings, the config will look like this:
```yaml
generic_model_args:
    renderer_class_type: LSTMRenderer
    renderer_LSTMRenderer_args:
        num_raymarch_steps: 10
        hidden_size: 16
```

See the documentation in `pytorch3d/implicitron/tools/config.py` for more details on the configuration system.

## Custom plugins

If you have an idea for another implementation of a replaceable component, it can be plugged in without changing the core code.
For that, you need to set up Implicitron through option 2 or 3 above.
Let's say you want to implement a renderer that accumulates opacities similar to an X-ray machine.
First, create a module `x_ray_renderer.py` with a class deriving from `BaseRenderer`:

```python
from pytorch3d.implicitron.tools.config import registry

@registry.register
class XRayRenderer(BaseRenderer, torch.nn.Module):
    n_pts_per_ray: int = 64

    # if there are other base classes, make sure to call `super().__init__()` explicitly
    def __post_init__(self):
        super().__init__()
        # custom initialization

    def forward(
        self,
        ray_bundle,
        implicit_functions=[],
        evaluation_mode: EvaluationMode = EvaluationMode.EVALUATION,
        **kwargs,
    ) -> RendererOutput:
        ...
```

Please note `@registry.register` decorator that registers the plug-in as an implementation of `Renderer`.
IMPORTANT: In order for it to run, the class (or its enclosing module) has to be imported in your launch script. Additionally, this has to be done before parsing the root configuration class `ExperimentConfig`.
Simply add `import .x_ray_renderer` in the beginning of `experiment.py`.

After that, you should be able to change the config with:
```yaml
generic_model_args:
    renderer_class_type: XRayRenderer
    renderer_XRayRenderer_args:
        n_pts_per_ray: 128
```

to replace the implementation and potentially override the parameters.

# Code and config structure

As per above, the config structure is parsed automatically from the module hierarchy.
In particular, model parameters are contained in `generic_model_args` node, and dataset parameters in `dataset_args` node.

Here is the class structure (single-line edges show aggregation, while double lines show available implementations):
```
generic_model_args: GenericModel
└-- sequence_autodecoder_args: Autodecoder
└-- raysampler_args: RaySampler
└-- renderer_*_args: BaseRenderer
    ╘== MultiPassEmissionAbsorptionRenderer
    ╘== LSTMRenderer
    ╘== SignedDistanceFunctionRenderer
        └-- ray_tracer_args: RayTracing
        └-- ray_normal_coloring_network_args: RayNormalColoringNetwork
└-- implicit_function_*_args: ImplicitFunctionBase
    ╘== NeuralRadianceFieldImplicitFunction
    ╘== SRNImplicitFunction
        └-- raymarch_function_args: SRNRaymarchFunction
        └-- pixel_generator_args: SRNPixelGenerator
    ╘== SRNHyperNetImplicitFunction
        └-- hypernet_args: SRNRaymarchHyperNet
        └-- pixel_generator_args: SRNPixelGenerator
    ╘== IdrFeatureField
└-- image_feature_extractor_args: ResNetFeatureExtractor
└-- view_sampler_args: ViewSampler
└-- feature_aggregator_*_args: FeatureAggregatorBase
    ╘== IdentityFeatureAggregator
    ╘== AngleWeightedIdentityFeatureAggregator
    ╘== AngleWeightedReductionFeatureAggregator
    ╘== ReductionFeatureAggregator
solver_args: init_optimizer
dataset_args: dataset_zoo
dataloader_args: dataloader_zoo
```

Please look at the annotations of the respective classes or functions for the lists of hyperparameters.

# Reproducing CO3D experiments

Common Objects in 3D (CO3D) is a large-scale dataset of videos of rigid objects grouped into 50 common categories.
Implicitron provides implementations and config files to reproduce the results from [the paper](https://arxiv.org/abs/2109.00512).
Please follow [the link](https://github.com/facebookresearch/co3d#automatic-batch-download) for the instructions to download the dataset.
In training and evaluation scripts, use the download location as `<DATASET_ROOT>`.
It is also possible to define environment variable `CO3D_DATASET_ROOT` instead of specifying it.
To reproduce the experiments from the paper, use the following configs. For single-sequence experiments:

| Method          |   config file                       |
|-----------------|-------------------------------------|
| NeRF            | repro_singleseq_nerf.yaml           |
| NeRF + WCE      | repro_singleseq_nerf_wce.yaml       |
| NerFormer       | repro_singleseq_nerformer.yaml      |
| IDR             | repro_singleseq_idr.yaml            |
| SRN             | repro_singleseq_srn_noharm.yaml     |
| SRN + γ         | repro_singleseq_srn.yaml            |
| SRN + WCE       | repro_singleseq_srn_wce_noharm.yaml |
| SRN + WCE + γ   | repro_singleseq_srn_wce_noharm.yaml |

For multi-sequence experiments (without generalisation to new sequences):

| Method          |   config file                              |
|-----------------|--------------------------------------------|
| NeRF + AD       | repro_multiseq_nerf_ad.yaml                |
| SRN + AD        | repro_multiseq_srn_ad_hypernet_noharm.yaml |
| SRN + γ + AD    | repro_multiseq_srn_ad_hypernet.yaml        |

For multi-sequence experiments (with generalisation to new sequences):

| Method          |   config file                        |
|-----------------|--------------------------------------|
| NeRF + WCE      | repro_multiseq_nerf_wce.yaml         |
| NerFormer       | repro_multiseq_nerformer.yaml        |
| SRN + WCE       | repro_multiseq_srn_wce_noharm.yaml   |
| SRN + WCE + γ   | repro_multiseq_srn_wce.yaml          |
