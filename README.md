# ComFe: An Interpretable Head for Vision Transformers

[arxiv paper](https://arxiv.org/abs/2403.04125) - [TMLR paper](https://openreview.net/forum?id=cI4wrDYFqE)

<p float="center">
  <img src="imgs/True_72_3709_raw.png" width="150" />
  <img src="imgs/True_72_3709_class.png" width="150" /> 
  <img src="imgs/True_72_3709_class_conf.png" width="150" /> 
  <img src="imgs/True_72_3709_prototypes.png" width="150" />
</p>

## Installation

Using uv, install following [this guide](https://github.com/astral-sh/uv).

First, install the correct python environment

    uv python install 3.10.18
    uv python pin 3.10.18

Then create the virtual environment

    deactivate
    uv venv --python 3.10.18
    source .venv/bin/activate
    uv pip install -r pyproject.toml

and then to activate the environment run

    source .venv/bin/activate

## Fitting ComFe


To fit a ComFe head to the Flowers-102 dataset using frozen DINOv2 embeddings, run

	python main.py +run=base.yaml +R=comfe +model/dataset=torchvision_flowers +model/networks=dinov2_vits_14  seed=1

This should fit pretty quickly on a local GPU, and to visualize the results use

	python main.py +run=base.yaml +R=comfe-view +model/dataset=torchvision_flowers +model/networks=dinov2_vits_14

Which will save the explanations for the test images in ../output/local/comfe-view_0/tensorboard/version_0/explanations.

