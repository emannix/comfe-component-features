# ComFe: Interpretable Image Classifiers With Foundation Models

<p float="center">
  <img src="imgs/True_72_3709_raw" width="150" />
  <img src="imgs/True_72_3709_class.png" width="150" /> 
  <img src="imgs/True_72_3709_class_conf.png" width="150" /> 
  <img src="imgs/True_72_3709_prototypes.png" width="150" />
</p>

## Installation

Create a new Python 3.9 environment and install the packages in the requirements.txt file. 


## Fitting ComFe


To fit a ComFe head to the Flowers-102 dataset using frozen DINOv2 embeddings, run

	python main.py +run=base.yaml +R=comfe +model/dataset=torchvision_flowers +model/networks=dinov2_vits_14  seed=1

This should fit pretty quickly on a local GPU, and to visualize the results use

	python main.py +run=base.yaml +R=comfe-view +model/dataset=torchvision_flowers +model/networks=dinov2_vits_14

Which will save the explanations for the test images in ../output/local/comfe-view_0/tensorboard/version_0/explanations.

