# Exercise 9: Explainable AI and Knowledge Extraction

## Overview

In this exercise we will:
1. Use a gradient-based attribution method to try to find out what parts of an image contribute to its classification
2. Train a CycleGAN to create counterfactual images
3. Run a discriminative attribution from counterfactuals


## Setup

Before anything else, in the super-repository called `DL-MBL-2024`:
```
git pull
git submodule update --init 08_knowledge_extraction
```

Then, if you have any other exercises still running, please save your progress and shut down those kernels.
This is a GPU-hungry exercise so you're going to need all the GPU memory you can get.

Next, run the setup script. It might take a few minutes.
```
cd 08_knowledge_extraction
source setup.sh
```
This will:
- Create a `mamba` environment for this exercise
- Download and unzip data and pre-trained network
Feel free to have a look at the `setup.sh` script to see the details.


Next, begin a Jupyter Lab instance:
```
jupyter lab
```
...and continue with the instructions in the notebook.


### Acknowledgments

This notebook was written by Jan Funke and modified by Tri Nguyen and Diane Adjavon, using code from Nils Eckstein and a modified version of the [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) implementation.
