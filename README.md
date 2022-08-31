# Exercise 9: Explainable AI and Knowledge Extraction

## Setup

Before anything else, in the super-repository called `DL-MBL-2022`:
```
git pull
git submodule update --init 09_knowledge_extraction
```

Then, if you have any other exercises still running, please save your progress and shut down those kernels.
This is a GPU-hungry exercise so you're going to need all the GPU memory you can get.

Next, run the setup script. It might take a few minutes.
```
cd 09_knowledge_extraction
bash setup.sh
```
This will:
- Create a `conda` environment for this exercise
- Download and unzip data and pre-trained network
Feel free to have a look at the `setup.sh` script to see the details.


Start you environment, and begin a Jupyter Lab instance:
```
conda activate 09_knowledge_extraction
jupyter lab
```
...and continue with the instructions in the notebook.

## Overview

In this exercise we will:
1. Train a classifier to predict, from 2D EM images of synapses, which neurotransmitter is (mostly) used at that synapse
2. Use a gradient-based attribution method to try to find out what parts of the images contribute to the prediction
3. Train a CycleGAN to create counterfactual images
4. Run a discriminative attribution from counterfactuals
