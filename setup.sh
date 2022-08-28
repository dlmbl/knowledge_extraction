#!/bin/bash
echo "Creating conda environment"
conda create -f environment.yaml

# # get the CycleGAN code and dependencies
git clone https://github.com/funkey/neuromatch_xai
mv neuromatch_xai/cycle_gan .


# Doanload checkpoints and data
wget 'https://www.dropbox.com/sh/ucpjfd3omjieu80/AAAvZynLtzvhyFx7_jwVhUK2a?dl=0&preview=data.zip' -O resources.zip
# Unzip the checkpoints
unzip -o checkpoints.zip 'checkpoints/synapses/*'
unzip -o data.zip 'data/raw/synapses/*'
# make sure the order of classes matches the pretrained model
mv data/raw/synapses/gaba data/raw/synapses/0_gaba
mv data/raw/synapses/acetylcholine data/raw/synapses/1_acetylcholine
mv data/raw/synapses/glutamate data/raw/synapses/2_glutamate
mv data/raw/synapses/serotonin data/raw/synapses/3_serotonin
mv data/raw/synapses/octopamine data/raw/synapses/4_octopamine
mv data/raw/synapses/dopamine data/raw/synapses/5_dopamine
