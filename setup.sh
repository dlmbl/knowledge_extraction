#!/usr/bin/env -S bash -i
echo "Creating conda environment"
mamba env create -f environment.yaml

# get the CycleGAN code and dependencies
git clone https://github.com/funkey/neuromatch_xai
mv neuromatch_xai/cycle_gan .
rm -rf neuromatch_xai

# Download checkpoints and data
wget 'https://www.dropbox.com/s/fbloj6iitlh2kpw/resources.zip?dl=1' -O resources.zip
# Unzip the checkpoints and data
unzip -o resources.zip data.zip
unzip -o resources.zip checkpoints.zip
unzip -o checkpoints.zip 'checkpoints/synapses/*'
unzip -o data.zip 'data/raw/synapses/*'
# make sure the order of classes matches the pretrained model
mv data/raw/synapses/gaba data/raw/synapses/0_gaba
mv data/raw/synapses/acetylcholine data/raw/synapses/1_acetylcholine
mv data/raw/synapses/glutamate data/raw/synapses/2_glutamate
mv data/raw/synapses/serotonin data/raw/synapses/3_serotonin
mv data/raw/synapses/octopamine data/raw/synapses/4_octopamine
mv data/raw/synapses/dopamine data/raw/synapses/5_dopamine
