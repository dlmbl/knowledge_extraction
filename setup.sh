#!/usr/bin/env -S bash -i
echo "Creating conda environment"
conda create -n 08_knowledge_extraction -y python=3.11 
eval "$(conda shell.bash hook)"
conda activate 08_knowledge_extraction
# Check if the environment is activated
echo "Environment activated: $(which python)"

conda install -y pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt

echo "Training classifier model"
python extras/train_classifier.py

conda deactivate
