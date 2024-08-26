#!/usr/bin/env -S bash -i
echo "Creating conda environment"
conda create -n 08_knowledge_extraction -y python=3.11 
eval "$(conda shell.bash hook)"
conda activate 08_knowledge_extraction
# Check if the environment is activated
if [[ "$CONDA_DEFAULT_ENV" == "08_knowledge_extraction" ]]; then
    echo "Environment activated successfully for package installation"
else
    echo "Failed to activate environment for package installation. Dependencies not installed!"
    exit
fi
echo "Training classifier model"
conda install -y pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
python extras/train_classifier.py

conda deactivate
