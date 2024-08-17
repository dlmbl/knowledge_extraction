#!/usr/bin/env -S bash -i
echo "Creating conda environment"
mamba create -n 08_knowledge_extraction python=3.11 pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
mamba activate 08_knowledge_extraction
pip install -r requirements.txt