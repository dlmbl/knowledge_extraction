# Contains the steps that I used to create the environment, for memory
mamba create -n 08_knowledge_extraction python=3.11 pytorch torchvision pytorch-cuda=12.1 -c conda-forge -c pytorch -c nvidia
mamba activate 08_knowledge_extraction
pip install -r requirements.txt
mamba env export > environment.yaml
