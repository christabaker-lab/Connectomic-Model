#!/bin/bash
#BSUB -n 256
#BSUB -W 20
#BSUB -R span[hosts=1]
#BSUB -J connectomic-model-simulation
#BSUB -o stdout.%J
#BSUB -e stderr.%J
source ~/.bashrc
conda activate /usr/local/usrapps/bakerlab/apathak4/connectomic-model-hpc
python new_model.py config.json
conda deactivate