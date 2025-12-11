#!/bin/bash
#BSUB -n 1
#BSUB -W 2
#BSUB -J submit
#BSUB -o out.%J
#BSUB -e err.%J

python submitSimulationAssay.py
