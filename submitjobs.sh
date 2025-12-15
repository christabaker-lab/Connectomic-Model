#!/bin/bash
#BSUB -n 1
#BSUB -W 2
#BSUB -J submit
#BSUB -o /rs1/researchers/c/cbaker5/Ayush/Connectomic-Model/out_files/out.%J
#BSUB -e /rs1/researchers/c/cbaker5/Ayush/Connectomic-Model/err_files/err.%J

python submitSimulationAssay.py
