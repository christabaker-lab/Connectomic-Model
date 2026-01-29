#!/usr/bin/env python
import os
import subprocess

HPC_PROJECT_DIR = "/rs1/researchers/c/cbaker5/Ayush/Connectomic-Model"

script_content = f"""#!/bin/bash

#BSUB -n 4
#BSUB -W 120
#BSUB -R "rusage[mem=32GB]"
#BSUB -R "span[hosts=1]"
#BSUB -J postprocess_all
#BSUB -o {HPC_PROJECT_DIR}/out_files/postprocess_out.%J
#BSUB -e {HPC_PROJECT_DIR}/err_files/postprocess_err.%J

source ~/.bashrc
conda activate /usr/local/usrapps/bakerlab/apathak4/connectomic-model-conda

echo "Starting post-processing of all experiments..."
echo "========================================"

python new_model.py --postprocess --experiment Exp_0126_LR50-50_Rate200_JO100-100

echo "========================================"
echo "Post-processing complete!"

conda deactivate
"""

# Write script
script_path = "postprocess_job.sh"
with open(script_path, 'w') as f:
    f.write(script_content)

print(f"Created {script_path}")

# Submit
try:
    subprocess.run(["which", "bsub"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    cmd = f"bsub < {script_path}"
    print(f"Submitting: {cmd}")
    os.system(cmd)
except subprocess.CalledProcessError:
    print("[Info] 'bsub' not found. Script created but not submitted.")