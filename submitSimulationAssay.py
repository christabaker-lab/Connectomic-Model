import os
import subprocess
import math

# --- Configuration ---
TOTAL_TRIALS = 10000
TRIALS_PER_BATCH = 128
# ---------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RUN_CONFIGS_DIR = os.path.join(BASE_DIR, "run_configs")
HPC_PROJECT_DIR = "/rs1/researchers/c/cbaker5/Ayush/Connectomic-Model"

def make_sh_file(filename, text):
    filepath = os.path.join(RUN_CONFIGS_DIR, filename)
    try:
        with open(filepath, 'w') as file:
            file.write(text)
        print(f"script '{filepath}' created")
    except Exception as e:
        print(f"An error occurred creating {filename}: {e}")

if not os.path.exists(RUN_CONFIGS_DIR):
    print(f"Error: run_configs directory not found at {RUN_CONFIGS_DIR}")
    exit(1)

jobs = os.listdir(RUN_CONFIGS_DIR)
num_batches = math.ceil(TOTAL_TRIALS / TRIALS_PER_BATCH)
print(f"Plan: Running {TOTAL_TRIALS} trials across {num_batches} jobs (array) for each config.")

for cfile in jobs:
    if cfile.startswith(".") or cfile.startswith("_") or not cfile.endswith(".json"):
        continue

    text = f"""#!/bin/bash

            #BSUB -n 32
            #BSUB -W 20
            #BSUB -R "rusage[mem=1GB/task]"
            #BSUB -R "span[hosts=1]"
            #BSUB -J {cfile}[1-{num_batches}]
            #BSUB -o {HPC_PROJECT_DIR}/out_files/out.%J.%I
            #BSUB -e {HPC_PROJECT_DIR}/err_files/err.%J.%I

            source ~/.bashrc
            conda activate /usr/local/usrapps/bakerlab/apathak4/connectomic-model-conda

            python new_model.py run_configs/{cfile} --batch_id $LSB_JOBINDEX --trials {TRIALS_PER_BATCH}

            conda deactivate
    """
    
    script_name = f"{cfile[:-5]}.sh"
    make_sh_file(script_name, text)
    
    cmd = f"bsub < {HPC_PROJECT_DIR}/run_configs/{script_name}"
    
    try:
        subprocess.run(["which", "bsub"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"Submitting: {cmd}")
        os.system(cmd)
    except subprocess.CalledProcessError:
        print(f"[Info] 'bsub' command not found. Created script '{script_name}' but did not submit.")