import os
import subprocess

def make_sh_file(filename, text):

    try:
        with open(filename, 'w') as file:
            file.write(text)

        print(f"script '{filename}' created")

    except Exception as e:
        print(f"An error occured: {e}")

jobs = os.listdir("/rs1/researchers/c/cbaker5/Ayush/Connectomic-Model/run_configs")
for cfile in jobs:
    text = f"""
            #!/bin/bash

            #BSUB -n 64
            #BSUB -W 240
            #BSUB -R "rusage[mem=192GB]"
            #BSUB -R "rusage[mem=3GB/task]"
            #BSUB -R span[hosts=1]
            #BSUB -J {cfile}
            #BSUB -o /rs1/researchers/c/cbaker5/Ayush/Connectomic-Model/out_files/out.%J
            #BSUB -e /rs1/researchers/c/cbaker5/Ayush/Connectomic-Model/err_files/err.%J

            source ~/.bashrc
            conda activate /usr/local/usrapps/bakerlab/apathak4/connectomic-model-hpc
            python new_model.py run_configs/{cfile}
            conda deactivate
"""
    
    make_sh_file(f"{cfile}.sh", text)
    
    os.system(f"bsub < {cfile}.sh")

