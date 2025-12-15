import os
import subprocess

def make_sh_file(filename, text):
    folder = f"/rs1/researchers/c/cbaker5/Ayush/Connectomic-Model/run_configs"
    filename = os.path.join(folder, filename)
    try:
        with open(filename, 'w') as file:
            file.write(text)

        print(f"script '{filename}' created")

    except Exception as e:
        print(f"An error occured: {e}")

jobs = os.listdir("/rs1/researchers/c/cbaker5/Ayush/Connectomic-Model/run_configs")
for cfile in jobs:

    if cfile.startswith(".") or not cfile.endswith(".json"):
        continue

    text = f"""
    #!/bin/bash

    #BSUB -n 8
    #BSUB -W 240

    #BSUB -R "rusage[mem=4GB/task]"
    #BSUB -R span[hosts=1]  
    #BSUB -J {cfile}
    #BSUB -o /rs1/researchers/c/cbaker5/Ayush/Connectomic-Model/out_files/out.%J
    #BSUB -e /rs1/researchers/c/cbaker5/Ayush/Connectomic-Model/err_files/err.%J

    source ~/.bashrc
    conda activate /usr/local/usrapps/bakerlab/apathak4/connectomic-model-conda
    python new_model.py run_configs/{cfile}
    conda deactivate
"""
    
    make_sh_file(f"{cfile[:-5]}.sh", text)
    
    os.system(f"bsub < /rs1/researchers/c/cbaker5/Ayush/Connectomic-Model/run_configs/{cfile[:-5]}.sh")

    #BSUB -R "rusage[mem=96GB]"