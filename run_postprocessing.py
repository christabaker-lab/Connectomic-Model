#!/usr/bin/env python3
"""
Submit parallel post-processing jobs for all experiments.
Each experiment runs post-processing independently on its own node.
"""

import os
import subprocess
from glob import glob

HPC_PROJECT_DIR = "/rs1/researchers/c/cbaker5/Ayush/Connectomic-Model"
SIM_SAVED_DIR = os.path.join(HPC_PROJECT_DIR, "simulation_results")
SCRIPTS_DIR = os.path.join(HPC_PROJECT_DIR, "postprocess_scripts")


def create_postprocess_script(exp_name):
    """Create a post-processing submission script for a single experiment."""
    
    script_content = f"""#!/bin/bash

    #BSUB -n 1
    #BSUB -W 30
    #BSUB -R "rusage[mem=32GB]"
    #BSUB -J postprocess_{exp_name}
    #BSUB -o {HPC_PROJECT_DIR}/out_files/postprocess_out.%J
    #BSUB -e {HPC_PROJECT_DIR}/err_files/postprocess_err.%J

    source ~/.bashrc
    conda activate /usr/local/usrapps/bakerlab/apathak4/connectomic-model-conda

    echo "========================================"
    echo "Post-processing: {exp_name}"
    echo "Start time: $(date)"
    echo "========================================"

    cd {HPC_PROJECT_DIR}

    python new_model.py --postprocess --experiment {exp_name}

    EXIT_CODE=$?

    echo "========================================"
    echo "End time: $(date)"
    echo "Exit code: $EXIT_CODE"
    echo "========================================"

    conda deactivate

    exit $EXIT_CODE
    """
    
    # Create scripts directory if needed
    os.makedirs(SCRIPTS_DIR, exist_ok=True)
    
    script_path = os.path.join(SCRIPTS_DIR, f"{exp_name}_postprocess.sh")
    
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    return script_path


def check_experiment_ready(exp_dir):
    """Check if experiment has batch files ready for post-processing."""
    batch_files = glob(os.path.join(exp_dir, "batch_*_statistics.h5"))
    
    if not batch_files:
        return False, "No batch files found"
    
    # Check if already post-processed
    summary_file = os.path.join(exp_dir, "summary_analysis.csv")
    per_neuron_file = os.path.join(exp_dir, "per_neuron_statistics.csv")
    
    if os.path.exists(summary_file) and os.path.exists(per_neuron_file):
        return False, f"Already processed ({len(batch_files)} batches)"
    
    return True, f"Ready ({len(batch_files)} batches)"


def main():
    """Submit post-processing jobs for all experiments."""
    
    if not os.path.exists(SIM_SAVED_DIR):
        print(f"Error: Simulation directory not found: {SIM_SAVED_DIR}")
        return
    
    # Find all experiment directories
    experiments = []
    for item in sorted(os.listdir(SIM_SAVED_DIR)):
        if item.startswith('.') or item.startswith('_'):
            continue
        
        exp_dir = os.path.join(SIM_SAVED_DIR, item)
        if not os.path.isdir(exp_dir):
            continue
        
        experiments.append(item)
    
    if not experiments:
        print(f"Error: No experiments found in {SIM_SAVED_DIR}")
        return
    
    print("="*70)
    print(f"PARALLEL POST-PROCESSING SUBMISSION")
    print("="*70)
    print(f"Found {len(experiments)} experiments\n")
    
    # Check which experiments need processing
    ready_experiments = []
    skipped_experiments = []
    
    for exp_name in experiments:
        exp_dir = os.path.join(SIM_SAVED_DIR, exp_name)
        is_ready, reason = check_experiment_ready(exp_dir)
        
        if is_ready:
            ready_experiments.append(exp_name)
            print(f"  ✓ {exp_name}: {reason}")
        else:
            skipped_experiments.append((exp_name, reason))
    
    if skipped_experiments:
        print(f"\nSkipped {len(skipped_experiments)} experiments:")
        for exp_name, reason in skipped_experiments:
            print(f"  ✗ {exp_name}: {reason}")
    
    if not ready_experiments:
        print("\nNo experiments need post-processing!")
        return
    
    print(f"\n{'='*70}")
    print(f"SUBMITTING {len(ready_experiments)} POST-PROCESSING JOBS")
    print(f"{'='*70}\n")
    
    # Create and submit jobs
    submitted = 0
    failed = 0
    
    for exp_name in ready_experiments:
        # Create script
        try:
            script_path = create_postprocess_script(exp_name)
            print(f"  Created: {os.path.basename(script_path)}")
        except Exception as e:
            print(f"  ✗ Failed to create script for {exp_name}: {e}")
            failed += 1
            continue
        
        # Submit job
        try:
            cmd = f"bsub < {script_path}"
            print(f"Submitting: {cmd}")
            os.system(cmd)
            submitted+=1
                
        except subprocess.CalledProcessError as e:
            print(f"  ✗ Failed to submit {exp_name}: {e}")
            failed += 1
        except FileNotFoundError:
            print(f"  [Info] 'bsub' not found - scripts created but not submitted")
            print(f"         Submit manually: bsub {script_path}")
            break
    
    print(f"\n{'='*70}")
    print(f"SUBMISSION SUMMARY")
    print(f"{'='*70}")
    print(f"  Jobs submitted: {submitted}")
    print(f"  Jobs failed: {failed}")
    print(f"  Jobs skipped: {len(skipped_experiments)}")
    print(f"{'='*70}\n")
    
    
    print(f"\nScripts saved in: {SCRIPTS_DIR}/")


if __name__ == "__main__":
    main()