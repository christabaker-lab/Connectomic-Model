import os
import json
import itertools
import sys
from datetime import datetime
from copy import deepcopy

def set_nested_value(d, key, value):
    """Sets a value in a nested dictionary using a dot-separated key."""
    keys = key.split('.')
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value

def create_lsf_script(job_name, n_cores, mem_gb, config_path_abs, project_dir_abs):
    """Generates the content for an LSF submission script using your template."""
    output_log_path = os.path.join(project_dir_abs, "lsf_logs", f"{job_name}_%J.out")
    hpc_runner_path = os.path.join(project_dir_abs, "new_model.py") # Your script's name
    
    # !!! IMPORTANT: EDIT the conda path below to match your HPC environment !!!
    conda_path = "/usr/local/usrapps/bakerlab/apathak4/connectomic-model-hpc"
    
    lsf_script_content = f"""#!/bin/bash
#BSUB -n {n_cores}
#BSUB -W 180
#BSUB -R "rusage[mem={mem_gb}GB]"
#BSUB -J {job_name}
#BSUB -o {output_log_path}
#BSUB -e {output_log_path}

# --- Setup Environment ---
source ~/.bashrc
conda activate {conda_path}

# --- Run Simulation ---
python {hpc_runner_path} {config_path_abs}

# --- Deactivate Environment ---
conda deactivate
"""
    return lsf_script_content

def main(sweep_config_path):
    """Generates all config and LSF files for a parameter sweep."""
    project_dir = os.path.dirname(os.path.abspath(__file__))
    
    with open(sweep_config_path, 'r') as f:
        sweep_config = json.load(f)

    base_config = sweep_config['base_config']
    sweep_params = sweep_config['sweep_parameters']
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir_name = f"{sweep_config['sweep_name']}_{timestamp}"
    base_output_dir = base_config['output_config']['base_output_directory']
    sweep_output_path = os.path.join(base_output_dir, sweep_dir_name)
    
    configs_path = os.path.join(sweep_output_path, "configs")
    submission_scripts_path = os.path.join(sweep_output_path, "submission_scripts")
    os.makedirs(os.path.join(sweep_output_path, "lsf_logs"), exist_ok=True)
    os.makedirs(configs_path, exist_ok=True)
    os.makedirs(submission_scripts_path, exist_ok=True)

    print(f"Sweep files will be generated in: '{sweep_output_path}'", file=sys.stderr)
    
    param_names = sorted(sweep_params.keys())
    param_value_lists = [sweep_params[k] for k in param_names]
    param_combinations = list(itertools.product(*param_value_lists))
    
    print(f"Generating {len(param_combinations)} configurations and submission scripts...", file=sys.stderr)

    for i, combo in enumerate(param_combinations):
        run_config = deepcopy(base_config)
        run_name_parts = []
        
        for param_name, value in zip(param_names, combo):
            set_nested_value(run_config, param_name, value)
            short_name = param_name.split('.')[-1].replace('_ms', '').replace('_pa', '')
            run_name_parts.append(f"{short_name}{value}")
        
        run_name = "_".join(run_name_parts)
        print(f"  - Generating files for job {i+1}/{len(param_combinations)}: {run_name}", file=sys.stderr)
        
        run_config['output_config']['base_output_directory'] = sweep_output_path
        run_config['output_config']['output_directory_name'] = run_name

        config_filename = os.path.join(configs_path, f"{run_name}.json")
        with open(config_filename, 'w') as f:
            json.dump(run_config, f, indent=2)

        n_cores = run_config['simulation_parameters'].get('n_cores', 1)
        mem_gb = n_cores * 2
        
        lsf_content = create_lsf_script(run_name, n_cores, mem_gb, os.path.abspath(config_filename), project_dir)
        lsf_filename = os.path.join(submission_scripts_path, f"{run_name}.sh")
        with open(lsf_filename, 'w') as f:
            f.write(lsf_content)
    
    print(f"\nSuccessfully generated {len(param_combinations)} LSF scripts.", file=sys.stderr)
    print(os.path.abspath(submission_scripts_path))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python generate_sweep.py <path_to_sweep.json>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])