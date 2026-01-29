import json
import os
from copy import deepcopy

# Base configuration template
BASE_CONFIG = {
    "output_config": {
        "base_output_directory": "simulation_results",
        "output_directory_name": "PLACEHOLDER"
    },
    "simulation_parameters": {
        "n_cores": -1,
        "n_trials": 10,
        "t_run_ms": 2000,
        "v_th_mv": -45.0,
        "t_rfc_ms": 2.2,
        "tau_ms": 5.0,
        "w_syn_mv": 1,
        "f_poi": 250
    },
    "file_paths": {
        "completeness_file": "data/CB_completeness.csv",
        "connectivity_file": "data/CB_connectivity.parquet",
        "neuron_ranges_pickle": "data/neuron_ranges.pkl",
        "jo_cluster_file": "data/JO_left_cluster_list_ordered_by_dendrogram_new_synapse_7-16.csv"
    },
    "stimulation_config": {
        "left_hemisphere": {
            "activate": True,
            "groups": [
                {"group_name": "JO-A", "random_selection_percent": 50, "poisson_rate_hz": 100},
                {"group_name": "JO-B", "random_selection_percent": 50, "poisson_rate_hz": 100}
            ]
        },
        "right_hemisphere": {
            "activate": True,
            "groups": [
                {"group_name": "JO-A", "random_selection_percent": 50, "poisson_rate_hz": 100},
                {"group_name": "JO-B", "random_selection_percent": 50, "poisson_rate_hz": 100}
            ]
        }
    },
    "silencing_config": {
        "left_hemisphere": {"activate": False, "groups": []},
        "right_hemisphere": {"activate": False, "groups": []}
    },
    "raster_plot_config": {
        "enabled": True,
        "groups_to_plot": [
            "PVLP_pr03-1_L", "PVLP_pr03-1_R",
            "PVLP_pr03-2_L", "PVLP_pr03-2_R",
            "PVLP_pr03-3_L", "PVLP_pr03-3_R",
            "pMN2_L", "pMN2_R",
            "AVLP_pr05-1_L", "AVLP_pr05-1_R",
            "AVLP_pr23_L", "AVLP_pr23_R",
            "A2_L", "A2_R"
        ]
    }
}

# LAYER 1: Left vs Right hemisphere activation percentages
# Format: (left_percent, right_percent)
LR_RATIOS = [
    (0, 100),    # Only right
    (25, 75),    # Mostly right
    (50, 50),    # Balanced
    (75, 25),    # Mostly left
    (100, 0)     # Only left
]

# LAYER 2: Poisson firing rates
POISSON_RATES = [28, 100, 150, 200, 250, 400]

# LAYER 3: JO-A vs JO-B selection percentages within each active hemisphere
# Format: (JO-A percent, JO-B percent)
JO_RATIOS = [
    (0, 100),    # Only JO-B
    (25, 75),    # Mostly JO-B
    (50, 50),    # Balanced
    (75, 25),    # Mostly JO-A
    (100, 0),    # Only JO-A
    (100, 100),  # Both at 100%
    (75, 75),    # Both at 75%
    (25, 25)     # Both at 25%
]


def generate_config(lr_left, lr_right, poisson_rate, joa_percent, job_percent, experiment_id):
    """
    Generate a single config file.
    
    Args:
        lr_left: Percentage of left hemisphere activation (0-100)
        lr_right: Percentage of right hemisphere activation (0-100)
        poisson_rate: Firing rate in Hz
        joa_percent: JO-A selection percentage (0-100)
        job_percent: JO-B selection percentage (0-100)
        experiment_id: Unique experiment number
    """
    config = deepcopy(BASE_CONFIG)
    
    # Descriptive experiment name
    config["output_config"]["output_directory_name"] = (
        f"Exp_{experiment_id:04d}_LR{lr_left}-{lr_right}_Rate{poisson_rate}_JO{joa_percent}-{job_percent}"
    )
    
    # Configure left hemisphere
    config["stimulation_config"]["left_hemisphere"]["activate"] = (lr_left > 0)
    if lr_left > 0:
        # Scale the JO percentages by the L/R ratio
        # This ensures that when lr_left=50, we get half the neurons compared to lr_left=100
        for i, (joa, job) in enumerate([(joa_percent, job_percent)] * 2):
            group_name = "JO-A" if i == 0 else "JO-B"
            percent = joa if i == 0 else job
            
            # Scale by lr_left percentage
            scaled_percent = (percent * lr_left) / 100.0
            
            config["stimulation_config"]["left_hemisphere"]["groups"][i] = {
                "group_name": group_name,
                "random_selection_percent": scaled_percent,
                "poisson_rate_hz": poisson_rate
            }
    
    # Configure right hemisphere
    config["stimulation_config"]["right_hemisphere"]["activate"] = (lr_right > 0)
    if lr_right > 0:
        for i, (joa, job) in enumerate([(joa_percent, job_percent)] * 2):
            group_name = "JO-A" if i == 0 else "JO-B"
            percent = joa if i == 0 else job
            
            # Scale by lr_right percentage
            scaled_percent = (percent * lr_right) / 100.0
            
            config["stimulation_config"]["right_hemisphere"]["groups"][i] = {
                "group_name": group_name,
                "random_selection_percent": scaled_percent,
                "poisson_rate_hz": poisson_rate
            }
    
    return config


def main():
    """Generate all config file combinations."""
    output_dir = "run_configs"
    os.makedirs(output_dir, exist_ok=True)
    
    experiment_id = 1
    config_files = []
    
    print("="*80)
    print("GENERATING CONFIG FILES - 3-LAYER SWEEP")
    print("="*80)
    print(f"Layer 1 (L/R): {len(LR_RATIOS)} combinations")
    print(f"Layer 2 (Rates): {len(POISSON_RATES)} rates")
    print(f"Layer 3 (JO-A/JO-B): {len(JO_RATIOS)} combinations")
    print(f"TOTAL: {len(LR_RATIOS) * len(POISSON_RATES) * len(JO_RATIOS)} experiments")
    print("="*80 + "\n")
    
    # Triple nested loop: L/R -> Rates -> JO-A/JO-B
    for lr_left, lr_right in LR_RATIOS:
        print(f"\n--- L/R Ratio: {lr_left}/{lr_right} ---")
        
        for poisson_rate in POISSON_RATES:
            print(f"  Rate: {poisson_rate} Hz")
            
            for joa_percent, job_percent in JO_RATIOS:
                # Generate config
                config = generate_config(
                    lr_left, lr_right, 
                    poisson_rate, 
                    joa_percent, job_percent, 
                    experiment_id
                )
                
                # Save to file
                filename = (
                    f"exp_{experiment_id:04d}_"
                    f"lr{lr_left}-{lr_right}_"
                    f"rate{poisson_rate}_"
                    f"jo{joa_percent}-{job_percent}.json"
                )
                filepath = os.path.join(output_dir, filename)
                
                with open(filepath, 'w') as f:
                    json.dump(config, f, indent=2)
                
                config_files.append(filename)
                print(f"    [{experiment_id:4d}] JO-A/B: {joa_percent}/{job_percent} -> {filename}")
                
                experiment_id += 1
    
    print("\n" + "="*80)
    print(f"COMPLETE: {len(config_files)} config files generated in '{output_dir}/'")
    print("="*80)
    
    # Save a summary file
    summary = {
        "total_experiments": len(config_files),
        "layer_1_lr_ratios": LR_RATIOS,
        "layer_2_poisson_rates": POISSON_RATES,
        "layer_3_jo_ratios": JO_RATIOS,
        "config_files": config_files
    }
    
    with open(os.path.join(output_dir, "_experiment_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary saved to: {output_dir}/_experiment_summary.json")
    print("\nTo submit all jobs, run:")
    print("  python submit_jobs.py")


if __name__ == "__main__":
    main()