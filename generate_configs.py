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
            "groups": []
        },
        "right_hemisphere": {
            "activate": True,
            "groups": []
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

# Poisson firing rates (same as before)
POISSON_RATES = [28, 100, 150, 200, 250, 400]

# Left/Right ratios for each neuron group
LR_RATIOS = [
    (0, 100),    # Only right
    (25, 75),    # Mostly right
    (50, 50),    # Balanced
    (75, 25),    # Mostly left
    (100, 0)     # Only left
]


def generate_config_joa_only(lr_left, lr_right, poisson_rate, experiment_id):
    """Generate config for JO-A only experiments."""
    config = deepcopy(BASE_CONFIG)
    
    config["output_config"]["output_directory_name"] = (
        f"Exp_{experiment_id:04d}_JOA_LR{lr_left}-{lr_right}_Rate{poisson_rate}"
    )
    
    # Configure left hemisphere (JO-A only)
    if lr_left > 0:
        config["stimulation_config"]["left_hemisphere"]["groups"] = [{
            "group_name": "JO-A",
            "random_selection_percent": lr_left,
            "poisson_rate_hz": poisson_rate
        }]
    else:
        config["stimulation_config"]["left_hemisphere"]["activate"] = False
    
    # Configure right hemisphere (JO-A only)
    if lr_right > 0:
        config["stimulation_config"]["right_hemisphere"]["groups"] = [{
            "group_name": "JO-A",
            "random_selection_percent": lr_right,
            "poisson_rate_hz": poisson_rate
        }]
    else:
        config["stimulation_config"]["right_hemisphere"]["activate"] = False
    
    return config


def generate_config_job_only(lr_left, lr_right, poisson_rate, experiment_id):
    """Generate config for JO-B only experiments."""
    config = deepcopy(BASE_CONFIG)
    
    config["output_config"]["output_directory_name"] = (
        f"Exp_{experiment_id:04d}_JOB_LR{lr_left}-{lr_right}_Rate{poisson_rate}"
    )
    
    # Configure left hemisphere (JO-B only)
    if lr_left > 0:
        config["stimulation_config"]["left_hemisphere"]["groups"] = [{
            "group_name": "JO-B",
            "random_selection_percent": lr_left,
            "poisson_rate_hz": poisson_rate
        }]
    else:
        config["stimulation_config"]["left_hemisphere"]["activate"] = False
    
    # Configure right hemisphere (JO-B only)
    if lr_right > 0:
        config["stimulation_config"]["right_hemisphere"]["groups"] = [{
            "group_name": "JO-B",
            "random_selection_percent": lr_right,
            "poisson_rate_hz": poisson_rate
        }]
    else:
        config["stimulation_config"]["right_hemisphere"]["activate"] = False
    
    return config


def generate_config_both(lr_left, lr_right, poisson_rate, experiment_id):
    """Generate config for JO-A + JO-B combined experiments."""
    config = deepcopy(BASE_CONFIG)
    
    config["output_config"]["output_directory_name"] = (
        f"Exp_{experiment_id:04d}_JOAB_LR{lr_left}-{lr_right}_Rate{poisson_rate}"
    )
    
    # Configure left hemisphere (both JO-A and JO-B)
    if lr_left > 0:
        config["stimulation_config"]["left_hemisphere"]["groups"] = [
            {
                "group_name": "JO-A",
                "random_selection_percent": lr_left,
                "poisson_rate_hz": poisson_rate
            },
            {
                "group_name": "JO-B",
                "random_selection_percent": lr_left,
                "poisson_rate_hz": poisson_rate
            }
        ]
    else:
        config["stimulation_config"]["left_hemisphere"]["activate"] = False
    
    # Configure right hemisphere (both JO-A and JO-B)
    if lr_right > 0:
        config["stimulation_config"]["right_hemisphere"]["groups"] = [
            {
                "group_name": "JO-A",
                "random_selection_percent": lr_right,
                "poisson_rate_hz": poisson_rate
            },
            {
                "group_name": "JO-B",
                "random_selection_percent": lr_right,
                "poisson_rate_hz": poisson_rate
            }
        ]
    else:
        config["stimulation_config"]["right_hemisphere"]["activate"] = False
    
    return config


def main():
    """Generate all config file combinations."""
    output_dir = "run_configs"
    os.makedirs(output_dir, exist_ok=True)
    
    experiment_id = 1
    config_files = []
    
    print("="*80)
    print("GENERATING CONFIG FILES - NEW STRUCTURE")
    print("="*80)
    print(f"Neuron Groups: JO-A only, JO-B only, JO-A+JO-B combined")
    print(f"L/R Ratios: {len(LR_RATIOS)} combinations")
    print(f"Poisson Rates: {len(POISSON_RATES)} rates")
    print(f"TOTAL: 3 groups × {len(LR_RATIOS)} L/R × {len(POISSON_RATES)} rates = {3 * len(LR_RATIOS) * len(POISSON_RATES)} experiments")
    print("="*80 + "\n")
    
    # Three categories of experiments
    experiment_types = [
        ("JOA", generate_config_joa_only),
        ("JOB", generate_config_job_only),
        ("JOAB", generate_config_both)
    ]
    
    for exp_type, generator_func in experiment_types:
        print(f"\n{'='*80}")
        print(f"Generating {exp_type} experiments")
        print(f"{'='*80}")
        
        for poisson_rate in POISSON_RATES:
            print(f"\n  Rate: {poisson_rate} Hz")
            
            for lr_left, lr_right in LR_RATIOS:
                # Generate config
                config = generator_func(lr_left, lr_right, poisson_rate, experiment_id)
                
                # Save to file
                filename = (
                    f"exp_{experiment_id:04d}_"
                    f"{exp_type.lower()}_"
                    f"lr{lr_left}-{lr_right}_"
                    f"rate{poisson_rate}.json"
                )
                filepath = os.path.join(output_dir, filename)
                
                with open(filepath, 'w') as f:
                    json.dump(config, f, indent=2)
                
                config_files.append(filename)
                print(f"    [{experiment_id:4d}] {exp_type} L/R: {lr_left}/{lr_right} → {filename}")
                
                experiment_id += 1
    
    print("\n" + "="*80)
    print(f"COMPLETE: {len(config_files)} config files generated in '{output_dir}/'")
    print("="*80)
    
    # Save a summary file
    summary = {
        "total_experiments": len(config_files),
        "experiment_types": ["JO-A only", "JO-B only", "JO-A + JO-B combined"],
        "lr_ratios": LR_RATIOS,
        "poisson_rates": POISSON_RATES,
        "config_files": config_files,
        "breakdown": {
            "JOA_only": len(LR_RATIOS) * len(POISSON_RATES),
            "JOB_only": len(LR_RATIOS) * len(POISSON_RATES),
            "JOAB_combined": len(LR_RATIOS) * len(POISSON_RATES)
        }
    }
    
    with open(os.path.join(output_dir, "_experiment_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nBreakdown:")
    print(f"  JO-A only:      {summary['breakdown']['JOA_only']} experiments")
    print(f"  JO-B only:      {summary['breakdown']['JOB_only']} experiments")
    print(f"  JO-A+JO-B:      {summary['breakdown']['JOAB_combined']} experiments")
    print(f"\nSummary saved to: {output_dir}/_experiment_summary.json")
    print("\nTo submit all jobs, run:")
    print("  python submitSimulationAssay.py")


if __name__ == "__main__":
    main()