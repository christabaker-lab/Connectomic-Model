import os
import json
import pickle
import random
import argparse
import pandas as pd
import numpy as np
from textwrap import dedent
from datetime import datetime
from glob import glob

# Import Brian2 components
from brian2 import (
    NeuronGroup, Synapses, PoissonInput, SpikeMonitor, Network,
    start_scope, mV, ms, Hz, store, restore, prefs, seed, defaultclock
)

# ---------------------------------------------------------
# OPTIMIZATION: Enable CUDA or C++ Compilation
# ---------------------------------------------------------
def setup_brian2_backend(use_cuda=False, gpu_id=0):
    """
    Configure Brian2 to use either CUDA or C++ compilation.
    
    Args:
        use_cuda: Whether to use CUDA (GPU) acceleration
        gpu_id: Which GPU device to use (0, 1, 2, etc.)
    """
    if use_cuda:
        try:
            import brian2cuda
            prefs.codegen.target = 'cuda'
            prefs.devices.cuda_standalone.cuda_backend.cuda_device = gpu_id
            print(f"  [Info] Brian2CUDA enabled on GPU {gpu_id}")
        except ImportError:
            print("  [Warning] brian2cuda not installed. Install with: pip install brian2cuda")
            print("  [Info] Falling back to C++/Cython")
            use_cuda = False
    
    if not use_cuda:
        try:
            import cython
            prefs.codegen.target = 'cython'
            print("  [Info] Using Cython backend")
        except ImportError:
            print("  [Info] Using numpy backend (slowest)")
            pass
    
    return use_cuda


def load_config(config_path):
    """Loads the JSON configuration file."""
    print(f"  [Info] Loading configuration from: {config_path}")
    with open(config_path, "r") as f:
        return json.load(f)


def load_data(paths):
    """Loads all necessary data files specified in the config."""
    print("  [Info] Loading data files...")
    try:
        completeness_df = pd.read_csv(
            paths["completeness_file"], index_col=0, dtype={"root_id": "str"}
        )
        connectivity_df = pd.read_parquet(paths["connectivity_file"])
        jo_clusters_df = pd.read_csv(paths["jo_cluster_file"])

        with open(paths["neuron_ranges_pickle"], "rb") as f:
            neuron_ranges = pickle.load(f)

        if "index" in completeness_df.columns:
            completeness_df = completeness_df.set_index("index")

        idx_to_id = completeness_df["root_id"].to_dict()
        id_to_idx = {v: k for k, v in idx_to_id.items()}

        return {
            "completeness": completeness_df,
            "connectivity": connectivity_df,
            "jo_clusters": jo_clusters_df,
            "neuron_ranges": neuron_ranges,
            "idx_to_id": idx_to_id,
            "id_to_idx": id_to_idx,
        }
    except FileNotFoundError as e:
        print(f"  [Error] Data file not found: {e}")
        raise
    except Exception as e:
        print(f"  [Error] Failed to load data: {e}")
        raise


def prepare_stimulation(config, data, batch_seed_offset):
    """Processes the stimulation config to get a list of neurons to activate."""
    random.seed(batch_seed_offset)
    
    neurons_to_activate = []
    stimulation_plan = config["stimulation_config"]
    neuron_ranges = data["neuron_ranges"]
    id_to_idx = data["id_to_idx"]

    for side, hemisphere_config in stimulation_plan.items():
        if not hemisphere_config["activate"]: continue
        
        for group_config in hemisphere_config["groups"]:
            base_group_name = group_config['group_name']

            if base_group_name.lower().startswith('cluster'):
                group_name_to_lookup = base_group_name
                if group_name_to_lookup not in neuron_ranges:
                    continue
            else:
                side_suffix = '_L' if 'left' in side else '_R'
                sided_name = base_group_name + side_suffix
                
                if sided_name in neuron_ranges:
                    group_name_to_lookup = sided_name
                elif base_group_name in neuron_ranges:
                    group_name_to_lookup = base_group_name
                else:
                    continue

            candidate_root_ids = neuron_ranges[group_name_to_lookup]
            candidate_indices = [id_to_idx[str(rid)] for rid in candidate_root_ids if str(rid) in id_to_idx]

            if not candidate_indices: continue

            percent = group_config["random_selection_percent"]
            num_to_select = int(len(candidate_indices) * (percent / 100.0))
            if num_to_select == 0 and len(candidate_indices) > 0: num_to_select = 1
            
            selected_indices = random.sample(candidate_indices, num_to_select)
            
            rate = group_config["poisson_rate_hz"] * Hz
            for idx in selected_indices:
                neurons_to_activate.append({"index": idx, "rate": rate})

    unique_neurons = {item["index"]: item for item in neurons_to_activate}.values()
    return list(unique_neurons)


def prepare_silencing(config, data, batch_seed_offset):
    """Processes the silencing config to get a list of neuron indices to silence."""
    random.seed(batch_seed_offset)
    
    indices_to_silence = set()
    if "silencing_config" not in config: return indices_to_silence
        
    silencing_plan = config["silencing_config"]
    neuron_ranges = data["neuron_ranges"]
    id_to_idx = data["id_to_idx"]

    for side, hemisphere_config in silencing_plan.items():
        if not hemisphere_config.get("activate", False): continue
            
        for group_config in hemisphere_config.get("groups", []):
            base_group_name = group_config['group_name']

            if base_group_name.lower().startswith('cluster'):
                group_name_to_lookup = base_group_name
                if group_name_to_lookup not in neuron_ranges:
                    continue
            else:
                side_suffix = '_L' if 'left' in side else '_R'
                sided_name = base_group_name + side_suffix
                
                if sided_name in neuron_ranges:
                    group_name_to_lookup = sided_name
                elif base_group_name in neuron_ranges:
                    group_name_to_lookup = base_group_name
                else:
                    continue

            candidate_root_ids = neuron_ranges[group_name_to_lookup]
            candidate_indices = [id_to_idx[str(rid)] for rid in candidate_root_ids if str(rid) in id_to_idx]
            
            if not candidate_indices: continue

            percent = group_config.get("random_selection_percent", 100)
            num_to_select = int(len(candidate_indices) * (percent / 100.0))
            if num_to_select == 0 and len(candidate_indices) > 0: num_to_select = 1
            
            selected_indices = random.sample(candidate_indices, num_to_select)
            indices_to_silence.update(selected_indices)

    return indices_to_silence


def build_network(params, data):
    """Build the Brian2 network once per batch."""
    print("  [Info] Building network connectivity (This happens ONLY ONCE per batch)...")
    start_scope()
    defaultclock.dt = 0.1 * ms
    
    brian_params = {
        "v_0": -52 * mV, "v_rst": -52 * mV, "t_mbr": 20 * ms,
        "v_th": params["v_th_mv"] * mV, "t_rfc": params["t_rfc_ms"] * ms,
        "tau": params["tau_ms"] * ms, "w_syn": params["w_syn_mv"] * mV, "t_dly": 1.8 * ms,
    }
    
    model_eqs = dedent(""" 
        dv/dt = (v_0 - v + g) / t_mbr : volt (unless refractory)
        dg/dt = -g / tau              : volt (unless refractory)
        rfc                           : second 
    """)

    neu = NeuronGroup(
        N=len(data["completeness"]),
        model=model_eqs,
        threshold="v > v_th",
        reset="v = v_rst; g = 0 * mV",
        refractory="rfc",
        namespace=brian_params,
    )
    neu.v = brian_params["v_0"]
    neu.g = 0 * mV
    neu.rfc = brian_params["t_rfc"]

    syn = Synapses(neu, neu, "w : volt", on_pre="g += w", delay=brian_params["t_dly"])
    
    syn.connect(
        i=data["connectivity"]["Presynaptic_Index"].values,
        j=data["connectivity"]["Postsynaptic_Index"].values,
    )
    syn.w = data["connectivity"]["Connectivity x Excitatory"].values * brian_params["w_syn"]

    spk_mon = SpikeMonitor(neu)
    
    net = Network(neu, syn, spk_mon)
    net.store("initial_state")
    
    return net, neu, syn, spk_mon, brian_params


def post_process(spike_df, data, save_dir, n_trials, t_run_s):
    """Calculates summary statistics including firing rates and saves them."""
    print("  [Info] Performing post-processing analysis...")
    if spike_df.empty:
        print("  [Warning] No spikes were recorded. Creating empty summary file.")
        pd.DataFrame(
            columns=[
                "group",
                "side",
                "avg_number_of_spikes",
                "avg_unique_spiking_neurons",
                "mean_spike_rate_hz",
                "std_spike_rate_hz",
                "first_spike_time_ms",
                "first_spike_neuron_label",
                "first_spike_neuron_id",
            ]
        ).to_csv(os.path.join(save_dir, "summary_analysis.csv"), index=False)
        return

    summary_list = []
    analysis_groups = {}

    for group_name, root_ids in data["neuron_ranges"].items():
        indices = [
            data["id_to_idx"][str(rid)]
            for rid in root_ids
            if str(rid) in data["id_to_idx"]
        ]
        if indices:
            analysis_groups[group_name] = {"indices": indices, "type": "group"}

    for group_name, group_info in analysis_groups.items():
        indices = group_info["indices"]
        group_spikes = spike_df[spike_df["neuron_index"].isin(indices)]

        spikes_per_trial = (
            group_spikes.groupby("trial").size().reindex(range(n_trials), fill_value=0)
        )
        avg_spikes = spikes_per_trial.mean()

        unique_spikers_per_trial = (
            group_spikes.groupby("trial")["neuron_index"]
            .nunique()
            .reindex(range(n_trials), fill_value=0)
        )
        avg_unique_spikers = unique_spikers_per_trial.mean()

        rates_per_trial = spikes_per_trial / t_run_s
        mean_rate, std_rate = rates_per_trial.mean(), rates_per_trial.std()

        if not group_spikes.empty:
            first_spike = group_spikes.loc[group_spikes["spike_time_ms"].idxmin()]
            first_spike_time = first_spike["spike_time_ms"]
            first_spike_label = first_spike["neuron_label"]
            first_spike_id = np.int64(first_spike["neuron_id"])
        else:
            first_spike_time, first_spike_label, first_spike_id = np.nan, None, None

        side = "N/A"
        if group_info["type"] == "group":
            if group_name.endswith("_L"):
                side = "L"
            elif group_name.endswith("_R"):
                side = "R"

        summary_list.append(
            {
                "group": group_name,
                "side": side,
                "avg_number_of_spikes": avg_spikes,
                "avg_unique_spiking_neurons": avg_unique_spikers,
                "mean_spike_rate_hz": mean_rate,
                "std_spike_rate_hz": std_rate,
                "first_spike_time_ms": first_spike_time,
                "first_spike_neuron_label": first_spike_label,
                "first_spike_neuron_id": first_spike_id,
            }
        )

    summary_df = pd.DataFrame(summary_list)

    print("  [Info] Finding top 20 most active ungrouped neurons...")

    all_grouped_indices = set()
    for group_info in analysis_groups.values():
        all_grouped_indices.update(group_info["indices"])

    ungrouped_spikes_df = spike_df[~spike_df["neuron_index"].isin(all_grouped_indices)]

    if not ungrouped_spikes_df.empty:
        total_spikes_per_neuron = ungrouped_spikes_df.groupby("neuron_index").size()
        avg_rate_per_neuron = total_spikes_per_neuron / (n_trials * t_run_s)

        top_20_ungrouped_neurons = avg_rate_per_neuron.sort_values(
            ascending=False
        ).head(20)

        top_20_list = []
        for neuron_index, mean_rate in top_20_ungrouped_neurons.items():
            neuron_spikes = ungrouped_spikes_df[
                ungrouped_spikes_df["neuron_index"] == neuron_index
            ]

            spikes_per_trial = (
                neuron_spikes.groupby("trial")
                .size()
                .reindex(range(n_trials), fill_value=0)
            )
            rates_per_trial = spikes_per_trial / t_run_s
            std_rate = rates_per_trial.std()
            avg_spikes = spikes_per_trial.mean()

            first_spike = neuron_spikes.loc[neuron_spikes["spike_time_ms"].idxmin()]

            neuron_label = first_spike["neuron_label"]
            side = "N/A"
            if isinstance(neuron_label, str) and neuron_label.endswith(("_L", "_R")):
                side = neuron_label[-1]

            top_20_list.append(
                {
                    "group": f"INDIVIDUAL: {neuron_label}",
                    "side": side,
                    "avg_number_of_spikes": avg_spikes,
                    "avg_unique_spiking_neurons": 1.0,
                    "mean_spike_rate_hz": mean_rate,
                    "std_spike_rate_hz": std_rate,
                    "first_spike_time_ms": first_spike["spike_time_ms"],
                    "first_spike_neuron_label": neuron_label,
                    "first_spike_neuron_id": np.int64(first_spike["neuron_id"]),
                }
            )

        if top_20_list:
            top_20_df = pd.DataFrame(top_20_list)
            separator = pd.DataFrame([{"group": "--- TOP 20 INDIVIDUAL NEURONS ---"}])
            summary_df = pd.concat(
                [summary_df, separator, top_20_df], ignore_index=True
            )

    if not summary_df.empty:
        summary_df["first_spike_neuron_id"] = (
            summary_df["first_spike_neuron_id"].fillna(0).astype(np.int64)
        )
        summary_df['avg_number_of_spikes'] = np.ceil(summary_df['avg_number_of_spikes'])

    summary_df.to_csv(
        os.path.join(save_dir, "summary_analysis.csv"), index=False
    )
    print("  [Info] Post-processing complete. Summary file saved.")


def create_raster_plot(spike_df, save_dir, data, plot_config, trial_to_plot=0):
    """Creates a raster plot for user-selected neuron groups."""
    import matplotlib.pyplot as plt
    
    groups_to_plot = plot_config.get("groups_to_plot", [])
    if not plot_config.get("enabled", False) or not groups_to_plot:
        print("  [Info] Raster plot disabled or no groups specified in config. Skipping.")
        return

    print(f"  [Info] Remapping Y-axis and creating raster plot for groups: {', '.join(groups_to_plot)}...")

    trial_spikes = spike_df[spike_df['trial'] == trial_to_plot].copy()
    if trial_spikes.empty:
        print(f"  [Warning] No spikes in trial {trial_to_plot}. Skipping raster plot.")
        return

    y_cursor = 0
    y_tick_locations, y_tick_labels = [], []
    index_to_plot_y_map = {}
    index_to_group_name_map = {}
    
    OVERLAP_GROUP_NAME = "OVERLAP" 
    gap_between_groups = 10 

    for group_name in groups_to_plot:
        if group_name not in data["neuron_ranges"]:
            if group_name + "_L" in data["neuron_ranges"]:
                group_name = group_name + "_L"
            elif group_name + "_R" in data["neuron_ranges"]:
                group_name = group_name + "_R"
            else:
                print(f"  [Warning] Group '{group_name}' not found in data. Skipping this group.")
                continue
                
        root_ids = data["neuron_ranges"][group_name]
        group_indices = sorted([data["id_to_idx"][str(rid)] for rid in root_ids if str(rid) in data["id_to_idx"]])
        
        if not group_indices:
            continue
            
        for i, original_index in enumerate(group_indices):
            index_to_plot_y_map[original_index] = y_cursor + i
            
            if original_index in index_to_group_name_map:
                index_to_group_name_map[original_index] = OVERLAP_GROUP_NAME
            else:
                index_to_group_name_map[original_index] = group_name
        
        num_neurons_in_group = len(group_indices)
        y_tick_locations.append(y_cursor + (num_neurons_in_group - 1) / 2)
        y_tick_labels.append(group_name)

        y_cursor += num_neurons_in_group + gap_between_groups

    if not index_to_plot_y_map:
        print("  [Warning] None of the selected groups contained valid neurons. Skipping plot.")
        return
        
    trial_spikes['plot_y'] = trial_spikes['neuron_index'].map(index_to_plot_y_map)
    trial_spikes['group_name'] = trial_spikes['neuron_index'].map(index_to_group_name_map)
    
    trial_spikes.dropna(subset=['plot_y', 'group_name'], inplace=True) 

    if trial_spikes.empty:
        print("  [Warning] No spikes found for any of the selected groups. No plot will be generated.")
        return

    num_labels = len(y_tick_labels)
    base_height_inches = 4
    height_per_label_inches = 0.3
    
    dynamic_height = base_height_inches + num_labels * height_per_label_inches
    dynamic_height = max(10, min(80, dynamic_height)) 
    
    print(f"  [Info] Plotting {num_labels} groups. Setting figure height to {dynamic_height:.1f} inches.")
    
    plt.figure(figsize=(15, dynamic_height)) 
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(groups_to_plot)))
    group_to_color = {name: color for name, color in zip(groups_to_plot, colors)}
    group_to_color[OVERLAP_GROUP_NAME] = '#FFFFFF' 

    spike_colors = trial_spikes['group_name'].map(group_to_color)
    
    plt.scatter(
        trial_spikes['spike_time_ms'],
        trial_spikes['plot_y'],
        s=20,
        c='#000000',
        marker='.'
    )
    
    plt.xlabel("Time (ms)", fontsize=12)
    plt.ylabel("Neuron Group", fontsize=12)
    plt.title(f"Raster Plot of Selected Neuron Groups (Trial {trial_to_plot})", fontsize=16)
    
    plt.yticks(ticks=y_tick_locations, labels=y_tick_labels, fontsize=8) 
    
    ax = plt.gca()
    ax.set_facecolor('#FFFFFF') 
    
    plt.grid(True, linestyle='--', alpha=0.2, axis='x') 
    plt.tight_layout()
    
    plot_path = os.path.join(save_dir, "raster_plot_selected_groups.svg")
    plt.savefig(plot_path, dpi=300, format='svg')
    plt.close()
    print(f"  [Info] Raster plot saved to: {plot_path}")


def run_batch(config_path, batch_id, trials_per_batch, use_cuda=False, gpu_id=0):
    """
    Run a batch of trials sequentially (GPU already provides parallelism).
    
    Args:
        config_path: Path to config JSON
        batch_id: ID of this batch
        trials_per_batch: Number of trials to run
        use_cuda: Whether to use CUDA acceleration
        gpu_id: Which GPU to use (if multiple available)
    """
    # Setup backend
    cuda_enabled = setup_brian2_backend(use_cuda=use_cuda, gpu_id=gpu_id)
    
    config = load_config(config_path)
    data = load_data(config["file_paths"])
    params = config["simulation_parameters"]
    
    print(f"  [Info] Starting Batch {batch_id} with {trials_per_batch} trials...")
    if cuda_enabled:
        print(f"  [Info] Using GPU {gpu_id} for acceleration")
    
    # Build network once
    net, neu, syn, spk_mon, brian_params = build_network(params, data)
    
    all_spikes = []
    
    for i in range(trials_per_batch):
        trial_seed = 1000 + (batch_id * 1000) + i
        seed(trial_seed)
        np.random.seed(trial_seed)
        
        # Restore clean state
        net.restore("initial_state")
        
        # Apply silencing (vectorized)
        silenced_indices = prepare_silencing(config, data, trial_seed)
        if silenced_indices:
            silenced_arr = np.array(list(silenced_indices))
            syn.w[silenced_arr, :] = 0 * mV
            
        # Apply stimulation
        stimulated_neurons = prepare_stimulation(config, data, trial_seed)
        active_inputs = []
        for neuron_info in stimulated_neurons:
            idx = neuron_info["index"]
            p_input = PoissonInput(
                target=neu[idx],
                target_var="v",
                N=1,
                rate=neuron_info["rate"],
                weight=brian_params["w_syn"] * params["f_poi"],
            )
            neu[idx].rfc = 0 * ms
            net.add(p_input)
            active_inputs.append(p_input)
            
        # Run simulation
        net.run(duration=params["t_run_ms"] * ms)
        
        # Collect data
        global_trial_num = i + ((batch_id - 1) * trials_per_batch)
        df_trial = pd.DataFrame({
            "neuron_index": np.array(spk_mon.i, dtype=np.int32),
            "spike_time_ms": np.array(spk_mon.t / ms, dtype=np.float32),
            "trial": global_trial_num
        })
        
        user_indices = {n["index"] for n in stimulated_neurons}
        df_trial["is_user_activated"] = df_trial["neuron_index"].isin(user_indices)
        
        all_spikes.append(df_trial)
        
        # Cleanup inputs
        for p_input in active_inputs:
            net.remove(p_input)
        
        if (i + 1) % 10 == 0:
            print(f"    [Progress] Completed {i + 1}/{trials_per_batch} trials")

    # Save results with experiment name in directory structure
    if all_spikes:
        spike_df = pd.concat(all_spikes, ignore_index=True)
        spike_df["neuron_id"] = spike_df["neuron_index"].map(data["idx_to_id"])
        
        # Create experiment-specific directory structure (NO TIMESTAMP!)
        out_conf = config["output_config"]
        experiment_name = out_conf["output_directory_name"]
        
        # Save directory: base_output_directory/experiment_name/
        save_dir = os.path.join(
            out_conf["base_output_directory"], 
            experiment_name  # NO TIMESTAMP - keeps all batches in same folder
        )
        os.makedirs(save_dir, exist_ok=True)
        
        # Save config in experiment directory (only once per batch 1)
        config_save_path = os.path.join(save_dir, "config.json")
        if batch_id == 1 and not os.path.exists(config_save_path):
            with open(config_save_path, 'w') as f:
                json.dump(config, f, indent=2)
        
        # Filename includes batch_id
        save_path = os.path.join(save_dir, f"spikes_batch_{batch_id}.parquet")
        spike_df.to_parquet(save_path, compression="gzip", engine="pyarrow")
        print(f"  [Info] Batch {batch_id} Complete. Saved to {save_path}")
    else:
        print(f"  [Warning] Batch {batch_id} produced no spikes.")


def aggregate_and_postprocess(base_output_dir=None, experiment_name=None):
    """
    Aggregates all batch parquet files and runs post-processing.
    Can process a single experiment or all experiments in the base directory.
    
    Args:
        base_output_dir: Base directory containing experiment folders (e.g., "simulation_results")
        experiment_name: Specific experiment to process, or None to process all
    """
    if base_output_dir is None:
        base_output_dir = "simulation_results"
    
    if not os.path.exists(base_output_dir):
        print(f"  [Error] Base output directory not found: {base_output_dir}")
        return
    
    # Find all experiment directories to process
    if experiment_name:
        # Process single experiment
        experiment_dirs = [os.path.join(base_output_dir, experiment_name)]
    else:
        # Process all experiments in base directory
        experiment_dirs = [
            os.path.join(base_output_dir, d) 
            for d in os.listdir(base_output_dir) 
            if os.path.isdir(os.path.join(base_output_dir, d))
        ]
    
    if not experiment_dirs:
        print(f"  [Error] No experiment directories found in {base_output_dir}")
        return
    
    print("\n" + "="*70)
    print(f"POST-PROCESSING {len(experiment_dirs)} EXPERIMENT(S)")
    print("="*70 + "\n")
    
    for exp_dir in experiment_dirs:
        exp_name = os.path.basename(exp_dir)
        
        # Check if this directory has batch files
        batch_files = sorted(glob(os.path.join(exp_dir, "spikes_batch_*.parquet")))
        if not batch_files:
            print(f"  [Skip] No batch files found in {exp_name}")
            continue
        
        print(f"\n{'='*70}")
        print(f"Processing: {exp_name}")
        print(f"{'='*70}")
        
        # Load config from experiment directory
        config_path = os.path.join(exp_dir, "config.json")
        if not os.path.exists(config_path):
            print(f"  [Error] Config file not found in {exp_dir}")
            continue
        
        config = load_config(config_path)
        data = load_data(config["file_paths"])
        params = config["simulation_parameters"]
        
        print(f"  [Info] Found {len(batch_files)} batch files to aggregate")
        
        # Load and concatenate all batches
        all_batches = []
        for batch_file in batch_files:
            print(f"  [Info] Loading {os.path.basename(batch_file)}...")
            df = pd.read_parquet(batch_file)
            all_batches.append(df)
        
        spike_df = pd.concat(all_batches, ignore_index=True)
        print(f"  [Info] Aggregated {len(spike_df)} total spikes across {spike_df['trial'].nunique()} trials")
        
        # Add neuron labels
        id_to_label_map = data["completeness"].set_index("root_id")["label"]
        spike_df["neuron_label"] = spike_df["neuron_id"].map(id_to_label_map)
        
        # Save combined file
        combined_path = os.path.join(exp_dir, "spikes_combined.parquet")
        spike_df.to_parquet(combined_path, compression="gzip", engine="pyarrow")
        print(f"  [Info] Saved combined spikes to: {combined_path}")
        
        # Run post-processing
        n_trials = spike_df['trial'].nunique()
        t_run_s = params["t_run_ms"] / 1000.0
        post_process(spike_df, data, exp_dir, n_trials, t_run_s)
        
        # Create raster plot if enabled
        if "raster_plot_config" in config and config["raster_plot_config"].get("enabled", False):
            create_raster_plot(spike_df, exp_dir, data, config["raster_plot_config"])
        
        print(f"  [✓] Completed processing: {exp_name}\n")
    
    print("\n" + "="*70)
    print("ALL POST-PROCESSING COMPLETE")
    print("="*70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", nargs='?', default=None, help="Path to config JSON (for batch mode)")
    parser.add_argument("--batch_id", type=int, default=None, help="The ID of this job batch")
    parser.add_argument("--trials", type=int, default=50, help="Trials per batch")
    parser.add_argument("--use_cuda", action="store_true", help="Use CUDA/GPU acceleration")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU device ID to use")
    parser.add_argument("--postprocess", action="store_true", help="Run post-processing")
    parser.add_argument("--base_dir", type=str, default="simulation_results", 
                       help="Base output directory containing experiments (default: simulation_results)")
    parser.add_argument("--experiment", type=str, default=None, 
                       help="Specific experiment name to process (default: process all)")
    args = parser.parse_args()

    if args.postprocess:
        # Post-processing mode - aggregate all experiments or specific one
        aggregate_and_postprocess(
            base_output_dir=args.base_dir,
            experiment_name=args.experiment
        )
    elif args.batch_id is not None and args.config_path is not None:
        # Batch mode - run trials
        run_batch(args.config_path, args.batch_id, args.trials, 
                  use_cuda=args.use_cuda, gpu_id=args.gpu_id)
    else:
        print("  [Error] Must specify either --postprocess or (config_path and --batch_id)")
        print("\n  Examples:")
        print("    Batch mode:  python new_model.py config.json --batch_id 1 --trials 50")
        print("    Post-process all: python new_model.py --postprocess")
        print("    Post-process one: python new_model.py --postprocess --experiment MultiRun1")
        exit(1)