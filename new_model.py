import os
import json
import pickle
import random
import time
import argparse
import pandas as pd
import numpy as np
import h5py
from textwrap import dedent
from datetime import datetime
from glob import glob

# Import Brian2 components
from brian2 import (
    NeuronGroup, Synapses, PoissonInput, SpikeMonitor, Network,
    start_scope, mV, ms, Hz, store, restore, prefs, seed, defaultclock
)

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


def post_process(stats_file, data, save_dir, n_trials, t_run_s):
    """Calculates summary statistics from aggregated HDF5 data."""
    print("  [Info] Performing post-processing analysis from aggregated statistics...")
    
    # Read aggregated statistics from HDF5
    with h5py.File(stats_file, 'r') as f:
        # Collect all batch data
        all_spike_counts = []
        all_first_spikes = []
        
        batch_names = [k for k in sorted(f.keys()) if k.startswith('batch_')]
        
        if not batch_names:
            print("  [Warning] No batch data found in HDF5 file.")
            pd.DataFrame(
                columns=[
                    "group", "side", "avg_number_of_spikes", "avg_unique_spiking_neurons",
                    "mean_spike_rate_hz", "std_spike_rate_hz", "first_spike_time_ms",
                    "first_spike_neuron_label", "first_spike_neuron_id",
                ]
            ).to_csv(os.path.join(save_dir, "summary_analysis.csv"), index=False)
            return
        
        print(f"  [Info] Found {len(batch_names)} batches: {', '.join(batch_names)}")
        
        for batch_name in batch_names:
            batch_group = f[batch_name]
            all_spike_counts.append(batch_group['spike_counts'][:])
            all_first_spikes.append(batch_group['first_spikes'][:])
        
        if not all_spike_counts:
            print("  [Warning] No batch data found in HDF5 file.")
            pd.DataFrame(
                columns=[
                    "group", "side", "avg_number_of_spikes", "avg_unique_spiking_neurons",
                    "mean_spike_rate_hz", "std_spike_rate_hz", "first_spike_time_ms",
                    "first_spike_neuron_label", "first_spike_neuron_id",
                ]
            ).to_csv(os.path.join(save_dir, "summary_analysis.csv"), index=False)
            return
        
        # Stack all batches: shape will be (n_trials, n_neurons)
        spike_counts_array = np.vstack(all_spike_counts)
        first_spikes_array = np.vstack(all_first_spikes)
    
    print(f"  [Info] Loaded statistics for {spike_counts_array.shape[0]} trials and {spike_counts_array.shape[1]} neurons")
    
    summary_list = []
    analysis_groups = {}

    # Build analysis groups
    for group_name, root_ids in data["neuron_ranges"].items():
        indices = [
            data["id_to_idx"][str(rid)]
            for rid in root_ids
            if str(rid) in data["id_to_idx"]
        ]
        if indices:
            analysis_groups[group_name] = {"indices": indices, "type": "group"}

    # Analyze each group
    for group_name, group_info in analysis_groups.items():
        indices = group_info["indices"]
        
        # Get spike counts for this group across all trials
        group_spike_counts = spike_counts_array[:, indices].sum(axis=1)  # Sum across neurons in group
        avg_spikes = group_spike_counts.mean()
        
        # Count unique spiking neurons per trial
        unique_spikers_per_trial = (spike_counts_array[:, indices] > 0).sum(axis=1)
        avg_unique_spikers = unique_spikers_per_trial.mean()
        
        # Calculate firing rates
        rates_per_trial = group_spike_counts / t_run_s
        mean_rate, std_rate = rates_per_trial.mean(), rates_per_trial.std()
        
        # Find first spike across all trials and neurons in this group
        group_first_spikes = first_spikes_array[:, indices]
        first_spike_time = np.nanmin(group_first_spikes)
        
        if not np.isnan(first_spike_time):
            # Find which neuron had the first spike
            trial_idx, neuron_idx_in_group = np.where(group_first_spikes == first_spike_time)
            if len(trial_idx) > 0:
                neuron_global_idx = indices[neuron_idx_in_group[0]]
                first_spike_id = data["idx_to_id"][neuron_global_idx]
                first_spike_label = data["completeness"].loc[
                    data["completeness"]["root_id"] == first_spike_id, "label"
                ].values[0] if "label" in data["completeness"].columns else None
            else:
                first_spike_id = None
                first_spike_label = None
        else:
            first_spike_time = np.nan
            first_spike_id = None
            first_spike_label = None
        
        # Determine side
        side = "N/A"
        if group_info["type"] == "group":
            if group_name.endswith("_L"):
                side = "L"
            elif group_name.endswith("_R"):
                side = "R"
        
        summary_list.append({
            "group": group_name,
            "side": side,
            "avg_number_of_spikes": avg_spikes,
            "avg_unique_spiking_neurons": avg_unique_spikers,
            "mean_spike_rate_hz": mean_rate,
            "std_spike_rate_hz": std_rate,
            "first_spike_time_ms": first_spike_time,
            "first_spike_neuron_label": first_spike_label,
            "first_spike_neuron_id": np.int64(first_spike_id) if first_spike_id else 0,
        })

    summary_df = pd.DataFrame(summary_list)

    print("  [Info] Finding top 20 most active ungrouped neurons...")

    # Get all grouped indices
    all_grouped_indices = set()
    for group_info in analysis_groups.values():
        all_grouped_indices.update(group_info["indices"])

    # Find ungrouped neurons
    all_neuron_indices = set(range(spike_counts_array.shape[1]))
    ungrouped_indices = list(all_neuron_indices - all_grouped_indices)

    if ungrouped_indices:
        # Calculate total spikes and rates for ungrouped neurons
        ungrouped_spike_counts = spike_counts_array[:, ungrouped_indices]
        total_spikes_per_neuron = ungrouped_spike_counts.sum(axis=0)
        avg_rate_per_neuron = total_spikes_per_neuron / (n_trials * t_run_s)
        
        # Get top 20
        top_20_idx = np.argsort(avg_rate_per_neuron)[-20:][::-1]
        
        top_20_list = []
        for idx_in_ungrouped in top_20_idx:
            neuron_global_idx = ungrouped_indices[idx_in_ungrouped]
            mean_rate = avg_rate_per_neuron[idx_in_ungrouped]
            
            # Calculate stats for this neuron
            neuron_spike_counts = ungrouped_spike_counts[:, idx_in_ungrouped]
            rates_per_trial = neuron_spike_counts / t_run_s
            std_rate = rates_per_trial.std()
            avg_spikes = neuron_spike_counts.mean()
            
            # Get first spike time
            first_spike_time = np.nanmin(first_spikes_array[:, neuron_global_idx])
            
            # Get neuron info
            neuron_id = data["idx_to_id"][neuron_global_idx]
            neuron_label = data["completeness"].loc[
                data["completeness"]["root_id"] == neuron_id, "label"
            ].values[0] if "label" in data["completeness"].columns else str(neuron_id)
            
            side = "N/A"
            if isinstance(neuron_label, str) and neuron_label.endswith(("_L", "_R")):
                side = neuron_label[-1]
            
            top_20_list.append({
                "group": f"INDIVIDUAL: {neuron_label}",
                "side": side,
                "avg_number_of_spikes": avg_spikes,
                "avg_unique_spiking_neurons": 1.0,
                "mean_spike_rate_hz": mean_rate,
                "std_spike_rate_hz": std_rate,
                "first_spike_time_ms": first_spike_time,
                "first_spike_neuron_label": neuron_label,
                "first_spike_neuron_id": np.int64(neuron_id),
            })
        
        if top_20_list:
            top_20_df = pd.DataFrame(top_20_list)
            separator = pd.DataFrame([{"group": "--- TOP 20 INDIVIDUAL NEURONS ---"}])
            summary_df = pd.concat([summary_df, separator, top_20_df], ignore_index=True)

    if not summary_df.empty:
        summary_df["first_spike_neuron_id"] = (
            summary_df["first_spike_neuron_id"].fillna(0).astype(np.int64)
        )
        summary_df['avg_number_of_spikes'] = np.ceil(summary_df['avg_number_of_spikes'])

    summary_df.to_csv(os.path.join(save_dir, "summary_analysis.csv"), index=False)
    print("  [Info] Post-processing complete. Summary file saved.")


def create_raster_plot(raster_data_file, save_dir, data, plot_config, trial_to_plot=0):
    """Creates a raster plot for user-selected neuron groups from saved raster data."""
    import matplotlib.pyplot as plt
    
    groups_to_plot = plot_config.get("groups_to_plot", [])
    if not plot_config.get("enabled", False) or not groups_to_plot:
        print("  [Info] Raster plot disabled or no groups specified in config. Skipping.")
        return

    print(f"  [Info] Creating raster plot for trial {trial_to_plot} from saved data...")

    # Load the raster data parquet file
    if not os.path.exists(raster_data_file):
        print(f"  [Warning] Raster data file not found: {raster_data_file}")
        return
    
    spike_df = pd.read_parquet(raster_data_file)
    
    # Filter to the specific trial
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
    
    plot_path = os.path.join(save_dir, f"raster_plot_trial_{trial_to_plot}.svg")
    plt.savefig(plot_path, dpi=300, format='svg')
    plt.close()
    print(f"  [Info] Raster plot saved to: {plot_path}")


def run_single_trial_isolated(config, data, params, brian_params_dict, trial_seed, global_trial_num, save_full_spikes=False):
    """
    Run a single trial in complete isolation.
    
    Args:
        save_full_spikes: If True, returns full spike data for rasterization (trials 0-4)
                         If False, returns only aggregated statistics
    """
    # Each worker needs its own Brian2 scope
    start_scope()
    defaultclock.dt = 0.1 * ms
    
    # Use numpy backend to avoid Cython issues
    prefs.codegen.target = 'numpy'
    
    # Seed this trial
    seed(trial_seed)
    np.random.seed(trial_seed)
    random.seed(trial_seed)
    
    # Build brian_params with Brian2 units
    brian_params = {
        "v_0": brian_params_dict["v_0"] * mV,
        "v_rst": brian_params_dict["v_rst"] * mV,
        "t_mbr": brian_params_dict["t_mbr"] * ms,
        "v_th": brian_params_dict["v_th"] * mV,
        "t_rfc": brian_params_dict["t_rfc"] * ms,
        "tau": brian_params_dict["tau"] * ms,
        "w_syn": brian_params_dict["w_syn"] * mV,
        "t_dly": brian_params_dict["t_dly"] * ms,
    }
    
    model_eqs = dedent(""" 
        dv/dt = (v_0 - v + g) / t_mbr : volt (unless refractory)
        dg/dt = -g / tau              : volt (unless refractory)
        rfc                           : second 
    """)
    
    # Build neuron group
    n_neurons = len(data["completeness"])
    neu = NeuronGroup(
        N=n_neurons,
        model=model_eqs,
        threshold="v > v_th",
        reset="v = v_rst; g = 0 * mV",
        refractory="rfc",
        namespace=brian_params,
    )
    neu.v = brian_params["v_0"]
    neu.g = 0 * mV
    neu.rfc = brian_params["t_rfc"]
    
    # Build synapses
    syn = Synapses(neu, neu, "w : volt", on_pre="g += w", delay=brian_params["t_dly"])
    syn.connect(
        i=data["connectivity"]["Presynaptic_Index"].values,
        j=data["connectivity"]["Postsynaptic_Index"].values,
    )
    syn.w = data["connectivity"]["Connectivity x Excitatory"].values * brian_params["w_syn"]
    
    # Apply silencing (vectorized)
    silenced_indices = prepare_silencing(config, data, trial_seed)
    if silenced_indices:
        silenced_arr = np.array(list(silenced_indices))
        syn.w[silenced_arr, :] = 0 * mV
    
    # Apply stimulation
    stimulated_neurons = prepare_stimulation(config, data, trial_seed)
    poisson_inputs = []
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
        poisson_inputs.append(p_input)
    
    # Setup spike monitor
    spk_mon = SpikeMonitor(neu)
    
    # Build network
    net = Network(neu, syn, spk_mon, *poisson_inputs)
    
    # Run simulation
    net.run(duration=params["t_run_ms"] * ms)
    
    # Extract spike data
    spikes_i = np.array(spk_mon.i, dtype=np.int32)
    spikes_t = np.array(spk_mon.t / ms, dtype=np.float32)
    
    # Calculate aggregated statistics for ALL trials
    neuron_spike_counts = np.bincount(spikes_i, minlength=n_neurons).astype(np.uint16)
    
    # Calculate first spike time per neuron - VECTORIZED VERSION
    first_spike_times = np.full(n_neurons, np.nan, dtype=np.float32)
    if len(spikes_i) > 0:
        # Sort by time to get first spikes efficiently
        sort_idx = np.argsort(spikes_t)
        sorted_neurons = spikes_i[sort_idx]
        sorted_times = spikes_t[sort_idx]
        
        # Get indices of first occurrence of each neuron
        _, first_spike_idx = np.unique(sorted_neurons, return_index=True)
        
        # Assign first spike times
        first_spike_neurons = sorted_neurons[first_spike_idx]
        first_spike_times[first_spike_neurons] = sorted_times[first_spike_idx]
    
    # Build return dictionary with aggregated stats
    result = {
        "trial": global_trial_num,
        "neuron_spike_counts": neuron_spike_counts,
        "first_spike_times": first_spike_times,
        "total_spikes": len(spikes_i),
    }
    
    # If this is one of the first 5 trials, also save full spike data for rasterization
    if save_full_spikes:
        result["full_spikes"] = {
            "neuron_index": spikes_i,
            "spike_time_ms": spikes_t,
            "trial": global_trial_num,
        }
    
    return result


def run_batch(config_path, batch_id, trials_per_batch):
    """
    Run a batch of trials in parallel using multiprocessing.
    Saves aggregated statistics to HDF5 and full spike data for first 5 trials.
    
    Args:
        config_path: Path to config JSON
        batch_id: ID of this batch
        trials_per_batch: Number of trials to run
    """
    from multiprocessing import Pool
    import multiprocessing as mp
    
    config = load_config(config_path)
    data = load_data(config["file_paths"])
    params = config["simulation_parameters"]
    
    # Get number of cores
    n_cores = params.get("n_cores", -1)
    if n_cores <= 0:
        n_cores = mp.cpu_count()
    
    print(f"  [Info] Starting Batch {batch_id} with {trials_per_batch} trials on {n_cores} cores...")
    
    # Prepare brian_params as a dictionary (not Brian2 objects, for pickling)
    brian_params_dict = {
        "v_0": -52,
        "v_rst": -52,
        "t_mbr": 20,
        "v_th": params["v_th_mv"],
        "t_rfc": params["t_rfc_ms"],
        "tau": params["tau_ms"],
        "w_syn": params["w_syn_mv"],
        "t_dly": 1.8,
    }
    
    # Create experiment directory structure
    out_conf = config["output_config"]
    experiment_name = out_conf["output_directory_name"]
    save_dir = os.path.join(out_conf["base_output_directory"], experiment_name)
    os.makedirs(save_dir, exist_ok=True)
    
    # Save config in experiment directory (only once for batch 1)
    config_save_path = os.path.join(save_dir, "config.json")
    if batch_id == 1 and not os.path.exists(config_save_path):
        with open(config_save_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    # Prepare arguments for each trial
    trial_args = []
    
    # Check if we should save raster data from this batch
    # Save from batch_1 normally, OR from current batch if batch_1 data doesn't exist
    raster_file = os.path.join(save_dir, "raster_data_trials_0_to_4.parquet")
    should_save_raster = (batch_id == 1) or (not os.path.exists(raster_file) and batch_id <= 5)
    
    if should_save_raster and batch_id > 1:
        print(f"  [Info] Raster data missing - will save trials 0-4 from batch {batch_id}")
    
    for i in range(trials_per_batch):
        trial_seed = 1000 + (batch_id * 1000) + i
        global_trial_num = i + ((batch_id - 1) * trials_per_batch)
        
        # Save full spikes for first 5 trials of this batch if it's batch_1 OR if raster data is missing
        if should_save_raster:
            save_full_spikes = (i < 5)  # First 5 trials of THIS batch
        else:
            save_full_spikes = False
        
        trial_args.append((config, data, params, brian_params_dict, trial_seed, global_trial_num, save_full_spikes))
    
    # Run trials in parallel
    print(f"  [Info] Running {trials_per_batch} trials in parallel across {n_cores} cores...")
    with Pool(processes=n_cores) as pool:
        results = pool.starmap(run_single_trial_isolated, trial_args)
    
    if not results:
        print(f"  [Warning] Batch {batch_id} produced no results.")
        return
    
    # Separate full spike data from aggregated stats
    full_spike_results = [r for r in results if "full_spikes" in r]
    
    # Collect aggregated statistics
    spike_counts_batch = np.array([r["neuron_spike_counts"] for r in results])
    first_spikes_batch = np.array([r["first_spike_times"] for r in results])
    
    # Save aggregated statistics to HDF5
    stats_file = os.path.join(save_dir, "experiment_statistics.h5")
    print(f"  [Info] Saving aggregated statistics to HDF5: {stats_file}")
    
    # Retry logic for file locking
    import time
    max_retries = 30
    retry_delay = 10  # seconds
    
    for attempt in range(max_retries):
        try:
            with h5py.File(stats_file, 'a') as f:
                batch_group_name = f"batch_{batch_id}"
                
                # Delete if exists (for reruns)
                if batch_group_name in f:
                    print(f"  [Info] Batch {batch_id} already exists, overwriting...")
                    del f[batch_group_name]
                
                batch_group = f.create_group(batch_group_name)
                batch_group.create_dataset(
                    "spike_counts", 
                    data=spike_counts_batch, 
                    compression="gzip",
                    compression_opts=9
                )
                batch_group.create_dataset(
                    "first_spikes", 
                    data=first_spikes_batch, 
                    compression="gzip",
                    compression_opts=9
                )
                
                # Store metadata
                batch_group.attrs["batch_id"] = batch_id
                batch_group.attrs["n_trials"] = trials_per_batch
                batch_group.attrs["first_global_trial"] = (batch_id - 1) * trials_per_batch
                
                # Debug: Print what's in the file
                print(f"  [Debug] HDF5 file now contains groups: {list(f.keys())}")
            
            # Success - break out of retry loop
            print(f"  [Info] Batch {batch_id} aggregated statistics saved to HDF5")
            break
            
        except (BlockingIOError, OSError) as e:
            if attempt < max_retries - 1:
                wait_time = retry_delay + random.randint(0, 5)  # Add jitter
                print(f"  [Warning] File locked, retrying in {wait_time}s (attempt {attempt+1}/{max_retries})...")
                time.sleep(wait_time)
            else:
                print(f"  [Error] Failed to write to HDF5 after {max_retries} attempts")
                raise
    
    # Save full spike data for rasterization (first 5 trials only)
    if full_spike_results:
        print(f"  [Info] Saving full spike data for {len(full_spike_results)} trials for rasterization...")
        print(f"  [Debug] Full spike trials: {[r['trial'] for r in full_spike_results]}")
        
        # Combine all full spike data
        all_full_spikes = []
        for result in full_spike_results:
            fs = result["full_spikes"]
            df_trial = pd.DataFrame({
                "neuron_index": fs["neuron_index"],
                "spike_time_ms": fs["spike_time_ms"],
                "trial": fs["trial"],
            })
            all_full_spikes.append(df_trial)
        
        if all_full_spikes:
            full_spike_df = pd.concat(all_full_spikes, ignore_index=True)
            
            # Add neuron IDs and labels
            full_spike_df["neuron_id"] = full_spike_df["neuron_index"].map(data["idx_to_id"])
            id_to_label_map = data["completeness"].set_index("root_id")["label"]
            full_spike_df["neuron_label"] = full_spike_df["neuron_id"].map(id_to_label_map)
            
            # Save with zstd compression
            raster_file = os.path.join(save_dir, "raster_data_trials_0_to_4.parquet")
            
            # Append if file exists, otherwise create new
            if os.path.exists(raster_file):
                existing_df = pd.read_parquet(raster_file)
                full_spike_df = pd.concat([existing_df, full_spike_df], ignore_index=True)
            
            full_spike_df.to_parquet(
                raster_file,
                compression="zstd",
                compression_level=9,
                engine="pyarrow",
                index=False
            )
            print(f"  [Info] Full spike data saved to: {raster_file}")
    
    print(f"  [✓] Batch {batch_id} Complete")


def aggregate_and_postprocess(base_output_dir=None, experiment_name=None):
    """
    Runs post-processing on experiments using HDF5 statistics.
    
    Args:
        base_output_dir: Base directory containing experiment folders (e.g., "simulation_results")
        experiment_name: Specific experiment to process, or None to process all
    """
    import gc  # Garbage collection to free memory
    
    if base_output_dir is None:
        base_output_dir = "simulation_results"
    
    if not os.path.exists(base_output_dir):
        print(f"  [Error] Base output directory not found: {base_output_dir}")
        return
    
    # Find all experiment directories to process
    if experiment_name:
        experiment_dirs = [os.path.join(base_output_dir, experiment_name)]
    else:
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
        
        # Check if this directory has HDF5 statistics file
        stats_file = os.path.join(exp_dir, "experiment_statistics.h5")
        if not os.path.exists(stats_file):
            print(f"  [Skip] No HDF5 statistics file found in {exp_name}")
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
        
        # Count trials from HDF5
        with h5py.File(stats_file, 'r') as f:
            batch_names = [k for k in f.keys() if k.startswith('batch_')]
            n_batches = len(batch_names)
            
            if n_batches == 0:
                print(f"  [Warning] No batches found in HDF5 file. Skipping {exp_name}")
                continue
            
            # Get number of trials from first batch
            first_batch_name = sorted(batch_names)[0]
            first_batch = f[first_batch_name]
            trials_per_batch = first_batch['spike_counts'].shape[0]
            n_trials = n_batches * trials_per_batch
        
        print(f"  [Info] Found {n_batches} batches with {trials_per_batch} trials each = {n_trials} total trials")
        
        # Run post-processing
        t_run_s = params["t_run_ms"] / 1000.0
        post_process(stats_file, data, exp_dir, n_trials, t_run_s)
        
        # Create raster plot if enabled
        if "raster_plot_config" in config and config["raster_plot_config"].get("enabled", False):
            raster_data_file = os.path.join(exp_dir, "raster_data_trials_0_to_4.parquet")
            if os.path.exists(raster_data_file):
                print(f"  [Info] Creating raster plots from saved data...")
                # Create plots for trials 0-4
                for trial_num in range(5):
                    create_raster_plot(raster_data_file, exp_dir, data, config["raster_plot_config"], trial_to_plot=trial_num)
            else:
                print(f"  [Warning] Raster data file not found: {raster_data_file}")
        
        # Clean up memory before next experiment
        del data
        gc.collect()
        
        print(f"  [✓] Completed processing: {exp_name}\n")
    
    print("\n" + "="*70)
    print("ALL POST-PROCESSING COMPLETE")
    print("="*70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", nargs='?', default=None, help="Path to config JSON (for batch mode)")
    parser.add_argument("--batch_id", type=int, default=None, help="The ID of this job batch")
    parser.add_argument("--trials", type=int, default=50, help="Trials per batch")
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
        run_batch(args.config_path, args.batch_id, args.trials)
    else:
        print("  [Error] Must specify either --postprocess or (config_path and --batch_id)")
        print("\n  Examples:")
        print("    Batch mode:  python new_model_optimized.py config.json --batch_id 1 --trials 50")
        print("    Post-process all: python new_model_optimized.py --postprocess")
        print("    Post-process one: python new_model_optimized.py --postprocess --experiment MultiRun1")
        exit(1)