import os
import json
import pickle
import random
from textwrap import dedent
from datetime import datetime
import pandas as pd
import numpy as np
from joblib import Parallel, delayed, parallel_backend
import matplotlib.pyplot as plt

# Import Brian2 components
from brian2 import (
    NeuronGroup,
    Synapses,
    PoissonInput,
    SpikeMonitor,
    Network,
    start_scope,
    mV,
    ms,
    Hz,
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


def prepare_stimulation(config, data):
    """Processes the stimulation config to get a list of neurons to activate."""
    print("  [Info] Preparing stimulation based on config...")
    neurons_to_activate = []
    
    stimulation_plan = config["stimulation_config"]
    neuron_ranges = data["neuron_ranges"]
    id_to_idx = data["id_to_idx"]

    for side, hemisphere_config in stimulation_plan.items():
        if not hemisphere_config["activate"]:
            continue
            
        for group_config in hemisphere_config["groups"]:
            base_group_name = group_config['group_name']

            if base_group_name.lower().startswith('cluster'):
                group_name_to_lookup = base_group_name
            else:
                side_suffix = '_L' if 'left' in side else '_R'
                group_name_to_lookup = base_group_name + side_suffix
            
            if group_name_to_lookup not in neuron_ranges:
                print(f"  [Warning] Group '{group_name_to_lookup}' not found in neuron_ranges. Skipping.")
                continue

            candidate_root_ids = neuron_ranges[group_name_to_lookup]
            candidate_indices = [id_to_idx[str(rid)] for rid in candidate_root_ids if str(rid) in id_to_idx]

            if not candidate_indices:
                continue

            percent = group_config["random_selection_percent"]
            num_to_select = int(len(candidate_indices) * (percent / 100.0))
            if num_to_select == 0 and len(candidate_indices) > 0: num_to_select = 1
            
            selected_indices = random.sample(candidate_indices, num_to_select)
            
            rate = group_config["poisson_rate_hz"] * Hz
            for idx in selected_indices:
                neurons_to_activate.append({"index": idx, "rate": rate})

    unique_neurons = {item["index"]: item for item in neurons_to_activate}.values()
    print(f"  [Info] Total unique neurons to be stimulated: {len(unique_neurons)}")
    return list(unique_neurons)


def prepare_silencing(config, data):
    """Processes the silencing config to get a list of neuron indices to silence."""
    print("  [Info] Preparing silencing based on config...")
    indices_to_silence = set()
    
    if "silencing_config" not in config:
        return indices_to_silence
        
    silencing_plan = config["silencing_config"]
    neuron_ranges = data["neuron_ranges"]
    id_to_idx = data["id_to_idx"]

    for side, hemisphere_config in silencing_plan.items():
        if not hemisphere_config.get("activate", False):
            continue
            
        for group_config in hemisphere_config.get("groups", []):
            base_group_name = group_config['group_name']

            if base_group_name.lower().startswith('cluster'):
                group_name_to_lookup = base_group_name
            else:
                side_suffix = '_L' if 'left' in side else '_R'
                group_name_to_lookup = base_group_name + side_suffix
            
            if group_name_to_lookup not in neuron_ranges:
                print(f"  [Warning] Silencing group '{group_name_to_lookup}' not found. Skipping.")
                continue

            candidate_root_ids = neuron_ranges[group_name_to_lookup]
            candidate_indices = [id_to_idx[str(rid)] for rid in candidate_root_ids if str(rid) in id_to_idx]

            if not candidate_indices:
                continue

            percent = group_config.get("random_selection_percent", 100)
            num_to_select = int(len(candidate_indices) * (percent / 100.0))
            if num_to_select == 0 and len(candidate_indices) > 0: num_to_select = 1
            
            selected_indices = random.sample(candidate_indices, num_to_select)
            indices_to_silence.update(selected_indices)

    print(f"  [Info] Total unique neurons to be silenced: {len(indices_to_silence)}")
    return indices_to_silence


def run_single_trial(params, data, stimulated_neurons, silenced_indices):
    """Builds and runs the Brian2 model for one trial."""
    start_scope()
    brian_params = {
        "v_0": -52 * mV,
        "v_rst": -52 * mV,
        "t_mbr": 20 * ms,
        "v_th": params["v_th_mv"] * mV,
        "t_rfc": params["t_rfc_ms"] * ms,
        "tau": params["tau_ms"] * ms,
        "w_syn": params["w_syn_mv"] * mV,
        "t_dly": 1.8 * ms,
    }
    model_eqs = dedent(
        """ dv/dt = (v_0 - v + g) / t_mbr : volt (unless refractory)
                           dg/dt = -g / tau              : volt (unless refractory)
                           rfc                           : second """
    )
    neu = NeuronGroup(
        N=len(data["completeness"]),
        model=model_eqs,
        method="linear",
        threshold="v > v_th",
        reset="v = v_rst; g = 0 * mV",
        refractory="rfc",
        namespace=brian_params,
    )
    neu.v, neu.g, neu.rfc = brian_params["v_0"], 0 * mV, brian_params["t_rfc"]
    syn = Synapses(neu, neu, "w : volt", on_pre="g += w", delay=brian_params["t_dly"])
    syn.connect(
        i=data["connectivity"]["Presynaptic_Index"].values,
        j=data["connectivity"]["Postsynaptic_Index"].values,
    )
    syn.w = (
        data["connectivity"]["Connectivity x Excitatory"].values * brian_params["w_syn"]
    )
    if silenced_indices:
        for neuron_idx in silenced_indices:
            syn.w[f"i == {neuron_idx}"] = 0 * mV
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
    spk_mon = SpikeMonitor(neu)
    net = Network(neu, syn, spk_mon, *poisson_inputs)
    net.run(duration=params["t_run_ms"] * ms)
    return spk_mon


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

    # for cluster_id, cluster_group in data["jo_clusters"].groupby("Cluster"):
    #     root_ids = cluster_group["pre_root_id"].unique()
    #     indices = [
    #         data["id_to_idx"][str(rid)]
    #         for rid in root_ids
    #         if str(rid) in data["id_to_idx"]
    #     ]
    #     if indices:
    #         analysis_groups[f"Cluster_{cluster_id}"] = {
    #             "indices": indices,
    #             "type": "cluster",
    #         }

    for group_name, group_info in analysis_groups.items():
        indices = group_info["indices"]
        group_spikes = spike_df[spike_df["neuron_index"].isin(indices)]

        # Calculate spikes per trial, reindex to include trials with 0 spikes, then get the average
        spikes_per_trial = (
            group_spikes.groupby("trial").size().reindex(range(n_trials), fill_value=0)
        )
        avg_spikes = spikes_per_trial.mean()

        # Calculate unique spiking neurons per trial, reindex, then get the average
        unique_spikers_per_trial = (
            group_spikes.groupby("trial")["neuron_index"]
            .nunique()
            .reindex(range(n_trials), fill_value=0)
        )
        avg_unique_spikers = unique_spikers_per_trial.mean()

        # This rate calculation is already an average of per-trial rates, so it's correct
        rates_per_trial = spikes_per_trial / t_run_s
        mean_rate, std_rate = rates_per_trial.mean(), rates_per_trial.std()

        if not group_spikes.empty and avg_spikes == 0:
            print("\n" + "="*20 + " DEBUGGING " + "="*20)
            print(f"Found group with spikes but zero stats: {group_name}")
            print(f"Total spikes found for this group: {len(group_spikes)}")
            print("Spikes per trial calculation:")
            print(spikes_per_trial)
            print("="*51 + "\n")

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

    # Section to get unassigned top 20 most active neurons
    # Get a set of all neuron indices that are already in a group
    all_grouped_indices = set()
    for group_info in analysis_groups.values():
        all_grouped_indices.update(group_info["indices"])

    # Filter for spikes from ungrouped neurons
    ungrouped_spikes_df = spike_df[~spike_df["neuron_index"].isin(all_grouped_indices)]

    if not ungrouped_spikes_df.empty:
        # 3. Calculate average firing rate for each ungrouped neuron
        total_spikes_per_neuron = ungrouped_spikes_df.groupby("neuron_index").size()
        avg_rate_per_neuron = total_spikes_per_neuron / (n_trials * t_run_s)

        # 4. Get the top 20
        top_20_ungrouped_neurons = avg_rate_per_neuron.sort_values(
            ascending=False
        ).head(20)

        top_20_list = []
        for neuron_index, mean_rate in top_20_ungrouped_neurons.items():
            neuron_spikes = ungrouped_spikes_df[
                ungrouped_spikes_df["neuron_index"] == neuron_index
            ]

            # Calculate stats for this individual neuron
            spikes_per_trial = (
                neuron_spikes.groupby("trial")
                .size()
                .reindex(range(n_trials), fill_value=0)
            )
            rates_per_trial = spikes_per_trial / t_run_s
            std_rate = rates_per_trial.std()
            avg_spikes = spikes_per_trial.mean()

            first_spike = neuron_spikes.loc[neuron_spikes["spike_time_ms"].idxmin()]

            # Format for the summary file
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
            # Add a separator row for clarity in the CSV
            separator = pd.DataFrame([{"group": "--- TOP 20 INDIVIDUAL NEURONS ---"}])
            summary_df = pd.concat(
                [summary_df, separator, top_20_df], ignore_index=True
            )

    if not summary_df.empty:
        summary_df["first_spike_neuron_id"] = (
            summary_df["first_spike_neuron_id"].fillna(0).astype(np.int64)
        )

        # Round up the average spikes column to the next whole number
        summary_df['avg_number_of_spikes'] = np.ceil(summary_df['avg_number_of_spikes'])

    summary_df.to_csv(
        os.path.join(save_dir, "summary_analysis.csv"), index=False
    )
    print("  [Info] Post-processing complete. Summary file saved.")


def create_raster_plot(spike_df, save_dir, data, plot_config, trial_to_plot=0):
    """
    Creates a raster plot for user-selected neuron groups, remapping the Y-axis
    to group neurons together visually. Neurons belonging to more than one
    plotted group are colored white.
    
    **This version dynamically adjusts figure height to prevent Y-axis label overlap.**
    """
    groups_to_plot = plot_config.get("groups_to_plot", [])
    if not plot_config.get("enabled", False) or not groups_to_plot:
        print("  [Info] Raster plot disabled or no groups specified in config. Skipping.")
        return

    print(f"  [Info] Remapping Y-axis and creating raster plot for groups: {', '.join(groups_to_plot)}...")

    trial_spikes = spike_df[spike_df['trial'] == trial_to_plot].copy()
    if trial_spikes.empty:
        print(f"  [Warning] No spikes in trial {trial_to_plot}. Skipping raster plot.")
        return

    # --- Remapping Logic ---
    y_cursor = 0
    y_tick_locations, y_tick_labels = [], []
    index_to_plot_y_map = {}
    index_to_group_name_map = {} # This will store the *final* group for coloring
    
    OVERLAP_GROUP_NAME = "OVERLAP" 
    gap_between_groups = 10 

    for group_name in groups_to_plot:
        if group_name not in data["neuron_ranges"]:
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
        
    # Apply mappings
    trial_spikes['plot_y'] = trial_spikes['neuron_index'].map(index_to_plot_y_map)
    trial_spikes['group_name'] = trial_spikes['neuron_index'].map(index_to_group_name_map)
    
    trial_spikes.dropna(subset=['plot_y', 'group_name'], inplace=True) 

    if trial_spikes.empty:
        print("  [Warning] No spikes found for any of the selected groups. No plot will be generated.")
        return

    # --- Plotting Logic ---
    
    # --- NEW: Calculate dynamic figure height ---
    num_labels = len(y_tick_labels)
    # Set a base height (for title, x-axis, margins) and add height per label
    base_height_inches = 4
    height_per_label_inches = 0.3  # 0.3 inches per label
    
    dynamic_height = base_height_inches + num_labels * height_per_label_inches
    
    # Set a reasonable minimum and maximum height
    dynamic_height = max(10, min(80, dynamic_height)) # Min 10in, Max 80in
    
    print(f"  [Info] Plotting {num_labels} groups. Setting figure height to {dynamic_height:.1f} inches.")
    # --- END NEW ---
    
    plt.figure(figsize=(15, dynamic_height)) # <-- DYNAMIC HEIGHT IS USED HERE
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(groups_to_plot)))
    group_to_color = {name: color for name, color in zip(groups_to_plot, colors)}
    group_to_color[OVERLAP_GROUP_NAME] = '#FFFFFF' # Set color to white

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
    
    # --- CHANGED: Reduced font size from 9 to 8 ---
    plt.yticks(ticks=y_tick_locations, labels=y_tick_labels, fontsize=8) 
    
    # Set plot background to dark to see the white dots
    ax = plt.gca()
    ax.set_facecolor('#FFFFFF') # Dark grey background
    
    plt.grid(True, linestyle='--', alpha=0.2, axis='x') # Fainter grid on x-axis only
    plt.tight_layout()
    
    plot_path = os.path.join(save_dir, "raster_plot_selected_groups.svg")
    plt.savefig(plot_path, dpi=300, format='svg')
    plt.close()
    print(f"  [Info] Raster plot saved to: {plot_path}")

def run_experiment(config_path):
    """Orchestrates the entire multi-trial experiment from a config file."""
    config = load_config(config_path)
    data = load_data(config["file_paths"])

    params = config["simulation_parameters"]
    n_trials = params.get("n_trials", 1)
    n_cores = params.get("n_cores", -1)

    print(
        f"\n  [Info] Starting experiment with {n_trials} trials on {n_cores if n_cores > 0 else 'all available'} cores..."
    )
    with parallel_backend("loky", n_jobs=n_cores):
        trial_results = Parallel()(
            delayed(run_trial_wrapper)(i, config, data, params) for i in range(n_trials)
        )
    print("  [Info] All trials complete.")

    print("  [Info] Aggregating and saving results...")
    all_spikes = []

    for i, result_dict in enumerate(trial_results):
        df_trial = pd.DataFrame(
            {
                "neuron_index": result_dict["spike_indices"],
                "spike_time_ms": result_dict["spike_times_ms"],
            }
        )
        df_trial["trial"] = i

        stimulated_in_trial = result_dict["stimulated_neurons"]
        user_indices_for_this_trial = {n["index"] for n in stimulated_in_trial}

        df_trial["activation_type"] = np.where(
            df_trial["neuron_index"].isin(user_indices_for_this_trial),
            "user",
            "natural",
        )
        all_spikes.append(df_trial)

    if not all_spikes:
        spike_df = pd.DataFrame()
    else:
        spike_df = pd.concat(all_spikes, ignore_index=True)
        spike_df["neuron_id"] = spike_df["neuron_index"].map(data["idx_to_id"])

        id_to_label_map = data["completeness"].set_index("root_id")["label"]
        spike_df["neuron_label"] = spike_df["neuron_id"].map(id_to_label_map)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(
        config["output_config"]["base_output_directory"],
        config["output_config"]["output_directory_name"],
    )
    save_dir = save_dir + "--" + timestamp
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    if not spike_df.empty:
        spike_df["neuron_id"] = spike_df["neuron_id"].fillna(0).astype(np.int64)

    spike_df.to_parquet(
        os.path.join(save_dir, "spikes.parquet"), compression="gzip", engine="pyarrow"
    )
    print(f"  [Info] Aggregated spike data saved to: {save_dir}")

    if not spike_df.empty and "raster_plot_config" in config:
        create_raster_plot(spike_df, save_dir, data, config["raster_plot_config"])

    t_run_s = params["t_run_ms"] / 1000.0
    post_process(spike_df, data, save_dir, n_trials, t_run_s)


def run_trial_wrapper(trial_num, config, data, params):
    """A helper function that prepares and runs a single trial."""
    print(f"    - Starting trial {trial_num + 1}...")
    stimulated_neurons_trial = prepare_stimulation(config, data)
    silenced_indices_trial = prepare_silencing(config, data)
    spk_mon = run_single_trial(
        params, data, stimulated_neurons_trial, silenced_indices_trial
    )

    # Extract raw data and return a simple dictionary instead of the Brian2 object
    return {
        "spike_indices": np.array(spk_mon.i),
        "spike_times_ms": np.array(spk_mon.t / ms),
        "stimulated_neurons": stimulated_neurons_trial,
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python new_model.py <path_to_config.json>")
        sys.exit(1)

    config_file_path = sys.argv[1]

    print("=" * 50)
    print("Starting Connectomics Experiment")
    print("=" * 50)

    run_experiment(config_file_path)

    print("\n" + "=" * 50)
    print("Experiment Finished.")
    print("=" * 50)
