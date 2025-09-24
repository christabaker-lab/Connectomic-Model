# analyze_results.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import pickle
import questionary

def plot_analysis(ax_raster, ax_psth, spike_df, n_trials, duration_ms, title):
    """Generates a raster and PSTH plot for a given set of spikes."""
    
    # --- Raster Plot ---
    if not spike_df.empty:
        # Create a unique integer for each neuron to plot on the y-axis
        unique_neurons = spike_df['neuron_index'].unique()
        neuron_y_map = {neuron_idx: i for i, neuron_idx in enumerate(unique_neurons)}
        
        spike_df['y_pos'] = spike_df['neuron_index'].map(neuron_y_map)
        
        ax_raster.scatter(spike_df['spike_time_ms'], 
                          spike_df['trial'] * len(unique_neurons) + spike_df['y_pos'], 
                          marker='.', s=2, c='black', alpha=0.7)
    
    ax_raster.set_title(f"Raster Plot - {title}")
    ax_raster.set_xlabel("Time (ms)")
    ax_raster.set_ylabel("Neuron / Trial")
    ax_raster.set_xlim(0, duration_ms)
    
    # --- PSTH Plot ---
    bin_size_ms = 10
    bins = np.arange(0, duration_ms + bin_size_ms, bin_size_ms)
    
    if not spike_df.empty:
        spike_counts, _ = np.histogram(spike_df['spike_time_ms'], bins=bins)
        psth = spike_counts / (n_trials * (bin_size_ms / 1000.0))
    else:
        psth = np.zeros(len(bins) - 1)
        
    ax_psth.bar(bins[:-1], psth, width=bin_size_ms, align='edge', edgecolor='black')
    ax_psth.set_title(f"PSTH - {title}")
    ax_psth.set_xlabel("Time (ms)")
    ax_psth.set_ylabel("Firing Rate (Hz)")
    ax_psth.set_xlim(0, duration_ms)


def main(old_path, new_path, ranges_path):
    print("Loading data...")
    try:
        df_old = pd.read_parquet(old_path)
        df_new = pd.read_parquet(new_path)
        with open(ranges_path, 'rb') as f:
            neuron_ranges = pickle.load(f)
    except FileNotFoundError as e:
        print(f"Error loading file: {e}")
        return
    
    # --- Get Simulation Parameters ---
    n_trials_old = df_old['trial'].max() + 1 if not df_old.empty else 1
    n_trials_new = df_new['trial'].max() + 1 if not df_new.empty else 1
    duration_ms = max(df_old['spike_time_ms'].max() if not df_old.empty else 0, 
                      df_new['spike_time_ms'].max() if not df_new.empty else 0)
    if duration_ms == 0: duration_ms = 1000

    # --- Select a Group to Analyze ---
    available_groups = sorted(neuron_ranges.keys())
    
    group_to_analyze = questionary.select(
        "Which neuron group would you like to analyze?",
        choices=available_groups
    ).ask()

    # <<< THE FIX IS IN THIS BLOCK >>>

    # 1. Get the root IDs for the selected group, converting them to STRINGS.
    root_ids_in_group_str = {str(rid) for rid in neuron_ranges[group_to_analyze]}
    
    # 2. Filter the DataFrames by converting their 'neuron_id' column to STRINGS for the comparison.
    spikes_old = df_old[df_old['neuron_id'].astype(str).isin(root_ids_in_group_str)]
    spikes_new = df_new[df_new['neuron_id'].astype(str).isin(root_ids_in_group_str)]
    
    # <<< END OF FIX >>>
    
    print(f"\nAnalyzing group '{group_to_analyze}':")
    print(f"  - Old Model: Found {len(spikes_old)} spikes from {spikes_old['neuron_id'].nunique()} neurons.")
    print(f"  - New Model: Found {len(spikes_new)} spikes from {spikes_new['neuron_id'].nunique()} neurons.")

    # --- Create Plots ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=True)
    fig.suptitle(f"Analysis of Neuron Group: {group_to_analyze}", fontsize=16, fontweight='bold')
    
    plot_analysis(axes[0, 0], axes[1, 0], spikes_old, n_trials_old, duration_ms, "Old Model")
    plot_analysis(axes[0, 1], axes[1, 1], spikes_new, n_trials_new, duration_ms, "New Model")
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    
if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("\nUsage: python analyze_results.py <path_to_old_spikes.parquet> <path_to_new_spikes.parquet> <path_to_neuron_ranges.pkl>\n")
        sys.exit(1)
    
    main(sys.argv[1], sys.argv[2], sys.argv[3])