import pickle
import random
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from textwrap import dedent
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import networkx as nx

from brian2 import NeuronGroup, Synapses, PoissonInput, SpikeMonitor, StateMonitor, Network, TimedArray, start_scope
from brian2 import mV, ms, Hz

class ConnectomicsModel:
    def __init__(self, data_path='.'):
        # Load data
        try:
            self.aud_label_root_id = pd.read_csv(os.path.join(data_path, 'aud_label_root_id.csv'), index_col=0)
            self.aud_filtered_princeton = pd.read_csv(os.path.join(data_path, 'aud_filtered_princeton.csv'), index_col=0)
            
            # Load pickled data
            with open(os.path.join(data_path, 'id_idx_dict.pickle'), 'rb') as fi1:
                self.id_idx_dict = pickle.load(fi1)
            
            with open(os.path.join(data_path, 'idx_id_dict.pickle'), 'rb') as fi2:
                self.idx_id_dict = pickle.load(fi2)
            
            with open(os.path.join(data_path, 'neuron_ranges.pickle'), 'rb') as fi3:
                self.neurons_ranges = pickle.load(fi3)
            
            with open(os.path.join(data_path, 'neuron_groupings.pickle'), 'rb') as fi4:
                self.neuron_groupings = pickle.load(fi4)
        except FileNotFoundError as e:
            print(f"Error loading data: {e}")
            # Set defaults if files are missing
            self.neuron_groupings = ['JO-A', 'JO-B', 'JO-C', 'JO-D', 'JO-E']
            raise
        
        # Default parameters
        self.default_params = {
            't_run': 500 * ms,              # duration of trial
            'n_run': 1,                     # number of runs
            'v_0': -52 * mV,                # resting potential
            'v_rst': -52 * mV,              # reset potential after spike
            'v_th': -45 * mV,               # threshold for spiking
            't_mbr': 20 * ms,               # membrane time scale
            'tau': 5 * ms,                  # time constant
            't_rfc': 2.2 * ms,              # refractory period
            't_dly': 1.8 * ms,              # delay for changes in post-synaptic neuron
            'w_syn': 0.8 * mV,              # weight per synapse
            'r_poi': 250 * Hz,              # default rate of the Poisson inputs
            'r_poi2': 0 * Hz,               # default rate of a 2nd class of Poisson inputs
            'f_poi': 150,                   # scaling factor for Poisson synapse
            'A': 2.5 * mV,
            'f': 10 * Hz,
            'sine_frequency': 500 * Hz,      # Frequency of the sine wave
            'sine_amplitude': 1 * mV,        # Amplitude for exc neurons
        }

    def get_case(self, activation_neuron_list, neuron_group_activation, activate_both_sides, activation_side, random_selection, random_selection_percent):
        """Determine which case of neuron activation to use"""
        # Case 2: Direct neuron list provided
        if activation_neuron_list and len(activation_neuron_list) > 0:
            return 2
        
        # Check for valid neuron group
        if neuron_group_activation not in self.neuron_groupings:
            print(f'Incorrect neuron group selected: {neuron_group_activation}')
            return -1

        # Check for valid random selection parameters
        if random_selection and (random_selection_percent <= 0 or random_selection_percent > 100):
            print('Invalid random selection percentage')
            return -1

        # Case 1: Activate both sides
        if activate_both_sides:
            return 1
        
        # Case 3: Activate one side
        if activation_side in ['L', 'R']:
            return 3
        else:
            print(f'Invalid activation side: {activation_side}')
            return -1

    def get_excitatory_neurons(self, activation_neuron_list=[], neuron_group_activation='JO-A', 
                              activate_both_sides=True, activation_side='L', 
                              random_selection=True, random_selection_percent=70):
        """Get the list of neurons to activate based on specified parameters"""
        activated_neuron_labels, activated_neuron_ids, activated_neuron_idx = [], [], []
        
        # Determine which case to use
        case = self.get_case(activation_neuron_list, neuron_group_activation, activate_both_sides, 
                            activation_side, random_selection, random_selection_percent)

        if case == -1:
            return [], [], []

        # Case 2: Use provided neuron list
        if case == 2:
            try:
                activated_neuron_ids = activation_neuron_list
                activated_neuron_idx = [self.id_idx_dict.get(n) for n in activation_neuron_list if n in self.id_idx_dict]
                
                # Filter out any None values that might have resulted from missing keys
                activated_neuron_idx = [idx for idx in activated_neuron_idx if idx is not None]
                
                activated_neuron_labels = self.aud_label_root_id[
                    self.aud_label_root_id['root_id'].isin(activated_neuron_ids)
                ]['label'].to_list()
            except Exception as e:
                print(f'Error processing neuron list: {e}')
            
            return activated_neuron_labels, activated_neuron_ids, activated_neuron_idx

        # Handle cases 1 and 3
        if case in (1, 3):
            try:
                if case == 1:
                    # Both sides
                    left_key = f'{neuron_group_activation}_L'
                    right_key = f'{neuron_group_activation}_R'
                    
                    if left_key not in self.neurons_ranges or right_key not in self.neurons_ranges:
                        print(f"Missing neuron group: {left_key} or {right_key}")
                        return [], [], []
                    
                    all_neurons = self.neurons_ranges[left_key] + self.neurons_ranges[right_key]
                else:
                    # One side
                    key = f'{neuron_group_activation}_{activation_side}'
                    
                    if key not in self.neurons_ranges:
                        print(f"Missing neuron group: {key}")
                        return [], [], []
                    
                    all_neurons = self.neurons_ranges[key]

                # Apply random selection
                if random_selection and all_neurons:
                    amount_to_select = max(1, int((random_selection_percent / 100) * len(all_neurons)))
                    activated_neuron_idx = random.sample(all_neurons, amount_to_select)
                else:
                    activated_neuron_idx = all_neurons

                # Get IDs and labels
                activated_neuron_ids = [self.idx_id_dict.get(n) for n in activated_neuron_idx if n in self.idx_id_dict]
                activated_neuron_ids = [id for id in activated_neuron_ids if id is not None]
                
                activated_neuron_labels = self.aud_label_root_id[
                    self.aud_label_root_id['root_id'].isin(activated_neuron_ids)
                ]['label'].to_list()
            except Exception as e:
                print(f'Error selecting neurons: {e}')
                return [], [], []

        return activated_neuron_labels, activated_neuron_ids, activated_neuron_idx

    def save_simulation_data(self, spk_mon, state_mon, user_activated_indices, 
                           naturally_activated_indices, params, config):
        """Save simulation data to a timestamped folder"""
        # Create timestamp folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
        save_dir = os.path.join('simulation_results', timestamp)
        os.makedirs(save_dir, exist_ok=True)
        
        # Save spike data with more detailed information
        spike_data = pd.DataFrame({
            'neuron_index': spk_mon.i[:],
            'spike_time_ms': spk_mon.t[:] / ms,
            'neuron_id': [self.idx_id_dict.get(idx, 'Unknown') for idx in spk_mon.i[:]],
            'activation_type': ['user' if idx in user_activated_indices 
                              else 'natural' for idx in spk_mon.i[:]]
        })
        
        # Add neuron labels if available
        neuron_labels = []
        for idx in spk_mon.i[:]:
            neuron_id = self.idx_id_dict.get(idx, 'Unknown')
            if neuron_id != 'Unknown':
                label_row = self.aud_label_root_id[self.aud_label_root_id['root_id'] == neuron_id]
                if not label_row.empty:
                    neuron_labels.append(label_row['label'].iloc[0])
                else:
                    neuron_labels.append('Unknown')
            else:
                neuron_labels.append('Unknown')
        
        spike_data['neuron_label'] = neuron_labels
        spike_data.to_csv(os.path.join(save_dir, 'spikes.csv'), index=False)
        
        # Save state monitor data (voltage traces)
        if len(state_mon.t) > 0:
            state_data = pd.DataFrame({
                'time_ms': state_mon.t[:] / ms
            })
            
            # Add voltage data for each monitored neuron
            for i, neuron_idx in enumerate(state_mon.record):
                neuron_id = self.idx_id_dict.get(neuron_idx, 'Unknown')
                state_data[f'neuron_{neuron_idx}_id_{neuron_id}_voltage_mV'] = state_mon.v[i, :] / mV
            
            state_data.to_csv(os.path.join(save_dir, 'voltage_traces.csv'), index=False)
        
        # Save neuron activation info with labels
        user_activated_ids = [self.idx_id_dict.get(idx, 'Unknown') for idx in user_activated_indices]
        naturally_activated_ids = [self.idx_id_dict.get(idx, 'Unknown') for idx in naturally_activated_indices]
        
        # Get labels for activated neurons
        user_activated_labels = []
        for neuron_id in user_activated_ids:
            if neuron_id != 'Unknown':
                label_row = self.aud_label_root_id[self.aud_label_root_id['root_id'] == neuron_id]
                if not label_row.empty:
                    user_activated_labels.append(label_row['label'].iloc[0])
                else:
                    user_activated_labels.append('Unknown')
            else:
                user_activated_labels.append('Unknown')
        
        naturally_activated_labels = []
        for neuron_id in naturally_activated_ids:
            if neuron_id != 'Unknown':
                label_row = self.aud_label_root_id[self.aud_label_root_id['root_id'] == neuron_id]
                if not label_row.empty:
                    naturally_activated_labels.append(label_row['label'].iloc[0])
                else:
                    naturally_activated_labels.append('Unknown')
            else:
                naturally_activated_labels.append('Unknown')
        
        activation_df = pd.DataFrame({
            'neuron_index': list(user_activated_indices) + list(naturally_activated_indices),
            'neuron_id': user_activated_ids + naturally_activated_ids,
            'neuron_label': user_activated_labels + naturally_activated_labels,
            'activation_type': ['user'] * len(user_activated_indices) + 
                             ['natural'] * len(naturally_activated_indices)
        })
        activation_df.to_csv(os.path.join(save_dir, 'activated_neurons.csv'), index=False)
        
        # Save hyperparameters with better handling - FIX THE BRIAN2 UNITS ISSUE
        params_data = {}
        for key, value in params.items():
            if hasattr(value, 'dim'):  # Brian2 quantity
                if value.dim.is_dimensionless:
                    params_data[key] = float(value)
                else:
                    # Use the in_unit method to get the value with proper units
                    if str(value.dim) == 'second':
                        params_data[f'{key}_value'] = float(value / ms)
                        params_data[f'{key}_unit'] = 'ms'
                    elif str(value.dim) == 'volt':
                        params_data[f'{key}_value'] = float(value / mV)
                        params_data[f'{key}_unit'] = 'mV'
                    elif str(value.dim) == 'hertz':
                        params_data[f'{key}_value'] = float(value / Hz)
                        params_data[f'{key}_unit'] = 'Hz'
                    else:
                        params_data[f'{key}_value'] = float(value)
                        params_data[f'{key}_unit'] = str(value.dim)
            else:
                params_data[key] = value
        
        # Add configuration data with proper serialization
        params_data['config'] = config
        params_data['timestamp'] = timestamp
        params_data['total_neurons'] = len(self.aud_label_root_id)
        params_data['total_connections'] = len(self.aud_filtered_princeton)
        
        # Save as JSON for easy reading
        import json
        with open(os.path.join(save_dir, 'hyperparameters.json'), 'w') as f:
            json.dump(params_data, f, indent=2, default=str)
        
        # Also save as pickle for exact reconstruction
        with open(os.path.join(save_dir, 'hyperparameters.pickle'), 'wb') as f:
            pickle.dump({'params': params, 'config': config}, f)
        
        # Save summary statistics
        summary_stats = {
            'simulation_duration_ms': float(params['t_run'] / ms),
            'total_spikes': len(spk_mon.t),
            'user_activated_neurons': len(user_activated_indices),
            'naturally_activated_neurons': len(naturally_activated_indices),
            'unique_spiking_neurons': len(set(spk_mon.i[:])),
            'spike_rate_hz': len(spk_mon.t) / (float(params['t_run'] / ms) / 1000.0),
            'config_summary': {
                'left_active': config.get('left_active', False),
                'right_active': config.get('right_active', False),
                'neuron_groups_left': [g['neuron_group'] for g in config.get('left_groups', [])],
                'neuron_groups_right': [g['neuron_group'] for g in config.get('right_groups', [])]
            }
        }
        
        with open(os.path.join(save_dir, 'summary_stats.json'), 'w') as f:
            json.dump(summary_stats, f, indent=2)
        
        print(f"Simulation data saved to: {save_dir}")
        return save_dir

    def run_model(self, activation_neuron_list=[], neuron_group_activation='JO-A', 
                  activate_both_sides=True, activation_side='L', 
                  random_selection=True, random_selection_percent=70,
                  custom_params=None, save_data=False, config=None):
        """Run the connectomics model simulation"""
        # Update parameters with custom parameters if provided
        params = self.default_params.copy()
        if custom_params:
            params.update(custom_params)

        # Get activated neurons (these are the "user activated" neurons)
        activated_neuron_labels, activated_neuron_ids, activated_neuron_idx = self.get_excitatory_neurons(
            activation_neuron_list, neuron_group_activation, activate_both_sides, 
            activation_side, random_selection, random_selection_percent
        )
        
        if not activated_neuron_idx:
            raise ValueError("No neurons activated. Check your parameters.")

        # Run model setup and simulation
        start_scope()
        neu = NeuronGroup(
            N=len(self.aud_label_root_id),
            model=dedent('''
                dv/dt = (v_0 - v + g) / t_mbr : volt (unless refractory)
                dg/dt = -g / tau               : volt (unless refractory)
                rfc                            : second
            '''),
            method='linear',
            threshold='v > v_th',
            reset='v = v_rst; g = 0 * mV',
            refractory='rfc',
            namespace=params
        )
        neu.v = params['v_0']
        neu.g = 0
        neu.rfc = params['t_rfc']
        
        # Create synapses
        syn = Synapses(neu, neu, 'w : volt', on_pre='g += w', delay=params['t_dly'])
        
        i_pre = self.aud_filtered_princeton['Presynaptic_Index'].values
        i_post = self.aud_filtered_princeton['Postsynaptic_Index'].values
        syn.connect(i=i_pre, j=i_post)
        syn.w = self.aud_filtered_princeton['Signed_Connectivity'].values * params['w_syn']
        
        # Spike monitor
        spk_mon = SpikeMonitor(neu)
        
        # State monitor for voltage traces (monitor a subset of neurons to avoid memory issues)
        monitored_neurons = activated_neuron_idx[:min(20, len(activated_neuron_idx))]  # Monitor up to 20 neurons
        state_mon = StateMonitor(neu, 'v', record=monitored_neurons)
        
        # Poisson inputs for user activated neurons
        pois = []
        for i in activated_neuron_idx:
            p = PoissonInput(
                target=neu[i],
                target_var='v',
                N=1,
                rate=params['r_poi'],
                weight=params['w_syn'] * params['f_poi']
            )
            neu[i].rfc = 0 * ms
            pois.append(p)
        
        # Run network
        net = Network(neu, syn, spk_mon, state_mon, *pois)
        net.run(duration=params['t_run'])
        
        # Get neurons that fired as a result of the simulation
        all_fired_neurons = set(spk_mon.i[:])
        user_activated_neurons = set(activated_neuron_idx)
        naturally_activated_neurons = all_fired_neurons - user_activated_neurons

        # Save data if requested
        if save_data:
            save_dir = self.save_simulation_data(
                spk_mon, state_mon, user_activated_neurons, 
                naturally_activated_neurons, params, config
            )
        
        return spk_mon, state_mon, activated_neuron_labels, list(naturally_activated_neurons)

    def plot_spike_raster(self, spk_mon):
        """Plot spike raster diagram"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if len(spk_mon.t) > 0:
            ax.scatter(spk_mon.t/ms, spk_mon.i, color='blue', s=2)
            ax.set_ylabel('Neuron Index')
            ax.set_xlabel('Time (ms)')
            ax.set_title('Spiking Activity of Neurons')
        else:
            ax.text(0.5, 0.5, 'No spikes detected in this simulation run', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes)
        
        return fig

    def plot_combined_spike_raster(self, spike_monitors_list):
        """Plot combined spike raster diagram from multiple simulations
        
        Shows user activated neurons in different colors from naturally activated neurons
        """
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Define colors and markers for different types
        user_color = '#2E86AB'      # Blue for user activated
        natural_color = '#A23B72'   # Purple/magenta for naturally activated
        user_marker = 'o'           # Circle for user activated
        natural_marker = 's'        # Square for naturally activated
        user_size = 8               # Slightly larger for user activated
        natural_size = 6            # Slightly smaller for naturally activated
        
        # Check if we have any spikes to plot
        has_spikes = False
        
        # Keep track of labels for legend
        labels_used = set()
        
        # Collect all unique neuron indices that spiked for Y-axis labels
        all_spiking_indices = set()
        
        # Plot spikes from each monitor
        for i, [spk_mon, side, user_activated_neurons, naturally_activated_neurons] in enumerate(spike_monitors_list):
            if len(spk_mon.t) == 0:
                continue
                
            has_spikes = True
            
            # Create a mapping of all spikes and their neuron indices
            spike_indices = spk_mon.i[:]
            spike_times = spk_mon.t/ms
            
            # Collect all spiking indices
            all_spiking_indices.update(spike_indices)
            
            # Get mask for user activated neurons
            user_mask = np.isin(spike_indices, list(user_activated_neurons))
            
            # Get mask for naturally activated neurons
            natural_mask = np.isin(spike_indices, list(naturally_activated_neurons))
            
            # Plot user activated neurons with circles
            if any(user_mask):
                user_label = 'User Activated (Direct Stimulation)'
                ax.scatter(
                    spike_times[user_mask], 
                    spike_indices[user_mask], 
                    color=user_color, 
                    s=user_size, 
                    alpha=0.8, 
                    marker=user_marker,
                    edgecolors='white',
                    linewidth=0.5,
                    label=user_label if user_label not in labels_used else None,
                    zorder=3  # Plot on top
                )
                labels_used.add(user_label)
            
            # Plot naturally activated neurons with squares
            if any(natural_mask):
                natural_label = 'Naturally Activated (Network Response)'
                ax.scatter(
                    spike_times[natural_mask], 
                    spike_indices[natural_mask], 
                    color=natural_color, 
                    s=natural_size, 
                    alpha=0.7, 
                    marker=natural_marker,
                    edgecolors='white',
                    linewidth=0.3,
                    label=natural_label if natural_label not in labels_used else None,
                    zorder=2  # Plot below user activated
                )
                labels_used.add(natural_label)
        
        if has_spikes:
            ax.set_xlabel('Time (ms)', fontsize=12, fontweight='bold')
            ax.set_title('Neural Spiking Activity: Direct Stimulation vs Network Response', 
                        fontsize=14, fontweight='bold', pad=20)
            
            # Create Y-axis with neuron names instead of indices
            if all_spiking_indices:
                # Sort indices for consistent display
                sorted_indices = sorted(list(all_spiking_indices))
                
                # Get neuron names for Y-axis labels
                y_labels = []
                y_positions = []
                
                for idx in sorted_indices:
                    neuron_id = self.idx_id_dict.get(idx, 'Unknown')
                    if neuron_id != 'Unknown':
                        label_row = self.aud_label_root_id[self.aud_label_root_id['root_id'] == neuron_id]
                        if not label_row.empty:
                            neuron_name = label_row['label'].iloc[0]
                            # Truncate long names for readability
                            if len(neuron_name) > 20:
                                neuron_name = neuron_name[:17] + '...'
                            y_labels.append(f"{neuron_name}")
                        else:
                            y_labels.append(f"ID_{neuron_id}")
                    else:
                        y_labels.append(f"Idx_{idx}")
                    y_positions.append(idx)
                
                # Set Y-axis labels
                ax.set_ylabel('Neurons', fontsize=12, fontweight='bold')
                
                # If we have too many neurons, show only a subset
                if len(y_positions) > 50:
                    # Show every nth label to avoid overcrowding
                    step = max(1, len(y_positions) // 30)
                    selected_positions = y_positions[::step]
                    selected_labels = y_labels[::step]
                    ax.set_yticks(selected_positions)
                    ax.set_yticklabels(selected_labels, fontsize=8)
                else:
                    ax.set_yticks(y_positions)
                    ax.set_yticklabels(y_labels, fontsize=8)
                    
                # Rotate labels if needed
                plt.setp(ax.get_yticklabels(), rotation=0, ha='right')
            
            # Add legend if we have spikes - customize legend appearance
            if labels_used:
                legend = ax.legend(loc='upper right', frameon=True, fancybox=True, 
                                 shadow=True, fontsize=10, markerscale=1.5)
                legend.get_frame().set_facecolor('white')
                legend.get_frame().set_alpha(0.9)
                legend.get_frame().set_edgecolor('gray')
                
            # Add grid for better readability
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            
            # Improve overall aesthetics
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(0.5)
            ax.spines['bottom'].set_linewidth(0.5)
            
            # Add subtle background color
            ax.set_facecolor('#fafafa')
            
            # Adjust layout to prevent label cutoff
            plt.tight_layout()
        else:
            ax.text(0.5, 0.5, 'No spikes detected in simulation run', 
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=14, fontweight='bold')
            ax.set_ylabel('Neuron Index')
        
        return fig


# Example usage
if __name__ == '__main__':
    model = ConnectomicsModel()
    spk_mon, state_mon, labels, naturally_activated = model.run_model(save_data=True)
    plt.close('all')  # Prevent multiple plot windows