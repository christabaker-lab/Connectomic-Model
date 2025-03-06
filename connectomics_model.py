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

from brian2 import NeuronGroup, Synapses, PoissonInput, SpikeMonitor, Network, TimedArray, start_scope
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

    def run_model(self, activation_neuron_list=[], neuron_group_activation='JO-A', 
                  activate_both_sides=True, activation_side='L', 
                  random_selection=True, random_selection_percent=70,
                  custom_params=None):
        """Run the connectomics model simulation"""
        # Update parameters with custom parameters if provided
        params = self.default_params.copy()
        if custom_params:
            params.update(custom_params)

        # Get activated neurons
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
        
        # Poisson inputs
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
        net = Network(neu, syn, spk_mon, *pois)
        net.run(duration=params['t_run'])
        
        
        # Get neurons that fired as a result of the simulation
        all_fired_neurons = set(spk_mon.i[:])

        # Determine neurons that fired **due to synaptic transmission**
        downstream_fired_neurons = all_fired_neurons - set(activated_neuron_idx)

        return spk_mon, activated_neuron_labels, list(downstream_fired_neurons)
        # return spk_mon, activated_neuron_labels

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

    def plot_connection_graph(self, data_path='.'):
        """Plot connection graph between neurons"""
        try:
            adj_matrix = pd.read_csv(os.path.join(data_path, 'Updated_named_heatmap_AUD_JONs.csv'), index_col=1)
            G = nx.Graph()
            
            # Add nodes
            for node in adj_matrix.index:
                G.add_node(node)
            
            # Add edges with weights from the adjacency matrix
            for i, row in enumerate(adj_matrix.index):
                for j, col in enumerate(adj_matrix.columns[1:], 1):
                    weight = adj_matrix.iloc[i, j]
                    if weight > 0:
                        G.add_edge(row, adj_matrix.columns[j], weight=weight)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Adjust layout and node sizes based on graph size
            if len(G.nodes) > 20:
                layout = nx.spring_layout(G)
                node_size = 1000
                font_size = 8
            else:
                layout = nx.spring_layout(G, k=0.5)
                node_size = 2000
                font_size = 10
            
            nx.draw(G, pos=layout, with_labels=True, 
                    node_color='skyblue', node_size=node_size,
                    font_size=font_size, font_weight='bold', 
                    edge_color='gray', width=0.5, ax=ax)
            
            ax.set_title('Connection Graph')
            ax.axis('off')
            
            return fig
        except FileNotFoundError:
            # Create an empty figure with a message if the data file is not found
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, 'Connection graph data not available', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes)
            ax.axis('off')
            return fig
        
        # Add to connectomics_model.py

    def plot_combined_spike_raster(self, spike_monitors_list):
        """Plot combined spike raster diagram from multiple simulations
        
        Shows direct activated neurons in different colors from downstream activated neurons
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Define a color cycle for different simulation runs
        side_colors = {'L': 'blue', 'R': 'red'}
        
        # Check if we have any spikes to plot
        has_spikes = False
        
        # Keep track of labels for legend
        labels_used = set()
        
        # Plot spikes from each monitor
        for i, [spk_mon, side, activated_neurons, downstream_neurons] in enumerate(spike_monitors_list):
            if len(spk_mon.t) == 0:
                continue
                
            has_spikes = True
            
            # Create a mapping of all spikes and their neuron indices
            spike_indices = spk_mon.i[:]
            spike_times = spk_mon.t/ms
            
            # Get mask for directly activated neurons
            direct_mask = np.isin(spike_indices, list(activated_neurons))
            
            # Get mask for downstream activated neurons
            downstream_mask = np.isin(spike_indices, list(downstream_neurons))
            
            # Plot directly activated neurons
            if any(direct_mask):
                direct_label = f'Direct {side}'
                ax.scatter(
                    spike_times[direct_mask], 
                    spike_indices[direct_mask], 
                    color=side_colors[side], 
                    s=4, 
                    alpha=0.9, 
                    marker='o',
                    label=direct_label if direct_label not in labels_used else None
                )
                labels_used.add(direct_label)
            
            # Plot downstream activated neurons
            if any(downstream_mask):
                downstream_label = f'Downstream {side}'
                ax.scatter(
                    spike_times[downstream_mask], 
                    spike_indices[downstream_mask], 
                    color=side_colors[side], 
                    s=3, 
                    alpha=0.6, 
                    marker='x',
                    label=downstream_label if downstream_label not in labels_used else None
                )
                labels_used.add(downstream_label)
        
        if has_spikes:
            ax.set_ylabel('Neuron Index')
            ax.set_xlabel('Time (ms)')
            ax.set_title('Spiking Activity: Direct vs. Downstream Neurons')
            
            # Add legend if we have spikes
            if labels_used:
                ax.legend(loc='upper right')
        else:
            ax.text(0.5, 0.5, 'No spikes detected in any simulation run', 
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes)
        
        return fig


# Example usage
if __name__ == '__main__':
    model = ConnectomicsModel()
    spk_mon, labels = model.run_model()
    plt.close('all')  # Prevent multiple plot windows