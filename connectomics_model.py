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
        if activation_neuron_list:
            return 2
        
        if neuron_group_activation not in self.neuron_groupings:
            print('Incorrect neuron group selected')
            return -1

        if not random_selection or not random_selection_percent:
            print('Random selection percentage not given')
            return -1

        return 1 if activate_both_sides else 3

    def get_excitatory_neurons(self, activation_neuron_list, neuron_group_activation, activate_both_sides, activation_side, random_selection, random_selection_percent):
        activated_neuron_labels, activated_neuron_ids, activated_neuron_idx = [], [], []
        case = self.get_case(activation_neuron_list, neuron_group_activation, activate_both_sides, activation_side, random_selection, random_selection_percent)

        if case == -1:
            return [], [], []

        if case == 2:
            if not activation_neuron_list:
                return [], [], []
            try:
                activated_neuron_ids = activation_neuron_list
                activated_neuron_idx = [self.id_idx_dict[n] for n in activation_neuron_list]
                activated_neuron_labels = self.aud_label_root_id[self.aud_label_root_id['root_id'].isin(activated_neuron_ids)]['label'].to_list()
            except KeyError as e:
                print(f'ID: {e.args[0]} is not present in the dictionary')
            return activated_neuron_labels, activated_neuron_ids, activated_neuron_idx

        # Handle cases 1 and 3
        if case in (1, 3):
            if case == 1:
                all_neurons = self.neurons_ranges[f'{neuron_group_activation}_L'] + self.neurons_ranges[f'{neuron_group_activation}_R']
            else:
                all_neurons = self.neurons_ranges[f'{neuron_group_activation}_{activation_side}']

            amount_to_select = int((random_selection_percent / 100) * len(all_neurons))
            activated_neuron_idx = random.sample(all_neurons, amount_to_select)
            activated_neuron_ids = [self.idx_id_dict[n] for n in activated_neuron_idx]
            activated_neuron_labels = self.aud_label_root_id[self.aud_label_root_id['root_id'].isin(activated_neuron_ids)]['label'].to_list()

        return activated_neuron_labels, activated_neuron_ids, activated_neuron_idx

    def run_model(self, activation_neuron_list=[], neuron_group_activation='JO-A', 
                  activate_both_sides=True, activation_side='L', 
                  random_selection=True, random_selection_percent=70,
                  custom_params=None):
        # Update parameters with custom parameters if provided
        params = self.default_params.copy()
        if custom_params:
            params.update(custom_params)

        # Get activated neurons
        activated_neuron_labels, activated_neuron_ids, activated_neuron_idx = self.get_excitatory_neurons(
            activation_neuron_list, neuron_group_activation, activate_both_sides, 
            activation_side, random_selection, random_selection_percent
        )

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
            reset='v = v_rst; w = 0; g = 0 * mV',
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
        
        return spk_mon, activated_neuron_labels

    def plot_spike_raster(self, spk_mon):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(spk_mon.t/ms, spk_mon.i, color='blue', s=2)
        ax.set_ylabel('Neuron Index')
        ax.set_xlabel('Time (ms)')
        ax.set_title('Spiking Activity of Neurons')
        return fig

    def plot_connection_graph(self, data_path='.'):
        adj_matrix = pd.read_csv(os.path.join(data_path, 'Updated_named_heatmap_AUD_JONs.csv'), index_col=1)
        G = nx.Graph(adj_matrix.iloc[:,1:])
        
        fig, ax = plt.subplots(figsize=(8, 6))
        nx.draw(G, with_labels=True, node_color='skyblue', node_size=2000,
                font_size=16, font_weight='bold', edge_color='gray', ax=ax)
        ax.set_title('Connection Graph')
        ax.axis('off')
        return fig

# Example usage
if __name__ == '__main__':
    model = ConnectomicsModel()
    spk_mon, labels = model.run_model()
    plt.close('all')  # Prevent multiple plot windows