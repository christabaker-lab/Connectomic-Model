import os
import io
import base64
import json
from flask import Flask, render_template, request, jsonify
import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from brian2 import mV, ms, Hz

# Import the ConnectomicsModel from the previous script
from connectomics_model import ConnectomicsModel

app = Flask(__name__)

# Initialize the model
model = ConnectomicsModel('data')

def fig_to_base64(fig):
    """Convert matplotlib figure to base64 encoded image"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)  # Ensure figure is closed properly
    return img_base64

@app.route('/', methods=['GET', 'POST'])
def index():
    # Get list of neuron groups for dropdown
    neuron_groups = model.neuron_groupings
    
    # Default parameters
    default_config = {
        'left_active': True,
        'right_active': True,
        'left_groups': [
            {
                'neuron_group': 'JO-A',
                'random_percent': 70,
                'r_poi': 250,
                'w_syn': 0.8
            }
        ],
        'right_groups': [
            {
                'neuron_group': 'JO-A',
                'random_percent': 70,
                'r_poi': 250,
                'w_syn': 0.8
            }
        ]
    }

    # Default model parameters with Brian2 units
    default_model_params = {
        't_run': 500 * ms,
        'tau': 5 * ms,
        'v_th': -45 * mV,
        't_rfc': 2.2 * ms,
    }

    if request.method == 'POST':
        try:
            # Extract hemisphere activation status
            left_active = request.form.get('left_active') == 'on'
            right_active = request.form.get('right_active') == 'on'
            
            if not (left_active or right_active):
                return render_template('index.html', 
                                    neuron_groups=neuron_groups, 
                                    error="At least one hemisphere must be activated",
                                    config=default_config,
                                    model_params=default_model_params,
                                    ms=ms, 
                                    mV=mV,
                                    Hz=Hz)
            
            # Extract global model parameters
            global_params = {
                't_run': float(request.form.get('t_run', default_model_params['t_run'] / ms)) * ms,
                'tau': float(request.form.get('tau', default_model_params['tau'] / ms)) * ms,
                'v_th': float(request.form.get('v_th', default_model_params['v_th'] / mV)) * mV,
                't_rfc': float(request.form.get('t_rfc', default_model_params['t_rfc'] / ms)) * ms,
            }
            
            # Process left hemisphere groups
            left_groups = []
            if left_active:
                # Extract data from the dynamic form fields
                left_group_count = 0
                while f'left_groups[{left_group_count}][neuron_group]' in request.form:
                    group_data = {
                        'neuron_group': request.form.get(f'left_groups[{left_group_count}][neuron_group]'),
                        'random_percent': float(request.form.get(f'left_groups[{left_group_count}][random_percent]', 70)),
                        'r_poi': float(request.form.get(f'left_groups[{left_group_count}][r_poi]', 250)) * Hz,
                        'w_syn': float(request.form.get(f'left_groups[{left_group_count}][w_syn]', 0.8)) * mV,
                    }
                    left_groups.append(group_data)
                    left_group_count += 1
            
            # Process right hemisphere groups
            right_groups = []
            if right_active:
                # Extract data from the dynamic form fields
                right_group_count = 0
                while f'right_groups[{right_group_count}][neuron_group]' in request.form:
                    group_data = {
                        'neuron_group': request.form.get(f'right_groups[{right_group_count}][neuron_group]'),
                        'random_percent': float(request.form.get(f'right_groups[{right_group_count}][random_percent]', 70)),
                        'r_poi': float(request.form.get(f'right_groups[{right_group_count}][r_poi]', 250)) * Hz,
                        'w_syn': float(request.form.get(f'right_groups[{right_group_count}][w_syn]', 0.8)) * mV,
                    }
                    right_groups.append(group_data)
                    right_group_count += 1
            
            # Run the simulation with all configured groups
            all_activated_neurons = []
            left_activated_neurons = []
            right_activated_neurons = []
            all_spikes = []
            
            # Collect downstream activated neurons
            left_downstream_neurons = []
            right_downstream_neurons = []
            all_downstream_neurons = []
            
            # Function to run simulation for a specific group and side
            def run_group_simulation(group_data, side):
                # Create custom parameters for this specific group
                custom_params = global_params.copy()
                custom_params.update({
                    'r_poi': group_data['r_poi'],
                    'w_syn': group_data['w_syn']
                })
                
                # Run the model for this group
                spk_mon, labels, downstream_fired_neurons = model.run_model(
                    neuron_group_activation=group_data['neuron_group'],
                    activate_both_sides=False,
                    activation_side=side,
                    random_selection=True,
                    random_selection_percent=group_data['random_percent'],
                    custom_params=custom_params
                )
                
                return spk_mon, labels, downstream_fired_neurons
            
            # Run simulations for all left groups
            for group_data in left_groups:
                spk_mon, labels, downstream_neurons = run_group_simulation(group_data, 'L')
                left_activated_neurons.extend(labels)
                all_activated_neurons.extend(labels)
                
                # Get indices for all activated neurons in this simulation
                _, _, activated_indices = model.get_excitatory_neurons(
                    neuron_group_activation=group_data['neuron_group'],
                    activate_both_sides=False,
                    activation_side='L',
                    random_selection=True,
                    random_selection_percent=group_data['random_percent']
                )
                
                # Store downstream neurons
                left_downstream_neurons.extend([model.idx_id_dict.get(idx) for idx in downstream_neurons if idx in model.idx_id_dict])
                all_downstream_neurons.extend([model.idx_id_dict.get(idx) for idx in downstream_neurons if idx in model.idx_id_dict])
                
                # Store spikes with activated neuron indices
                all_spikes.append([spk_mon, 'L', set(activated_indices), set(downstream_neurons)])
            
            # Run simulations for all right groups
            for group_data in right_groups:
                spk_mon, labels, downstream_neurons = run_group_simulation(group_data, 'R')
                right_activated_neurons.extend(labels)
                all_activated_neurons.extend(labels)
                
                # Get indices for all activated neurons in this simulation
                _, _, activated_indices = model.get_excitatory_neurons(
                    neuron_group_activation=group_data['neuron_group'],
                    activate_both_sides=False,
                    activation_side='R',
                    random_selection=True,
                    random_selection_percent=group_data['random_percent']
                )
                
                # Store downstream neurons
                right_downstream_neurons.extend([model.idx_id_dict.get(idx) for idx in downstream_neurons if idx in model.idx_id_dict])
                all_downstream_neurons.extend([model.idx_id_dict.get(idx) for idx in downstream_neurons if idx in model.idx_id_dict])
                
                # Store spikes with activated neuron indices
                all_spikes.append([spk_mon, 'R', set(activated_indices), set(downstream_neurons)])
            
            # Generate combined spike raster plot from all simulations
            combined_spike_raster_fig = model.plot_combined_spike_raster(all_spikes)
            spike_raster_base64 = fig_to_base64(combined_spike_raster_fig)
            plt.close(combined_spike_raster_fig)

            # Generate connection graph
            connection_graph_fig = model.plot_connection_graph(data_path='data')
            connection_graph_base64 = fig_to_base64(connection_graph_fig)
            plt.close(connection_graph_fig)

            # Create config object for template rendering
            config = {
                'left_active': left_active,
                'right_active': right_active,
                'left_groups': left_groups,
                'right_groups': right_groups
            }

            # Get downstream neuron labels
            left_downstream_labels = model.aud_label_root_id[
                model.aud_label_root_id['root_id'].isin(left_downstream_neurons)
            ]['label'].tolist() if left_downstream_neurons else []
            
            right_downstream_labels = model.aud_label_root_id[
                model.aud_label_root_id['root_id'].isin(right_downstream_neurons)
            ]['label'].tolist() if right_downstream_neurons else []

            return render_template('index.html', 
                                neuron_groups=neuron_groups, 
                                config=config, 
                                model_params=global_params,
                                spike_raster=spike_raster_base64,
                                connection_graph=connection_graph_base64,
                                left_activated_neurons=left_activated_neurons,
                                right_activated_neurons=right_activated_neurons,
                                left_downstream_neurons=left_downstream_labels,
                                right_downstream_neurons=right_downstream_labels,
                                ms=ms, 
                                mV=mV,
                                Hz=Hz)
                                
        except Exception as e:
            return render_template('index.html', 
                                neuron_groups=neuron_groups, 
                                error=str(e),
                                config=default_config,
                                model_params=default_model_params,
                                ms=ms, 
                                mV=mV,
                                Hz=Hz)

    return render_template('index.html', 
                        neuron_groups=neuron_groups, 
                        config=default_config,
                        model_params=default_model_params, 
                        ms=ms, 
                        mV=mV,
                        Hz=Hz)
if __name__ == '__main__':
    app.run(debug=True)