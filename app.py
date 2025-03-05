import os
import io
import base64
from flask import Flask, render_template, request, jsonify
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
    return base64.b64encode(buf.getvalue()).decode('utf-8')

@app.route('/', methods=['GET', 'POST'])
def index():
    # Get list of neuron groups for dropdown
    neuron_groups = model.neuron_groupings
    
    # Default parameters
    default_config = {
        'neuron_group_activation': 'JO-A',
        'activate_both_sides': True,
        'activation_side': 'L',
        'random_selection': True,
        'random_selection_percent': 70,
        'activation_neuron_list': []
    }

    # Default model parameters
    default_model_params = {
        't_run': 500,  # ms
        'r_poi': 250,  # Hz
        'w_syn': 0.8,  # mV
        'tau': 5,      # ms
    }

    if request.method == 'POST':
        # Collect neuron activation parameters
        neuron_config = {
            'neuron_group_activation': request.form.get('neuron_group', default_config['neuron_group_activation']),
            'activate_both_sides': request.form.get('both_sides') == 'on',
            'activation_side': request.form.get('side', default_config['activation_side']),
            'random_selection': request.form.get('random_selection') == 'on',
            'random_selection_percent': float(request.form.get('random_percent', default_config['random_selection_percent'])),
        }

        # Collect custom model parameters
        custom_params = {
            't_run': float(request.form.get('t_run', default_model_params['t_run'])) * ms,
            'r_poi': float(request.form.get('r_poi', default_model_params['r_poi'])) * Hz,
            'w_syn': float(request.form.get('w_syn', default_model_params['w_syn'])) * mV,
            'tau': float(request.form.get('tau', default_model_params['tau'])) * ms,
        }

        # Run model and generate plots
        try:
            spk_mon, labels = model.run_model(**neuron_config, custom_params=custom_params)
            
            # Generate spike raster plot
            spike_raster_fig = model.plot_spike_raster(spk_mon)
            spike_raster_base64 = fig_to_base64(spike_raster_fig)
            plt.close(spike_raster_fig)

            # Generate connection graph
            connection_graph_fig = model.plot_connection_graph(data_path='data')
            connection_graph_base64 = fig_to_base64(connection_graph_fig)
            plt.close(connection_graph_fig)

            return render_template('index.html', 
                                   neuron_groups=neuron_groups, 
                                   config=neuron_config, 
                                   model_params=custom_params,
                                   spike_raster=spike_raster_base64,
                                   connection_graph=connection_graph_base64,
                                   activated_neurons=labels, 
                                   ms=ms, 
                                   mV=mV,
                                   Hz=Hz)
        except Exception as e:
            return render_template('index.html', 
                                   neuron_groups=neuron_groups, 
                                   error=str(e),
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