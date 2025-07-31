import os
import io
import base64
import json
import tempfile
import uuid
import time
from flask import Flask, render_template, request, jsonify, redirect, url_for, session, send_file
import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from brian2 import mV, ms, Hz

# Import the ConnectomicsModel from the previous script
from connectomics_model import ConnectomicsModel

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Required for sessions

# Initialize the model
model = ConnectomicsModel('data')

def cleanup_old_plots():
    """Clean up plot files older than 1 hour"""
    try:
        for filename in os.listdir('simulation_results'):
            if filename.startswith('temp_plot_'):
                filepath = os.path.join('simulation_results', filename)
                if os.path.getmtime(filepath) < time.time() - 3600:  # 1 hour
                    os.remove(filepath)
                    print(f"Cleaned up old plot: {filename}")
    except Exception as e:
        print(f"Error cleaning up plots: {e}")

@app.route('/plot/<filename>')
def serve_plot(filename):
    """Serve temporary plot files"""
    try:
        # Security check - only allow temp_plot_ files
        if not filename.startswith('temp_plot_'):
            return "Invalid filename", 400
            
        plot_path = os.path.join('simulation_results', filename)
        if os.path.exists(plot_path):
            return send_file(plot_path, mimetype='image/png')
        else:
            return "Plot not found", 404
    except Exception as e:
        return f"Error serving plot: {str(e)}", 500

@app.route('/', methods=['GET'])
def index():
    """GET route - only shows the form, never runs simulation"""
    print("GET request to / - showing form only, NO simulation")
    
    # Get list of neuron groups for dropdown
    neuron_groups = model.neuron_groupings
    
    # Default configuration
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
    
    # Check for error messages first
    error_config = session.pop('error_config', None)
    error_model_params = session.pop('error_model_params', None)
    error_message = session.pop('error', None)
    
    if error_message:
        print(f"Displaying error: {error_message}")
        return render_template('index.html',
                              neuron_groups=neuron_groups,
                              config=error_config or default_config,
                              model_params={
                                  't_run': error_model_params['t_run'] * ms,
                                  'tau': error_model_params['tau'] * ms,
                                  'v_th': error_model_params['v_th'] * mV,
                                  't_rfc': error_model_params['t_rfc'] * ms
                              } if error_model_params else default_model_params,
                              error=error_message,
                              simulation_run=False,
                              ms=ms, mV=mV, Hz=Hz)
    
    # Check if we have simulation results from session (after redirect)
    simulation_results = session.pop('simulation_results', None)
    
    if simulation_results:
        print("Displaying simulation results from session")
        return render_template('index.html', 
                              neuron_groups=neuron_groups, 
                              config=simulation_results['config'],
                              model_params={
                                  't_run': simulation_results['model_params']['t_run'] * ms,
                                  'tau': simulation_results['model_params']['tau'] * ms,
                                  'v_th': simulation_results['model_params']['v_th'] * mV,
                                  't_rfc': simulation_results['model_params']['t_rfc'] * ms
                              },
                              plot_filename=simulation_results['plot_filename'],
                              left_activated_neurons=simulation_results['left_activated_neurons'],
                              right_activated_neurons=simulation_results['right_activated_neurons'],
                              left_naturally_activated_neurons=simulation_results['left_naturally_activated_neurons'],
                              right_naturally_activated_neurons=simulation_results['right_naturally_activated_neurons'],
                              all_activated_neurons=simulation_results['all_activated_neurons'],
                              all_naturally_activated_neurons=simulation_results['all_naturally_activated_neurons'],
                              simulation_run=True,
                              ms=ms, 
                              mV=mV,
                              Hz=Hz)
    else:
        print("Showing default form")
        return render_template('index.html', 
                              neuron_groups=neuron_groups, 
                              config=default_config,
                              model_params=default_model_params,
                              simulation_run=False,
                              ms=ms, 
                              mV=mV,
                              Hz=Hz)

@app.route('/run_simulation', methods=['POST'])
def run_simulation():
    """POST route - processes form and runs simulation, then redirects"""
    print("="*60)
    print("POST request to /run_simulation - RUNNING SIMULATION")
    print("="*60)
    
    # Clean up old plots at the start of new simulation
    cleanup_old_plots()
    
    # Get list of neuron groups for dropdown
    neuron_groups = model.neuron_groupings
    
    # Default model parameters
    default_model_params = {
        't_run': 500 * ms,
        'tau': 5 * ms,
        'v_th': -45 * mV,
        't_rfc': 2.2 * ms,
    }
    
    try:
        # Extract hemisphere activation status
        left_active = request.form.get('left_active') == 'on'
        right_active = request.form.get('right_active') == 'on'
        
        print(f"Hemisphere activation - Left: {left_active}, Right: {right_active}")
        
        if not (left_active or right_active):
            session['error'] = "At least one hemisphere must be activated"
            return redirect(url_for('index'))
        
        # Extract global model parameters - preserve user input
        global_params = {
            't_run': float(request.form.get('t_run', default_model_params['t_run'] / ms)) * ms,
            'tau': float(request.form.get('tau', default_model_params['tau'] / ms)) * ms,
            'v_th': float(request.form.get('v_th', default_model_params['v_th'] / mV)) * mV,
            't_rfc': float(request.form.get('t_rfc', default_model_params['t_rfc'] / ms)) * ms,
        }
        
        print(f"Global params: t_run={global_params['t_run']}, tau={global_params['tau']}")
        
        # Process left hemisphere groups
        left_groups = []
        if left_active:
            left_group_count = 0
            while f'left_groups[{left_group_count}][neuron_group]' in request.form:
                group_data = {
                    'neuron_group': request.form.get(f'left_groups[{left_group_count}][neuron_group]'),
                    'random_percent': float(request.form.get(f'left_groups[{left_group_count}][random_percent]', 70)),
                    'r_poi': float(request.form.get(f'left_groups[{left_group_count}][r_poi]', 250)),
                    'w_syn': float(request.form.get(f'left_groups[{left_group_count}][w_syn]', 0.8)),
                }
                left_groups.append(group_data)
                left_group_count += 1
            print(f"Left groups configured: {len(left_groups)}")
        
        # Process right hemisphere groups
        right_groups = []
        if right_active:
            right_group_count = 0
            while f'right_groups[{right_group_count}][neuron_group]' in request.form:
                group_data = {
                    'neuron_group': request.form.get(f'right_groups[{right_group_count}][neuron_group]'),
                    'random_percent': float(request.form.get(f'right_groups[{right_group_count}][random_percent]', 70)),
                    'r_poi': float(request.form.get(f'right_groups[{right_group_count}][r_poi]', 250)),
                    'w_syn': float(request.form.get(f'right_groups[{right_group_count}][w_syn]', 0.8)),
                }
                right_groups.append(group_data)
                right_group_count += 1
            print(f"Right groups configured: {len(right_groups)}")

        # Check if we have any groups to simulate
        if not left_groups and not right_groups:
            session['error'] = "No neuron groups configured for simulation"
            return redirect(url_for('index'))
        
        print(f"Running SINGLE COMBINED simulation with {len(left_groups)} left groups and {len(right_groups)} right groups")
        
        # COLLECT ALL NEURONS FROM ALL GROUPS FOR SINGLE SIMULATION
        all_activated_neuron_indices = []
        all_activated_neuron_labels = []
        config_summary = {
            'left_active': left_active,
            'right_active': right_active,
            'left_groups': left_groups,
            'right_groups': right_groups
        }
        
        # Process all left groups and collect their neurons (NO individual simulations)
        for group_data in left_groups:
            print(f"Collecting neurons from left group: {group_data['neuron_group']}")
            _, _, user_activated_indices = model.get_excitatory_neurons(
                neuron_group_activation=group_data['neuron_group'],
                activate_both_sides=False,
                activation_side='L',
                random_selection=True,
                random_selection_percent=group_data['random_percent']
            )
            all_activated_neuron_indices.extend(user_activated_indices)
            
            # Get labels for these neurons
            activated_ids = [model.idx_id_dict.get(idx) for idx in user_activated_indices if idx in model.idx_id_dict]
            activated_ids = [id for id in activated_ids if id is not None]
            labels = model.aud_label_root_id[
                model.aud_label_root_id['root_id'].isin(activated_ids)
            ]['label'].to_list()
            all_activated_neuron_labels.extend(labels)
        
        # Process all right groups and collect their neurons (NO individual simulations)
        for group_data in right_groups:
            print(f"Collecting neurons from right group: {group_data['neuron_group']}")
            _, _, user_activated_indices = model.get_excitatory_neurons(
                neuron_group_activation=group_data['neuron_group'],
                activate_both_sides=False,
                activation_side='R',
                random_selection=True,
                random_selection_percent=group_data['random_percent']
            )
            all_activated_neuron_indices.extend(user_activated_indices)
            
            # Get labels for these neurons
            activated_ids = [model.idx_id_dict.get(idx) for idx in user_activated_indices if idx in model.idx_id_dict]
            activated_ids = [id for id in activated_ids if id is not None]
            labels = model.aud_label_root_id[
                model.aud_label_root_id['root_id'].isin(activated_ids)
            ]['label'].to_list()
            all_activated_neuron_labels.extend(labels)
        
        # Remove duplicates while preserving order
        all_activated_neuron_indices = list(dict.fromkeys(all_activated_neuron_indices))
        all_activated_neuron_labels = list(dict.fromkeys(all_activated_neuron_labels))
        
        print(f"Total unique neurons collected for SINGLE simulation: {len(all_activated_neuron_indices)}")
        
        # Convert all activated neuron indices to IDs for the model
        all_activated_neuron_ids = [model.idx_id_dict.get(idx) for idx in all_activated_neuron_indices if idx in model.idx_id_dict]
        all_activated_neuron_ids = [id for id in all_activated_neuron_ids if id is not None]
        
        # Calculate weighted averages for parameters
        all_groups = left_groups + right_groups
        avg_r_poi = sum(g['r_poi'] for g in all_groups) / len(all_groups) if all_groups else 250.0
        avg_w_syn = sum(g['w_syn'] for g in all_groups) / len(all_groups) if all_groups else 0.8
        
        # Create unified parameters for the single simulation
        unified_params = global_params.copy()
        unified_params.update({
            'r_poi': avg_r_poi * Hz,
            'w_syn': avg_w_syn * mV
        })
        
        print(f"Running SINGLE simulation with avg_r_poi={avg_r_poi}, avg_w_syn={avg_w_syn}")
        print(f"Total neurons to activate: {len(all_activated_neuron_ids)}")
        
        # RUN ONE SINGLE SIMULATION WITH ALL COLLECTED NEURONS
        spk_mon, state_mon, _, naturally_activated_neurons = model.run_model(
            activation_neuron_list=all_activated_neuron_ids,  # Use collected IDs directly
            custom_params=unified_params,
            save_data=True,  # SAVES ONE SET OF FILES
            config=config_summary
        )
        
        print(f"Simulation completed. Total spikes: {len(spk_mon.t) if spk_mon.t is not None else 0}")
        
        # Separate left and right results for display
        left_activated_neurons = []
        right_activated_neurons = []
        
        # Categorize activated neurons by hemisphere based on which groups they came from
        for i, label in enumerate(all_activated_neuron_labels):
            # This is a simplified categorization - you might need to adjust based on your data structure
            found_in_left = False
            found_in_right = False
            
            # Check if this neuron came from left groups
            for group_data in left_groups:
                if group_data['neuron_group'] in label:
                    left_activated_neurons.append(label)
                    found_in_left = True
                    break
            
            # Check if this neuron came from right groups (and not already added to left)
            if not found_in_left:
                for group_data in right_groups:
                    if group_data['neuron_group'] in label:
                        right_activated_neurons.append(label)
                        found_in_right = True
                        break
            
            # If we can't determine, add based on active hemispheres
            if not found_in_left and not found_in_right:
                if left_active and right_active:
                    # Split evenly or use some other logic
                    if i % 2 == 0:
                        left_activated_neurons.append(label)
                    else:
                        right_activated_neurons.append(label)
                elif left_active:
                    left_activated_neurons.append(label)
                elif right_active:
                    right_activated_neurons.append(label)
        
        # Get naturally activated neuron labels
        naturally_activated_ids = [model.idx_id_dict.get(idx) for idx in naturally_activated_neurons if idx in model.idx_id_dict]
        naturally_activated_labels = model.aud_label_root_id[
            model.aud_label_root_id['root_id'].isin(naturally_activated_ids)
        ]['label'].tolist() if naturally_activated_ids else []
        
        # For display purposes, split naturally activated by hemisphere (simplified)
        left_naturally_activated_neurons = []
        right_naturally_activated_neurons = []
        for label in naturally_activated_labels:
            if left_active and right_active:
                # Split evenly
                if len(left_naturally_activated_neurons) <= len(right_naturally_activated_neurons):
                    left_naturally_activated_neurons.append(label)
                else:
                    right_naturally_activated_neurons.append(label)
            elif left_active:
                left_naturally_activated_neurons.append(label)
            elif right_active:
                right_naturally_activated_neurons.append(label)
        
        # Generate spike raster plot - create a single entry for the combined simulation
        all_spikes = [[spk_mon, 'Both', set(all_activated_neuron_indices), set(naturally_activated_neurons)]]
        combined_spike_raster_fig = model.plot_combined_spike_raster(all_spikes)
        
        # Generate unique filename for this simulation plot
        plot_filename = f"temp_plot_{uuid.uuid4().hex}.png"
        plot_path = os.path.join('simulation_results', plot_filename)
        
        # Save plot directly to file instead of base64 to avoid session size issues
        combined_spike_raster_fig.savefig(plot_path, format='png', bbox_inches='tight', dpi=100)
        plt.close(combined_spike_raster_fig)
        
        print(f"Plot saved to: {plot_path}")

        # Create config object for template rendering (preserve user state)
        config = {
            'left_active': left_active,
            'right_active': right_active,
            'left_groups': left_groups,
            'right_groups': right_groups
        }

        # Store results in session for redirect - NO LARGE BASE64 DATA
        session['simulation_results'] = {
            'config': config,
            'model_params': {
                't_run': float(global_params['t_run'] / ms),
                'tau': float(global_params['tau'] / ms),
                'v_th': float(global_params['v_th'] / mV),
                't_rfc': float(global_params['t_rfc'] / ms)
            },
            'plot_filename': plot_filename,  # Store filename instead of base64 data
            'left_activated_neurons': left_activated_neurons,
            'right_activated_neurons': right_activated_neurons,
            'left_naturally_activated_neurons': left_naturally_activated_neurons,
            'right_naturally_activated_neurons': right_naturally_activated_neurons,
            'all_activated_neurons': all_activated_neuron_labels,
            'all_naturally_activated_neurons': naturally_activated_labels
        }
        
        print("Simulation results stored in session, redirecting to prevent resubmission")
        
        # REDIRECT TO PREVENT FORM RESUBMISSION (Post-Redirect-Get pattern)
        return redirect(url_for('index'))

    except Exception as e:
        print(f"Simulation error: {str(e)}")
        session['error'] = f"Simulation error: {str(e)}"
        
        # Store user input in session to preserve form state on error - CONVERT BRIAN2 UNITS
        session['error_config'] = {
            'left_active': request.form.get('left_active') == 'on',
            'right_active': request.form.get('right_active') == 'on',
            'left_groups': left_groups if 'left_groups' in locals() else [],
            'right_groups': right_groups if 'right_groups' in locals() else []
        }
        
        # Store model parameters from form - CONVERT BRIAN2 UNITS TO PLAIN NUMBERS
        session['error_model_params'] = {
            't_run': float(request.form.get('t_run', default_model_params['t_run'] / ms)),
            'tau': float(request.form.get('tau', default_model_params['tau'] / ms)),
            'v_th': float(request.form.get('v_th', default_model_params['v_th'] / mV)),
            't_rfc': float(request.form.get('t_rfc', default_model_params['t_rfc'] / ms))
        }
        
        return redirect(url_for('index'))

if __name__ == '__main__':
    # Create simulation results directory if it doesn't exist
    os.makedirs('simulation_results', exist_ok=True)
    
    # Clean up old plots on startup
    cleanup_old_plots()
    
    app.run(debug=True)