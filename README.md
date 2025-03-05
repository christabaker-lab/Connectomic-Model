# Connectomics Model Simulation

## Setup and Installation

1. Clone the repository
2. Create a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install requirements
```bash
pip install -r requirements.txt
```

4. Run the Flask application
```bash
python app.py
```

## Project Structure
- `connectomics_model.py`: Core model implementation
- `app.py`: Flask web application
- `templates/index.html`: Web interface template
- Required data files:
  - `aud_label_root_id.csv`
  - `aud_filtered_princeton.csv`
  - `id_idx_dict.pickle`
  - `idx_id_dict.pickle`
  - `neuron_ranges.pickle`
  - `neuron_groupings.pickle`
  - `Updated_named_heatmap_AUD_JONs.csv`

## Features
- Interactive neuron group selection
- Toggle activation for both/single brain sides
- Random neuron selection
- Customizable model parameters
- Spike raster and connection graph visualization