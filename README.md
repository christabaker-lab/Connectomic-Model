# Connectomics Model Simulation

## Setup and Installation

1. (One time step) Clone the repository 
- Go to the folder where you want to download the project
- Right click in the folder and select "Open Terminal".
- Run the git clone command with the git repo link: ```git clone https://github.com/christabaker-lab/Connectomic-Model.git```
- In the terminal, run the command to change directory into the recently downloaded project: ```cd Connectomic-Model```

2. (One time step) Create a virtual environment, this create a private directory to handle all the code and its dependencies. You should be able to see "(venv)" in the Terminal, before the folder structure.
```bash
python3 -m venv venv
.\venv\Scripts\activate
```

3. (One time step) Install requirements and dependencies
```bash
pip3 install -r requirements.txt
```

### Everything after this needs to be done every time to start the model

1. Open Terminal inside the Connectomic-Model folder (ignore if you are currently in the installation process)
2. Activate the environment
```bash
.\venv\Scripts\activate
```
3. Run the Flask application
```bash
python app.py
```
-> If this doesnt work, try out 
```bash
python3 app.py
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
