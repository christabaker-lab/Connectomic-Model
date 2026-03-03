# Connectomics Simulation Pipeline: Technical Reference and Documentation

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture and Data Flow](#2-system-architecture-and-data-flow)
3. [File-by-File Reference](#3-file-by-file-reference)
   - [connectomic_preprocessing.py](#31-connectomic_preprocessingpy)
   - [connectomics_model.py](#32-connectomics_modelpy)
   - [new_model.py](#33-new_modelpy)
   - [generate_configs.py](#34-generate_configspy)
   - [submitSimulationAssay.py](#35-submitsimulationassaypy)
   - [submitjobs.sh](#36-submitjobssh)
   - [run_postprocessing.py](#37-run_postprocessingpy)
4. [Data File Reference](#4-data-file-reference)
5. [Configuration Schema Reference](#5-configuration-schema-reference)
6. [HPC Deployment Guide](#6-hpc-deployment-guide)

---

## 1. Project Overview

This pipeline implements a large-scale, biologically grounded spiking neural network simulation of the *Drosophila melanogaster* auditory system. It is built on the FlyWire connectome dataset, a nanometer-resolution electron microscopy reconstruction of the full adult fly brain (FAFB), from which every synaptic connection between neurons is known. The pipeline simulates how auditory stimuli, applied to Johnston's Organ (JO) neurons, propagate through downstream neural circuits via realistic leaky integrate-and-fire dynamics.

The project has two operational modes. The first is an exploratory, interactive mode implemented in `connectomics_model.py`, designed for rapid prototyping and single-run visualisation. The second is a high-throughput batch mode implemented in `new_model.py`, designed to run thousands of stochastic trials across a High-Performance Computing (HPC) cluster using IBM Platform LSF job scheduling, and to aggregate results into statistically robust summary files.

The full computational workflow proceeds as follows: raw connectome data is retrieved and cleaned in preprocessing; a master neuron roster and indexed connectivity table are constructed; JSON configuration files define each experiment's stimulation parameters; batch jobs are submitted to the HPC cluster; each job runs isolated, parallel simulation trials using Brian2; aggregated statistics are written to HDF5 and Parquet files; post-processing reads those files and produces per-group and per-neuron summary CSVs and raster plots.

### Key Technologies

| Component | Technology |
|---|---|
| Neural simulation | Brian2 (Python) |
| Connectome data access | fafbseg / FlyWire, CAVEclient |
| Data storage | Parquet (PyArrow), HDF5 (h5py), Pickle, CSV |
| Parallelism (trial-level) | Python multiprocessing |
| Parallelism (job-level) | IBM LSF (bsub, job arrays) |
| Configuration | JSON |

---

## 2. System Architecture and Data Flow

The pipeline is divided into five logical stages, each handled by a distinct file or set of files.

### Stage 1: Preprocessing (`connectomic_preprocessing.py`)

Raw lookup tables from the FlyWire connectome and supplementary spreadsheets are loaded. Neuron root IDs, which change as the connectome is proofread, are updated to their current values via the FlyWire API. All data sources are merged into a single master neuron roster (`CB_completeness.csv`) and a connectivity table (`CB_connectivity.parquet`). A neuron-grouping dictionary (`neuron_ranges.pkl`) is serialised for use downstream. We use the  [princeton_connections](https://drive.google.com/file/d/1hxv_lTuEnjjsSV75vhJXUtFtzJiRgREK/view?usp=sharing) neuron connections list published by flywire.

### Stage 2: Experiment Design (`generate_configs.py`)

A parametric sweep over neuron groups (JO-A, JO-B, combined), left/right activation ratios, and Poisson input firing rates produces one JSON configuration file per experimental condition. These files live in `run_configs/` and are the single source of truth for every simulation parameter.

### Stage 3: Job Submission (`submitSimulationAssay.py`, `submitjobs.sh`)

For each configuration file, an LSF job array is generated and submitted. Each array element is one batch of simulation trials. The batch system distributes work across cluster nodes without requiring inter-node communication.

### Stage 4: Simulation (`new_model.py`)

Each batch job loads the shared data, constructs a Brian2 spiking neural network, applies Poisson stimulation to the selected neurons, runs the simulation for the configured duration, and writes aggregated spike-count and first-spike-time statistics to a per-batch HDF5 file. Full spike trains for the first five trials are written to a Parquet raster file for subsequent visualisation.

### Stage 5: Post-Processing (`run_postprocessing.py`, `new_model.py --postprocess`)

All per-batch HDF5 files for each experiment are read and stacked. Summary statistics (mean, standard deviation, median spike counts; mean firing rates; first-spike times) are computed per neuron group and per individual neuron. Results are written to `summary_analysis.csv` and `per_neuron_statistics.csv`. Raster plots are generated as SVG files.

---

## 3. File-by-File Reference

---

### 3.1 `connectomic_preprocessing.py`

**Purpose:** Fetches, reconciles, and indexes all connectome and annotation data to produce the canonical input files used by all downstream simulation code.

---

```python
from fafbseg import flywire
from caveclient import CAVEclient
```
Imports the FlyWire Python client (`fafbseg`) and the Connectome Annotation Versioning Engine client (`CAVEclient`). These libraries handle authentication and API calls to the FlyWire connectome server, which hosts the live, continuously updated FAFB dataset.

```python
import os
import numpy as np
import pickle
import pandas as pd
from tqdm import tqdm
from textwrap import dedent
from datetime import datetime, timedelta
```
Standard scientific Python imports. `pickle` is used to serialise the neuron-grouping dictionary. `tqdm` provides progress bars for long-running loops. `datetime` and `timedelta` are used for date-stamped filename suffixes.

```python
pd.options.mode.chained_assignment = None
```
Suppresses the pandas `SettingWithCopyWarning`, which is a spurious warning in contexts where in-place modification is intentional.

```python
client = CAVEclient()
client.auth.save_token(token="5be2f6e6cfe49c7c7ccc0c8e791a60be", overwrite=True)
```
Initialises a CAVEclient session and saves the API authentication token to the local credential store. This token authorises read access to the FlyWire production dataset. The `overwrite=True` flag replaces any previously stored token.

```python
datastack_name = "flywire_fafb_production"
client = CAVEclient(datastack_name)
```
Re-instantiates the client, this time bound to the `flywire_fafb_production` datastack, which is the authoritative, proofread version of the full adult fly brain connectome.

```python
synapse_table = client.info.get_datastack_info()['synapse_table']
print(synapse_table)
```
Queries the server for metadata about the datastack, including the name of the synapse annotation table currently in use. This is printed for verification and audit purposes.

```python
date = datetime.today().strftime('%Y%m%d')
todays_date = date[2:]
date_30d_ago = (datetime.today() - timedelta(days=30)).strftime('%Y%m%d')[2:]
```
Generates date strings for use in time-stamped filenames. `todays_date` is the current date with the century digits stripped (e.g., `250714`). `date_30d_ago` is the date thirty days prior in the same format, used for incremental update logic.

```python
lockwood = pd.read_csv('data_setup/Auditory neurons Lockwood et al - Sheet1.csv')
princeton = pd.read_csv('data_setup/connections_princeton.csv')
consolidated_types = pd.read_csv('data_setup/consolidated_cell_types.csv')
joa_clusters = pd.read_csv('data_setup/JO_left_cluster_list_ordered_by_dendrogram_new_synapse_7-16.csv')
names = pd.read_csv('data_setup/names.csv')
```
Loads five source annotation tables. `lockwood` contains auditory neuron identities from Lockwood et al. `princeton` contains the raw pairwise connectivity data (pre/post synaptic neuron pairs with synapse counts) as provided by the Princeton FlyWire team. `consolidated_types` contains curated cell type assignments. `joa_clusters` provides Johnston's Organ A (JO-A) neuron cluster assignments ordered by dendrogram. `names` contains human-readable name overrides.

```python
lockwood.rename(columns={'Codex June 2024': 'root_id'}, inplace=True)
joa_clusters.rename(columns={'pre_root_id': 'root_id'}, inplace=True)
```
Standardises the root ID column name to `root_id` across both tables so that subsequent merge operations use a consistent key.

```python
def update_ids_in_chunks(ids_series, chunk_size=500000):
```
Defines a helper function for chunked ID updating. FlyWire root IDs are not permanent: as the connectome is proofread and neurons are merged or split, their IDs change. This function handles large series of IDs that exceed API call size limits by splitting them into batches of up to 500,000 IDs.

```python
    for i in range(num_chunks):
        ...
        updated_chunk = flywire.update_ids(id=chunk)
        updated_ids_list.append(updated_chunk['new_id'])
```
For each chunk, calls `flywire.update_ids` which queries the FlyWire server for the most current root ID corresponding to each input ID. The `new_id` column of the returned DataFrame contains the updated values.

```python
lockwood['root_id'] = flywire.update_ids(id=lockwood['root_id'])['new_id'].values
joa_clusters['root_id'] = flywire.update_ids(id=joa_clusters['root_id'])['new_id'].values
names['root_id'] = flywire.update_ids(id=names['root_id'])['new_id'].values
consolidated_types['root_id'] = flywire.update_ids(id=consolidated_types['root_id'])['new_id'].values
```
Updates the root IDs in each annotation table in place. This ensures that all annotation tables reference the same version of each neuron's identity, preventing mismatches between data sources that were downloaded at different times.

```python
all_ids = set()
all_ids.update(lockwood['root_id'])
all_ids.update(joa_clusters['root_id'])
...
completeness = pd.DataFrame(list(all_ids), columns=['root_id'])
completeness.dropna(inplace=True)
```
Constructs the master neuron roster by taking the union of all unique root IDs appearing in any annotation source. Using a Python `set` automatically deduplicates. The resulting DataFrame, `completeness`, will have one row per unique neuron.

```python
completeness = completeness.merge(lockwood, on='root_id', how='left')
completeness = completeness.merge(consolidated_types, on='root_id', how='left')
completeness = completeness.merge(joa_clusters, on='root_id', how='left')
completeness = completeness.merge(names, on='root_id', how='left')
```
Performs a series of left joins to attach all available annotation columns to each neuron. Left joins preserve all neurons in the master roster even if they lack annotations in a given source.

```python
completeness.drop_duplicates(subset=['root_id'], keep='first', inplace=True)
```
Strictly deduplicates the merged table on `root_id`. Joins can introduce duplicate rows if source tables themselves contain duplicate IDs; this line ensures the final table has exactly one row per neuron.

```python
completeness['label'] = completeness['Neuron name (Lockwood et al)'].combine_first(
    completeness['additional_type(s)']
).combine_first(completeness['primary_type'])
completeness['label'] = completeness['label'].fillna(completeness['root_id'].astype(str))
```
Generates a human-readable label for each neuron using a priority cascade. The Lockwood name is used if available; otherwise the additional type annotation is used; otherwise the primary type. Any neuron with no annotation at all receives its numeric root ID as a fallback label.

```python
completeness = completeness[['root_id', 'label', 'Cluster']]
completeness = completeness.reset_index(drop=True)
completeness['index'] = completeness.index
id2idx = dict(zip(completeness['root_id'], completeness['index']))
```
Reduces the master table to the three columns needed downstream and creates a zero-based integer index for each neuron. The dictionary `id2idx` provides O(1) lookup from root ID to simulation array index. This index is how Brian2 and the connectivity matrices reference neurons.

```python
princeton['Presynaptic_Index'] = princeton['pre_root_id'].map(id2idx).fillna(-1).astype(int)
princeton['Postsynaptic_Index'] = princeton['post_root_id'].map(id2idx).fillna(-1).astype(int)
```
Maps every presynaptic and postsynaptic root ID in the connectivity table to its integer index. Connections involving neurons not present in the master roster receive index `-1` and are excluded downstream.

```python
nt_function_map = {'GABA': -1, 'ACH': 1, 'GLUT': -1, 'OCT': 1, 'SER': 1, 'DA': 1}
princeton['Excitatory'] = princeton['nt_type'].map(nt_function_map)
princeton['Connectivity x Excitatory'] = princeton['Connectivity'] * princeton['Excitatory']
```
Assigns a sign to each connection based on its predicted neurotransmitter type. GABAergic and glutamatergic synapses are inhibitory (sign -1); cholinergic, octopaminergic, serotonergic, and dopaminergic synapses are excitatory (sign +1). The signed product `Connectivity x Excitatory` is used as the synaptic weight in the Brian2 model.

```python
connectivity.to_parquet('data/CB_connectivity.parquet')
completeness.to_csv('data/CB_completeness.csv')
```
Saves the final connectivity table and neuron roster. Parquet format is used for the connectivity table because it supports efficient column-wise access and compression of the large, sparse connection matrix.

```python
lockwood_sided = lockwood.copy(deep=True)
lockwood_sided['Side'] = lockwood_sided['Neuron name (Lockwood et al)'].str.extract(r'_(R|L)')
```
Creates a working copy of the Lockwood table and extracts the lateralisation suffix (L or R) from each neuron name using a regular expression. Neurons without a lateralisation suffix (e.g., midline neurons) will have `NaN` in this column.

```python
lockwood_sided["GroupKey"] = lockwood_sided.apply(
    lambda row: f"{row['Cell type (Lockwood et al)']}_{row['Side']}"
    if pd.notna(row['Side'])
    else row['Cell type (Lockwood et al)'],
    axis=1
)
```
Constructs a group key for each neuron. For lateralised neurons the key is `CellType_L` or `CellType_R`. For midline neurons without a side, the key is simply the cell type name. This correctly groups neurons while preserving bilateral symmetry information.

```python
neuron_ranges = lockwood_sided.groupby('GroupKey')['root_id'].apply(list).to_dict()
joa_clusters['Cluster'] = joa_clusters['Cluster'].apply(lambda x: f'Cluster{x}')
neuron_ranges.update(joa_clusters.groupby('Cluster')['root_id'].apply(list).to_dict())
```
Builds the `neuron_ranges` dictionary, which maps each group name (e.g., `JO-A_L`, `Cluster3`) to the list of root IDs belonging to that group. JO-A cluster assignments are also incorporated. This dictionary is the primary mechanism by which stimulation and post-processing code selects neurons to act upon.

```python
with open('data/neuron_ranges.pkl', 'wb') as file:
    pickle.dump(neuron_ranges, file, protocol=pickle.HIGHEST_PROTOCOL)
```
Serialises the `neuron_ranges` dictionary to disk using the highest available pickle protocol for compact, fast I/O. This file is loaded by every downstream simulation and post-processing script.

---

### 3.2 `connectomics_model.py`

**Purpose:** Provides an object-oriented, interactive interface for single-run simulation, visualisation, and exploratory analysis. This is the prototype-phase model, used before the high-throughput batch infrastructure was finalised.

---

```python
from brian2 import NeuronGroup, Synapses, PoissonInput, SpikeMonitor, StateMonitor, Network, TimedArray, start_scope
from brian2 import mV, ms, Hz
```
Imports the core Brian2 components used in the spiking network model. `NeuronGroup` defines the neuron population and its differential equations. `Synapses` defines connections and their plasticity rules. `PoissonInput` applies stochastic spike trains to selected neurons. `SpikeMonitor` records spike times. `StateMonitor` records continuous state variables (membrane voltage). `start_scope` resets the Brian2 namespace between runs to prevent state contamination.

```python
class ConnectomicsModel:
    def __init__(self, data_path='data/'):
```
Defines the main simulation class. The constructor accepts a `data_path` argument allowing deployment in directories with different folder structures.

```python
        self.aud_label_root_id = pd.read_csv(os.path.join(data_path, 'aud_label_root_id.csv'), index_col=0)
        self.aud_filtered_princeton = pd.read_csv(os.path.join(data_path, 'aud_filtered_princeton.csv'), index_col=0)
```
Loads the auditory-circuit-filtered versions of the neuron roster and connectivity table. These are pre-filtered to contain only the neurons and connections relevant to the auditory pathway, reducing simulation size.

```python
        with open(os.path.join(data_path, 'id_idx_dict.pickle'), 'rb') as fi1:
            self.id_idx_dict = pickle.load(fi1)
        with open(os.path.join(data_path, 'idx_id_dict.pickle'), 'rb') as fi2:
            self.idx_id_dict = pickle.load(fi2)
```
Loads bidirectional lookup dictionaries: `id_idx_dict` maps root ID to simulation index, and `idx_id_dict` maps simulation index back to root ID. Both directions of lookup are needed frequently during results annotation.

```python
        self.default_params = {
            't_run': 500 * ms,
            'v_0': -52 * mV,
            'v_rst': -52 * mV,
            'v_th': -45 * mV,
            't_mbr': 20 * ms,
            'tau': 5 * ms,
            't_rfc': 2.2 * ms,
            't_dly': 1.8 * ms,
            'w_syn': 0.8 * mV,
            'r_poi': 250 * Hz,
            'f_poi': 150,
            ...
        }
```
[See Shiu et. al](https://www.biorxiv.org/content/10.1101/2023.05.02.539144v1) 
This defines the default biophysical parameters for the leaky integrate-and-fire (LIF) model. The resting potential `v_0` and reset potential `v_rst` are both set to -52 mV. The spike threshold `v_th` is -45 mV, giving a 7 mV depolarisation margin. `t_mbr` is the membrane time constant (20 ms), governing how quickly the membrane potential decays to rest. `tau` is the synaptic conductance decay time constant (5 ms). `t_rfc` is the absolute refractory period (2.2 ms). `t_dly` is the synaptic transmission delay (1.8 ms). `w_syn` is the per-synapse weight (0.8 mV per spike per synapse). `r_poi` is the default Poisson input firing rate applied to directly stimulated neurons. `f_poi` is a scaling factor multiplying the Poisson synapse weight.

```python
    def get_case(self, activation_neuron_list, neuron_group_activation, activate_both_sides, activation_side, random_selection, random_selection_percent):
```
A routing function that translates the combination of input arguments into one of three integer case codes. Case 2 means a direct list of neuron IDs was provided. Case 1 means both hemispheres of a named group should be activated. Case 3 means only one hemisphere should be activated. Returns -1 if the inputs are invalid.

```python
    def get_excitatory_neurons(self, ...):
```
Uses the case code from `get_case` to select the appropriate neuron indices from `neurons_ranges`. If random selection is enabled, a random sample of the specified percentage is drawn without replacement using `random.sample`. Returns three parallel lists: human-readable labels, root IDs, and simulation indices.

```python
    def save_simulation_data(self, spk_mon, state_mon, user_activated_indices, naturally_activated_indices, params, config):
```
Saves all outputs from a single simulation run to a timestamped subdirectory under `simulation_results/`. Outputs include `spikes.csv` (one row per spike, with neuron label and activation type), `voltage_traces.csv` (membrane voltage over time for up to 20 monitored neurons), `activated_neurons.csv` (roster of directly and indirectly activated neurons), `hyperparameters.json` (all simulation parameters with units), `hyperparameters.pickle` (exact Brian2 objects for reconstruction), and `summary_stats.json` (aggregate statistics).

```python
        for key, value in params.items():
            if hasattr(value, 'dim'):
                if str(value.dim) == 'second':
                    params_data[f'{key}_value'] = float(value / ms)
                    params_data[f'{key}_unit'] = 'ms'
```
Handles the serialisation of Brian2 `Quantity` objects, which carry physical units and cannot be directly serialised by the standard `json` module. Each quantity is converted to a plain float by dividing by its unit, and the unit string is stored in a companion key.

```python
    def run_model(self, ...):
        start_scope()
        neu = NeuronGroup(
            N=len(self.aud_label_root_id),
            model=dedent('''
                dv/dt = (v_0 - v + g) / t_mbr : volt (unless refractory)
                dg/dt = -g / tau               : volt (unless refractory)
                rfc                            : second
            '''),
            ...
        )
```
The core simulation entry point. `start_scope()` clears the Brian2 namespace. A `NeuronGroup` is created with one neuron for every row in the filtered neuron roster. The model equations implement a leaky integrate-and-fire neuron with a separate synaptic conductance variable `g`. The membrane voltage `v` decays toward resting potential `v_0` with time constant `t_mbr`, driven by synaptic input `g`. The conductance `g` decays to zero with time constant `tau`. Both variables are frozen during the refractory period (`unless refractory`).

```python
        syn = Synapses(neu, neu, 'w : volt', on_pre='g += w', delay=params['t_dly'])
        syn.connect(i=i_pre, j=i_post)
        syn.w = self.aud_filtered_princeton['Signed_Connectivity'].values * params['w_syn']
```
Creates the synapse object connecting the population to itself. The `on_pre` rule increments the postsynaptic conductance `g` by weight `w` each time a presynaptic spike arrives. The weight for each synapse is the signed connectivity (excitatory positive, inhibitory negative) multiplied by the base synaptic weight `w_syn`.

```python
        for i in activated_neuron_idx:
            p = PoissonInput(target=neu[i], target_var='v', N=1, rate=params['r_poi'], weight=params['w_syn'] * params['f_poi'])
            neu[i].rfc = 0 * ms
            pois.append(p)
```
Applies a `PoissonInput` directly to the membrane voltage `v` of each stimulated neuron at rate `r_poi`. The large weight (`w_syn * f_poi`) ensures that Poisson spikes reliably drive the neuron above threshold, simulating direct sensory drive. The refractory period for stimulated neurons is set to zero so that high-rate Poisson input can drive them at physiologically plausible rates without artificial rate limiting.

```python
    def post_processing(self, simulation_dir):
```
Reads the `spikes.csv` file from a completed simulation run and produces `summary_spikes_per_group_and_cluster.csv`. For each neuron group and JO-A cluster, it computes the total spike count and the time and identity of the first spike.

```python
    def plot_combined_spike_raster(self, spike_monitors_list):
```
Produces a dual-colour scatter plot raster diagram. User-activated neurons (directly stimulated via Poisson input) are plotted as blue circles. Naturally activated neurons (recruited by network propagation) are plotted as purple squares. The Y-axis shows neuron names rather than bare indices. If more than 50 neurons spiked, labels are subsampled for legibility.

---

### 3.3 `new_model.py`

**Purpose:** The production-grade simulation engine designed for large-scale HPC batch execution. Replaces `connectomics_model.py` for all quantitative experiments.

---

```python
import h5py
```
Imports the h5py library for reading and writing HDF5 files. HDF5 is used for aggregated simulation statistics because it supports efficient compression of large numerical arrays and atomic writes, which is important when many parallel processes are writing to shared storage.

```python
def load_config(config_path):
    with open(config_path, "r") as f:
        return json.load(f)
```
Reads the JSON configuration file for the current experiment. Every simulation parameter, file path, and stimulation specification is drawn from this file, making each run fully reproducible and self-documenting.

```python
def load_data(paths):
    completeness_df = pd.read_csv(paths["completeness_file"], index_col=0, dtype={"root_id": "str"})
    connectivity_df = pd.read_parquet(paths["connectivity_file"])
    jo_clusters_df = pd.read_csv(paths["jo_cluster_file"])
    with open(paths["neuron_ranges_pickle"], "rb") as f:
        neuron_ranges = pickle.load(f)
```
Loads all data files referenced in the configuration. Root IDs are loaded as strings (`dtype={"root_id": "str"}`) to prevent silent precision loss from integer overflow when handling the large 64-bit FlyWire neuron IDs. The connectivity table is read from Parquet for speed.

```python
    idx_to_id = completeness_df["root_id"].to_dict()
    id_to_idx = {v: k for k, v in idx_to_id.items()}
```
Constructs both directions of the index-to-ID lookup dictionary. `idx_to_id` maps integer simulation index to root ID string. `id_to_idx` is its inverse. These are built once at load time and reused across all trials in a batch.

```python
def prepare_stimulation(config, data, batch_seed_offset):
    random.seed(batch_seed_offset)
    ...
    for side, hemisphere_config in stimulation_plan.items():
        if not hemisphere_config["activate"]: continue
        for group_config in hemisphere_config["groups"]:
```
Reads the stimulation configuration and selects the neurons to stimulate for a given trial. The function iterates over left and right hemisphere configurations. For each group, it looks up the corresponding root IDs in `neuron_ranges`, converts them to simulation indices, applies random subsampling according to `random_selection_percent`, and records the Poisson firing rate for each selected neuron. The random seed is set from `batch_seed_offset` to ensure reproducibility across reruns.

```python
            if base_group_name.lower().startswith('cluster'):
                group_name_to_lookup = base_group_name
            else:
                side_suffix = '_L' if 'left' in side else '_R'
                sided_name = base_group_name + side_suffix
```
Handles two classes of neuron groups: JO-A clusters (which are sideless, as the cluster file contains only left-side neurons) and standard lateralised groups (which require appending `_L` or `_R` to the base group name).

```python
def prepare_silencing(config, data, batch_seed_offset):
```
Analogous to `prepare_stimulation` but for the silencing configuration. Returns a set of neuron indices to silence. Silencing is implemented downstream by setting outgoing synaptic weights to zero, preventing silenced neurons from influencing the network while allowing them to receive input.

```python
def run_single_trial_isolated(config, data, params, brian_params_dict, trial_seed, global_trial_num, save_full_spikes=False):
    start_scope()
    defaultclock.dt = 0.1 * ms
    prefs.codegen.target = 'numpy'
    seed(trial_seed)
```
Runs one complete simulation trial in full isolation. `start_scope()` clears Brian2 state. The simulation clock is set to 0.1 ms resolution. The numpy code generation backend is selected to avoid Cython compilation conflicts when running many processes in parallel. The Brian2 random seed, NumPy seed, and Python random seed are all set from `trial_seed` for reproducibility.

```python
    brian_params = {
        "v_0": brian_params_dict["v_0"] * mV,
        ...
    }
```
Reconstructs Brian2 `Quantity` objects from the plain-number dictionary. Plain numbers must be passed between processes because Brian2 `Quantity` objects are not picklable, and Python's multiprocessing uses pickle for inter-process communication.

```python
    model_eqs = dedent("""
        dv/dt = (v_0 - v + g) / t_mbr : volt (unless refractory)
        dg/dt = -g / tau              : volt (unless refractory)
        rfc                           : second
    """)
```
The LIF model equations in Brian2's differential equation syntax. These are identical to those in `connectomics_model.py` for consistency.

```python
    syn.connect(
        i=data["connectivity"]["Presynaptic_Index"].values,
        j=data["connectivity"]["Postsynaptic_Index"].values,
    )
    syn.w = data["connectivity"]["Connectivity x Excitatory"].values * brian_params["w_syn"]
```
Connects all synapses using precomputed index arrays. This vectorised approach is substantially faster than iterative connection methods. The signed connectivity values (positive for excitatory, negative for inhibitory) are multiplied by the base weight to set each synapse's strength and polarity.

```python
    silenced_indices = prepare_silencing(config, data, trial_seed)
    if silenced_indices:
        silenced_arr = np.array(list(silenced_indices))
        syn.w[silenced_arr, :] = 0 * mV
```
Implements neuron silencing by zeroing out the synaptic weights of all outgoing connections from each silenced neuron. This is equivalent to pharmacological silencing in the sense that silenced neurons can still receive (but not transmit) signals.

```python
    neuron_spike_counts = np.bincount(spikes_i, minlength=n_neurons).astype(np.uint16)
```
Computes the spike count for each neuron using `np.bincount`, which is substantially faster than groupby operations for this purpose. The result is stored as `uint16` (0 to 65535) to minimise memory usage, as spike counts within a single trial will not approach this limit.

```python
    if len(spikes_i) > 0:
        sort_idx = np.argsort(spikes_t)
        sorted_neurons = spikes_i[sort_idx]
        sorted_times = spikes_t[sort_idx]
        _, first_spike_idx = np.unique(sorted_neurons, return_index=True)
        first_spike_neurons = sorted_neurons[first_spike_idx]
        first_spike_times[first_spike_neurons] = sorted_times[first_spike_idx]
```
Computes the first spike time per neuron using a vectorised approach. Spikes are sorted by time; `np.unique` with `return_index=True` efficiently finds the index of the first occurrence of each neuron ID in the sorted array, giving the time of the neuron's first spike in O(n log n) time.

```python
    if save_full_spikes:
        result["full_spikes"] = {
            "neuron_index": spikes_i,
            "spike_time_ms": spikes_t,
            "trial": global_trial_num,
        }
```
For the first five trials of the first batch, the complete spike train (every spike time and neuron index) is returned in addition to the aggregated statistics. This data is used to generate raster plots. For all other trials, only the aggregated statistics are returned to minimise memory usage.

```python
def run_batch(config_path, batch_id, trials_per_batch):
    from multiprocessing import Pool
    ...
    with Pool(processes=n_cores) as pool:
        results = pool.starmap(run_single_trial_isolated, trial_args)
```
The batch runner. A multiprocessing `Pool` distributes `trials_per_batch` independent simulation trials across all available CPU cores. Each worker calls `run_single_trial_isolated` with a unique seed, ensuring that trials are statistically independent. The `starmap` method unpacks the argument tuples automatically.

```python
    stats_file = os.path.join(save_dir, f"batch_{batch_id}_statistics.h5")
    with h5py.File(stats_file, 'w') as f:
        batch_group = f.create_group("data")
        batch_group.create_dataset("spike_counts", data=spike_counts_batch, compression="gzip", compression_opts=9)
        batch_group.create_dataset("first_spikes", data=first_spikes_batch, compression="gzip", compression_opts=9)
```
Writes the aggregated statistics for this batch to a dedicated HDF5 file. Each batch produces its own file (named `batch_N_statistics.h5`), avoiding write conflicts between simultaneously running batch jobs. Maximum gzip compression (level 9) is used because the spike count arrays are sparse and compress extremely well.

```python
def post_process(exp_dir, data, n_trials, t_run_s):
    batch_files = sorted(glob(os.path.join(exp_dir, "batch_*_statistics.h5")))
    ...
    spike_counts_array = np.vstack(all_spike_counts)
    first_spikes_array = np.vstack(all_first_spikes)
```
Reads all per-batch HDF5 files and stacks them vertically into two arrays of shape `(n_trials, n_neurons)`. `spike_counts_array[t, n]` is the number of spikes fired by neuron `n` in trial `t`. `first_spikes_array[t, n]` is the time of neuron `n`'s first spike in trial `t`, or `NaN` if it did not fire.

```python
    group_spike_counts = spike_counts_array[:, indices].sum(axis=1)
    avg_spikes = group_spike_counts.mean()
    std_spikes = group_spike_counts.std()
    median_spikes = np.median(group_spike_counts)
```
For each neuron group, extracts the columns corresponding to its member neurons, sums across neurons within each trial to get the group's total spike count per trial, and then computes the mean, standard deviation, and median across trials.

```python
    unique_spikers_per_trial = (spike_counts_array[:, indices] > 0).sum(axis=1)
```
Counts the number of distinct neurons in each group that fired at least once per trial, providing a measure of population recruitment breadth as distinct from total spike count.

```python
    rates_per_trial = group_spike_counts / t_run_s
```
Converts spike counts to firing rates in Hz by dividing by the trial duration in seconds.

```python
    ungrouped_indices = list(all_neuron_indices - all_grouped_indices)
    ...
    top_20_idx = np.argsort(avg_rate_per_neuron)[-20:][::-1]
```
Identifies the top 20 most active neurons that are not members of any named neuron group. These are appended to the summary file as individually labelled entries, providing visibility into highly active neurons that fall outside the pre-defined anatomical groups.

```python
def create_raster_plot(raster_data_file, save_dir, data, plot_config, trial_to_plot=0):
```
Generates an SVG raster plot for a specified trial from the saved Parquet spike data. Neurons are grouped by membership and displayed as horizontal rows separated by gaps. Each neuron group receives a distinct colour from the matplotlib `tab10` palette. The plot is saved as an SVG (scalable vector graphics) file for publication-quality reproduction.

```python
def aggregate_and_postprocess(base_output_dir=None, experiment_name=None):
```
A convenience wrapper that iterates over all experiment directories in `base_output_dir`, checks for the presence of batch HDF5 files, loads the experiment's own `config.json`, and calls `post_process` and `create_raster_plot`. Can be targeted at a single named experiment using the `experiment_name` argument.

---

### 3.4 `generate_configs.py`

**Purpose:** Programmatically generates all JSON configuration files for a parametric sweep experiment, ensuring systematic coverage of the parameter space without manual file creation.

---

```python
BASE_CONFIG = {
    "output_config": { ... },
    "simulation_parameters": { ... },
    "file_paths": { ... },
    "stimulation_config": { ... },
    "silencing_config": { ... },
    "raster_plot_config": { ... }
}
```
Defines the template configuration dictionary from which all experiment-specific configs are derived. This template specifies default values for all parameters that are not swept, including file paths, base simulation parameters, silencing settings (disabled by default), and raster plot group names.

```python
POISSON_RATES = [28, 100, 150, 200, 250, 400]
LR_RATIOS = [(0, 100), (25, 75), (50, 50), (75, 25), (100, 0)]
```
Defines the two axes of the parameter sweep. `POISSON_RATES` spans a physiologically relevant range of Poisson input firing rates in Hz, from low (28 Hz, near spontaneous) to high (400 Hz, intense stimulation). `LR_RATIOS` defines five bilateral activation patterns ranging from fully right-lateralised to fully left-lateralised, with balanced bilateral activation in the centre.

```python
def generate_config_joa_only(lr_left, lr_right, poisson_rate, experiment_id):
    config = deepcopy(BASE_CONFIG)
    config["output_config"]["output_directory_name"] = (
        f"Exp_{experiment_id:04d}_JOA_LR{lr_left}-{lr_right}_Rate{poisson_rate}"
    )
```
`deepcopy` is critical here: without it, modifications to one config would mutate the shared `BASE_CONFIG` template, causing all subsequently generated configs to inherit unintended changes. The output directory name encodes all swept parameters for human readability.

```python
    if lr_left > 0:
        config["stimulation_config"]["left_hemisphere"]["groups"] = [{
            "group_name": "JO-A",
            "random_selection_percent": lr_left,
            "poisson_rate_hz": poisson_rate
        }]
    else:
        config["stimulation_config"]["left_hemisphere"]["activate"] = False
```
Sets the left hemisphere stimulation configuration. When `lr_left` is zero, the hemisphere is fully deactivated rather than being configured with a zero-percent selection, which avoids edge cases in the stimulation preparation code.

```python
experiment_types = [
    ("JOA", generate_config_joa_only),
    ("JOB", generate_config_job_only),
    ("JOAB", generate_config_both)
]
for exp_type, generator_func in experiment_types:
    for poisson_rate in POISSON_RATES:
        for lr_left, lr_right in LR_RATIOS:
```
The main generation loop iterates over all combinations of experiment type, Poisson rate, and L/R ratio. The total number of configurations is 3 experiment types × 6 rates × 5 L/R ratios = 90 experiments.

```python
with open(filepath, 'w') as f:
    json.dump(config, f, indent=2)
```
Writes each configuration to a uniquely named JSON file with 2-space indentation for human readability. The filename encodes experiment ID, type, L/R ratio, and rate, making it possible to identify any experiment from its filename alone.

---

### 3.5 `submitSimulationAssay.py`

**Purpose:** Generates and submits LSF job array scripts for every configuration file in `run_configs/`, orchestrating the HPC batch execution of the full simulation assay.

---

```python
TOTAL_TRIALS = 10000
TRIALS_PER_BATCH = 128
```
Defines the total number of stochastic trials per experiment and the number of trials assigned to each LSF array element. With 10,000 trials and 128 trials per batch, each experiment requires `ceil(10000 / 128) = 79` LSF array jobs.

```python
num_batches = math.ceil(TOTAL_TRIALS / TRIALS_PER_BATCH)
```
Computes the number of array elements needed. `math.ceil` ensures that the last batch covers any remainder if `TOTAL_TRIALS` is not exactly divisible by `TRIALS_PER_BATCH`.

```python
text = f"""#!/bin/bash
    #BSUB -n 32
    #BSUB -W 20
    #BSUB -R "rusage[mem=1GB/task]"
    #BSUB -R "span[hosts=1]"
    #BSUB -J {cfile}[1-{num_batches}]
    #BSUB -o {HPC_PROJECT_DIR}/out_files/out.%J.%I
    #BSUB -e {HPC_PROJECT_DIR}/err_files/err.%J.%I
```
The LSF job script header. `-n 32` requests 32 CPU cores per array element. `-W 20` sets a 20-minute wall-clock time limit. `-R "rusage[mem=1GB/task]"` requests 1 GB RAM per core (32 GB total). `-R "span[hosts=1]"` constrains all 32 cores to a single physical node, which is required because the Python multiprocessing pool uses shared memory. The `[1-{num_batches}]` suffix creates a job array. `%J` is the master job ID; `%I` is the array element index.

```python
    python new_model.py run_configs/{cfile} --batch_id $LSB_JOBINDEX --trials {TRIALS_PER_BATCH}
```
The command executed by each array element. `$LSB_JOBINDEX` is the LSF environment variable containing this element's index within the array (1-based), which is passed as `--batch_id`. This ensures each batch produces a uniquely named HDF5 output file.

```python
    try:
        subprocess.run(["which", "bsub"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.system(cmd)
    except subprocess.CalledProcessError:
        print(f"[Info] 'bsub' command not found. Created script '{script_name}' but did not submit.")
```
Checks for the presence of the `bsub` command before attempting submission. This allows the script to run safely on non-HPC machines (e.g., for testing), where it will create scripts but not attempt to submit them.

---

### 3.6 `submitjobs.sh`

**Purpose:** An LSF wrapper script that submits `submitSimulationAssay.py` as a cluster job, ensuring that the submission process itself has access to the cluster's file system and environment.

---

```bash
#BSUB -n 1
#BSUB -W 2
#BSUB -J submit
```
Requests a single CPU core with a 2-minute wall-clock time limit. Submission itself is fast; this is merely a bootstrapping job to run the Python submission script from within the cluster environment, ensuring that all `bsub` calls are made from a node with access to the correct paths and modules.

```bash
python submitSimulationAssay.py
```
Executes the submission script, which iterates over all JSON configs and submits a job array for each.

---

### 3.7 `run_postprocessing.py`

**Purpose:** Submits a separate, independent LSF post-processing job for each completed experiment, enabling parallel post-processing of multiple experiments simultaneously.

---

```python
HPC_PROJECT_DIR = "/rs1/researchers/c/cbaker5/Ayush/Connectomic-Model"
SIM_SAVED_DIR = os.path.join(HPC_PROJECT_DIR, "simulation_results")
SCRIPTS_DIR = os.path.join(HPC_PROJECT_DIR, "postprocess_scripts")
```
Hard-coded paths to the project directory and relevant subdirectories on the HPC cluster's shared file system. These paths are embedded in the generated shell scripts.

```python
def check_experiment_ready(exp_dir):
    batch_files = glob(os.path.join(exp_dir, "batch_*_statistics.h5"))
    if not batch_files:
        return False, "No batch files found"
    summary_file = os.path.join(exp_dir, "summary_analysis.csv")
    per_neuron_file = os.path.join(exp_dir, "per_neuron_statistics.csv")
    if os.path.exists(summary_file) and os.path.exists(per_neuron_file):
        return False, f"Already processed ({len(batch_files)} batches)"
    return True, f"Ready ({len(batch_files)} batches)"
```
Determines whether an experiment directory needs post-processing. An experiment is skipped if it has no batch HDF5 files (not yet simulated) or if both output CSVs already exist (already post-processed). This makes `run_postprocessing.py` safe to re-run without overwriting already completed work.

```python
script_content = f"""#!/bin/bash
    #BSUB -n 1
    #BSUB -W 30
    #BSUB -R "rusage[mem=32GB]"
    #BSUB -J postprocess_{exp_name}
```
Each post-processing job requests 1 core, 30 minutes of wall time, and 32 GB RAM. The 32 GB allocation is necessary because `post_process` loads the full spike count array (`n_trials × n_neurons`) into memory, which can reach tens of gigabytes for large experiments.

```python
    python new_model.py --postprocess --experiment {exp_name}
```
Calls the post-processing mode of `new_model.py`, targeting only the specific experiment for this job. This is equivalent to calling `aggregate_and_postprocess(experiment_name=exp_name)`.

```python
    cmd = f"bsub < {script_path}"
    os.system(cmd)
```
Submits the generated shell script to the LSF scheduler using input redirection. The `bsub < script.sh` form is the standard LSF submission pattern.

---

## 4. Data File Reference

| File | Format | Description |
|---|---|---|
| `data/CB_completeness.csv` | CSV | Master neuron roster: root_id, label, Cluster |
| `data/CB_connectivity.parquet` | Parquet | Pairwise connectivity: indices, synapse count, NT type, signed weight |
| `data/neuron_ranges.pkl` | Pickle | Dict mapping group names to lists of root IDs |
| `run_configs/*.json` | JSON | One config file per experiment |
| `simulation_results/<exp>/batch_N_statistics.h5` | HDF5 | Per-batch spike counts and first spike times |
| `simulation_results/<exp>/raster_data_trials_0_to_4.parquet` | Parquet | Full spike trains for first 5 trials |
| `simulation_results/<exp>/summary_analysis.csv` | CSV | Per-group mean/std/median firing statistics |
| `simulation_results/<exp>/per_neuron_statistics.csv` | CSV | Per-neuron firing statistics across all trials |
| `simulation_results/<exp>/raster_plot_trial_N.svg` | SVG | Raster plots for trials 0-4 |

---

## 5. Configuration Schema Reference

The JSON configuration file controls all aspects of a simulation experiment.

```json
{
  "output_config": {
    "base_output_directory": "simulation_results",
    "output_directory_name": "Exp_0001_JOA_LR50-50_Rate250"
  },
  "simulation_parameters": {
    "n_cores": -1,
    "n_trials": 10,
    "t_run_ms": 2000,
    "v_th_mv": -45.0,
    "t_rfc_ms": 2.2,
    "tau_ms": 5.0,
    "w_syn_mv": 1,
    "f_poi": 250
  },
  "file_paths": {
    "completeness_file": "data/CB_completeness.csv",
    "connectivity_file": "data/CB_connectivity.parquet",
    "neuron_ranges_pickle": "data/neuron_ranges.pkl",
    "jo_cluster_file": "data/JO_left_cluster_list_ordered_by_dendrogram_new_synapse_7-16.csv"
  },
  "stimulation_config": {
    "left_hemisphere": {
      "activate": true,
      "groups": [
        {
          "group_name": "JO-A",
          "random_selection_percent": 50,
          "poisson_rate_hz": 250
        }
      ]
    },
    "right_hemisphere": {
      "activate": true,
      "groups": []
    }
  },
  "silencing_config": {
    "left_hemisphere": { "activate": false, "groups": [] },
    "right_hemisphere": { "activate": false, "groups": [] }
  },
  "raster_plot_config": {
    "enabled": true,
    "groups_to_plot": ["PVLP_pr03-1_L", "PVLP_pr03-1_R"]
  }
}
```

**Key fields:**

`n_cores`: Number of CPU cores for multiprocessing. `-1` uses all available cores.

`t_run_ms`: Simulation duration in milliseconds per trial.

`v_th_mv`: Spike threshold in millivolts.

`w_syn_mv`: Base synaptic weight in millivolts per synapse per spike. Multiplied by the signed connectivity value and `f_poi` (for Poisson inputs).

`random_selection_percent`: Percentage of the group's neurons to randomly select for stimulation or silencing in each trial. Enables stochastic sampling to model trial-to-trial variability.

---

## 6. HPC Deployment Guide

### Initial Setup (NOT NEEDED ON THE RESEARCH DRIVE, USE rs1/…/Ayush/Connectomic-Model)

1. Setup the HPC environment as per instructions, using the requirements file to install all needed packages.
2. Clone repository with the multi-runner branch to HPC project directory
3. Run preprocessing (once, not needed if all of the data is present): `python connectomic_preprocessing.py`
4. Generate experiment configuration files: `python generate_configs.py`
5. Create output directories: `mkdir -p out_files err_files simulation_results`

### Submitting Simulations

Submit the submission script as a cluster job:

```bash
python submitSimulationAssay.py
```

This runs `submitSimulationAssay.py` on the cluster, which generates and submits one job array per configuration file. Monitor job status with `bjobs`.

### Running Post-Processing

After all simulation jobs have completed:

```bash
python run_postprocessing.py
```

This submits one post-processing job per completed experiment. Alternatively, post-process a single experiment interactively:

```bash
python new_model.py --postprocess --experiment Exp_0001_JOA_LR50-50_Rate250
```