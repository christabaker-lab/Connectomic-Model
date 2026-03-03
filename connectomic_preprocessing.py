from fafbseg import flywire
from caveclient import CAVEclient
import os
import numpy as np
import pickle
import pandas as pd
from tqdm import tqdm
from textwrap import dedent
from datetime import datetime, timedelta


pd.options.mode.chained_assignment = None  # default='warn'

# --- 1. IMPORTS ---
client = CAVEclient()
client.auth.save_token(token="5be2f6e6cfe49c7c7ccc0c8e791a60be", overwrite=True)
# CAVEclient authorization, code from the documentation
client = CAVEclient()
client.auth.save_token(token="5be2f6e6cfe49c7c7ccc0c8e791a60be",overwrite=True)
# client.auth.save_token(token="cf5a53b7c589672a22207c9ec5bdea33",overwrite=True)

datastack_name = "flywire_fafb_production"
client = CAVEclient(datastack_name)

synapse_table = client.info.get_datastack_info()['synapse_table']
print(synapse_table)

## datetime and extension for filenames

date = datetime.today().strftime('%Y%m%d')
todays_date = date[2:]
date_30d_ago = (datetime.today() - timedelta(days=30)).strftime('%Y%m%d')[2:]

print(date_30d_ago)

# --- 2. LOAD DATA ---
print("Loading source files...")
lockwood = pd.read_csv('data_setup/Auditory neurons Lockwood et al - Sheet1.csv')
princeton = pd.read_csv('data_setup/connections_princeton.csv')
consolidated_types = pd.read_csv('data_setup/consolidated_cell_types.csv')
joa_clusters = pd.read_csv('data_setup/JO_left_cluster_list_ordered_by_dendrogram_new_synapse_7-16.csv')
names = pd.read_csv('data_setup/names.csv')

lockwood.rename(columns={'Codex June 2024': 'root_id'}, inplace=True)
joa_clusters.rename(columns={'pre_root_id': 'root_id'}, inplace=True)

# --- 3. ID UPDATING ---
def update_ids_in_chunks(ids_series, chunk_size=500000):
    updated_ids_list = []
    num_chunks = int(np.ceil(len(ids_series) / chunk_size))
    print(f"Updating {len(ids_series)} IDs in {num_chunks} chunks...")

    for i in range(num_chunks):
        start = i * chunk_size
        end = (i + 1) * chunk_size
        chunk = ids_series[start:end]
        updated_chunk = flywire.update_ids(id=chunk)
        updated_ids_list.append(updated_chunk['new_id'])

    return pd.concat(updated_ids_list, ignore_index=True)

print("Updating root IDs...")
lockwood['root_id'] = flywire.update_ids(id=lockwood['root_id'])['new_id'].values
joa_clusters['root_id'] = flywire.update_ids(id=joa_clusters['root_id'])['new_id'].values
names['root_id'] = flywire.update_ids(id=names['root_id'])['new_id'].values
consolidated_types['root_id'] = flywire.update_ids(id=consolidated_types['root_id'])['new_id'].values
# princeton['pre_root_id'] = update_ids_in_chunks(princeton['pre_root_id'])

print("Finished updating IDs.")

# --- 4. MASTER LIST CREATION ---
print("Creating a master list of all unique neuron IDs...")
all_ids = set()
all_ids.update(lockwood['root_id'])
all_ids.update(joa_clusters['root_id'])
all_ids.update(names['root_id'])
all_ids.update(consolidated_types['root_id'])
all_ids.update(princeton['pre_root_id'])
all_ids.update(princeton['post_root_id'])

completeness = pd.DataFrame(list(all_ids), columns=['root_id'])
completeness.dropna(inplace=True)

# --- 5. MERGING ANNOTATIONS ---
print("Merging all annotation data...")
# NOTE: Merges can reintroduce duplicates if source files have duplicate IDs
completeness = completeness.merge(lockwood, on='root_id', how='left')
completeness = completeness.merge(consolidated_types, on='root_id', how='left')
completeness = completeness.merge(joa_clusters, on='root_id', how='left')
completeness = completeness.merge(names, on='root_id', how='left')

# === STRICT DEDUPLICATION ===
# This ensures that even if merges created duplicates, we strip them before saving.
print(f"Rows before deduplication: {len(completeness)}")
completeness.drop_duplicates(subset=['root_id'], keep='first', inplace=True)
print(f"Rows after deduplication: {len(completeness)}")

# Generate labels
print("Generating labels...")
completeness['label'] = completeness['Neuron name (Lockwood et al)'].combine_first(
    completeness['additional_type(s)']
).combine_first(completeness['primary_type'])

completeness['label'] = completeness['label'].fillna(completeness['root_id'].astype(str))

# Select final columns
completeness = completeness[['root_id', 'label', 'Cluster']]
completeness = completeness.reset_index(drop=True)
completeness['index'] = completeness.index

id2idx = dict(zip(completeness['root_id'], completeness['index']))

# --- 6. PROCESS CONNECTIVITY ---
print("Processing connectivity indices...")
princeton['Presynaptic_Index'] = princeton['pre_root_id'].map(id2idx).fillna(-1).astype(int)
princeton['Postsynaptic_Index'] = princeton['post_root_id'].map(id2idx).fillna(-1).astype(int)

princeton.rename(columns={'pre_root_id': 'Presynaptic_ID', 'post_root_id':'Postsynaptic_ID', 'syn_count':'Connectivity'}, inplace=True)

nt_function_map = {'GABA': -1, 'ACH': 1, 'GLUT': -1, 'OCT': 1, 'SER': 1, 'DA': 1}
princeton['Excitatory'] = princeton['nt_type'].map(nt_function_map)
princeton['Connectivity x Excitatory'] = princeton['Connectivity'] * princeton['Excitatory']

connectivity = princeton[['Presynaptic_ID', 'Postsynaptic_ID', 'Presynaptic_Index', 'Postsynaptic_Index', 'Connectivity', 'Excitatory', 'Connectivity x Excitatory']]

print("Saving Parquet/CSV files...")
connectivity.to_parquet('data/CB_connectivity.parquet')
completeness.to_csv('data/CB_completeness.csv')

# --- 7. PROCESS NEURON RANGES (SIDELESS NEURON FIX) ---
print("Processing neuron ranges for grouping...")
lockwood_sided = lockwood.copy(deep=True)
lockwood_sided['Side'] = lockwood_sided['Neuron name (Lockwood et al)'].str.extract(r'_(R|L)')

# Logic to handle neurons without sides (e.g. WV-WV-1)
lockwood_sided["GroupKey"] = lockwood_sided.apply(
    lambda row: f"{row['Cell type (Lockwood et al)']}_{row['Side']}"
    if pd.notna(row['Side'])
    else row['Cell type (Lockwood et al)'],
    axis=1
)

neuron_ranges = lockwood_sided.groupby('GroupKey')['root_id'].apply(list).to_dict()

joa_clusters['Cluster'] = joa_clusters['Cluster'].apply(lambda x: f'Cluster{x}')
neuron_ranges.update(joa_clusters.groupby('Cluster')['root_id'].apply(list).to_dict())

with open('data/neuron_ranges.pkl', 'wb') as file:
  pickle.dump(neuron_ranges, file, protocol=pickle.HIGHEST_PROTOCOL)

print("Preprocessing complete.")