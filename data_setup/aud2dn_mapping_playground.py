import os
from fafbseg import flywire
from caveclient import CAVEclient
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from tqdm import tqdm
import tqdm
from datetime import datetime, timedelta
from tqdm import tqdm
from pyvis.network import Network
from tqdm.notebook import tqdm
from tqdm.auto import tqdm
import networkx as nx
import requests
from bs4 import BeautifulSoup
import re
import pickle


# CAVEclient authorization, code from the documentation
client = CAVEclient()
client.auth.save_token(token="5be2f6e6cfe49c7c7ccc0c8e791a60be", overwrite=True)

datastack_name = "flywire_fafb_production"
client = CAVEclient(datastack_name)

synapse_table = client.info.get_datastack_info()["synapse_table"]
print(synapse_table)

date = datetime.today().strftime("%Y%m%d")
todays_date = date[2:]
date_30d_ago = (datetime.today() - timedelta(days=300)).strftime("%Y%m%d")[2:]

print(date_30d_ago, todays_date)


def get_new_synapses(root_ids):
    inputs = root_ids

    # fetch connectivity
    syn0 = flywire.get_synapses(
        inputs,
        clean=True,
        filtered=True,
        transmitters=True,
        batch_size=7,
        min_score=50,
        materialization="latest",
    )

    # pair counts
    syn = pd.DataFrame(
        syn0.value_counts(["pre", "post"])
    )  # sum counts across unique post-presynaptic pairs
    syn.reset_index(inplace=True)
    syn = syn.rename(columns={0: "count"})

    return syn0, syn


def get_lab_aud_neurons(update=False):
    if update:
        lab_aud_neurons = pd.read_csv('data_setup/250730_lab_aud_ids_updated.csv')
        aud_table = lab_aud_neurons[['Neuron name','Codex July 2025']]
        updated_aud_id_table = flywire.update_ids(id=aud_table.iloc[:, 1])
        aud_table['new_id'] = updated_aud_id_table['new_id'].copy()
        aud_table.to_csv('{0}_lab_aud_ids_updated.csv'.format(todays_date))
    else:
        aud_table = pd.read_csv('data_setup/250730_lab_aud_ids_updated.csv', index_col=0)

    # print(aud_ids)
    return aud_table


def update_aud_neurons(update=False):
    aud_ids_path = "aud_ids/"
    aud_neuron_ids_file_path = "aud_ids/240215_aud_ids.csv"
    aud_table_path = "aud_table/"

    files_in_directory = os.listdir(aud_ids_path)
    if update:
        aud_ids = pd.read_csv(aud_neuron_ids_file_path)
        updated_aud_id_table = flywire.update_ids(id=aud_ids.iloc[:, 1])
        updated_aud_id_table.to_csv(
            aud_ids_path + "{0}_aud_ids.csv".format(todays_date)
        )
        print(updated_aud_id_table.head(10))
    else:
        latest_files = [
            files
            for files in files_in_directory
            if date_30d_ago <= files[:6] <= todays_date
        ]
        aud_id_table_fname = sorted(latest_files)[-1]
        updated_aud_id_table = pd.read_csv(
            aud_ids_path + "{0}".format(aud_id_table_fname)
        )
        print("Read: ", aud_id_table_fname)

    files_in_aud_info_directory = os.listdir(aud_table_path)
    if update:
        aud_table, aud_pairs = get_new_synapses(updated_aud_id_table["new_id"])
        aud_table.to_csv(aud_table_path + "{0}_aud_info_table.csv".format(todays_date))
        aud_pairs.to_csv(
            aud_table_path + "{0}_aud_pair_info_table.csv".format(todays_date)
        )
    else:
        latest_aud_files = [
            files
            for files in files_in_aud_info_directory
            if date_30d_ago <= files[:6] <= todays_date
        ]
        aud_info_table, aud_pair_info_table = sorted(latest_aud_files)[-2:]
        print("Read: ", aud_info_table, aud_pair_info_table)
        aud_table = pd.read_csv(aud_table_path + "{0}".format(aud_info_table))
        aud_pairs = pd.read_csv(aud_table_path + "{0}".format(aud_pair_info_table))

    return [aud_table, aud_pairs, updated_aud_id_table]


def update_dn_neurons(update=False):
    dn_neuron_ids_file_path = "data_setup/250730_dn_ids.csv"
    if update:
        dn_ids = pd.read_csv(dn_neuron_ids_file_path, sep=",", index_col=0)
        print(dn_ids)
        updated_dn_id_table = flywire.update_ids(id=dn_ids.iloc[1:, 0])
        updated_dn_id_table.to_csv("{0}_dn_ids.csv".format(todays_date))
        print(updated_dn_id_table.head(10))
    else:
        updated_dn_id_table = pd.read_csv('data_setup/250730_dn_ids.csv', index_col=0)

    # files_in_dn_info_directory = os.listdir(dn_table_path)
    # if update:
    #     dn_table, dn_pairs = get_new_synapses(updated_dn_id_table["new_id"])
    #     dn_table.to_csv(dn_table_path + "{0}_dn_info_table.csv".format(todays_date))
    #     dn_pairs.to_csv(
    #         dn_table_path + "{0}_dn_pair_info_table.csv".format(todays_date)
    #     )
    # else:
    #     latest_dn_files = [
    #         files
    #         for files in files_in_dn_info_directory
    #         if date_30d_ago <= files[:6] <= todays_date
    #     ]
    #     dn_info_table, dn_pair_info_table = sorted(latest_dn_files)[-2:]
    #     dn_table = pd.read_csv(dn_table_path + "{0}".format(dn_info_table))
    #     dn_pairs = pd.read_csv(dn_table_path + "{0}".format(dn_pair_info_table))
    #     print("Read: ", dn_info_table, dn_pair_info_table)
    return updated_dn_id_table
    # return [dn_table, dn_pairs, updated_dn_id_table]


def update_connectivity_table(updated_dn_id_table):
    connectivity_df = pd.read_csv("240130_aud_connectivity.csv")
    aud_dn_connectivity_df = pd.DataFrame(columns=["pre", "post", "weight"])
    for dn_id in tqdm(updated_dn_id_table["new_id"].values):
        aud_dn_pairs = connectivity_df[(connectivity_df["post"]) == dn_id]
        for idx, row in aud_dn_pairs.iterrows():
            new_row = {
                "pre": row["pre"],
                "post": row["post"],
                "weight": row["weight"],
            }
            aud_dn_connectivity_df.loc[len(aud_dn_connectivity_df)] = new_row
    print(len(connectivity_df))
    return connectivity_df


def create_complete_adj_matrix(connectivity_list):
    uq_aud_ids = np.unique([edge[0] for edge in connectivity_list])
    uq_dn_ids = np.unique([edge[1] for edge in connectivity_list])

    adj_matrix = np.zeros((len(uq_aud_ids), len(uq_dn_ids)), dtype=int)
    print("Creating Adjacency Matrix")
    for edge in tqdm(connectivity_list):
        aud_id = edge[0]
        dn_id = edge[1]
        weight = edge[2]

        vertical_idx = np.where(aud_id == uq_aud_ids)[0][0]
        horizontal_idx = np.where(uq_dn_ids == dn_id)[0][0]
        adj_matrix[vertical_idx, horizontal_idx] = weight

    print(np.shape(adj_matrix))

    return [adj_matrix, uq_aud_ids, uq_dn_ids]


def update_group_ref_book():
    group_ref_df = pd.read_csv("ref_ids.csv")
    source_ids = group_ref_df.iloc[:570, -2]
    source_ids = source_ids.dropna()
    source_ids = source_ids.astype("float64").astype("int64").astype("str")
    print(source_ids)
    updated_ids = flywire.update_ids(id=source_ids)
    print(updated_ids)
    return updated_ids


def get_aud_dn_ids(updated_aud_id_table, updated_dn_id_table):
    aud_ids = updated_aud_id_table["new_id"].values
    dn_ids = updated_dn_id_table["new_id"].values
    str_aud_ids = list(map(str, aud_ids.tolist()))
    str_dn_ids = list(map(str, dn_ids.tolist()))
    return [aud_ids, dn_ids, str_aud_ids, str_dn_ids]


def create_aud_dn_graph(aud_ids, dn_ids, str_aud_ids, str_dn_ids):
    aud_dn_graph = nx.DiGraph()
    aud_dn_graph.add_nodes_from(str_aud_ids, node_type="Auditory_Neurons")
    aud_dn_graph.add_nodes_from(str_dn_ids, node_type="Descending_Neurons")

    connectivity_df = pd.read_csv("240130_aud_connectivity.csv")
    adj_matrix_list = []
    count = 0
    threshold = 5

    for aud in tqdm(aud_ids):
        filtered_rows = connectivity_df[connectivity_df["pre"] == aud]
        post_values = filtered_rows["post"].tolist()
        for neuron in post_values:
            if neuron in dn_ids:
                aud_dn_pair_row = connectivity_df.loc[
                    (connectivity_df["pre"] == aud)
                    & (connectivity_df["post"] == neuron)
                ]
                weight = aud_dn_pair_row["weight"].values[0]
                count += 1
                if int(weight) > threshold:
                    adj_matrix_list.append([aud, neuron, int(weight)])
                    aud_dn_graph.add_edge(str(aud), str(neuron), weight=int(weight))

    print(count, len(adj_matrix_list))
    adj_matrix, uq_aud_ids, uq_dn_ids = create_complete_adj_matrix(adj_matrix_list)

    nodes_to_remove = [
        node
        for node in aud_dn_graph.nodes()
        if sum(
            aud_dn_graph.get_edge_data(edge[0], edge[1])["weight"]
            for edge in aud_dn_graph.in_edges(node)
        )
        + sum(
            aud_dn_graph.get_edge_data(edge[0], edge[1])["weight"]
            for edge in aud_dn_graph.out_edges(node)
        )
        < threshold
    ]

    aud_dn_graph.remove_nodes_from(nodes_to_remove)

    return [aud_dn_graph, connectivity_df, adj_matrix, uq_aud_ids, uq_dn_ids]


def create_normalized_adj_matrix(adj_matrix, uq_aud_ids, uq_dn_ids, update=False):
    if update:
        norm_matrix = adj_matrix.astype(float).copy()
        dn_connectivity_df = pd.read_csv("data_setup/250730_dn_connectivity.csv")
        grouped_data = dn_connectivity_df.groupby("post")["weight"].sum()
        resulting_dict = {}

        for post_id, total_weight in grouped_data.items():
            resulting_dict[post_id] = total_weight

        for dn_id in tqdm(uq_dn_ids):
            for aud_id in uq_aud_ids:
                (aud_idx,) = np.where(uq_aud_ids == aud_id)
                (dn_idx,) = np.where(uq_dn_ids == dn_id)
                norm_matrix.iloc[aud_idx, dn_idx] = np.float16(adj_matrix.iloc[aud_idx, dn_idx] / resulting_dict[int(dn_id)])

        max_values = norm_matrix.apply(max, axis=1)
        max_val = max(max_values)
        min_values = norm_matrix.apply(min, axis=1)
        min_val = max(min_values)
        updated_norm_matrix = norm_matrix.map(lambda x: ((x - min_val) / (max_val - min_val)))
        updated_norm_matrix.to_csv('{0}_normalized_matrix.csv'.format(todays_date))
    else:
        updated_norm_matrix = pd.read_csv('data_setup/250730_normalized_matrix.csv', index_col=0)

    return updated_norm_matrix


def make_pyvis_graph(graph):
    net = Network(select_menu=True, filter_menu=True)
    net.from_nx(graph)
    net.show_buttons()

    for node in net.get_nodes():
        data = net.get_node(node)
        if data["node_type"] == "Auditory_Neurons":
            data["color"] = "#70b922"
        elif data["node_type"] == "Descending_Neurons":
            data["color"] = "#b9224a"

    return net


def make_heatmap(adj_matrix, norm_matrix, xticks, yticks):
    tick_font_size = 10
    label_tick_font_size = 10
    header_font_size = 50
    fig, axs = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(200, 200))

    im1 = axs[0].imshow(adj_matrix, cmap="inferno", interpolation="none")
    axs[0].set_title("Adjacency Matrix", fontsize=header_font_size)
    axs[0].set_xticks(np.arange(len(xticks)), xticks, rotation=90, fontsize=label_tick_font_size)
    axs[0].set_xticklabels(xticks, rotation=90, ha='center', fontsize=label_tick_font_size)
    axs[0].set_yticks(np.arange(len(yticks)), yticks, fontsize=label_tick_font_size)
    axs[0].set_yticklabels(yticks, fontsize=label_tick_font_size)
    axs[0].set_xlabel("Descending Ids", fontsize=20)
    axs[0].set_ylabel("Auditory Ids", fontsize=20)
    cb1 = fig.colorbar(im1, ax=axs[0])
    cb1.ax.tick_params(labelsize=tick_font_size)

    im2 = axs[1].imshow(norm_matrix, cmap="inferno", interpolation="none")
    axs[1].set_title("Normalized Adjacency Matrix", fontsize=header_font_size)
    axs[1].set_xticks(np.arange(len(xticks)), xticks, rotation=90, fontsize=label_tick_font_size)
    axs[1].set_xticklabels(xticks, rotation=90, ha='center', fontsize=label_tick_font_size)
    axs[1].set_yticks(np.arange(len(yticks)), yticks, fontsize=label_tick_font_size)
    axs[1].set_yticklabels(yticks, fontsize=label_tick_font_size)
    axs[1].set_xlabel("Descending Ids", fontsize=20)
    axs[1].set_ylabel("Auditory Ids", fontsize=20)
    cb2 = fig.colorbar(im2, ax=axs[1])
    cb2.ax.tick_params(labelsize=tick_font_size)


    plt.savefig('heatmap_50_threshold.png')
    plt.show()


def scrape_neuron_names(id):
    cookies = {
        'g_state': '{"i_p":1701196386976,"i_l":2}',
        '_ga_PRYCNHR0ZH': 'GS1.1.1701281186.1.1.1701281259.0.0.0',
        '_ga': 'GA1.1.1170215572.1700599937',
        '_ga_CQFQDNDHNZ': 'GS1.1.1708020879.18.0.1708020879.0.0.0',
        'session': '.eJytVNtO4zAU_JXKz6WNnaRJKq1EQerLAgKxLCtVKHIduzF17CixW7pV_32P0wsUeNw3X8Yzk3PmZIvymjcV1VxbNLaN431UUEtzyhhv27ymG2VogcZbRItKajQWVLUAokJIJamVRrdoPJu9vPQRr6hUaIxarsLLupGacWv0gBcO9dGiMa72UFRwQZ2ycCbUZi0bDqtbqmh-6xpbbvIbOkfAJkGVkBT3USXbVupFbo1_v-0MttzmHpH0T1tNKw7qlWQNmMprN1eSATc866Bkv_wetgPFw82jqUvJezeGLdfGFMBQ0wbq05FopxQcQM28qe7jt2g6ubvOW6qLuXnrdKbTOzQG5zdOF91CUDHvbvziDHqowdGHB9dSLzc4COB-d6aVr8hXuRnihfTVXEm-9oXrxGen7d7D-35v5curc2Nfrz_ZfOd7d_vpkfcuoZ6-t719b3sKetuHgDQrybgPmXE-eF2odufRs2bJIW8ooZgwktEgDaNRmAke0iyNRpjFJB7RFCMflVxqYbqUOvhYFMajJEgSgOPgQmVmJRJBqCUrRkIROvIKAGqWTcTnWTagdd0OFsYsFHdgjRltod0DZirgpn_r_0t4nBJaU1vSZXSpWesOQ9Ld5SveSCF5cRxI_gYWcBJgjMM469JUSbU5Jvm-4_EjJldcH08nG9eWcFj6enyQgJk9kAVZ5skgW4Aora3b8XB4aMnR_sHzq_WNpHHMwALPMh6mmERRkEQkCFkkeFFAW6BvOE7wCJ1GqTPROxnUc3HSHmVd0pl1Df-gr8rw-8oN6XByvUgN-_k8uW_S9u3xKZo8_q02f6Z3Dw25egp_23r961mVycPk6oFr8fSjzUYX_g_QOkg8wkGYjOI0IFFK0gTjJIwygna7f2_on4k.Zc5fcQ.Mr60AqUGmGyvlX3O9VDsT9EPhbg',
    }

    headers = {
        'authority': 'codex.flywire.ai',
        'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
        'accept-language': 'en-US,en;q=0.9',
        'cache-control': 'max-age=0',
        # 'cookie': 'g_state={"i_p":1701196386976,"i_l":2}; _ga_PRYCNHR0ZH=GS1.1.1701281186.1.1.1701281259.0.0.0; _ga=GA1.1.1170215572.1700599937; _ga_CQFQDNDHNZ=GS1.1.1708020879.18.0.1708020879.0.0.0; session=.eJytVNtO4zAU_JXKz6WNnaRJKq1EQerLAgKxLCtVKHIduzF17CixW7pV_32P0wsUeNw3X8Yzk3PmZIvymjcV1VxbNLaN431UUEtzyhhv27ymG2VogcZbRItKajQWVLUAokJIJamVRrdoPJu9vPQRr6hUaIxarsLLupGacWv0gBcO9dGiMa72UFRwQZ2ycCbUZi0bDqtbqmh-6xpbbvIbOkfAJkGVkBT3USXbVupFbo1_v-0MttzmHpH0T1tNKw7qlWQNmMprN1eSATc866Bkv_wetgPFw82jqUvJezeGLdfGFMBQ0wbq05FopxQcQM28qe7jt2g6ubvOW6qLuXnrdKbTOzQG5zdOF91CUDHvbvziDHqowdGHB9dSLzc4COB-d6aVr8hXuRnihfTVXEm-9oXrxGen7d7D-35v5curc2Nfrz_ZfOd7d_vpkfcuoZ6-t719b3sKetuHgDQrybgPmXE-eF2odufRs2bJIW8ooZgwktEgDaNRmAke0iyNRpjFJB7RFCMflVxqYbqUOvhYFMajJEgSgOPgQmVmJRJBqCUrRkIROvIKAGqWTcTnWTagdd0OFsYsFHdgjRltod0DZirgpn_r_0t4nBJaU1vSZXSpWesOQ9Ld5SveSCF5cRxI_gYWcBJgjMM469JUSbU5Jvm-4_EjJldcH08nG9eWcFj6enyQgJk9kAVZ5skgW4Aora3b8XB4aMnR_sHzq_WNpHHMwALPMh6mmERRkEQkCFkkeFFAW6BvOE7wCJ1GqTPROxnUc3HSHmVd0pl1Df-gr8rw-8oN6XByvUgN-_k8uW_S9u3xKZo8_q02f6Z3Dw25egp_23r961mVycPk6oFr8fSjzUYX_g_QOkg8wkGYjOI0IFFK0gTjJIwygna7f2_on4k.Zc5fcQ.Mr60AqUGmGyvlX3O9VDsT9EPhbg',
        'referer': 'https://codex.flywire.ai/app/cell_details?cell_names_or_id=720575940626232153',
        'sec-ch-ua': '"Not A(Brand";v="99", "Google Chrome";v="121", "Chromium";v="121"',
        'sec-ch-ua-arch': '"x86"',
        'sec-ch-ua-bitness': '"64"',
        'sec-ch-ua-full-version-list': '"Not A(Brand";v="99.0.0.0", "Google Chrome";v="121.0.6167.162", "Chromium";v="121.0.6167.162"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-model': '""',
        'sec-ch-ua-platform': '"Windows"',
        'sec-ch-ua-platform-version': '"10.0.0"',
        'sec-ch-ua-wow64': '?0',
        'sec-fetch-dest': 'document',
        'sec-fetch-mode': 'navigate',
        'sec-fetch-site': 'same-origin',
        'sec-fetch-user': '?1',
        'upgrade-insecure-requests': '1',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
    }

    response = requests.get("https://codex.flywire.ai/app/cell_details?cell_names_or_id={0}".format(id), headers=headers, cookies=cookies)
    soup = BeautifulSoup(response.content, "html.parser")
    td_tags = soup.find_all('td')
    if not td_tags:
        return ''
    else:
        cell_name = re.sub(r'<[^>]*>', '', str(td_tags[1]))

    return cell_name


def get_annotations(id_list, name):
    annotated_list = []


    if name == "dn_ids":
        annotations_list = flywire.search_annotations(id_list, materialization='latest')
        for id in tqdm(id_list):
            annotations = annotations_list[annotations_list['root_id'] == id]
            if (any(annotations['cell_type'].notna()) or any(annotations['hemibrain_type'].notna())) and not annotations.empty:
                if all(annotations['cell_type'].notna()):
                    annotated_list.append(str(annotations['cell_type'].iloc[0]) + ' ' + str(annotations['side'].iloc[0]))
                else : 
                    annotated_list.append(str(annotations['hemibrain_type'].iloc[0]) + ' ' + str(annotations['side'].iloc[0]))
            else:
                neuron_name = scrape_neuron_names(id)
                if neuron_name:
                    annotated_list.append(neuron_name)
                else :
                    annotated_list.append(id)
    elif name == 'aud_ids':
        lab_aud_neurons = get_lab_aud_neurons(update=False)
        matching_rows = lab_aud_neurons[lab_aud_neurons['new_id'].isin(id_list)]
        # print(matching_rows)
        annotated_list = matching_rows['Neuron name'].tolist()
    else:
        annotated_list = id_list
    return annotated_list


def get_desc_connectivity(dn_list):
    dn_conn = flywire.synapses.get_connectivity(dn_list, materialization='latest', upstream=True, downstream=False)
    dn_conn.to_csv('{0}_dn_connectivity.csv'.format(todays_date))


def update_adjacency_id_based(aud_ids, dn_ids, update=False):
    aud_cum_threshold = dn_cum_threshold = 1
    if not update:
        adj_matrix = pd.read_csv('data_setup/250730_adjacency_matrix.csv', index_col=0)
    else:
        adj_matrix = flywire.synapses.get_adjacency(aud_ids, dn_ids, materialization='latest')
        adj_matrix.to_csv('{0}_adjacency_matrix.csv'.format(todays_date))

    adj_matrix_filtered_aud = adj_matrix.loc[:, adj_matrix.sum(axis=0) >= aud_cum_threshold].copy()

    adj_matrix_filtered_dn = adj_matrix_filtered_aud.loc[adj_matrix_filtered_aud.sum(axis=1) >= dn_cum_threshold].copy()

    filtered_aud_ids = adj_matrix_filtered_dn.index
    filtered_dn_ids = adj_matrix_filtered_dn.columns
    return adj_matrix_filtered_dn, filtered_aud_ids, filtered_dn_ids


def create_aggregate_vectors(adj_matrix, update=False):
    if update:
        col_sum = adj_matrix.sum(axis=0)
        dn_connectivity_df = pd.read_csv("data_setup/250730_dn_connectivity.csv")
        grouped_data = dn_connectivity_df.groupby("post")["weight"].sum()
        
        # Create a new Series to store the results
        normalized_col_sum = col_sum.copy()
        for id in col_sum.index:
            normalized_col_sum.loc[str(id)] = col_sum.loc[str(id)] / grouped_data.loc[int(id)]
        
        col_min = normalized_col_sum.min()
        col_max = normalized_col_sum.max()
        normalized_series = (normalized_col_sum - col_min)/(col_max - col_min)
        normalized_series.to_csv('{0}_normalized_col_sum_vector.csv'.format(todays_date))
    else:
        normalized_series = pd.read_csv('data_setup/250730_normalized_col_sum_vector.csv')
    
    return normalized_series




def final_run():
    updated_aud_id_table = get_lab_aud_neurons(update=False)
    updated_dn_id_table = update_dn_neurons(update=False)
    aud_ids, dn_ids, str_aud_ids, str_dn_ids = get_aud_dn_ids(updated_aud_id_table, updated_dn_id_table)
    print('Updating Adjacency')
    adj_matrix, filtered_aud_ids, filtered_dn_ids = update_adjacency_id_based(aud_ids, dn_ids, update=False)
    # get_desc_connectivity(dn_ids)

    print("Creating Normalized matrix")
    normalized_matrix = create_normalized_adj_matrix(adj_matrix, filtered_aud_ids, filtered_dn_ids, update=False)
    norm_idx, norm_cols = normalized_matrix.index, normalized_matrix.columns
    adj_idx, adj_cols = adj_matrix.index, adj_matrix.columns
    dn_name_list, aud_name_list = [], []

    print("Creating aggregation columns")
    create_aggregate_vectors(adj_matrix, update=False)

    update_annotations = True
    update_matrices = True

    print('Getting cell types')
    if norm_idx.equals(adj_idx) and norm_cols.equals(adj_cols):
        print('Fetching AUD Annotations')
        if update_annotations:
            norm_idx = norm_idx.astype('int64')
            norm_cols = norm_cols.astype('int64').tolist()
            aud_name_list = get_annotations(norm_idx, name='aud_ids')
            # aud_name_list.sort(key=lambda x: x.lower())
            print('Fetching DN Annotations')
            dn_name_list = get_annotations(norm_cols, name='dn_ids')
            # dn_name_list.sort(key=lambda x: x.lower())
            with open("data_setup/aud_name_list", "wb") as fp:   #Pickling
                pickle.dump(aud_name_list, fp)
            with open("data_setup/dn_name_list", "wb") as dp:   #Pickling
                pickle.dump(dn_name_list, dp)
        else:
            with open("data_setup/aud_name_list", "rb") as fp:   #UnPickling
                aud_name_list = pickle.load(fp)
            with open("data_setup/dn_name_list", "rb") as dp:   #UnPickling
                dn_name_list = pickle.load(dp)

    else:
        print('IDX and COLS not same')
        return
    normalized_matrix.columns, adj_matrix.columns = dn_name_list, dn_name_list
    normalized_matrix.index, adj_matrix.index = aud_name_list, aud_name_list
    adj_matrix.sort_index(axis=1, inplace=True, key=lambda x: x.str.lower())
    adj_matrix.sort_index(axis=0, inplace=True, key=lambda x: x.str.lower())
    normalized_matrix.sort_index(axis=1, inplace=True, key=lambda x: x.str.lower())
    normalized_matrix.sort_index(axis=0, inplace=True, key=lambda x: x.str.lower())


    if update_matrices:
        adj_matrix.to_csv('data_setup/{0}_final_adj_matrix.csv'.format(todays_date))
        normalized_matrix.to_csv('data_setup/{0}_final_norm_matrix.csv'.format(todays_date))

    print('Making Heatmaps')
    make_heatmap(adj_matrix, normalized_matrix, adj_matrix.columns, adj_matrix.index)


if __name__ == "__main__":
  final_run()



'''
link_elements = str(soup.select("td"))
        anti_small_pattern  = "<small>(.*?)</small>"
        cleaned_html_content = re.sub(anti_small_pattern, "", link_elements)
        side_pattern = r"Side: <b>(.*?)</b>"
        cell_type_pattern = r""
        matches = re.findall(side_pattern, link_elements)
        # print(matches)
        print(cleaned_html_content)
        # print(id, link_elements)
        # print(id, str(link_elements[1])[4:-5])
        # try:
        #     id_names_dict.append(str(link_elements[1])[4:-5])
        # except:
        #     print('not found')
        #     id_names_dict.append(str(id))
'''
