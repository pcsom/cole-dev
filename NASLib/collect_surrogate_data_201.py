#read every json file in run_nb201/nasbench201/cifar-10/LLM_NB201_Predictor_CustomMLP_bananas

import json
import os
import numpy as np
import pandas as pd

def compute_nb201_index(op_indices):
    index = 0
    for i, op in enumerate(op_indices):
        index += int(op) * (5 ** i)
    return index

# Define the path to the directory containing the JSON files
directory_path10 = 'run_nb201/run_nb201/nasbench201/cifar10/LLM_NB201_Predictor_CustomMLP_bananas'
directory_path100 = 'run_nb201/run_nb201/nasbench201/cifar100/LLM_NB201_Predictor_CustomMLP_bananas'
# Initialize an empty list to store the data
data_list10 = []
data_list100 = []
trajectory_list10 = []
trajectory_list100 = []
# Iterate through each file  and direcory in the directory
for root, dirs, files in os.walk(directory_path10):
    for filename in files:
        if filename.endswith(".json") and filename.startswith("candidate_log"):
            file_path = os.path.join(root, filename)
            with open(file_path, 'r') as f:
                data = json.load(f)
                data_list10.append(data)
        elif filename.endswith(".json") and filename.startswith("trajectory"):
            file_path = os.path.join(root, filename)
            with open(file_path, 'r') as f:
                data = json.load(f)
                trajectory_list10.append(data)
for root, dirs, files in os.walk(directory_path100):
    for filename in files:
        if filename.endswith(".json") and filename.startswith("candidate_log"):
            file_path = os.path.join(root, filename)
            with open(file_path, 'r') as f:
                data = json.load(f)
                data_list100.append(data)
        elif filename.endswith(".json") and filename.startswith("trajectory"):
            file_path = os.path.join(root, filename)
            with open(file_path, 'r') as f:
                data = json.load(f)
                trajectory_list100.append(data)
print(f"Total number of candidate_log JSON files read: {len(data_list10) + len(data_list100)}")
print(f"Total number of trajectory JSON files read: {len(trajectory_list10) + len(trajectory_list100)}")
#create a dictionary with the following structure: "arch_index": {"predicted_accuracy" "generationfirstdiscovery", "generation_lastdiscovery"} match predicted accuracy to the entry with last discovery and first discovery
arch_dict_10 = {}
for data in data_list10:
    for item in data:
        arch_index = item['arch_index']
        predicted_accuracy = item['predicted_accuracy']
        generation_firstdiscovery = item['generation']
        generation_lastdiscovery = item['generation']
        if arch_index not in arch_dict_10:
            arch_dict_10[arch_index] = {"predicted_accuracy": predicted_accuracy, "generation_firstdiscovery": generation_firstdiscovery, "generation_lastdiscovery": generation_lastdiscovery, "showed_up": 1, "average_predicted_accuracy": predicted_accuracy, "average_generation": generation_firstdiscovery}
        else:
            arch_dict_10[arch_index]["showed_up"] += 1
            arch_dict_10[arch_index]["average_predicted_accuracy"] = (arch_dict_10[arch_index]["average_predicted_accuracy"] * (arch_dict_10[arch_index]["showed_up"] - 1) + predicted_accuracy) / arch_dict_10[arch_index]["showed_up"]
            arch_dict_10[arch_index]["average_generation"] = (arch_dict_10[arch_index]["average_generation"] * (arch_dict_10[arch_index]["showed_up"] - 1) + generation_firstdiscovery) / arch_dict_10[arch_index]["showed_up"]
            if generation_firstdiscovery < arch_dict_10[arch_index]["generation_firstdiscovery"]:
                arch_dict_10[arch_index]["generation_firstdiscovery"] = generation_firstdiscovery
            if generation_lastdiscovery > arch_dict_10[arch_index]["generation_lastdiscovery"]:
                arch_dict_10[arch_index]["generation_lastdiscovery"] = generation_lastdiscovery
                arch_dict_10[arch_index]["predicted_accuracy"] = predicted_accuracy
    
    #convert to pandas dataframe
arch_dict_100 = {}
for data in data_list100:
    for item in data:
        arch_index = item['arch_index']
        predicted_accuracy = item['predicted_accuracy']
        generation_firstdiscovery = item['generation']
        generation_lastdiscovery = item['generation']
        if arch_index not in arch_dict_100:
            arch_dict_100[arch_index] = {"predicted_accuracy": predicted_accuracy, "generation_firstdiscovery": generation_firstdiscovery, "generation_lastdiscovery": generation_lastdiscovery, "showed_up": 1, "average_predicted_accuracy": predicted_accuracy, "average_generation": generation_firstdiscovery}
        else:
            arch_dict_100[arch_index]["showed_up"] += 1
            arch_dict_100[arch_index]["average_predicted_accuracy"] = (arch_dict_100[arch_index]["average_predicted_accuracy"] * (arch_dict_100[arch_index]["showed_up"] - 1) + predicted_accuracy) / arch_dict_100[arch_index]["showed_up"]
            arch_dict_100[arch_index]["average_generation"] = (arch_dict_100[arch_index]["average_generation"] * (arch_dict_100[arch_index]["showed_up"] - 1) + generation_firstdiscovery) / arch_dict_100[arch_index]["showed_up"]
            if generation_firstdiscovery < arch_dict_100[arch_index]["generation_firstdiscovery"]:
                arch_dict_100[arch_index]["generation_firstdiscovery"] = generation_firstdiscovery
            if generation_lastdiscovery > arch_dict_100[arch_index]["generation_lastdiscovery"]:
                arch_dict_100[arch_index]["generation_lastdiscovery"] = generation_lastdiscovery
                arch_dict_100[arch_index]["predicted_accuracy"] = predicted_accuracy

# --- TRAJECTORY DICTS (only the chosen/evaluated individuals) ---
# Each trajectory entry has: op_indices, generation, predicted_accuracy, true_accuracy
def build_trajectory_dict(trajectory_list):
    traj_dict = {}
    for data in trajectory_list:
        for item in data:
            if item.get('op_indices') is None:
                continue
            arch_index = compute_nb201_index(item['op_indices'])
            generation = item['generation']
            predicted_accuracy = item.get('predicted_accuracy')
            true_accuracy = item.get('true_accuracy')
            if arch_index not in traj_dict:
                traj_dict[arch_index] = {
                    "true_accuracy": true_accuracy,
                    "predicted_accuracy": predicted_accuracy,
                    "generation_firstchosen": generation,
                    "generation_lastchosen": generation,
                    "times_chosen": 1,
                    "average_predicted_accuracy": predicted_accuracy if predicted_accuracy is not None else 0,
                    "average_generation": generation
                }
            else:
                traj_dict[arch_index]["times_chosen"] += 1
                n = traj_dict[arch_index]["times_chosen"]
                if predicted_accuracy is not None:
                    traj_dict[arch_index]["average_predicted_accuracy"] = (
                        traj_dict[arch_index]["average_predicted_accuracy"] * (n - 1) + predicted_accuracy
                    ) / n
                traj_dict[arch_index]["average_generation"] = (
                    traj_dict[arch_index]["average_generation"] * (n - 1) + generation
                ) / n
                if generation < traj_dict[arch_index]["generation_firstchosen"]:
                    traj_dict[arch_index]["generation_firstchosen"] = generation
                if generation > traj_dict[arch_index]["generation_lastchosen"]:
                    traj_dict[arch_index]["generation_lastchosen"] = generation
                    traj_dict[arch_index]["true_accuracy"] = true_accuracy
                    traj_dict[arch_index]["predicted_accuracy"] = predicted_accuracy
    return traj_dict

traj_dict_10 = build_trajectory_dict(trajectory_list10)
traj_dict_100 = build_trajectory_dict(trajectory_list100)
print(f"Unique trajectory architectures (cifar10): {len(traj_dict_10)}")
print(f"Unique trajectory architectures (cifar100): {len(traj_dict_100)}")

#now join the two dictionaries on the arch_index and create a pandas dataframe with the following columns: "arch_index", "predicted_accuracy_cifar10", "generation_firstdiscovery_cifar10", "generation_lastdiscovery_cifar10", "predicted_accuracy_cifar100", "generation_firstdiscovery_cifar100", "generation_lastdiscovery_cifar100"
def get_traj_fields(traj_dict, arch_index, suffix):
    if arch_index in traj_dict:
        t = traj_dict[arch_index]
        return {
            f"traj_true_accuracy_{suffix}": t["true_accuracy"],
            f"traj_predicted_accuracy_{suffix}": t["predicted_accuracy"],
            f"traj_generation_firstchosen_{suffix}": t["generation_firstchosen"],
            f"traj_generation_lastchosen_{suffix}": t["generation_lastchosen"],
            f"traj_times_chosen_{suffix}": t["times_chosen"],
            f"traj_avg_predicted_accuracy_{suffix}": t["average_predicted_accuracy"],
            f"traj_avg_generation_{suffix}": t["average_generation"],
        }
    else:
        return {
            f"traj_true_accuracy_{suffix}": -1000,
            f"traj_predicted_accuracy_{suffix}": -1000,
            f"traj_generation_firstchosen_{suffix}": -1000,
            f"traj_generation_lastchosen_{suffix}": -1000,
            f"traj_times_chosen_{suffix}": -1000,
            f"traj_avg_predicted_accuracy_{suffix}": -1000,
            f"traj_avg_generation_{suffix}": -1000,
        }

data_combined = []
all_arch_indices = set(arch_dict_10.keys()) | set(arch_dict_100.keys())
for arch_index in all_arch_indices:
    row = {"arch_index": arch_index}

    # Candidate log fields
    if arch_index in arch_dict_10:
        row.update({
            "predicted_accuracy_cifar10": arch_dict_10[arch_index]["predicted_accuracy"],
            "generation_firstdiscovery_cifar10": arch_dict_10[arch_index]["generation_firstdiscovery"],
            "generation_lastdiscovery_cifar10": arch_dict_10[arch_index]["generation_lastdiscovery"],
            "average_predicted_accuracy_cifar10": arch_dict_10[arch_index]["average_predicted_accuracy"],
            "average_generation_cifar10": arch_dict_10[arch_index]["average_generation"],
            "showed_up_cifar10": arch_dict_10[arch_index]["showed_up"],
        })
    else:
        row.update({
            "predicted_accuracy_cifar10": -1000,
            "generation_firstdiscovery_cifar10": -1000,
            "generation_lastdiscovery_cifar10": -1000,
            "average_predicted_accuracy_cifar10": -1000,
            "average_generation_cifar10": -1000,
            "showed_up_cifar10": -1000,
        })

    if arch_index in arch_dict_100:
        row.update({
            "predicted_accuracy_cifar100": arch_dict_100[arch_index]["predicted_accuracy"],
            "generation_firstdiscovery_cifar100": arch_dict_100[arch_index]["generation_firstdiscovery"],
            "generation_lastdiscovery_cifar100": arch_dict_100[arch_index]["generation_lastdiscovery"],
            "average_predicted_accuracy_cifar100": arch_dict_100[arch_index]["average_predicted_accuracy"],
            "average_generation_cifar100": arch_dict_100[arch_index]["average_generation"],
            "showed_up_cifar100": arch_dict_100[arch_index]["showed_up"],
        })
    else:
        row.update({
            "predicted_accuracy_cifar100": -1000,
            "generation_firstdiscovery_cifar100": -1000,
            "generation_lastdiscovery_cifar100": -1000,
            "average_predicted_accuracy_cifar100": -1000,
            "average_generation_cifar100": -1000,
            "showed_up_cifar100": -1000,
        })

    # Trajectory fields (only chosen/evaluated individuals)
    row.update(get_traj_fields(traj_dict_10, arch_index, "cifar10"))
    row.update(get_traj_fields(traj_dict_100, arch_index, "cifar100"))

    data_combined.append(row)
df_combined = pd.DataFrame(data_combined)

#save to csv
# print("Saving to CSV... at /storage/ice-shared/vip-vvk/data/AOT/mgullapalli6/codenas/collecteddata.csv")
#print the first 5 rows of the dataframe
print(df_combined.head())   
#unify this csv with the one at /storage/ice-shared/vip-vvk/data/AOT/psomu3/codenas2/nasbench201_corpus_onnx_paper_embedded,csv and save it as collecteddata_unified.csv and also keep all data in the psomu file, and for any
#rows in psomu file that are not in the combined file, fill the predicted accuracy and generation discovery fields with NaN, and the discovery file with 600.
df_psomu = pd.read_csv('/storage/ice-shared/vip-vvk/data/AOT/psomu3/codenas/nasbench201_corpus_pytorch_corrected.csv')
df_unified = pd.merge(df_psomu, df_combined, on='arch_index', how='left')
#fill NaN values in the predicted accuracy and generation discovery fields with -1000
df_unified['predicted_accuracy_cifar10'] = df_unified['predicted_accuracy_cifar10'].fillna(-1000)
df_unified['generation_firstdiscovery_cifar10'] = df_unified['generation_firstdiscovery_cifar10'].fillna(-1000)
df_unified['generation_lastdiscovery_cifar10'] = df_unified['generation_lastdiscovery_cifar10'].fillna(-1000)
df_unified['predicted_accuracy_cifar100'] = df_unified['predicted_accuracy_cifar100'].fillna(-1000)
df_unified['generation_firstdiscovery_cifar100'] = df_unified['generation_firstdiscovery_cifar100'].fillna(-1000)
df_unified['generation_lastdiscovery_cifar100'] = df_unified['generation_lastdiscovery_cifar100'].fillna(-1000) 
df_unified['average_predicted_accuracy_cifar10'] = df_unified['average_predicted_accuracy_cifar10'].fillna(-1000)
df_unified['average_generation_cifar10'] = df_unified['average_generation_cifar10'].fillna(-1000)
df_unified['average_predicted_accuracy_cifar100'] = df_unified['average_predicted_accuracy_cifar100'].fillna(-1000)
df_unified['average_generation_cifar100'] = df_unified['average_generation_cifar100'].fillna(-1000)
df_unified['showed_up_cifar10'] = df_unified['showed_up_cifar10'].fillna(-1000)
df_unified['showed_up_cifar100'] = df_unified['showed_up_cifar100'].fillna(-1000)

# Fill trajectory columns
traj_cols = [
    'traj_true_accuracy_cifar10', 'traj_predicted_accuracy_cifar10',
    'traj_generation_firstchosen_cifar10', 'traj_generation_lastchosen_cifar10',
    'traj_times_chosen_cifar10', 'traj_avg_predicted_accuracy_cifar10', 'traj_avg_generation_cifar10',
    'traj_true_accuracy_cifar100', 'traj_predicted_accuracy_cifar100',
    'traj_generation_firstchosen_cifar100', 'traj_generation_lastchosen_cifar100',
    'traj_times_chosen_cifar100', 'traj_avg_predicted_accuracy_cifar100', 'traj_avg_generation_cifar100',
]
for col in traj_cols:
    if col in df_unified.columns:
        df_unified[col] = df_unified[col].fillna(-1000)
print(df_unified.head())

import os
# user env variable
user = os.getenv("USER")
df_unified.to_csv(f'/storage/ice-shared/vip-vvk/data/AOT/{user}/codenas/collecteddata_unified.csv', index=False)