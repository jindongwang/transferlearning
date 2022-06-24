import os, sys
import os.path as osp
import pandas as pd
from pdb import set_trace as st
import numpy as np
np.set_printoptions(precision = 1)

root = "results/nc_adv_eval_test"
coverages = ["neuron_coverage", "top_k_coverage", "strong_coverage"]
coverage_names = ["NC", "TKNC", "SNAC"]
strategies = ["deepxplore", "dlfuzz", "dlfuzzfirst"]
strategy_names = ["DeepXplore", "DLFuzzRR", "DLFuzz"]
datasets = ["mit67", "stanford40"]
dataset_names = ["Scenes", "Actions"]
methods = ["Finetune", "DELTA", "Magprune", "ReMoS"]

dataset = "mit67"

# for idx, dataset in enumerate(datasets):
#     dataset_df = pd.DataFrame(coverage_names)
#     for strategy_name, strategy in zip(strategy_names, strategies):
#         s_dirs = []
#         for coverage in coverages:
#             path = osp.join(root, f"{strategy}_{coverage}_{dataset}_resnet18", "result.csv")
#             df = pd.read_csv(path, index_col=0)
#             dir = np.around(df['dir'].to_numpy(), 1)
#             s_dirs.append(dir)
#         dataset_df[strategy_name] = s_dirs
#     print(f"Results for {dataset_names[idx]}")
#     print(dataset_df)

m_indexes = pd.MultiIndex.from_product([dataset_names, strategy_names, methods], names=["Dataset", "Strategy", "Techniques"])
result = pd.DataFrame(np.random.randn(3, 6*4), index=coverage_names, columns=m_indexes)
for dataset_name, dataset in zip(dataset_names, datasets):
    for strategy_name, strategy in zip(strategy_names, strategies):
        s_dirs = []
        for coverage_name, coverage in zip(coverage_names, coverages):
            path = osp.join(root, f"{strategy}_{coverage}_{dataset}_resnet18", "result.csv")
            df = pd.read_csv(path, index_col=0)
            dir = np.around(df['dir'].to_numpy(), 1)
            for idx, method in enumerate(methods):
                result[(dataset_name,strategy_name, method)][coverage_name] = round(dir[idx],1)

print(f"Results for Scenes")
print(result["Scenes"])
print(f"\nResults for Actions")
print(result["Actions"])