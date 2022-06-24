import os, sys
import os.path as osp
import pandas as pd
from pdb import set_trace as st
import numpy as np

np.set_printoptions(precision = 1)

root = "remos"
methods = ["finetune", "weight", "retrain", "renofeation", "remos"]
method_names = ["Finetune", "Magprune", "Retrain", "Renofeation", "ReMoS"]
datasets = ["mit67", "cub200", "stanford40"]
dataset_names = ["Scenes", "Birds", "Actions"]

m_indexes = pd.MultiIndex.from_product([dataset_names, method_names], names=["Dataset", "Techniques"])
result = pd.DataFrame(np.random.randn(2, 15), index=["Acc", "DIR"], columns=m_indexes)
for dataset_name, dataset in zip(dataset_names, datasets):
    for method_name, method in zip(method_names, methods):
        path = osp.join(root, method, dataset, "test.tsv")
        with open(path) as f:
            lines = f.readlines()
        acc = float(lines[-1].split()[-1])
        dir = float(lines[-7].split()[-1])
        result[(dataset_name, method_name)]["Acc"] = acc
        result[(dataset_name, method_name)]["DIR"] = dir
print(result)
