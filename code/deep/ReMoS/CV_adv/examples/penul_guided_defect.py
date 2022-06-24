import os, sys
import os.path as osp
import pandas as pd
from pdb import set_trace as st
import numpy as np
np.set_printoptions(precision = 1)


datasets = ["mit67", "cub200", "flower102", "sdog120", "stanford40"]
dataset_names = ["Scenes", "Birds", "Flowers", "Dogs", "Actions"]
methods = ["finetune", "delta", "weight", "retrain", "deltar", "renofeation", "remos"]
method_names = ["Finetune", "DELTA", "Magprune", "Retrain", "DELTA-R", "Renofeation", "ReMoS"]

model = "resnet18"
root = "results/res18_models"
m_indexes = pd.MultiIndex.from_product([dataset_names, method_names], names=["Dataset", "Techniques"])
result = pd.DataFrame(np.random.randn(2, 5*7), index=["Acc", "DIR"], columns=m_indexes)
for dataset_name, dataset in zip(dataset_names, datasets):
    for method_name, method in zip(method_names, methods):
        path = osp.join(root, method, f"{model}_{dataset}", "posttrain_eval.txt")
        with open(path) as f:
            line = f.readline()
            info = line.split("|")
            acc = float(info[0].split()[-1])
            dir = float(info[-1].split()[-1])
            result[(dataset_name, method_name)]["Acc"] = acc
            result[(dataset_name, method_name)]["DIR"] = dir
print("="*30, "  ResNet18  ", "="*30)
for dataset_name in dataset_names:
    print(f"Dataset {dataset_name}:")
    print(result[dataset_name])


model = "resnet50"
root = "results/res50_models"
m_indexes = pd.MultiIndex.from_product([dataset_names, method_names], names=["Dataset", "Techniques"])
result = pd.DataFrame(np.random.randn(2, 5*7), index=["Acc", "DIR"], columns=m_indexes)
for dataset_name, dataset in zip(dataset_names, datasets):
    for method_name, method in zip(method_names, methods):
        path = osp.join(root, method, f"{model}_{dataset}", "posttrain_eval.txt")
        with open(path) as f:
            line = f.readline()
            info = line.split("|")
            acc = float(info[0].split()[-1])
            dir = float(info[-1].split()[-1])
            result[(dataset_name, method_name)]["Acc"] = acc
            result[(dataset_name, method_name)]["DIR"] = dir
print("="*30, "  ResNet50  ", "="*30)
for dataset_name in dataset_names:
    print(f"Dataset {dataset_name}:")
    print(result[dataset_name])
