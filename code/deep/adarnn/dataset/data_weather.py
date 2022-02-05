
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from pandas.core.frame import DataFrame
from torch.utils.data import Dataset, DataLoader
import torch
import pickle
import datetime



class data_loader(Dataset):
    def __init__(self, df_feature, df_label, df_label_reg, t=None):

        assert len(df_feature) == len(df_label)
        assert len(df_feature) == len(df_label_reg)

        # df_feature = df_feature.reshape(df_feature.shape[0], df_feature.shape[1] // 6, df_feature.shape[2] * 6)
        self.df_feature=df_feature
        self.df_label=df_label
        self.df_label_reg = df_label_reg

        self.T=t
        self.df_feature=torch.tensor(
            self.df_feature, dtype=torch.float32)
        self.df_label=torch.tensor(
            self.df_label, dtype=torch.float32)
        self.df_label_reg=torch.tensor(
            self.df_label_reg, dtype=torch.float32)

    def __getitem__(self, index):
        sample, target, label_reg =self.df_feature[index], self.df_label[index], self.df_label_reg[index]
        if self.T:
            return self.T(sample), target
        else:
            return sample, target, label_reg

    def __len__(self):
        return len(self.df_feature)


def create_dataset(df, station, start_date, end_date, mean=None, std=None):
    data=df[station]
    feat, label, label_reg =data[0], data[1], data[2]
    referece_start_time=datetime.datetime(2013, 3, 1, 0, 0)
    referece_end_time=datetime.datetime(2017, 2, 28, 0, 0)

    assert (pd.to_datetime(start_date) - referece_start_time).days >= 0
    assert (pd.to_datetime(end_date) - referece_end_time).days <= 0
    assert (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days >= 0
    index_start=(pd.to_datetime(start_date) - referece_start_time).days
    index_end=(pd.to_datetime(end_date) - referece_start_time).days
    feat=feat[index_start: index_end + 1]
    label=label[index_start: index_end + 1]
    label_reg=label_reg[index_start: index_end + 1]

    # ori_shape_1, ori_shape_2=feat.shape[1], feat.shape[2]
    # feat=feat.reshape(-1, feat.shape[2])
    # feat=(feat - mean) / std
    # feat=feat.reshape(-1, ori_shape_1, ori_shape_2)

    return data_loader(feat, label, label_reg)



def create_dataset_shallow(df, station, start_date, end_date, mean=None, std=None):
    data=df[station]
    feat, label, label_reg =data[0], data[1], data[2]
    referece_start_time=datetime.datetime(2013, 3, 1, 0, 0)
    referece_end_time=datetime.datetime(2017, 2, 28, 0, 0)

    assert (pd.to_datetime(start_date) - referece_start_time).days >= 0
    assert (pd.to_datetime(end_date) - referece_end_time).days <= 0
    assert (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days >= 0
    index_start=(pd.to_datetime(start_date) - referece_start_time).days
    index_end=(pd.to_datetime(end_date) - referece_start_time).days
    feat=feat[index_start: index_end + 1]
    label=label[index_start: index_end + 1]
    label_reg=label_reg[index_start: index_end + 1]

    # ori_shape_1, ori_shape_2=feat.shape[1], feat.shape[2]
    # feat=feat.reshape(-1, feat.shape[2])
    # feat=(feat - mean) / std
    # feat=feat.reshape(-1, ori_shape_1, ori_shape_2)

    return feat,  label_reg

def get_dataset_statistic(df, station, start_date, end_date):
    data=df[station]
    feat, label =data[0], data[1]
    referece_start_time=datetime.datetime(2013, 3, 1, 0, 0)
    referece_end_time=datetime.datetime(2017, 2, 28, 0, 0)

    assert (pd.to_datetime(start_date) - referece_start_time).days >= 0
    assert (pd.to_datetime(end_date) - referece_end_time).days <= 0
    assert (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days >= 0
    index_start=(pd.to_datetime(start_date) - referece_start_time).days
    index_end=(pd.to_datetime(end_date) - referece_start_time).days
    feat=feat[index_start: index_end + 1]
    label=label[index_start: index_end + 1]
    feat=feat.reshape(-1, feat.shape[2])
    mu_train=np.mean(feat, axis=0)
    sigma_train=np.std(feat, axis=0)

    return mu_train, sigma_train


def get_weather_data(data_file, station, start_time, end_time, batch_size, shuffle=True, mean=None, std=None):
    df=pd.read_pickle(data_file)

    dataset=create_dataset(df, station, start_time,
                             end_time, mean=mean, std=std)
    train_loader=DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle)
    return train_loader


def get_weather_data_shallow(data_file, station, start_time, end_time, batch_size, shuffle=True, mean=None, std=None):
    df=pd.read_pickle(data_file)

    feat, label_reg =create_dataset_shallow(df, station, start_time,
                             end_time, mean=mean, std=std)

    return feat, label_reg


def get_weather_data_statistic(data_file, station, start_time, end_time):
    df=pd.read_pickle(data_file)
    mean_train, std_train =get_dataset_statistic(
        df, station, start_time, end_time)
    return mean_train, std_train

