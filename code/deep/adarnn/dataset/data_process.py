# encoding=utf-8
import os
import dataset.data_act as data_act
import pandas as pd
import dataset.data_weather as data_weather
import datetime


def load_act_data(data_folder, batch_size=64, domain="1_20"):
    x_train, y_train, x_test, y_test = data_act.load_data(data_folder, domain)
    x_train, x_test = x_train.reshape(
        (-1, x_train.shape[2], 1, x_train.shape[1])), x_test.reshape((-1, x_train.shape[2], 1, x_train.shape[1]))
    transform = None
    train_set = data_act.data_loader(x_train, y_train, transform)
    test_set = data_act.data_loader(x_test, y_test, transform)
    train_loader = data_act.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = data_act.DataLoader(
        test_set, batch_size=batch_size, shuffle=False)
    return train_loader, train_loader, test_loader



def load_weather_data(file_path, batch_size=6, station='Changping'):
    data_file = os.path.join(file_path, "PRSA_Data_1.pkl")
    mean_train, std_train = data_weather.get_weather_data_statistic(data_file, station=station, start_time='2013-3-1 0:0',
                                                                    end_time='2016-10-30 23:0')
    train_loader = data_weather.get_weather_data(data_file, station=station, start_time='2013-3-6 0:0',
                                                 end_time='2015-5-31 23:0', batch_size=batch_size, mean=mean_train, std=std_train)
    valid_train_loader = data_weather.get_weather_data(data_file, station=station, start_time='2015-6-2 0:0',
                                                       end_time='2016-6-30 23:0', batch_size=batch_size, mean=mean_train, std=std_train)
    valid_vld_loader = data_weather.get_weather_data(data_file, station=station, start_time='2016-7-2 0:0',
                                                     end_time='2016-10-30 23:0', batch_size=batch_size, mean=mean_train, std=std_train)
    test_loader = data_weather.get_weather_data(data_file, station=station, start_time='2016-11-2 0:0',
                                                end_time='2017-2-28 23:0', batch_size=batch_size, mean=mean_train, std=std_train)
    return train_loader, valid_train_loader, valid_vld_loader, test_loader


def get_split_time(num_domain=2, mode='pre_process'):
    spilt_time = {
        '2': [('2013-3-6 0:0', '2015-5-31 23:0'), ('2015-6-2 0:0', '2016-6-30 23:0')],
        # '3': [('2013-3-1 0:0', '2014-4-11 23:0'), ('2014-4-13 0:0', '2015-5-22 23:0'), ('2015-5-24 0:0', '2016-6-30 23:0')],
    }
    if mode == 'pre_process':
        return spilt_time[str(num_domain)]
    else:
        print("error in mode")





def load_weather_data_multi_domain(file_path, batch_size=6, station='Changping', number_domain=2, mode='pre_process'):
    # mode: 'auto', 'pre_process'
    #
    data_file = os.path.join(file_path, "PRSA_Data_1.pkl")
    mean_train, std_train = data_weather.get_weather_data_statistic(data_file, station=station, start_time='2013-3-1 0:0',
                                                                    end_time='2016-10-30 23:0')
    split_time_list = get_split_time(number_domain, mode=mode)
    train_list = []
    for i in range(len(split_time_list)):
        time_temp = split_time_list[i]
        train_loader = data_weather.get_weather_data(data_file, station=station, start_time=time_temp[0],
                                                     end_time=time_temp[1], batch_size=batch_size, mean=mean_train, std=std_train)
        train_list.append(train_loader)

    valid_vld_loader = data_weather.get_weather_data(data_file, station=station, start_time='2016-7-2 0:0',
                                                     end_time='2016-10-30 23:0', batch_size=batch_size, mean=mean_train, std=std_train)
    test_loader = data_weather.get_weather_data(data_file, station=station, start_time='2016-11-2 0:0',
                                                end_time='2017-2-28 23:0', batch_size=batch_size, mean=mean_train, std=std_train, shuffle=False)
    return train_list, valid_vld_loader, test_loader
