# AdaRNN: Adaptive Learning and Forecasting for Time Series

This project implements our paper [AdaRNN: Adaptive Learning and Forecasting for Time Series](https://arxiv.org/abs/2108.04443). Please refer to our paper [1] for the method and technical details.

## Request
- cuda 10.1 
- Python 3.7.7
- Pytorch == 1.6.0
- Torchvision == 0.7.0

The required packages are listed in requirements.txt 


## Dataset 

The original air-quality dataset is downloaded from [Beijing Multi-Site Air-Quality Data Data Set](https://archive.ics.uci.edu/ml/datasets/Beijing+Multi-Site+Air-Quality+Data) . The air-quality dataset contains hourly air quality information collected from 12 stations in Beijing from 03/2013 to 02/2017. We randomly chose four stations (Dongsi, Tiantan, Nongzhanguan, and Dingling) and select six features (PM2.5, PM10, S02, NO2, CO, and O3). Since there are some missing data, we simply fill the empty slots using averaged values. Then, the dataset is normalized before feeding into the network to scale all features into the same range. This process is accomplished by max-min normalization and ranges data between 0 and 1. The processed  air-quality dataset can be downloaded at [dataset link](https://box.nju.edu.cn/f/2239259e06dd4f4cbf64/?dl=1). 



## How to run

The code for air-quality dataset is in `train_weather.py`. After downloading the dataset, you can change args.data_path in `train_weather.py` to the folder where you place the data.

Then you can run the code. Taking Dongsi station as example, you can run 

`python train_weather.py --model_name 'AdaRNN' --station 'Dongsi' --pre_epoch 50 --dw 0.5`


# Contact
- dz1833005@smail.nju.edu.cn
- jindongwang@outlook.com

# References
[1] Yuntao Du, Jindong Wang, Wenjie Feng, Sinno Pan, Tao Qin, Chongjun Wang, "AdaRNN: Adaptive Learning and Forecasting for Time Series", CIKM 2021.
