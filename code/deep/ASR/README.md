# EasyEspnet

ESPNet is a popular tool for end-to-end speech processing. However, it is not that easy to install, learn, and use. For instance, it is in Kaldi style that must run in shell scripts (i.e., its `run.sh` file). This makes it not easy to use, debug, and deploy in online environments.

We provide a wraper for ESPNet, which we call **EasyEspnet**, for easier usage of ESPNet. This code base will make it easier to write/run/debug your codes in a more friendly Python style.

## Requirements

Of course we are not an independent tool. So you need to correctly install [ESPNet](https://espnet.github.io/espnet/installation.html) first. But we know that the installation of ESPNet is also not that easy (slow; tedious configurations etc.). Thus, we provide a all-in-one docker image for your to use. All you need to do is install [docker](https://docs.docker.com/engine/install/). Then, pull [our ESPNet image](https://hub.docker.com/r/jindongwang/espnet):

```
docker pull jindongwang/espnet:all11
```

Then, you can directly run ESPNet in this docker. Note that this docker itself already contains the ESPNet codebase. So you do not need to install it again.
Docker makes it much easier to submit speech recognition jobs in a cloud environment since most of the cloud computing platforms support docker.

## Run

Currently, this repo supports ASR tasks only. All you need is to extract features using Espnet and set the data folder path. To extract features using ESPNet, you can run `bash run.sh --stop_state 2` inside an example of ESPNet such as `egs/an4/asr1/`.

There are three main Python files to use:

- `train.py`:
- `data_load.py`:
- `utils.py`:

You need to check or modify in `train.py arg_list`, config should be in ESPnet config style (remember to include decoding information if you want to compute cer/wer), then, you can run train.py. For example, 

```
python train.py --root_path an4/asr1 --dataset an4
```

Done. Results (log, model, snapshots) are saved in results_(dataset)/(config_name) by default.

### Demo

We provide the processed features using an4 as demo.

To run this demo, please execute:

Download and unzip the features:

```
mkdir data; cd data; 
wget https://transferlearningdrive.blob.core.windows.net/teamdrive/dataset/speech/an4_features.tar.gz
tar -zxvf an4_features.tar.gz; rm an4_features.tar.gz; cd ..
```
Start training with EasyEspnet: 

```
python train.py --root_path data/an4/asr1/ --dataset an4
```

## Decoding and WER/CER evaluation

Set `--decoding_mode` to `true` to perform decoding and CER/WER evaluation. For example:

```
python train.py --decoding_mode true
```

## Distributed training

EasyEspnet supports multi-GPU training by default using Pytorch `DataParallel`, but it also supports PyTorch `DistributedDataParallel` training which is much faster. For example, using 2 GPUs, 1 node: 

```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py --dist_train true
```

## Acknowledgement

- ESPNet: https://github.com/espnet/espnet

## Contact

- Wenxin Hou, 
- Jindong Wang, 
