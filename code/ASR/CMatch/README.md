# CMatch: Cross-domain Speech Recognition with Unsupervised Character-level Distribution Matching

This project implements our paper [Cross-domain Speech Recognition with Unsupervised Character-level Distribution Matching](https://arxiv.org/abs/2104.07491) based on [EasyEspnet](https://github.com/jindongwang/EasyEspnet). Please refer to [EasyEspnet](https://github.com/jindongwang/EasyEspnet) for the program introduction, installation and usage. And our paper [1] for the method and technical details.

## Run

After extracting features using Espnet and set the data folder path as introduced in [EasyEspnet](https://github.com/jindongwang/EasyEspnet).

You need to check or modify in `train.py arg_list`, config should be in ESPnet config style (remember to include decoding information if you want to compute cer/wer), then, you can run train.py. For example, 

```
python train.py --root_path libriadapt/asr1 --dataset libriadapt_en_us_clean_matrix --config config/train.yaml
```

This commands pre-trains an ASR model on the specified dataset. Results (log, model, snapshots) are saved in results_(dataset)/(config_name) by default.

### Cross-domain Adaptation

We provide three methods for cross-domain adaptation in this implementation: adversarial training (adv), maximum mean discrepancy (mmd) and our CMatch.

For the adversarial and mmd, you can refer to and modify the target dataset, pre-trained model setting in `config/{mmd, adv}_example.yaml`, and then execute this command for example:

```
python train.py --root_path libriadapt/asr1 --dataset libriadapt_en_us_clean_matrix --config config/mmd_example.yaml
```

For the CMatch method, there are two steps.

#### Step 1:
First we need to obtain the pseudo labels, for example, suppose we trained the source model using the command above, we need to re-execute it with the `--pseudo_labeling` being set `true` and the `--tgt_dataset` specifying the target dataset:

```
python train.py --root_path libriadapt/asr1 --dataset libriadapt_en_us_clean_matrix --config config/train.yaml --pseudo_labeling true --tgt_dataset libriadapt_en_us_clean_respeaker
```

So that we will get the pseudo labels for `libriadapt_en_us_clean_respeaker` dataset.

#### Step 2:
Then we can refer to and modify the target dataset, pre-trained model, non-character symbol, and pseudo label setting in `config/{ctc_align, pseudo_ctc_pred, frame_average}_example.yaml`, and then execute this command for example:
```
python train.py --root_path libriadapt/asr1 --dataset libriadapt_en_us_clean_matrix --config config/ctc_align_example.yaml
```

### Demo

TBD

## Decoding and WER/CER evaluation

Set `--decoding_mode` to `true` to perform decoding and CER/WER evaluation. For example:

```
python train.py --root_path libriadapt/asr1 --dataset libriadapt_en_us_clean_matrix --config config/train.yaml --decoding_mode true
```

## Distributed training

Following EasyEspnet, you can also perform distributed training which is much faster. For example, using 4 GPUs, 1 node: 

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 train.py --dist_train true --root_path libriadapt/asr1 --dataset libriadapt_en_us_clean_matrix --config config/mmd_example.yaml
```

## Acknowledgement

- ESPNet: https://github.com/espnet/espnet
- EasyEspnet: https://github.com/jindongwang/EasyEspnet
- NeuralSP: https://github.com/hirofumi0810/neural_sp

## Contact

- [Wenxin Hou](https://houwenxin.github.io/): houwx001@gmail.com
- [Jindong Wang](http://www.jd92.wang/): jindongwang@outlook.com



## References

[1] Wenxin Hou, Jindong Wang, Xu Tan, Tao Qin, Takahiro Shinozaki, "Cross-domain Speech Recognition with Unsupervised Character-level Distribution Matching", submitted to INTERSPEECH 2021.
