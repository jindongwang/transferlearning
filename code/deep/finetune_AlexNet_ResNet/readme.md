# Fine-tune on AlexNet and ResNet

This directory contains the Pytorch code for fine-tuning **AlexNet** and **ResNet** on certain datasets. With the development of deep transfer learning, a lot of new approaches are regarding AlexNet and ResNet as the baselines. Therefore, we hope this directory could be of help.

The code are designed to be easy to follow and understand, as I always do.

## Requirements

Python 3.6, Pytorch 0.4.0 or above.

All can be installed using `conda` or `pip`.

## Usage

1. Download the [Office-31](https://pan.baidu.com/s/1o8igXT4#list/path=%2F) dataset and extract it into the `data` foder. Other datasets are welcome.
2. In your terminal, type `python office31.py`. That's all.
3. You can switch `alexnet` or `resnet` in the commond line.

More usage: you can run 

`python finetune_office31.py -m resnet -b 64 -g 3 -src amazon -tar webcam`

which means you use resnet with the batch size of 64 and the 3rd gpu device on source amazon and target webcam.

## About the code

The code consists of 2 files:

- `data_loader.py`: Load the dataset.
- `office31.py`: The main file.

## Results

**Protocol**: We use a **full-training** protocol, which is taking all the samples from one domain as the source or target domain. 

Here are our results using the full training prototol.

AlexNet:

| Method | A - W | D - W | W - D | A - D | D - A | W - A | Average |
|:--------------:|:-----:|:-----:|:-----:|:-----:|:----:|:----:|:-------:|
| AlexNet (reported in previous work)| 61.6 | 95.4  | 99.0 | 63.8 | 51.1 | 49.8 | 70.1 |
| AlexNet (ours) | 51.3 | 91.1 | 96.8 | 50.0  | 49.8  | 39.1  | 63.0 |


ResNet:

| Method | A - W | D - W | W - D | A - D | D - A | W - A | Average |
|:--------------:|:-----:|:-----:|:-----:|:-----:|:----:|:----:|:-------:|
| ResNet-50 (reported in previous work)| 68.4 | 96.7  | 99.3 | 68.9 | 62.5 | 60.7 | 76.1 |
| ResNet-50 (ours)| 76.7 | 91.8  | 99.0 | 78.9 | 63.5 | 65.0 | 79.2 |

**Training:** The finetune takes a standard training process: divide the source domain into `train` and `validation` set (8:2), and then treat the target domain as the `test` set.

**Parameter setting:** Learning rate=0.0001 for all the layers except the fc layer, which is 0.001 learning rate. Batch size = 64. Using SGD as the optimizer with momentum = 0.9 without weight decay.

*Remark:* The results of AlexNet is clearly different from those reported in previous work. This is because the pretrained AlexNet model provided in PyTorch is different from that in Caffe, while previous work adopted the results from this framework.