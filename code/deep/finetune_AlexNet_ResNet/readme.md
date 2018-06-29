# Fine-tune on AlexNet and ResNet

This directory contains the Pytorch code for fine-tuning AlexNet and ResNet on certain datasets. With the development of deep transfer learning, a lot of new approaches are regarding AlexNet and ResNet as the baselines. Therefore, we hope this directory could be of help.

The code are designed to be easy to follow and understand, as I always do.

After learning fine-tune, we can build other `bottleneck` layers to perform transfer learning. This will be added soon.

## Requirements

Python 3.6, Pytorch 0.4.0, tqdm.

All can be installed using `conda` or `pip`.

## Run

1. Download the [Office-31](https://pan.baidu.com/s/1o8igXT4#list/path=%2F) dataset and extract it into the `data` foder.
2. In your terminal, type `python office31.py`. That's all.
3. You can switch `alexnet` or `resnet` in the file to run respective network.

## About the code

The code consists of 4 `.py` files:

- `data_loader.py`: Load the dataset.
- `AlexNet.py`: AlexNet class for fine-tune.
- `ResNet.py`: ResNet class for fine-tune.
- `office31.py`: the main file, contains train and test.

## Results

*Update: Thanks to @Wogong, I changed the data precessing step. The current accuracy is around 51%.*

|             Method            | A - W |
|:-----------------------------:|:-----:|
| AlexNet (DDC, DAN, JAN paper) | 61.6% |
|          Our AlexNet          |  51%  |
|    ResNet (DAN, JAN paper)    |  80%  |
|           Our ResNet          |  67%  |


This is a **mystery**. I tried so hard to change the learning rate, batch size, and weight decay, but I failed to reimplement the results reported in most articles. In fact, I did **NOT** see any efforts in Pytorch to reimplement them successfully.

Current learning rate is set according to DDC and DAN. The learning rate of last layer is 10 times of the rest of network. And it decays according to a certain formula. (see the code)

If anyone successfully reimplements the results in the paper, **PLEASE** contact me ASAP!

## References

[1] DDC paper: Tzeng E, Hoffman J, Zhang N, et al. Deep domain confusion: Maximizing for domain invariance[J]. arXiv preprint arXiv:1412.3474, 2014.

[2] DAN paper: Long M, Cao Y, Wang J, et al. Learning transferable features with deep adaptation networks[J]. arXiv preprint arXiv:1502.02791, 2015.