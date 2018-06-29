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

**Protocol**: We use a **full-training** protocol, which is taking all the samples from one domain as the source or target domain. Another similar protocol is **down-sample** protocol, which is choosing 20 or 8 samples per category to use as the domain data. Almost all the famous deep transfer learning methods (DDC, DAN, JAN, RTN, DCORAL etc.) are adopting the **down-sample** protocol. The results from two protocols are absolutely **different**.

However, some publised papers are just copying the results (mostly the down-sample results) without paying attention to the protocols.

Here's our results.

|             Method            | A - W |
|:-----------------------------:|:-----:|
|          Our AlexNet (full-training)          |  51%  |
|           Our ResNet (full-training)        |  71%  |
|           AlexNet in DDC, DAN, JAN, DCORAL... (down-sample)        |  61%  |
|           ResNet in DAN, JAN... (down-sample)        |  80%  |

## References

[1] DDC paper: Tzeng E, Hoffman J, Zhang N, et al. Deep domain confusion: Maximizing for domain invariance[J]. arXiv preprint arXiv:1412.3474, 2014.

[2] DAN paper: Long M, Cao Y, Wang J, et al. Learning transferable features with deep adaptation networks[J]. arXiv preprint arXiv:1502.02791, 2015.
