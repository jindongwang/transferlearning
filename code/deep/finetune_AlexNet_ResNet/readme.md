# Fine-tune on AlexNet and ResNet

This directory contains the Pytorch code for fine-tuning AlexNet and ResNet on certain datasets. With the development of deep transfer learning, a lot of new approaches are regarding AlexNet and ResNet as the baselines. Therefore, we hope this directory could be of help.

The code are designed to be easy to follow and understand, as I always do.

After learning fine-tune, we can build other `bottleneck` layers to perform transfer learning. This will be added soon.

## Requirements

Python 3.6, Pytorch 0.4.0 or above.

All can be installed using `conda` or `pip`.

## Usage

1. Download the [Office-31](https://pan.baidu.com/s/1o8igXT4#list/path=%2F) dataset and extract it into the `data` foder.
2. In your terminal, type `python office31.py`. That's all.
3. You can switch `alexnet` or `resnet` in the commond line.

More usage: you can run `python office31.py -m resnet -b 64 -g 3`, which means you use resnet with the batch size of 64 and the 3rd of gpu device.

## About the code

The code consists of 4 `.py` files:

- `data_loader.py`: Load the dataset.
- `office31.py`: the main file.

## Results

**Protocol**: We use a **full-training** protocol, which is taking all the samples from one domain as the source or target domain. Another similar protocol is **down-sample** protocol, which is choosing 20 or 8 samples per category to use as the domain dataThe results from two protocols are absolutely **different**.

Here are our results using the full training prototol.

|             Method            | A - W |
|:-----------------------------:|:-----:|
|          Our AlexNet         |  51%  |
|           Our ResNet       |  77.3%  |
|           AlexNet in DDC, DAN, JAN, DCORAL...       |  61%  |
|           ResNet in DAN, JAN...       |  80%  |

The results are clearly much lower than the results reported in exising papers such as DDC and DAN. I don't know why. Anyone who can successfully reimplement the results, **PLEASE** contact me!

## References

[1] DDC paper: Tzeng E, Hoffman J, Zhang N, et al. Deep domain confusion: Maximizing for domain invariance[J]. arXiv preprint arXiv:1412.3474, 2014.

[2] DAN paper: Long M, Cao Y, Wang J, et al. Learning transferable features with deep adaptation networks[J]. arXiv preprint arXiv:1502.02791, 2015.
