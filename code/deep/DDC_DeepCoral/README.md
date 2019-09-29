# Deep Coral and DDC
A PyTorch implementation of '[Deep CORAL Correlation Alignment for Deep Domain Adaptation](https://arxiv.org/pdf/1607.01719.pdf)'.
The contributions of this paper are summarized as fol-
lows.
* They extend CORAL to incorporate it directly into deep networks by constructing a differentiable loss function that minimizes the difference between source and target correlations–the CORAL loss.
* Compared to CORAL, Deep CORAL approach learns a non-linear transformation that is more powerful and also works seamlessly with deep CNNs.

By simply replacing the CORAL loss with MMD, we can re-implemented the DDC (Deep Domain Confusion) paper [Deep Domain Confusion: Maximizing for Domain Invariance](https://arxiv.org/abs/1412.3474).

## Requirement
* python 3
* pytorch 1.0
* torchvision 0.2.0

## Usage

Before you run, you need to take some time to look at the `config.py` file, where you can set some configs.

1. You can download Office31 dataset [here](https://pan.baidu.com/s/1o8igXT4#list/path=%2F). And then unrar dataset in ./dataset/.
2. You can change the `source_name` and `target_name` in `DeepCoral.py` to set different transfer tasks.
3. Run `python main.py`.

In the `main.py` file, you can replace your `adaptation_loss` with either `mmd` or `coral`. We support both alexnet and resnet50.

## Results on Office31
| Method | A - W | D - W | W - D | A - D | D - A | W - A | Average |
|:--------------:|:-----:|:-----:|:-----:|:-----:|:----:|:----:|:-------:|
| DCORAL | 77.7±0.3 | 97.6±0.2 | 99.7±0.1 | 81.1±0.4 | 64.6±0.3 | 64.0±0.4 | 80.8 |

> Please note that the results are run by myself. To compared to other methods, I add the coral loss after the average pool layer in ResNet50. In the paper, they add the coral loss after the fc8 in AlexNet.