# Revgrad
A PyTorch implementation of '[Unsupervised Domain Adaptation by Backpropagation](https://arxiv.org/pdf/1409.7495.pdf)'.
The contributions of this paper are summarized as follows. 
* They use adversarial training to find domain–invariant representations innetwork. Their Revgrad exhibit an architecture whose first few feature extraction layers are shared by two classifiers(domain classifier and label classifier) trained simultaneously.

## Requirement
* python 3
* pytorch 0.3.1
* torchvision 0.2.0

## Usage
1. You can download Office31 dataset [here](https://pan.baidu.com/s/1o8igXT4#list/path=%2F). And then unrar dataset in ./dataset/.
2. You can change the `source_name` and `target_name` in `Revgrad.py` to set different transfer tasks.
3. Run `python Revgrad.py`.

## Results on Office31
| Method | A - W | D - W | W - D | A - D | D - A | W - A | Average |
|:--------------:|:-----:|:-----:|:-----:|:-----:|:----:|:----:|:-------:|
| Revgradori | 82.0±0.4 | 96.9±0.2 | 99.1±0.1 | 79.7±0.4 | 68.2±0.4 | 67.4±0.5 | 82.2 |
| Revgradlast | 81.5±0.9 | 97.2±0.4 | 99.9±0.1 |  82.7±0.9 | 65.6±0.7 | 65.2±0.5 | 82.0 |
| Revgradmax | 82.6±0.9 | 97.8±0.2 | 100.0±0.0 | 83.3±0.9 | 66.8±0.1 | 66.1±0.5 | 82.8 |

> Note that the results **Revgradori** comes from [paper](http://ise.thss.tsinghua.edu.cn/~mlong/doc/multi-adversarial-domain-adaptation-aaai18.pdf) which has the same author as DAN. The **Revgradlast** is the results of the last epoch, and **Revgradmax** is the results of the max results in all epoches. Both **Revgradlast** and **Revgradmax** are run by myself with the code.

> Please note that the code is different from the paper. In the paper, they use a layer named Revgrad layer. However, the adversarial training is not stable. [Tzeng et al](https://www.robots.ox.ac.uk/~vgg/rg/papers/Tzeng_ICCV2015.pdf). mentioned when source and target feature mappings share their architectures, the domain confusion can be introduced to replace the adversarial objective, which performs stable. Hence, we use the domain confusion instead of the Revgrad layer.
