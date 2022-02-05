# DAN
A PyTorch implementation of '[Multi-representationadaptationnetworkforcross-domainimage
classification](https://www.sciencedirect.com/science/article/pii/S0893608019301984)'.
The contributions of this paper are summarized as follows. 
* We are the first to learn multiple different domain-invariant representations by Inception
Adaptation Module (IAM) for cross-domain image classification.
* A novel Multi-Representation Adaptation Network (MRAN) is proposed to align distributions of multiple different representations which might contain more information about the images.

## Requirement
* python 3
* pytorch 1.0
* torchvision 0.2.0

## Usage
1. You can download Office31 dataset [here](https://pan.baidu.com/s/1o8igXT4#list/path=%2F). And then unrar dataset in ./dataset/.
2. You can change the `source_name` and `target_name` in `MRAN.py` to set different transfer tasks.
3. Run `python MRAN.py`.

## Results on Office31
| Method | A - W | D - W | W - D | A - D | D - A | W - A | Average |
|:--------------:|:-----:|:-----:|:-----:|:-----:|:----:|:----:|:-------:|
| MRAN | 91.4±0.1 | 96.9±0.3 | 99.8±0.2 | 86.4±0.6 | 68.3±0.5 | 70.9±0.6 | 85.6 |
