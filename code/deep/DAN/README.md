# DAN
A PyTorch implementation of '[Learning Transferable Features with Deep Adaptation Networks](http://ise.thss.tsinghua.edu.cn/~mlong/doc/deep-adaptation-networks-icml15.pdf)'.
The contributions of this paper are summarized as fol-
lows. 
* They propose a novel deep neural network architecture for domain adaptation, in which all the layers corresponding to task-specific features are adapted in a layerwise manner, hence benefiting from “deep adaptation.”
* They explore multiple kernels for adapting deep representations, which substantially enhances adaptation effectiveness compared to single kernel methods. Our model can yield unbiased deep features with statistical guarantees.

## Requirement
* python 3
* pytorch 0.3.1
* torchvision 0.2.0

## Usage
1. You can download Office31 dataset [here](https://pan.baidu.com/s/1o8igXT4#list/path=%2F). And then unrar dataset in ./dataset/.
2. You can change the `source_name` and `target_name` in `DAN.py` to set different transfer tasks.
3. Run `python DAN.py`.

## Results on Office31
| Method | A - W | D - W | W - D | A - D | D - A | W - A | Average |
|:--------------:|:-----:|:-----:|:-----:|:-----:|:----:|:----:|:-------:|
| DAN | 83.8±0.4 | 96.8±0.2 | 99.5±0.1 | 78.4±0.2 | 66.7±0.3 | 62.7±0.2 | 81.3 |
