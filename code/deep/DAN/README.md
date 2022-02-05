# DAN
A PyTorch implementation of '[Learning Transferable Features with Deep Adaptation Networks](http://ise.thss.tsinghua.edu.cn/~mlong/doc/deep-adaptation-networks-icml15.pdf)'.
The contributions of this paper are summarized as follows. 
* They propose a novel deep neural network architecture for domain adaptation, in which all the layers corresponding to task-specific features are adapted in a layerwise manner, hence benefiting from “deep adaptation.”
* They explore multiple kernels for adapting deep representations, which substantially enhances adaptation effectiveness compared to single kernel methods. Our model can yield unbiased deep features with statistical guarantees.

## Requirement
* python 3
* pytorch 1.0

## Usage
1. You can download Office31 dataset [here](https://pan.baidu.com/s/1o8igXT4#list/path=%2F). And then unrar dataset in ./dataset/.
2. You can change the `source_name` and `target_name` in `DAN.py` to set different transfer tasks.
3. Run `python DAN.py`.

## Results on Office31
| Method | A - W | D - W | W - D | A - D | D - A | W - A | Average |
|:--------------:|:-----:|:-----:|:-----:|:-----:|:----:|:----:|:-------:|
| DANori | 83.8±0.4 | 96.8±0.2 | 99.5±0.1 | 78.4±0.2 | 66.7±0.3 | 62.7±0.2 | 81.3 |
| DANlast | 81.6±0.7 | 97.2±0.1 | 99.5±0.1 | 80.0±0.7 | 66.2±0.6 | 65.6±0.4 | 81.7 |
| DANmax | 82.6±0.7 | 97.7±0.1 | 100.0±0.0 | 83.1±0.9 | 66.8±0.3 | 66.6±0.4 | 82.8 |

> Note that the results **DANori** comes from [paper](http://ise.thss.tsinghua.edu.cn/~mlong/doc/multi-adversarial-domain-adaptation-aaai18.pdf) which has the same author as DAN. The **DANlast** is the results of the last epoch, and **DANmax** is the results of the max results in all epoches. Both **DANlast** and **DANmax** are run by myself with the code.
