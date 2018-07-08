# DDC
A PyTorch implementation of '[Deep Domain Confusion Maximizing for Domain Invariance](https://arxiv.org/pdf/1412.3474.pdf)'.
The contributions of this paper are summarized as follows. 
* They propose a new CNN architecture, which uses an adaptation layer along with a domain confusion loss based on maximum mean discrepancy(MMD) to automatically learn a representation jointly trained to optimize for classification and domain invariance.

## Requirement
* python 3
* pytorch 0.3.1
* torchvision 0.2.0

## Usage
1. You can download Office31 dataset [here](https://pan.baidu.com/s/1o8igXT4#list/path=%2F). And then unrar dataset in ./dataset/.
2. You can change the `source_name` and `target_name` in `DDC.py` to set different transfer tasks.
3. Run `python DDC.py`.

## Results on Office31
| Method | A - W | D - W | W - D | A - D | D - A | W - A | Average |
|:--------------:|:-----:|:-----:|:-----:|:-----:|:----:|:----:|:-------:|
| DDCori | 75.8±0.2 | 95.0±0.2 | 98.2±0.1 | 77.5±0.3 | 67.4±0.4 | 64.0±0.5 | 79.7 |
| DDClast | 66.7±0.3 | 96.0±0.7 | 99.5±0.3 | 73.3±0.3 | 59.9±0.4 | 57.2±0.8 | 75.4 |
| DDCmax | 78.3±0.4 | 97.1±0.1 | 100.0±0.0 | 81.7±0.9 | 65.2±0.6 | 65.1±0.4 | 81.2 |

> I conduct the experiment as the deep transfer methods, such as DAN, JAN, Revgrad.

> Note that the results **DDCori** comes from [paper](http://ise.thss.tsinghua.edu.cn/~mlong/doc/multi-adversarial-domain-adaptation-aaai18.pdf). The **DDClast** is the results of the last epoch, and **DDCmax** is the results of the max results in all epoches. Both **DDClast** and **DDCmax** are run by myself with the code.

> You could note that there is a large gap between **DDClast** and **DDCmax**. I don't know the reason of the phenomenon. And the phenomenon doesn't appear in other methods such as DAN, DeepCoral. If you know the reason, please create an Issue or send us an Email.
