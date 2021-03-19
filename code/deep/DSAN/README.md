# DSAN
A PyTorch implementation of 'Deep Subdomain Adaptation Network for Image Classification' which has published on IEEE Transactions on Neural Networks and Learning Systems.
The contributions of this paper are summarized as follows. 
* They propose a novel deep neural network architecture for Subdomain Adaptation, which can extend the ability of deep adaptation networks by capturing the fine-grained information for each category.
* They show that DSAN which is a non-adversarial method can achieve the remarkable results. In addition, their DSAN is very simple and easy to implement.
## Requirement
* python 3
* pytorch 1.0

## Usage
1. You can download Office31 dataset [here](https://pan.baidu.com/s/1o8igXT4#list/path=%2F). And then unrar dataset in ./dataset/.
2. You can change the `source_name` and `target_name` in `Config.py` to set different transfer tasks.
3. Run `python DSAN.py`.

## Results on Office31
| Method | A - W | D - W | W - D | A - D | D - A | W - A | Average |
|:--------------:|:-----:|:-----:|:-----:|:-----:|:----:|:----:|:-------:|
| DSAN | 93.6±0.2 | 98.4±0.1 | 100.0±0.0 | 90.2±0.7 | 73.5±0.5 | 74.8±0.4 | 88.4 |

> Note that for tasks D-A and W-A, setting epochs = 800 or larger could achieve better performance.

## Reference

```
Zhu Y, Zhuang F, Wang J, et al. Deep Subdomain Adaptation Network for Image Classification[J]. IEEE Transactions on Neural Networks and Learning Systems, 2020.
```

or in bibtex style:

```
@article{zhu2020deep,
  title={Deep Subdomain Adaptation Network for Image Classification},
  author={Zhu, Yongchun and Zhuang, Fuzhen and Wang, Jindong and Ke, Guolin and Chen, Jingwu and Bian, Jiang and Xiong, Hui and He, Qing},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2020},
  publisher={IEEE}
}
```