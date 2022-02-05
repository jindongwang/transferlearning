# DeepMEDA (DDAN)

A PyTorch implementation of **Transfer Learning with Dynamic Distribution Adaptation** which has published on ACM Transactions on Intelligent Systems and Technology.

This is also called **DDAN (Deep Dynamic Adaptation Network)**.

Matlab version is [HERE](https://github.com/jindongwang/transferlearning/tree/master/code/traditional/MEDA/matlab).

## Requirement

* python 3
* pytorch 1.x
* Numpy, scikit-learn

## Usage

1. You can download Office31 dataset [here](https://pan.baidu.com/s/1o8igXT4#list/path=%2F). Also, other datasets are supported in [here](https://github.com/jindongwang/transferlearning/tree/master/data).
2. Run `python main.py --src dslr --tar amazon --batch_size 32`.

> Note that for tasks D-A and W-A, setting epochs = 800 or larger could achieve better performance.

## Reference

```
Wang J, Chen Y, Feng W, et al. Transfer learning with dynamic distribution adaptation[J]. 
ACM Transactions on Intelligent Systems and Technology (TIST), 2020, 11(1): 1-25.
```

or in bibtex style:

```
@article{wang2020transfer,
  title={Transfer learning with dynamic distribution adaptation},
  author={Wang, Jindong and Chen, Yiqiang and Feng, Wenjie and Yu, Han and Huang, Meiyu and Yang, Qiang},
  journal={ACM Transactions on Intelligent Systems and Technology (TIST)},
  volume={11},
  number={1},
  pages={1--25},
  year={2020},
  publisher={ACM New York, NY, USA}
}
```