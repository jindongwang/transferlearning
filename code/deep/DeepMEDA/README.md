# DeepMEDA

A PyTorch implementation of 'Transfer Learning with Dynamic Distribution Adaptation' which has published on ACM Transactions on Intelligent Systems and Technology.

Matlab version is [HERE](https://github.com/jindongwang/transferlearning/tree/master/code/traditional/MEDA/matlab).

## Requirement

* python 3
* pytorch 1.x
* Numpy, scikit-learn

## Usage

1. You can download Office31 dataset [here](https://pan.baidu.com/s/1o8igXT4#list/path=%2F). And then unrar dataset in ./dataset/.
2. You can change the `source_name` and `target_name` in `Config.py` to set different transfer tasks.
3. Run `python main.py`.

> Note that for tasks D-A and W-A, setting epochs = 800 or larger could achieve better performance.

## Reference

```
Wang J, Chen Y, Feng W, et al. Transfer learning with dynamic distribution adaptation[J]. ACM Transactions on Intelligent Systems and Technology (TIST), 2020, 11(1): 1-25.
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