# DeepDG: Deep Domain Generalization Toolkit

An easy-to-learn, easy-to-extend, and for-fair-comparison toolkit based on PyTorch for domain generalization (DG).

For a complete survey on these and more DG algorithms, please refer to this survey published at *IEEE TKDE 2022*: [Generalizing to Unseen Domains: A Survey on Domain Generalization](https://arxiv.org/abs/2103.03097). More recent news can be obtained from our [IJCAI 2022 tutorial on domain generalization](https://dgresearch.github.io/).

## Implemented Algorithms

We currently support the following algoirthms. We are working on more algorithms. Of course, you are welcome to add your algorithms here.

1. ERM
2. DDC (Deep Domain Confusion, arXiv 2014) [1]
3. CORAL (COrrelation Alignment, ECCV-16) [2]
4. DANN (Domain-adversarial Neural Network, JMLR-16) [3]
5. MLDG (Meta-learning Domain Generalization, AAAI-18) [4]
6. Mixup (ICLR-18) [5]
7. RSC (Representation Self-Challenging, ECCV-20) [6]
8. GroupDRO (ICLR-20) [7]
9. ANDMask (ICLR-21) [8]
10. VREx (ICML-21) [9]
11. DIFEX (TMLR-22) [10]

## Installation

You can either clone this whole big repo:

```
git clone https://github.com/jindongwang/transferlearning.git
cd code/DeepDG
```

Or *if you just want to use this folder* (i.e., no other things in this big transferlearning repo), you can go to [this site](https://minhaskamal.github.io/DownGit/#/home) and paste the url of this DeepDG folder (https://github.com/jindongwang/transferlearning/edit/master/code/DeepDG) and then download only this folder!

We recommend to use `Python 3.8.8` which is our development environment. 
It is better to use the same environment following (https://hub.docker.com/r/jindongwang/docker).

## Datasets

Our code supports the following dataset:

* [Office-31](https://github.com/jindongwang/transferlearning/tree/master/data#office-31)
* [Office-Home](https://github.com/jindongwang/transferlearning/tree/master/data#office-home)
* [Office-Caltech](https://github.com/jindongwang/transferlearning/tree/master/data#office-caltech10)
* [PACS](https://drive.google.com/uc?id=0B6x7gtvErXgfbF9CSk53UkRxVzg)
* [Digit-Five](https://wjdcloud.blob.core.windows.net/dataset/dg5.tar.gz)
* [VLCS](https://drive.google.com/uc?id=1skwblH1_okBwxWxmRsp9_qi15hyPpxg8)

If you want to use your own dataset, please organize your data in the following structure.

```
RootDir
└───Domain1Name
│   └───Class1Name
│       │   file1.jpg
│       │   file2.jpg
│       │   ...
│   ...
└───Domain2Name
|   ...    
```

And then, modifty `util/util.py` to contain the dataset.

## Usage

1. Modify the file in the scripts
2. The main script file is `train.py`, which can be runned by using `run.sh` from `scripts/run.sh`: `cd scripts; bash run.sh`.

## Customization

It is easy to design your own method following the steps:

1. Add your method (a Python file) to `alg/algs`, and add the reference to it in the `alg/alg.py`

2. Modify `utils/util.py` to make it adapt your own parameters

3. Midify `scripts/run.sh` and execuate it

## Results

We present results of our implementations on 2 popular benchmarks: **PACS** and **Office-Home**. We did not perform careful parameter tuning. You can easily reproduce our results using the hyperparameters [here](https://github.com/jindongwang/transferlearning/blob/master/code/DeepDG/scripts/paramsref.md).

### Results on PACS (ResNet-18)

| Method   | A     | C     | P     | S     | AVG   |
|----------|-------|-------|-------|-------|-------|
| ERM      | 81.1  | 77.94 | 95.03 | 76.94 | 82.75 |
| DANN     | 82.86 | 78.33 | 96.11 | 76.99 | 83.57 |
| Mixup    | 81.84 | 75.43 | 95.27 | 76.51 | 82.26 |
| RSC      | 82.13 | 77.99 | 94.43 | 79.87 | 83.6  |
| MMD      | 80.32 | 76.45 | 92.46 | 83.63 | 83.21 |
| CORAL    | 79.39 | 77.9  | 91.98 | 82.03 | 82.83 |
| GroupDRO | 79.15 | 76.75 | 91.32 | 81.52 | 82.19 |
| ANDMask  | 80.81 | 73.29 | 95.81 | 71.95 | 80.47 |
| Vrex     | 81.54 | 78.11 | 95.39 | 80.35 | 83.85 |
|DIFEX-ori | 82.86 | 78.46 | 94.97 | 79.41 | 83.93 |
|DIFEX-norm| 83.40 | 79.74 | 95.03 | 79.10 | 84.32 |

### Results on Office-Home (ResNet-18)

| Method   | A     | C     | P     | R     | AVG   |
|----------|-------|-------|-------|-------|-------|
| ERM      | 57.77 | 50.63 | 71.3  | 74.45 | 63.54 |
| DANN     | 57.6  | 48.52 | 71.16 | 72.99 | 62.57 |
| Mixup    | 58.71 | 51    | 72.2  | 75.42 | 64.33 |
| RSC      | 57.07 | 50.77 | 71.93 | 73.63 | 63.35 |
| MMD      | 59.29 | 50.52 | 72.34 | 74.43 | 64.15 |
| CORAL    | 59.29 | 50.15 | 72.25 | 74.2  | 63.97 |
| GroupDRO | 59.09 | 50.22 | 71.91 | 74.48 | 63.92 |
| ANDMask  | 53.61 | 47.54 | 69.36 | 72.23 | 60.69 |
| Vrex     | 59.09 | 49.81 | 71.64 | 74.82 | 63.84 |

### Results on PACS (ResNet-50)

| Method   | A     | C     | P     | S     | AVG   |
|----------|-------|-------|-------|-------|-------|
| ERM      | 83.2  | 81.7  | 96.65 | 83.69 | 86.31 |
| DANN     | 87.26 | 83.45 | 95.33 | 84.35 | 87.6  |
| Mixup    | 89.36 | 82.08 | 96.65 | 84.63 | 88.18 |
| RSC      | 87.84 | 80.33 | 97.72 | 81.5  | 86.85 |
| MMD      | 85.74 | 83.58 | 95.51 | 83.46 | 87.07 |
| CORAL    | 86.77 | 84.04 | 94.85 | 85.95 | 87.9  |
| GroupDRO | 84.18 | 83.15 | 96.11 | 83.94 | 86.84 |
| ANDMask  | 84.91 | 76.45 | 97.72 | 81.19 | 85.07 |
| Vrex     | 87.11 | 82.85 | 96.95 | 84.07 | 87.75 |

### Results on Office-Home (ResNet-50)

| Method   | A     | C     | P     | R     | AVG   |
|----------|-------|-------|-------|-------|-------|
| ERM      | 67.66 | 55.92 | 77.7  | 80.47 | 70.44 |
| DANN     | 67.49 | 56.66 | 76.73 | 79.21 | 70.02 |
| Mixup    | 67.41 | 58.24 | 78.46 | 80.84 | 71.24 |
| RSC      | 66.3  | 55.21 | 76.95 | 79    | 69.36 |
| MMD      | 66.71 | 56.54 | 78.37 | 79.8  | 70.36 |
| CORAL    | 66.58 | 56.17 | 78.55 | 79.76 | 70.27 |
| GroupDRO | 66.87 | 57.04 | 77.97 | 79.69 | 70.39 |
| ANDMask  | 62.3  | 54.78 | 74.99 | 78.31 | 67.59 |
| Vrex     | 68.19 | 56.29 | 78.35 | 80.42 | 70.81 |

## Contribution

The toolkit is under active development and contributions are welcome! Feel free to submit issues and PRs to ask questions or contribute your code. If you would like to implement new features, please submit a issue to discuss with us first. You are welcome to our another related project: [DeepDA](https://github.com/jindongwang/transferlearning/edit/master/code/DeepDA).

## Acknowledgment

Great thanks to [DomainBed](https://github.com/facebookresearch/DomainBed). We simplify their work to make it easy to perform experiments and extensions. Moreover, we add some new features.

## Reference

[1] Tzeng, Eric, et al. "Deep domain confusion: Maximizing for domain invariance." arXiv preprint arXiv:1412.3474 (2014).

[2] Sun, Baochen, and Kate Saenko. "Deep coral: Correlation alignment for deep domain adaptation." European conference on computer vision. Springer, Cham, 2016.

[3] Ganin, Yaroslav, and Victor Lempitsky. "Unsupervised domain adaptation by backpropagation." International conference on machine learning. PMLR, 2015.

[4] Li, Da, et al. "Learning to generalize: Meta-learning for domain generalization." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 32. No. 1. 2018.

[5] Zhang, Hongyi, et al. "mixup: Beyond empirical risk minimization." ICLR 2018.

[6] Huang, Zeyi, et al. "Self-challenging improves cross-domain generalization." ECCV 2020.

[7] Sagawa S, Koh P W, Hashimoto T B, et al. Distributionally robust neural networks for group shifts: On the importance of regularization for worst-case generalization, ICLR 2020.

[8] Parascandolo G, Neitz A, ORVIETO A, et al. Learning explanations that are hard to vary[C]//International Conference on Learning Representations. 2020.

[9] Krueger D, Caballero E, Jacobsen J H, et al. Out-of-distribution generalization via risk extrapolation (rex)[C]//International Conference on Machine Learning. PMLR, 2021.

[10] Wang Lu, Jindong Wang, et al. Domain-invariant Feature Exploration for Domain Generalization. TMLR, 2022.

## Citation

If you think this toolkit or the results are helpful to you and your research, please cite us!

```
@Misc{deepdg,
howpublished = {\url{https://github.com/jindongwang/transferlearning/tree/master/code/DeepDG}},   
title = {DeepDG: Deep Domain Generalization Toolkit},  
author = {Wang, Jindong and Wang Lu}
}  
```

## Contact

- Wang lu: luwang@ict.ac.cn
- [Jindong Wang](http://www.jd92.wang/): jindongwang@outlook.com
