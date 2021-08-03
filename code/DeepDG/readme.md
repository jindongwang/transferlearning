# DeepDG: Deep Domain Generalization Toolkit

An easy-to-learn, easy-to-extend, and for-fair-comparison toolkit based on PyTorch for domain generalization (DG).

For a complete survey on these and more DG algorithms, please refer to this survey published at IJCAI 2021 survey track: [Generalizing to Unseen Domains: A Survey on Domain Generalization](https://arxiv.org/abs/2103.03097).

## Implemented Algorithm

As initial version, we support the following algoirthms. We are working on more algorithms. Of course, you are welcome to add your algorithms here.

1. ERM
2. DDC [1]
3. CORL [2]
4. DANN [3]
5. MLDG [4]
6. Mixup [5]
7. RSC [6]
8. GroupDRO [7]

## Installation

```
git clone https://github.com/jindongwang/transferlearning.git
cd code/DeepDG
pip install -r requirements.txt
```
We recommend to use `Python 3.8.8` which is our development environment.

## Dataset

Our code supports the following dataset:

* [office](https://mega.nz/file/dSpjyCwR#9ctB4q1RIE65a4NoJy0ox3gngh15cJqKq1XpOILJt9s)

* [office-home](https://www.hemanthdv.org/officeHomeDataset.html)

* [office-caltech](https://pan.baidu.com/s/14JEGQ56LJX7LMbd6GLtxCw)

* [PACS](https://drive.google.com/uc?id=0B6x7gtvErXgfbF9CSk53UkRxVzg)

* [dg5](https://transferlearningdrive.blob.core.windows.net/teamdrive/dataset/dg5.tar.gz)

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

And then, modifty util/util.py to contain the dataset.

## Usage

1. Modify the file in the scripts
2. `bash run.sh`

## Customization

It is easy to design your own method following the steps:

1. Add your method to alg/algs, and add the reference to it in the alg/alg.py

2. Modify utils/util.py to make it adapt your own parameters

3. Midify scripts/run.sh and execuate it

## Results

We present results of our implementations on 2 popular benchmarks: PACS and Office-Home. We did not perform careful parameter tuning. You can easily reproduce our results!

1. Results on PACS (resnet-18)

| Method | A | C | P | S | AVG |
|----------|----------|----------|----------|----------|----------|
| ERM | 60.84 | 54.31 | 93.59 | 39.70 | 62.11 |
| DANN | 63.82 | 57.68 | 93.65 | 38.61 | 63.44 |
| Mixup | 63.87 | 54.65 | 93.05 | 43.57 | 63.79 |
| RSC | 63.38 | 55.12 | 93.95 | 40.19 | 63.16 |
| CORAL | 58.54 | 58.70 | 89.94 | 36.04 | 60.80 |
| MMD | 58.94 | 57.81 | 89.70 | 36.04 | 60.62 |
| MLDG | 58.79 | 55.80 | 95.39 | 52.94 | 65.73 |
| GroupDRO | 54.15 | 58.66 | 88.38 | 42.99 | 61.04 |

2. Results on Office-Home (resnet-18)

| Method | A | C | P | R | AVG |
|----------|----------|----------|----------|----------|----------|
| ERM | 51.38 | 37.69 | 64.83 | 67.62 | 55.38 |
| DANN | 51.59 | 38.01 | 64.56 | 67.41 | 55.39 |
| Mixup | 50.97 | 37.64 | 62.81 | 66.28 | 54.43 |
| RSC | 52.45 | 39.15 | 65.24 | 67.96 | 56.20 |
| CORAL | 51.71 | 38.53 | 63.84 | 67.00 | 55.27 |
| MMD | 51.83 | 38.05 | 63.82 | 67.00 | 55.17 |
| MLDG | 45.65 | 32.28 | 58.62 | 60.64 | 49.30 |
|GroupDRO | 49.40 | 37.75 | 59.16 | 64.54 | 52.71|

## Contribution

The toolkit is under active development and contributions are welcome! Feel free to submit issues and PRs to ask questions or contribute your code. If you would like to implement new features, please submit a issue to discuss with us first.

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
