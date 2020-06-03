# Benchmark

This file contains some benchmark results of popular transfer learning (domain adaptation) methods gathered from published papers. Right now there are only results of the most popular Office+Caltech10 datasets. You're welcome to add more results.

The full list of datasets can be found in [datasets](https://github.com/jindongwang/transferlearning/blob/master/data/dataset.md).

Here, we provide benchmark results for the following datasets:

- [Benchmark](#benchmark)
  - [Office-31 dataset](#office-31-dataset)
  - [Office-Home](#office-home)
  - [Image-CLEF DA](#image-clef-da)
  - [Office+Caltech](#officecaltech)
    - [SURF](#surf)
    - [Decaf6](#decaf6)
  - [MNIST+USPS](#mnistusps)
  - [References](#references)

## Office-31 dataset 

Using ResNet-50 features (compare with the latest deep methods with ResNet-50 as backbone). It seems **MEDA** is the only traditional method that can challenge these heavy deep adversarial methods.

Finetuned ResNet-50 models For Office-31 dataset: [BaiduYun](https://pan.baidu.com/s/1mRVDYOpeLz3siIId3tni6Q) | [Mega](https://mega.nz/file/dSpjyCwR#9ctB4q1RIE65a4NoJy0ox3gngh15cJqKq1XpOILJt9s)

| Cite        | Method    | A-W | D-W | W-D    | A-D | D-A | W-A   | AVG   |
|---------|-----------|-------|-------|--------|-------|-------|-------|-------|
| cvpr16  | ResNet-50 | 68.4  | 96.7  | 99.3   | 68.9  | 62.5  | 60.7  | 76.1  |
| icml15[17]  | DAN       | 80.5  | 97.1  | 99.6   | 78.6  | 63.6  | 62.8  | 80.4  |
| nips16[18]  | RTN       | 84.5  | 96.8  | 99.4   | 77.5  | 66.2  | 64.8  | 81.6  |
| icml15[19]  | DANN      | 82.0  | 96.9  | 99.1   | 79.7  | 68.2  | 67.4  | 82.2  |
| cvpr17[20]  | ADDA      | 86.2  | 96.2  | 98.4   | 77.8  | 69.5  | 68.9  | 82.9  |
| icml17[21]  | JAN       | 85.4  | 97.4  | 99.8   | 84.7  | 68.6  | 70.0  | 84.3  |
| cvpr17[22]  | GTA       | 89.5  | 97.9  | 99.8   | 87.7  | 72.8  | 71.4  | 86.5  |
| nips18[23]  | CDAN-RM   | 93.0  | 98.4  | 100.0  | 89.2  | 70.2  | 67.4  | 86.4  |
| nips18[23]  | CDAN-M    | 93.1  | 98.6  | 100.0  | 92.9  | 71.0  | 69.3  | 87.5  |
| cvpr18[24]  | CAN       | 81.5  | 98.2  | 99.7   | 85.5  | 65.9  | 63.4  | 82.4  |
| aaai19[25]  | JDDA      | 82.6  | 95.2  | 99.7   | 79.8  | 57.4  | 66.7  | 80.2  |
| aaai18[26]  | MADA      | 90.1  | 97.4  | 99.6   | 87.8  | 70.3  | 66.4  | 85.2  |
| acmmm18[27] | MEDA | 86.2  | 97.2  | 99.4  | 85.3  | 72.4  | 74.0  | 85.8 |

## Office-Home

Using ResNet-50 features (compare with the latest deep methods with ResNet-50 as backbone). Again, it seems that **MEDA** achieves the best performance. 

Finetuned ResNet-50 models For Office-Home dataset: [BaiduYun](https://pan.baidu.com/s/1i_g-QC2HZ0ZUhTnnySFIWw) | [Mega](https://mega.nz/#F!pGIkjIxC!MDD3ps6RzTXWobMfHh0Slw)

|  Cite       | Method    | Ar-Cl | Ar-Pr | Ar-Rw | Cl-Ar | Cl-Pr | Cl-Rw | Pr-Ar | Pr-Cl | Pr-Rw | Rw-Ar | Rw-Cl | Rw-Pr | Avg   |
|---------|-----------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
|  nips12       | AlexNet   | 26.4  | 32.6  | 41.3  | 22.1  | 41.7  | 42.1  | 20.5  | 20.3  | 51.1  | 31.0  | 27.9  | 54.9  | 34.3  |
| icml15[17]  | DAN       | 31.7  | 43.2  | 55.1  | 33.8  | 48.6  | 50.8  | 30.1  | 35.1  | 57.7  | 44.6  | 39.3  | 63.7  | 44.5  |
| icml15[19]  | DANN      | 36.4  | 45.2  | 54.7  | 35.2  | 51.8  | 55.1  | 31.6  | 39.7  | 59.3  | 45.7  | 46.4  | 65.9  | 47.3  |
| icml17[21]  | JAN       | 35.5  | 46.1  | 57.7  | 36.4  | 53.3  | 54.5  | 33.4  | 40.3  | 60.1  | 45.9  | 47.4  | 67.9  | 48.2  |
| nips18[23]  | CDAN-RM   | 36.2  | 47.3  | 58.6  | 37.3  | 54.4  | 58.3  | 33.2  | 43.9  | 62.1  | 48.2  | 48.1  | 70.7  | 49.9  |
| nips18[23]  | CDAN-M    | 38.1  | 50.3  | 60.3  | 39.7  | 56.4  | 57.8  | 35.5  | 43.1  | 63.2  | 48.4  | 48.5  | 71.1  | 51.0  |
| cvpr16  | ResNet-50 | 34.9  | 50.0  | 58.0  | 37.4  | 41.9  | 46.2  | 38.5  | 31.2  | 60.4  | 53.9  | 41.2  | 59.9  | 46.1  |
| icml15[17]  | DAN       | 43.6  | 57.0  | 67.9  | 45.8  | 56.5  | 60.4  | 44.0  | 43.6  | 67.7  | 63.1  | 51.5  | 74.3  | 56.3  |
| icml15[19]  | DANN      | 45.6  | 59.3  | 70.1  | 47.0  | 58.5  | 60.9  | 46.1  | 43.7  | 68.5  | 63.2  | 51.8  | 76.8  | 57.6  |
| icml17[21]  | JAN       | 45.9  | 61.2  | 68.9  | 50.4  | 59.7  | 61.0  | 45.8  | 43.4  | 70.3  | 63.9  | 52.4  | 76.8  | 58.3  |
| nips18[23]  | CDAN-RM   | 49.2  | 64.8  | 72.9  | 53.8  | 62.4  | 62.9  | 49.8  | 48.8  | 71.5  | 65.8  | 56.4  | 79.2  | 61.5  |
| nips18[23]  | CDAN-M    | 50.6  | 65.9  | 73.4  | 55.7  | 62.7  | 64.2  | 51.8  | 49.1  | 74.5  | 68.2  | 56.9  | 80.7  | 62.8  |
| acmmm18[27] | MEDA | **55.2**  | **76.2**  | **77.3**  | **58.0**  | **73.7**  | **71.9**  | **59.3**  | **52.4**  | **77.9**  | **68.2**  | **57.5**  | **81.8**  | **67.5**  |

## Image-CLEF DA

using ResNet-50 features (compare with the latest deep methods with ResNet-50 as backbone). Again, it seems that **MEDA** achieves the best performance. 

Finetuned ResNet-50 models For ImageCLEF dataset: [BaiduYun](https://pan.baidu.com/s/1y9tqyzBL7LZTd7Td380fxA) | [Mega](https://mega.nz/#F!QPJCzShS!b6qQUXWnCCGBMVs0m6MdQw)

| Cite | Method    | I-P   | P-I   | I-C   | C-I   | C-P   | P-C   | Avg   |
|---------|-----------|-------|-------|-------|-------|-------|-------|-------|
| nips12 | AlexNet   | 66.2  | 70.0  | 84.3  | 71.3  | 59.3  | 84.5  | 73.9  |
| icml15[17] | DAN       | 67.3  | 80.5  | 87.7  | 76.0  | 61.6  | 88.4  | 76.9  |
| icml15[19] | DANN      | 66.5  | 81.8  | 89.0  | 79.8  | 63.5  | 88.7  | 78.2  |
| icml17[21] | JAN       | 67.2  | 82.8  | 91.3  | 80.0  | 63.5  | 91.0  | 79.3  |
| nips18[23]| CDAN-RM   | 67.0  | 84.8  | 92.4  | 81.3  | 64.7  | 91.6  | 80.3  |
| nips18[23] | CDAN-M    | 67.7  | 83.3  | 91.8  | 81.5  | 63.0  | 91.5  | 79.8  |
| cvpr16 |  ResNet-50 | 74.8  | 83.9  | 91.5  | 78.0  | 65.5  | 91.2  | 80.7  |
| icml15[17]  | DAN       | 74.5  | 82.2  | 92.8  | 86.3  | 69.2  | 89.8  | 82.5  |
| icml15[19]  | DANN      | 75.0  | 86.0  | 96.2  | 87.0  | 74.3  | 91.5  | 85.0  |
| nips16[18]  | RTN       | 75.6  | 86.8  | 95.3  | 86.9  | 72.7  | 92.2  | 84.9  |
| icml17[19] | JAN       | 76.8  | 88.0  | 94.7  | 89.5  | 74.2  | 91.7  | 85.8  |
| aaai18[26] | MADA      | 75.0  | 87.9  | 96.0  | 88.8  | 75.2  | 92.2  | 85.8  |
| nips18[23] | CDAN-RM   | 77.2  | 88.3  | **98.3**  | 90.7  | 76.7  | 94.0  | 87.5  |
| nips18[23]  | CDAN-M    | 78.3  | 91.2  | 96.7  | 91.2  | 77.2  | 93.7  | 88.1  |
| cvpr18[24]  | CAN       | 78.2  | 87.5  | 94.2  | 89.5  | 75.8  | 89.2  | 85.7  |
| cvpr18[24] | iCAN      | 79.5  | 89.7  | 94.7  | 89.9  | 78.5  | 92.0  | 87.4  |
| acmmm18[27] | MEDA      | **80.2**  | **91.5**  | 96.2  | **92.7**  | **79.1**  | **95.8**  | **89.3**  |

## Office+Caltech

We provide results on SURF and DeCaf features.

### SURF

| **Dim** | **Method** | **C-A** | **C-W** | **C-D** | **A-C** | **A-W** | **A-D** | **W-C** | **W-A** | **W-D** | **D-C** | **D-A** | **D-W** |
|:---:|:--------:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| 100 | PCA+1NN | 36.95 | 32.54 | 38.22 | 34.73 | 35.59 | 27.39 | 26.36 | 31 | 77.07 | 29.65 | 32.05 | 75.93 |
| 100 | GFK+1NN | 41.02 | 40.68 | 38.85 | 40.25 | 38.98 | 36.31 | 30.72 | 29.75 | 80.89 | 30.28 | 32.05 | 75.59 |
| 100 | TCA+1NN | 38.2 | 38.64 | 41.4 | 37.76 | 37.63 | 33.12 | 29.3 | 30.06 | 87.26 | 31.7 | 32.15 | 86.1 |
| 100 | TSL+1NN | 44.47 | 34.24 | 43.31 | 37.58 | 33.9 | 26.11 | 29.83 | 30.27 | 87.26 | 28.5 | 27.56 | 85.42 |
| 100 | JDA+1NN | 44.78 | 41.69 | 45.22 | 39.36 | 37.97 | 39.49 | 31.17 | 32.78 | 89.17 | 31.52 | 33.09 | 89.49 |
| 100 | UDA+1NN | 47.39 | 46.56 | 48.41 | 41.41 | 43.05 | 42.04 | 32.41 | 34.45 | 91.08 | 34.19 | 34.24 | 90.85 |
| 30 | SA+1NN | 49.27 | 40 | 39.49 | 39.98 | 33.22 | 33.76 | 35.17 | 39.25 | 75.16 | 34.55 | 39.87 | 76.95 |
| 30 | SDA+1NN | 49.69 | 38.98 | 40.13 | 39.54 | 30.85 | 33.76 | 34.73 | 39.25 | 75.8 | 35.89 | 38.73 | 76.95 |
| 30 | GFK+1NN | 46.03 | 36.95 | 40.76 | 40.69 | 36.95 | 40.13 | 24.76 | 27.56 | 85.35 | 29.3 | 28.71 | 80.34 |
| 30 | TCA+1NN | 45.82 | 31.19 | 34.39 | 42.39 | 36.27 | 33.76 | 29.39 | 28.91 | 89.17 | 30.72 | 31 | 86.1 |
| 30 | JDA+1NN | 45.62 | 41.69 | 45.22 | 39.36 | 37.97 | 39.49 | 31.17 | 32.78 | 89.17 | 31.52 | 33.09 | 89.49 |
| 30 | TJM+1NN | 46.76 | 38.98 | 44.59 | 39.45 | 42.03 | 45.22 | 30.19 | 29.96 | 89.17 | 31.43 | 32.78 | 85.42 |
| 30 | SCA+1NN | 45.62 | 40 | 47.13 | 39.72 | 34.92 | 39.49 | 31.08 | 29.96 | 87.26 | 30.72 | 31.63 | 84.41 |
| 30 | JGSA+1NN | 53.13 | 48.47 | 48.41 | 41.5 | 45.08 | 45.22 | 33.57 | 40.81 | 88.54 | 30.28 | 38.73 | 93.22 |
| 20 | PCA+1NN | 36.95 | 32.54 | 38.22 | 34.73 | 35.59 | 27.39 | 26.36 | 29.35 | 77.07 | 29.65 | 32.05 | 75.93 |
| 20 | FSSL+1NN | 35.88 | 32.32 | 37.53 | 33.91 | 34.35 | 26.37 | 25.85 | 29.53 | 76.79 | 27.89 | 30.61 | 74.99 |
| 20 | TCA+1NN | 45.82 | 30.51 | 35.67 | 40.07 | 35.25 | 34.39 | 29.92 | 28.81 | 85.99 | 32.06 | 31.42 | 86.44 |
| 20 | GFK+1NN | 41.02 | 40.68 | 38.85 | 40.25 | 38.98 | 36.31 | 30.72 | 29.75 | 80.89 | 30.28 | 32.05 | 75.59 |
| 20 | TJM+1NN | 46.76 | 38.98 | 44.59 | 39.45 | 42.03 | 45.22 | 30.19 | 29.96 | 89.17 | 31.43 | 32.78 | 85.42 |
| 20 | VDA+1NN | 46.14 | 46.1 | 51.59 | 42.21 | 51.19 | 48.41 | 27.6 | 26.1 | 89.18 | 31.26 | 37.68 | 90.85 |
| no | 1NN | 23.7 | 25.76 | 25.48 | 26 | 29.83 | 25.48 | 19.86 | 22.96 | 59.24 | 26.27 | 28.5 | 63.39 |
| no | SVM | 55.64 | 45.22 | 43.73 | 45.77 | 42.04 | 39.66 | 31.43 | 34.76 | 82.8 | 29.39 | 26.62 | 63.39 |
| no | LapSVM | 56.27 | 45.8 | 43.73 | 44.23 | 42.74 | 39.79 | 31.99 | 34.77 | 83.43 | 29.49 | 27.37 | 64.31 |
| no | TKL | 54.28 | 46.5 | 51.19 | 45.59 | 49.04 | 46.44 | 34.82 | 40.92 | 83.44 | 35.8 | 40.71 | 84.75 |
| no | KMM | 48.32 | 45.78 | 53.53 | 42.21 | 42.38 | 42.72 | 29.01 | 31.94 | 71.98 | 31.61 | 32.2 | 72.88 |
| no | DTMKL | 54.33 | 42.04 | 44.74 | 45.01 | 36.94 | 40.85 | 32.5 | 36.53 | 88.85 | 32.1 | 34.03 | 81.69 |
| no | SKM+SVM | 53.97 | 43.31 | 43.05 | 44.7 | 37.58 | 42.37 | 31.34 | 35.07 | 89.81 | 30.37 | 30.27 | 81.02 |

**Results are coming from:**

- 1~5：[4]
- 6~15: [11]
- 16~21: [12]
- 22~28: [13]

- - -

### Decaf6

Luckily, there is one article [16] that gathers the results of many popular methods on Decaf6 features. The benchmark is as the following image from that article:

![](https://raw.githubusercontent.com/jindongwang/transferlearning/master/png/result_office_caltech_decaf.jpg)

- - -

## MNIST+USPS

There are plenty of different configurations in MNIST+USPS datasets. Here we only show some the recent results with the same network (based on LeNet) and training/test split.

|  Method | MNIST-USPS |
|:-------:|:----------:|
|   DDC   |    79.1    |
|   DANN  |    77.1    |
|  CoGAN  |    91.2    |
|   ADDA  |    89.4    |
|   MSTN  |    92.9    |
|   MEDA  |    94.3    |
|  CyCADA |    95.6    |
| PixelDA |    95.9    |
|   UNIT  |    95.9    |

## References

[1] Gong B, Shi Y, Sha F, et al. Geodesic flow kernel for unsupervised domain adaptation[C]//Computer Vision and Pattern Recognition (CVPR), 2012 IEEE Conference on. IEEE, 2012: 2066-2073.

[2] Russell B C, Torralba A, Murphy K P, et al. LabelMe: a database and web-based tool for image annotation[J]. International journal of computer vision, 2008, 77(1): 157-173.

[3] Griffin G, Holub A, Perona P. Caltech-256 object category dataset[J]. 2007.

[4] Long M, Wang J, Ding G, et al. Transfer feature learning with joint distribution adaptation[C]//Proceedings of the IEEE International Conference on Computer Vision. 2013: 2200-2207.

[5] http://attributes.kyb.tuebingen.mpg.de/

[6] http://www.cs.cmu.edu/afs/cs/project/PIE/MultiPie/Multi-Pie/Home.html

[7] http://www.cs.dartmouth.edu/~chenfang/proj_page/FXR_iccv13/

[8] M. Everingham, L. Van-Gool, C. K. Williams, J. Winn, and A. Zisserman, “The pascal visual object classes (VOC) challenge,” Int. J. Comput. Vis., vol. 88, no. 2, pp. 303–338, 2010.

[9] M. J. Choi, J. J. Lim, A. Torralba, and A. S. Willsky, “Exploiting hierarchical context on a large database of object categories,” in Proc. IEEE Conf. Comput. Vis. Pattern Recogit., 2010, pp. 129–136

[10] http://www.uow.edu.au/~jz960/

[11] Zhang J, Li W, Ogunbona P. Joint Geometrical and Statistical Alignment for Visual Domain Adaptation[C]. CVPR 2017.

[12] Tahmoresnezhad J, Hashemi S. Visual domain adaptation via transfer feature learning[J]. Knowledge and Information Systems, 2017, 50(2): 585-605.

[13] Long M, Wang J, Sun J, et al. Domain invariant transfer kernel learning[J]. IEEE Transactions on Knowledge and Data Engineering, 2015, 27(6): 1519-1532.

[14] Venkateswara H, Eusebio J, Chakraborty S, et al. Deep hashing network for unsupervised domain adaptation[C]. CVPR 2017.

[15] Daumé III H. Frustratingly easy domain adaptation[J]. arXiv preprint arXiv:0907.1815, 2009.

[16] Luo L, Chen L, Hu S. Discriminative Label Consistent Domain Adaptation[J]. arXiv preprint arXiv:1802.08077, 2018.

[17] Mingsheng Long, Yue Cao, Jianmin Wang, and Michael Jordan. Learning transferable features with deep adaptation networks. In ICML, pages 97–105, 2015.

[18] Mingsheng Long, Han Zhu, Jianmin Wang, and Michael I. Jordan. Unsupervised domain adaptation with residual transfer networks. In NIPS, 2016.

[19] Yaroslav Ganin and Victor Lempitsky. Unsupervised domain adaptation by backpropagation. In ICML, pages 1180–1189, 2015.

[20] Eric Tzeng, Judy Hoffman, Kate Saenko, and Trevor Darrell. Adversarial discriminative domain adaptation. In Computer Vision and Pattern Recognition (CVPR), volume 1, page 4, 2017.

[21] Mingsheng Long, Han Zhu, Jianmin Wang, and Michael I Jordan. Deep transfer learning with joint adaptation networks. In International Conference on Machine Learning, pages 2208–2217, 2017.

[22] Swami Sankaranarayanan, Yogesh Balaji, Carlos D Castillo, and Rama Chellappa. Generate to adapt: Aligning domains using generative adversarial networks. In CVPR, 2018.

[23] Mingsheng Long, Zhangjie Cao, Jianmin Wang, and Michael I Jordan. Conditional adversarial domain adaptation. In Advances in Neural Information Processing Systems, pages 1645–1655, 2018.

[24] Weichen Zhang, Wanli Ouyang, Wen Li, and Dong Xu. Collaborative and adversarial network for unsupervised domain adaptation. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 3801–3809, 2018.

[25] Chao Chen, Zhihong Chen, Boyuan Jiang, and Xinyu Jin. Joint domain alignment and discriminative feature learning for unsupervised deep domain adaptation. In AAAI, 2019.

[26] Zhongyi Pei, Zhangjie Cao, Mingsheng Long, and Jianmin Wang. Multi-adversarial domain adaptation. In AAAI Conference on Artificial Intelligence, 2018.

[27] Wang, Jindong, et al. "Visual Domain Adaptation with Manifold Embedded Distribution Alignment." 2018 ACM Multimedia Conference on Multimedia Conference. ACM, 2018.


