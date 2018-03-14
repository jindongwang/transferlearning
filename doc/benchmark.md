# Benchmark

This file contains some benchmark results of popular transfer learning (domain adaptation) methods gathered from published papers. Right now there are only results of the most popular Office+Caltech10 datasets. You're welcome to add more results.

The full list of datasets can be found in [datasets](https://github.com/jindongwang/transferlearning/blob/master/doc/dataset.md).

## Office+Caltech SURF

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

## Office+Caltech10 Decaf6

Luckily, there is one article [16] that gathers the results of many popular methods on Decaf6 features. The benchmark is as the following image from that article:

![](https://raw.githubusercontent.com/jindongwang/transferlearning/master/png/result_office_caltech_decaf.jpg)

## Office-31

More and more researches chose to compare the accuracy on Office-31 datasets. Here is the comparison of both traditional and deep methods:

| Method | A - D | A - W | D - A | D - W | W-A | W-D | Average |
|:--------------:|:-----:|:-----:|:-----:|:-----:|:----:|:----:|:-------:|
| SVM | 55.7 | 50.6 | 46.5 | 93.1 | 43.0 | 97.4 | 64.4 |
| TCA | 45.4 | 40.5 | 36.5 | 78.2 | 34.1 | 84.0 | 53.1 |
| GFK | 52.0 | 48.2 | 41.8 | 86.5 | 38.6 | 87.5 | 59.1 |
| SA | 46.2 | 42.5 | 39.3 | 78.9 | 36.3 | 80.6 | 54.0 |
| DANN | 34.0 | 34.1 | 20.1 | 62.0 | 21.2 | 64.4 | 39.3 |
| CORAL | 57.1 | 53.1 | 51.1 | 94.6 | 47.3 | 98.2 | 66.9 |
| AlexNet | 63.8 | 61.6 | 51.1 | 95.4 | 49.8 | 99.0 | 70.1 |
| ResNet | 68.9 | 68.4 | 62.5 | 96.7 | 60.7 | 99.3 | 76.1 |
| DDC | 64.4 | 61.8 | 52.1 | 95.0 | 52.2 | 98.5 | 70.6 |
| DAN | 67.0 | 68.5 | 54.0 | 96.0 | 53.1 | 99.0 | 72.9 |
| RTN | 71.0 | 73.3 | 50.5 | 96.8 | 51.0 | 99.6 | 73.7 |
| RevGrad | 72.3 | 73.0 | 53.4 | 96.4 | 51.2 | 99.2 | 74.3 |
| DCORAL | 66.4 | 66.8 | 52.8 | 95.7 | 51.5 | 99.2 | 72.1 |
| DUCDA | 68.3 | 68.3 | 53.6 | 96.2 | 51.6 | 99.7 | 73.0 |
| JAN(AlexNet) | 71.8 | 74.9 | 58.3 | 96.6 | 55.0 | 99.5 | 76.0 |
| JAN-A(AlexNet) | 72.8 | 75.2 | 57.5 | 96.6 | 56.3 | 99.6 | 76.3 |
| JAN(ResNet) | 84.7 | 85.4 | 68.6 | 97.4 | 70.0 | 99.8 | 84.3 |
| JAN-A(ResNet) | 85.1 | 86.0 | 69.2 | 96.7 | 70.7 | 99.7 | 84.6 |

### References

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