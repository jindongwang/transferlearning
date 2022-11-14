# Benchmark

This file contains some benchmark results of popular transfer learning (domain adaptation) methods gathered from published papers. Right now there are only results of the most popular Office+Caltech10 datasets. You're welcome to add more results.

The full list of datasets can be found in [datasets](https://github.com/jindongwang/transferlearning/blob/master/data/dataset.md).

Here, we provide benchmark results for the following datasets:

- [Benchmark](#benchmark)
  - [Domain adaptation](#domain-adaptation)
    - [Adaptiope dataset](#adaptiope-dataset)
    - [Office-31 dataset](#office-31-dataset)
    - [Office-Home](#office-home)
    - [Image-CLEF DA](#image-clef-da)
    - [Office+Caltech](#officecaltech)
      - [SURF](#surf)
      - [Decaf6](#decaf6)
    - [MNIST+USPS](#mnistusps)
  - [Domain generalization](#domain-generalization)
    - [PACS (Resnet-18)](#pacs-resnet-18)
    - [Office-Home (Resnet-18)](#office-home-resnet-18)
  - [References](#references)

**Update at 2022-11:** You may want to check the *zero-shot* results by CLIP in [HERE](https://github.com/jindongwang/transferlearning/tree/master/code/clip#results), which I believe will blow your mind:)

## Domain adaptation

### Adaptiope dataset 

Using ResNet-50 features (compare with the latest deep methods with ResNet-50 as backbone). 


| Cite        | Method    | P-R | P-S | R-P    | R-S | S-P | S-R   | AVG   |
|---------|-----------|-------|-------|--------|-------|-------|-------|-------|
|   | Source Only | 63.6  | 26.7  | 85.3   | 27.6  | 7.6  | 2.0  | 35.5  |
| icml15[19]  | RSDA-DANN      | **78.6**  | 48.5  | 90.0   | 43.9  | 63.2  | 37.0  | 60.2  |
| icml18[30]  | RSDA-MSTN      | 73.8  | 59.2  | 87.5   | 50.3  | **69.5**  | 44.6  | 64.2  |
| TNNLS20[29] | DSAN | 77.8 | **60.1** | **91.9** | **55.7** | 68.8 | **47.8** | **67.0** |


### Office-31 dataset 

Using ResNet-50 features (compare with the latest deep methods with ResNet-50 as backbone). It seems **MEDA** is the only traditional method that can challenge these heavy deep adversarial methods.

Finetuned ResNet-50 models For Office-31 dataset: [BaiduYun](https://pan.baidu.com/s/1mRVDYOpeLz3siIId3tni6Q) | [Mega](https://mega.nz/file/dSpjyCwR#9ctB4q1RIE65a4NoJy0ox3gngh15cJqKq1XpOILJt9s)

**Results reported in original papers:**

| Cite        | Method    | A-W | D-W | W-D    | A-D | D-A | W-A   | AVG   |
|---------|-----------|-------|-------|--------|-------|-------|-------|-------|
| cvpr16  | ResNet-50 | 68.4  | 96.7  | 99.3   | 68.9  | 62.5  | 60.7  | 76.1  |
| icml15[17]  | DAN       | 80.5  | 97.1  | 99.6   | 78.6  | 63.6  | 62.8  | 80.4  |
| icml15[19]  | DANN      | 82.0  | 96.9  | 99.1   | 79.7  | 68.2  | 67.4  | 82.2  |
| cvpr17[20]  | ADDA      | 86.2  | 96.2  | 98.4   | 77.8  | 69.5  | 68.9  | 82.9  |
| icml17[21]  | JAN       | 85.4  | 97.4  | 99.8   | 84.7  | 68.6  | 70.0  | 84.3  |
| cvpr17[22]  | GTA       | 89.5  | 97.9  | 99.8   | 87.7  | 72.8  | 71.4  | 86.5  |
| cvpr18[24]  | CAN       | 81.5  | 98.2  | 99.7   | 85.5  | 65.9  | 63.4  | 82.4  |
| aaai19[25]  | JDDA      | 82.6  | 95.2  | 99.7   | 79.8  | 57.4  | 66.7  | 80.2  |
| acmmm18[27] | MEDA | 86.2  | 97.2  | 99.4  | 85.3  | 72.4  | 74.0  | 85.8 |
| neural network19[28] | MRAN | 91.4 | 96.9 | 99.8 | 86.4 | 68.3 | 70.9 | 85.6 |
| TNNLS20[29] | DSAN | **93.6** | **98.4** | **100.0** | **90.2** | **73.5** | **74.8** | **88.4** |

**Results using our unified and fair [codes](https://github.com/jindongwang/transferlearning/tree/master/code/DeepDA):**

|     Method        | D - A | D - W | A - W | W - A | A - D  | W - D  | Average |
|-------------|-------|-------|-------|-------|--------|--------|---------|
| Source-only | 66.17 | 97.61 | 80.63 | 65.07 | 82.73  | 100.00 | 82.03   |
| DAN [1]         | 68.16 | 97.48 | 85.79 | 66.56 | 84.34  | 100.00 | 83.72   |
| DeepCoral [2]       | 66.06 | 97.36 | 80.25 | 65.32 | 82.53  | 100.00 | 81.92   |
| DANN [3]        | 67.06 | 97.86 | 84.65 | 71.03 | 82.73  | 100.00 | 83.89   |
| DSAN [4]        | 76.04 | 98.49 | 94.34 | 72.91 | 89.96  | 100.00 | 88.62   |

### Office-Home

Using ResNet-50 features (compare with the latest deep methods with ResNet-50 as backbone). Again, it seems that **MEDA** achieves the best performance. 

Finetuned ResNet-50 models For Office-Home dataset: [BaiduYun](https://pan.baidu.com/s/1i_g-QC2HZ0ZUhTnnySFIWw) | [Mega](https://mega.nz/#F!pGIkjIxC!MDD3ps6RzTXWobMfHh0Slw)

**Results reported in original papers:**

|  Cite       | Method    | Ar-Cl | Ar-Pr | Ar-Rw | Cl-Ar | Cl-Pr | Cl-Rw | Pr-Ar | Pr-Cl | Pr-Rw | Rw-Ar | Rw-Cl | Rw-Pr | Avg   |
|---------|-----------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
|  nips12       | AlexNet   | 26.4  | 32.6  | 41.3  | 22.1  | 41.7  | 42.1  | 20.5  | 20.3  | 51.1  | 31.0  | 27.9  | 54.9  | 34.3  |
| icml15[17]  | DAN       | 31.7  | 43.2  | 55.1  | 33.8  | 48.6  | 50.8  | 30.1  | 35.1  | 57.7  | 44.6  | 39.3  | 63.7  | 44.5  |
| icml15[19]  | DANN      | 36.4  | 45.2  | 54.7  | 35.2  | 51.8  | 55.1  | 31.6  | 39.7  | 59.3  | 45.7  | 46.4  | 65.9  | 47.3  |
| icml17[21]  | JAN       | 35.5  | 46.1  | 57.7  | 36.4  | 53.3  | 54.5  | 33.4  | 40.3  | 60.1  | 45.9  | 47.4  | 67.9  | 48.2  |
| cvpr16  | ResNet-50 | 34.9  | 50.0  | 58.0  | 37.4  | 41.9  | 46.2  | 38.5  | 31.2  | 60.4  | 53.9  | 41.2  | 59.9  | 46.1  |
| icml15[17]  | DAN       | 43.6  | 57.0  | 67.9  | 45.8  | 56.5  | 60.4  | 44.0  | 43.6  | 67.7  | 63.1  | 51.5  | 74.3  | 56.3  |
| icml15[19]  | DANN      | 45.6  | 59.3  | 70.1  | 47.0  | 58.5  | 60.9  | 46.1  | 43.7  | 68.5  | 63.2  | 51.8  | 76.8  | 57.6  |
| icml17[21]  | JAN       | 45.9  | 61.2  | 68.9  | 50.4  | 59.7  | 61.0  | 45.8  | 43.4  | 70.3  | 63.9  | 52.4  | 76.8  | 58.3  |
| acmmm18[27] | MEDA | **55.2**  | **76.2**  | **77.3**  | 58.0  | **73.7**  | **71.9**  | 59.3  | 52.4  | 77.9  | 68.2  | 57.5  | 81.8  | **67.5**  |
| neural network19[28] | MRAN | 53.8 | 68.6 | 75.0 | 57.3 | 68.5 | 68.3 | 58.5 | 54.6 | 77.5 | 70.4 | 60.0 | 82.2 | 66.2  |
| TNNLS20[29] | DSAN | 54.4 | 70.8 | 75.4 | **60.4** | 67.8 | 68.0 | **62.6** | **55.9** | **78.5** | **73.8** | **60.6** | **83.1** | **67.6** |

**Results using our unified and fair [codes](https://github.com/jindongwang/transferlearning/tree/master/code/DeepDA):**

|     Method       | A - C | A - P | A - R | C - A | C - P | C - R | P - A | P - C | P - R | R - A | R - C | R - P | Average |
|-------------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|---------|
| Source-only | 51.04 | 68.21 | 74.85 | 54.22 | 63.64 | 66.84 | 53.65 | 45.41 | 74.57 | 65.68 | 53.56 | 79.34 | 62.58   |
| DAN [1]        | 52.51 | 68.48 | 74.82 | 57.48 | 65.71 | 67.82 | 55.42 | 47.51 | 75.28 | 66.54 | 54.36 | 79.91 | 63.82   |
| DeepCoral [2]      | 52.26 | 67.72 | 74.91 | 56.20 | 64.70 | 67.48 | 55.79 | 47.17 | 74.89 | 66.13 | 54.34 | 79.05 | 63.39   |
| DANN [3]        | 51.48 | 67.27 | 74.18 | 53.23 | 65.10 | 65.41 | 53.15 | 50.22 | 75.05 | 65.35 | 57.48 | 79.45 | 63.12   |
| DSAN [4]        | 54.48 | 71.12 | 75.37 | 60.53 | 70.92 | 68.53 | 62.71 | 56.04 | 78.29 | 74.37 | 60.34 | 82.99 | 67.97   |

### Image-CLEF DA

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
| icml17[19] | JAN       | 76.8  | 88.0  | 94.7  | 89.5  | 74.2  | 91.7  | 85.8  |
| cvpr18[24]  | CAN       | 78.2  | 87.5  | 94.2  | 89.5  | 75.8  | 89.2  | 85.7  |
| cvpr18[24] | iCAN      | 79.5  | 89.7  | 94.7  | 89.9  | 78.5  | 92.0  | 87.4  |
| acmmm18[27] | MEDA      | **80.2**  | 91.5  | 96.2  | 92.7  | 79.1  | 95.8  | 89.3  |
| neural network19[28] | MRAN      | 78.8 | 91.7 |  95.0 | 93.5 | 77.7 | 93.1 | 88.3 |
| TNNLS20[29] | DSAN      | **80.2** | **93.3** | **97.2** | **93.8** | **80.8** |**95.9** | **90.2** |

### Office+Caltech

We provide results on SURF and DeCaf features.

#### SURF

|            Task |     C - A   |     C - W   |     C - D   |     A - C   |     A - W   |     A - D   |     W - C |     W - A   |     W - D   |     D - C   |     D - A   |     D - W   |     Average |
|-----------------|-------------|-------------|-------------|-------------|-------------|-------------|-----------|-------------|-------------|-------------|-------------|-------------|-------------|
|     1NN         |     23.7    |     25.8    |     25.5    |     26      |     29.8    |     25.5    |     19.9  |     23      |     59.2    |     26.3    |     28.5    |     63.4    |     31.4    |
|     SVM         |     53.1    |     41.7    |     47.8    |     41.7    |     31.9    |     44.6    |     28.8  |     27.6    |     78.3    |     26.4    |     26.2    |     52.5    |     41.1    |
|     PCA         |     39.5    |     34.6    |     44.6    |     39      |     35.9    |     33.8    |     28.2  |     29.1    |     89.2    |     29.7    |     33.2    |     86.1    |     43.6    |
|     TCA         |     45.6    |     39.3    |     45.9    |     42      |     40      |     35.7    |     31.5  |     30.5    |     91.1    |     33      |     32.8    |     87.5    |     46.2    |
|     GFK         |     46      |     37      |     40.8    |     40.7    |     37      |     40.1    |     24.8  |     27.6    |     85.4    |     29.3    |     28.7    |     80.3    |     43.1    |
|     JDA         |     43.1    |     39.3    |     49      |     40.9    |     38      |     42      |     33    |     29.8    |     92.4    |     31.2    |     33.4    |     89.2    |     46.8    |
|     CORAL       |     52.1    |     46.4    |     45.9    |     45.1    |     44.4    |     39.5    |     33.7  |     36      |     86.6    |     33.8    |     37.7    |     84.7    |     48.8    |
|     SCA         |     45.6    |     40      |     47.1    |     39.7    |     34.9    |     39.5    |     31.1  |     30      |     87.3    |     30.7    |     31.6    |     84.4    |     45.2    |
|     JGSA        |     51.5    |     45.4    |     45.9    |     41.5    |     45.8    |     47.1    |     33.2  |     39.9    |     90.5    |     29.9    |     38      |     91.9    |     50      |
|     MEDA[27]        |     **56.5**    |     **53.9**    |     **50.3**    |     **43.9**    |     **53.2**    |     **45.9**    |     **34.0**    |     **42.7**    |     **88.5**    |     **34.9**    |     **41.2**    |     **87.5**    |     **52.7**    |
- - -

#### Decaf6

|        Method  |     C - A      |     C - W      |     C - D      |     A - C      |     A - W      |     A - D      |     W - C      |     W - A      |     W - D      |     D - C      |     D - A      |     D - W      |     Average    |
|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|
|     1NN        |        87.3    |        72.5    |        79.6    |        71.7    |        68.1    |        74.5    |        55.3    |        62.6    |        98.1    |        42.1    |        50      |        91.5    |        71.1    |
|        SVM     |        91.6    |        80.7    |        86      |        82.2    |        71.9    |        80.9    |        67.9    |        73.4    |        100     |        72.8    |        78.7    |        98.3    |        82      |
|        PCA     |        88.1    |        83.4    |        84.1    |        79.3    |        70.9    |        82.2    |        70.3    |        73.5    |        99.4    |        71.7    |        79.2    |        98      |        81.7    |
|        TCA     |        89.8    |        78.3    |        85.4    |        82.6    |        74.2    |        81.5    |        80.4    |        84.1    |        100     |        82.3    |        89.1    |        99.7    |        85.6    |
|        GFK     |        88.2    |        77.6    |        86.6    |        79.2    |        70.9    |        82.2    |        69.8    |        76.8    |        100     |        71.4    |        76.3    |        99.3    |        81.5    |
|        JDA     |        89.6    |        85.1    |        89.8    |        83.6    |        78.3    |        80.3    |        84.8    |        90.3    |        100     |        85.5    |        91.7    |        99.7    |        88.2    |
|        SCA     |        89.5    |        85.4    |        87.9    |        78.8    |        75.9    |        85.4    |        74.8    |        86.1    |        100     |        78.1    |        90      |        98.6    |        85.9    |
|        JGSA    |        91.4    |        86.8    |        **93.6**    |        84.9    |        81      |        88.5    |        85      |        90.7    |        100     |        86.2    |        92      |        99.7    |        90      |
|        CORAL   |        92      |        80      |        84.7    |        83.2    |        74.6    |        84.1    |        75.5    |        81.2    |        100     |        76.8    |        85.5    |        99.3    |        84.7    |
|        AlexNet |        91.9    |        83.7    |        87.1    |        83      |        79.5    |        87.4    |        73      |        83.8    |        100     |        79      |        87.1    |        97.7    |        86.1    |
|        DDC     |        91.9    |        85.4    |        88.8    |        85      |        86.1    |        89      |        78      |        84.9    |        100     |        81.1    |        89.5    |        98.2    |        88.2    |
|        DAN     |        92      |        90.6    |        89.3    |        84.1    |        91.8    |        91.7    |        81.2    |        92.1    |        100     |        80.3    |        90      |        98.5    |        90.1    |
|        DCORAL  |        92.4    |        91.1    |        91.4    |        84.7    |        -       |        -       |        79.3    |        -       |        -       |        82.8    |        -       |        -       |        -       |
|        MEDA[27]    |        **93.4**    |        **95.6**    |        91.1    |        **87.4**    |        88.1    |        88.1    |        **93.2**    |        **99.4**    |        99.4    |        **87.5**    |        **93.2**    |        97.6    |        **92.8**    |

- - -

### MNIST+USPS

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
|   DSAN  |    96.9    |

## Domain generalization

We get the results on PACS and Office-Home using our [codebase](https://github.com/jindongwang/transferlearning/blob/master/code/DeepDG).

### PACS (Resnet-18)

| Method | A | C | P | S | AVG |
|----------|----------|----------|----------|----------|----------|
| ERM | 76.90 | 76.41 | 93.83 | 65.26 | 78.10 |
| DANN | 79.30 | 77.13 | 92.93 | 77.20 | 81.64 |
| Mixup | 76.32 | 71.93 | 92.28 | 70.50 | 77.76 |
| RSC | 75.68 | 75.60 | 94.43 | 70.81 | 79.13 |
| CORAL | 73.34 | 76.62 | 87.96 | 73.86 | 77.94 |
| GroupDRO | 71.92 | 77.13 | 87.13 | 75.01 | 77.80 |

### Office-Home (Resnet-18)

| Method | A | C | P | R | AVG |
|----------|----------|----------|----------|----------|----------|
| ERM | 51.38 | 37.69 | 64.83 | 67.62 | 55.38 |
| DANN | 51.59 | 38.01 | 64.56 | 67.41 | 55.39 |
| Mixup | 50.97 | 37.64 | 62.81 | 66.28 | 54.43 |
| RSC | 52.45 | 39.15 | 65.24 | 67.96 | 56.20 |
| CORAL | 51.71 | 38.53 | 63.84 | 67.00 | 55.27 |
| MMD | 51.83 | 38.05 | 63.82 | 67.00 | 55.17 |
| MLDG | 45.65 | 32.28 | 58.62 | 60.64 | 49.30 |

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

[19] Yaroslav Ganin and Victor Lempitsky. Unsupervised domain adaptation by backpropagation. In ICML, pages 1180–1189, 2015.

[20] Eric Tzeng, Judy Hoffman, Kate Saenko, and Trevor Darrell. Adversarial discriminative domain adaptation. In Computer Vision and Pattern Recognition (CVPR), volume 1, page 4, 2017.

[21] Mingsheng Long, Han Zhu, Jianmin Wang, and Michael I Jordan. Deep transfer learning with joint adaptation networks. In International Conference on Machine Learning, pages 2208–2217, 2017.

[22] Swami Sankaranarayanan, Yogesh Balaji, Carlos D Castillo, and Rama Chellappa. Generate to adapt: Aligning domains using generative adversarial networks. In CVPR, 2018.

[24] Weichen Zhang, Wanli Ouyang, Wen Li, and Dong Xu. Collaborative and adversarial network for unsupervised domain adaptation. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 3801–3809, 2018.

[25] Chao Chen, Zhihong Chen, Boyuan Jiang, and Xinyu Jin. Joint domain alignment and discriminative feature learning for unsupervised deep domain adaptation. In AAAI, 2019.

[27] Wang, Jindong, et al. "Visual Domain Adaptation with Manifold Embedded Distribution Alignment." 2018 ACM Multimedia Conference on Multimedia Conference. ACM, 2018.

[28] Yongchun Zhu, Fuzhen Zhuang, Jindong Wang, et al. "Multi-representation adaptation network for cross-domain image classification." Neural Network 2019, 119.

[29] Yongchun Zhu, Fuzhen Zhuang, Jindong Wang, et al. "Deep Subdomain Adaptation Network for Image Classification." IEEE Transactions on Neural Networks and Learning Systems 2020.

[30] Xie S, Zheng Z, Chen L, et al. Learning semantic representations for unsupervised domain adaptation. International conference on machine learning. PMLR, 2018: 5423-5432.


