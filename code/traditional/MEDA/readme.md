# MEDA: Manifold Embedded Distribution Alignment

This directory contains the code for paper [Visual Domain Adaptation with Manifold Embedded Distribution Alignment](http://jd92.wang/assets/files/a11_mm18.pdf) published at ACM Multimedia conference (ACM MM) 2018 as an Oral presentation. This paper is also lucky to be ranked as **Top 10 papers**.

## Usage

The original code is written using Matlab R2017a. I think all versions after 2015 can run the code.

For Python users, I add a `MEDA.py` implementation. The Python version will need to import GFK module (can be found [here](https://github.com/jindongwang/transferlearning/tree/master/code/traditional/GFK)). However, this Python version is only for reference since the graph Laplacian (as exactly in Matlab) is not implemented.

## Demo

I offer a basic demo to run on the Office+Caltech10 datasets. Download the datasets [here](https://github.com/jindongwang/transferlearning/tree/master/data) and put the data (mat files) into the `data` folder.

Run `demo_office_caltech_surf.m`.

## Results

MEDA achieved **state-of-the-art** performances compared to a lot of traditional and deep methods as of 2018. The testing datasets are most popular domain adaptation and transfer learning datasets: Office+Caltech10, Office-31, USPS+MNIST, ImageNet+VOC2007.

The following results are from the original paper and its [supplementary file](https://www.jianguoyun.com/p/DRuWOFkQjKnsBRjkr2E).

## Office-31 dataset 

Using ResNet-50 features (compare with the latest deep methods with ResNet-50 as backbone). It seems **MEDA** is the only traditional method that can challenge these heavy deep adversarial methods.

[Download Office-31 ResNet-50 features](https://pan.baidu.com/s/1UoyJSqoCKCda-NcP-zraVg)

|         | Method    | A - W | D - W | W-D    | A - D | D - A | W-A   | AVG   |
|---------|-----------|-------|-------|--------|-------|-------|-------|-------|
| cvpr16  | ResNet-50 | 68.4  | 96.7  | 99.3   | 68.9  | 62.5  | 60.7  | 76.1  |
| icml15  | DAN       | 80.5  | 97.1  | 99.6   | 78.6  | 63.6  | 62.8  | 80.4  |
| nips16  | RTN       | 84.5  | 96.8  | 99.4   | 77.5  | 66.2  | 64.8  | 81.6  |
| icml15  | DANN      | 82.0  | 96.9  | 99.1   | 79.7  | 68.2  | 67.4  | 82.2  |
| cvpr17  | ADDA      | 86.2  | 96.2  | 98.4   | 77.8  | 69.5  | 68.9  | 82.9  |
| icml17  | JAN       | 85.4  | 97.4  | 99.8   | 84.7  | 68.6  | 70.0  | 84.3  |
| cvpr17  | GTA       | 89.5  | 97.9  | 99.8   | 87.7  | 72.8  | 71.4  | 86.5  |
| nips18  | CDAN-RM   | 93.0  | 98.4  | 100.0  | 89.2  | 70.2  | 67.4  | 86.4  |
| nips18  | CDAN-M    | 93.1  | 98.6  | 100.0  | 92.9  | 71.0  | 69.3  | 87.5  |
| cvpr18  | CAN       | 81.5  | 63.4  | 85.5   | 65.9  | 99.7  | 98.2  | 82.4  |
| aaai19  | JDDA      | 82.6  | 95.2  | 99.7   | 79.8  | 57.4  | 66.7  | 80.2  |
| aaai18  | MADA      | 90.1  | 97.4  | 99.6   | 87.8  | 70.3  | 66.4  | 85.2  |
| ACMMM18 | MEDA | 86.2  | 97.2  | 99.4  | 85.3  | 72.4  | 74.0  | 85.7 |

## Office-Home

Using ResNet-50 features (compare with the latest deep methods with ResNet-50 as backbone). Again, it seems that **MEDA** achieves the best performance. 

[Download Office-Home ResNet-50 pretrained features](https://pan.baidu.com/s/1qvcWJCXVG8JkZnoM4BVoGg)

|         | Method    | Ar-Cl | Ar-Pr | Ar-Rw | Cl-Ar | Cl-Pr | Cl-Rw | Pr-Ar | Pr-Cl | Pr-Rw | Rw-Ar | Rw-Cl | Rw-Pr | Avg   |
|---------|-----------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
|         | AlexNet   | 26.4  | 32.6  | 41.3  | 22.1  | 41.7  | 42.1  | 20.5  | 20.3  | 51.1  | 31.0  | 27.9  | 54.9  | 34.3  |
| icml15  | DAN       | 31.7  | 43.2  | 55.1  | 33.8  | 48.6  | 50.8  | 30.1  | 35.1  | 57.7  | 44.6  | 39.3  | 63.7  | 44.5  |
| icml15  | DANN      | 36.4  | 45.2  | 54.7  | 35.2  | 51.8  | 55.1  | 31.6  | 39.7  | 59.3  | 45.7  | 46.4  | 65.9  | 47.3  |
| icml17  | JAN       | 35.5  | 46.1  | 57.7  | 36.4  | 53.3  | 54.5  | 33.4  | 40.3  | 60.1  | 45.9  | 47.4  | 67.9  | 48.2  |
| nips18  | CDAN-RM   | 36.2  | 47.3  | 58.6  | 37.3  | 54.4  | 58.3  | 33.2  | 43.9  | 62.1  | 48.2  | 48.1  | 70.7  | 49.9  |
| nips18  | CDAN-M    | 38.1  | 50.3  | 60.3  | 39.7  | 56.4  | 57.8  | 35.5  | 43.1  | 63.2  | 48.4  | 48.5  | 71.1  | 51.0  |
| cvpr16  | ResNet-50 | 34.9  | 50.0  | 58.0  | 37.4  | 41.9  | 46.2  | 38.5  | 31.2  | 60.4  | 53.9  | 41.2  | 59.9  | 46.1  |
| icml15  | DAN       | 43.6  | 57.0  | 67.9  | 45.8  | 56.5  | 60.4  | 44.0  | 43.6  | 67.7  | 63.1  | 51.5  | 74.3  | 56.3  |
| icml15  | DANN      | 45.6  | 59.3  | 70.1  | 47.0  | 58.5  | 60.9  | 46.1  | 43.7  | 68.5  | 63.2  | 51.8  | 76.8  | 57.6  |
| icml17  | JAN       | 45.9  | 61.2  | 68.9  | 50.4  | 59.7  | 61.0  | 45.8  | 43.4  | 70.3  | 63.9  | 52.4  | 76.8  | 58.3  |
| nips18  | CDAN-RM   | 49.2  | 64.8  | 72.9  | 53.8  | 62.4  | 62.9  | 49.8  | 48.8  | 71.5  | 65.8  | 56.4  | 79.2  | 61.5  |
| nips18  | CDAN-M    | 50.6  | 65.9  | 73.4  | 55.7  | 62.7  | 64.2  | 51.8  | 49.1  | 74.5  | 68.2  | 56.9  | 80.7  | 62.8  |
| ACMMM18 | MEDA | **54.6**  | **75.2**  | **77.0**  | **56.5**  | **72.8**  | **72.3**  | **59.0**  | **51.9**  | **78.2**  | 67.7  | **57.2**  | **81.8**  | **67.0**  |

## Image-CLEF DA

using ResNet-50 features (compare with the latest deep methods with ResNet-50 as backbone). Again, it seems that **MEDA** achieves the best performance. 

[Download Image-CLEF ResNet-50 pretrained features](https://pan.baidu.com/s/16wBgDJI6drA0oYq537h4FQ)

| Method    | I-P   | P-I   | I-C   | C-I   | C-P   | P-C   | Avg   |
|-----------|-------|-------|-------|-------|-------|-------|-------|
| AlexNet   | 66.2  | 70.0  | 84.3  | 71.3  | 59.3  | 84.5  | 73.9  |
| DAN       | 67.3  | 80.5  | 87.7  | 76.0  | 61.6  | 88.4  | 76.9  |
| DANN      | 66.5  | 81.8  | 89.0  | 79.8  | 63.5  | 88.7  | 78.2  |
| JAN       | 67.2  | 82.8  | 91.3  | 80.0  | 63.5  | 91.0  | 79.3  |
| CDAN-RM   | 67.0  | 84.8  | 92.4  | 81.3  | 64.7  | 91.6  | 80.3  |
| CDAN-M    | 67.7  | 83.3  | 91.8  | 81.5  | 63.0  | 91.5  | 79.8  |
| ResNet-50 | 74.8  | 83.9  | 91.5  | 78.0  | 65.5  | 91.2  | 80.7  |
| DAN       | 74.5  | 82.2  | 92.8  | 86.3  | 69.2  | 89.8  | 82.5  |
| DANN      | 75.0  | 86.0  | 96.2  | 87.0  | 74.3  | 91.5  | 85.0  |
| RTN       | 75.6  | 86.8  | 95.3  | 86.9  | 72.7  | 92.2  | 84.9  |
| JAN       | 76.8  | 88.0  | 94.7  | 89.5  | 74.2  | 91.7  | 85.8  |
| MADA      | 75.0  | 87.9  | 96.0  | 88.8  | 75.2  | 92.2  | 85.8  |
| CDAN-RM   | 77.2  | 88.3  | **98.3**  | 90.7  | 76.7  | 94.0  | 87.5  |
| CDAN-M    | 78.3  | 91.2  | 96.7  | 91.2  | 77.2  | 93.7  | 88.1  |
| CAN       | 78.2  | 87.5  | 94.2  | 89.5  | 75.8  | 89.2  | 85.7  |
| iCAN      | 79.5  | 89.7  | 94.7  | 89.9  | 78.5  | 92.0  | 87.4  |
| MEDA      | **79.7**  | **92.5**  | 95.7  | **92.2**  | **78.5**  | **95.5**  | **89.0**  |

- Office-31 dataset using DECAF features (compare with deep methods with AlexNet):

![](https://raw.githubusercontent.com/jindongwang/transferlearning/master/code/traditional/MEDA/results/result2.png)

- Office+Caltech 10 datasets and MNIST+USPS and ImageNet+VOC:

![](https://raw.githubusercontent.com/jindongwang/transferlearning/master/code/traditional/MEDA/results/result1.png)

## Reference

If you use this code, please cite it as:

`
Jindong Wang, Wenjie Feng, Yiqiang Chen, Han Yu, Meiyu Huang, Philip S. Yu. Visual Domain Adaptation with Manifold Embedded Distribution Alignment. ACM Multimedia conference 2018.
`

Or in bibtex style:

```
@inproceedings{wang2018visual,
    title={Visual Domain Adaptation with Manifold Embedded Distribution Alignment},
    author={Wang, Jindong and Feng, Wenjie and Chen, Yiqiang and Yu, Han and Huang, Meiyu and Yu, Philip S},
    booktitle={ACM Multimedia Conference (ACM MM)},
    year={2018}
}
```
