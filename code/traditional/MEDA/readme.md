# MEDA: Manifold Embedded Distribution Alignment

This directory contains the code for paper [Visual Domain Adaptation with Manifold Embedded Distribution Alignment](http://jd92.wang/assets/files/a11_mm18.pdf) published at ACM Multimedia conference (ACM MM) 2018 as an Oral presentation. This paper is also lucky to be ranked as **Top 10 papers**.

## Usage

The original code is written using Matlab R2017a. I think all versions after 2015 can run the code.

For Python users, I add a `MEDA.py` implementation. The Python version will need to import GFK module (can be found [here](https://github.com/jindongwang/transferlearning/tree/master/code/traditional/GFK)). However, this Python version is only for reference since the graph Laplacian (as exactly in Matlab) is not implemented.

## Demo

I offer a basic demo to run on the Office+Caltech10 datasets. Download the datasets [here](https://pan.baidu.com/s/1bp4g7Av#list/path=%2F) and put the data (mat files) into the `data` folder.

Run `demo_office_caltech_surf.m`.

## Results

MEDA achieved **state-of-the-art** performances compared to a lot of traditional and deep methods as of 2018. The testing datasets are most popular domain adaptation and transfer learning datasets: Office+Caltech10, Office-31, USPS+MNIST, ImageNet+VOC2007.

The following results are from the original paper and its [supplementary file](https://www.jianguoyun.com/p/DRuWOFkQjKnsBRjkr2E).

- Office-31 dataset using ResNet-50 features (compare with the latest deep methods with ResNet-50 as backbone):

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
| cvpr18  | iCAN      | 92.5  | 69.9  | 90.1   | 72.1  | 100   | 98.8  | 87.2  |
| aaai19  | JDDA      | 82.6  | 95.2  | 99.7   | 79.8  | 57.4  | 66.7  | 80.2  |
| aaai18  | MADA      | 90.1  | 97.4  | 99.6   | 87.8  | 70.3  | 66.4  | 85.2  |
|         |           |       |       |        |       |       |       |       |
| ACMMM18 | MEDA      | 91.9  | 97.6  | 99.4   | 93.8  | 85.1  | 82.3  | 91.7  |

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
