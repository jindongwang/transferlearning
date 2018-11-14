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

![](https://raw.githubusercontent.com/jindongwang/transferlearning/master/code/traditional/MEDA/results/result2.png)

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
