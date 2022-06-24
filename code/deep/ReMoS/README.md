# ReMoS

> Reducing Defect Inheritance in Transfer Learning via Relevant Model Slicing

This is the implementation for the ICSE 2022 paper **ReMoS: Reducing Defect Inheritance in Transfer Learning via Relevant Model Slicing**. 

[[Paper](https://jd92.wang/assets/files/icse22-remos.pdf)] [[知乎解读](https://zhuanlan.zhihu.com/p/446453487)] [[Video](https://www.bilibili.com/video/BV1mi4y1C7bP)]

## Introduction

Transfer learning is a popular software reuse technique in the deep learning community that enables developers to build custom models (students) based on sophisticated pretrained models (teachers). However, some defects in the teacher model may also be inherited by students, such as well-known adversarial vulnerabilities and backdoors. We propose ReMoS, a relevant model slicing technique to reduce defect inheritance during transfer learning while retaining useful knowledge from the teacher model. Our experiments on seven DNN defects, four DNN models, and eight datasets demonstrate that ReMoS can reduce inherited defects effectively (by **63% to 86%** for CV tasks and by **40% to 61%** for NLP tasks) and efficiently with minimal sacrifice of accuracy (**3%** on average). 

## Requirements

See `requirements.txt`. All you need is a simple Pytorch environments with `advertorch` installed. Just `pip install -r requirements.txt`.

## Usage

See [`instructions.md`](instructions.md).

## Citation

If you think this work is helpful to your research, you can cite it as:

```
@inproceedings{zhang2022remos,
  title = {ReMoS: Reducing Defect Inheritance in Transfer Learning via Relevant Model Slicing},
  author = {Zhang, Ziqi and Li, Yuanchun and Wang, Jindong and Liu, Bingyan and Li, Ding and Chen, Xiangqun and Guo, Yao and Liu, Yunxin},
  booktitle = {44th International Conference on Software Engineering (ICSE)},
  year = {2022}
}
```


