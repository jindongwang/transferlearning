# EasyTL: Practically Easy Transfer Learning

This directory contains the code for paper [Easy Transfer Learning By Exploiting Intra-domain Structures](http://jd92.wang/assets/files/a13_icme19.pdf) published at IEEE International Conference on Multimedia & Expo (ICME) 2019.

## Requirements

There are two implementations of EasyTL: Matlab and Python.

### Matlab 

The original code is written using Matlab R2017a. I think all versions after 2015 can run the code.

### Python

Thanks to [@KodeWorker](https://github.com/KodeWorker) for providing a Python implementation.

The Python version can be found in [here](https://github.com/jindongwang/transferlearning/tree/master/code/traditional/pyEasyTL).

## Demo & Usage

For Matlab, I offer three basic demos to reproduce the experiments in this paper: 

- For Amazon Review dataset, please run `demo_amazon_review.m`. 
- For Office-Caltech dataset, please run `demo_office_caltech.m`. 
- For ImageCLEF-DA and Office-Home datasets, please run `demo_image.m`.

Note that this directory does **not** contains any dataset. You can download them at the following links, and then add the folder to your Matlab path before running the code.

For Python, the demo code is `.py`.

[Download Amazon Review dataset](https://mega.nz/#F!RS43DADD!4pWwFA0CBJP1oLhAR23bTA).

[Download Office-Caltech with SURF features](https://mega.nz/#F!AaJTGIzD!XHM2XMsSd9V-ljVi0EtvFg)

[Download Image-CLEF ResNet-50 pretrained features](https://mega.nz/#F!QPJCzShS!b6qQUXWnCCGBMVs0m6MdQw)

[Download Office-Home ResNet-50 pretrained features](https://mega.nz/#F!pGIkjIxC!MDD3ps6RzTXWobMfHh0Slw)

You are welcome to run EasyTL on other public datasets such as [here](https://github.com/jindongwang/transferlearning/blob/master/data/dataset.md). You can also use your own datasets.

### Reference

If you find this code helpful, please cite it as:

`
Jindong Wang, Yiqiang Chen, Han Yu, Meiyu Huang, Qiang Yang. Easy Transfer Learning By Exploiting Intra-domain Structures. IEEE International Conference on Multimedia & Expo (ICME) 2019.
`

Or in bibtex style:

```
@inproceedings{wang2019easytl,
    title={Easy Transfer Learning By Exploiting Intra-domain Structures},
    author={Wang, Jindong and Chen, Yiqiang and Yu, Han and Huang, Meiyu and Yang, Qiang},
    booktitle={IEEE International Conference on Multimedia & Expo (ICME)},
    year={2019}
}
```

## Results

EasyTL achieved **state-of-the-art** performances compared to a lot of traditional and deep methods as of March 2019:

![](https://s2.ax1x.com/2019/04/02/A6UMoF.png)

![](https://s2.ax1x.com/2019/04/02/A6VOIO.png)

![](https://s2.ax1x.com/2019/04/02/A6NbIe.png)

![](https://s2.ax1x.com/2019/04/02/A6ZrFO.png)