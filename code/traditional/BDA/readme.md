## Balanced Distributon Adaptation (BDA)

This directory contains the code for paper [Balanced Distribution Adaptation for Transfer Learning](http://jd92.wang/assets/files/a08_icdm17.pdf).

We support both Matlab and Python.

The test dataset can be downloaded [HERE](https://github.com/jindongwang/transferlearning/tree/master/data).

### Important Notice

Please note that the original BDA (with $\mu$) has been extensively extended by MEDA (Manifold Embedded Distribution Alignment) published in ACM International Conference on Multimedia (ACMMM) 2018. Its code can be found [here](https://github.com/jindongwang/transferlearning/tree/master/code/traditional/MEDA).

Hence, from now on, the name **BDA** will specifically refer to **W-BDA** (for imbalanced transfer) in this ICDM paper. Therefore, the code default will also be the imbalanced version. If still want to use the original BDA, you can change the settings in the code.

### Citation
You can cite this paper as

- Bibtex style:
  ```
  @inproceedings{wang2017balanced,
	title={Balanced Distribution Adaptation for Transfer Learning},
	author={Wang, Jindong and Chen, Yiqiang and Hao, Shuji and Feng, Wenjie and Shen, Zhiqi},
	booktitle={The IEEE International conference on data mining (ICDM)},
	year={2017},
	pages={1129--1134}
  }
	```

- GB/T 7714 style:
  
  Jindong Wang, Yiqiang Chen, Shuji Hao, Wenjie Feng, and Zhiqi Shen. Balanced Distribution Adaptation for Transfer Learning. ICDM 2017. pp. 1129-1134.


