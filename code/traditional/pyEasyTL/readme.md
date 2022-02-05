# pyEasyTL

## Introduction
This [project](https://github.com/KodeWorker/pyEasyTL) is the implementation of EasyTL in Python.
The EasyTL paper on the [website](http://transferlearning.xyz/code/traditional/EasyTL/) show this domain adaptation method is intuitive and parametric-free.
The MATLAB source code is on this [Repo](https://github.com/jindongwang/transferlearning/tree/master/code/traditional/EasyTL).

- scipy.optimize.linprog is slower than PuLP
- M^(-1/2) = (M^(-1))^(1/2) = scipy.linalg.sqrtm(np.linalg.inv(np.array(cov_src)))
- scipy.linalg.sqrtm will introduce complex number [#3549](https://github.com/scipy/scipy/issues/3549) and cause our Dct parameter to be a complex array.

## To Do
- PCA_map in intra_alignment.py
- GFK_map in intra_alignment.py

## Dev Log
- **2020/02/25** PuLP type conversion problems (can't convert complex to float) is fixed
- **2020/02/24** write label_prop_v2.py using PuLP
- **2020/02/05** implement get_ma_dist and get_cosine_dist in EasyTL.py (fixed)
- **2020/02/03** more distance measurement in get_class_center
- **2020/01/31** CORAL_map still has some issue. (fixed)
- **2020/01/31** The primitive results of Amazon dataset show that we'v successfully implemented the EasyTL(c).

# Reference:
1. Easy Transfer Learning By Exploiting Intra-domain Structures
2. Geodesic Flow Kernel for Unsupervised Domain Adaptation