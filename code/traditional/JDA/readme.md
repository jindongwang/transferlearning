# Joint Distribution Adaptation

This is the implementation of Joint Distribution Adaptation (JDA) in Python and Matlab.

Remark: The core of JDA is a generalized eigendecompsition problem. In Matlab, it can be solved by calling `eigs()` function. In Python, the implementation `scipy.linalg.eig()` function can do the same thing. However, they are a bit different. So the results may be different.

The test dataset can be downloaded [HERE](https://github.com/jindongwang/transferlearning/tree/master/code/traditional/data).

**Reference**

Long M, Wang J, Ding G, et al. Transfer feature learning with joint distribution adaptation[C]//Proceedings of the IEEE international conference on computer vision. 2013: 2200-2207.