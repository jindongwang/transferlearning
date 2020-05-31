# Geodesic Flow Kernwl (GFK)

This is the implementation of Geodesic Flow Kernel (GFK) in both Matlab and Python.

## Matlab version

Just use `GFK.m`. Note that `getGFKDim.m` is the implementation of *subspace disagreement measurement* proposed in GFK paper.

## Python version

**Note:** We may not be able to use `bob` any more. Therefore, the code of GFK in python is also not available since to achieve the same results in Matlab, we need to use `bob` library.

Therefore, I can do nothing about it except suggesting that you use the Matlab version instead. If you still want to use Python code, then you can try to call the Matlab code in your Python script by following some tutorials online.

See the `GFK.py` file.

Requirements:
- Python 3
- Numpy and Scipy
- Sklearn
- **Bob**

**Note:** This Python version is wrapped from Bob: https://www.idiap.ch/software/bob/docs/bob/bob.learn.linear/stable/_modules/bob/learn/linear/GFK.html#GFKMachine.

Therefore, if you want to run and use this program, you should install `bob` by following its official instructions in [here](https://www.idiap.ch/software/bob/docs/bob/docs/stable/bob/bob/doc/install.html).

There are many libraries in bob. A minimum request is to install `bob.math` and `bob.learn` by following the instructions.

## Run

The test dataset can be downloaded [HERE](https://github.com/jindongwang/transferlearning/tree/master/data). Then, you can run the file.

The python file can be used directly, while the matlab code just contains the core function of TCA. To use the matlab code, you need to learn from the code of [BDA](https://github.com/jindongwang/transferlearning/tree/master/code/BDA) and set out the parameters.

### Reference

Gong B, Shi Y, Sha F, et al. Geodesic flow kernel for unsupervised domain adaptation[C]//2012 IEEE Conference on Computer Vision and Pattern Recognition. IEEE, 2012: 2066-2073.

If you have any questions, please open an issue.