# Domain adversarial neural network (DANN/RevGrad)

This is a Pytorch implementation of Unsupervised domain adaptation by backpropagation (also know as *DANN* or *RevGrad*).

## Requirements

- Python 3.6
- Pytorch 0.4.0

## Usage

### Dataset

First, you need download two datasets: source dataset mnist,

```
cd dataset
mkdir mnist
cd mnist
wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
```

and target dataset mnist_m from [pan.baidu.com](https://pan.baidu.com/s/1eShdX0u) or [Google Drive](https://drive.google.com/open?id=0B_tExHiYS-0veklUZHFYT19KYjg)

```
cd dataset
mkdir mnist_m
cd mnist_m
tar -zvxf mnist_m.tar.gz
```

### Training and testing

Then, run `main.py`

## Results

On MNIST - MNIST_M, I run 100 epochs and get the following results, which is extremely high compared to the paper:

![](https://s1.ax1x.com/2018/11/19/FpiIAJ.png)

**Reference**

Ganin Y, Lempitsky V. Unsupervised domain adaptation by backpropagation. ICML 2015.

If you have any questions, please open an issue.