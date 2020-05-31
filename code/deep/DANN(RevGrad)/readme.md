# Domain adversarial neural network (DANN/RevGrad)

This is a Pytorch implementation of Unsupervised domain adaptation by backpropagation (also know as *DANN* or *RevGrad*).

## Requirements

- Python 3.6
- Pytorch 0.4.0

## Usage

### Dataset

First, you need download two datasets: source dataset mnist. To organize the datasets, create a new folder: `mkdir dataset`.

MNIST dataset can be automatically downloaded by Pytorch.

MNIST_M dataset can be obtained from [pan.baidu.com](https://pan.baidu.com/s/1eShdX0u) or [Google Drive](https://drive.google.com/open?id=0B_tExHiYS-0veklUZHFYT19KYjg). The downloaded file will be `mnist_m.tar.gz`.

After that, extract the file into the `dataset` folder:

```
cd dataset
tar -zvxf mnist_m.tar.gz
```

### Run

Then, `python main.py`.

## Results

On MNIST - MNIST_M, I run 100 epochs and the results is around `55%`, which is clearly lower than that reported in the original paper. Maybe more tuning can be of help. Anyway, this is just for demo.

**Reference**

Ganin Y, Lempitsky V. Unsupervised domain adaptation by backpropagation. ICML 2015.

If you have any questions, please open an issue.