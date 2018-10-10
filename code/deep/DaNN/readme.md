# DaNN (Domain Adaptive Neural Networks)

- - -

This is the implementation of **Domain Adaptive Neural Networks (DaNN)** using PyTorch. The original paper can be found at https://link.springer.com/chapter/10.1007/978-3-319-13560-1_76.

DaNN is rather a *simple* neural network (with only 1 hidden layer) for domain adaptation. But its idea is important that brings MMD (maximum mean discrpancy) for adaptation in neural network. From then on, many researches are following this idea to embed MMD or other measurements (e.g. CORAL loss, Wasserstein distance) into deeper (e.g. AlexNet, ResNet) networks.

I think if you are beginning to learn **deep transfer learning**, it is better to start with the most original and simple one. 

## Dataset

The original paper adopted the popular *Office+Caltech10* dataset. You can download them [HERE](https://github.com/jindongwang/transferlearning/blob/master/data/dataset.md#download) and put them into a new folder named `data`.

- - -

## Usage

Make sure you have Python 3.6 and PyTorch 0.3.0. As for other requirements, I'm sure you are satisfied.

- `DaNN.py` is the DaNN model
- `mmd.py` is the MMD measurement. You're welcome to change it to others.
- `data_loader.py` is the help function for loading data.
- `main.py` is the main training and test program. You can directly run this file by `python main.py`.

- - -

### Reference

Ghifary M, Kleijn W B, Zhang M. Domain adaptive neural networks for object recognition[C]//Pacific Rim International Conference on Artificial Intelligence. Springer, Cham, 2014: 898-904.
