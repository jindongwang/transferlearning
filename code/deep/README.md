# Deep Transfer Learning on PyTorch

This directory contains some re-implemented Pytorch codes for deep transfer learning.

## Codes

* [**DaNN**：Domain-adaptive Neural Network](https://github.com/jindongwang/transferlearning/tree/master/code/deep/DaNN)
* [**DAN**: Learning Transferable Features with Deep Adaptation Networks](https://github.com/jindongwang/transferlearning/tree/master/code/deep/DAN)
* [**Deep Coral**: Deep CORAL Correlation Alignment for Deep Domain Adaptation](https://github.com/jindongwang/transferlearning/tree/master/code/deep/DeepCoral)
* [**Finetune**: finetune using AlexNet and ResNet](https://github.com/jindongwang/transferlearning/tree/master/code/deep/finetune_AlexNet_ResNet)
* [**DANN/RevGrad**: Domain adversarial neural network](https://github.com/jindongwang/transferlearning/tree/master/code/deep/DANN(RevGrad))
* **MRAN** (Multi-representation adaptation network for cross-domain image classification, Neural Networks 2019) [72]
    - [Pytorch](https://github.com/jindongwang/transferlearning/tree/master/code/deep/MRAN)

## Results on Office31
| Method | A - W | D - W | W - D | A - D | D - A | W - A | Average |
|:--------------:|:-----:|:-----:|:-----:|:-----:|:----:|:----:|:-------:|
| DAN | 83.8±0.4 | 96.8±0.2 | 99.5±0.1 | 78.4±0.2 | 66.7±0.3 | 62.7±0.2 | 81.3 |
| DCORAL | 77.7±0.3 | 97.6±0.2 | 99.7±0.1 | 81.1±0.4 | 64.6±0.3 | 64.0±0.4 | 80.8 |

## Contact
If you have any problem about this library, please create an Issue or send us an Email at:
* zhuyc0204@gmail.com
* jindongwang@outlook.com