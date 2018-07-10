# Deep Transfer Learning on PyTorch
This is a PyTorch library for deep transfer learning. And this is a part of another repository---[transferlearning](https://github.com/jindongwang/transferlearning) which I will merge this into. If you need more code, please refer to [code](https://github.com/jindongwang/transferlearning/tree/master/code/deep). Here I have implemented some deep transfer methods as follows:
* DDC：Deep Domain Confusion Maximizing for Domain Invariance
* DAN: Learning Transferable Features with Deep Adaptation Networks
* Deep Coral: Deep CORAL Correlation Alignment for Deep Domain Adaptation

## Results on Office31
| Method | A - W | D - W | W - D | A - D | D - A | W - A | Average |
|:--------------:|:-----:|:-----:|:-----:|:-----:|:----:|:----:|:-------:|
| ResNet | 68.4±0.5 | 96.7±0.5 | 99.3±0.1 | 68.9±0.2 | 62.5±0.3 | 60.7±0.3 | 76.1 |
| DDC | 75.8±0.2 | 95.0±0.2 | 98.2±0.1 | 77.5±0.3 | 67.4±0.4 | 64.0±0.5 | 79.7 |
| DDCthis | 78.3±0.4 | 97.1±0.1 | 100.0±0.0 | 81.7±0.9 | 65.2±0.6 | 65.1±0.4 | 81.2 |
| DAN | 83.8±0.4 | 96.8±0.2 | 99.5±0.1 | 78.4±0.2 | 66.7±0.3 | 62.7±0.2 | 81.3 |
| DANthis | 82.6±0.7 | 97.7±0.1 | 100.0±0.0 | 83.1±0.9 | 66.8±0.3 | 66.6±0.4 | 82.8 |
| DCORALthis | 79.0±0.5 | 98.0±0.2 | 100.0±0.0 | 82.7±0.1 | 65.3±0.3 | 64.5±0.3 | 81.6 |

> Note that the results without **this** comes from [paper](http://ise.thss.tsinghua.edu.cn/~mlong/doc/multi-adversarial-domain-adaptation-aaai18.pdf). The results with **this** are run by myself with the code. 

## Contact
If you have any problem about this library, please create an Issue or send us an Email at:
* zhuyc0204@gmail.com
* jindongwang@outlook.com