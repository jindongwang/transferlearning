# [Transfer Learning with Dynamic Adversarial Adaptation Network](https://arxiv.org/abs/1909.08184)

## Prerequisites:

* Python3
* PyTorch == 1.0.0 (with suitable CUDA and CuDNN version)
* Numpy
* argparse
* PIL
* tqdm

## Training:

You can run "./scripts/train.sh" to train and evaluate on the task.

## Contribution:

The contributions of this paper are four-fold:  
1. We propose a novel dynamic adversarial adaptation network to learn domain-invariant features. DAAN is accurate and robust, and can be easily implemented by most deep learning libraries.  
2. We propose the dynamic adversarial factor to easily, dynamically, and quantitatively evaluate the relative importance of the marginal and conditional distributions in adversarial transfer learning.  
3. We theoretically analyze the effectiveness of DAAN, and it can also be explained in an attention stragegy.  
4. Extensive experiments on public datasets demonstrate the significant superiority of our DAAN in both classification accuracy and the estimation of the dynamic adversarial factor.  

## Results:

* The architecture of the proposed Dynamic Adversarial Adaptation Network (DAAN):  
![image](https://github.com/Jindongwang/transferlearning/blob/master/code/deep/DAAN/assets/arch.png)  

* The classification accuracy on the ImageCLEF-DA dataset based on ResNet:  
![image](https://github.com/Jindongwang/transferlearning/blob/master/code/deep/DAAN/assets/imclef.png)  

* The classification accuracy on the OfficeHome dataset based on ResNet:  
![image](https://github.com/Jindongwang/transferlearning/blob/master/code/deep/DAAN/assets/officehome.png)  

* Ablation study of DAAN:  
![image](https://github.com/Jindongwang/transferlearning/blob/master/code/deep/DAAN/assets/ablation.png)  
We compare the performance of DAAN with DANN(ω= 0), MADA(ω= 1), and JAN(ω= 0.5). All these methods can be seen as special cases of our DAAN. The average results on each dataset indicate that it is not enough to only align the marginal or conditional distributions, or aligning them with equal weights.  


## Citation:

If you use this code for your research, please consider citing:

```
@inproceedings{yu2019transfer,
    title={Transfer Learning with Dynamic Adversarial Adaptation Network},
    author={Yu, Chaohui and Wang, Jindong and Chen, Yiqiang and Huang, Meiyu},
    booktitle={The IEEE International Conference on Data Mining (ICDM)},
    year={2019}
}
```
