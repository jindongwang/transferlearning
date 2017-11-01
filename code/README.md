# code_transfer_learning

*Transfer learning toolbox; some useful transfer learning and domain adaptation codes*

> It is a waste of time looking for the codes from others. So I **clean** and **reimplement** them here in a way that you can **easily understand** and use. The following are some of the popular transfer learning (domain adaptation) methods in recent years, and I know most of them will be chosen to compare with your own method.

> It is still **on the go**. You are welcome to contribute and suggest other methods.

- - -

## Availiable codes for:

#### Non-deep learning

- **TCA** (Transfer Component Anaysis) [1]
	- [Matlab(Recommended!)](https://github.com/jindongwang/transferlearning/blob/master/code/MyTCA.m) | [Python](https://github.com/jindongwang/transferlearning/tree/master/code/TCA_python)
- **GFK** (Geodesic Flow Kernel) [2]
	- [Matlab](https://github.com/jindongwang/transferlearning/blob/master/code/MyGFK.m)
- **JDA** (Joint Distribution Adaptation) [3]
	- [Matlab](https://github.com/jindongwang/transferlearning/blob/master/code/MyJDA.m)
- **TJM** (Transfer Joint Matching) [4]
	- [Matlab](https://github.com/jindongwang/transferlearning/blob/master/code/MyTJM.m)
- **CORAL** (CORrelation ALignment) [5]
	- [Matlab](https://github.com/jindongwang/transferlearning/blob/master/code/MyCORAL.m) | [Github](https://github.com/VisionLearningGroup/CORAL)
- **JGSA** (Joint Geometrical and Statistical Alignment) [6]
	- [Matlab](https://github.com/jindongwang/transferlearning/blob/master/code/MyJGSA.m)
- **ARTL** (Adaptation Regularization) [7]
	- [Matlab](https://github.com/jindongwang/transferlearning/tree/master/code/MyARTL)
- **TrAdaBoost** [8]
	- [Python](https://github.com/chenchiwei/tradaboost)
- **SA** (Subspace Alignment) [11]
	- [Matlab](http://users.cecs.anu.edu.au/~basura/DA_SA/)
- **BDA** (Balanced Distribution Adaptation for Transfer Learning) [15]
	- [Matlab](https://github.com/jindongwang/transferlearning/tree/master/code/BDA)
- **MTLF** (Metric Transfer Learning) [16]
	- [Matlab](https://github.com/xyh2016/MTLF)

#### Deep learning

- **DAN/JAN** (Deep Adaptation Network/Joint Adaptation Network) [9,10]
	- [Caffe](https://github.com/thuml/Xlearn)
- **RTN** (Unsupervised Domain Adaptation with Residual Transfer Networks) [12]
	- [Caffe](https://github.com/thuml/Xlearn)
- **ADDA** (Adversarial Discriminative Domain Adaptation) [13]
	- [Python(Tensorflow)](https://github.com/erictzeng/adda)
- Unsupervised Domain Adaptation by Backpropagation [14]
	- [Python(Tensorflow)](https://github.com/shucunt/domain_adaptation)

- - -

#### [Code from HKUST](http://www.cse.ust.hk/TL/) [a bit old]

- - -

Testing **dataset** can be found [here](https://github.com/jindongwang/transferlearning/blob/master/doc/dataset.md).

- - -

#### References

[1] Pan S J, Tsang I W, Kwok J T, et al. Domain adaptation via transfer component analysis[J]TNN, 2011, 22(2): 199-210.

[2] Gong B, Shi Y, Sha F, et al. Geodesic flow kernel for unsupervised domain adaptation[C]//CVPR, 2012: 2066-2073.

[3] Long M, Wang J, Ding G, et al. Transfer feature learning with joint distribution adaptation[C]//ICCV. 2013: 2200-2207.

[4] Long M, Wang J, Ding G, et al. Transfer joint matching for unsupervised domain adaptation[C]//CVPR. 2014: 1410-1417.

[5] Sun B, Feng J, Saenko K. Return of Frustratingly Easy Domain Adaptation[C]//AAAI. 2016, 6(7): 8.

[6] Zhang J, Li W, Ogunbona P. Joint Geometrical and Statistical Alignment for Visual Domain Adaptation[C]//CVPR 2017.

[7] Long M, Wang J, Ding G, et al. Adaptation regularization: A general framework for transfer learning[J]//TKDE, 2014, 26(5): 1076-1089.

[8] Dai W, Yang Q, Xue G R, et al. Boosting for transfer learning[C]//ICML, 2007: 193-200.

[9] Long M, Cao Y, Wang J, et al. Learning transferable features with deep adaptation networks[C]//ICML. 2015: 97-105.

[10] Long M, Wang J, Jordan M I. Deep transfer learning with joint adaptation networks[J]//ICML 2017.

[11] Fernando B, Habrard A, Sebban M, et al. Unsupervised visual domain adaptation using subspace alignment[C]//ICCV. 2013: 2960-2967.

[12] Long M, Zhu H, Wang J, et al. Unsupervised domain adaptation with residual transfer networks[C]//NIPS. 2016.

[13] Tzeng E, Hoffman J, Saenko K, et al. Adversarial discriminative domain adaptation[J]. arXiv preprint arXiv:1702.05464, 2017.

[14] Ganin Y, Lempitsky V. Unsupervised domain adaptation by backpropagation[C]//International Conference on Machine Learning. 2015: 1180-1189.

[15] Jindong Wang, Yiqiang Chen, Shuji Hao, Wenjie Feng, and Zhiqi Shen. Balanced Distribution Adaptation for Transfer Learning. ICDM 2017.

[16] Y. Xu et al., "A Unified Framework for Metric Transfer Learning," in IEEE Transactions on Knowledge and Data Engineering, vol. 29, no. 6, pp. 1158-1171, June 1 2017. doi: 10.1109/TKDE.2017.2669193