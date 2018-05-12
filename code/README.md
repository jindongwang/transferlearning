# code_transfer_learning

*Some useful transfer learning and domain adaptation codes*

> It is a waste of time looking for the codes from others. So I **collect** or **reimplement** them here in a way that you can **easily** use. The following are some of the popular transfer learning (domain adaptation) methods in recent years, and I know most of them will be chosen to **compare** with your own method.

> It is still **on the go**. You are welcome to contribute and suggest other methods.

This document contains codes from several aspects: **tutorial**, **theory**, **traditional** methods, and **deep** methods.

- - -

## Tutorial

- [基于深度学习和迁移学习的识花实践(Tensorflow)](https://cosx.org/2017/10/transfer-learning/)

- [基于Pytorch的图像分类](https://github.com/miguelgfierro/sciblog_support/blob/master/A_Gentle_Introduction_to_Transfer_Learning/Intro_Transfer_Learning.ipynb)

- [使用Pytorch进行finetune](https://github.com/Spandan-Madan/Pytorch_fine_tuning_Tutorial)

- **Tensorflow Hub** (Tensorflow library released by Google for transfer learning)
	- [Tensorflow](https://github.com/tensorflow/hub)
- **Pytorch CNN finetune** (Finetune tutorial for pytorch)
	- [Pytorch](https://github.com/creafz/pytorch-cnn-finetune)

## Theory

- MMD及多核MMD代码：[Matlab](https://github.com/lopezpaz/classifier_tests/tree/master/code/unit_test_mmd) | [Python](https://github.com/jindongwang/transferlearning/tree/master/code/basic/mmd.py)

## Traditional transfer learning methods

- **SVM** (baseline)
	- [Matlab](https://github.com/jindongwang/transferlearning/tree/master/code/SVM.m)
- **TCA** (Transfer Component Anaysis, TNN-11) [1]
	- [Matlab(Recommended!)](https://github.com/jindongwang/transferlearning/blob/master/code/MyTCA.m) | [Python](https://github.com/jindongwang/transferlearning/tree/master/code/TCA_python)
- **GFK** (Geodesic Flow Kernel, CVPR-12) [2]
	- [Matlab](https://github.com/jindongwang/transferlearning/blob/master/code/MyGFK.m)
- **DA-NBNN** (Frustratingly Easy NBNN Domain Adaptation, ICCV-13) [39]
	- [Matlab](https://github.com/enoonIT/nbnn-nbnl/tree/master/DANBNN_demo)
- **JDA** (Joint Distribution Adaptation, ICCV-13) [3]
	- [Matlab](https://github.com/jindongwang/transferlearning/blob/master/code/MyJDA.m)
- **TJM** (Transfer Joint Matching, CVPR-14) [4]
	- [Matlab](https://github.com/jindongwang/transferlearning/blob/master/code/MyTJM.m)
- **CORAL** (CORrelation ALignment, AAAI-15) [5]
	- [Matlab](https://github.com/jindongwang/transferlearning/blob/master/code/CORAL) | [Github](https://github.com/VisionLearningGroup/CORAL)
- **JGSA** (Joint Geometrical and Statistical Alignment, CVPR-17) [6]
	- [Matlab](https://www.uow.edu.au/~jz960/codes/JGSA-r.rar)
- **ARTL** (Adaptation Regularization, TKDE-14) [7]
	- [Matlab](https://github.com/jindongwang/transferlearning/tree/master/code/MyARTL)
- **TrAdaBoost** (ICML-07)[8]
	- [Python](https://github.com/chenchiwei/tradaboost)
- **SA** (Subspace Alignment, ICCV-13) [11]
	- [Matlab(official)](http://users.cecs.anu.edu.au/~basura/DA_SA/) | [Matlab](https://github.com/jindongwang/transferlearning/tree/master/code/SA_SVM.m)
- **BDA** (Balanced Distribution Adaptation for Transfer Learning, ICDM-17) [15]
	- [Matlab](https://github.com/jindongwang/transferlearning/tree/master/code/BDA)
- **MTLF** (Metric Transfer Learning, TKDE-17) [16]
	- [Matlab](https://github.com/xyh2016/MTLF)
- **Open Set Domain Adaptation** (ICCV-17) [19]
	- [Matlab(official)](https://github.com/Heliot7/open-set-da)
- **TAISL** (When Unsupervised Domain Adaptation Meets Tensor Representations, ICCV-17) [21]
	- [Matlab(official)](https://github.com/poppinace/TAISL)
- **STL** (Stratified Transfer Learning for Cross-domain Activity Recognition, PerCom-18) [22]
	- [Matlab](https://github.com/jindongwang/activityrecognition/tree/master/code/percom18_stl)
- **LSA** (Landmarks-based kernelized subspace alignment for unsupervised domain adaptation, CVPR-15) [29]
	- [Matlab](http://homes.esat.kuleuven.be/~raljundi/papers/LSA%20Clean%20Code.zip)
- **OTL** (Online Transfer Learning, ICML-10) [31]
	- [Matlab(official)](http://stevenhoi.org/otl)


## Deep transfer learning methods

- **DaNN** (Domain Adaptive Neural Network, PRICAI-14) [41]
	- [PyTorch](https://github.com/jindongwang/transferlearning/tree/master/code/deep/DaNN)
- **DeepCORAL** (Deep CORAL: Correlation Alignment for Deep Domain Adaptation) [33]
	- [PyTorch](https://github.com/SSARCandy/DeepCORAL) | [中文解读](https://ssarcandy.tw/2017/10/31/deep-coral/)
- **DAN/JAN** (Deep Adaptation Network/Joint Adaptation Network, ICML-15,17) [9,10]
	- [PyTorch(Official)](https://github.com/thuml/Xlearn/tree/master/pytorch) | [Caffe(Official)](https://github.com/thuml/Xlearn)
- **RTN** (Unsupervised Domain Adaptation with Residual Transfer Networks, NIPS-16) [12]
	- [Caffe](https://github.com/thuml/Xlearn)
- **ADDA** (Adversarial Discriminative Domain Adaptation, arXiv-17) [13]
	- [Tensorflow(Official)](https://github.com/erictzeng/adda) | [Pytorch](https://github.com/corenel/pytorch-adda)
- **RevGrad** (Unsupervised Domain Adaptation by Backpropagation, ICML-15) [14]
	- [Caffe(Official)](https://github.com/ddtm/caffe/tree/grl)|[Tensorflow(third party)](https://github.com/shucunt/domain_adaptation) | [PyTorch](https://github.com/fungtion/DANN)
- **DANN** Domain-Adversarial Training of Neural Networks (JMLR-16)[17] 
	- [Python(pure)](https://github.com/GRAAL-Research/domain_adversarial_neural_network) | [Tensorflow](https://github.com/jindongwang/tf-dann)
- Associative Domain Adaptation (ICCV-17) [18]
	- [Tensorflow](https://github.com/haeusser/learning_by_association)
- Deep Hashing Network for Unsupervised Domain (CVPR-17) [20]
	- [Matlab](https://github.com/hemanthdv/da-hash)
- **CCSL** (Unified Deep Supervised Domain Adaptation and Generalization, ICCV-17) [23]
	- [Python(Keras)](https://github.com/samotiian/CCSA)
- **MRN** (Learning Multiple Tasks with Multilinear Relationship Networks, NIPS-17) [24]
	- [Pytorch](https://github.com/thuml/MTlearn)
- **AutoDIAL** (Automatic DomaIn Alignment Layers, ICCV-17) [25]
	- [Caffe](https://github.com/ducksoup/autodial)
- **DSN** (Domain Separation Networks, NIPS-16) [26]
	- [Tensorflow](https://github.com/tensorflow/models/tree/master/research/domain_adaptation)
- **DRCN** (Deep Reconstruction-Classification Networks for Unsupervised Domain Adaptation, ECCV-16) [27]
	- [Keras](https://github.com/ghif/drcn) | [Pytorch](https://github.com/fungtion/DRCN)
- Multi-task Autoencoders for Domain Generalization (ICCV-15) [28]
	- [Keras](https://github.com/ghif/mtae)
- Encoder based lifelong learning (ICCV-17) [30]
	- [Matlab](https://github.com/rahafaljundi/Encoder-Based-Lifelong-learning)
- **MECA** (Minimal-Entropy Correlation Alignment, ICLR-18) [32]
	- [Python](https://github.com/pmorerio/minimal-entropy-correlation-alignment)
- **WAE** (Wasserstein Auto-Encoders, ICLR-18) [34]
	- [Python(Tensorflow)](https://github.com/tolstikhin/wae)
- **ATDA** (Asymmetric Tri-training for Unsupervised Domain Adaptation, ICML-15) [35]
	- [Pytorch](https://github.com/corenel/pytorch-atda#pytorch-atda)
- **PixelDA_GAN** (Unsupervised pixel-level domain adaptation with GAN, CVPR-17) [36]
	- [Pytorch](https://github.com/vaibhavnaagar/pixelDA_GAN)
- **ARDA** (Adversarial Representation Learning for Domain Adaptation) [37]
	- [Pytorch](https://github.com/corenel/pytorch-arda)
- **DiscoGAN** (Learning to Discover Cross-Domain Relations with Generative Adversarial Networks) [38]
	- [Pytorch](https://github.com/carpedm20/DiscoGAN-pytorch)
- **MADA** (Multi-Adversarial Domain Adaptation, AAAI-18) [40]
	- [Caffe(official)](https://github.com/thuml/mada)
- **MCD** (Maximum Classifier Discrepancy, CVPR-18) [42]
	- [Pytorch(official)](https://github.com/mil-tokyo/MCD_DA)
- Adversarial Feature Augmentation for Unsupervised Domain Adaptation (CVPR-18) [43]
	- [Tensorflow](https://github.com/ricvolpi/adversarial-feature-augmentation)
- Deep Mutual Learning (CVPR 2018) [44]
	- [Tensorflow](https://github.com/YingZhangDUT/Deep-Mutual-Learning)
- Self-ensembling for visual domain adaptation (ICLR 2018) [45]
	- [Pytorch](https://github.com/Britefury/self-ensemble-visual-domain-adapt)

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

[17] Ganin Y, Ustinova E, Ajakan H, et al. Domain-adversarial training of neural networks[J]. Journal of Machine Learning Research, 2016, 17(59): 1-35.

[18] Haeusser P, Frerix T, Mordvintsev A, et al. Associative Domain Adaptation[C]. ICCV, 2017.

[19] Pau Panareda Busto, Juergen Gall. Open set domain adaptation. ICCV 2017.

[20] Venkateswara H, Eusebio J, Chakraborty S, et al. Deep hashing network for unsupervised domain adaptation[C]. CVPR 2017.

[21] H. Lu, L. Zhang, et al. When Unsupervised Domain Adaptation Meets Tensor Representations. ICCV 2017.

[22] J. Wang, Y. Chen, L. Hu, X. Peng, and P. Yu. Stratified Transfer Learning for Cross-domain Activity Recognition. 2018 IEEE International Conference on Pervasive Computing and Communications (PerCom).

[23] Motiian S, Piccirilli M, Adjeroh D A, et al. Unified deep supervised domain adaptation and generalization[C]//The IEEE International Conference on Computer Vision (ICCV). 2017, 2.

[24] Long M, Cao Z, Wang J, et al. Learning Multiple Tasks with Multilinear Relationship Networks[C]//Advances in Neural Information Processing Systems. 2017: 1593-1602.

[25] Maria Carlucci F, Porzi L, Caputo B, et al. AutoDIAL: Automatic DomaIn Alignment Layers[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2017: 5067-5075.

[26] Bousmalis K, Trigeorgis G, Silberman N, et al. Domain separation networks[C]//Advances in Neural Information Processing Systems. 2016: 343-351.

[27] M. Ghifary, W. B. Kleijn, M. Zhang, D. Balduzzi, and W. Li. "Deep Reconstruction-Classification Networks for Unsupervised Domain Adaptation (DRCN)", European Conference on Computer Vision (ECCV), 2016

[28] M. Ghifary, W. B. Kleijn, M. Zhang, D. Balduzzi.
Domain Generalization for Object Recognition with Multi-task Autoencoders,
accepted in International Conference on Computer Vision (ICCV 2015), Santiago, Chile.

[29] Aljundi R, Emonet R, Muselet D, et al. Landmarks-based kernelized subspace alignment for unsupervised domain adaptation[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2015: 56-63.

[30] Rannen A, Aljundi R, Blaschko M B, et al. Encoder based lifelong learning[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2017: 1320-1328.

[31] Peilin Zhao and Steven C.H. Hoi. OTL: A Framework of Online Transfer Learning. ICML 2010.

[32] Pietro Morerio, Jacopo Cavazza, Vittorio Murino. Minimal-Entropy Correlation Alignment for Unsupervised Deep Domain Adaptation. ICLR 2018.

[33] Sun B, Saenko K. Deep coral: Correlation alignment for deep domain adaptation[C]//European Conference on Computer Vision. Springer, Cham, 2016: 443-450.

[34] Tolstikhin I, Bousquet O, Gelly S, et al. Wasserstein Auto-Encoders[J]. arXiv preprint arXiv:1711.01558, 2017.

[35] Saito K, Ushiku Y, Harada T. Asymmetric tri-training for unsupervised domain adaptation[J]. arXiv preprint arXiv:1702.08400, 2017.

[36] Bousmalis K, Silberman N, Dohan D, et al. Unsupervised pixel-level domain adaptation with generative adversarial networks[C]//The IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 2017, 1(2): 7.

[37] Shen J, Qu Y, Zhang W, et al. Adversarial representation learning for domain adaptation[J]. arXiv preprint arXiv:1707.01217, 2017.

[38] Kim T, Cha M, Kim H, et al. Learning to discover cross-domain relations with generative adversarial networks[J]. arXiv preprint arXiv:1703.05192, 2017.

[39] Tommasi T, Caputo B. Frustratingly Easy NBNN Domain Adaptation[C]. international conference on computer vision, 2013: 897-904.

[40] Pei Z, Cao Z, Long M, et al. Multi-Adversarial Domain Adaptation[C] // AAAI 2018.

[41] Ghifary M, Kleijn W B, Zhang M. Domain adaptive neural networks for object recognition[C]//Pacific Rim International Conference on Artificial Intelligence. Springer, Cham, 2014: 898-904.

[42] Saito K, Watanabe K, Ushiku Y, et al. Maximum Classifier Discrepancy for Unsupervised Domain Adaptation[J]. arXiv preprint arXiv:1712.02560, 2017.

[43] Volpi R, Morerio P, Savarese S, et al. Adversarial Feature Augmentation for Unsupervised Domain Adaptation[J]. arXiv preprint arXiv:1711.08561, 2017.

[44] Zhang Y, Xiang T, Hospedales T M, et al. Deep Mutual Learning[C]. CVPR 2018.

[45] French G, Mackiewicz M, Fisher M. Self-ensembling for visual domain adaptation[C]//International Conference on Learning Representations. 2018.