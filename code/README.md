# code_transfer_learning

*Some useful transfer learning and domain adaptation codes*

> It is a waste of time looking for the codes from others. So I **collect** or **reimplement** them here in a way that you can **easily** use. The following are some of the popular transfer learning (domain adaptation) methods in recent years, and I know most of them will be chosen to **compare** with your own method.

> You are welcome to contribute and suggest other methods.

This document contains codes from several aspects: **tutorial**, **theory**, **traditional** methods, and **deep** methods.

Testing **dataset** can be found [here](https://github.com/jindongwang/transferlearning/blob/master/doc/dataset.md).

- - -

## Fine-tune 最简单的深度迁移学习

- Fine-tune using **AlexNet** and **ResNet**
	- [PyTorch](https://github.com/jindongwang/transferlearning/tree/master/code/deep/finetune_AlexNet_ResNet)
	- [Another example using Pytorch](https://github.com/Spandan-Madan/Pytorch_fine_tuning_Tutorial)
	- [Pytorch CNN fine-tune](https://github.com/creafz/pytorch-cnn-finetune)

- Fast learn transfer learning:
	- [Pytorch](https://github.com/miguelgfierro/sciblog_support/blob/master/A_Gentle_Introduction_to_Transfer_Learning/Intro_Transfer_Learning.ipynb) | [Tensorflow](https://cosx.org/2017/10/transfer-learning/)

- **Google's Tensorflow Hub** (Tensorflow library released by Google for transfer learning)
	- [Tensorflow](https://github.com/tensorflow/hub)

## Deep feature extractor 提取深度网络特征用于传统方法

[Deep feature extractor](https://github.com/jindongwang/transferlearning/blob/master/code/feature_extractor/readme.md)

## Basic distance 常用的距离度量

- MMD and MK-MMD：[Python](https://github.com/jindongwang/transferlearning/blob/master/code/distance/mmd_numpy_sklearn.py) | [Pytorch](https://github.com/jindongwang/transferlearning/blob/master/code/distance/mmd_pytorch.py) | [Matlab](https://github.com/jindongwang/transferlearning/blob/master/code/distance/mmd_matlab.m)
- $A$-distance: [Python](https://github.com/jindongwang/transferlearning/tree/master/code/distance/proxy_a_distance.py)
- CORAL loss: [Pytorch](https://github.com/jindongwang/transferlearning/tree/master/code/distance/coral_loss.py)
- Several metric learning algorithms: [Python](https://github.com/metric-learn/metric-learn)
- Wasserstein distance (earch mover's distance):
	- Scipy built-in function: [scipy.stats.wasserstein_distance](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wasserstein_distance.html)
	- OpenCV built-in function: `cv.CalcEMD2`
	- Google's implementation: [Tensorflow](https://github.com/google/wasserstein-dist)

## Traditional transfer learning methods  非深度迁移

- **SVM** (baseline)
	- [Matlab](https://github.com/jindongwang/transferlearning/tree/master/code/traditional/SVM.m)
- **TCA** (Transfer Component Anaysis, TNN-11) [1]
	- [Matlab and Python](https://github.com/jindongwang/transferlearning/tree/master/code/traditional/TCA)
- **KMM** (Kernel Mean Matching, NIPS-06) [67]
    - [Python](https://github.com/jindongwang/transferlearning/tree/master/code/traditional/KMM.py)
- **GFK** (Geodesic Flow Kernel, CVPR-12) [2]
	- [Matlab and Python](https://github.com/jindongwang/transferlearning/blob/master/code/traditional/GFK)
- **DA-NBNN** (Frustratingly Easy NBNN Domain Adaptation, ICCV-13) [39]
	- [Matlab](https://github.com/enoonIT/nbnn-nbnl/tree/master/DANBNN_demo)
- **JDA** (Joint Distribution Adaptation, ICCV-13) [3]
	- [Matlab and Python](https://github.com/jindongwang/transferlearning/blob/master/code/traditional/JDA)
- **TJM** (Transfer Joint Matching, CVPR-14) [4]
	- [Matlab](https://github.com/jindongwang/transferlearning/blob/master/code/traditional/MyTJM.m)
- **CORAL** (CORrelation ALignment, AAAI-15) [5]
	- [Matlab and Python](https://github.com/jindongwang/transferlearning/blob/master/code/traditional/CORAL) | [Github](https://github.com/VisionLearningGroup/CORAL)
- **JGSA** (Joint Geometrical and Statistical Alignment, CVPR-17) [6]
	- [Matlab(official)](https://www.uow.edu.au/~jz960/codes/JGSA-r.rar) | [Matlab(easy)](https://github.com/jindongwang/transferlearning/blob/master/code/traditional/CORA/MyJGSA.m)
- **TrAdaBoost** (ICML-07)[8]
	- [Python](https://github.com/chenchiwei/tradaboost)
- **SA** (Subspace Alignment, ICCV-13) [11]
	- [Matlab(official)](http://users.cecs.anu.edu.au/~basura/DA_SA/) | [Matlab](https://github.com/jindongwang/transferlearning/tree/master/code/traditional/SA_SVM.m)
- **BDA** (Balanced Distribution Adaptation for Transfer Learning, ICDM-17) [15]
	- [Matlab(official)](https://github.com/jindongwang/transferlearning/tree/master/code/BDA)
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
- **RWA** (Random Walking, arXiv, simple but powerful) [46]
	- [Matlab](https://github.com/twanvl/rwa-da/tree/master/src)
- **MEDA** (Manifold Embedded Distribution Alignment, ACM MM-18) [47]
	- [Matlab(Official)](https://github.com/jindongwang/transferlearning/tree/master/code/traditional/MEDA)
- **EasyTL** (Practically Easy Transfer Learning, ICME-19) [63]
    - [Matlab(Official)](https://github.com/jindongwang/transferlearning/tree/master/code/traditional/EasyTL)

- **SCA** (Scatter Component Analysis, TPAMI-17) [79]
    - [Matlab](https://github.com/amber0309/SCA)


## Deep transfer learning methods  深度迁移

- **DaNN** (Domain Adaptive Neural Network, PRICAI-14) [41]
	- [PyTorch](https://github.com/jindongwang/transferlearning/tree/master/code/deep/DaNN)
- **DDC** (Deep Domain Confusion, arXiv-14)
    - [PyTorch](https://github.com/jindongwang/transferlearning/tree/master/code/deep/DDC_DeepCoral)
- **DeepCORAL** (Deep CORAL: Correlation Alignment for Deep Domain Adaptation) [33]
	- [PyTorch(recommend)](https://github.com/jindongwang/transferlearning/tree/master/code/deep/DeepCoral) | [PyTorch](https://github.com/SSARCandy/DeepCORAL) | [中文解读](https://ssarcandy.tw/2017/10/31/deep-coral/)
- **DAN/JAN** (Deep Adaptation Network/Joint Adaptation Network, ICML-15,17) [9,10]
	- [PyTorch(Official)](https://github.com/thuml/Xlearn/tree/master/pytorch) | [Caffe(Official)](https://github.com/thuml/Xlearn) | [PyTorch(DAN)(recommend)](https://github.com/jindongwang/transferlearning/tree/master/code/deep/DAN)
- **RTN** (Unsupervised Domain Adaptation with Residual Transfer Networks, NIPS-16) [12]
	- [Caffe](https://github.com/thuml/Xlearn)
- **ADDA** (Adversarial Discriminative Domain Adaptation, arXiv-17) [13]
	- [Tensorflow(Official)](https://github.com/erictzeng/adda) | [Pytorch](https://github.com/corenel/pytorch-adda) | [Pytorch(another)](https://github.com/jvanvugt/pytorch-domain-adaptation/blob/master/adda.py)
- **DANN/RevGrad** (Unsupervised Domain Adaptation by Backpropagation, ICML-15) [14]
	- [Caffe(Official)](https://github.com/ddtm/caffe/tree/grl) | [PyTorch](https://github.com/jindongwang/transferlearning/tree/master/code/deep/DANN(RevGrad)) | [Pytorch(another)](https://github.com/jvanvugt/pytorch-domain-adaptation/blob/master/revgrad.py) | [Tensorflow(third party)](https://github.com/shucunt/domain_adaptation) 
- **DANN** Domain-Adversarial Training of Neural Networks (JMLR-16)[17] 
	- [Python(official)](https://github.com/GRAAL-Research/domain_adversarial_neural_network) | [Tensorflow](https://github.com/jindongwang/tf-dann) | [PyTorch](https://github.com/CuthbertCai/pytorch_DANN)
- Associative Domain Adaptation (ICCV-17) [18]
	- [Tensorflow](https://github.com/haeusser/learning_by_association)
- Deep Hashing Network for Unsupervised Domain (CVPR-17) [20]

	- [Matlab](https://github.com/hemanthdv/da-hash)
- **CCSA** (Unified Deep Supervised Domain Adaptation and Generalization, ICCV-17) [23]
	- [Python(Keras)](https://github.com/samotiian/CCSA)
- **MRN** (Learning Multiple Tasks with Multilinear Relationship Networks, NIPS-17) [24]
	- [Pytorch](https://github.com/thuml/MTlearn)
- **AutoDIAL** (Automatic DomaIn Alignment Layers, ICCV-17) [25]
	- [Caffe](https://github.com/ducksoup/autodial)
- **DSN** (Domain Separation Networks, NIPS-16) [26]
	- [Pytorch](https://github.com/fungtion/DSN) | [Tensorflow](https://github.com/tensorflow/models/tree/master/research/domain_adaptation)
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
- **DML** (Deep Mutual Learning, CVPR-18) [44]
	- [Tensorflow](https://github.com/YingZhangDUT/Deep-Mutual-Learning)
- Self-ensembling for visual domain adaptation (ICLR 2018) [45]
	- [Pytorch](https://github.com/Britefury/self-ensemble-visual-domain-adapt)
- **PADA** (Partial Adversarial Domain Adaptation, ECCV-18) [48]
	- [Pytorch(Official)](https://github.com/thuml/PADA)
- **iCAN** (Incremental Collaborative and Adversarial Network for Unsupervised Domain Adaptation, CVPR-18) [49]
	- [Pytorch](https://github.com/mahfuj9346449/iCAN)
- **WeightedGAN** (Importance Weighted Adversarial Nets for Partial Domain Adaptation, CVPR-18) [50]
	- [Caffe](https://github.com/hellojing89/weightedGANpartialDA)
- **OpenSet** (Open Set Domain Adaptation by Backpropagation) [51]
	- [Tensorflow](https://github.com/Mid-Push/Open_set_domain_adaptation)
- **WDGRL** (Wasserstein Distance Guided Representation Learning, AAAI-18) [52]
	- [Pytorch](https://github.com/jvanvugt/pytorch-domain-adaptation/blob/master/wdgrl.py)
- **JDDA** (Joint Domain Alignment and Discriminative Feature Learning) [53]
	- [Tensorflow](https://github.com/A-bone1/JDDA)
- Multi-modal Cycle-consistent Generalized Zero-Shot Learning (ECCV-18) [54]
	- [Tensorflow](https://github.com/rfelixmg/frwgan-eccv18)
- **MSTN** (Moving Semantic Transfer Network, ICML-18) [55]
	- [Tensorflow](https://github.com/Mid-Push/Moving-Semantic-Transfer-Network) | [Pytorch](https://github.com/EasonApolo/mstn)
- **SAN** (Partial Transfer Learning With Selective Adversarial Networks, CVPR-18) [56]
	- [Caffe, Pytorch](https://github.com/thuml/SAN)
- **M-ADDA** (Metric-based Adversarial Discriminative Domain Adaptation, ICML-18 workshop) [57]
    - [Pytorch](https://github.com/IssamLaradji/M-ADDA)
- **Openset_DA** (Open Set Domain Adaptation by Backpropagation) [58]
    - [Pytorch](https://github.com/YU1ut/openset-DA)
- **DIRT-T** (A DIRT-T Approach to Unsupervised Domain Adaptation, ICLR-18) [59]
    - [Tensorflow](https://github.com/RuiShu/dirt-t)
- **CDAN** (Conditional Adversarial Domain Adaptation, NeurIPS-18) [60]
	- [Pytorch(official)](https://github.com/thuml/CDAN) | [Pytorch(third party)](https://github.com/thuml/CDAN)
- **CMD** (Central Moment Discrepancy, ICLR-17 and InfSc-19) [61], [62]
    - [Keras(Theano)](https://github.com/wzell/cmd) | [Keras(Theano, journal extension)](https://github.com/wzell/mann)
- **OPDA_BP** (Open Set Domain Adaptation by Back-propagation, ECCV-18) [64]
    - [Pytorch(Official)](https://github.com/ksaito-ut/OPDA_BP)
- **TCP** (Transfer Channel Prunning, IJCNN-19) [65]
    - [Pytorch(Official)](https://github.com/jindongwang/transferlearning/tree/master/code/deep/TCP)
- **MTAN** (Multi-Task Attention Network, CVPR-19) [66]
    - [Python](https://github.com/lorenmt/mtan)
- **L2T_ww** (Learning What and Where to Transfer, ICML-19) [68]
    - [Pytorch](https://github.com/alinlab/L2T-ww)  
- **SSDA_MME** (Semi-supervised Domain Adaptation via Minimax Entropy, ICCV-19) [71]
    - [Pytorch](https://github.com/VisionLearningGroup/SSDA_MME)

- **MRAN** (Multi-representation adaptation network for cross-domain image classification, Neural Networks 2019) [72]
    - [Pytorch](https://github.com/jindongwang/transferlearning/tree/master/code/deep/MRAN)
- **TA<sup>3</sup>N** (Temporal Attentive Alignment for Large-Scale Video Domain Adaptation, ICCV-19) [73]
    - [Pytorch](https://github.com/cmhungsteve/TA3N)
- **MDAN** (Multiple Source Domain Adaptation with Adversarial Learning, NeurIPS-18) [74]
    - [Pytorch](https://github.com/KeiraZhao/MDAN)

- Deep model transferribility from attribution maps (NeurIPS-19) [75]
    - [Tensorflow](https://github.com/DeepDarkFantasy20/TransferbilityFromAttributionMaps)

- **DIVA** (Domain Invariant Variational Autoencoders, arXiv-19) [76]
    - [Pytorch](https://github.com/AMLab-Amsterdam/DIVA)

- **CDCL** (Cross-Domain Complementary Learning with Synthetic Data for Multi-Person Part Segmentation, arXiv, ICCV-19 Demo) [77]
    - [Tensorflow](https://github.com/kevinlin311tw/CDCL-human-part-segmentation)

- **DTA** (Drop to Adapt: Learning Discriminative Features for Unsupervised Domain Adaptation, arXiv, ICCV-19) [78]
    - [PyTorch](https://github.com/postBG/DTA.pytorch)

- **DAAN** (Dynamic Adversarial Adaptation Network, ICDM 2019) [80]
    - [Pytorch](https://github.com/jindongwang/transferlearning/tree/master/code/deep/DAAN)

## Applications

- Learning to select data for transfer learning with Bayesian Optimization (EMNLP-17) [69]
	- [Python](https://github.com/sebastianruder/learn-to-select-data)

- **SDG4DA** (Reinforced Training Data Selection for Domain Adaptation, ACL-19) [70]
    - [Tensorflow](https://github.com/timerstime/SDG4DA)

- - -

#### [Code from HKUST](http://www.cse.ust.hk/TL/) [a bit old]

- - -

#### References

[1] Pan S J, Tsang I W, Kwok J T, et al. Domain adaptation via transfer component analysis[J]TNN, 2011, 22(2): 199-210.

[2] Gong B, Shi Y, Sha F, et al. Geodesic flow kernel for unsupervised domain adaptation[C]//CVPR, 2012: 2066-2073.

[3] Long M, Wang J, Ding G, et al. Transfer feature learning with joint distribution adaptation[C]//ICCV. 2013: 2200-2207.

[4] Long M, Wang J, Ding G, et al. Transfer joint matching for unsupervised domain adaptation[C]//CVPR. 2014: 1410-1417.

[5] Sun B, Feng J, Saenko K. Return of Frustratingly Easy Domain Adaptation[C]//AAAI. 2016, 6(7): 8.

[6] Zhang J, Li W, Ogunbona P. Joint Geometrical and Statistical Alignment for Visual Domain Adaptation[C]//CVPR 2017.

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

[46] van Laarhoven T, Marchiori E. Unsupervised Domain Adaptation with Random Walks on Target Labelings[J]. arXiv preprint arXiv:1706.05335, 2017.

[47] Jindong Wang, Wenjie Feng, Yiqiang Chen, Han Yu, Meiyu Huang, Philip S. Yu. Visual Domain Adaptation with Manifold Embedded Distribution Alignment. ACM Multimedia conference 2018.

[48] Zhangjie Cao, Mingsheng Long, et al. Partial Adversarial Domain Adaptation. ECCV 2018.

[49] Zhang W, Ouyang W, Li W, et al. Collaborative and Adversarial Network for Unsupervised domain adaptation[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018: 3801-3809.

[50] Zhang J, Ding Z, Li W, et al. Importance Weighted Adversarial Nets for Partial Domain Adaptation[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018: 8156-8164.

[51] Saito K, Yamamoto S, Ushiku Y, et al. Open Set Domain Adaptation by Backpropagation[J]. arXiv preprint arXiv:1804.10427, 2018.

[52] Shen J, Qu Y, Zhang W, et al. Wasserstein Distance Guided Representation Learning for Domain Adaptation[C]//AAAI. 2018.

[53] Chen C, Chen Z, Jiang B, et al. Joint Domain Alignment and Discriminative Feature Learning for Unsupervised Deep Domain Adaptation[J]. arXiv preprint arXiv:1808.09347, 2018.

[54] Felix R, Vijay Kumar B G, Reid I, et al. Multi-modal Cycle-consistent Generalized Zero-Shot Learning. ECCV 2018.

[55] Xie S, Zheng Z, Chen L, et al. Learning Semantic Representations for Unsupervised Domain Adaptation[C]//International Conference on Machine Learning. 2018: 5419-5428.

[56] Cao Z, Long M, Wang J, et al. Partial transfer learning with selective adversarial networks. CVPR 2018.

[57] Issam Laradji, Reza Babanezhad. M-ADDA: Unsupervised Domain Adaptation with Deep Metric Learning. ICML 2018 workshop.

[58] Saito K, Yamamoto S, Ushiku Y, et al. Open Set Domain Adaptation by Backpropagation[J]. arXiv preprint arXiv:1804.10427, 2018.

[59] Shu R, Bui H H, Narui H, et al. A DIRT-T Approach to Unsupervised Domain Adaptation[J]. arXiv preprint arXiv:1802.08735, 2018.

[60] Mingsheng Long, et al. Conditional Adversarial Domain Adaptation. NeurIPS 2018.

[61] W.Zellinger, T. Grubinger, E. Lughofer, T. Natschlaeger, and Susanne Saminger-Platz, "Central moment discrepancy (cmd) for domain-invariant representation learning," ICLR 2017.

[62] W. Zellinger, B.A. Moser, T. Grubinger, E. Lughofer, T. Natschlaeger, and S. Saminger-Platz, "Robust unsupervised domain adaptation for neural networks via moment alignment," Information Sciences (in press), 2019, https://doi.org/10.1016/j.ins.2019.01.025, arXiv preprint arxiv:1711.06114

[63] Jindong Wang, Yiqiang Chen, Han Yu, Meiyu Huang, Qiang Yang. Easy Transfer Learning By Exploiting Intra-domain Structures. IEEE International Conference on Multimedia & Expo (ICME) 2019.

[64] Saito K, Yamamoto S, Ushiku Y, et al. Open set domain adaptation by backpropagation[C]//Proceedings of the European Conference on Computer Vision (ECCV). 2018: 153-168.

[65] Chaohui Yu, Jindong Wang, Yiqiang Chen, Zijing Wu. Accelerating Deep Unsupervised Domain Adaptation with Transfer Channel Pruning. IJCNN 2019.

[66] Shikun Liu, Edward Johns, and Andrew Davison. End-to-End Multi-Task Learning with Attention. CVPR 2019.

[67] Huang J, Gretton A, Borgwardt K, et al. Correcting sample selection bias by unlabeled data[C]//Advances in neural information processing systems. 2007: 601-608.

[68] Yunhun Jang, Hankook Lee, Sung Ju Hwang, Jinwoo Shin. Learning what and where to transfer. ICML 2019.

[69] Sebastian Ruder, Barbara Plank (2017). Learning to select data for transfer learning with Bayesian Optimization. In Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing, Copenhagen, Denmark.

[70] Liu M, Song Y, Zou H, et al. Reinforced Training Data Selection for Domain Adaptation[C]//Proceedings of the 57th Conference of the Association for Computational Linguistics. 2019: 1957-1968.

[71] Saito K, Kim D, Sclaroff S, et al. Semi-supervised Domain Adaptation via Minimax Entropy. ICCV 2019.

[72] Zhu Y, Zhuang F, Wang J, et al. Multi-representation adaptation network for cross-domain image classification[J]. Neural Networks, 2019.

[73] Min-Hung Chen, Zsolt Kira, Ghassan AlRegib, et al. Temporal Attentive Alignment for Large-Scale Video Domain Adaptation. ICCV 2019.

[74] Zhao H, Zhang S, Wu G, et al. Multiple source domain adaptation with adversarial learning. NeurIPS 2018.

[75] Jie Song, et al. Deep model transferrability from attirbution maps. NeurIPS 2019.

[76] Ilse, M., Tomczak, J. M., C. Louizos & Welling, M. (2018). DIVA: Domain Invariant Variational Autoencoders. arXiv preprint arXiv:1905.10427

[77] Lin K., et al. Cross-Domain Complementary Learning with Synthetic Data for Multi-Person Part Segmentation[J]. arXiv preprint arXiv:1907.05193, ICCV demo, 2019.

[78] Lee S., Kim D., et al. Drop to Adapt: Learning Discriminative Features for Unsupervised Domain Adaptation. ICCV 2019.

[79] Ghifary M, Balduzzi D, Kleijn W B, et al. Scatter component analysis: A unified framework for domain adaptation and domain generalization[J]. IEEE transactions on pattern analysis and machine intelligence, 2016, 39(7): 1414-1430.

[80] Chaohui Yu, Jindong Wang, Yiqiang Chen, Meihu Huang. Transfer learnign with dynamic adversarial adaptation network. ICDM 2019.
