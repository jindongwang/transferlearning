# code_transfer_learning

*Some useful transfer learning and domain adaptation codes*

> It is a waste of time looking for the codes from others. So I **collect** or **reimplement** them here in a way that you can **easily** use. The following are some of the popular transfer learning (domain adaptation) methods in recent years, and I know most of them will be chosen to **compare** with your own method.

> It is still **on the go**. You are welcome to contribute and suggest other methods.

- - -

## Availiable codes for:

#### Non-deep learning

- **TCA** (Transfer Component Anaysis, TNN-11) [1]
	- [Matlab(Recommended!)](https://github.com/jindongwang/transferlearning/blob/master/code/MyTCA.m) | [Python](https://github.com/jindongwang/transferlearning/tree/master/code/TCA_python)
- **GFK** (Geodesic Flow Kernel, CVPR-12) [2]
	- [Matlab](https://github.com/jindongwang/transferlearning/blob/master/code/MyGFK.m)
- **JDA** (Joint Distribution Adaptation, ICCV-13) [3]
	- [Matlab](https://github.com/jindongwang/transferlearning/blob/master/code/MyJDA.m)
- **TJM** (Transfer Joint Matching, CVPR-14) [4]
	- [Matlab](https://github.com/jindongwang/transferlearning/blob/master/code/MyTJM.m)
- **CORAL** (CORrelation ALignment, AAAI-15) [5]
	- [Matlab](https://github.com/jindongwang/transferlearning/blob/master/code/MyCORAL.m) | [Github](https://github.com/VisionLearningGroup/CORAL)
- **JGSA** (Joint Geometrical and Statistical Alignment, CVPR-17) [6]
	- [Matlab](https://github.com/jindongwang/transferlearning/blob/master/code/MyJGSA.m)
- **ARTL** (Adaptation Regularization, TKDE-14) [7]
	- [Matlab](https://github.com/jindongwang/transferlearning/tree/master/code/MyARTL)
- **TrAdaBoost** (ICML-07)[8]
	- [Python](https://github.com/chenchiwei/tradaboost)
- **SA** (Subspace Alignment, ICCV-13) [11]
	- [Matlab](http://users.cecs.anu.edu.au/~basura/DA_SA/)
- **BDA** (Balanced Distribution Adaptation for Transfer Learning, ICDM-17) [15]
	- [Matlab](https://github.com/jindongwang/transferlearning/tree/master/code/BDA)
- **MTLF** (Metric Transfer Learning, TKDE-17) [16]
	- [Matlab](https://github.com/xyh2016/MTLF)
- **Open Set Domain Adaptation** (ICCV-17) [19]
	- [Matlab(official, but not available now)](https://github.com/Heliot7/open-set-da) | [Matlab(My implementation)](https://github.com/jindongwang/transferlearning/tree/master/code/open_set_da)
- **TAISL** (When Unsupervised Domain Adaptation Meets Tensor Representations, ICCV-17) [21]
	- [Matlab(official)](https://github.com/poppinace/TAISL)
- **STL** (Stratified Transfer Learning for Cross-domain Activity Recognition, PerCom-18) [22]
	- [Matlab](https://github.com/jindongwang/activityrecognition/tree/master/code/percom18_stl)
- **LSA** (Landmarks-based kernelized subspace alignment for unsupervised domain adaptation, CVPR-15) [29]
	- [Matlab](http://homes.esat.kuleuven.be/~raljundi/papers/LSA%20Clean%20Code.zip)


#### Deep learning

- **DAN/JAN** (Deep Adaptation Network/Joint Adaptation Network, ICML-15,17) [9,10]
	- [Caffe](https://github.com/thuml/Xlearn)
- **RTN** (Unsupervised Domain Adaptation with Residual Transfer Networks, NIPS-16) [12]
	- [Caffe](https://github.com/thuml/Xlearn)
- **ADDA** (Adversarial Discriminative Domain Adaptation, arXiv-17) [13]
	- [Python(Tensorflow)](https://github.com/erictzeng/adda)
- Unsupervised Domain Adaptation by Backpropagation (ICML-15) [14]
	- [Caffe(from author)](https://github.com/ddtm/caffe/tree/grl)|[Tensorflow(third party)](https://github.com/shucunt/domain_adaptation)
- Domain-Adversarial Training of Neural Networks (JMLR-16)[17] 
	- [Tensorflow](https://github.com/jindongwang/tf-dann)
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
	- [Keras](https://github.com/ghif/drcn)
- Multi-task Autoencoders for Domain Generalization (ICCV-15) [28]
	- [Keras](https://github.com/ghif/mtae)
- Encoder based lifelong learning (ICCV-17) [30]
	- [Matlab](https://github.com/rahafaljundi/Encoder-Based-Lifelong-learning)


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