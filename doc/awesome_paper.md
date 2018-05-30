# Awesome transfer learning papers

Let's read some awesome transfer learning / domain adaptation papers!

这里收录了迁移学习各个研究领域的最新文章。

- - -

## 目录

- [arXiv专区](#arXiv专区)

- [普通迁移学习 General Transfer Learning](#普通迁移学习) 

- [领域自适应(非深度) Domain Adaptation (Shallow)](#领域自适应)

- [在线迁移学习 Online transfer learning](#在线迁移学习)

- [终身迁移学习 Lifelong transfer learning](#终身迁移学习)

- [异构迁移学习 Heterogeneous Transfer Learning](#异构迁移学习)

- [深度迁移学习 Deep Transfer Learning](#深度迁移学习)
    
    - [深度对抗迁移迁移学习 Deep Adversarial transfer learning](#对抗迁移学习)

- [传递迁移学习 Transitive Transfer Learning](#传递迁移学习)

- [迁移学习用于强化学习 Transfer Learning with Reinforcement Learning](#强化迁移学习)

- [应用 Applications](#应用)

- - -

### 普通迁移学习

- 20180524 arXiv 探索了Multi-source迁移学习的一些理论：[Algorithms and Theory for Multiple-Source Adaptation](https://arxiv.org/abs/1805.08727)

- 20180524 arXiv 杨强团队、与之前的learning to learning类似，这里提供了一个从经验中学习的learning to multitask框架：[Learning to Multitask](https://arxiv.org/abs/1805.07541)

- 20180403 arXiv 选择最优的子类生成方便迁移的特征：[Class Subset Selection for Transfer Learning using Submodularity](https://arxiv.org/abs/1804.00060)

- 20180326 ICMLA-17 在类别不平衡情况下比较了一些迁移学习和传统方法的性能，并做出一些结论：[Comparing Transfer Learning and Traditional Learning Under Domain Class Imbalance](http://ieeexplore.ieee.org/abstract/document/8260654/)

- - - 

### 领域自适应

- 20180510 IEEE Trans. Cybernetics 提出一个通用的迁移学习框架，对不同的domain进行不同的特征变换：[Transfer Independently Together: A Generalized Framework for Domain Adaptation](https://ieeexplore-ieee-org.lib.ezproxy.ust.hk/abstract/document/8337102/)

- 20180403 TIP-18 一篇传统方法做domain adaptation的文章，比很多深度方法结果都好：[An Embarrassingly Simple Approach to Visual Domain Adaptation](http://ieeexplore.ieee.org/abstract/document/8325317/)

- 20180326 ICMLA-17 利用subsapce alignment进行迁移学习：[Transfer Learning for Large Scale Data Using Subspace Alignment](http://ieeexplore.ieee.org/abstract/document/8260772)

- 20180321 CVPR-18 构建了一个迁移学习算法，用于解决跨数据集之间的person-reidenfication: [Unsupervised Cross-dataset Person Re-identification by Transfer Learning of Spatial-Temporal Patterns](https://arxiv.org/abs/1803.07293)

- 20180316 arXiv 用optimal transport解决domain adaptation中类别不平衡的问题：[Optimal Transport for Multi-source Domain Adaptation under Target Shift](https://arxiv.org/abs/1803.04899)

- 20180315 ICLR-17 一篇综合进行two-sample stest的文章：[Revisiting Classifier Two-Sample Tests](https://arxiv.org/abs/1610.06545)

- 20180228 arXiv 一篇通过标签一致性和MMD准则进行domain adaptation的文章: [Discriminative Label Consistent Domain Adaptation](https://arxiv.org/abs/1802.08077)

- 20180226 AAAI-18 清华龙明盛组最新工作：[Unsupervised Domain Adaptation with Distribution Matching Machines](http://ise.thss.tsinghua.edu.cn/~mlong/doc/distribution-matching-machines-aaai18.pdf)

- 20180110 arXiv 一篇比较新的传统方法做domain adaptation的文章 [Close Yet Discriminative Domain Adaptation](https://arxiv.org/abs/1704.04235)

- 20180105 arXiv 最优的贝叶斯迁移学习 [Optimal Bayesian Transfer Learning](https://arxiv.org/abs/1801.00857)

- 20171216 arXiv [Zero-Shot Deep Domain Adaptation](https://arxiv.org/abs/1707.01922)
    - 当target domain的数据不可用时，如何用相关domain的数据进行辅助学习？

- 20171214 arXiv [Investigating the Impact of Data Volume and Domain Similarity on Transfer Learning Applications](https://arxiv.org/abs/1712.04008)
    - 在实验中探索了数据量多少，和相似度这两个因素对迁移学习效果的影响

- 20171210 AAAI-18 [Learning to Generalize: Meta-Learning for Domain Generalization](https://arxiv.org/pdf/1710.03463.pdf)
    - 将Meta-Learning与domain generalization结合的文章，可以联系到近期较为流行的few-shot learning进行下一步思考。

- 20171201 ICCV-17 [When Unsupervised Domain Adaptation Meets Tensor Representations](http://openaccess.thecvf.com/content_iccv_2017/html/Lu_When_Unsupervised_Domain_ICCV_2017_paper.html)
    - 第一篇将Tensor与domain adaptation结合的文章。[代码](https://github.com/poppinace/TAISL)
    - [我的解读](https://zhuanlan.zhihu.com/p/31834244)

- 201711 ICLR-18 [GENERALIZING ACROSS DOMAINS VIA CROSS-GRADIENT TRAINING](https://openreview.net/pdf?id=r1Dx7fbCW)
    - 由于双盲审的缘故，目前处于匿名状态。不同于以往的工作，本文运用贝叶斯网络建模label和domain的依赖关系，抓住training、inference 两个过程，有效引入domain perturbation来实现domain adaptation。

- 201711 ICCV-17 [Open set domain adaptation](http://openaccess.thecvf.com/content_iccv_2017/html/Busto_Open_Set_Domain_ICCV_2017_paper.html)。
    - 当source和target只共享某一些类别时，怎么处理？这个文章获得了ICCV 2017的Marr Prize Honorable Mention，值得好好研究。
    - [我的解读](https://zhuanlan.zhihu.com/p/31230331)

- 201710 [Domain Adaptation in Computer Vision Applications](https://books.google.com.hk/books?id=7181DwAAQBAJ&pg=PA95&lpg=PA95&dq=Learning+Domain+Invariant+Embeddings+by+Matching%E2%80%A6&source=bl&ots=fSc1yvZxU3&sig=XxmGZkrfbJ2zSsJcsHhdfRpjaqk&hl=zh-CN&sa=X&ved=0ahUKEwjzvODqkI3XAhUCE5QKHYStBywQ6AEIRDAE#v=onepage&q=Learning%20Domain%20Invariant%20Embeddings%20by%20Matching%E2%80%A6&f=false) 里面收录了若干篇domain adaptation的文章，可以大概看看。

- [学习迁移](https://arxiv.org/abs/1708.05629)(Learning to Transfer, L2T)
	- 迁移学习领域的新方向：与在线、增量学习结合
	- [我的解读](https://zhuanlan.zhihu.com/p/28888554)
    
- 201707 [Mutual Alignment Transfer Learning](https://arxiv.org/abs/1707.07907)

- 201708 [Learning Invariant Riemannian Geometric Representations Using Deep Nets](https://arxiv.org/abs/1708.09485)

- 20170812 香港科技大学的最新文章：[Learning To Transfer](https://arxiv.org/abs/1708.05629)，将迁移学习和增量学习的思想结合起来，为迁移学习的发展开辟了一个崭新的研究方向。[我的解读](https://zhuanlan.zhihu.com/p/28888554)

- AAAI-18 [Learning to Generalize: Meta-Learning for Domain Generalization](https://arxiv.org/pdf/1710.03463.pdf)

- ICLR-18 [generalizing across domains via cross-gradient training](https://openreview.net/pdf?id=r1Dx7fbCW)

- NIPS-17 [Learning Multiple Tasks with Multilinear Relationship Networks](http://papers.nips.cc/paper/6757-learning-multiple-tasks-with-deep-relationship-networks) 

- NIPS-17 [JDOT: Joint distribution optimal transportation for domain adaptation](https://arxiv.org/pdf/1705.08848.pdf)

- ICCV-17 [Open set domain adaptation](http://openaccess.thecvf.com/content_iccv_2017/html/Busto_Open_Set_Domain_ICCV_2017_paper.html)

- AAAI-16 [Return of Frustratingly Easy Domain Adaptation](https://arxiv.org/abs/1511.05547)

- JMLR-16 [Distribution-Matching Embedding for Visual Domain Adaptation](http://www.jmlr.org/papers/volume17/15-207/15-207.pdf)

- CoRR abs/1610.04420 (2016) [Theoretical Analysis of Domain Adaptation with Optimal Transport](https://arxiv.org/pdf/1610.04420.pdf)

- CVPR-14 [Transfer Joint Matching for Unsupervised Domain Adaptation](http://ise.thss.tsinghua.edu.cn/~mlong/doc/transfer-joint-matching-cvpr14.pdf)

- ICCV-13 [Transfer Feature Learning with Joint Distribution Adaptation](http://ise.thss.tsinghua.edu.cn/~mlong/doc/joint-distribution-adaptation-iccv13.pdf)

- 迁移成分分析方法(Transfer component analysis, TCA)
	- [Domain adaptation via tranfer component analysis](https://mega.nz/#!JTwElLrL!j5-TanhHCMESsGBNvY6I_hX6uspsrTxyopw8bPQ2azU)
	- 发表在IEEE Trans. Neural Network期刊上（现改名为IEEE trans. Neural Network and Learning System），前作会议文章发在AAAI-09上
	- [我的解读](https://zhuanlan.zhihu.com/p/26764147?group_id=844611188275965952)

- 联合分布适配方法（joint distribution adaptation，JDA）
	- [Transfer Feature Learning with Joint Distribution Adaptation](http://ise.thss.tsinghua.edu.cn/~mlong/doc/joint-distribution-adaptation-iccv13.pdf)
	- 发表在2013年的ICCV上
	- [我的解读](https://zhuanlan.zhihu.com/p/27336930)

- 测地线流式核方法(Geodesic flow kernel, GFK)
	- [Geodesic flow kernel for unsupervised domain adaptation](https://mega.nz/#!tDY1lCSD!flMSgl-0uIswpSFL3sdZgKi6fOyFVLtcO8P6SE0OUPU)
	- 发表在CVPR-12上
	- [我的解读](https://zhuanlan.zhihu.com/p/27782708)

- 领域不变性迁移核学习(Transfer Kernel Learning, TKL)
	- [Domain invariant transfer kernel learning](https://mega.nz/#!tOoCCRhB!YyoorOUcp6XIPPd6A0s7qglYnaSiRJFEQBphtZ2c58Q)
	- 发表在IEEE Trans. Knowledge and Data Engineering期刊上
	- 
- 深度适配网络（Deep Adaptation Network, DAN）
	- 发表在ICML-15上：learning transferable features with deep adaptation networks
	- [我的解读](https://zhuanlan.zhihu.com/p/27657910)

- - -

### 在线迁移学习

- 20180326 考虑主动获取label的budget情况下的在线迁移学习：[Online domain adaptation by exploiting labeled features and pro-active learning](https://dl.acm.org/citation.cfm?id=3152507)

- 20180128 **第一篇**在线迁移学习的文章，发表在ICML-10上，系统性地定义了在线迁移学习的任务，给出了进行在线同构和异构迁移学习的两种学习模式。[Online Transfer Learning](https://dl.acm.org/citation.cfm?id=3104478) 
    - 扩充的期刊文章发在2014年的AIJ上：[Online Transfer Learning](https://www.sciencedirect.com/science/article/pii/S0004370214000800)
    - [我的解读](https://zhuanlan.zhihu.com/p/33557802?group_id=943152232741535744)
	- 文章代码：[OTL](http://stevenhoi.org/otl)

- 20180126 两篇在线迁移学习：
	- [Online transfer learning by leveraging multiple source domains](https://link.springer.com/article/10.1007/s10115-016-1021-1)
	- [Online Heterogeneous Transfer by Hedge Ensemble of Offline and Online Decisions](http://ieeexplore.ieee.org/document/8064213/)

- 20180126 TKDE-17 同时有多个同构和异构源域时的在线迁移学习：[Online Transfer Learning with Multiple Homogeneous or Heterogeneous Sources](http://ieeexplore.ieee.org/abstract/document/7883886/)

- KIS-17 Online transfer learning by leveraging multiple source domains 提出一种综合衡量多个源域进行在线迁移学习的方法。文章的related work是很不错的survey。

- CIKM-13 OMS-TL: A Framework of Online Multiple Source Transfer Learning 第一次在mulitple source上做online transfer，也是用的分类器集成。

- ICLR-17 ONLINE BAYESIAN TRANSFER LEARNING FOR SEQUENTIAL DATA MODELING 用贝叶斯的方法学习在线的HMM迁移学习模型，并应用于行为识别、睡眠监测，以及未来流量分析。

- KDD-14 Scalable Hands-Free Transfer Learning for Online Advertising 提出一种无参数的SGD方法，预测广告量

- TNNLS-17 Online Feature Transformation Learning for Cross-Domain Object Category Recognition 在线feature transformation方法

- ICPR-12 Online Transfer Boosting for Object Tracking 在线transfer 样本

- TKDE-14 Online Feature Selection and Its Applications 在线特征选择

- AAAI-15 Online Transfer Learning in Reinforcement Learning Domains 应用于强化学习的在线迁移学习

- AAAI-15 Online Boosting Algorithms for Anytime Transfer and Multitask Learning 一种通用的在线迁移学习方法，可以适配在现有方法的后面

- IJSR-13 Knowledge Transfer Using Cost Sensitive Online Learning Classification 探索在线迁移方法，用样本cost

- Multi-Source Iterative Adaptation for Cross-Domain Classification 多源迁移学习方法

- - -

### 终身迁移学习

- 20180323 arXiv 终身迁移学习与增量学习结合：[Incremental Learning-to-Learn with Statistical Guarantees](https://arxiv.org/abs/1803.08089)

- 20180111 arXiv 一种新的终身学习框架，与L2T的思路有一些类似 [Lifelong Learning for Sentiment Classification](https://arxiv.org/abs/1801.02808)


- - -

### 异构迁移学习

- 20180403 Neural Processing Letters-18 异构迁移学习：[Label Space Embedding of Manifold Alignment for Domain Adaption](https://link.springer.com/article/10.1007/s11063-018-9822-8)

- 20180105 arXiv 异构迁移学习 [Heterogeneous transfer learning](https://arxiv.org/abs/1701.02511)


- - -

### 深度迁移学习

- 20180530 arXiv 用于深度网络的鲁棒性domain adaptation方法：[Robust Unsupervised Domain Adaptation for Neural Networks via Moment Alignment](https://arxiv.org/abs/1711.06114)

- 20180522 arXiv 用CNN进行跨领域的属性学习：[Cross-domain attribute representation based on convolutional neural network](https://arxiv.org/abs/1805.07295)

- 20180428 CVPR-18 相互协同学习：[Deep Mutual Learning](https://github.com/YingZhangDUT/Deep-Mutual-Learning)

- 20180428 ICLR-18 自集成学习用于domain adaptation：[Self-ensembling for visual domain adaptation](https://github.com/Britefury/self-ensemble-visual-domain-adapt)

- 20180428 IJCAI-18 将knowledge distilation用于transfer learning，然后进行视频分类：[Better and Faster: Knowledge Transfer from Multiple Self-supervised Learning Tasks via Graph Distillation for Video Classification](https://arxiv.org/abs/1804.10069)

- 20180426 arXiv 深度学习中的参数如何进行迁移？（杨强团队）：[Parameter Transfer Unit for Deep Neural Networks](https://arxiv.org/abs/1804.08613)

- 20180425 CVPR-18(oral) 对不同的视觉任务进行建模，从而可以进行深层次的transfer：[Taskonomy: Disentangling Task Transfer Learning](https://arxiv.org/abs/1804.08328)

- 20180425 arXiv 探索各个层对于迁移任务的作用，方便以后的迁移。比较有意思：[CactusNets: Layer Applicability as a Metric for Transfer Learning](https://arxiv.org/abs/1804.07846)

- 20180410 ICLR-17 第一篇用可变RNN进行多维时间序列迁移的文章：[Variational Recurrent Adversarial Deep Domain Adaptation](https://openreview.net/forum?id=rk9eAFcxg&noteId=SJN7BGyPl)

- 20180403 arXiv 本地和云端CNN迁移融合的图片分类：[Hierarchical Transfer Convolutional Neural Networks for Image Classification](https://arxiv.org/abs/1804.00021)

- 20180402 CVPR-18 渐进式domain adaptation：[Cross-Domain Weakly-Supervised Object Detection through Progressive Domain Adaptation](https://arxiv.org/abs/1803.11365)

- 20180329 arXiv 基于attention机制的多任务学习：[End-to-End Multi-Task Learning with Attention](https://arxiv.org/abs/1803.10704)

- 20180326 arXiv 将迁移学习用于Faster R-CNN对象识别中：[Domain Adaptive Faster R-CNN for Object Detection in the Wild](https://arxiv.org/abs/1803.03243)

- 20180326 Pattern Recognition-17 多标签迁移学习方法应用于脸部属性分类：[Multi-label Learning Based Deep Transfer Neural Network for Facial Attribute Classification](https://www.sciencedirect.com/science/article/pii/S0031320318301080)

- 20180326 类似于ResNet的思想，在传统layer的ReLU之前加一个additive layer进行domain adaptation，思想简洁，效果非常好：[Layer-wise domain correction for unsupervised domain adaptation](https://link.springer.com/article/10.1631/FITEE.1700774)

- 20180326 Pattern Recognition-17 基于Batch normalization提出了AdaBN，很简单：[Adaptive Batch Normalization for practical domain adaptation](http://ieeexplore.ieee.org/abstract/document/8168121/)

- 20180309 arXiv 利用已有网络的先验知识来加速目标网络的训练：[Transfer Automatic Machine Learning](https://arxiv.org/abs/1803.02780)

- 2018 ICLR-18 最小熵领域对齐方法 [Minimal-Entropy Correlation Alignment for Unsupervised Deep Domain Adaptation](https://openreview.net/forum?id=rJWechg0Z) [code](https://github.com/pmorerio/minimal-entropy-correlation-alignment/tree/master/svhn2mnist)

- 2018 arXiv 最新发表在arXiv上的深度domain adaptation的综述：[Deep Visual Domain Adaptation: A Survey](https://arxiv.org/abs/1802.03601)

- ICCV-17 [AutoDIAL: Automatic DomaIn Alignment Layers](https://arxiv.org/pdf/1704.08082.pdf)

- ICCV-17 [CCSA: Unified Deep Supervised Domain Adaptation and Generalization](http://vision.csee.wvu.edu/~motiian/papers/CCSA.pdf)

- ICML-17 [JAN: Deep Transfer Learning with Joint Adaptation Networks](http://ise.thss.tsinghua.edu.cn/~mlong/doc/joint-adaptation-networks-icml17.pdf)

- 2017 Google: [Learning Transferable Architectures for Scalable Image Recognition](https://arxiv.org/abs/1707.07012)

- NIPS-16 [RTN: Unsupervised Domain Adaptation with Residual Transfer Networks](http://ise.thss.tsinghua.edu.cn/~mlong/doc/residual-transfer-network-nips16.pdf)

- CoRR abs/1603.04779 (2016) [AdaBN: Revisiting batch normalization for practical domain adaptation](https://arxiv.org/pdf/1603.04779.pdf)

- JMLR-16 [DANN: Domain-adversarial training of neural networks](http://www.jmlr.org/papers/volume17/15-239/15-239.pdf)

- 20171226 NIPS 2016 把传统工作搬到深度网络中的范例：不是只学习domain之间的共同feature，还学习每个domain specific的feature。这篇文章写得非常清楚，通俗易懂！ [Domain Separation Networks](http://papers.nips.cc/paper/6254-domain-separation-networks) | [代码](https://github.com/tensorflow/models/tree/master/research/domain_adaptation)

- 20171222 NIPS 2017 用adversarial网络，当target中有很少量的label时如何进行domain adaptation：[Few-Shot Adversarial Domain Adaptation](http://papers.nips.cc/paper/7244-few-shot-adversarial-domain-adaptation)

- 20171222 ICCV 2017 对于target中只有很少量的标记数据，用深度网络结合孪生网络的思想进行泛化：[Unified Deep Supervised Domain Adaptation and Generalization](http://openaccess.thecvf.com/content_ICCV_2017/papers/Motiian_Unified_Deep_Supervised_ICCV_2017_paper.pdf) | [代码和数据](https://github.com/samotiian/CCSA)

- 20171126 NIPS-17 [Label Efficient Learning of Transferable Representations acrosss Domains and Tasks](http://papers.nips.cc/paper/6621-label-efficient-learning-of-transferable-representations-acrosss-domains-and-tasks)    
    - 李飞飞小组发在NIPS 2017的文章。针对不同的domain、不同的feature、不同的label space，统一学习一个深度网络进行表征。

- 201711 一个很好的深度学习+迁移学习的实践教程，有代码有数据，可以直接上手：[基于深度学习和迁移学习的识花实践](https://cosx.org/2017/10/transfer-learning/)

- ECCV-16 [Deep CORAL: Correlation Alignment for Deep Domain Adaptation](https://arxiv.org/abs/1607.01719.pdf)

- ECCV-16 [DRCN: Deep Reconstruction-Classification Networks for Unsupervised Domain Adaptation](https://arxiv.org/abs/1607.03516.pdf)

- ICML-15 [DAN: Learning Transferable Features with Deep Adaptation Networks](http://ise.thss.tsinghua.edu.cn/~mlong/doc/deep-adaptation-networks-icml15.pdf)

- ICML-15 [GRL: Unsupervised Domain Adaptation by Backpropagation](http://sites.skoltech.ru/compvision/projects/grl/files/paper.pdf)

- ICCV-15 [Simultaneous Deep Transfer Across Domains and Tasks](https://people.eecs.berkeley.edu/~jhoffman/papers/Tzeng_ICCV2015.pdf)

- NIPS-14 [How transferable are features in deep neural networks?](http://yosinski.com/media/papers/Yosinski__2014__NIPS__How_Transferable_with_Supp.pdf)

- CoRR abs/1412.3474 (2014) [Deep Domain Confusion(DDC): Maximizing for Domain Invariance](http://www.arxiv.org/pdf/1412.3474.pdf)

- ICML-14 著名的DeCAF特征：[DeCAF: A Deep Convolutional Activation Feature for Generic Visual Recognition](https://arxiv.org/abs/1310.1531.pdf)

- [Simultaneous Deep Transfer Across Domains and Tasks](https://www.cv-foundation.org/openaccess/content_iccv_2015/html/Tzeng_Simultaneous_Deep_Transfer_ICCV_2015_paper.html)
	- 发表在ICCV-15上，在传统深度迁移方法上又加了新东西
	- [我的解读](https://zhuanlan.zhihu.com/p/30621691)

- 深度适配网络（Deep Adaptation Network, DAN）
	- 发表在ICML-15上：learning transferable features with deep adaptation networks
	- [我的解读](https://zhuanlan.zhihu.com/p/27657910)

- [深度联合适配网络](http://proceedings.mlr.press/v70/long17a.html)（Joint Adaptation Network, JAN）
	- Deep Transfer Learning with Joint Adaptation Networks
	- 发表在ICML 2017上，作者也是龙明盛
	- 延续了之前的DAN工作，这次考虑联合适配
- - -

#### 对抗迁移学习

- 20180508 arXiv 利用GAN，从有限数据中生成另一个domain的数据：[Transferring GANs: generating images from limited data](https://arxiv.org/abs/1805.01677)

- 20180501 arXiv Open set domain adaptation的对抗网络版本：[Open Set Domain Adaptation by Backpropagation](https://arxiv.org/abs/1804.10427)

- 20180427 arXiv 提出了adversarial residual的概念，进行深度对抗迁移：[Unsupervised Domain Adaptation with Adversarial Residual Transform Networks](https://arxiv.org/abs/1804.09578)

- 20180424 CVPR-18 用GAN和迁移学习进行图像增强：[Adversarial Feature Augmentation for Unsupervised Domain Adaptation](https://arxiv.org/pdf/1711.08561.pdf)

- 20180413 arXiv 一种思想非常简单的深度迁移方法，仅考虑进行domain之间的类别概率矫正就能取得非常好的效果：[Simple Domain Adaptation with Class Prediction Uncertainty Alignment](https://arxiv.org/abs/1804.04448)

- 20180413 arXiv Mingming Gong提出的用因果生成网络进行深度迁移：[Causal Generative Domain Adaptation Networks](https://arxiv.org/abs/1804.04333)

- 20180410 CVPR-18(oral) 用两个分类器进行对抗迁移：[Maximum Classifier Discrepancy for Unsupervised Domain Adaptation](https://arxiv.org/abs/1712.02560) [代码](https://github.com/mil-tokyo/MCD_DA)

- 20180403 CVPR-18 将样本权重应用于对抗partial transfer中：[Importance Weighted Adversarial Nets for Partial Domain Adaptation](https://arxiv.org/abs/1803.09210)

- 20180326 MLSP-17 把domain separation network和对抗结合起来，提出了一个对抗网络进行迁移学习：[Adversarial domain separation and adaptation](http://ieeexplore.ieee.org/abstract/document/8168121/)

- 20180326 ICIP-17 类似于domain separation network，加入了对抗判别训练，同时优化分类、判别、相似度三个loss：[Semi-supervised domain adaptation via convolutional neural network](http://ieeexplore.ieee.org/abstract/document/8296801/)

- 20180312 arXiv 来自Google Brain团队的[Wasserstein Auto-Encoders](https://arxiv.org/abs/1711.01558) [代码](https://arxiv.org/abs/1711.01558)

- 20180226 CVPR-18 当源域的类别大于目标域的类别时，如何进行迁移学习？[Partial Transfer Learning with Selective Adversarial Networks](https://arxiv.org/abs/1707.07901)

- 20180116 ICLR-18 用对偶的形式替代对抗训练中原始问题的表达，从而进行分布对齐 [Stable Distribution Alignment using the Dual of the Adversarial Distance](https://arxiv.org/abs/1707.04046)

- 20180111 arXiv 在GAN中用原始问题的对偶问题替换max问题，使得梯度更好收敛 [Stable Distribution Alignment Using the Dual of the Adversarial Distance](https://arxiv.org/abs/1707.04046)

- 20180110 AAAI-18 将Wasserstein GAN用到domain adaptaiton中 [Wasserstein Distance Guided Representation Learning for Domain Adaptation](https://arxiv.org/abs/1707.01217)

- 20171218 arXiv [Partial Transfer Learning with Selective Adversarial Networks](https://arxiv.org/abs/1707.07901)
    - 假设target domain中的class是包含在source domain中，然后进行选择性的对抗学习

- 201707 发表在CVPR-17上，目前最好的对抗迁移学习文章：[Adversarial Representation Learning For Domain Adaptation](https://arxiv.org/abs/1707.01217)

- AAAI-18 [Multi-Adversarial Domain Adaptation](http://ise.thss.tsinghua.edu.cn/~mlong/doc/multi-adversarial-domain-adaptation-aaai18.pdf)

- CoRR abs/1711.09020 (2017) [StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation](https://arxiv.org/pdf/1707.01217.pdf)

- NIPS-17 [FADA: Few-Shot Adversarial Domain Adaptation](http://vision.csee.wvu.edu/~motiian/papers/FADA.pdf)

- ICCV-17 [CycleGAN: Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/pdf/1703.10593.pdf)

- ICCV-17 [DualGAN: DualGAN: Unsupervised Dual Learning for Image-to-Image Translation](https://arxiv.org/pdf/1704.02510.pdf)

- CVPR-17 [Asymmetric Tri-training for Unsupervised Domain Adaptation](https://arxiv.org/abs/1702.08400.pdf)

- ICML-17 [DiscoGAN: Learning to Discover Cross-Domain Relations with Generative Adversarial Networks](https://arxiv.org/abs/1703.05192)


- - -

### 传递迁移学习

- 传递迁移学习的第一篇文章，来自杨强团队，发表在KDD-15上：[Transitive Transfer Learning](http://dl.acm.org/citation.cfm?id=2783295)

- AAAI-17 杨强团队最新的传递迁移学习：[Distant Domain Transfer Learning](http://www3.ntu.edu.sg/home/sinnopan/publications/[AAAI17]Distant%20Domain%20Transfer%20Learning.pdf)


- - -

### 强化迁移学习

- 20180530 ICML-18 强化迁移学习：[Importance Weighted Transfer of Samples in Reinforcement Learning](https://arxiv.org/abs/1805.10886)

- 20180524 arXiv 用深度强化学习的方法学习domain adaptation中的采样策略：[Learning Sampling Policies for Domain Adaptation](https://arxiv.org/abs/1805.07641)

- 20180516 arXiv 探索了强化学习中的任务迁移：[Adversarial Task Transfer from Preference](https://arxiv.org/abs/1805.04686)

- 20180413 NIPS-17 基于后继特征迁移的强化学习：[Successor Features for Transfer in Reinforcement Learning](https://arxiv.org/abs/1606.05312)

- 20180404 IEEE TETCI-18 用迁移学习来玩星际争霸游戏：[StarCraft Micromanagement with Reinforcement Learning and Curriculum Transfer Learning](https://arxiv.org/abs/1804.00810)

- - -


### 应用

- 20180530 MNRAS 用迁移学习检测银河星系兼并：[Using transfer learning to detect galaxy mergers](https://arxiv.org/abs/1805.10289)

- 20180529 arXiv 迁移学习用于表情识别：[Meta Transfer Learning for Facial Emotion Recognition](https://arxiv.org/abs/1805.09946)

- 20180524 KDD-18 用迁移学习方法进行人们的ID迁移：[Learning and Transferring IDs Representation in E-commerce](https://arxiv.org/abs/1712.08289)

- 20180519 arXiv 用迁移学习进行物体检测，200帧/秒：[Object detection at 200 Frames Per Second](https://arxiv.org/abs/1804.04775)

- 20180519 arXiv 用迁移学习进行肢体语言识别：[Optimization of Transfer Learning for Sign Language Recognition Targeting Mobile Platform](https://arxiv.org/abs/1805.06618)

- 20180516 ACL-18 将对抗迁移学习用于危机状态下的舆情分析：[Domain Adaptation with Adversarial Training and Graph Embeddings](https://arxiv.org/abs/1805.05151)

- 20180504 arXiv 用迁移学习进行心脏病检测分类：[ECG Heartbeat Classification: A Deep Transferable Representation](https://arxiv.org/abs/1805.00794)

- 20180427 CVPR-18(workshop) 将深度迁移学习用于Person-reidentification： [Adaptation and Re-Identification Network: An Unsupervised Deep Transfer Learning Approach to Person Re-Identification](https://arxiv.org/abs/1804.09347)

- 20180426 arXiv 迁移学习用于医学名字实体检测；[Label-aware Double Transfer Learning for Cross-Specialty Medical Named Entity Recognition](https://arxiv.org/abs/1804.09021)

- 20180425 arXiv 将bagging和dropping结合起来进行迁移的一个深度网络：[A New Channel Boosted Convolution Neural Network using Transfer Learning](https://arxiv.org/abs/1804.08528)

- 20180425 arXiv 迁移学习应用于自然语言任务：[Dropping Networks for Transfer Learning](Dropping Networks for Transfer Learning)

- 20180421 arXiv 采用联合分布适配的深度迁移网络用于工业生产中的错误诊断：[Deep Transfer Network with Joint Distribution Adaptation: A New Intelligent Fault Diagnosis Framework for Industry Application](https://arxiv.org/abs/1804.07265)

- 20180419 arXiv 跨领域的推荐系统：[CoNet: Collaborative Cross Networks for Cross-Domain Recommendation](https://arxiv.org/abs/1804.06769)

- 20180413 arXiv 跨模态检索：[Cross-Modal Retrieval with Implicit Concept Association](https://arxiv.org/abs/1804.04318)

- 20180410 arXiv 用迁移学习进行犯罪现场的图像匹配：[Cross-Domain Image Matching with Deep Feature Maps](https://arxiv.org/abs/1804.02367)

- 20180408 ASRU-18 用迁移学习中的domain separation network进行speech recognition：[Unsupervised Adaptation with Domain Separation Networks for Robust Speech Recognition](https://arxiv.org/abs/1711.08010)

- 20180408 arXiv 小数据集上的迁移学习手写体识别：[Boosting Handwriting Text Recognition in Small Databases with Transfer Learning](https://arxiv.org/abs/1804.01527)

- 20180404 arXiv 用迁移学习进行物体检测：[Transferring Common-Sense Knowledge for Object Detection](https://arxiv.org/abs/1804.01077)

- 20180402 arXiv 将迁移学习用于癌症检测：[Improve the performance of transfer learning without fine-tuning using dissimilarity-based multi-view learning for breast cancer histology images](https://arxiv.org/abs/1803.11241)

- [迁移学习用于行为识别 Transfer learning for activity recognition](https://github.com/jindongwang/activityrecognition/blob/master/notes/%E8%BF%81%E7%A7%BB%E5%AD%A6%E4%B9%A0%E7%94%A8%E4%BA%8E%E8%A1%8C%E4%B8%BA%E8%AF%86%E5%88%AB.md)
