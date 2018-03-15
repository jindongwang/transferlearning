# Awesome transfer learning papers

Let's read some awesome transfer learning / domain adaptation papers!

这里收录了迁移学习各个研究领域的最新文章。

- - -

## 目录

- [领域自适应(非深度) Domain Adaptation (Shallow)](#领域自适应)

- [在线迁移学习 Online transfer learning](#在线迁移学习)

- [终身迁移学习 Lifelong transfer learning](#终身迁移学习)

- [异构迁移学习 Heterogeneous Transfer Learning](#异构迁移学习)

- [深度迁移学习 Deep Transfer Learning](#深度迁移学习)
    
    - [深度对抗迁移迁移学习 Deep Adversarial transfer learning](#对抗迁移学习)

- [传递迁移学习 Transitive Transfer Learning](#传递迁移学习)

- - -

### 领域自适应

- 20180316 arXiv 用optimal transport解决domain adaptation中类别不平衡的问题：[Optimal Transport for Multi-source Domain Adaptation under Target Shift](https://arxiv.org/abs/1803.04899)

- 20180315 ICLR-17 一篇综合进行two-sample test的文章：[Revisiting Classifier Two-Sample Tests](https://arxiv.org/abs/1610.06545)

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

- 20180111 arXiv 一种新的终身学习框架，与L2T的思路有一些类似 [Lifelong Learning for Sentiment Classification](https://arxiv.org/abs/1801.02808)


- - -

### 异构迁移学习

- 20180105 arXiv 异构迁移学习 [Heterogeneous transfer learning](https://arxiv.org/abs/1701.02511)


- - -

### 深度迁移学习

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


