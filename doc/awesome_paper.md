# Awesome Transfer Learning Papers

Let's read some awesome transfer learning / domain adaptation papers.

这里收录了迁移学习各个研究领域的最新文章。

- - -

- [Awesome Transfer Learning Papers](#awesome-transfer-learning-papers)
	- [General Transfer Learning (普通迁移学习)](#general-transfer-learning-%E6%99%AE%E9%80%9A%E8%BF%81%E7%A7%BB%E5%AD%A6%E4%B9%A0)
		- [Theory (理论)](#theory-%E7%90%86%E8%AE%BA)
		- [Others (其他)](#others-%E5%85%B6%E4%BB%96)
	- [Domain Adaptation (领域自适应)](#domain-adaptation-%E9%A2%86%E5%9F%9F%E8%87%AA%E9%80%82%E5%BA%94)
		- [Traditional Methods (传统迁移方法)](#traditional-methods-%E4%BC%A0%E7%BB%9F%E8%BF%81%E7%A7%BB%E6%96%B9%E6%B3%95)
		- [Deep / Adversarial Methods (深度/对抗迁移方法)](#deep--adversarial-methods-%E6%B7%B1%E5%BA%A6%E5%AF%B9%E6%8A%97%E8%BF%81%E7%A7%BB%E6%96%B9%E6%B3%95)
	- [Domain Generalization](#domain-generalization)
	- [Multi-source Transfer Learning (多源迁移学习)](#multi-source-transfer-learning-%E5%A4%9A%E6%BA%90%E8%BF%81%E7%A7%BB%E5%AD%A6%E4%B9%A0)
	- [Heterogeneous Transfer Learning (异构迁移学习)](#heterogeneous-transfer-learning-%E5%BC%82%E6%9E%84%E8%BF%81%E7%A7%BB%E5%AD%A6%E4%B9%A0)
	- [Online Transfer Learning (在线迁移学习)](#online-transfer-learning-%E5%9C%A8%E7%BA%BF%E8%BF%81%E7%A7%BB%E5%AD%A6%E4%B9%A0)
	- [Zero-shot / Few-shot Learning](#zero-shot--few-shot-learning)
	- [Deep Transfer Learning (深度迁移学习)](#deep-transfer-learning-%E6%B7%B1%E5%BA%A6%E8%BF%81%E7%A7%BB%E5%AD%A6%E4%B9%A0)
		- [Non-Adversarial Transfer Learning (非对抗深度迁移)](#non-adversarial-transfer-learning-%E9%9D%9E%E5%AF%B9%E6%8A%97%E6%B7%B1%E5%BA%A6%E8%BF%81%E7%A7%BB)
		- [Deep Adversarial Transfer Learning (对抗迁移学习)](#deep-adversarial-transfer-learning-%E5%AF%B9%E6%8A%97%E8%BF%81%E7%A7%BB%E5%AD%A6%E4%B9%A0)
	- [Multi-task Learning (多任务学习)](#multi-task-learning-%E5%A4%9A%E4%BB%BB%E5%8A%A1%E5%AD%A6%E4%B9%A0)
	- [Transfer Reinforcement Learning (强化迁移学习)](#transfer-reinforcement-learning-%E5%BC%BA%E5%8C%96%E8%BF%81%E7%A7%BB%E5%AD%A6%E4%B9%A0)
	- [Transfer Metric Learning (迁移度量学习)](#transfer-metric-learning-%E8%BF%81%E7%A7%BB%E5%BA%A6%E9%87%8F%E5%AD%A6%E4%B9%A0)
	- [Transitive Transfer Learning (传递迁移学习)](#transitive-transfer-learning-%E4%BC%A0%E9%80%92%E8%BF%81%E7%A7%BB%E5%AD%A6%E4%B9%A0)
	- [Lifelong Learning (终身迁移学习)](#lifelong-learning-%E7%BB%88%E8%BA%AB%E8%BF%81%E7%A7%BB%E5%AD%A6%E4%B9%A0)
	- [Negative Transfer (负迁移)](#negative-transfer-%E8%B4%9F%E8%BF%81%E7%A7%BB)
	- [Transfer Learning Applications (应用)](#transfer-learning-applications-%E5%BA%94%E7%94%A8)

- - -

## General Transfer Learning (普通迁移学习)

### Theory (理论)

- 20181219 arXiv [PAC Learning Guarantees Under Covariate Shift](https://arxiv.org/abs/1812.06393)
    - PAC learning theory for covariate shift
    - Covariate shift问题的PAC学习理论

- 20181206 arXiv [Transferring Knowledge across Learning Processes](https://arxiv.org/abs/1812.01054)
	-  Transfer learning across learning processes
	- 学习过程中的知识迁移

- 20181128 arXiv [Theoretical Guarantees of Transfer Learning](https://arxiv.org/abs/1810.05986)
	-  Some theoretical analysis of transfer learning
	- 一些关于迁移学习的理论分析

- 20181117 arXiv [Theoretical Perspective of Deep Domain Adaptation](https://arxiv.org/abs/1811.06199)
	-  Providing some theory analysis on deep domain adaptation
	- 对deep domain adaptaiton做出了一些理论上的分析
  
- 20181106 workshop [GENERALIZATION BOUNDS FOR DOMAIN ADAPTATION VIA DOMAIN TRANSFORMATIONS](https://ieeexplore.ieee.org/abstract/document/8517092)
	-  Analyze some generalization bound for domain adaptation
	- 对domain adaptation进行了一些理论上的分析

### Others (其他)

- 20190118 arXiv [Domain Adaptation for Structured Output via Discriminative Patch Representations](https://arxiv.org/abs/1901.05427)
    - Domain adaptation for structured output
    - Domain adaptation用于结构化输出

- 20190111 arXiv [Low-Cost Transfer Learning of Face Tasks](https://arxiv.org/abs/1901.02675)
    - Infer what task transfers better and how to transfer
    - 探索对于一个预训练好的网络来说哪个任务适合迁移、如何迁移

- 20190111 arXiv [Transfer Representation Learning with TSK Fuzzy System](https://arxiv.org/abs/1901.02703)
    - Transfer learning with fuzzy system
    - 基于模糊系统的迁移学习

- 20190102 arXiv [An introduction to domain adaptation and transfer learning](https://arxiv.org/abs/1812.11806)
    - Another introduction to transfer learning
    - 另一个迁移学习和domain adaptation综述

- 20181217 arXiv [When Semi-Supervised Learning Meets Transfer Learning: Training Strategies, Models and Datasets](https://arxiv.org/abs/1812.05313)
    - Combining semi-supervised learning and transfer learning
    - 将半监督方法应用于迁移学习

- 20181127 arXiv [Privacy-preserving Transfer Learning for Knowledge Sharing](https://arxiv.org/abs/1811.09491)
	-  First work on privacy preserving in transfer learning
	- 第一篇探讨迁移学习中隐私保护的文章(第四范式、杨强)

- 20181121 arXiv [An Efficient Transfer Learning Technique by Using Final Fully-Connected Layer Output Features of Deep Networks](https://arxiv.org/abs/1811.07459)
    -  Using final fc layer to perform transfer learning
    - 使用最后一层全连接层进行迁移学习

- 20181121 arXiv [Not just a matter of semantics: the relationship between visual similarity and semantic similarity](https://arxiv.org/abs/1811.07120)
    -  Interpreting relationships between visual similarity and semantic similarity
    - 解释了视觉相似性和语义相似性的不同

- 20181008 arXiv [Unsupervised Learning via Meta-Learning](https://arxiv.org/abs/1810.02334)
	-  Meta-learning for unsupervised learning
	- 用于无监督学习的元学习

- 20101008 arXiv [Concept-drifting Data Streams are Time Series; The Case for Continuous Adaptation](https://arxiv.org/abs/1810.02266)
	-  Continuous adaptation for time series data
	- 对时间序列进行连续adaptation

- 20180925 arXiv [DT-LET: Deep Transfer Learning by Exploring where to Transfer](https://arxiv.org/pdf/1809.08541.pdf)
	-  Explore the suitable layers to transfer
	- 探索深度网络中效果表现好的对应的迁移层

- 20180919 JMLR [Invariant Models for Causal Transfer Learning](http://jmlr.csail.mit.edu/papers/volume19/16-432/16-432.pdf)
	-  Invariant models for causal transfer learning
	- 针对causal transfer learning提出不变模型



- 20180912 arXiv [Transfer Learning with Neural AutoML](https://arxiv.org/abs/1803.02780)
	- Applying transfer learning into autoML search
	- 将迁移学习思想应用于automl

- 20190904 arXiv [On the Minimal Supervision for Training Any Binary Classifier from Only Unlabeled Data](https://arxiv.org/abs/1808.10585)
	- Train binary classifiers from only unlabeled data
	- 仅从无标记数据训练二分类器

- 20180904 arXiv [Learning Data-adaptive Nonparametric Kernels](https://arxiv.org/abs/1808.10724)
	-  Learn a kernel that can do adaptation
	- 学习一个可以自适应的kernel

- 20180901 arXiv [Distance Based Source Domain Selection for Sentiment Classification](https://arxiv.org/abs/1808.09271)
	-  Propose a new domain selection method by combining existing distance functions
	- 提出一种混合已有多种距离公式的源领域选择方法

- 20180901 KBS [Transfer subspace learning via low-rank and discriminative reconstruction matrix](https://www.sciencedirect.com/science/article/pii/S0950705118304222)
	-  Transfer subspace learning via low-rank and discriminative reconstruction matrix
	- 通过低秩和重构进行迁移学习

- 20180825 arXiv [Transfer Learning for Estimating Causal Effects using Neural Networks](https://arxiv.org/abs/1808.07804)
	-  Using transfer learning for casual effect learning
	- 用迁移学习进行因果推理

- 20180724 ICPKR-18 [Knowledge-based Transfer Learning Explanation](https://arxiv.org/abs/1807.08372)
	-  Explain transfer learning things with some knowledge-based theory
	- 用一些基于knowledge的方法解释迁移学习

- 20180628 arXiv 提出Office数据集的实验室又放出一个数据集用于close set、open set、以及object detection的迁移学习：[Syn2Real: A New Benchmark forSynthetic-to-Real Visual Domain Adaptation](https://arxiv.org/abs/1806.09755)

- 20180605 arXiv 解决federated learning中的数据不同分布的问题：[Federated Learning with Non-IID Data](https://arxiv.org/abs/1806.00582)

- 20180604 arXiv 在Open set domain adaptation中，用共享和私有部分重建进行问题的解决：[Learning Factorized Representations for Open-set Domain Adaptation](https://arxiv.org/abs/1805.12277)

- 20180403 arXiv 选择最优的子类生成方便迁移的特征：[Class Subset Selection for Transfer Learning using Submodularity](https://arxiv.org/abs/1804.00060)

- 20180326 ICMLA-17 在类别不平衡情况下比较了一些迁移学习和传统方法的性能，并做出一些结论：[Comparing Transfer Learning and Traditional Learning Under Domain Class Imbalance](http://ieeexplore.ieee.org/abstract/document/8260654/)

- - - 

## Domain Adaptation (领域自适应)

Including domain adaptation and partial domain adaptation.

### Traditional Methods (传统迁移方法)

- 20180724 ACMMM-18 [Visual Domain Adaptation with Manifold Embedded Distribution Alignment](https://arxiv.org/abs/1807.07258)
	-  The state-of-the-art results of domain adaptation, better than most traditional and deep methods
	- 目前效果最好的非深度迁移学习方法，领先绝大多数最近的方法
	- Code: [MEDA](https://github.com/jindongwang/transferlearning/tree/master/code/traditional/MEDA)

- 20180912 arXiv [Unsupervised Domain Adaptation Based on Source-guided Discrepancy](https://arxiv.org/abs/1809.03839)
	-  Using source domain information to help domain adaptation
	- 使用源领域数据辅助目标领域进行domain adaptation

- 20181219 arXiv [Domain Adaptation on Graphs by Learning Graph Topologies: Theoretical Analysis and an Algorithm](https://arxiv.org/abs/1812.06944)
    - Domain adaptation on graphs
    - 在图上的领域自适应

- 20181121 arXiv [Deep Discriminative Learning for Unsupervised Domain Adaptation](https://arxiv.org/abs/1811.07134)
    -  Deep discriminative learning for domain adaptation
    - 同时进行源域和目标域上的分类判别

- 20181114 arXiv [Multiple Subspace Alignment Improves Domain Adaptation](https://arxiv.org/abs/1811.04491)
	-  Project domains into multiple subsapce to do domain adaptation
	- 将domain映射到多个subsapce上然后进行adaptation

- 20180912 ICIP-18 [Structural Domain Adaptation with Latent Graph Alignment](https://ieeexplore.ieee.org/abstract/document/8451245/)
    -  Using graph alignment for domain adaptation
    - 使用图对齐方式进行domain adaptation

- 20180912 IEEE Access [Unsupervised Domain Adaptation by Mapped Correlation Alignment](https://ieeexplore.ieee.org/abstract/document/8434290/)
    -  Mapped correlation alignment for domain adaptation
    - 用映射的关联对齐进行domain adaptation

- 20180912 ICALIP-18 [Domain Adaptation for Gaussian Process Classification](https://ieeexplore.ieee.org/abstract/document/8455721/)
    -  Domain Adaptation for Gaussian Process Classification
    - 在高斯过程分类中进行domain adaptation

- 20180701 arXiv 对domain adaptation问题，基于optimal transport提出一种新的特征选择方法：[Feature Selection for Unsupervised Domain Adaptation using Optimal Transport](https://arxiv.org/abs/1806.10861)

- 20180510 IEEE Trans. Cybernetics 提出一个通用的迁移学习框架，对不同的domain进行不同的特征变换：[Transfer Independently Together: A Generalized Framework for Domain Adaptation](https://ieeexplore-ieee-org.lib.ezproxy.ust.hk/abstract/document/8337102/)

- 20180403 TIP-18 一篇传统方法做domain adaptation的文章，比很多深度方法结果都好：[An Embarrassingly Simple Approach to Visual Domain Adaptation](http://ieeexplore.ieee.org/abstract/document/8325317/)

- 20180326 ICMLA-17 利用subsapce alignment进行迁移学习：[Transfer Learning for Large Scale Data Using Subspace Alignment](http://ieeexplore.ieee.org/abstract/document/8260772)

- 20180228 arXiv 一篇通过标签一致性和MMD准则进行domain adaptation的文章: [Discriminative Label Consistent Domain Adaptation](https://arxiv.org/abs/1802.08077)

- 20180226 AAAI-18 清华龙明盛组最新工作：[Unsupervised Domain Adaptation with Distribution Matching Machines](http://ise.thss.tsinghua.edu.cn/~mlong/doc/distribution-matching-machines-aaai18.pdf)

- 20180110 arXiv 一篇比较新的传统方法做domain adaptation的文章 [Close Yet Discriminative Domain Adaptation](https://arxiv.org/abs/1704.04235)

- 20180105 arXiv 最优的贝叶斯迁移学习 [Optimal Bayesian Transfer Learning](https://arxiv.org/abs/1801.00857)

- 20171201 ICCV-17 [When Unsupervised Domain Adaptation Meets Tensor Representations](http://openaccess.thecvf.com/content_iccv_2017/html/Lu_When_Unsupervised_Domain_ICCV_2017_paper.html)
    - 第一篇将Tensor与domain adaptation结合的文章。[代码](https://github.com/poppinace/TAISL)
    - [我的解读](https://zhuanlan.zhihu.com/p/31834244)

- 201711 ICCV-17 [Open set domain adaptation](http://openaccess.thecvf.com/content_iccv_2017/html/Busto_Open_Set_Domain_ICCV_2017_paper.html)。
    - 当source和target只共享某一些类别时，怎么处理？这个文章获得了ICCV 2017的Marr Prize Honorable Mention，值得好好研究。
    - [我的解读](https://zhuanlan.zhihu.com/p/31230331)

- 201710 [Domain Adaptation in Computer Vision Applications](https://books.google.com.hk/books?id=7181DwAAQBAJ&pg=PA95&lpg=PA95&dq=Learning+Domain+Invariant+Embeddings+by+Matching%E2%80%A6&source=bl&ots=fSc1yvZxU3&sig=XxmGZkrfbJ2zSsJcsHhdfRpjaqk&hl=zh-CN&sa=X&ved=0ahUKEwjzvODqkI3XAhUCE5QKHYStBywQ6AEIRDAE#v=onepage&q=Learning%20Domain%20Invariant%20Embeddings%20by%20Matching%E2%80%A6&f=false) 里面收录了若干篇domain adaptation的文章，可以大概看看。

- [学习迁移](https://arxiv.org/abs/1708.05629)(Learning to Transfer, L2T)
	- 迁移学习领域的新方向：与在线、增量学习结合
	- [我的解读](https://zhuanlan.zhihu.com/p/28888554)
    
- 201707 [Mutual Alignment Transfer Learning](https://arxiv.org/abs/1707.07907)

- 201708 [Learning Invariant Riemannian Geometric Representations Using Deep Nets](https://arxiv.org/abs/1708.09485)

- 20170812 ICML-18 [Learning To Transfer](https://arxiv.org/abs/1708.05629)，将迁移学习和增量学习的思想结合起来，为迁移学习的发展开辟了一个崭新的研究方向。[我的解读](https://zhuanlan.zhihu.com/p/28888554)

- NIPS-17 [JDOT: Joint distribution optimal transportation for domain adaptation](https://arxiv.org/pdf/1705.08848.pdf)

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

### Deep / Adversarial Methods (深度/对抗迁移方法)

- 20190102 WSDM-19 [Learning to Selectively Transfer: Reinforced Transfer Learning for Deep Text Matching](https://arxiv.org/abs/1812.11561)
    - Reinforced transfer learning for deep text matching
    - 迁移学习进行深度文本匹配

- 20190102 arXiv [DART: Domain-Adversarial Residual-Transfer Networks for Unsupervised Cross-Domain Image Classification](https://arxiv.org/abs/1812.11478)
    - Adversarial + residual for domain adaptation
    - 对抗+残差进行domain adaptation

- 20181220 arXiv [TWINs: Two Weighted Inconsistency-reduced Networks for Partial Domain Adaptation](https://arxiv.org/abs/1812.07405)
    - Two weighted inconsistency-reduced networks for partial domain adaptation
    - 两个权重网络用于部分domain adaptation

- 20181127 arXiv [Learning Grouped Convolution for Efficient Domain Adaptation](https://arxiv.org/abs/1811.09341)
	-  Group convolution for domain adaptation
	- 群体卷积进行domain adaptation

- 20181121 arXiv [Unsupervised Domain Adaptation: An Adaptive Feature Norm Approach](https://arxiv.org/abs/1811.07456)
    -  A nonparametric method for domain adaptation
    - 一种无参数的domain adaptation方法

- 20181121 arXiv [Domain Adaptive Transfer Learning with Specialist Models](https://arxiv.org/abs/1811.07056)
    -  Sample reweighting methods for domain adaptative
    - 样本权重更新法进行domain adaptation

- 20180926 ICLR-18 [Self-ensembling for visual domain adaptation](https://arxiv.org/abs/1706.05208)
	-  Self-ensembling for domain adaptation
	- 将self-ensembling应用于da

- 20180620 CVPR-18 用迁移学习进行fine tune：[Large Scale Fine-Grained Categorization and Domain-Specific Transfer Learning](https://arxiv.org/abs/1806.06193)

- 20180321 CVPR-18 构建了一个迁移学习算法，用于解决跨数据集之间的person-reidenfication: [Unsupervised Cross-dataset Person Re-identification by Transfer Learning of Spatial-Temporal Patterns](https://arxiv.org/abs/1803.07293)

- 20180315 ICLR-17 一篇综合进行two-sample stest的文章：[Revisiting Classifier Two-Sample Tests](https://arxiv.org/abs/1610.06545)

- 20171214 arXiv [Investigating the Impact of Data Volume and Domain Similarity on Transfer Learning Applications](https://arxiv.org/abs/1712.04008)
    - 在实验中探索了数据量多少，和相似度这两个因素对迁移学习效果的影响

- NIPS-17 [Learning Multiple Tasks with Multilinear Relationship Networks](http://papers.nips.cc/paper/6757-learning-multiple-tasks-with-deep-relationship-networks) 

- 深度适配网络（Deep Adaptation Network, DAN）
	- 发表在ICML-15上：learning transferable features with deep adaptation networks
	- [我的解读](https://zhuanlan.zhihu.com/p/27657910)

- - -

## Domain Generalization

- 20180701 arXiv 做迁移时，只用source数据，不用target数据训练：[Generalizing to Unseen Domains via Adversarial Data Augmentation](https://arxiv.org/abs/1805.12018)

- 201711 ICLR-18 [GENERALIZING ACROSS DOMAINS VIA CROSS-GRADIENT TRAINING](https://openreview.net/pdf?id=r1Dx7fbCW)
    - 不同于以往的工作，本文运用贝叶斯网络建模label和domain的依赖关系，抓住training、inference 两个过程，有效引入domain perturbation来实现domain adaptation。

- ICLR-18 [generalizing across domains via cross-gradient training](https://openreview.net/pdf?id=r1Dx7fbCW)

- 20181106 PRCV-18 [Domain Attention Model for Domain Generalization in Object Detection](https://link.springer.com/chapter/10.1007/978-3-030-03341-5_3)
	-  Adding attention for domain generalization
	- 在domain generalization中加入了attention机制

- 20181225 WACV-19 [Multi-component Image Translation for Deep Domain Generalization](https://arxiv.org/abs/1812.08974)
    - Using GAN generated images for domain generalization
    - 用GAN生成的图像进行domain generalization

- 20180724 arXiv [Domain Generalization via Conditional Invariant Representation](https://arxiv.org/abs/1807.08479)
	-  Using Conditional Invariant Representation for domain generalization
	- 生成条件不变的特征表达，用于domain generalization问题

- 20181212 arXiv [Beyond Domain Adaptation: Unseen Domain Encapsulation via Universal Non-volume Preserving Models](https://arxiv.org/abs/1812.03407)
    - Domain generalization method
    - 一种针对于unseen domain的学习方法

- 20171210 AAAI-18 [Learning to Generalize: Meta-Learning for Domain Generalization](https://arxiv.org/pdf/1710.03463.pdf)
    - 将Meta-Learning与domain generalization结合的文章，可以联系到近期较为流行的few-shot learning进行下一步思考。

- - -

## Multi-source Transfer Learning (多源迁移学习)

- 20181212 AIKP [Multi-source Transfer Learning](https://link.springer.com/chapter/10.1007/978-3-030-00734-8_8)
    - Multi-source transfer

- 20181207 arXiv [Moment Matching for Multi-Source Domain Adaptation](https://arxiv.org/abs/1812.01754)
	- Moment matching and propose a new large dataset for domain adaptation
	- 提出一种moment matching的网络，并且提出一种新的domain adaptation数据集，很大

- CoRR abs/1711.09020 (2017) [StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation](https://arxiv.org/pdf/1707.01217.pdf)

- 20180524 arXiv 探索了Multi-source迁移学习的一些理论：[Algorithms and Theory for Multiple-Source Adaptation](https://arxiv.org/abs/1805.08727)

- 20181117 AAAI-19 [Robust Optimization over Multiple Domains](https://arxiv.org/abs/1805.07588)
	-  Optimization on multi domains
	- 针对多个domain建模并优化

- 20180912 arXiv [Multi-target Unsupervised Domain Adaptation without Exactly Shared Categories](https://arxiv.org/abs/1809.00852)
    -  Multi-target domain adaptation
    - 多目标的domain adaptation

- 20180316 arXiv 用optimal transport解决domain adaptation中类别不平衡的问题：[Optimal Transport for Multi-source Domain Adaptation under Target Shift](https://arxiv.org/abs/1803.04899)

- - -

## Heterogeneous Transfer Learning (异构迁移学习)

- 20181113 ACML-18 [Unsupervised Heterogeneous Domain Adaptation with Sparse Feature Transformation](http://proceedings.mlr.press/v95/shen18b/shen18b.pdf)
	- Heterogeneous domain adaptation
	- 异构domain adaptation

- 20180901 TKDE [A General Domain Specific Feature Transfer Framework for Hybrid Domain Adaptation](https://ieeexplore.ieee.org/abstract/document/8432087/)
	- Hybrid DA: special case in Heterogeneous DA
	- 提出一种新的混合DA问题和方法

- 20180606 arXiv 一篇最近的对非对称情况下的异构迁移学习综述：[Asymmetric Heterogeneous Transfer Learning: A Survey](https://arxiv.org/abs/1804.10834)

- 20180403 Neural Processing Letters-18 异构迁移学习：[Label Space Embedding of Manifold Alignment for Domain Adaption](https://link.springer.com/article/10.1007/s11063-018-9822-8)

- 20180105 arXiv 异构迁移学习 [Heterogeneous transfer learning](https://arxiv.org/abs/1701.02511)

- - -

## Online Transfer Learning (在线迁移学习)

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

- - -

## Zero-shot / Few-shot Learning

- 20180612 CVPR-18 泛化的Zero-shot learning：[Generalized Zero-Shot Learning via Synthesized Examples](https://arxiv.org/abs/1712.03878)

- 20181106 arXiv [Zero-Shot Transfer VQA Dataset](https://arxiv.org/abs/1811.00692)
	- English: A dataset for zero-shot VQA transfer
	- 中文：一个针对zero-shot VQA的迁移学习数据集

- 20171222 NIPS 2017 用adversarial网络，当target中有很少量的label时如何进行domain adaptation：[Few-Shot Adversarial Domain Adaptation](http://papers.nips.cc/paper/7244-few-shot-adversarial-domain-adaptation)

- 20181225 arXiv [Learning Compositional Representations for Few-Shot Recognition](https://arxiv.org/abs/1812.09213)
    - Few-shot recognition

- 20181127 WACV-19 [Self Paced Adversarial Training for Multimodal Few-shot Learning](https://arxiv.org/abs/1811.09192)
	-  Multimodal training for single modal testing
	- 用多模态数据针对单一模态进行迁移

- 20180728 arXiv [Meta-learning autoencoders for few-shot prediction](https://arxiv.org/abs/1807.09912)
	-  Using meta-learning for few-shot transfer learning
	- 用元学习进行迁移学习

- 20171216 arXiv [Zero-Shot Deep Domain Adaptation](https://arxiv.org/abs/1707.01922)
    - 当target domain的数据不可用时，如何用相关domain的数据进行辅助学习？

- - -

## Deep Transfer Learning (深度迁移学习)

### Non-Adversarial Transfer Learning (非对抗深度迁移)

- 20190109 InfSc [Robust Unsupervised Domain Adaptation for Neural Networks via Moment Alignment](https://doi.org/10.1016/j.ins.2019.01.025)
    - Extension of Central Moment Discrepancy (ICLR-17) approach

- 20181212 ICONIP-18 [Domain Adaptation via Identical Distribution Across Models and Tasks](https://link.springer.com/chapter/10.1007/978-3-030-04167-0_21)
    -  Transfer from large net to small net
    - 从大网络迁移到小网络

- 20181212 AIKP [Deep Domain Adaptation](https://link.springer.com/chapter/10.1007/978-3-030-00734-8_9)
    -  Low-rank + deep nn for domain adaptation
    - Low-rank用于深度迁移

- 20181211 arXiv [Deep Variational Transfer: Transfer Learning through Semi-supervised Deep Generative Models](https://arxiv.org/abs/1812.03123)
    -  Transfer learning with deep generative model
    - 通过深度生成模型进行迁移学习

- 20181207 arXiv [Feature Matters: A Stage-by-Stage Approach for Knowledge Transfer](https://arxiv.org/abs/1812.01819)
	-  Feature transfer in student-teacher networks
	- 在学生-教师网络中进行特征迁移

- 20181128 arXiv [Low-resolution Face Recognition in the Wild via Selective Knowledge Distillation](https://arxiv.org/abs/1811.09998)
	-  Knowledge distilation for low-resolution face recognition
	- 将知识蒸馏应用于低分辨率的人脸识别

- 20181128 arXiv [One Shot Domain Adaptation for Person Re-Identification](https://arxiv.org/abs/1811.10144)
	-  One shot learning for REID
	- One shot for再识别

- 20181123 arXiv [SpotTune: Transfer Learning through Adaptive Fine-tuning](https://arxiv.org/abs/1811.08737)
	-  Very interesting work: how exactly determine the finetune process?
	- 很有意思的工作：如何决定finetune的策略？

- 20181121 arXiv [Integrating domain knowledge: using hierarchies to improve deep classifiers](https://arxiv.org/abs/1811.07125)
    -  Using hierarchies to help deep learning
    - 借助于层次关系来帮助深度网络进行学习

- 20181117 arXiv [AdapterNet - learning input transformation for domain adaptation](https://arxiv.org/abs/1805.11601)
	-  Learning input transformation for domain adaptation
	- 对domain adaptation任务学习输入的自适应

- 20181115 AAAI-19 [Exploiting Local Feature Patterns for Unsupervised Domain Adaptation](https://arxiv.org/abs/1811.05042)
	-  Local domain alignment for domain adaptation
	- 局部领域自适应

- 20181115 NIPS-18 [Co-regularized Alignment for Unsupervised Domain Adaptation](https://arxiv.org/abs/1811.05443)
	-  The idea is the same with the above one...
	- 仍然是局部对齐。。。

- 20181113 NIPS-18 [Generalized Zero-Shot Learning with Deep Calibration Network](http://ise.thss.tsinghua.edu.cn/~mlong/doc/deep-calibration-network-nips18.pdf)
	-  Deep calibration network for zero-shot learning
	- 提出deep calibration network进行zero-shot learning

- 20181110 AAAI-19 [Knowledge Transfer via Distillation of Activation Boundaries Formed by Hidden Neurons](https://arxiv.org/abs/1811.03233)
	-  Transfer learning for bounding neuron activation boundaries
	- 使用迁移学习进行神经元激活边界判定

- 20181109 PAMI-18 [Transferable Representation Learning with Deep Adaptation Networks](https://ieeexplore.ieee.org/abstract/document/8454781/authors#authors)
	-  Journal version of DAN paper
	- DAN的Journal版本

- 20181108 arXiv [Deep feature transfer between localization and segmentation tasks](https://arxiv.org/abs/1811.02539)
	-  Feature transfer between localization and segmentation
	- 在定位与分割任务间进行迁移

- 20181107 BigData-18 [Transfer learning for time series classification](https://arxiv.org/abs/1811.01533)
	-  First work on deep transfer learning for time series classification
	- 第一个将深度迁移学习用于时间序列分类

- 20181106 PRCV-18 [Deep Local Descriptors with Domain Adaptation](https://link.springer.com/chapter/10.1007/978-3-030-03335-4_30)
	-  Adding MMD layers to conv and fc layers
	- 在卷积和全连接层都加入MMD

- 20181106 LNCS-18 [LSTN: Latent Subspace Transfer Network for Unsupervised Domain Adaptation](https://link.springer.com/chapter/10.1007/978-3-030-03335-4_24)
	-  Combine subspace learning and neural network for DA
	- 将子空间表示和深度网络结合起来用于DA



- 20181105 SIGGRAPI-18 [Unsupervised representation learning using convolutional and stacked auto-encoders: a domain and cross-domain feature space analysis](https://arxiv.org/abs/1811.00473)
	-  Representation learning for cross-domains
	- 跨领域的特征学习

- 20181105 arXiv [Progressive Memory Banks for Incremental Domain Adaptation](https://arxiv.org/abs/1811.00239)
	-  Progressive memory bank in RNN for incremental DA
	- 针对增量的domain adaptation，进行记忆单元的RNN学习

- 20180909 arXiv [A Survey on Deep Transfer Learning](https://arxiv.org/abs/1808.01974)
	-  A survey on deep transfer learning
	- 深度迁移学习的survey

- 20180901 arXiv [Joint Domain Alignment and Discriminative Feature Learning for Unsupervised Deep Domain Adaptation](https://arxiv.org/abs/1808.09347)
	-  deep domain adaptation + intra-class / inter-class distance
	- 深度domain adaptation再加上类内类间距离学习

- 20180819 arXiv [Conceptual Domain Adaptation Using Deep Learning](https://arxiv.org/abs/1808.05355)
	-  A search framework for deep transfer learning
	- 提出一个可以搜索的framework进行迁移学习

- 20180731 ECCV-18 [DeepJDOT: Deep Joint Distribution Optimal Transport for Unsupervised Domain Adaptation](https://arxiv.org/abs/1803.10081)
	-  Deep + Joint distribution adaptation + optimal transport
	- 深度 + 联合分布适配 + optimal transport
	
- 20180731 ICLR-18 [Few Shot Learning with Simplex](https://arxiv.org/abs/1807.10726)
	-  Represent deep learning using the simplex
	- 用单纯性表征深度学习

- 20180724 AIAI-18 [Improving Deep Models of Person Re-identification for Cross-Dataset Usage](https://arxiv.org/abs/1807.08526)
	-  apply deep models to cross-dataset RE-ID
	- 将深度迁移学习应用于跨数据集的Re-ID

- 20180724 ECCV-18 [Zero-Shot Deep Domain Adaptation](https://arxiv.org/abs/1707.01922)
	-  Perform zero-shot domain adaptation when there is no target domain data available 
	- 当目标领域的数据不可用时如何进行domain adaptation :

- 20180724 ICCSE-18 [Deep Transfer Learning for Cross-domain Activity Recognition](https://arxiv.org/abs/1807.07963)
	-  Provide source domain selection and activity recognition for cross-domain activity recognition
	- 提出了跨领域行为识别中的深度方法模型，以及相关的领域选择方法

- 20180530 arXiv 用于深度网络的鲁棒性domain adaptation方法：[Robust Unsupervised Domain Adaptation for Neural Networks via Moment Alignment](https://arxiv.org/abs/1711.06114)

- 20180522 arXiv 用CNN进行跨领域的属性学习：[Cross-domain attribute representation based on convolutional neural network](https://arxiv.org/abs/1805.07295)

- 20180428 CVPR-18 相互协同学习：[Deep Mutual Learning](https://github.com/YingZhangDUT/Deep-Mutual-Learning)

- 20180428 ICLR-18 自集成学习用于domain adaptation：[Self-ensembling for visual domain adaptation](https://github.com/Britefury/self-ensemble-visual-domain-adapt)

- 20180428 IJCAI-18 将knowledge distilation用于transfer learning，然后进行视频分类：[Better and Faster: Knowledge Transfer from Multiple Self-supervised Learning Tasks via Graph Distillation for Video Classification](https://arxiv.org/abs/1804.10069)

- 20180426 arXiv 深度学习中的参数如何进行迁移？（杨强团队）：[Parameter Transfer Unit for Deep Neural Networks](https://arxiv.org/abs/1804.08613)

- 20180425 CVPR-18(oral) 对不同的视觉任务进行建模，从而可以进行深层次的transfer：[Taskonomy: Disentangling Task Transfer Learning](https://arxiv.org/abs/1804.08328)

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

- ICLR-17 [Central Moment Discrepancy (CMD) for Domain-Invariant Representation Learning](https://openreview.net/pdf?id=SkB-_mcel)

- ICCV-17 [AutoDIAL: Automatic DomaIn Alignment Layers](https://arxiv.org/pdf/1704.08082.pdf)

- ICCV-17 [CCSA: Unified Deep Supervised Domain Adaptation and Generalization](http://vision.csee.wvu.edu/~motiian/papers/CCSA.pdf)

- ICML-17 [JAN: Deep Transfer Learning with Joint Adaptation Networks](http://ise.thss.tsinghua.edu.cn/~mlong/doc/joint-adaptation-networks-icml17.pdf)

- 2017 Google: [Learning Transferable Architectures for Scalable Image Recognition](https://arxiv.org/abs/1707.07012)

- NIPS-16 [RTN: Unsupervised Domain Adaptation with Residual Transfer Networks](http://ise.thss.tsinghua.edu.cn/~mlong/doc/residual-transfer-network-nips16.pdf)

- CoRR abs/1603.04779 (2016) [AdaBN: Revisiting batch normalization for practical domain adaptation](https://arxiv.org/pdf/1603.04779.pdf)

- JMLR-16 [DANN: Domain-adversarial training of neural networks](http://www.jmlr.org/papers/volume17/15-239/15-239.pdf)

- 20171226 NIPS 2016 把传统工作搬到深度网络中的范例：不是只学习domain之间的共同feature，还学习每个domain specific的feature。这篇文章写得非常清楚，通俗易懂！ [Domain Separation Networks](http://papers.nips.cc/paper/6254-domain-separation-networks) | [代码](https://github.com/tensorflow/models/tree/master/research/domain_adaptation)

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

### Deep Adversarial Transfer Learning (对抗迁移学习)

- 20181217 arXiv [DLOW: Domain Flow for Adaptation and Generalization](https://arxiv.org/abs/1812.05418)
    - Domain flow for adaptation and generalization
    - 域流方法应用于领域自适应和扩展

- 20181212 arXiv [Learning Transferable Adversarial Examples via Ghost Networks](https://arxiv.org/abs/1812.03413)
    - Use ghost networks to learn transferrable adversarial examples
    - 使用ghost网络来学习可迁移的对抗样本

- 20181211 arXiv [Adversarial Transfer Learning](https://arxiv.org/abs/1812.02849)
    -  A survey on adversarial domain adaptation
    - 一个关于对抗迁移的综述，特别用在domain adaptation上

- 20181205 arXiv [Unsupervised Domain Adaptation using Generative Models and Self-ensembling](https://arxiv.org/abs/1812.00479)
	-  UDA using CycleGAN
	- 基于CycleGAN的domain adaptation

- 20181205 arXiv [VADRA: Visual Adversarial Domain Randomization and Augmentation](https://arxiv.org/abs/1812.00491)
	-  Domain randomization and augmentation
	- Domain randomization和增强

- 20181130 arXiv [Identity Preserving Generative Adversarial Network for Cross-Domain Person Re-identification](https://arxiv.org/abs/1811.11510)
	-  Cross-domain reID
	- 跨领域的行人再识别

- 20181129 AAAI-19 [Exploiting Coarse-to-Fine Task Transfer for Aspect-level Sentiment Classification](https://arxiv.org/abs/1811.10999)
	-  Aspect-level sentiment classification
	- 迁移学习用于情感分类

- 20181128 arXiv [Geometry-Consistent Generative Adversarial Networks for One-Sided Unsupervised Domain Mapping](https://arxiv.org/abs/1809.05852)
	-  CycleGAN for domain adaptation
	- CycleGAN用于domain adaptation

- 20181127 arXiv [Distorting Neural Representations to Generate Highly Transferable Adversarial Examples](https://arxiv.org/abs/1811.09020)
	-  Generate transferrable examples to fool networks
	- 生成一些可迁移的对抗样本来迷惑神经网络，在各个网络上都表现好

- 20181123 arXiv [Progressive Feature Alignment for Unsupervised Domain Adaptation](https://arxiv.org/abs/1811.08585)
		-  Progressively selecting confident pseudo labeled samples for transfer
		- 渐进式选择置信度高的伪标记进行迁移

- 20181113 NIPS-18 [Conditional Adversarial Domain Adaptation](http://ise.thss.tsinghua.edu.cn/~mlong/doc/conditional-adversarial-domain-adaptation-nips18.pdf)
	-  Using conditional GAN for domain adaptation
	- 用conditional GAN进行domain adaptation

- 20181107 NIPS-18 [Invariant Representations without Adversarial Training](https://arxiv.org/abs/1805.09458)
	-  Get invariant representations without adversarial training
	- 不进行对抗训练获得不变特征表达

- 20181105 arXiv [Efficient Multi-Domain Dictionary Learning with GANs](https://arxiv.org/abs/1811.00274)
	-  Dictionary learning for multi-domains using GAN
	- 用GAN进行多个domain的字典学习

- 20181012 arXiv [Domain Confusion with Self Ensembling for Unsupervised Adaptation](https://arxiv.org/abs/1810.04472)
	-  Domain confusion and self-ensembling for DA
	- 用于Domain adaptation的confusion和self-ensembling方法

- 20180912 arXiv [Improving Adversarial Discriminative Domain Adaptation](https://arxiv.org/abs/1809.03625)
	-  Improve ADDA using source domain labels
	- 提高ADDA方法的精度，使用source domain的label

- 20180731 ECCV-18 [Dist-GAN: An Improved GAN using Distance Constraints](https://arxiv.org/abs/1803.08887)
	-  Embed an autoencoder in GAN to improve its stability in training and propose two distances
	- 将autoencoder集成到GAN中，提出相应的两种距离进行度量，提高了GAN的稳定性
	- Code: [Tensorflow](https://github.com/tntrung/gan)

- 20180724 arXiv [Generalization Bounds for Unsupervised Cross-Domain Mapping with WGANs](https://arxiv.org/abs/1807.08501)
	-  Provide a generalization bound for unsupervised WGAN in transfer learning
	- 对迁移学习中无监督的WGAN进行了一些理论上的分析

- 20180724 ECCV-18 [Unsupervised Image-to-Image Translation with Stacked Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1807.08536)
	-  Using stacked CycleGAN to perform image-to-image translation
	- 用stacked cycleGAN进行image-to-image的翻译

- 20180628 ICML-18 Pixel-level和feature-level的domain adaptation：[CyCADA: Cycle-Consistent Adversarial Domain Adaptation](https://arxiv.org/abs/1711.03213)

- 20180619 CVPR-18 将optimal transport加入adversarial中进行domain adaptation：[Re-weighted Adversarial Adaptation Network for Unsupervised Domain Adaptation](http://openaccess.thecvf.com/content_cvpr_2018/CameraReady/1224.pdf)

- 20180616 CVPR-18 用GAN进行domain adaptation：[Generate To Adapt: Aligning Domains using Generative Adversarial Networks](https://arxiv.org/abs/1704.01705)

- 20180612 ICML-18 利用多个数据集辅助，从而提高目标领域的学习能力：[RadialGAN: Leveraging multiple datasets to improve target-specific predictive models using Generative Adversarial Networks](https://arxiv.org/abs/1802.06403)

- 20180612 ICML-18 利用GAN进行多个domain的联合分布优化：[JointGAN: Multi-Domain Joint Distribution Learning with Generative Adversarial Nets](https://arxiv.org/abs/1806.02978)

- 20180605 arXiv [NAM: Non-Adversarial Unsupervised Domain Mapping](https://arxiv.org/abs/1806.00804)

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

- ICCV-17 [CycleGAN: Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/pdf/1703.10593.pdf)

- ICCV-17 [DualGAN: DualGAN: Unsupervised Dual Learning for Image-to-Image Translation](https://arxiv.org/pdf/1704.02510.pdf)

- CVPR-17 [Asymmetric Tri-training for Unsupervised Domain Adaptation](https://arxiv.org/abs/1702.08400.pdf)

- ICML-17 [DiscoGAN: Learning to Discover Cross-Domain Relations with Generative Adversarial Networks](https://arxiv.org/abs/1703.05192)

- - -

## Multi-task Learning (多任务学习)

- 20181128 arXiv [A Framework of Transfer Learning in Object Detection for Embedded Systems](https://arxiv.org/abs/1811.04863)
	-  A Framework of Transfer Learning in Object Detection for Embedded Systems
	- 一个用于嵌入式系统的迁移学习框架

- 20181012 NIPS-18 [Multi-Task Learning as Multi-Objective Optimization](https://arxiv.org/abs/1810.04650)
	-  Solve the multi-task learning as a multi-objective optimization problem
	- 将多任务问题看成一个多目标优化问题进行求解

- 20181008 PSB-19 [The Effectiveness of Multitask Learning for Phenotyping with Electronic Health Records Data](https://arxiv.org/abs/1808.03331)
	-  Evaluate the effectiveness of multitask learning for phenotyping
	- 评估多任务学习对于表型的作用

- 20180828 arXiv [Self-Paced Multi-Task Clustering](https://arxiv.org/abs/1808.08068)
	-  Multi-task clustering
	- 多任务聚类

- 20180622 arXiv 探索了多任务迁移学习中的不确定性：[Uncertainty in Multitask Transfer Learning](https://arxiv.org/abs/1806.07528)

- 20180524 arXiv 杨强团队、与之前的learning to learning类似，这里提供了一个从经验中学习的learning to multitask框架：[Learning to Multitask](https://arxiv.org/abs/1805.07541)

- - -

## Transfer Reinforcement Learning (强化迁移学习)

- 20181212 NeurIPS-18 workshop [Efficient transfer learning and online adaptation with latent variable models for continuous control](https://arxiv.org/abs/1812.03399)
    -  Reinforcement transfer learning with latent models
    - 隐变量模型用于迁移强化学习的控制

- 20181128 arXiv [Hardware Conditioned Policies for Multi-Robot Transfer Learning](https://arxiv.org/abs/1811.09864)
	-  Hardware Conditioned Policies for Multi-Robot Transfer Learning
	- 多个机器人之间的迁移学习

- 20180926 arXiv [Target Transfer Q-Learning and Its Convergence Analysis](https://arxiv.org/abs/1809.08923)
	-  Analyze the risk of transfer q-learning
	- 提供了在Q learning的任务迁移中一些理论分析

- 20180926 arXiv [Domain Adaptation in Robot Fault Diagnostic Systems](https://arxiv.org/abs/1809.08626)
	-  Apply domain adaptation in robot fault diagnostic system
	- 将domain adaptation应用于机器人故障检测系统

- 20180912 arXiv [VPE: Variational Policy Embedding for Transfer Reinforcement Learning](https://arxiv.org/abs/1809.03548)
	-  Policy transfer in reinforcement learning
	- 增强学习中的策略迁移

- 20180909 arXiv [Transferring Deep Reinforcement Learning with Adversarial Objective and Augmentation](https://arxiv.org/abs/1809.00770)
	-  deep + adversarial + reinforcement learning transfer
	- 深度对抗迁移学习用于强化学习

- 20180530 ICML-18 强化迁移学习：[Importance Weighted Transfer of Samples in Reinforcement Learning](https://arxiv.org/abs/1805.10886)

- 20180524 arXiv 用深度强化学习的方法学习domain adaptation中的采样策略：[Learning Sampling Policies for Domain Adaptation](https://arxiv.org/abs/1805.07641)

- 20180516 arXiv 探索了强化学习中的任务迁移：[Adversarial Task Transfer from Preference](https://arxiv.org/abs/1805.04686)

- 20180413 NIPS-17 基于后继特征迁移的强化学习：[Successor Features for Transfer in Reinforcement Learning](https://arxiv.org/abs/1606.05312)

- 20180404 IEEE TETCI-18 用迁移学习来玩星际争霸游戏：[StarCraft Micromanagement with Reinforcement Learning and Curriculum Transfer Learning](https://arxiv.org/abs/1804.00810)

- - -

## Transfer Metric Learning (迁移度量学习)

- 20181012 arXiv [Transfer Metric Learning: Algorithms, Applications and Outlooks](https://arxiv.org/abs/1810.03944)
	-  A survey on transfer metric learning
	- 一篇迁移度量学习的综述

- 20180622 arXiv 基于深度迁移学习的度量学习：[DEFRAG: Deep Euclidean Feature Representations through Adaptation on the Grassmann Manifold](https://arxiv.org/abs/1806.07688)

- 20181117 arXiv [Distance Measure Machines](https://arxiv.org/abs/1803.00250)
	-  Machines that measures distances
	- 衡量距离的算法

- 20180605 KDD-10 迁移度量学习：[Transfer metric learning by learning task relationships](https://dl.acm.org/citation.cfm?id=1835954)

- 20180606 arXiv 将流形和统计信息联合起来构成一个domain adaptation框架：[A Unified Framework for Domain Adaptation using Metric Learning on Manifolds](https://arxiv.org/abs/1804.10834)

- 20180605 CVPR-15 深度度量迁移学习：[Deep metric transfer learning](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Hu_Deep_Transfer_Metric_2015_CVPR_paper.pdf)

- 20180425 arXiv 探索各个层对于迁移任务的作用，方便以后的迁移。比较有意思：[CactusNets: Layer Applicability as a Metric for Transfer Learning](https://arxiv.org/abs/1804.07846)

- - -

## Transitive Transfer Learning (传递迁移学习)

- 传递迁移学习的第一篇文章，来自杨强团队，发表在KDD-15上：[Transitive Transfer Learning](http://dl.acm.org/citation.cfm?id=2783295)

- AAAI-17 杨强团队最新的传递迁移学习：[Distant Domain Transfer Learning](http://www3.ntu.edu.sg/home/sinnopan/publications/[AAAI17]Distant%20Domain%20Transfer%20Learning.pdf)

- 20180819 LNCS-2018 [Distant Domain Adaptation for Text Classification](https://link.springer.com/chapter/10.1007/978-3-319-99365-2_5)
	-  Propose a selected algorithm for distant domain text classification
	- 提出一个用于远域的文本分类方法

- - -

## Lifelong Learning (终身迁移学习)

- 20180323 arXiv 终身迁移学习与增量学习结合：[Incremental Learning-to-Learn with Statistical Guarantees](https://arxiv.org/abs/1803.08089)

- 20180111 arXiv 一种新的终身学习框架，与L2T的思路有一些类似 [Lifelong Learning for Sentiment Classification](https://arxiv.org/abs/1801.02808)

- - -

## Negative Transfer (负迁移)

- 20181128 arXiv [Characterizing and Avoiding Negative Transfer](https://arxiv.org/abs/1811.09751)
	-  Analyzing and formalizing negative transfer, then propose a new method
	- 分析并形式化负迁移，进而提出自己的方法

- - -

## Transfer Learning Applications (应用)

See [HERE](https://github.com/jindongwang/transferlearning/blob/master/doc/transfer_learning_application.md) for a full list of transfer learning applications.
