# 1. Awesome Transfer Learning Papers

Let's read some awesome transfer learning / domain adaptation papers.

这里收录了迁移学习各个研究领域的最新文章。

- [1. Awesome Transfer Learning Papers](#1-awesome-transfer-learning-papers)
	- [1.1. General Transfer Learning (普通迁移学习)](#11-general-transfer-learning-普通迁移学习)
		- [1.1.1. Theory (理论)](#111-theory-理论)
		- [1.1.2. Others (其他)](#112-others-其他)
	- [1.2. Domain Adaptation (领域自适应)](#12-domain-adaptation-领域自适应)
		- [1.2.1. Traditional Methods (传统迁移方法)](#121-traditional-methods-传统迁移方法)
		- [1.2.2. Deep / Adversarial Methods (深度/对抗迁移方法)](#122-deep--adversarial-methods-深度对抗迁移方法)
	- [1.3. Domain Generalization](#13-domain-generalization)
	- [1.4. Multi-source Transfer Learning (多源迁移学习)](#14-multi-source-transfer-learning-多源迁移学习)
	- [1.5. Heterogeneous Transfer Learning (异构迁移学习)](#15-heterogeneous-transfer-learning-异构迁移学习)
	- [1.6. Online Transfer Learning (在线迁移学习)](#16-online-transfer-learning-在线迁移学习)
	- [1.7. Zero-shot / Few-shot Learning](#17-zero-shot--few-shot-learning)
		- [1.7.1. Zero-shot Learning based on Data Synthesis (基于样本生成的零样本学习)](#171-zero-shot-learning-based-on-data-synthesis-基于样本生成的零样本学习)
	- [1.8. Deep Transfer Learning (深度迁移学习)](#18-deep-transfer-learning-深度迁移学习)
		- [1.8.1. Non-Adversarial Transfer Learning (非对抗深度迁移)](#181-non-adversarial-transfer-learning-非对抗深度迁移)
		- [1.8.2. Deep Adversarial Transfer Learning (对抗迁移学习)](#182-deep-adversarial-transfer-learning-对抗迁移学习)
	- [1.9. Multi-task Learning (多任务学习)](#19-multi-task-learning-多任务学习)
	- [1.10. Transfer Reinforcement Learning (强化迁移学习)](#110-transfer-reinforcement-learning-强化迁移学习)
	- [1.11. Transfer Metric Learning (迁移度量学习)](#111-transfer-metric-learning-迁移度量学习)
	- [1.12. Transitive Transfer Learning (传递迁移学习)](#112-transitive-transfer-learning-传递迁移学习)
	- [1.13. Lifelong Learning (终身迁移学习)](#113-lifelong-learning-终身迁移学习)
	- [1.14. Negative Transfer (负迁移)](#114-negative-transfer-负迁移)
	- [1.15. Transfer Learning Applications (应用)](#115-transfer-learning-applications-应用)

## 1.1. General Transfer Learning (普通迁移学习)

### 1.1.1. Theory (理论)

- 20200702 [ICML-20] [Few-shot domain adaptation by causal mechanism transfer](https://arxiv.org/pdf/2002.03497.pdf)
  	- The first work on causal transfer learning
  	- 日本理论组大佬Sugiyama的工作，causal transfer learning

- 20191008 CVPR-19 [Characterizing and Avoiding Negative Transfer](https://arxiv.org/abs/1811.09751)
  	- Characterizing and avoid negative transfer
  	- 形式化并提出如何避免负迁移

- 20190301 ALT-19 [A Generalized Neyman-Pearson Criterion for Optimal Domain Adaptation](https://arxiv.org/abs/1810.01545)
    - A new criterion for domain adaptation
    - 提出一种新的可以强化domain adaptation表现的度量

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

### 1.1.2. Others (其他)

- 20200706 [Learn Faster and Forget Slower via Fast and Stable Task Adaptation](https://arxiv.org/abs/2007.01388)
- 20200706 [In Search of Lost Domain Generalization](https://arxiv.org/abs/2007.01434)
- 20200706 [ICML-20] [Continuously Indexed Domain Adaptation](https://arxiv.org/abs/2007.01807)
- 20200706 [Interactive Knowledge Distillation](https://arxiv.org/abs/2007.01476)
- 20200706 [Domain Adaptation without Source Data](https://arxiv.org/abs/2007.01524)

- 20200629 [ICML-20] [Do We Really Need to Access the Source Data? Source Hypothesis Transfer for Unsupervised Domain Adaptation](https://arxiv.org/abs/2002.08546)
    	- Source-free adaptation
    	- 在adaptation过程中不访问source data
- 20200629 [Transfer learning via L1 regulaziation](https://arxiv.org/abs/2006.14845)
	- Using L1 regularizationg for transfer learning
- 20200629 [ICML-20] [Graph Optimal Transport for Cross-Domain Alignment])(https://arxiv.org/abs/2006.14744)
	- Graph OT for cross-domain alignment

- 20200615 [Rethinking Pre-training and Self-training](https://arxiv.org/abs/2006.06882)

- 20200615 [Double Double Descent: On Generalization Errors in Transfer Learning between Linear Regression Tasks](https://arxiv.org/abs/2006.07002)

- 20200412 ICML-19 [Towards understanding knowledge distillation](http://proceedings.mlr.press/v97/phuong19a.html)
  	- Some theoretical and empirical understanding to knowledge distllation
  	- 对知识蒸馏的一些理论和实验的分析

- 20200210 WACVW-20 [Impact of ImageNet Model Selection on Domain Adaptation](https://arxiv.org/abs/2002.02559)
  	- A good experiment paper to indicate the power of representations
  	- 一篇很好的实验paper，揭示了深度特征+传统方法的有效性

- 20191202 AAAI-20 [Towards Oracle Knowledge Distillation with Neural Architecture Search](https://arxiv.org/abs/1911.13019)
   - Using NAS for knowledge Distillation
   - 用NAS帮助知识蒸馏

- 20191202 AAAI-20 [Stable Learning via Sample Reweighting](https://arxiv.org/abs/1911.12580)
   - Theoretical sample reweigting
   - 理论和方法，用于sample reweight

- 20191202 arXiv [Domain-invariant Stereo Matching Networks](https://arxiv.org/abs/1911.13287)
   - Domain-invariant stereo matching networks
   - 领域不变的匹配网络

- 20191202 arXiv [Learning Generalizable Representations via Diverse Supervision](https://arxiv.org/abs/1911.12911)
   - Diverse supervision helps to learn generalizable representations

- 20191202 arXiv [Domain-Aware Dynamic Networks](https://arxiv.org/abs/1911.13237)
     - Edge devices adaptative computing
     - 边缘计算上的自适应计算

- 20191029 [Adversarial Feature Alignment: Avoid Catastrophic Forgetting in Incremental Task Lifelong Learning](https://arxiv.org/abs/1910.10986)
  	- Avoid catastrophic forgeeting in incremental task lifelong learning
  	- 在终身学习中避免灾难遗忘

- 20191029 [Reducing Domain Gap via Style-Agnostic Networks](https://arxiv.org/abs/1910.11645)
  	- Use style-agnostic networks to avoid domain gap
  	- 通过风格无关的网络来避免领域的gap

- 20191015 arXiv [The Visual Task Adaptation Benchmark](https://arxiv.org/abs/1910.04867)
  	- A new large benchmark for visual adaptation tasks by Google
  	- Google提出的一个巨大的视觉迁移任务数据集

- 20191011 arXiv [Estimating Transfer Entropy via Copula Entropy](https://arxiv.org/abs/1910.04375)
  	- Evaluate the transfer entopy via copula entropy
  	- 评估迁移熵

- 20191011 arXiv [Learning to Remember from a Multi-Task Teacher](https://arxiv.org/abs/1910.04650)
  	- Dealing with the catastrophic forgetting during sequential learning
  	- 在序列学习时处理灾难遗忘

- 20191008 arXiv [DIVA: Domain Invariant Variational Autoencoders](https://arxiv.org/abs/1905.10427)
  	- Domain invariant variational autoencoders
  	- 领域不变的变分自编码器

- 20190821 arXiv [Transfer Learning-Based Label Proportions Method with Data of Uncertainty](https://arxiv.org/abs/1908.06603)
  	- Transfer learning with source and target having uncertainty
  	- 当source和target都有不确定label时进行迁移

- 20190806 KDD-19 [Relation Extraction via Domain-aware Transfer Learning](https://dl.acm.org/citation.cfm?id=3330890)
    - Relation extraction using transfer learning for knowledge base construction
    - 利用迁移学习进行关系抽取

- 20190703 arXiv [Inferred successor maps for better transfer learning](https://arxiv.org/abs/1906.07663)
  	- Inferred successor maps for better transfer learning

- 20190626 arXiv [Transfer of Machine Learning Fairness across Domains](https://arxiv.org/abs/1906.09688)
  	- Transfer of machine learning fairness across domains
  	- 机器学习的公平性的迁移

- 20190531 IJCAI-19 [Adversarial Imitation Learning from Incomplete Demonstrations](https://arxiv.org/abs/1905.12310)
  	- Adversarial imitation learning from imcomplete demonstrations
  	- 基于不完整实例的对抗模仿学习

- 20190517 arXiv [Budget-Aware Adapters for Multi-Domain Learning](https://arxiv.org/abs/1905.06242)
    - Budget-Aware Adapters for Multi-Domain Learning

- 20190409 arXiv [Improving Image Classification Robustness through Selective CNN-Filters Fine-Tuning](https://arxiv.org/abs/1904.03949)
    - Improving Image Classification Robustness through Selective CNN-Filters Fine-Tuning
    - 通过可选择的CNN滤波器进行图像分类的fine-tuning

- 20190401 arXiv [Distilling Task-Specific Knowledge from BERT into Simple Neural Networks](https://arxiv.org/abs/1903.12136)
    - Distill knowledge from BERT to simple neural networks
    - 从BERT模型中迁移知识到简单网络中

- 20190305 arXiv [Let's Transfer Transformations of Shared Semantic Representations](https://arxiv.org/abs/1903.00793)
    - Transfer transformations from shared semantic representations
    - 从共享的语义表示中进行特征迁移

- 20190301 SysML-19 [FixyNN: Efficient Hardware for Mobile Computer Vision via Transfer Learning](https://arxiv.org/abs/1902.11128)
    - An efficient hardware for mobile computer vision applications using transfer learning
    - 提出一个高效的用于移动计算机视觉应用的硬件

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

## 1.2. Domain Adaptation (领域自适应)

Including domain adaptation and partial domain adaptation.

### 1.2.1. Traditional Methods (传统迁移方法)

- 20200615 [Deep Transfer Learning with Ridge Regression](https://arxiv.org/abs/2006.06791)

- 20200324 [Domain Adaptation by Class Centroid Matching and Local Manifold Self-Learning](https://arxiv.org/abs/2003.09391)
  	- Domain adaptation by class centroid matching and local manifold self-learning
  	- 集合了聚类、中心匹配，及自学习的DA

- 20191204 arXiv [Transferability versus Discriminability: Joint Probability Distribution Adaptation (JPDA)](https://arxiv.org/abs/1912.00320)
     - Joint adaptation with different weights
     - 不同权重的联合概率适配

- 20191125 AAAI-20 [Unsupervised Domain Adaptation via Structured Prediction Based Selective Pseudo-Labeling](https://arxiv.org/abs/1911.07982)
  	- DA with selective pseudo label
  	- 结构化和选择性的伪标签用于DA

- 20190703 arXiv [Domain Adaptation via Low-Rank Basis Approximation](https://arxiv.org/abs/1907.01343)
  	- Domain adaptation with low-rank basis approximation
  	- 低秩分解进行domain adaptation

- 20190508 IJCNN-19 [Unsupervised Domain Adaptation using Graph Transduction Games](https://arxiv.org/abs/1905.02036)
  	- Domain adaptation using graph transduction games
  	- 用图转换博弈进行domain adaptation

- 20190403 ICME-19 [Easy Transfer Learning By Exploiting Intra-domain Structures](https://arxiv.org/abs/1904.01376) [Code](http://transferlearning.xyz/code/traditional/EasyTL)
    - An easy transfer learning approach with good performance
    - 一个非常简单但效果很好的迁移方法

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

### 1.2.2. Deep / Adversarial Methods (深度/对抗迁移方法)

- 20200529 TNNLS [Deep Subdomain Adaptation Network for Image Classification](https://github.com/easezyc/deep-transfer-learning/tree/master/UDA/pytorch1.0/DSAN)
  	- A fine-grained adaptation method with LMMD, which is very simple and effective
  	- 一种细粒度自适应的方法，使用LMMD进行对齐，该方法非常简单有效

- 20200420 arXiv [One-vs-Rest Network-based Deep Probability Model for Open Set Recognition](https://arxiv.org/abs/2004.08067)
  	- One-vs-rest deep model for open set recognition
  	- 用于开放集的识别的深度网络

- 20200414 ICLR-20 [Gradient as features for deep representation learning](https://openreview.net/pdf?id=BkeoaeHKDS)
  	- Gradients as features for deep representation learning on pretrained models
  	- 在预训练模型基础上，将梯度作为额外的feature，提高学习表现

- 20200414 ICLR-20 [Domain adaptive multi-branch networks](https://openreview.net/forum?id=rJxycxHKDS)
  	- A domain adaptation framework using a multi-branch cascade structure
  	- 一个用了多层级联、多分支结构的DA框架

- 20200405 CVPR-20 [Towards Discriminability and Diversity: Batch Nuclear-norm Maximization under Label Insufficient Situations](https://arxiv.org/abs/2003.12237)
  	- A simple regularization-based adaptation method
  	- 一个非常简单的基于能量最小化的adaptation方法

- 20200210 AAAI-20 [Bi-Directional Generation for Unsupervised Domain Adaptation](https://arxiv.org/abs/2002.04869)
  	- Bidirectional GANs for domain adaptation
  	- 双向的GAN用来做DA

- 20191202 PR-19 [Correlation-aware Adversarial Domain Adaptation and Generalization](https://arxiv.org/abs/1911.12983)
     - CORAL and adversarial for adaptation and generalization
     - 基于CORAL和对抗网络的DA和DG

- 20191201 BMVC-19 [Domain Adaptation for Object Detection via Style Consistency](https://arxiv.org/abs/1911.10033)
     - Use style consistency for domain adaptation
     - 通过结构一致性来进行domain adaptation

- 20191124 AAAI-20 [Knowledge Graph Transfer Network for Few-Shot Recognition](https://arxiv.org/abs/1911.09579)
  	- GNN for semantic transfer for few-shot learning
  	- 用GNN进行类别的语义迁移用于few-shot learning

- 20191124 arXiv [Improving Unsupervised Domain Adaptation with Variational Information Bottleneck](https://arxiv.org/abs/1911.09310)
  	- Information bottleneck for unsupervised da
  	- 用了信息瓶颈来进行DA

- 20191124 AAAI-20 (AdaFilter: Adaptive Filter Fine-tuning for Deep Transfer Learning)(https://arxiv.org/abs/1911.09659)
  	- Adaptively determine which layer to transfer or finetune
  	- 自适应地决定迁移哪个层或微调哪个层

- 20191113 arXiv [Knowledge Distillation for Incremental Learning in Semantic Segmentation](https://arxiv.org/abs/1911.03462)
  	- Knowledge distillation for incremental learning in semantic segmentation
  	- 在语义分割问题中针对增量学习进行知识蒸馏

- 20191111 NIPS-19 [PointDAN: A Multi-Scale 3D Domain Adaption Network for Point Cloud Representation](https://arxiv.org/abs/1911.02744)
  	- Multi-scale 3D DA network for point cloud representation

- 20191111 CCIA-19 [Feature discriminativity estimation in CNNs for transfer learning](https://arxiv.org/abs/1911.03332)
  	- Feature discriminativity estimation in CNN for TL

- 20191012 ICCV-19 [Drop to Adapt: Learning Discriminative Features for Unsupervised Domain Adaptation](https://arxiv.org/abs/1910.05562)
	- Drop to Adapt: Learning Discriminative Features for Unsupervised Domain Adaptation
	- 直接適應：學習非監督域自適應的判別功能

- 20191015 arXiv [Deep Kernel Transfer in Gaussian Processes for Few-shot Learning](https://arxiv.org/abs/1910.05199)
  	- Deep kernel transfer learing in Gaussian process
  	- 高斯过程中的深度迁移学习

- 20191008 EMNLP-19 workshop [Domain Differential Adaptation for Neural Machine Translation](https://arxiv.org/abs/1910.02555)
  	- Embrace the difference between domains for adaptation
  	- 拥抱domain的不同，并利用这些不同帮助adaptation

- 20191008 BMVC-19 [Multi-Weight Partial Domain Adaptation](https://bmvc2019.org/wp-content/uploads/papers/0406-paper.pdf)
  	- Class and sample weight contribution for partial domain adaptation
  	- 同时考虑类别和样本比重用于部分迁移学习

- 20190813 ICCV-19 oral [UM-Adapt: Unsupervised Multi-Task Adaptation Using Adversarial Cross-Task Distillation](https://arxiv.org/abs/1908.03884)
  	- A unified framework for domain adaptation
  	- 一个统一的用于domain adaptation的框架

- 20190809 arXiv [Multi-Purposing Domain Adaptation Discriminators for Pseudo Labeling Confidence](https://arxiv.org/abs/1907.07802)
  	- Improve pseudo label confidence using multi-purposing DA
  	- 用多目标DA提高伪标签准确率

- 20190809 arXiv [Semi-supervised representation learning via dual autoencoders for domain adaptation](https://arxiv.org/abs/1908.01342)
  	- Semi-supervised learning via autoencoders
  	- 半监督autoencoder用于DA

- 20190809 arXiv [Mind2Mind : transfer learning for GANs](https://arxiv.org/abs/1906.11613)
  	- Transfer learning using GANs
  	- 用GAN进行迁移学习

- 20190809 arXiv [Self-supervised Domain Adaptation for Computer Vision Tasks](https://arxiv.org/abs/1907.10915)
  	- Self-supervised DA
  	- 自监督DA

- 20190809 arXiv [Hidden Covariate Shift: A Minimal Assumption For Domain Adaptation](https://arxiv.org/abs/1907.12299)
  	- Hidden covariate shift 
  	- 一种新的DA假设

- 20190809 PR-19 [Cross-domain Network Representations](https://arxiv.org/abs/1908.00205)
  - Cross-domain network representation learning
  - 跨领域网络表达学习

- 20190809 ICCV-19 [Larger Norm More Transferable: An Adaptive Feature Norm Approach for Unsupervised Domain Adaptation](https://arxiv.org/abs/1811.07456)
  - Adaptive Feature Norm Approach for Unsupervised Domain Adaptation
  - 自适应的特征归一化用于DA

- 20190731 MICCAI-19 Unsupervised Domain Adaptation via Disentangled Representations: Application to Cross-Modality Liver Segmentation
  	- Disentangled representations for unsupervised domain adaptation
    - 基于解耦表征的无监督领域自适应

- 20190719 arXiv [Agile Domain Adaptation](https://arxiv.org/abs/1907.04978)
  	- Domain adaptation by considering the difficulty in classification
  	- 通过考虑不同样本分离的难度进行domain adaptation

- 20190718 arXiv [Measuring the Transferability of Adversarial Examples](https://arxiv.org/abs/1907.06291)
  	- Measure the transferability of adversarial examples
  	- 度量对抗样本的可迁移性

- 20190604 IJCAI-19 [DANE: Domain Adaptive Network Embedding](https://arxiv.org/abs/1906.00684)
  	- Transfered network embeddings for different networks
  	- 不同网络表达的迁移

- 20190604 arXiv [Learning to Transfer: Unsupervised Meta Domain Translation](https://arxiv.org/abs/1906.00181)
  	- Unsupervised meta domain translation
  	- 无监督领域翻译

- 20190530 arXiv [Learning Bregman Divergences](https://arxiv.org/abs/1905.11545)
  	- Learning Bregman divergence
  	- 学习Bregman差异

- 20190530 arXiv [Adversarial Domain Adaptation Being Aware of Class Relationships](https://arxiv.org/abs/1905.11931)
  	- Using class relationship for adversarial domain adaptation
  	- 利用类别关系进行对抗的domain adaptaition

- 20190530 arXiv [Cross-Domain Transferability of Adversarial Perturbations](https://arxiv.org/abs/1905.11736)
  	- Cross-Domain Transferability of Adversarial Perturbations

- 20190525 PAMI-19 [Learning More Universal Representations for Transfer-Learning](https://ieeexplore.ieee.org/abstract/document/8703078)
  	- Learning more universal representations for transfer learning
  	- 对迁移学习设计2种方式学到更通用的表达

- 20190517 ICML-19 [Learning What and Where to Transfer](https://arxiv.org/abs/1905.05901)
  	- Learning what and where to transfer in deep networks
  	- 学习深度网络从何处迁移

- 20190517 ICML-19 [Zero-Shot Voice Style Transfer with Only Autoencoder Loss](https://arxiv.org/abs/1905.05879)
  	- Zero-shot voice style transfer with only autoencoder loss
  	- 零次声音迁移

- 20190515 CVPR-19 [Diversify and Match: A Domain Adaptive Representation Learning Paradigm for Object Detection](https://arxiv.org/abs/1905.05396)
  	- Domain adaptation for object detection
  	- 领域自适应用于物体检测
  
- 20190507 NAACL-HLT 19 [Transfer of Adversarial Robustness Between Perturbation Types](https://arxiv.org/abs/1905.01034)
  	- Transfer of Adversarial Robustness Between Perturbation Types

- 20190416 arXiv [ACE: Adapting to Changing Environments for Semantic Segmentation](https://arxiv.org/abs/1904.06268)
  	- Propose a new method that can adapt to new environments
  	- 提出一种可以适配不同环境的方法

- 20190416 arXiv [Incremental multi-domain learning with network latent tensor factorization](https://arxiv.org/abs/1904.06345)
  	- Incremental multi-domain learning with network latent tensor factorization
  	- 网络隐性tensor分解应用于多领域增量学习

- 20190415 PAKDD-19 [Parameter Transfer Unit for Deep Neural Networks](https://link.springer.com/chapter/10.1007/978-3-030-16145-3_7)
  	- Propose a parameter transfer unit for DNN
  	- 对深度网络提出参数迁移单元

- 20190412 PAMI-19 [Beyond Sharing Weights for Deep Domain Adaptation](https://ieeexplore.ieee.org/abstract/document/8310033)
  	- Domain adaptation by not sharing weights
  	- 通过不共享权重来进行domain adaptation

- 20190405 IJCNN-19 [Accelerating Deep Unsupervised Domain Adaptation with Transfer Channel Pruning](https://arxiv.org/abs/1904.02654)
    - The first work to accelerate transfer learning
    - 第一个加速迁移学习的工作

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

## 1.3. Domain Generalization

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

## 1.4. Multi-source Transfer Learning (多源迁移学习)

- 20200427 [TriGAN: Image-to-Image Translation for Multi-Source Domain Adaptation](https://arxiv.org/abs/2004.08769)
  	- A cycle-gan style multi-source DA
  	- 类似于cyclegan的多源领域适应

- 20190902 AAAI-19 [Aligning Domain-Specific Distribution and Classifier for Cross-Domain Classification from Multiple Sources](https://www.aaai.org/ojs/index.php/AAAI/article/download/4551/4429)
  	- Multi-source domain adaptation using both features and classifier adaptation
  	- 利用特征和分类器同时适配进行多源迁移

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

## 1.5. Heterogeneous Transfer Learning (异构迁移学习)

- 20190717 AAAI [Heterogeneous Transfer Learning via Deep Matrix Completion with Adversarial Kernel Embedding](https://144.208.67.177/ojs/index.php/AAAI/article/view/4880)
	- Transfer Learning via Deep Matrix Completion with Adversarial Kernel Embedding
	- 异构迁移学习中用对抗核嵌入的深度矩阵

- 20190829 ACMMM-19 [Heterogeneous Domain Adaptation via Soft Transfer Network](https://arxiv.org/abs/1908.10552)
  	- Soft-mmd loss in heterogeneous domain adaptation
  	- 异构迁移学习中用soft-mmd loss

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

## 1.6. Online Transfer Learning (在线迁移学习)

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

## 1.7. Zero-shot / Few-shot Learning

- 20200608 ICML-20 [Few-Shot Learning as Domain Adaptation: Algorithm and Analysis](https://arxiv.org/abs/2002.02050)
    - Using domain adaptation to solve the few-shot learning

- 20200408 ICLR-20 [A Baseline for Few-Shot Image Classification](https://openreview.net/forum?id=rylXBkrYDS)
      - A simple finetune+entropy minimization approach with strong baseline
      - 一个微调+最小化熵的小样本学习方法，结果很强

- 20200405 ICCV-19 [Variational few-shot learning](http://openaccess.thecvf.com/content_ICCV_2019/html/Zhang_Variational_Few-Shot_Learning_ICCV_2019_paper.html)
	- Variational few-shot learning
	- 变分小样本学习

- 20200405 ICLR-20 [A baseline for few-shot image classification](https://openreview.net/forum?id=rylXBkrYDS&noteId=rylXBkrYDS)
	- A simple but powerful baseline for few-shot image classification
	- 一个简单但是很有效的few-shot baseline

- 20200324 IEEE TNNLS [Few-Shot Learning with Geometric Constraints](https://arxiv.org/abs/2003.09151)
  	- Few-shot learning with geometric constraints
  	- 用了一些几何约束进行小样本学习

- 20190813 arXiv [Domain-Specific Embedding Network for Zero-Shot Recognition](https://arxiv.org/abs/1908.04174)
  	- Domain-specific embedding network for zero-shot learning
  	- 领域自适应的zero-shot learning

- 20190401 TIp-19 [Few-Shot Deep Adversarial Learning for Video-based Person Re-identification](https://arxiv.org/abs/1903.12395)
    - Few-shot deep adversarial learning
    - Few-shot对抗学习

- 20190305 arXiv [Zero-Shot Task Transfer](https://arxiv.org/abs/1903.01092)
    - Zero-shot task transfer
    - Zero-shot任务迁移学习

- 20190221 arXiv [Adaptive Cross-Modal Few-Shot Learning](https://arxiv.org/abs/1902.07104)
    - Adaptive cross-modal few-shot learning
    - 跨模态的few-shot

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

### 1.7.1. Zero-shot Learning based on Data Synthesis (基于样本生成的零样本学习)
[详细介绍](https://github.com/PatrickZH/Zero-shot-Learning) <br>

- 20191204 arXiv [MetAdapt: Meta-Learned Task-Adaptive Architecture for Few-Shot Classification](https://arxiv.org/abs/1912.00412)
     - Task adaptive structure for few-shot learning
     - 目标自适应的结构用于小样本学习

- 20190409 ICLR-19 [A Closer Look at Few-shot Classification](https://arxiv.org/abs/1904.04232)
    - Give some important conclusions on few-shot classification
    - 在few-shot上给了一些有用的结论

- 20190401 IJCNN-19 [Zero-shot Image Recognition Using Relational Matching, Adaptation and Calibration](https://arxiv.org/abs/1903.11701)
    - Zero-shot image recognition
    - 零次学习的图像识别

- 20190301 NeurIPS-18 workshp [One-Shot Federated Learning](https://arxiv.org/abs/1902.11175)
    - One-shot federated learning

- 20171022 ICCVW-17 [Zero-shot learning posed as a missing data problem](http://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w38/Zhao_Zero-Shot_Learning_Posed_ICCV_2017_paper.pdf)
    - 算法首先学习 semantic embeddings 的结构性知识，利用学习到的知识和已知类的 image features 合成未知类的 image features。再利用无标记的未知类数据对合成数据进行修正。 算法假设未知类数据呈混合高斯分布，用 GMM-EM 算法进行无监督修正。
    
- 20180516 arXiv-18 [A Large-scale Attribute Dataset for Zero-shot Learning](https://arxiv.org/pdf/1804.04314v2.pdf)
    - 传统 ZSL 数据集（如 AwA, CUB）存在规模小，属性标注不丰富等问题。本文提出一个新的属性数据集 LAD 用于测试零样本学习算法。新数据集包含 230 类， 78,017 张图片，标注了 359 种属性。基于此数据集举办了 AI Challenger 零样本学习竞赛。 110+ 支来自海内外的参赛队伍提交了成绩。
    
- 20180710 ICML-18 [MSplit LBI: Realizing Feature Selection and Dense Estimation Simultaneously in Few-shot and Zero-shot Learning](https://arxiv.org/pdf/1806.04360.pdf)
    - 针对 L1 （欠拟合） 和 L2 （无特征选择、有偏） 正则项存在的问题，提出 MSplit LBI 用于同时实现特征选择和密集估计。在 Few-shot Learning 和 Zero-shot Learning 两个问题上进行了实验。实验表明 MSplit LBI 由优于 L1 和 L2。针对 ZSL 进行了特征可视化实验。
    
- 20190108 WACV-19 [Zero-shot Learning via Recurrent Knowledge Transfer](https://drive.google.com/open?id=1cUsQWX80zeCxTyVSCcYlqEWZP-Hq0KzR)
    - 基于样本合成的零样本学习算法通常将 semantic embeddings 的知识迁移到 image features 以实现 ZSL。然而，这种 training 和 testing space 的不一致，会导致这种迁移失效。因此，本文提出 Space Shift Problem，并针对此问题，提出一种（在 image feature space 和 semantic embedding space 之间）递归传递知识的解决方案。

- - -

## 1.8. Deep Transfer Learning (深度迁移学习)

### 1.8.1. Non-Adversarial Transfer Learning (非对抗深度迁移)

- 20191222 arXiv [Dreaming to Distill: Data-free Knowledge Transfer via DeepInversion](https://arxiv.org/abs/1912.08795)
   - Generate data without priors for transfer learning based on deep dream
   - 只用网络架构不用原来数据，生成新数据用于迁移

- 20191222 AAAI-20 [Improved Knowledge Distillation via Teacher Assistant](https://arxiv.org/abs/1902.03393)
    - Teacher assistant helps knowledge distillation

- 20191204 AAAI-20 [Online Knowledge Distillation with Diverse Peers](https://arxiv.org/abs/1912.00350)
    - Online Knowledge Distillation with Diverse Peers
- 20191201 arXiv [A Unified Framework for Lifelong Learning in Deep Neural Networks](https://arxiv.org/abs/1911.09704)
     - A unified framework for life-long learing in DNN

- 20191201 arXiv [ReMixMatch: Semi-Supervised Learning with Distribution Alignment and Augmentation Anchoring](https://arxiv.org/abs/1911.09785)
     - Semi-Supervised Learning with Distribution Alignment and Augmentation Anchoring

- 20191119 ICDM-19 [Towards Making Deep Transfer Learning Never Hurt](https://arxiv.org/abs/1911.07489)
  	- Towards making deep transfer learning never hurt
  	- 通过正则避免负迁移

- 20191119 NIPS-19 workshop [Collaborative Unsupervised Domain Adaptation for Medical Image Diagnosis](https://arxiv.org/abs/1911.07293)
  	- Ensemble DA using noise labels
  	- 在ensemble中出现noise label时如何处理

- 20191119 NIPS-19 [Transferable Normalization: Towards Improving Transferability of Deep Neural Networks](http://scholar.google.com/scholar_url?url=https://papers.nips.cc/paper/8470-transferable-normalization-towards-improving-transferability-of-deep-neural-networks.pdf&hl=en&sa=X&d=9221290800687054760&scisig=AAGBfm0aM7wBD0WhQjvvfe_vxKxIF8STLw&nossl=1&oi=scholaralrt&hist=hBZ_tKsAAAAJ:17338659929613568812:AAGBfm3MBWiQSh14vpyLyxgl0RCrplQyWg)
  	- Transfer normalization

- 20191029 KBS [Semi-supervised representation learning via dual autoencoders for domain adaptation](https://arxiv.org/abs/1908.01342)
  	- Semi-supervised domain adaptation with autoencoders
  	- 用自动编码器进行半监督的DA

- 20190929 NeurIPS-19 [Deep Model Transferability from Attribution Maps](https://arxiv.org/abs/1909.11902)
  	- Using attribution map for network similarity
  	- 与cvpr18的taskmony类似，这次用了属性图的方式探索网络的相似性

- 20190926 arXiv [Learning a Domain-Invariant Embedding for Unsupervised Domain Adaptation Using Class-Conditioned Distribution Alignment](https://arxiv.org/abs/1907.02271)
  	- Use class-conditional DA for domain adaptation
  	- 使用类条件对齐进行domain adaptation

- 20190926 arXiv [A Deep Learning-Based Approach for Measuring the Domain Similarity of Persian Texts](https://arxiv.org/abs/1909.09690)
  	- Deep learning based domain similarity learning
  	- 利用深度学习进行领域相似度的学习

- 20190926 arXiv [Transfer Learning across Languages from Someone Else's NMT Model](https://arxiv.org/abs/1909.10955)
  	- Transfer learning across languages from NMT pretrained model
  	- 利用预训练好的NMT模型进行迁移学习

- 20190926 arXiv [FEED: Feature-level Ensemble for Knowledge Distillation](https://arxiv.org/abs/1909.10754)
  	- Feature-level knowledge distillation
  	- 特征层面的知识蒸馏

- 20190926 ICCV-19 [Self-similarity Grouping: A Simple Unsupervised Cross Domain Adaptation Approach for Person Re-identification](https://arxiv.org/abs/1811.10144)
  	- A simple approach for domain adaptation
  	- 一个很简单的DA方法

- 20190910 BMVC-19 [Curriculum based Dropout Discriminator for Domain Adaptation](https://arxiv.org/abs/1907.10628)
  	- Curriculum dropout for domain adaptation
  	- 基于课程学习的dropout用于DA

- 20190909 IJCAI-FML-19 [FedHealth: A Federated Transfer Learning Framework for Wearable Healthcare](http://jd92.wang/assets/files/a15_ijcai19.pdf)
  	- The first work on federated transfer learning for wearable healthcare
  	- 第一个将联邦迁移学习用于可穿戴健康监护的工作

- 20190909 PAMI [Inferring Latent Domains for Unsupervised Deep Domain Adaptation](https://ieeexplore.ieee.org/abstract/document/8792192)
  	- Inferring latent domains for unsupervised deep domain
  	- 在深度迁移学习中推断隐含领域

- 20190729 ICCV workshop [Multi-level Domain Adaptive learning for Cross-Domain Detection](https://arxiv.org/abs/1907.11484)
  	- Multi-level domain adaptation for cross-domain Detection
  	- 多层次的domain adaptation

- 20190626 IJCAI-19 [Bayesian Uncertainty Matching for Unsupervised Domain Adaptation](https://arxiv.org/abs/1906.09693)
  	- Bayesian uncertainty matching for da
  	- 贝叶斯网络用于da

- 20190419 CVPR-19 [DDLSTM: Dual-Domain LSTM for Cross-Dataset Action Recognition](https://arxiv.org/abs/1904.08634)
  	- Dual-Domain LSTM for Cross-Dataset Action Recognition
  	- 跨数据集的动作识别

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
	- 延续了之前的DAN工作，这次考虑联合适配
- - -

### 1.8.2. Deep Adversarial Transfer Learning (对抗迁移学习)

- 20191214 arXiv [Learning Domain Adaptive Features with Unlabeled Domain Bridges](https://arxiv.org/abs/1912.05004)
    - Learning domain adaptive features with unlabeled CycleGAN

- 20191214 AAAI-20 [Adversarial Domain Adaptation with Domain Mixup](https://arxiv.org/abs/1912.01805)
    - Domain adaptation with data mixup

- 20190916 arXiv [Compound Domain Adaptation in an Open World](https://arxiv.org/abs/1909.03403)
  	- Domain adaptation using the target domain knowledge
  	- 使用目标域的知识来进行domain adaptation

- 20101008 ICCV-19 [Enhancing Adversarial Example Transferability with an Intermediate Level Attack](https://arxiv.org/abs/1907.10823)
  	- Enhancing adversarial examples transerability
  	- 增强对抗样本的可迁移性

- 20190531 arXiv [Adaptive Deep Kernel Learning](https://arxiv.org/abs/1905.12131)
  	- Adaptive deep kernel learning
  	- 自适应深度核学习

- 20190531 arXiv [Multi-task Learning in Deep Gaussian Processes with Multi-kernel Layers](https://arxiv.org/abs/1905.12407)
  	- Multi-task learning in deep Gaussian process
  	- 深度高斯过程中的多任务学习

- 20190531 arXiv [Image Alignment in Unseen Domains via Domain Deep Generalization](https://arxiv.org/abs/1905.12028)
  	- Deep domain generalization for image alignment
  	- 深度领域泛化用于图像对齐

- 20190408 arXiv [DeceptionNet: Network-Driven Domain Randomization](https://arxiv.org/abs/1904.02750)
    - Using only source data for domain randomization
    - 仅利用源域数据进行domain randomization

- 20190220 arXiv [Fully-Featured Attribute Transfer](https://arxiv.org/abs/1902.06258)
    - Fully-featured image attribute transfer
	- 图像特征迁移

- 20190220 arXiv [Unsupervised Domain Adaptation using Deep Networks with Cross-Grafted Stacks](https://arxiv.org/abs/1902.06328)
    - Domain adaptation using deep learning with cross-grafted stacks
    - 用跨领域嫁接栈进行domain adaptation

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

## 1.9. Multi-task Learning (多任务学习)

- 20191202 arXiv [AdaShare: Learning What To Share For Efficient Deep Multi-Task Learning](https://arxiv.org/abs/1911.12423)
   - Learning what to share for multi-task learning
   - 对多任务学习如何share

- 20191125 AAAI-20 [Adaptive Activation Network and Functional Regularization for Efficient and Flexible Deep Multi-Task Learning](https://arxiv.org/abs/1911.08065)
  	- Adaptive activation network for deep multi-task learning
  	- 自适应的激活网络用于深度多任务学习

- 20191015 arXiv [Gumbel-Matrix Routing for Flexible Multi-task Learning](https://arxiv.org/abs/1910.04915)
  	- Effective method for flexible multi-task learning
  	- 一种很有效的方法用于多任务学习

- 20190718 arXiv [Task Selection Policies for Multitask Learning](https://arxiv.org/abs/1907.06214)
  	- Task selection in multitask learning
  	- 在多任务学习中的任务选择机制

- 20190509 FG-19 [Multi-task human analysis in still images: 2D/3D pose, depth map, and multi-part segmentation](https://arxiv.org/abs/1905.03003)
  	- Multi-task human analysis in still images
  	- 多任务人体静止图像分析

- 20190409 NAACL-19 [AutoSeM: Automatic Task Selection and Mixing in Multi-Task Learning](https://arxiv.org/abs/1904.04153)
    - Automatic Task Selection and Mixing in Multi-Task Learning
    - 多任务学习中自动任务选择和混淆

- 20190409 TNNLS-19 [Heterogeneous Multi-task Metric Learning across Multiple Domains](https://arxiv.org/abs/1904.04081)
    - Heterogeneous Multi-task Metric Learning across Multiple Domains
    - 在多个领域之间进行异构多任务度量学习

- 20190409 NeurIPS-18 [Synthesized Policies for Transfer and Adaptation across Tasks and Environments](https://arxiv.org/abs/1904.03276)
    - Transfer across tasks and environments
    - 通过任务和环境之间进行迁移

- 20190408 ICMR-19 [Learning Task Relatedness in Multi-Task Learning for Images in Context](https://arxiv.org/abs/1904.03011)
    - Using task relatedness in multi-task learning
    - 在多任务学习中学习任务之间的相关性

- 20190408 CVPR-19 [End-to-End Multi-Task Learning with Attention](https://arxiv.org/abs/1803.10704)
    - End-to-End Multi-Task Learning with Attention
    - 基于attention的端到端的多任务学习

- 20190401 arXiv [Many Task Learning with Task Routing](https://arxiv.org/abs/1903.12117)
    - From multi-task leanring to many-task learning
    - 许多任务同时学习

- 20190324 arXiv [A Principled Approach for Learning Task Similarity in Multitask Learning](https://arxiv.org/abs/1903.09109)
    - Provide some theoretical analysis of the similarity learning in multi-task learning
    - 为多任务学习中的相似度学习提供了一些理论分析

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

## 1.10. Transfer Reinforcement Learning (强化迁移学习)

- 20191214 arXiv [Does Knowledge Transfer Always Help to Learn a Better Policy?](https://arxiv.org/abs/1912.02986)
    - Transfer learning in reinforcement learning

- 20191212 AAAI-20 [Transfer value iteration networks](https://arxiv.org/abs/1911.05701)
    - Transferred value iteration networks

- 20190821 arXiv [Transfer in Deep Reinforcement Learning using Knowledge Graphs](https://arxiv.org/abs/1908.06556)
  	- Use knowledge graph to transfer in reinforcement learning
  	- 用知识图谱进行强化迁移

- 20190320 arXiv [Learning to Augment Synthetic Images for Sim2Real Policy Transfer](https://arxiv.org/abs/1903.07740)
    - Augment synthetic images for sim to real policy transfer
    - 学习对于策略迁移如何合成图像

- 20190305 arXiv [Sim-to-Real Transfer for Biped Locomotion]
    - Transfer learning for robot locomotion
    - 用迁移学习进行机器人定位

- 20190220 arXiv [DIViS: Domain Invariant Visual Servoing for Collision-Free Goal Reaching](https://arxiv.org/abs/1902.05947)
    - Transfer learning for robot reinforcement learning
    - 迁移学习用于机器人的强化学习目标搜寻

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

## 1.11. Transfer Metric Learning (迁移度量学习)

- 20190515 TNNLS-19 [A Distributed Approach towards Discriminative Distance Metric Learning](https://arxiv.org/abs/1905.05177)
  	- Discriminative distance metric learning
  	- 分布式度量学习

- 20190409 TNNLS-19 [Heterogeneous Multi-task Metric Learning across Multiple Domains](https://arxiv.org/abs/1904.04081)
    - Heterogeneous Multi-task Metric Learning across Multiple Domains
    - 在多个领域之间进行异构多任务度量学习

- 20190409 PAMI-19 [Transferring Knowledge Fragments for Learning Distance Metric from A Heterogeneous Domain](https://arxiv.org/abs/1904.04061)
    - Heterogeneous transfer metric learning by transferring fragments
    - 通过迁移知识片段来进行异构迁移度量学习

- 20190409 arXiv [Decomposition-Based Transfer Distance Metric Learning for Image Classification](https://arxiv.org/abs/1904.03846)
    - Transfer metric learning based on decomposition
    - 基于特征向量分解的迁移度量学习

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

## 1.12. Transitive Transfer Learning (传递迁移学习)

- 传递迁移学习的第一篇文章，来自杨强团队，发表在KDD-15上：[Transitive Transfer Learning](http://dl.acm.org/citation.cfm?id=2783295)

- AAAI-17 杨强团队最新的传递迁移学习：[Distant Domain Transfer Learning](http://www3.ntu.edu.sg/home/sinnopan/publications/[AAAI17]Distant%20Domain%20Transfer%20Learning.pdf)

- 20180819 LNCS-2018 [Distant Domain Adaptation for Text Classification](https://link.springer.com/chapter/10.1007/978-3-319-99365-2_5)
	-  Propose a selected algorithm for distant domain text classification
	- 提出一个用于远域的文本分类方法

- - -

## 1.13. Lifelong Learning (终身迁移学习)

- 20190912 NeurIPS-19 [Meta-Learning with Implicit Gradients](https://arxiv.org/abs/1909.04630)
  - Meta-learning with implicit gradients
  - 隐式梯度的元学习

- 20180323 arXiv 终身迁移学习与增量学习结合：[Incremental Learning-to-Learn with Statistical Guarantees](https://arxiv.org/abs/1803.08089)

- 20180111 arXiv 一种新的终身学习框架，与L2T的思路有一些类似 [Lifelong Learning for Sentiment Classification](https://arxiv.org/abs/1801.02808)

- - -

## 1.14. Negative Transfer (负迁移)

- 20181128 arXiv [Characterizing and Avoiding Negative Transfer](https://arxiv.org/abs/1811.09751)
	-  Analyzing and formalizing negative transfer, then propose a new method
	- 分析并形式化负迁移，进而提出自己的方法

- - -

## 1.15. Transfer Learning Applications (应用)

See [HERE](https://github.com/jindongwang/transferlearning/blob/master/doc/transfer_learning_application.md) for a full list of transfer learning applications.
