# 迁移学习 Transfer Learning  

<h1 align="center">
  <br>
  <img src="png/logo.jpg" alt="Overleaf" width="500">
</h1>

<h4 align="center">Everything about Transfer Learning.</h4>

<p align="center">
  <strong><a href="#0papers-论文">Papers</a></strong> •
  <strong><a href="#1introduction-and-tutorials-简介与教程">Tutorials</a></strong> •
  <a href="#2transfer-learning-areas-and-papers-研究领域与相关论文">Research areas</a> •
  <a href="#3theory-and-survey-理论与综述">Theory</a> •
  <a href="#3theory-and-survey-理论与综述">Survey</a> •
  <strong><a href="#4code-代码">Code</a></strong> •
  <strong><a href="#7datasets-and-benchmarks-数据集与评测结果">Dataset & benchmark</a></strong>
</p>
<p align="center">
<a href="#6transfer-learning-thesis-硕博士论文">Thesis</a> •
  <a href="#5transfer-learning-scholars-著名学者">Scholars</a> •
  <a href="#8transfer-learning-challenges-迁移学习比赛">Contests</a> •
  <a href="#applications-迁移学习应用">Applications</a> •
  <a href="#other-resources-其他资源">Others</a> •
  <a href="#contributing-欢迎参与贡献">Contributing</a>
</p>

```
@Misc{transferlearning.xyz,
howpublished = {\url{http://transferlearning.xyz}},   
title = {Everything about Transfer Learning and Domain Adapation},  
author = {Wang, Jindong and others}  
}  
```

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re) [![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT) [![LICENSE](https://img.shields.io/badge/license-Anti%20996-blue.svg)](https://github.com/996icu/996.ICU/blob/master/LICENSE) [![996.icu](https://img.shields.io/badge/link-996.icu-red.svg)](https://996.icu) More related repos：[Activity recognition](https://github.com/jindongwang/activityrecognition)｜[Machine learning](https://github.com/jindongwang/MachineLearning)

- - -

**NOTE:** You can directly open the code in [Gihub Codespaces](https://docs.github.com/en/codespaces/getting-started/quickstart#introduction) on the web to run them without downloading! Also, try [github.dev](https://github.dev/jindongwang/transferlearning).

## 0.Papers (论文)

[Awesome transfer learning papers (迁移学习文章汇总)](https://github.com/jindongwang/transferlearning/tree/master/doc/awesome_paper.md)

- [Paperweekly](http://www.paperweekly.site/collections/231/papers): A good website to recommend and read paper notes (一个推荐、分享论文的网站比较好，上面会持续整理相关的文章并分享阅读笔记)

**Latest papers**: (All papers are also put in [doc/awesome_papers.md](https://github.com/jindongwang/transferlearning/blob/master/doc/awesome_paper.md))

<details>
  <summary>Latest papers (2021-10-21)</summary>

  - WACV-21 [Domain Generalization through Audio-Visual Relative Norm Alignment in First Person Action Recognition](https://arxiv.org/abs/2110.10101)
    - Domain generalization by audio-visual alignment
    - 通过音频-视频对齐进行domain generalization

  - BMVC-21 [Dynamic Feature Alignment for Semi-supervised Domain Adaptation](https://arxiv.org/abs/2110.09641)
    - Dynamic feature alignment for semi-supervised DA
    - 动态特征对齐用于半监督DA

</details>

<details>
  <summary>Latest papers (2021-10-19)</summary>

  - NeurIPS-21 [FlexMatch: Boosting Semi-Supervised Learning with Curriculum Pseudo Labeling](https://arxiv.org/abs/2110.08263) [知乎解读](https://zhuanlan.zhihu.com/p/422930830) [code](https://github.com/TorchSSL/TorchSSL)
    - Curriculum pseudo label with a unified codebase TorchSSL
    - 半监督方法FlexMatch和统一算法库TorchSSL

</details>

<details>
  <summary>Latest papers (2021-10-14)</summary>

  - [Rethinking supervised pre-training for better downstream transferring](https://arxiv.org/abs/2110.06014)
    - Rethink better finetune
    - 重新思考预训练以便更好finetune

  - [Music Sentiment Transfer](https://arxiv.org/abs/2110.05765)
    - Music sentiment transfer learning
    - 迁移学习用于音乐sentiment

</details>

<details>
  <summary>Latest papers (2021-10-11)</summary>

  - NeurIPS-21 [Model Adaptation: Historical Contrastive Learning for Unsupervised Domain Adaptation without Source Data](http://arxiv.org/abs/2110.03374)
    - Source-free domain adaptation using constrastive learning
    - 无源域数据的DA，利用对比学习

  - [Understanding Domain Randomization for Sim-to-real Transfer](http://arxiv.org/abs/2110.03239)
    - Understanding domain randomizationfor sim-to-real transfer
    - 对强化学习中的sim-to-real transfer进行理论上的分析

  - [Dynamically Decoding Source Domain Knowledge For Unseen Domain Generalization](http://arxiv.org/abs/2110.03027)
    - Ensemble learning for domain generalization
    - 用集成学习进行domain generalization

  - [Scale Invariant Domain Generalization Image Recapture Detection](http://arxiv.org/abs/2110.03496)
    - Scale invariant domain generalizaiton
    - 尺度不变的domain generalization

</details>

<details>
  <summary>Latest papers (2021-09)</summary>

  - IEEE TIP-21 [Joint Clustering and Discriminative Feature Alignment for Unsupervised Domain Adaptation](https://ieeexplore.ieee.org/abstract/document/9535218)
    - Clustering and discriminative alignment for DA
    - 聚类与判定式对齐用于DA

  - IEEE TNNLS-21 [Entropy Minimization Versus Diversity Maximization for Domain Adaptation](https://ieeexplore.ieee.org/abstract/document/9537640)
    - Entropy minimization versus diversity max for DA
    - 熵最小化与diversity最大化

  - [Adversarial Domain Feature Adaptation for Bronchoscopic Depth Estimation](https://arxiv.org/abs/2109.11798)
    - Adversarial domain adaptation for bronchoscopic depth estimation
    - 用对抗领域自适应进行支气管镜的深度估计

  - EMNLP-21 [Few-Shot Intent Detection via Contrastive Pre-Training and Fine-Tuning](https://arxiv.org/abs/2109.06349)
    - Few-shot intent detection using pretrain and finetune
    - 用迁移学习进行少样本意图检测

  - EMNLP-21 [Non-Parametric Unsupervised Domain Adaptation for Neural Machine Translation](https://arxiv.org/abs/2109.06604)
    - UDA for machine translation
    - 用领域自适应进行机器翻译

  - [KroneckerBERT: Learning Kronecker Decomposition for Pre-trained Language Models via Knowledge Distillation](https://arxiv.org/abs/2109.06243)
    - Using Kronecker decomposition and knowledge distillation for pre-trained language models compression
    - 用Kronecker分解和知识蒸馏来进行语言模型的压缩

  - [Cross-Region Domain Adaptation for Class-level Alignment](https://arxiv.org/abs/2109.06422)
    - Cross-region domain adaptation for class-level alignment
    - 跨区域的领域自适应用于类级别的对齐

  - [Unsupervised domain adaptation for cross-modality liver segmentation via joint adversarial learning and self-learning](https://arxiv.org/abs/2109.05664)
    - Domain adaptation for cross-modality liver segmentation
    - 使用domain adaptation进行肝脏的跨模态分割

  - [CDTrans: Cross-domain Transformer for Unsupervised Domain Adaptation](https://arxiv.org/abs/2109.06165)
    - Cross-domain transformer for domain adaptation
    - 基于transformer进行domain adaptation

  - ICCV-21 [Shape-Biased Domain Generalization via Shock Graph Embeddings](https://arxiv.org/abs/2109.05671)
    - Domain generalization based on shape information
    - 基于形状进行domain generalization

  - [Domain and Content Adaptive Convolution for Domain Generalization in Medical Image Segmentation](https://arxiv.org/abs/2109.05676)
    - Domain generalization for medical image segmentation
    - 领域泛化用于医学图像分割

  - [Class-conditioned Domain Generalization via Wasserstein Distributional Robust Optimization](https://arxiv.org/abs/2109.03676)
    - Domain generalization with wasserstein DRO
    - 使用Wasserstein DRO进行domain generalization

  - [FedZKT: Zero-Shot Knowledge Transfer towards Heterogeneous On-Device Models in Federated Learning](https://arxiv.org/abs/2109.03775)
    - Zero-shot transfer in heterogeneous federated learning
    - 零次迁移用于联邦学习

  - [Fishr: Invariant Gradient Variances for Out-of-distribution Generalization](https://arxiv.org/abs/2109.02934)
    - Invariant gradient variances for OOD generalization
    - 不变梯度方差，用于OOD

  - [How Does Adversarial Fine-Tuning Benefit BERT?](https://arxiv.org/abs/2108.13602)
    - Examine how does adversarial fine-tuning help BERT
    - 探索对抗性finetune如何帮助BERT

  - [Contrastive Domain Adaptation for Question Answering using Limited Text Corpora](https://arxiv.org/abs/2108.13854)
    - Contrastive domain adaptation for QA
    - QA任务中应用对比domain adaptation

</details>

<details>
  <summary>Latest papers (2021-08)</summary>

  - [Robust Ensembling Network for Unsupervised Domain Adaptation](https://arxiv.org/abs/2108.09473)
    - Ensembling network for domain adaptation
    - 集成嵌入网络用于domain adaptation

  - [Federated Multi-Task Learning under a Mixture of Distributions](https://arxiv.org/abs/2108.10252)
    - Federated multi-task learning
    - 联邦多任务学习

  - [Fine-tuning is Fine in Federated Learning](http://arxiv.org/abs/2108.07313)
    - Finetuning in federated learning
    - 在联邦学习中进行finetune

  - [Federated Multi-Target Domain Adaptation](http://arxiv.org/abs/2108.07792)
    - Federated multi-target DA
    - 联邦学习场景下的多目标DA

  - [Learning Transferable Parameters for Unsupervised Domain Adaptation](https://arxiv.org/abs/2108.06129)
    - Learning partial transfer parameters for DA
    - 学习适用于迁移部分的参数做UDA任务

  - MICCAI-21 [A Systematic Benchmarking Analysis of Transfer Learning for Medical Image Analysis](https://arxiv.org/abs/2108.05930)
    - A benchmark of transfer learning for medical image
    - 一个详细的迁移学习用于医学图像的benchmark

  - [TVT: Transferable Vision Transformer for Unsupervised Domain Adaptation](https://arxiv.org/abs/2108.05988)
    - Vision transformer for domain adaptation
    - 用视觉transformer进行DA

  - CIKM-21 [AdaRNN: Adaptive Learning and Forecasting of Time Series](https://arxiv.org/abs/2108.04443) [Code](https://github.com/jindongwang/transferlearning/tree/master/code/deep/adarnn) [知乎文章](https://zhuanlan.zhihu.com/p/398036372) [Video](https://www.bilibili.com/video/BV1Gh411B7rj/)
    - A new perspective to using transfer learning for time series analysis
    - 一种新的建模时间序列的迁移学习视角

  - TKDE-21 [Unsupervised Deep Anomaly Detection for Multi-Sensor Time-Series Signals](https://arxiv.org/abs/2107.12626)
    - Anomaly detection using semi-supervised and transfer learning
    - 半监督学习用于无监督异常检测

  - SemDIAL-21 [Generating Personalized Dialogue via Multi-Task Meta-Learning](https://arxiv.org/abs/2108.03377)
    - Generate personalized dialogue using multi-task meta-learning
    - 用多任务元学习生成个性化的对话

  - ICCV-21 [BiMaL: Bijective Maximum Likelihood Approach to Domain Adaptation in Semantic Scene Segmentation](https://arxiv.org/abs/2108.03267)
    - Bijective MMD for domain adaptation
    - 双射MMD用于语义分割

  - [A Survey on Cross-domain Recommendation: Taxonomies, Methods, and Future Directions](https://arxiv.org/abs/2108.03357)
    - A survey on cross-domain recommendation
    - 跨领域的推荐的综述

  - [A Data Augmented Approach to Transfer Learning for Covid-19 Detection](https://arxiv.org/abs/2108.02870)
    - Data augmentation to transfer learning for COVID
    - 迁移学习使用数据增强，用于COVID-19

  - MM-21 [Few-shot Unsupervised Domain Adaptation with Image-to-class Sparse Similarity Encoding](https://arxiv.org/abs/2108.02953)
    - Few-shot DA with image-to-class sparse similarity encoding
    - 小样本的领域自适应

  - [Dual-Tuning: Joint Prototype Transfer and Structure Regularization for Compatible Feature Learning](https://arxiv.org/abs/2108.02959)
    - Prototype transfer and structure regularization
    - 原型的迁移学习

  - [Finetuning Pretrained Transformers into Variational Autoencoders](https://arxiv.org/abs/2108.02446)
    - Finetune transformer to VAE
    - 把transformer迁移到VAE

  - [Pre-trained Models for Sonar Images](http://arxiv.org/abs/2108.01111)
    - Pre-trained models for sonar images
    - 针对声纳图像的预训练模型

  - [Domain Adaptor Networks for Hyperspectral Image Recognition](http://arxiv.org/abs/2108.01555)
    - Finetune for hyperspectral image recognition
    - 针对高光谱图像识别的迁移学习

</details>

<details>
  <summary>Latest papers (2021-07)</summary>

  - CVPR-21 [Efficient Conditional GAN Transfer With Knowledge Propagation Across Classes](https://openaccess.thecvf.com/content/CVPR2021/html/Shahbazi_Efficient_Conditional_GAN_Transfer_With_Knowledge_Propagation_Across_Classes_CVPR_2021_paper.html)
    - Transfer conditional GANs to unseen classes
    - 通过知识传递，迁移预训练的conditional GAN到新类别

  - CVPR-21 [Ego-Exo: Transferring Visual Representations From Third-Person to First-Person Videos](https://openaccess.thecvf.com/content/CVPR2021/html/Li_Ego-Exo_Transferring_Visual_Representations_From_Third-Person_to_First-Person_Videos_CVPR_2021_paper.html)
    - Transfer learning from third-person to first-person video
    - 从第三人称视频迁移到第一人称

  - [Toward Co-creative Dungeon Generation via Transfer Learning](http://arxiv.org/abs/2107.12533)
    - Game scene generation with transfer learning
    - 用迁移学习生成游戏场景

  - [Transfer Learning in Electronic Health Records through Clinical Concept Embedding](https://arxiv.org/abs/2107.12919)
    - Transfer learning in electronic health record
    - 迁移学习用于医疗记录管理

  - CVPR-21 [Conditional Bures Metric for Domain Adaptation](https://openaccess.thecvf.com/content/CVPR2021/html/Luo_Conditional_Bures_Metric_for_Domain_Adaptation_CVPR_2021_paper.html)
    - A new metric for domain adaptation
    - 提出一个新的metric用于domain adaptation

  - CVPR-21 [Wasserstein Barycenter for Multi-Source Domain Adaptation](https://openaccess.thecvf.com/content/CVPR2021/html/Montesuma_Wasserstein_Barycenter_for_Multi-Source_Domain_Adaptation_CVPR_2021_paper.html)
    - Use Wasserstein Barycenter for multi-source domain adaptation
    - 利用Wasserstein Barycenter进行DA

  - CVPR-21 [Generalized Domain Adaptation](https://openaccess.thecvf.com/content/CVPR2021/html/Mitsuzumi_Generalized_Domain_Adaptation_CVPR_2021_paper.html)
    - A general definition for domain adaptation
    - 一个更抽象更一般的domain adaptation定义

  - CVPR-21 [Reducing Domain Gap by Reducing Style Bias](https://openaccess.thecvf.com/content/CVPR2021/html/Nam_Reducing_Domain_Gap_by_Reducing_Style_Bias_CVPR_2021_paper.html)
    - Syle-invariant training for adaptation and generalization
    - 通过训练图像对style无法辨别来进行DA和DG

  - CVPR-21 [Uncertainty-Guided Model Generalization to Unseen Domains](https://openaccess.thecvf.com/content/CVPR2021/html/Qiao_Uncertainty-Guided_Model_Generalization_to_Unseen_Domains_CVPR_2021_paper.html)
    - Uncertainty-guided generalization
    - 基于不确定性的domain generalization

  - CVPR-21 [Adaptive Methods for Real-World Domain Generalization](https://openaccess.thecvf.com/content/CVPR2021/html/Dubey_Adaptive_Methods_for_Real-World_Domain_Generalization_CVPR_2021_paper.html)
    - Adaptive methods for domain generalization
    - 动态算法，用于domain generalization

  - 20210716 ICML-21 [Continual Learning in the Teacher-Student Setup: Impact of Task Similarity](https://arxiv.org/abs/2107.04384)
    - Investigating task similarity in teacher-student learning
    - 调研在continual learning下teacher-student learning问题的任务相似度

  - 20210716 BMCV-extend [Exploring Dropout Discriminator for Domain Adaptation](https://arxiv.org/abs/2107.04231)
    - Using multiple discriminators for domain adaptation
    - 用分布估计代替点估计来做domain adaptation

  - 20210716 TPAMI-21 [Lifelong Teacher-Student Network Learning](https://arxiv.org/abs/2107.04689)
    - Lifelong distillation
    - 持续的知识蒸馏

  - 20210716 MICCAI-21 [Few-Shot Domain Adaptation with Polymorphic Transformers](https://arxiv.org/abs/2107.04805)
    - Few-shot domain adaptation with polymorphic transformer
    - 用多模态transformer做少样本的domain adaptation

  - 20210716 InterSpeech-21 [Speech2Video: Cross-Modal Distillation for Speech to Video Generation](https://arxiv.org/abs/2107.04806)
    - Cross-model distillation for video generation
    - 跨模态蒸馏用于语音到video的生成

  - 20210716 ICML-21 workshop [Leveraging Domain Adaptation for Low-Resource Geospatial Machine Learning](https://arxiv.org/abs/2107.04983)
    - Using domain adaptation for geospatial ML
    - 用domain adaptation进行地理空间的机器学习 

</details>

- - -

## 1.Introduction and Tutorials (简介与教程)

Want to quickly learn transfer learning？想尽快入门迁移学习？看下面的教程。

- Books 书籍
  - **《迁移学习导论》Introduction to Transfer Learning** [Homepage](http://jd92.wang/tlbook) [Buy](https://item.jd.com/13283188.html)

- Blogs 博客
  - [Zhihu blogs - 知乎专栏《小王爱迁移》系列文章](https://zhuanlan.zhihu.com/p/130244395)
	
- Video tutorials 视频教程 
  - [Recent advance of transfer learning - 2021年最新迁移学习发展现状探讨](https://www.bilibili.com/video/BV1N5411T7Sb)
  - [Domain generalization - 迁移学习新兴研究方向领域泛化](https://www.bilibili.com/video/BV1ro4y1S7dd/)
  
  - [Domain adaptation - 迁移学习中的领域自适应方法(中文)](https://www.bilibili.com/video/BV1T7411R75a/) 
  - [Transfer learning by Hung-yi Lee @ NTU - 台湾大学李宏毅的视频讲解(中文视频)](https://www.youtube.com/watch?v=qD6iD4TFsdQ)

- Brief introduction and slides 简介与ppt资料

  - [Recent advance of transfer learning](http://jd92.wang/assets/files/l15_jiqizhixin.pdf)
  
  - [Domain generalization survey](http://jd92.wang/assets/files/DGSurvey-ppt.pdf)
  
  - [Brief introduction in Chinese](https://github.com/jindongwang/transferlearning/blob/master/doc/%E8%BF%81%E7%A7%BB%E5%AD%A6%E4%B9%A0%E7%AE%80%E4%BB%8B.md)
	- [PPT (English)](http://jd92.wang/assets/files/l03_transferlearning.pdf) | [PPT (中文)](http://jd92.wang/assets/files/l08_tl_zh.pdf)
	
  - 迁移学习中的领域自适应方法 Domain adaptation: [PDF](http://jd92.wang/assets/files/l12_da.pdf) ｜ [Video on Bilibili](https://www.bilibili.com/video/BV1T7411R75a/) | [Video on Youtube](https://www.youtube.com/watch?v=RbIsHNtluwQ&t=22s)
	
  - Tutorial on transfer learning by Qiang Yang: [IJCAI'13](http://ijcai13.org/files/tutorial_slides/td2.pdf) | [2016 version](http://kddchina.org/file/IntroTL2016.pdf)

- Talk is cheap, show me the code 动手教程、代码、数据 
  - [Pytorch官方迁移学习示意代码](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
	- [Pytorch的finetune Fine-tune based on Alexnet and Resnet](https://github.com/jindongwang/transferlearning/tree/master/code/AlexNet_ResNet)
	- [用Pytorch进行深度特征提取](https://github.com/jindongwang/transferlearning/tree/master/code/feature_extractor)
	- [更多 More...](https://github.com/jindongwang/transferlearning/tree/master/code)

- [Transfer Learning Scholars and Labs - 迁移学习领域的著名学者、代表工作及实验室介绍](https://github.com/jindongwang/transferlearning/blob/master/doc/scholar_TL.md)

- [Negative transfer - 负迁移](https://www.zhihu.com/question/66492194/answer/242870418)

- - -

## 2.Transfer Learning Areas and Papers (研究领域与相关论文)

- [Survey](https://github.com/jindongwang/transferlearning/blob/master/doc/awesome_paper.md#survey)
- [Theory](#theory)
- [Per-training/Finetuning](https://github.com/jindongwang/transferlearning/blob/master/doc/awesome_paper.md#per-trainingfinetuning)
- [Knowledge distillation](https://github.com/jindongwang/transferlearning/blob/master/doc/awesome_paper.md#knowledge-distillation)
- [Traditional domain adaptation](https://github.com/jindongwang/transferlearning/blob/master/doc/awesome_paper.md#traditional-domain-adaptation)
- [Deep domain adaptation](https://github.com/jindongwang/transferlearning/blob/master/doc/awesome_paper.md#deep-domain-adaptation)
- [Domain generalization](https://github.com/jindongwang/transferlearning/blob/master/doc/awesome_paper.md#domain-generalization)
- [Source-free domain adaptation](https://github.com/jindongwang/transferlearning/blob/master/doc/awesome_paper.md#source-free-domain-adaptation)
- [Multi-source domain adaptation](https://github.com/jindongwang/transferlearning/blob/master/doc/awesome_paper.md#multi-source-domain-adaptation)
- [Heterogeneous transfer learning](https://github.com/jindongwang/transferlearning/blob/master/doc/awesome_paper.md#heterogeneous-transfer-learning)
- [Online transfer learning](https://github.com/jindongwang/transferlearning/blob/master/doc/awesome_paper.md#online-transfer-learning)
- [Zero-shot / few-shot learning](https://github.com/jindongwang/transferlearning/blob/master/doc/awesome_paper.md#zero-shot--few-shot-learning)
- [Multi-task learning](https://github.com/jindongwang/transferlearning/blob/master/doc/awesome_paper.md#multi-task-learning)
- [Transfer reinforcement learning](https://github.com/jindongwang/transferlearning/blob/master/doc/awesome_paper.md#transfer-reinforcement-learning)
- [Transfer metric learning](https://github.com/jindongwang/transferlearning/blob/master/doc/awesome_paper.md#transfer-metric-learning)
- [Federated transfer learning](https://github.com/jindongwang/transferlearning/blob/master/doc/awesome_paper.md#federated-transfer-learning)
- [Lifelong transfer learning](https://github.com/jindongwang/transferlearning/blob/master/doc/awesome_paper.md#lifelong-transfer-learning)
- [Transfer learning applications](https://github.com/jindongwang/transferlearning/blob/master/doc/awesome_paper.md#transfer-learning-applications)

- - -

## 3.Theory and Survey (理论与综述)

Here are some articles on transfer learning theory and survey.

**Survey (综述文章)：**

- 2021 Domain generalization: IJCAI-21 [Generalizing to Unseen Domains: A Survey on Domain Generalization](https://arxiv.org/abs/2103.03097) | [知乎文章](https://zhuanlan.zhihu.com/p/354740610) | [微信公众号](https://mp.weixin.qq.com/s/DsoVDYqLB1N7gj9X5UnYqw)
  - First survey on domain generalization
  - 第一篇对Domain generalization (领域泛化)的综述
- 2020 迁移学习最新survey，来自中科院计算所庄福振团队，发表在Proceedings of the IEEE: [A Comprehensive Survey on Transfer Learning](https://arxiv.org/abs/1911.02685)
- 2020 负迁移的综述：[Overcoming Negative Transfer: A Survey](https://arxiv.org/abs/2009.00909)
- 2020 知识蒸馏的综述: [Knowledge Distillation: A Survey](https://arxiv.org/abs/2006.05525)
- 用transfer learning进行sentiment classification的综述：[A Survey of Sentiment Analysis Based on Transfer Learning](https://ieeexplore.ieee.org/abstract/document/8746210) 
- 2019 一篇新survey：[Transfer Adaptation Learning: A Decade Survey](https://arxiv.org/abs/1903.04687)
- 2018 一篇迁移度量学习的综述: [Transfer Metric Learning: Algorithms, Applications and Outlooks](https://arxiv.org/abs/1810.03944)
- 2018 一篇最近的非对称情况下的异构迁移学习综述：[Asymmetric Heterogeneous Transfer Learning: A Survey](https://arxiv.org/abs/1804.10834)
- 2018 Neural style transfer的一个survey：[Neural Style Transfer: A Review](https://arxiv.org/abs/1705.04058)
- 2018 深度domain adaptation的一个综述：[Deep Visual Domain Adaptation: A Survey](https://www.sciencedirect.com/science/article/pii/S0925231218306684)
- 2017 多任务学习的综述，来自香港科技大学杨强团队：[A survey on multi-task learning](https://arxiv.org/abs/1707.08114)
- 2017 异构迁移学习的综述：[A survey on heterogeneous transfer learning](https://link.springer.com/article/10.1186/s40537-017-0089-0)
- 2017 跨领域数据识别的综述：[Cross-dataset recognition: a survey](https://arxiv.org/abs/1705.04396)
- 2016 [A survey of transfer learning](https://pan.baidu.com/s/1gfgXLXT)。其中交代了一些比较经典的如同构、异构等学习方法代表性文章。
- 2015 中文综述：[迁移学习研究进展](https://pan.baidu.com/s/1bpautob)
- 2010 [A survey on transfer learning](http://ieeexplore.ieee.org/abstract/document/5288526/)
- Survey on applications - 应用导向的综述：
	- 视觉domain adaptation综述：[Visual Domain Adaptation: A Survey of Recent Advances](https://pan.baidu.com/s/1o8BR7Vc)
	- 迁移学习应用于行为识别综述：[Transfer Learning for Activity Recognition: A Survey](https://pan.baidu.com/s/1kVABOYr)
	- 迁移学习与增强学习：[Transfer Learning for Reinforcement Learning Domains: A Survey](https://pan.baidu.com/s/1slfr0w1)
	- 多个源域进行迁移的综述：[A Survey of Multi-source Domain Adaptation](https://pan.baidu.com/s/1eSGREF4)。

**Theory （理论文章）:**

- ICML-20 [Few-shot domain adaptation by causal mechanism transfer](https://arxiv.org/pdf/2002.03497.pdf)
	- The first work on causal transfer learning
	- 日本理论组大佬Sugiyama的工作，causal transfer learning
- CVPR-19 [Characterizing and Avoiding Negative Transfer](https://arxiv.org/abs/1811.09751)
	- Characterizing and avoid negative transfer
	- 形式化并提出如何避免负迁移
- ICML-20 [On Learning Language-Invariant Representations for Universal Machine Translation](https://arxiv.org/abs/2008.04510)
  - Theory for universal machine translation
  - 对统一机器翻译模型进行了理论论证
- NIPS-06 [Analysis of Representations for Domain Adaptation](https://dl.acm.org/citation.cfm?id=2976474)
- ML-10 [A Theory of Learning from Different Domains](https://link.springer.com/article/10.1007/s10994-009-5152-4)
- NIPS-08 [Learning Bounds for Domain Adaptation](http://papers.nips.cc/paper/3212-learning-bounds-for-domain-adaptation)
- COLT-09 [Domain adaptation: Learning bounds and algorithms](https://arxiv.org/abs/0902.3430)
- MMD paper：[A Hilbert Space Embedding for Distributions](https://link.springer.com/chapter/10.1007/978-3-540-75225-7_5) and [A Kernel Two-Sample Test](http://www.jmlr.org/papers/v13/gretton12a.html)
- Multi-kernel MMD paper: [Optimal kernel choice for large-scale two-sample tests](http://papers.nips.cc/paper/4727-optimal-kernel-choice-for-large-scale-two-sample-tests)

_ _ _

## 4.Code (代码)

Unified codebases for:
- [Deep domain adaptation](https://github.com/jindongwang/transferlearning/tree/master/code/DeepDA)
- [Deep domain generalization](https://github.com/jindongwang/transferlearning/tree/master/code/DeepDG)

More: see [HERE](https://github.com/jindongwang/transferlearning/tree/master/code) and [HERE](https://colab.research.google.com/drive/1MVuk95mMg4ecGyUAIG94vedF81HtWQAr?usp=sharing) for an instant run using Google's Colab.

_ _ _

## 5.Transfer Learning Scholars (著名学者)

Here are some transfer learning scholars and labs.

**全部列表以及代表工作性见[这里](https://github.com/jindongwang/transferlearning/blob/master/doc/scholar_TL.md)** 

Please note that this list is far not complete. A full list can be seen in [here](https://github.com/jindongwang/transferlearning/blob/master/doc/scholar_TL.md). Transfer learning is an active field. *If you are aware of some scholars, please add them here.*

_ _ _

## 6.Transfer Learning Thesis (硕博士论文)

Here are some popular thesis on transfer learning.

[这里](https://pan.baidu.com/share/init?surl=iuzZhHdumrD64-yx_VAybA), 提取码：txyz。

- - -

## 7.Datasets and Benchmarks (数据集与评测结果)

Please see [HERE](https://github.com/jindongwang/transferlearning/blob/master/data) for the popular transfer learning **datasets and benchmark** results.

[这里](https://github.com/jindongwang/transferlearning/blob/master/data)整理了常用的公开数据集和一些已发表的文章在这些数据集上的实验结果。

- - -

## 8.Transfer Learning Challenges (迁移学习比赛)

- [Visual Domain Adaptation Challenge (VisDA)](http://ai.bu.edu/visda-2018/)

- - -

## Applications (迁移学习应用)

- [Computer vision](https://github.com/jindongwang/transferlearning/blob/master/doc/transfer_learning_application.md#computer-vision)
- [Medical and healthcare](https://github.com/jindongwang/transferlearning/blob/master/doc/transfer_learning_application.md#medical-and-healthcare)
- [Natural language processing](https://github.com/jindongwang/transferlearning/blob/master/doc/transfer_learning_application.md#natural-language-processing)
- [Time series](https://github.com/jindongwang/transferlearning/blob/master/doc/transfer_learning_application.md#time-series)
- [Speech](https://github.com/jindongwang/transferlearning/blob/master/doc/transfer_learning_application.md#speech)
- [Multimedia](https://github.com/jindongwang/transferlearning/blob/master/doc/transfer_learning_application.md#multimedia)
- [Recommendation](https://github.com/jindongwang/transferlearning/blob/master/doc/transfer_learning_application.md#recommendation)
- [Human activity recognition](https://github.com/jindongwang/transferlearning/blob/master/doc/transfer_learning_application.md#human-activity-recognition)
- [Autonomous driving](https://github.com/jindongwang/transferlearning/blob/master/doc/transfer_learning_application.md#autonomous-driving)
- [Others](https://github.com/jindongwang/transferlearning/blob/master/doc/transfer_learning_application.md#others)

See [HERE](https://github.com/jindongwang/transferlearning/blob/master/doc/transfer_learning_application.md) for transfer learning applications.

迁移学习应用请见[这里](https://github.com/jindongwang/transferlearning/blob/master/doc/transfer_learning_application.md)。

- - -

## Other Resources (其他资源)

- Call for papers:
  - [Advances in Transfer Learning: Theory, Algorithms, and Applications](https://www.frontiersin.org/research-topics/21133/advances-in-transfer-learning-theory-algorithms-and-applications), DDL: October 2021

- Related projects:
  - Salad: [A semi-supervised domain adaptation library](https://domainadaptation.org)

- - -

## Contributing (欢迎参与贡献)

If you are interested in contributing, please refer to [HERE](https://github.com/jindongwang/transferlearning/blob/master/CONTRIBUTING.md) for instructions in contribution.

- - -

### Copyright notice

> ***[Notes]This Github repo can be used by following the corresponding licenses. I want to emphasis that it may contain some PDFs or thesis, which were downloaded by me and can only be used for academic purposes. The copyrights of these materials are owned by corresponding publishers or organizations. All this are for better adademic research. If any of the authors or publishers have concerns, please contact me to delete or replace them.***
