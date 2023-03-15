[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]

<h1 align="center">
  <br>
  <img src="png/logo.jpg" alt="Transfer Leanring" width="500">
</h1>

<h4 align="center">Everything about Transfer Learning. 迁移学习.</h4>

<p align="center">
  <strong><a href="#0papers-论文">Papers</a></strong> •
  <strong><a href="#1introduction-and-tutorials-简介与教程">Tutorials</a></strong> •
  <a href="#2transfer-learning-areas-and-papers-研究领域与相关论文">Research areas</a> •
  <a href="#3theory-and-survey-理论与综述">Theory</a> •
  <a href="#3theory-and-survey-理论与综述">Survey</a> •
  <strong><a href="https://github.com/jindongwang/transferlearning/tree/master/code">Code</a></strong> •
  <strong><a href="#7datasets-and-benchmarks-数据集与评测结果">Dataset & benchmark</a></strong>
</p>
<p align="center">
  <a href="#6transfer-learning-thesis-硕博士论文">Thesis</a> •
  <a href="#5transfer-learning-scholars-著名学者">Scholars</a> •
  <a href="#8transfer-learning-challenges-迁移学习比赛">Contests</a> •
  <a href="#journals-and-conferences">Journal/conference</a> •
  <a href="#applications-迁移学习应用">Applications</a> •
  <a href="#other-resources-其他资源">Others</a> •
  <a href="#contributing-欢迎参与贡献">Contributing</a>
</p>

**Widely used by top conferences and journals:** 
- Conferences: [[CVPR'22](https://openaccess.thecvf.com/content/CVPR2022W/FaDE-TCV/html/Zhang_Segmenting_Across_Places_The_Need_for_Fair_Transfer_Learning_With_CVPRW_2022_paper.html)] [[NeurIPS'21](https://proceedings.neurips.cc/paper/2021/file/731b03008e834f92a03085ef47061c4a-Paper.pdf)] [[IJCAI'21](https://arxiv.org/abs/2103.03097)] [[ESEC/FSE'20](https://dl.acm.org/doi/abs/10.1145/3368089.3409696)] [[IJCNN'20](https://ieeexplore.ieee.org/abstract/document/9207556)] [[ACMMM'18](https://dl.acm.org/doi/abs/10.1145/3240508.3240512)] [[ICME'19](https://ieeexplore.ieee.org/abstract/document/8784776/)]
- Journals: [[IEEE TKDE](https://ieeexplore.ieee.org/abstract/document/9782500/)] [[ACM TIST](https://dl.acm.org/doi/abs/10.1145/3360309)] [[Information sciences](https://www.sciencedirect.com/science/article/pii/S0020025520308458)] [[Neurocomputing](https://www.sciencedirect.com/science/article/pii/S0925231221007025)] [[IEEE Transactions on Cognitive and Developmental Systems](https://ieeexplore.ieee.org/abstract/document/9659817)]

```
@Misc{transferlearning.xyz,
howpublished = {\url{http://transferlearning.xyz}},   
title = {Everything about Transfer Learning and Domain Adapation},  
author = {Wang, Jindong and others}  
}  
```

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re) [![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT) [![LICENSE](https://img.shields.io/badge/license-Anti%20996-blue.svg)](https://github.com/996icu/996.ICU/blob/master/LICENSE) [![996.icu](https://img.shields.io/badge/link-996.icu-red.svg)](https://996.icu) 

Related Codes: 
  - Robust machine learning: [[robustlearn: robust machine learning](https://github.com/microsoft/robustlearn)]
  - Semi-supervised learning: [[USB: unified semi-supervised learning benchmark](https://github.com/microsoft/Semi-supervised-learning)] | [[TorchSSL: a unified SSL library](https://github.com/TorchSSL/TorchSSL)] 
  - Federated learning: [[PersonalizedFL: library for personalized federated learning](https://github.com/microsoft/PersonalizedFL)]
  - Activity recognition and machine learning [[Activity recognition](https://github.com/jindongwang/activityrecognition)]｜[[Machine learning](https://github.com/jindongwang/MachineLearning)]

- - -

**NOTE:** You can directly open the code in [Gihub Codespaces](https://docs.github.com/en/codespaces/getting-started/quickstart#introduction) on the web to run them without downloading! Also, try [github.dev](https://github.dev/jindongwang/transferlearning).

## 0.Papers (论文)

[Awesome transfer learning papers (迁移学习文章汇总)](https://github.com/jindongwang/transferlearning/tree/master/doc/awesome_paper.md)

- [Paperweekly](http://www.paperweekly.site/collections/231/papers): A website to recommend and read paper notes

**Latest papers**: 

- By topic: [doc/awesome_papers.md](https://github.com/jindongwang/transferlearning/blob/master/doc/awesome_paper.md)
- By date: [[2023-01](https://github.com/jindongwang/transferlearning/blob/master/doc/awesome_paper_date.md#2023-01)] [[2022-12](https://github.com/jindongwang/transferlearning/blob/master/doc/awesome_paper_date.md#2022-12)] [[2022-11](https://github.com/jindongwang/transferlearning/blob/master/doc/awesome_paper_date.md#2022-11)] [[2022-10](https://github.com/jindongwang/transferlearning/blob/master/doc/awesome_paper_date.md#2022-10)] [[2022-09](https://github.com/jindongwang/transferlearning/blob/master/doc/awesome_paper_date.md#2022-09)] [[2022-08](https://github.com/jindongwang/transferlearning/blob/master/doc/awesome_paper_date.md#2022-08)] [[2022-07](https://github.com/jindongwang/transferlearning/blob/master/doc/awesome_paper_date.md#2022-07)] [[2022-06](https://github.com/jindongwang/transferlearning/blob/master/doc/awesome_paper_date.md#2022-06)] [[2022-05](https://github.com/jindongwang/transferlearning/blob/master/doc/awesome_paper_date.md#2022-05)] [[2022-04](https://github.com/jindongwang/transferlearning/blob/master/doc/awesome_paper_date.md#2022-04)] [[2022-03](https://github.com/jindongwang/transferlearning/blob/master/doc/awesome_paper_date.md#2022-03)] [[2022-02](https://github.com/jindongwang/transferlearning/blob/master/doc/awesome_paper_date.md#2022-02)] [[2022-01](https://github.com/jindongwang/transferlearning/blob/master/doc/awesome_paper_date.md#2022-01)] [[2021-12](https://github.com/jindongwang/transferlearning/blob/master/doc/awesome_paper_date.md#2021-12)] [[2021-11](https://github.com/jindongwang/transferlearning/blob/master/doc/awesome_paper_date.md#2021-11)] [[2021-10](https://github.com/jindongwang/transferlearning/blob/master/doc/awesome_paper_date.md#2021-10)] [[2021-09](https://github.com/jindongwang/transferlearning/blob/master/doc/awesome_paper_date.md#2021-09)] [[2021-08](https://github.com/jindongwang/transferlearning/blob/master/doc/awesome_paper_date.md#2021-08)] [[2021-07](https://github.com/jindongwang/transferlearning/blob/master/doc/awesome_paper_date.md#2021-07)]

*Updated at 2023-03-15:*

- Imbalanced Domain Generalization for Robust Single Cell Classification in Hematological Cytomorphology [[arxiv](https://arxiv.org/abs/2303.07771)]
  - Imbalanced domain generalization for single cell classification 不平衡的DG用于单细胞分类

- Revisit Parameter-Efficient Transfer Learning: A Two-Stage Paradigm [[arxiv](https://arxiv.org/abs/2303.07910)]
  - Parameter-efficient transfer learning: a two-stage approach 一种两阶段的参数高效迁移学习

- Unsupervised Cumulative Domain Adaptation for Foggy Scene Optical Flow [[arxiv](https://arxiv.org/abs/2303.07564)]
  - Domain adaptation for foggy scene optical flow 领域自适应用于雾场景的光流

- ICLR'23 AutoTransfer: AutoML with Knowledge Transfer -- An Application to Graph Neural Networks [[arxiv](https://arxiv.org/abs/2303.07669)]
  - GNN with autoML transfer learning 用于GNN的自动迁移学习

- Transfer Learning for Real-time Deployment of a Screening Tool for Depression Detection Using Actigraphy [[arxiv](https://arxiv.org/abs/2303.07847)]
  - Transfer learning for Depression detection 迁移学习用于脉动计焦虑检测

- Domain Generalization via Nuclear Norm Regularization [[arxiv](https://arxiv.org/abs/2303.07527)]
  - Domain generalization via nuclear norm regularization 使用核归一化进行domain generalization

*Updated at 2023-03-07:*

- To Stay or Not to Stay in the Pre-train Basin: Insights on Ensembling in Transfer Learning [[arxiv](https://arxiv.org/abs/2303.03374)]
  - Ensembling in transfer learning 调研迁移学习中的集成

- CVPR'13 Masked Images Are Counterfactual Samples for Robust Fine-tuning [[arxiv](https://arxiv.org/abs/2303.03052)]
  - Masked images for robust fine-tuning 调研masked image对于fine-tuning的影响

- FedCLIP: Fast Generalization and Personalization for CLIP in Federated Learning [[arxiv](https://arxiv.org/abs/2302.13485v1)]
  - Fast generalization for federated CLIP 在联邦中进行快速的CLIP训练

*Updated at 2023-03-02:*

- Robust Representation Learning with Self-Distillation for Domain Generalization [[arxiv](http://arxiv.org/abs/2302.06874)]
  - Robust representation learning with self-distillation

- ICLR-23 Temporal Coherent Test-Time Optimization for Robust Video Classification [[arxiv](http://arxiv.org/abs/2302.14309)]
  - Temporal distribution shift in video classification

- WSDM-23 A tutorial on domain generalization [[link](https://dl.acm.org/doi/10.1145/3539597.3572722)] | [[website](https://dgresearch.github.io/)]
  - A tutorial on domain generalization

*Updated at 2023-02-23:*

- On the Robustness of ChatGPT: An Adversarial and Out-of-distribution Perspective [[arxiv](https://arxiv.org/abs/2302.12095)] | [[code](https://github.com/microsoft/robustlearn)]
  - Adversarial and OOD evaluation of ChatGPT 对ChatGPT鲁棒性的评测


*Updated at 2023-02-08:*

- Transfer learning for process design with reinforcement learning [[arxiv](https://arxiv.org/abs/2302.03375)]
  - Transfer learning for process design with reinforcement learning 使用强化迁移学习进行过程设计

- Domain Adaptation for Time Series Under Feature and Label Shifts [[arxiv](https://arxiv.org/abs/2302.03133)]
  - Domain adaptation for time series 用于时间序列的domain adaptation

- How Reliable is Your Regression Model's Uncertainty Under Real-World Distribution Shifts? [[arxiv](https://arxiv.org/abs/2302.03679)]
  - Regression models uncertainty for distribution shift 回归模型对于分布漂移的不确定性

*Updated at 2023-02-02:*

- ICLR'23 SoftMatch: Addressing the Quantity-Quality Tradeoff in Semi-supervised Learning [[arxiv](https://arxiv.org/abs/2301.10921)]
  - Semi-supervised learning algorithm 解决标签质量问题的半监督学习方法

- Empirical Study on Optimizer Selection for Out-of-Distribution Generalization [[arxiv](http://arxiv.org/abs/2211.08583)]
  - Opimizer selection for OOD generalization OOD泛化中的学习器选择

- ICML'22 Understanding the failure modes of out-of-distribution generalization [[arxiv](https://openreview.net/forum?id=fSTD6NFIW_b)]
  - Understand the failure modes of OOD generalization 探索OOD泛化中的失败现象

- ICLR'23 Out-of-distribution Representation Learning for Time Series Classification [[arxiv](https://arxiv.org/abs/2209.07027)]
  - OOD for time series classification 时间序列分类的OOD算法

- - -

## 1.Introduction and Tutorials (简介与教程)

Want to quickly learn transfer learning？想尽快入门迁移学习？看下面的教程。

- Books 书籍
  - **《迁移学习》（杨强）** [[Buy](https://item.jd.com/12930984.html)] [[English version](https://www.cambridge.org/core/books/transfer-learning/CCFFAFE3CDBC245047F1DEC71D9EF3C7)]
  - **《迁移学习导论》(王晋东、陈益强著)** [[Homepage](http://jd92.wang/tlbook)] [[Buy](https://item.jd.com/13272157.html)]

- Blogs 博客
  - [Zhihu blogs - 知乎专栏《小王爱迁移》系列文章](https://zhuanlan.zhihu.com/p/130244395)
	
- Video tutorials 视频教程
  - Transfer learning 迁移学习:
    - [Recent advance of transfer learning - 2022年最新迁移学习发展现状探讨](https://www.bilibili.com/video/BV1nY411E7Uc/)
    - [Definitions of transfer learning area - 迁移学习领域名词解释](https://www.bilibili.com/video/BV1fu411o7BW) [[Article](https://zhuanlan.zhihu.com/p/428097044)]
    - [Transfer learning by Hung-yi Lee @ NTU - 台湾大学李宏毅的视频讲解(中文视频)](https://www.youtube.com/watch?v=qD6iD4TFsdQ)
  - Domain generalization 领域泛化：
    - [IJCAI-ECAI'22 tutorial on domain generalization - 领域泛化tutorial](https://dgresearch.github.io/)
    - [Domain generalization - 迁移学习新兴研究方向领域泛化](https://www.bilibili.com/video/BV1ro4y1S7dd/)
  - Domain adaptation 领域自适应：
    - [Domain adaptation - 迁移学习中的领域自适应方法(中文)](https://www.bilibili.com/video/BV1T7411R75a/) 
  

- Brief introduction and slides 简介与ppt资料
  - [Recent advance of transfer learning](https://jd92.wang/assets/files/l16_aitime.pdf)
  - [Domain generalization survey](http://jd92.wang/assets/files/DGSurvey-ppt.pdf)
  - [Brief introduction in Chinese](https://github.com/jindongwang/transferlearning/blob/master/doc/%E8%BF%81%E7%A7%BB%E5%AD%A6%E4%B9%A0%E7%AE%80%E4%BB%8B.md)
	- [PPT (English)](http://jd92.wang/assets/files/l03_transferlearning.pdf) | [PPT (中文)](http://jd92.wang/assets/files/l08_tl_zh.pdf)
  - 迁移学习中的领域自适应方法 Domain adaptation: [PDF](http://jd92.wang/assets/files/l12_da.pdf) ｜ [Video on Bilibili](https://www.bilibili.com/video/BV1T7411R75a/) | [Video on Youtube](https://www.youtube.com/watch?v=RbIsHNtluwQ&t=22s)
  - Tutorial on transfer learning by Qiang Yang: [IJCAI'13](http://ijcai13.org/files/tutorial_slides/td2.pdf) | [2016 version](http://kddchina.org/file/IntroTL2016.pdf)

- Talk is cheap, show me the code 动手教程、代码、数据 
  - [Pytorch tutorial on transfer learning](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
	- [Pytorch finetune](https://github.com/jindongwang/transferlearning/tree/master/code/AlexNet_ResNet)
	- [DeepDA: a unified deep domain adaptation toolbox](https://github.com/jindongwang/transferlearning/tree/master/code/DeepDA)
	- [DeepDG: a unified deep domain generalization toolbox](https://github.com/jindongwang/transferlearning/tree/master/code/DeepDG)
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
- [Safe transfer learning](https://github.com/jindongwang/transferlearning/blob/master/doc/awesome_paper.md#safe-transfer-learning)
- [Transfer learning applications](https://github.com/jindongwang/transferlearning/blob/master/doc/awesome_paper.md#transfer-learning-applications)

- - -

## 3.Theory and Survey (理论与综述)

Here are some articles on transfer learning theory and survey.

**Survey (综述文章)：**

- 2023 Source-Free Unsupervised Domain Adaptation: A Survey [[arxiv](http://arxiv.org/abs/2301.00265)]
- 2022 [Transfer Learning for Future Wireless Networks: A Comprehensive Survey](https://arxiv.org/abs/2102.07572)
- 2022 [A Review of Deep Transfer Learning and Recent Advancements](https://arxiv.org/abs/2201.09679)
- 2022 [Transferability in Deep Learning: A Survey](https://paperswithcode.com/paper/transferability-in-deep-learning-a-survey), from Mingsheng Long in THU.
- 2021 Domain generalization: IJCAI-21 [Generalizing to Unseen Domains: A Survey on Domain Generalization](https://arxiv.org/abs/2103.03097) | [知乎文章](https://zhuanlan.zhihu.com/p/354740610) | [微信公众号](https://mp.weixin.qq.com/s/DsoVDYqLB1N7gj9X5UnYqw)
  - First survey on domain generalization
  - 第一篇对Domain generalization (领域泛化)的综述
- 2021 Vision-based activity recognition: [A Survey of Vision-Based Transfer Learning in Human Activity Recognition](https://www.mdpi.com/2079-9292/10/19/2412)
- 2021 ICSAI [A State-of-the-Art Survey of Transfer Learning in Structural Health Monitoring](https://ieeexplore.ieee.org/abstract/document/9664171)
- 2020 [Transfer learning: survey and classification](https://link.springer.com/chapter/10.1007/978-981-15-5345-5_13), Advances in Intelligent Systems and Computing. 
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
- See all codes here: https://github.com/jindongwang/transferlearning/tree/master/code.

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

## Journals and Conferences

See [here](https://github.com/jindongwang/transferlearning/blob/master/doc/venues.md) for a full list of related journals and conferences.

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

[contributors-shield]: https://img.shields.io/github/contributors/jindongwang/transferlearning.svg?style=for-the-badge
[contributors-url]: https://github.com/jindongwang/transferlearning/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/jindongwang/transferlearning.svg?style=for-the-badge
[forks-url]: https://github.com/jindongwang/transferlearning/network/members
[stars-shield]: https://img.shields.io/github/stars/jindongwang/transferlearning.svg?style=for-the-badge
[stars-url]: https://github.com/jindongwang/transferlearning/stargazers
[issues-shield]: https://img.shields.io/github/issues/jindongwang/transferlearning.svg?style=for-the-badge
[issues-url]: https://github.com/jindongwang/transferlearning/issues
[license-shield]: https://img.shields.io/github/license/jindongwang/transferlearning.svg?style=for-the-badge
[license-url]: https://github.com/jindongwang/transferlearning/blob/main/LICENSE.txt
