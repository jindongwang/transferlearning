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
  - Large language model evaluation: [[llm-eval](https://llm-eval.github.io/)]
  - Large language model enhancement: [[llm-enhance](https://llm-enhance.github.io/)]
  - Robust machine learning: [[robustlearn: robust machine learning](https://github.com/microsoft/robustlearn)]
  - Semi-supervised learning: [[USB: unified semi-supervised learning benchmark](https://github.com/microsoft/Semi-supervised-learning)] | [[TorchSSL: a unified SSL library](https://github.com/TorchSSL/TorchSSL)] 
  - LLM benchmark: [[PromptBench: adverarial robustness of prompts of LLMs](https://github.com/microsoft/promptbench)]
  - Federated learning: [[PersonalizedFL: library for personalized federated learning](https://github.com/microsoft/PersonalizedFL)]
  - Activity recognition and machine learning [[Activity recognition](https://github.com/jindongwang/activityrecognition)]｜[[Machine learning](https://github.com/jindongwang/MachineLearning)]

- - -

**NOTE:** You can directly open the code in [Gihub Codespaces](https://docs.github.com/en/codespaces/getting-started/quickstart#introduction) on the web to run them without downloading! Also, try [github.dev](https://github.dev/jindongwang/transferlearning).

## 0.Papers (论文)

[Awesome transfer learning papers (迁移学习文章汇总)](https://github.com/jindongwang/transferlearning/tree/master/doc/awesome_paper.md)

- [Paperweekly](http://www.paperweekly.site/collections/231/papers): A website to recommend and read paper notes

**Latest papers**: 

- By topic: [doc/awesome_papers.md](/doc/awesome_paper.md)
- By date: [doc/awesome_paper_date.md](/doc/awesome_paper_date.md)

*Updated at 2024-03-26:*

- Deep Domain Adaptation: A Sim2Real Neural Approach for Improving Eye-Tracking Systems [[arxiv](https://arxiv.org/abs/2403.15947)]
  - Domain adaptation for eye-tracking systems 用DA进行眼球追踪

- EAGLE: A Domain Generalization Framework for AI-generated Text Detection [[arxiv](https://arxiv.org/abs/2403.15690)]
  - Domain generalization for AI content detection 用DG进行AI生成内容检测

- DPStyler: Dynamic PromptStyler for Source-Free Domain Generalization [[arxiv](https://arxiv.org/abs/2403.16697)]
  - Dynamic propmtstyler for source-free DG 动态prompt分格化用于source-free DG

*Updated at 2024-03-21:*

- Neurocomputing'24 Uncertainty-Aware Pseudo-Label Filtering for Source-Free Unsupervised Domain Adaptation [[arxiv](https://arxiv.org/abs/2403.11256)]
  - Unvertainty-aware source-free domain adaptation 基于不确定性伪标签的domain adaptation

- Efficient Domain Adaptation for Endoscopic Visual Odometry [[arxiv](https://arxiv.org/abs/2403.10860)]
  - Efficient domain adaptation for visual odometry 高效DA用于odometry

- Potential of Domain Adaptation in Machine Learning in Ecology and Hydrology to Improve Model Extrapolability [[arxiv](https://arxiv.org/abs/2403.11331)]
  - Domain adaptation in ecology and hydrology 研究生态学和水文学中的DA

- ICLR'24 SF(DA)2: Source-free Domain Adaptation Through the Lens of Data Augmentation [[arxiv](https://arxiv.org/abs/2403.10834)]
  - Source-free DA by data augmentation 通过数据增强来进行source-free DA

- CVPR'24 Universal Semi-Supervised Domain Adaptation by Mitigating Common-Class Bias [[arxiv](https://arxiv.org/abs/2403.11234)]
  - Unviersal semi-supervised DA 通过公共类bias进行半监督DA

- Domain Adaptation Using Pseudo Labels for COVID-19 Detection [[arxiv](https://arxiv.org/abs/2403.11498)]
  - Domain adaptation for COVID-19 detection 用DA进行covid-19检查

- Ensembling and Test Augmentation for Covid-19 Detection and Covid-19 Domain Adaptation from 3D CT-Scans [[arxiv](https://arxiv.org/abs/2403.11338)]
  - Covid-19 test using domain adaptation 使用集成和测试增强用于DA covid-19

- V2X-DGW: Domain Generalization for Multi-agent Perception under Adverse Weather Conditions [[arxiv](https://arxiv.org/abs/2403.11371)]
  - DG for multi-agent perception 领域泛化用于极端天气

- Bidirectional Multi-Step Domain Generalization for Visible-Infrared Person Re-Identification [[arxiv](https://arxiv.org/abs/2403.10782)]
  - Bidirectional multi-step DG for REID 双向领域泛化用于REID

- MedMerge: Merging Models for Effective Transfer Learning to Medical Imaging Tasks [[arxiv](https://arxiv.org/abs/2403.11646)]
  - Model merge for medical transfer learning 通过模型合并进行医学迁移学习

*Updated at 2024-03-05:*
- SPA: A Graph Spectral Alignment Perspective for Domain Adaptation [[NeurIPS 2023]](https://arxiv.org/abs/2310.17594) [[Pytorch]](https://github.com/CrownX/SPA)
  - Graph spectral alignment and neighbor-aware propagation for domain adaptation

*Updated at 2024-03-20:*

- Addressing Source Scale Bias via Image Warping for Domain Adaptation [[arxiv](https://arxiv.org/abs/2403.12712)]
  - Address the source scale bias for domain adaptation 解决源域的scale bias

*Updated at 2024-03-18:*

- ICLR'24 扩展版 Learning with Noisy Foundation Models [[arxiv](https://arxiv.org/abs/2403.06869)]
  - Fine-tune a noisy foundation model 基础模型有noisy的时候如何finetune

- Visual Foundation Models Boost Cross-Modal Unsupervised Domain Adaptation for 3D Semantic Segmentation [[arxiv](https://arxiv.org/abs/2403.10001)]
  - Foundation models help domain adaptation 基础模型帮助领域自适应


*Updated at 2024-03-12:*

- Attention Prompt Tuning: Parameter-efficient Adaptation of Pre-trained Models for Spatiotemporal Modeling [[arxiv](https://arxiv.org/abs/2403.06978)]
  - Parameter-efficient adaptation for spatiotemporal modeling

- ICASSP'24 Test-time Distribution Learning Adapter for Cross-modal Visual Reasoning [[arxiv](https://arxiv.org/abs/2403.06059)]
  - Test-time distribution learning adapter

- A Study on Domain Generalization for Failure Detection through Human Reactions in HRI [[arxiv](https://arxiv.org/abs/2403.06315)]
  - Domain generalization for failure detection through human reactions in HRI

- ICLR'24 Towards Robust Out-of-Distribution Generalization Bounds via Sharpness [[arxiv](https://arxiv.org/abs/2403.06392)]
  - Robust OOD generalization bounds

- Learning with Noisy Foundation Models [[arxiv](https://arxiv.org/abs/2403.06869)]
  - Learning with noisy foundation models

- - -

## 1.Introduction and Tutorials (简介与教程)

Want to quickly learn transfer learning？想尽快入门迁移学习？看下面的教程。

- Books 书籍
  - **Introduction to Transfer Learning: Algorithms and Practice** [[Buy or read](https://link.springer.com/book/9789811975837)]
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
