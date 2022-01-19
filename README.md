# 迁移学习 Transfer Learning  

<h1 align="center">
  <br>
  <img src="png/logo.jpg" alt="Transfer Leanring" width="500">
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
  <a href="#journals-and-conferences">Journal/conference</a> •
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

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re) [![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT) [![LICENSE](https://img.shields.io/badge/license-Anti%20996-blue.svg)](https://github.com/996icu/996.ICU/blob/master/LICENSE) [![996.icu](https://img.shields.io/badge/link-996.icu-red.svg)](https://996.icu) Related repos：[Activity recognition](https://github.com/jindongwang/activityrecognition)｜[Machine learning](https://github.com/jindongwang/MachineLearning)

- - -

**NOTE:** You can directly open the code in [Gihub Codespaces](https://docs.github.com/en/codespaces/getting-started/quickstart#introduction) on the web to run them without downloading! Also, try [github.dev](https://github.dev/jindongwang/transferlearning).

## 0.Papers (论文)

[Awesome transfer learning papers (迁移学习文章汇总)](https://github.com/jindongwang/transferlearning/tree/master/doc/awesome_paper.md)

- [Paperweekly](http://www.paperweekly.site/collections/231/papers): A website to recommend and read paper notes

**Latest papers**: (All papers are also put in [doc/awesome_papers.md](https://github.com/jindongwang/transferlearning/blob/master/doc/awesome_paper.md))


<details>
  <summary>Latest papers (2022-01-19)</summary>

- AAAI-22 [Knowledge Sharing via Domain Adaptation in Customs Fraud Detection](https://arxiv.org/abs/2201.06759)
  - Domain adaptation for fraud detection
  - 用领域自适应进行欺诈检测

- [Continual Coarse-to-Fine Domain Adaptation in Semantic Segmentation](https://arxiv.org/abs/2201.06974)
  - Domain adaptation in semantic segmentation
  - 领域自适应在语义分割的应用

</details>

<details>
  <summary>Latest papers (2022-01-13)</summary>

- KBS-22 [Intra-domain and cross-domain transfer learning for time series data -- How transferable are the features](https://arxiv.org/abs/2201.04449)
  - An overview of transfer learning for time series data
  - 一个用迁移学习进行时间序列分析的小综述

- [A Likelihood Ratio based Domain Adaptation Method for E2E Models](2201.03655)
  - Domain adaptation for speech recognition
  - 用domain adaptation进行语音识别

- [Transfer Learning for Scene Text Recognition in Indian Languages](2201.03180)
  - Transfer learning for scene text recognition in Indian languages
  - 用迁移学习进行印度语的场景文字识别

</details>

<details>
  <summary>Latest papers (2022-01-07)</summary>

- IEEE TMM-22 [Decompose to Adapt: Cross-domain Object Detection via Feature Disentanglement](https://arxiv.org/abs/2201.01929)
  - Invariant and shared components for Faster RCNN detection
  - 解耦公共和私有表征进行目标检测

- [Mixture of basis for interpretable continual learning with distribution shifts](https://arxiv.org/abs/2201.01853)
  - Incremental learning with mixture of basis
  - 用mixture of domains进行增量学习

</details>

<details>
  <summary>Latest papers (2022-01-04)</summary>

- TKDE-22 [Adaptive Memory Networks with Self-supervised Learning for Unsupervised Anomaly Detection](https://arxiv.org/abs/2201.00464)
  - Adaptiev memory network for anomaly detection
  - 自适应的记忆网络用于异常检测

- ICIP-22 [Meta-Learned Feature Critics for Domain Generalized Semantic Segmentation](https://arxiv.org/abs/2112.13538)
  - Meta-learning for domain generalization
  - 元学习用于domain generalization

- ICIP-22 [Few-Shot Classification in Unseen Domains by Episodic Meta-Learning Across Visual Domains](https://arxiv.org/abs/2112.13539)
  - Few-shot generalization using meta-learning
  - 用元学习进行小样本的泛化

- [Data-Free Knowledge Transfer: A Survey](https://arxiv.org/abs/2112.15278)
  - A survey on data-free distillation and source-free DA
  - 一篇关于data-free蒸馏和source-free DA的综述

- [An Ensemble of Pre-trained Transformer Models For Imbalanced Multiclass Malware Classification](https://arxiv.org/abs/2112.13236)
  - An ensemble of pre-trained transformer for malware classification
  - 预训练的transformer通过集成进行恶意软件检测

- [Optimal Representations for Covariate Shift](https://arxiv.org/abs/2201.00057)
  - Learning optimal representations for covariate shift
  - 为covariate shift学习最优的表达

- [Transfer-learning-based Surrogate Model for Thermal Conductivity of Nanofluids](https://arxiv.org/abs/2201.00435)
  - Transfer learning for thermal conductivity
  - 迁移学习用于热传导

- [Transfer learning of phase transitions in percolation and directed percolation](https://arxiv.org/abs/2112.15516)
  - Transfer learning of phase transitions in percolation and directed percolation
  - 迁移学习用于precolation

- [Transfer learning for cancer diagnosis in histopathological images](https://arxiv.org/abs/2112.15523)
  - Transfer learning for cancer diagnosis
  - 迁移学习用于癌症诊断

</details>

<details>
  <summary>Latest papers (2021-12)</summary>

- IEEE TASLP-22 [Exploiting Adapters for Cross-lingual Low-resource Speech Recognition](https://arxiv.org/abs/2105.11905)  [Zhihu article](https://zhuanlan.zhihu.com/p/448216624)
    - Cross-lingual speech recogntion using meta-learning and transfer learning
    - 用元学习和迁移学习进行跨语言的低资源语音识别

- [More is Better: A Novel Multi-view Framework for Domain Generalization](https://arxiv.org/abs/2112.12329)
    - Multi-view learning for domain generalization
    - 使用多视图学习来进行domain generalization

- [SLIP: Self-supervision meets Language-Image Pre-training](https://arxiv.org/abs/2112.12750)
    - Self-supervised learning + language image pretraining
    - 用自监督学习用于语言到图像的预训练

  - [Domain Prompts: Towards memory and compute efficient domain adaptation of ASR systems](https://arxiv.org/abs/2112.08718)
    - Prompt for domain adaptation in speech recognition
    - 用Prompt在语音识别中进行domain adaptation

  - [UMAD: Universal Model Adaptation under Domain and Category Shift](https://arxiv.org/abs/2112.08553)
    - Model adaptation under domain and category shift
    - 在domain和class都有shift的前提下进行模型适配

  - [Domain Adaptation on Point Clouds via Geometry-Aware Implicits](https://arxiv.org/abs/2112.09343)
    - Domain adaptation for point cloud
    - 针对点云的domain adaptation

  - [A Survey of Unsupervised Domain Adaptation for Visual Recognition](http://arxiv.org/abs/2112.06745)
    - A new survey article of domain adaptation
    - 对UDA的一个综述文章，来自作者博士论文

  - [VL-Adapter: Parameter-Efficient Transfer Learning for Vision-and-Language Tasks](http://arxiv.org/abs/2112.06825)
    - Vision-language efficient transfer learning
    - 参数高校的vision-language任务迁移

  - [Federated Learning with Adaptive Batchnorm for Personalized Healthcare](https://arxiv.org/abs/2112.00734)
    - Federated learning with adaptive batchnorm
    - 用自适应BN进行个性化联邦学习

  - [Unsupervised Domain Adaptation: A Reality Check](https://arxiv.org/abs/2111.15672)
    - Doing experiments to show the progress of DA methods over the years
    - 用大量的实验来验证近几年来DA方法的进展
  
  - [Hierarchical Optimal Transport for Unsupervised Domain Adaptation](https://arxiv.org/abs/2112.02073)
    - hierarchical optimal transport for UDA
    - 层次性的最优传输用于domain adaptation

  - [Unsupervised Domain Generalization by Learning a Bridge Across Domains](https://arxiv.org/abs/2112.02300)
    - Unsupervised domain generalization
    - 无监督的domain generalization

  - [Boosting Unsupervised Domain Adaptation with Soft Pseudo-label and Curriculum Learning](https://arxiv.org/abs/2112.01948)
    - Using soft pseudo-label and curriculum learning to boost UDA
    - 用软的伪标签和课程学习增强UDA方法
  
  - [Subtask-dominated Transfer Learning for Long-tail Person Search](https://arxiv.org/abs/2112.00527)
    - Subtask-dominated transfer for long-tail person search
    - 子任务驱动的长尾人物搜索

  - [Revisiting the Transferability of Supervised Pretraining: an MLP Perspective](https://arxiv.org/abs/2112.00496)
    - Revisit the transferability of supervised pretraining
    - 重新思考有监督预训练的可迁移性
  
  - [Multi-Agent Transfer Learning in Reinforcement Learning-Based Ride-Sharing Systems](https://arxiv.org/abs/2112.00424)
    - Multi-agent transfer in RL
    - 在RL中的多智能体迁移

  - NeurIPS-21 [On Learning Domain-Invariant Representations for Transfer Learning with Multiple Sources](https://arxiv.org/abs/2111.13822)
    - Theory and algorithm of domain-invariant learning for transfer learning
    - 对invariant representation的理论和算法

  - WACV-22 [Semi-supervised Domain Adaptation via Sample-to-Sample Self-Distillation](https://arxiv.org/abs/2111.14353)
    - Sample-level self-distillation for semi-supervised DA
    - 样本层次的自蒸馏用于半监督DA

  - [ROBIN : A Benchmark for Robustness to Individual Nuisancesin Real-World Out-of-Distribution Shifts](https://arxiv.org/abs/2111.14341)
    - A benchmark for robustness to individual OOD
    - 一个OOD的benchmark

  - ICML-21 workshop [Towards Principled Disentanglement for Domain Generalization](https://arxiv.org/abs/2111.13839)
    - Principled disentanglement for domain generalization
    - Principled解耦用于domain generalization
</details>

<details>
  <summary>Latest papers (2021-11)</summary>

  - NeurIPS-21 workshop [CytoImageNet: A large-scale pretraining dataset for bioimage transfer learning](https://arxiv.org/abs/2111.11646)
    - A large-scale dataset for bioimage transfer learning
    - 一个大规模的生物图像数据集用于迁移学习
  
  - NeurIPS-21 workshop [Component Transfer Learning for Deep RL Based on Abstract Representations](https://arxiv.org/abs/2111.11525)
    - Deep transfer learning for RL
    - 深度迁移学习用于强化学习

  - NeurIPS-21 workshop [Maximum Mean Discrepancy for Generalization in the Presence of Distribution and Missingness Shift](https://arxiv.org/abs/2111.10344)
    - MMD for covariate shift
    - 用MMD来解决covariate shift问题

  - [Combined Scaling for Zero-shot Transfer Learning](https://arxiv.org/abs/2111.10050)
    - Scaling up for zero-shot transfer learning
    - 增大训练规模用于zero-shot迁移学习

  - [Federated Learning with Domain Generalization](https://arxiv.org/abs/2111.10487)
    - Federated domain generalization
    - 联邦学习+domain generalization

  - [Semi-Supervised Domain Generalization in Real World:New Benchmark and Strong Baseline](https://arxiv.org/abs/2111.10221)
    - Semi-supervised domain generalization
    - 半监督+domain generalization

  - MICCAI-21 [Domain Generalization for Mammography Detection via Multi-style and Multi-view Contrastive Learning](https://arxiv.org/abs/2111.10827)
    - Domain generalization for mammography detection
    - 领域泛化用于乳房X射线检查

  - [On Representation Knowledge Distillation for Graph Neural Networks](https://arxiv.org/abs/2111.04964)
    - Knowledge distillation for GNN
    - 适用于GNN的知识蒸馏

  - BMVC-21 [Domain Attention Consistency for Multi-Source Domain Adaptation](https://arxiv.org/abs/2111.03911)
    - Multi-source domain adaptation using attention consistency
    - 用attention一致性进行多源的domain adaptation

  - [Action Recognition using Transfer Learning and Majority Voting for CSGO](https://arxiv.org/abs/2111.03882)
    - Using transfer learning and majority voting for action recognition
    - 使用迁移学习和多数投票进行动作识别
  
  - [Open-Set Crowdsourcing using Multiple-Source Transfer Learning](https://arxiv.org/abs/2111.04073)
    - Open-set crowdsourcing using multiple-source transfer learning
    - 使用多源迁移进行开放集的crowdsourcing

  - [Improved Regularization and Robustness for Fine-tuning in Neural Networks](https://arxiv.org/abs/2111.04578)
    - Improve regularization and robustness for finetuning
    - 针对finetune提高其正则和鲁棒性

  - [TimeMatch: Unsupervised Cross-Region Adaptation by Temporal Shift Estimation](https://arxiv.org/abs/2111.02682)
    - Temporal domain adaptation

  - NeurIPS-21 [Modular Gaussian Processes for Transfer Learning](https://arxiv.org/abs/2110.13515)
    - Modular Gaussian process for transfer learning
    - 在迁移学习中使用modular Gaussian过程

  - [Estimating and Maximizing Mutual Information for Knowledge Distillation](https://arxiv.org/abs/2110.15946)
    - Global and local mutual information maximation for knowledge distillation
    - 局部和全局互信息最大化用于蒸馏

  - [On Label Shift in Domain Adaptation via Wasserstein Distance](https://arxiv.org/abs/2110.15520)
    - Using Wasserstein distance to solve label shift in domain adaptation
    - 在DA领域中用Wasserstein distance去解决label shift问题

  - [Xi-Learning: Successor Feature Transfer Learning for General Reward Functions](https://arxiv.org/abs/2110.15701)
    - General reward function transfer learning in RL
    - 在强化学习中general reward function的迁移学习

  - [C-MADA: Unsupervised Cross-Modality Adversarial Domain Adaptation framework for medical Image Segmentation](https://arxiv.org/abs/2110.15823)
    - Cross-modality domain adaptation for medical image segmentation
    - 跨模态的DA用于医学图像分割

  - [Deep Transfer Learning for Multi-source Entity Linkage via Domain Adaptation](https://arxiv.org/abs/2110.14509)
    - Domain adaptation for multi-source entiry linkage
    - 用DA进行多源的实体链接

  - [Temporal Knowledge Distillation for On-device Audio Classification](https://arxiv.org/abs/2110.14131)
    - Temporal knowledge distillation for on-device ASR
    - 时序知识蒸馏用于设备端的语音识别

  - [Transferring Domain-Agnostic Knowledge in Video Question Answering](https://arxiv.org/abs/2110.13395)
    - Domain-agnostic learning for VQA
    - 在VQA任务中进行迁移学习

</details>

<details>
  <summary>Latest papers (2021-10)</summary>

  - BMVC-21 [SILT: Self-supervised Lighting Transfer Using Implicit Image Decomposition](https://arxiv.org/abs/2110.12914)
    - Lighting transfer using implicit image decomposition
    - 用隐式图像分解进行光照迁移

  - [Domain Adaptation in Multi-View Embedding for Cross-Modal Video Retrieval](https://arxiv.org/abs/2110.12812)
    - Domain adaptation for cross-modal video retrieval
    - 用领域自适应进行跨模态的视频检索

  - [Age and Gender Prediction using Deep CNNs and Transfer Learning](https://arxiv.org/abs/2110.12633)
    - Age and gender prediction using transfer learning
    - 用迁移学习进行年龄和性别预测

  - [Domain Adaptation for Rare Classes Augmented with Synthetic Samples](https://arxiv.org/abs/2110.12216)
    - Domain adaptation for rare class
    - 稀疏类的domain adaptation

  - WACV-22 [AuxAdapt: Stable and Efficient Test-Time Adaptation for Temporally Consistent Video Semantic Segmentation](https://arxiv.org/abs/2110.12369)
    - Test-time adaptation for video semantic segmentation
    - 测试时adaptation用于视频语义分割

  - NeurIPS-21 [Unsupervised Domain Adaptation with Dynamics-Aware Rewards in Reinforcement Learning](https://arxiv.org/abs/2110.12997)
    - Domain adaptation in reinforcement learning
    - 在强化学习中应用domain adaptation

  - WACV-21 [Domain Generalization through Audio-Visual Relative Norm Alignment in First Person Action Recognition](https://arxiv.org/abs/2110.10101)
      - Domain generalization by audio-visual alignment
      - 通过音频-视频对齐进行domain generalization

  - BMVC-21 [Dynamic Feature Alignment for Semi-supervised Domain Adaptation](https://arxiv.org/abs/2110.09641)
    - Dynamic feature alignment for semi-supervised DA
    - 动态特征对齐用于半监督DA

  - NeurIPS-21 [FlexMatch: Boosting Semi-Supervised Learning with Curriculum Pseudo Labeling](https://arxiv.org/abs/2110.08263) [知乎解读](https://zhuanlan.zhihu.com/p/422930830) [code](https://github.com/TorchSSL/TorchSSL)
    - Curriculum pseudo label with a unified codebase TorchSSL
    - 半监督方法FlexMatch和统一算法库TorchSSL

  - [Rethinking supervised pre-training for better downstream transferring](https://arxiv.org/abs/2110.06014)
    - Rethink better finetune
    - 重新思考预训练以便更好finetune

  - [Music Sentiment Transfer](https://arxiv.org/abs/2110.05765)
    - Music sentiment transfer learning
    - 迁移学习用于音乐sentiment

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
  - **《迁移学习》（杨强）** [[Buy](https://item.jd.com/12930984.html)] [[English version](https://www.cambridge.org/core/books/transfer-learning/CCFFAFE3CDBC245047F1DEC71D9EF3C7)]
  - **《迁移学习导论》(王晋东、陈益强著)** [[Homepage](http://jd92.wang/tlbook)] [[Buy](https://item.jd.com/13283188.html)]

- Blogs 博客
  - [Zhihu blogs - 知乎专栏《小王爱迁移》系列文章](https://zhuanlan.zhihu.com/p/130244395)
	
- Video tutorials 视频教程 
  - [Recent advance of transfer learning - 2021年最新迁移学习发展现状探讨](https://www.bilibili.com/video/BV1N5411T7Sb)
  - [Definitions of transfer learning area - 迁移学习领域名词解释](https://www.bilibili.com/video/BV1fu411o7BW) [[Article](https://zhuanlan.zhihu.com/p/428097044)]
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
