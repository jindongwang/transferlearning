# Awesome Transfer Learning Papers

Let's read some awesome transfer learning / domain adaptation papers.

Here, we list some papers by topic. For list by date, please refer to [papers by date](awesome_paper.md).

这里收录了迁移学习各个研究领域的最新文章。

- [Awesome Transfer Learning Papers](#awesome-transfer-learning-papers)
  - [Survey](#survey)
  - [Large models](#large-models)
  - [Theory](#theory)
  - [Per-training/Finetuning](#per-trainingfinetuning)
  - [Knowledge distillation](#knowledge-distillation)
  - [Traditional domain adaptation](#traditional-domain-adaptation)
  - [Deep domain adaptation](#deep-domain-adaptation)
  - [Domain generalization](#domain-generalization)
    - [Survey](#survey-1)
    - [Tutorial](#tutorial)
    - [Papers](#papers)
  - [Source-free domain adaptation](#source-free-domain-adaptation)
  - [Multi-source domain adaptation](#multi-source-domain-adaptation)
  - [Heterogeneous transfer learning](#heterogeneous-transfer-learning)
  - [Online transfer learning](#online-transfer-learning)
  - [Zero-shot / few-shot learning](#zero-shot--few-shot-learning)
  - [Multi-task learning](#multi-task-learning)
  - [Transfer reinforcement learning](#transfer-reinforcement-learning)
  - [Transfer metric learning](#transfer-metric-learning)
  - [Federated transfer learning](#federated-transfer-learning)
  - [Lifelong transfer learning](#lifelong-transfer-learning)
  - [Safe transfer learning](#safe-transfer-learning)
  - [Transfer learning applications](#transfer-learning-applications)


## Survey

- A Survey of Heterogeneous Transfer Learning [[arxiv](https://arxiv.org/abs/2310.08459v2)]
  - A recent survey of heterogeneous transfer learning 一篇最近的关于异构迁移学习的综述
- Review of Large Vision Models and Visual Prompt Engineering [[arxiv](https://arxiv.org/abs/2307.00855)]
  - A survey of large vision model and prompt tuning 一个关于大视觉模型的prompt tuning的综述
- IEEE TNNLS-22 [Towards Personalized Federated Learning](http://arxiv.org/abs/2103.00710)
  - A survey on personalized federated learning 一个关于个性化联邦学习的综述
- 2022 [Transfer Learning for Future Wireless Networks: A Comprehensive Survey](https://arxiv.org/abs/2102.07572)
- 2022 [A Review of Deep Transfer Learning and Recent Advancements](https://arxiv.org/abs/2201.09679)

- 2022 [Transferability in Deep Learning: A Survey](https://paperswithcode.com/paper/transferability-in-deep-learning-a-survey)

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

- [Data-Free Knowledge Transfer: A Survey](https://arxiv.org/abs/2112.15278)
  - A survey on data-free distillation and source-free DA
  - 一篇关于data-free蒸馏和source-free DA的综述

## Large models

- ICLR'24-spotlight Understanding and Mitigating the Label Noise in Pre-training on Downstream Tasks [[arxiv](https://arxiv.org/abs/2309.17002)]
  - Noisy model learning: fine-tuning to supress the bad effect of noisy pretraining data 通过使用轻量级finetune减少噪音预训练数据对下游任务的影响

- ZooPFL: Exploring Black-box Foundation Models for Personalized Federated Learning [[arxiv](https://arxiv.org/abs/2310.05143)]
  - Black-box foundation models for personalized federated learning 黑盒的blackbox模型进行个性化迁移学习

- IJCV'23 Exploring Vision-Language Models for Imbalanced Learning [[arxiv](https://arxiv.org/abs/2304.01457)] [[code](https://github.com/Imbalance-VLM/Imbalance-VLM)]
  - Explore vision-language models for imbalanced learning 探索视觉大模型在不平衡问题上的表现

- ICCV'23 Improving Generalization of Adversarial Training via Robust Critical Fine-Tuning [[arxiv](https://arxiv.org/abs/2308.02533)] [[code](https://github.com/microsoft/robustlearn)]
  - 达到对抗鲁棒性和泛化能力的trade off 

- Towards Realistic Unsupervised Fine-tuning with CLIP [[arxiv](http://arxiv.org/abs/2308.12919)]
  - Unsupervised fine-tuning of CLIP

## Theory

- [Improved OOD Generalization via Conditional Invariant Regularizer](https://arxiv.org/abs/2207.06687)
  - Improved OOD generalization via conditional invariant regularizer 通过条件不变正则进行OOD泛化

- [An Information-Theoretic Analysis for Transfer Learning: Error Bounds and Applications](https://arxiv.org/abs/2207.05377)
  - Information-theoretic analysis for transfer learning 用信息理论解释迁移学习

- [PAC-Bayesian Domain Adaptation Bounds for Multiclass Learners](https://arxiv.org/abs/2207.05685)
  - PAC-Bayesian domain adaptation 基于PAC-Bayesian的domain adaptation

- [Optimal Representations for Covariate Shift](https://arxiv.org/abs/2201.00057)
  - Learning optimal representations for covariate shift
  - 为covariate shift学习最优的表达

- NeurIPS-21 [On Learning Domain-Invariant Representations for Transfer Learning with Multiple Sources](https://arxiv.org/abs/2111.13822)
    - Theory and algorithm of domain-invariant learning for transfer learning
    - 对invariant representation的理论和算法

- 20210625 ICML-21 [f-Domain-Adversarial Learning: Theory and Algorithms](http://arxiv.org/abs/2106.11344)
    - New theory based on f-divergence
    - 基于f-divergence给出新的DA理论和算法

- 20210521 [When is invariance useful in an Out-of-Distribution Generalization problem ?](http://arxiv.org/abs/2008.01883)
    - When is invariant useful in OOD?
    - 理论上分析了在OOD问题中invariance什么时候有用

- 20200220 [Butterfly: One-step Approach towards Wildly Unsupervised Domain Adaptation](http://arxiv.org/abs/1905.07720)
    - Noisy domain adaptation
    - 用于噪声环境中的domain adaptation的方法

- 20210127 [A Unified Joint Maximum Mean Discrepancy for Domain Adaptation](http://arxiv.org/abs/2101.09979)
    - 一个理论上更一般化的MMD差异用于领域自适应
    - A more general MMD for domain adaptation

- 20200615 [Double Double Descent: On Generalization Errors in Transfer Learning between Linear Regression Tasks](https://arxiv.org/abs/2006.07002)

- 20200813 [A Boundary Based Out-of-Distribution Classifier for Generalized Zero-Shot Learning](https://arxiv.org/abs/2008.04872)
    - OOD classifier for generalized zero-shot learning

- 20200813 ICML-20 [On Learning Language-Invariant Representations for Universal Machine Translation](https://arxiv.org/abs/2008.04510)
    - Theory for universal machine translation
    - 对统一机器翻译模型进行了理论论证

- 20200702 ICML-20 [Few-shot domain adaptation by causal mechanism transfer](https://arxiv.org/pdf/2002.03497.pdf)
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

- 20180724 arXiv [Generalization Bounds for Unsupervised Cross-Domain Mapping with WGANs](https://arxiv.org/abs/1807.08501)
	-  Provide a generalization bound for unsupervised WGAN in transfer learning
	- 对迁移学习中无监督的WGAN进行了一些理论上的分析

## Per-training/Finetuning

- Facing the Elephant in the Room: Visual Prompt Tuning or Full Finetuning? [[arxiv](https://arxiv.org/abs/2401.12902)]
  - A comparison between visual prompt tuning and full finetuning 比较prompt tuning和全finetune

- ICLR'24 spotlight Understanding and Mitigating the Label Noise in Pre-training on Downstream Tasks [[arxiv](https://arxiv.org/abs/2309.17002)]
  - A new research direction of transfer learning in the era of foundation models 大模型时代一个新研究方向：研究预训练数据的噪声对下游任务影响

- NeurIPS'23 Geodesic Multi-Modal Mixup for Robust Fine-Tuning [[paper](https://openreview.net/forum?id=iAAXq60Bw1)]
  - Geodesic mixup for robust fine-tuning

- NeurIPS'23 Parameter and Computation Efficient Transfer Learning for Vision-Language Pre-trained Models [[paper](https://openreview.net/forum?id=TPeAmxwPK2)]
  - Parameter and computation efficient transfer learning by reinforcement learning

- Equivariant Adaptation of Large Pre-Trained Models [[arxiv](http://arxiv.org/abs/2310.01647)]
  - Equivariant adaptation of large pre-trained models 对大模型进行等边自适应

- Effective and Parameter-Efficient Reusing Fine-Tuned Models [[arxiv](http://arxiv.org/abs/2310.01886)]
  - Effective and parameter-efficient reusing fine-tuned models 高效使用预训练模型

- Understanding and Mitigating the Label Noise in Pre-training on Downstream Tasks [[arxiv](https://arxiv.org/abs/2309.17002)]
  - Noisy model learning: fine-tuning to supress the bad effect of noisy pretraining data 通过使用轻量级finetune减少噪音预训练数据对下游任务的影响

- DePT: Decomposed Prompt Tuning for Parameter-Efficient Fine-tuning [[arxiv](http://arxiv.org/abs/2309.05173)]
  - Decomposed prompt tuning for parameter-efficient fine-tuning 基于分解prompt tuning的参数高效微调

- Towards Realistic Unsupervised Fine-tuning with CLIP [[arxiv](http://arxiv.org/abs/2308.12919)]
  - Unsupervised fine-tuning of CLIP

- Fine-tuning can cripple your foundation model; preserving features may be the solution [[arxiv](http://arxiv.org/abs/2308.13320)]
  - Fine-tuning will cripple foundation model

- Unified Transfer Learning Models for High-Dimensional Linear Regression [[arxiv](https://arxiv.org/abs/2307.00238)]
  - Transfer learning for high-dimensional linar regression 迁移学习用于高维线性回归

- Review of Large Vision Models and Visual Prompt Engineering [[arxiv](https://arxiv.org/abs/2307.00855)]
  - A survey of large vision model and prompt tuning 一个关于大视觉模型的prompt tuning的综述

- Scaling Down to Scale Up: A Guide to Parameter-Efficient Fine-Tuning [[arxiv](http://arxiv.org/abs/2303.15647)]
  - A guide for parameter-efficient fine-tuning 一个对parameter efficient fine-tuning的全面介绍

- ICML'23 A Kernel-Based View of Language Model Fine-Tuning [[arxiv](http://arxiv.org/abs/2210.05643)]
  - A kernel-based view of language model fine-tuning 一种以kernel的视角来看待fine-tuning的方法

- ICML'23 Improving Visual Prompt Tuning for Self-supervised Vision Transformers [[arxiv](http://arxiv.org/abs/2306.05067)]
  - Improving visual prompt tuning for self-supervision 为自监督模型提高其 prompt tuning 表现

- Adapting Pre-trained Language Models to Vision-Language Tasks via Dynamic Visual Prompting [[arxiv](http://arxiv.org/abs/2306.00409)]
  - Using dynamic visual prompting for model adaptation 用动态视觉prompt进行模型适配

- ACL'23 Parameter-Efficient Fine-Tuning without Introducing New Latency [[arxiv](http://arxiv.org/abs/2305.16742)]
  - Parameter-efficient finetuning 参数高效的finetune

- Ahead-of-Time P-Tuning [[arxiv](http://arxiv.org/abs/2305.10835)]
  - Ahead-ot-time P-tuning for language models

- Parameter-Efficient Tuning Makes a Good Classification Head [[arxiv](http://arxiv.org/abs/2210.16771)]
  - Parameter-efficient tuning makes a good classification head 参数高效的迁移学习成就一个好的分类头

- CVPR'23 Trainable Projected Gradient Method for Robust Fine-tuning [[arxiv](http://arxiv.org/abs/2303.10720)]
  - Trainable PGD for robust fine-tuning 可训练的pgd用于鲁棒的微调技术

- ICLR'23 Contrastive Alignment of Vision to Language Through Parameter-Efficient Transfer Learning [[arxiv](http://arxiv.org/abs/2303.11866)]
  - Contrastive alignment for vision language models using transfer learning 使用参数高效迁移进行视觉语言模型的对比对齐

- ICLR'23 workshop SPDF: Sparse Pre-training and Dense Fine-tuning for Large Language Models [[arxiv](http://arxiv.org/abs/2303.10464)]
  - Sparse pre-training and dense fine-tuning

- A Unified Continual Learning Framework with General Parameter-Efficient Tuning [[arxiv](http://arxiv.org/abs/2303.10070)]
  - A continual learning framework for parameter-efficient tuning 一个对于参数高效迁移的连续学习框架

- Transfer Learning for Real-time Deployment of a Screening Tool for Depression Detection Using Actigraphy [[arxiv](https://arxiv.org/abs/2303.07847)]
  - Transfer learning for Depression detection 迁移学习用于脉动计焦虑检测

- ICLR'23 AutoTransfer: AutoML with Knowledge Transfer -- An Application to Graph Neural Networks [[arxiv](https://arxiv.org/abs/2303.07669)]
  - GNN with autoML transfer learning 用于GNN的自动迁移学习

- Revisit Parameter-Efficient Transfer Learning: A Two-Stage Paradigm [[arxiv](https://arxiv.org/abs/2303.07910)]
  - Parameter-efficient transfer learning: a two-stage approach 一种两阶段的参数高效迁移学习

- To Stay or Not to Stay in the Pre-train Basin: Insights on Ensembling in Transfer Learning [[arxiv](https://arxiv.org/abs/2303.03374)]
  - Ensembling in transfer learning 调研迁移学习中的集成

- CVPR'13 Masked Images Are Counterfactual Samples for Robust Fine-tuning [[arxiv](https://arxiv.org/abs/2303.03052)]
  - Masked images for robust fine-tuning 调研masked image对于fine-tuning的影响

- Finetune like you pretrain: Improved finetuning of zero-shot vision models [[arxiv]](http://arxiv.org/abs/2212.00638)]
  - Improved fine-tuning of zero-shot models 针对zero-shot model提高fine-tuneing

- CVPR'22 Does Robustness on ImageNet Transfer to Downstream Tasks? [[arxiv](https://openaccess.thecvf.com/content/CVPR2022/papers/Yamada_Does_Robustness_on_ImageNet_Transfer_to_Downstream_Tasks_CVPR_2022_paper.pdf)]
  - Does robustness on imagenet transfer lto downstream tasks?

- NeurIPS'22 Improved Fine-Tuning by Better Leveraging Pre-Training Data [[openreview](https://openreview.net/forum?id=YTXIIc7cAQ)]
  - Using pre-training data for fine-tuning 用预训练数据来做微调

- On Fine-Tuned Deep Features for Unsupervised Domain Adaptation [[arxiv](http://arxiv.org/abs/2210.14083)]
  - Fine-tuned features for domain adaptation 微调的特征用于域自适应

- Transfer of Machine Learning Fairness across Domains [[arxiv](http://arxiv.org/abs/1906.09688)]
  - Fairness transfer in transfer learning 迁移学习中的公平性迁移

- CVPR-20 Regularizing CNN Transfer Learning With Randomised Regression [[arxiv](https://openaccess.thecvf.com/content_CVPR_2020/html/Zhong_Regularizing_CNN_Transfer_Learning_With_Randomised_Regression_CVPR_2020_paper.html)]
  - Using randomized regression to regularize CNN 用随机回归约束CNN迁移学习

- AAAI-21 TransTailor: Pruning the Pre-trained Model for Improved Transfer Learning [[arxiv](https://ojs.aaai.org/index.php/AAAI/article/view/17046)]
  - Pruning pre-trained model for transfer learning 通过对预训练模型进行剪枝来进行迁移学习

- Test-Time Training with Masked Autoencoders [[arxiv](https://arxiv.org/abs/2209.07522)]
  - Test-time training with MAE MAE的测试时训练

- Test-Time Prompt Tuning for Zero-Shot Generalization in Vision-Language Models [[arxiv](https://arxiv.org/abs/2209.07511)]
  - Test-time prompt tuning 测试时的prompt tuning

- TeST: test-time self-training under distribution shift [[arxiv](https://assets.amazon.science/02/1c/b469914c4732a9c29ac765f948f9/test-test-time-self-training-under-distribution-shift.pdf)]
  - Test-time self-training 测试时的self-training

- [Conv-Adapter: Exploring Parameter Efficient Transfer Learning for ConvNets](https://arxiv.org/pdf/2208.07463.pdf)
  - Parameter efficient CNN adapter for transfer learning 参数高效的CNN adapter用于迁移学习

- [Hyper-Representations for Pre-Training and Transfer Learning](https://arxiv.org/abs/2207.10951)
  - Hyper-representation for pre-training and fine-tuning 对于预训练和微调的超表示

- [Zero-Shot AutoML with Pretrained Models](https://arxiv.org/abs/2206.08476)
  - 用预训练模型进行零样本的自动机器学习 

- [How robust are pre-trained models to distribution shift?](https://arxiv.org/abs/2206.08871)
  - How robust are pre-trained models to distribution shift 评估预训练模型对于distribution shift的鲁棒性

- [Wav2vec-S: Semi-Supervised Pre-Training for Speech Recognition](https://arxiv.org/abs/2110.04484)
  - Pretraining for speech recognition 用预训练模型进行语音识别

- [ScholarBERT: Bigger is Not Always Better](https://arxiv.org/abs/2205.11342)
  - Empirical study on fine-tuning experiments 提出ScholarBERT进行大规模finetuning实验

- [A Domain-adaptive Pre-training Approach for Language Bias Detection in News](https://arxiv.org/abs/2205.10773)
  - Domain-adaptive pre-training for language bias detection 领域适配预训练用于新闻语言偏见检测

- IJCAI-22 [Parameter-Efficient Sparsity for Large Language Models Fine-Tuning](https://arxiv.org/abs/2205.11005)
  - Parameter-efficient sparsity for language model fine-tuning 参数高效的稀疏学习用于语言模型微调

- NAACL-22 [Efficient Few-Shot Fine-Tuning for Opinion Summarization](https://arxiv.org/abs/2205.02170)
  - Few-shot fine-tuning for opinion summarization 小样本微调技术用于评论总结

- ACL-22 [Probing Simile Knowledge from Pre-trained Language Models](https://arxiv.org/abs/2204.12807)
  - Probe simile knowledge from pre-trained model 从预训练模型中找出明喻知识

- [Transfer Learning with Pre-trained Conditional Generative Models](https://arxiv.org/abs/2204.12833)
  - Transfer learning with pre-trained conditional generative models 条件生成模型用于迁移学习

- ICLR-22 [Towards a Unified View of Parameter-Efficient Transfer Learning](https://openreview.net/forum?id=0RDcd5Axok)
  - Unified view of parameter-efficient transfer learning 一个统一视角看待参数高效的迁移学习

- ICLR-22 [Exploring the Limits of Large Scale Pre-training](https://openreview.net/forum?id=V3C8p78sDa)
  - Many experiments to explore pre-training  许多实验来探索预训练

- [Just Fine-tune Twice: Selective Differential Privacy for Large Language Models](https://arxiv.org/abs/2204.07667)
  - Differential privacy by just fine-tune twice 通过微调两次进行差分隐私

- [On Effectively Learning of Knowledge in Continual Pre-training](https://arxiv.org/abs/2204.07994)
  - Continual per-training 持续的预训练

- NAACL-22 [GRAM: Fast Fine-tuning of Pre-trained Language Models for Content-based Collaborative Filtering](https://arxiv.org/abs/2204.04179)
  - Fast fine-tuning for content-based collaborative filtering
  - 快速的适用于协同过滤的微调

- AAAI-22 [Powering Finetuning in Few-shot Learning: Domain-Agnostic Feature Adaptation with Rectified Class Prototypes](https://arxiv.org/abs/2204.03749)
  - Finetuning in few-shot learning
  - 小样本学习中的微调

- CVPR-22 [Does Robustness on ImageNet Transfer to Downstream Tasks?](https://arxiv.org/abs/2204.03934)
  - Transfer learning robustness
  - 迁移学习鲁棒性

- [Blockchain as an Enabler for Transfer Learning in Smart Environments](https://arxiv.org/abs/2204.03959)
  - Blockchain transfer learning
  - 用区块链进行迁移学习

- ICLR-22 [Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution](https://openreview.net/forum?id=UYneFzXSJWh)
  - Fin-tuning and linear probing for ood generalization
  - 先linear probing最后一层再finetune对OOD任务最好

- [A Broad Study of Pre-training for Domain Generalization and Adaptation](https://arxiv.org/abs/2203.11819)
  - A broad study of pre-training models for DA and DG
  - 大量的实验进行DA和DG

- ACL-22 [Language-Agnostic Meta-Learning for Low-Resource Text-to-Speech with Articulatory Features](https://arxiv.org/abs/2203.03191)
  - Language-agnostic meta-learning for TTS
  - 语言无关的元学习用于TTS

- [Input-Tuning: Adapting Unfamiliar Inputs to Frozen Pretrained Models](https://arxiv.org/abs/2203.03131)
  - Adapt unfamiliar inputs to frozen pretrained models
  - 让固定的预训练模型适配不熟悉的输入

- [Pre-trained Token-replaced Detection Model as Few-shot Learner](https://arxiv.org/abs/2203.03235)
  - Pre-trained token-replaced detection model as few-shot learner
  - 预训练的替换token的检测模型

- ICLR-22 spotlight [Towards a Unified View of Parameter-Efficient Transfer Learning](https://openreview.net/pdf?id=0RDcd5Axok)
  - Detailed analysis of parameter-efficient transfer learning
  - 对参数高效的迁移学习进行分析

- ICLR-22 [BEiT: BERT Pre-Training of Image Transformers](https://openreview.net/forum?id=p-BhZSz59o4)
  - BERT pre-training of image transformers
  - 用BERT的方式pre-train transformer

- [Improved Fine-tuning by Leveraging Pre-training Data: Theory and Practice](http://arxiv.org/abs/2111.12292)
  - Using pre-training data to improve fine-tuning
  - 使用预训练数据来帮助finetune

- [An Ensemble of Pre-trained Transformer Models For Imbalanced Multiclass Malware Classification](https://arxiv.org/abs/2112.13236)
  - An ensemble of pre-trained transformer for malware classification
  - 预训练的transformer通过集成进行恶意软件检测

- [SLIP: Self-supervision meets Language-Image Pre-training](https://arxiv.org/abs/2112.12750)
    - Self-supervised learning + language image pretraining
    - 用自监督学习用于语言到图像的预训练

- [VL-Adapter: Parameter-Efficient Transfer Learning for Vision-and-Language Tasks](http://arxiv.org/abs/2112.06825)
    - Vision-language efficient transfer learning
    - 参数高校的vision-language任务迁移

- [Revisiting the Transferability of Supervised Pretraining: an MLP Perspective](https://arxiv.org/abs/2112.00496)
    - Revisit the transferability of supervised pretraining
    - 重新思考有监督预训练的可迁移性

- NeurIPS-21 workshop [Maximum Mean Discrepancy for Generalization in the Presence of Distribution and Missingness Shift](https://arxiv.org/abs/2111.10344)
    - MMD for covariate shift
    - 用MMD来解决covariate shift问题

- [Combined Scaling for Zero-shot Transfer Learning](https://arxiv.org/abs/2111.10050)
    - Scaling up for zero-shot transfer learning
    - 增大训练规模用于zero-shot迁移学习

- [Improved Regularization and Robustness for Fine-tuning in Neural Networks](https://arxiv.org/abs/2111.04578)
    - Improve regularization and robustness for finetuning
    - 针对finetune提高其正则和鲁棒性

- NeurIPS-21 [Modular Gaussian Processes for Transfer Learning](https://arxiv.org/abs/2110.13515)
    - Modular Gaussian process for transfer learning
    - 在迁移学习中使用modular Gaussian过程

- [Rethinking supervised pre-training for better downstream transferring](https://arxiv.org/abs/2110.06014)
    - Rethink better finetune
    - 重新思考预训练以便更好finetune

- EMNLP-21 [Few-Shot Intent Detection via Contrastive Pre-Training and Fine-Tuning](https://arxiv.org/abs/2109.06349)
    - Few-shot intent detection using pretrain and finetune
    - 用迁移学习进行少样本意图检测

- [KroneckerBERT: Learning Kronecker Decomposition for Pre-trained Language Models via Knowledge Distillation](https://arxiv.org/abs/2109.06243)
    - Using Kronecker decomposition and knowledge distillation for pre-trained language models compression
    - 用Kronecker分解和知识蒸馏来进行语言模型的压缩

- [How Does Adversarial Fine-Tuning Benefit BERT?](https://arxiv.org/abs/2108.13602)
    - Examine how does adversarial fine-tuning help BERT
    - 探索对抗性finetune如何帮助BERT

- [A Data Augmented Approach to Transfer Learning for Covid-19 Detection](https://arxiv.org/abs/2108.02870)
    - Data augmentation to transfer learning for COVID
    - 迁移学习使用数据增强，用于COVID-19

- [Finetuning Pretrained Transformers into Variational Autoencoders](https://arxiv.org/abs/2108.02446)
    - Finetune transformer to VAE
    - 把transformer迁移到VAE

- [Pre-trained Models for Sonar Images](http://arxiv.org/abs/2108.01111)
    - Pre-trained models for sonar images
    - 针对声纳图像的预训练模型

- [Domain Adaptor Networks for Hyperspectral Image Recognition](http://arxiv.org/abs/2108.01555)
  - Finetune for hyperspectral image recognition
  - 针对高光谱图像识别的迁移学习

- CVPR-21 [Efficient Conditional GAN Transfer With Knowledge Propagation Across Classes](https://openaccess.thecvf.com/content/CVPR2021/html/Shahbazi_Efficient_Conditional_GAN_Transfer_With_Knowledge_Propagation_Across_Classes_CVPR_2021_paper.html)
    - Transfer conditional GANs to unseen classes
    - 通过知识传递，迁移预训练的conditional GAN到新类别

- 20191011 arXiv [Estimating Transfer Entropy via Copula Entropy](https://arxiv.org/abs/1910.04375)
  	- Evaluate the transfer entopy via copula entropy
  	- 评估迁移熵

- 20180925 arXiv [DT-LET: Deep Transfer Learning by Exploring where to Transfer](https://arxiv.org/pdf/1809.08541.pdf)
	-  Explore the suitable layers to transfer
	- 探索深度网络中效果表现好的对应的迁移层

- 20200629 [Transfer learning via L1 regulaziation](https://arxiv.org/abs/2006.14845)
	- Using L1 regularizationg for transfer learning

- 20201016 [Deep Ensembles for Low-Data Transfer Learning](https://arxiv.org/abs/2010.06866)
    - Deep ensemble models for transfer learning

- 20210511 ACL-21 [Are Pre-trained Convolutions Better than Pre-trained Transformers?](https://arxiv.org/abs/2105.03322)
    - Empirically investigate pre-trained convolutions and Transformers
    - 设计实验探索预训练的卷积和Transformer的对比

- 20190111 arXiv [Transfer Representation Learning with TSK Fuzzy System](https://arxiv.org/abs/1901.02703)
    - Transfer learning with fuzzy system
    - 基于模糊系统的迁移学习

- 20201203 [Pre-Trained Image Processing Transformer](https://arxiv.org/abs/2012.00364)
    - 用transformer做low-level的图像任务

- 20200821 [Self-Supervised Learning Across Domains](https://arxiv.org/abs/2007.12368)
    - 跨领域自监督学习

- 20180403 arXiv 选择最优的子类生成方便迁移的特征：[Class Subset Selection for Transfer Learning using Submodularity](https://arxiv.org/abs/1804.00060)

- 20180326 ICMLA-17 在类别不平衡情况下比较了一些迁移学习和传统方法的性能，并做出一些结论：[Comparing Transfer Learning and Traditional Learning Under Domain Class Imbalance](http://ieeexplore.ieee.org/abstract/document/8260654/)

- 20190626 arXiv [Transfer of Machine Learning Fairness across Domains](https://arxiv.org/abs/1906.09688)
  	- Transfer of machine learning fairness across domains
  	- 机器学习的公平性的迁移

- 20200615 [Rethinking Pre-training and Self-training](https://arxiv.org/abs/2006.06882)

- 20200210 WACVW-20 [Impact of ImageNet Model Selection on Domain Adaptation](https://arxiv.org/abs/2002.02559)
  	- A good experiment paper to indicate the power of representations
  	- 一篇很好的实验paper，揭示了深度特征+传统方法的有效性

- 20191015 arXiv [The Visual Task Adaptation Benchmark](https://arxiv.org/abs/1910.04867)
  	- A new large benchmark for visual adaptation tasks by Google
  	- Google提出的一个巨大的视觉迁移任务数据集

- 20190305 arXiv [Let's Transfer Transformations of Shared Semantic Representations](https://arxiv.org/abs/1903.00793)
    - Transfer transformations from shared semantic representations
    - 从共享的语义表示中进行特征迁移

- 20190409 arXiv [Improving Image Classification Robustness through Selective CNN-Filters Fine-Tuning](https://arxiv.org/abs/1904.03949)
    - Improving Image Classification Robustness through Selective CNN-Filters Fine-Tuning
    - 通过可选择的CNN滤波器进行图像分类的fine-tuning

- 20190111 arXiv [Low-Cost Transfer Learning of Face Tasks](https://arxiv.org/abs/1901.02675)
    - Infer what task transfers better and how to transfer
    - 探索对于一个预训练好的网络来说哪个任务适合迁移、如何迁移

- 20191119 ICDM-19 [Towards Making Deep Transfer Learning Never Hurt](https://arxiv.org/abs/1911.07489)
  	- Towards making deep transfer learning never hurt
  	- 通过正则避免负迁移

- 20181123 arXiv [SpotTune: Transfer Learning through Adaptive Fine-tuning](https://arxiv.org/abs/1811.08737)
	-  Very interesting work: how exactly determine the finetune process?
	- 很有意思的工作：如何决定finetune的策略？

- 20180425 arXiv 探索各个层对于迁移任务的作用，方便以后的迁移。比较有意思：[CactusNets: Layer Applicability as a Metric for Transfer Learning](https://arxiv.org/abs/1804.07846)

- 传递迁移学习的第一篇文章，来自杨强团队，发表在KDD-15上：[Transitive Transfer Learning](http://dl.acm.org/citation.cfm?id=2783295)

- AAAI-17 杨强团队最新的传递迁移学习：[Distant Domain Transfer Learning](http://www3.ntu.edu.sg/home/sinnopan/publications/[AAAI17]Distant%20Domain%20Transfer%20Learning.pdf)

- 20180819 LNCS-2018 [Distant Domain Adaptation for Text Classification](https://link.springer.com/chapter/10.1007/978-3-319-99365-2_5)
	-  Propose a selected algorithm for distant domain text classification
	- 提出一个用于远域的文本分类方法

- 20190220 arXiv [Fully-Featured Attribute Transfer](https://arxiv.org/abs/1902.06258)
    - Fully-featured image attribute transfer
	- 图像特征迁移

- 20190926 arXiv [Transfer Learning across Languages from Someone Else's NMT Model](https://arxiv.org/abs/1909.10955)
  	- Transfer learning across languages from NMT pretrained model
  	- 利用预训练好的NMT模型进行迁移学习

- 20190929 NeurIPS-19 [Deep Model Transferability from Attribution Maps](https://arxiv.org/abs/1909.11902)
  	- Using attribution map for network similarity
  	- 与cvpr18的taskmony类似，这次用了属性图的方式探索网络的相似性

- 20181121 arXiv [An Efficient Transfer Learning Technique by Using Final Fully-Connected Layer Output Features of Deep Networks](https://arxiv.org/abs/1811.07459)
    -  Using final fc layer to perform transfer learning
    - 使用最后一层全连接层进行迁移学习

- ICML-14 著名的DeCAF特征：[DeCAF: A Deep Convolutional Activation Feature for Generic Visual Recognition](https://arxiv.org/abs/1310.1531.pdf)

- [Simultaneous Deep Transfer Across Domains and Tasks](https://www.cv-foundation.org/openaccess/content_iccv_2015/html/Tzeng_Simultaneous_Deep_Transfer_ICCV_2015_paper.html)
	- 发表在ICCV-15上，在传统深度迁移方法上又加了新东西
	- [我的解读](https://zhuanlan.zhihu.com/p/30621691)

## Knowledge distillation

- NeurIPS'22 Respecting Transfer Gap in Knowledge Distillation [[arxiv](http://arxiv.org/abs/2210.12787)]
  - Transfer gap in distillation 知识蒸馏中的迁移gap

- [Cross-Architecture Knowledge Distillation](https://arxiv.org/abs/2207.05273)
  - Cross-architecture knowledge distillation 跨架构的知识蒸馏

- ECCV-22 [Knowledge Condensation Distillation](https://arxiv.org/abs/2207.05409)
  - Knowledge condensation distillation 知识压缩蒸馏

- [FreeMatch: Self-adaptive Thresholding for Semi-supervised Learning](https://arxiv.org/abs/2205.07246)
  - Self-adaptive thresholding for semi-supervised learning 新的自适应阈值半监督方法

- TIP-22 [Spot-adaptive Knowledge Distillation](https://arxiv.org/abs/2205.02399)
  - Spot-adaptive knowledge distillation 层次自适应的知识蒸馏

- CVPR-22 [Decoupled Knowledge Distillation](https://arxiv.org/abs/2203.08679)
  - Decoupled knowledge distillation
  - 解耦的知识蒸馏

- [On Representation Knowledge Distillation for Graph Neural Networks](https://arxiv.org/abs/2111.04964)
    - Knowledge distillation for GNN
    - 适用于GNN的知识蒸馏

- [Estimating and Maximizing Mutual Information for Knowledge Distillation](https://arxiv.org/abs/2110.15946)
    - Global and local mutual information maximation for knowledge distillation
    - 局部和全局互信息最大化用于蒸馏

- 20210426 [Distill on the Go: Online knowledge distillation in self-supervised learning](http://arxiv.org/abs/2104.09866)
    - Online knowledge distillation in self-supervised learning
    - 自监督学习中的在线知识蒸馏

- 20210202 ICLR-21 [Rethinking Soft Labels for Knowledge Distillation: A Bias-Variance Tradeoff Perspective](https://arxiv.org/abs/2102.00650)
    - Rethink soft labels for KD in a bias-variance tradeoff perspective
    - 从偏差-方差的角度重新思考蒸馏中的软标签

- 20200706 [Interactive Knowledge Distillation](https://arxiv.org/abs/2007.01476)

- 20200412 ICML-19 [Towards understanding knowledge distillation](http://proceedings.mlr.press/v97/phuong19a.html)
  	- Some theoretical and empirical understanding to knowledge distllation
  	- 对知识蒸馏的一些理论和实验的分析

- 20191202 AAAI-20 [Towards Oracle Knowledge Distillation with Neural Architecture Search](https://arxiv.org/abs/1911.13019)
   - Using NAS for knowledge Distillation
   - 用NAS帮助知识蒸馏

- 20190401 arXiv [Distilling Task-Specific Knowledge from BERT into Simple Neural Networks](https://arxiv.org/abs/1903.12136)
    - Distill knowledge from BERT to simple neural networks
    - 从BERT模型中迁移知识到简单网络中

- 20181207 arXiv [Feature Matters: A Stage-by-Stage Approach for Knowledge Transfer](https://arxiv.org/abs/1812.01819)
	-  Feature transfer in student-teacher networks
	- 在学生-教师网络中进行特征迁移

- 20191204 AAAI-20 [Online Knowledge Distillation with Diverse Peers](https://arxiv.org/abs/1912.00350)
    - Online Knowledge Distillation with Diverse Peers

- 20191222 AAAI-20 [Improved Knowledge Distillation via Teacher Assistant](https://arxiv.org/abs/1902.03393)
    - Teacher assistant helps knowledge distillation

- - -

## Traditional domain adaptation

- [On Label Shift in Domain Adaptation via Wasserstein Distance](https://arxiv.org/abs/2110.15520)
    - Using Wasserstein distance to solve label shift in domain adaptation
    - 在DA领域中用Wasserstein distance去解决label shift问题

- 20210319 [Cross-domain Activity Recognition via Substructural Optimal Transport](https://arxiv.org/abs/2102.03353) | [知乎文章](https://zhuanlan.zhihu.com/p/356904023) | [微信公众号](https://mp.weixin.qq.com/s/QuVrqnPruHgfolYltI1Peg)
    - Using sub-structures for domain adaptation
    - 采用子结构进行domain adaptation，比传统方法快5倍

- 20210607 ICML-21 [Sequential Domain Adaptation by Synthesizing Distributionally Robust Experts](http://arxiv.org/abs/2106.00322)
  - Sequential DA using distributionally robust experts
  - 用鲁棒专家模型进行连续式领域自适应

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

## Deep domain adaptation

- Unsupervised Domain Adaptation within Deep Foundation Latent Spaces [[arxiv](https://arxiv.org/abs/2402.14976)]
  - Domain adaptation using foundation models

- LanDA: Language-Guided Multi-Source Domain Adaptation [[arxiv](https://arxiv.org/abs/2401.14148)]
  - Language guided multi-source DA  在多源域自适应中使用语言指导

- AdaEmbed: Semi-supervised Domain Adaptation in the Embedding Space [[arxiv](https://arxiv.org/abs/2401.12421)]
  - Semi-spuervised domain adaptation in the embedding space 在嵌入空间中进行半监督域自适应

- Inter-Domain Mixup for Semi-Supervised Domain Adaptation [[arxiv](https://arxiv.org/abs/2401.11453)]
  - Inter-domain mixup for semi-supervised domain adaptation 跨领域mixup用于半监督域自适应

- Source-Free and Image-Only Unsupervised Domain Adaptation for Category Level Object Pose Estimation [[arxiv](https://arxiv.org/abs/2401.10848)]
  - Source-free and image-only unsupervised domain adaptation 

- Multi-Source Domain Adaptation with Transformer-based Feature Generation for Subject-Independent EEG-based Emotion Recognition [[arxiv](https://arxiv.org/abs/2401.02344)]
  - Multi-source DA with Transformer-based feature generation

- Multi-Modal Domain Adaptation Across Video Scenes for Temporal Video Grounding [[arxiv](https://arxiv.org/abs/2312.13633)]
  - Multi-modal domain adaptation 多模态领域自适应

- Domain Adaptive Graph Classification [[arxiv](https://arxiv.org/abs/2312.13536)]
  - Domain adaptive graph classification 域适应的图分类

- Understanding and Estimating Domain Complexity Across Domains [[arxiv](https://arxiv.org/abs/2312.13487)]
  - Understanding and estimating domain complexity 解释领域复杂性

- Prompt-based Domain Discrimination for Multi-source Time Series Domain Adaptation [[arxiv](https://arxiv.org/abs/2312.12276v1)]
  - Prompt-based domain discrimination for time series domain adaptation 基于prompt的时间序列域自适应

- NeurIPS'23 SwapPrompt: Test-Time Prompt Adaptation for Vision-Language Models [[arxiv](https://openreview.net/forum?id=EhdNQiOWgQ)]
  - Test-time prompt adaptation for vision language models 对视觉-语言大模型的测试时prompt自适应

- DARNet: Bridging Domain Gaps in Cross-Domain Few-Shot Segmentation with Dynamic Adaptation [[arxiv](http://arxiv.org/abs/2312.04813)]
  - Dynamic adaptation for cross-domain few-shot segmentation 动态适配用于跨领域小样本分割

- A Unified Framework for Unsupervised Domain Adaptation based on Instance Weighting [[arxiv](http://arxiv.org/abs/2312.05024)]
  - Instance weighting for domain adaptation 样本加权用于领域自适应

- Target-agnostic Source-free Domain Adaptation for Regression Tasks [[arxiv](http://arxiv.org/abs/2312.00540)]
  - Target-agnostic source-free DA for regression 用于回归任务的source-free DA

- Proposal-Level Unsupervised Domain Adaptation for Open World Unbiased Detector [[arxiv](https://arxiv.org/abs/2311.02342)]
  - Proposal-level unsupervised domain adaptation

- Better Practices for Domain Adaptation [[arxiv](http://arxiv.org/abs/2309.03879)]
  - Better practice for domain adaptation

- Domain Adaptation for Efficiently Fine-tuning Vision Transformer with Encrypted Images [[arxiv](http://arxiv.org/abs/2309.02556)]
  - Domain adaptation for efficient ViT

- Robust Activity Recognition for Adaptive Worker-Robot Interaction using Transfer Learning [[arxiv](http://arxiv.org/abs/2308.14843)]
  - Activity recognition using domain adaptation

- Unsupervised Domain Adaptation via Domain-Adaptive Diffusion [[arxiv](http://arxiv.org/abs/2308.13893)]
  - Domain-adaptive diffusion for domain adaptation 领域自适应的diffusion

- SAM-DA: UAV Tracks Anything at Night with SAM-Powered Domain Adaptation [[arxiv](https://arxiv.org/abs/2307.01024)]
  - Using SAM for domain adaptation 使用segment anything进行domain adaptation

- Cross-Database and Cross-Channel ECG Arrhythmia Heartbeat Classification Based on Unsupervised Domain Adaptation [[arxiv](http://arxiv.org/abs/2306.04433)]
  - EEG using unsupervised domain adaptation 用无监督DA来进行EEG心跳分类

- Real-Time Online Unsupervised Domain Adaptation for Real-World Person Re-identification [[arxiv](http://arxiv.org/abs/2306.03993)]
  - Real-time online unsupervised domain adaptation for REID 无监督DA用于REID

- Can We Evaluate Domain Adaptation Models Without Target-Domain Labels? A Metric for Unsupervised Evaluation of Domain Adaptation [[arxiv](http://arxiv.org/abs/2305.18712)]
  - Evaluate domain adaptation models 评测domain adaptation的模型

- Universal Test-time Adaptation through Weight Ensembling, Diversity Weighting, and Prior Correction [[arxiv](http://arxiv.org/abs/2306.00650)]
  - Universal test-time adaptation

- Universal Domain Adaptation from Foundation Models [[arxiv](http://arxiv.org/abs/2305.11092)]
  - Using foundation models for universal domain adaptation

- Multi-Source to Multi-Target Decentralized Federated Domain Adaptation [[arxiv](http://arxiv.org/abs/2304.12422)]
  - Decentralized federated domain adaptation 

- Multi-Source to Multi-Target Decentralized Federated Domain Adaptation [[arxiv](https://arxiv.org/abs/2304.12422)]
  - Multi-source to multi-target federated domain adaptation 多源多目标的联邦域自适应

- ICML'23 AdaNPC: Exploring Non-Parametric Classifier for Test-Time Adaptation [[arxiv](https://arxiv.org/abs/2304.12566)]
  - Adaptive test-time adaptation 非参数化分类器进行测试时adaptation

- CVPR'23 Zero-shot Generative Model Adaptation via Image-specific Prompt Learning [[arxiv](http://arxiv.org/abs/2304.03119)]
  - Zero-shot generative model adaptation via image-specific prompt learning 零样本的生成模型adaptation

- Source-free Domain Adaptation Requires Penalized Diversity [[arxiv](http://arxiv.org/abs/2304.02798)]
  - Source-free DA requires penalized diversity

- Complementary Domain Adaptation and Generalization for Unsupervised Continual Domain Shift Learning [[arxiv](http://arxiv.org/abs/2303.15833)]
  - Continual domain shift learning using adaptation and generalization 使用 adaptation和DG进行持续分布变化的学习

- CVPR'23 Feature Alignment and Uniformity for Test Time Adaptation [[arxiv](http://arxiv.org/abs/2303.10902)]
  - Feature alignment for test-time adaptation 使用特征对齐进行测试时adaptation

- TempT: Temporal consistency for Test-time adaptation [[arxiv](http://arxiv.org/abs/2303.10536)]
  - Temporeal consistency for test-time adaptation 时间一致性用于test-time adaptation

- CVPR'23 A New Benchmark: On the Utility of Synthetic Data with Blender for Bare Supervised Learning and Downstream Domain Adaptation [[arxiv](http://arxiv.org/abs/2303.09165)]
  - A new benchmark for domain adaptation 一个对于domain adaptation最新的benchmark

- Unsupervised domain adaptation by learning using privileged information [[arxiv](http://arxiv.org/abs/2303.09350)]
  - Domain adaptation by privileged information 使用高级信息进行domain adaptation

- Probabilistic Domain Adaptation for Biomedical Image Segmentation [[arxiv](http://arxiv.org/abs/2303.11790)]
  - Probabilistic domain adaptation for biomedical image segmentation 概率的domain adaptation用于生物医疗图像分割

- Unsupervised Cumulative Domain Adaptation for Foggy Scene Optical Flow [[arxiv](https://arxiv.org/abs/2303.07564)]
  - Domain adaptation for foggy scene optical flow 领域自适应用于雾场景的光流

- Domain Adaptation for Time Series Under Feature and Label Shifts [[arxiv](https://arxiv.org/abs/2302.03133)]
  - Domain adaptation for time series 用于时间序列的domain adaptation

- TPAMI'23 Source-Free Unsupervised Domain Adaptation: A Survey [[arxiv](http://arxiv.org/abs/2301.00265)]
  - A survey on source-free domain adaptation 关于source-free DA的一个最新综述

- Discriminative Radial Domain Adaptation [[arxiv](http://arxiv.org/abs/2301.00383)]
  - Discriminative radial domain adaptation 判别性的放射式domain adaptation

- WACV'23 Cross-Domain Video Anomaly Detection without Target Domain Adaptation [[arxiv](https://arxiv.org/abs/2212.07010)]
  - Cross-domain video anomaly detection without target domain adaptation 跨域视频异常检测

- Co-Learning with Pre-Trained Networks Improves Source-Free Domain Adaptation [[arxiv](https://arxiv.org/abs/2212.07585)]
  - Pre-trained models for source-free domain adaptation 用预训练模型进行source-free DA

- CONDA: Continual Unsupervised Domain Adaptation Learning in Visual Perception for Self-Driving Cars [[arxiv](https://arxiv.org/abs/2212.00621)]
  - Continual DA for self-driving cars 连续的domain adaptation用于自动驾驶

- Robust Mean Teacher for Continual and Gradual Test-Time Adaptation [[arxiv](https://arxiv.org/abs/2211.13081)]
  - Mean teacher for test-time adaptation 在测试时用mean teacher进行适配

- ECCV-22 DecoupleNet: Decoupled Network for Domain Adaptive Semantic Segmentation [[arXiv](https://arxiv.org/pdf/2207.09988.pdf)] [[Code](https://github.com/dvlab-research/DecoupleNet)]
  - Domain adaptation in semantic segmentation 语义分割域适应

- NeurIPS'22 Divide and Contrast: Source-free Domain Adaptation via Adaptive Contrastive Learning [[openreview](https://openreview.net/forum?id=NjImFaBEHl)]
  - Adaptive contrastive learning for source-free DA 自适应的对比学习用于source-free DA

- NeurIPS'22 MetaTeacher: Coordinating Multi-Model Domain Adaptation for Medical Image Classification [[openreview](https://openreview.net/forum?id=AQd4ugzALQ1)]
  - Multi-model domain adaptation mor medical image classification 多模型DA用于医疗数据

- NeurIPS'22 Domain Adaptation under Open Set Label Shift [[openreview](https://openreview.net/forum?id=OMZG4vsKmm7)]
  - Domain adaptation under open set label shift 在开放集的label shift中的DA

- NeurIPS'22 Test Time Adaptation via Conjugate Pseudo-labels [[openreview](https://openreview.net/forum?id=2yvUYc-YNUH)]
  - Test-time adaptation with conjugate pseudo-labels 用伪标签进行测试时adaptation

- On Fine-Tuned Deep Features for Unsupervised Domain Adaptation [[arxiv](http://arxiv.org/abs/2210.14083)]
  - Fine-tuned features for domain adaptation 微调的特征用于域自适应

- WACV-23 ConfMix: Unsupervised Domain Adaptation for Object Detection via Confidence-based Mixing [[arxiv](https://arxiv.org/abs/2210.11539)]
  - Domain adaptation for object detection using confidence mixing 用置信度mix做domain adaptation

- Unsupervised Domain Adaptation for COVID-19 Information Service with Contrastive Adversarial Domain Mixup [[arxiv](https://arxiv.org/abs/2210.03250)]
  - Domain adaptation for COVID-19 用DA进行COVID-19预测

- ICONIP'22 IDPL: Intra-subdomain adaptation adversarial learning segmentation method based on Dynamic Pseudo Labels [[arxiv](https://arxiv.org/abs/2210.03435)]
  - Intra-domain adaptation for segmentation 子领域对抗Adaptation

- NeurIPS'22 Polyhistor: Parameter-Efficient Multi-Task Adaptation for Dense Vision Tasks [[arxiv](https://arxiv.org/abs/2210.03265)]
  - Parameter-efficient multi-task adaptation 参数高效的多任务adaptation

- Deep Domain Adaptation for Detecting Bomb Craters in Aerial Images [[arxiv](https://arxiv.org/abs/2209.11299)]
  - Bomb craters detection using domain adaptation 用DA检测遥感图像中的炮弹弹坑

- WACV-23 TeST: Test-time Self-Training under Distribution Shift [[arxiv](https://arxiv.org/abs/2209.11459)]
  - Test-time self-training 测试时训练

- Robust Domain Adaptation for Machine Reading Comprehension [[arxiv](https://arxiv.org/abs/2209.11615)]
  - Domain adaptation for machine reading comprehension 机器阅读理解的domain adaptation

- IEEE-TMM'22 Uncertainty Modeling for Robust Domain Adaptation Under Noisy Environments [[IEEE](https://ieeexplore.ieee.org/abstract/document/9882310)]
  - Uncertainty modeling for domain adaptation 噪声环境下的domain adaptation

- MM-22 [Making the Best of Both Worlds: A Domain-Oriented Transformer for Unsupervised Domain Adaptation](https://arxiv.org/abs/2208.01195)
  - Transformer for domain adaptation 用transformer进行DA

- NeurIPS-21 [The balancing principle for parameter choice in distance-regularized domain adaptation](https://papers.nips.cc/paper/2021/hash/ae0909a324fb2530e205e52d40266418-Abstract.html)
  - Hyperparameter selection for domain adaptation 对adaptation中的正则项系数进行选择

- ECCV-22 [Prototype-Guided Continual Adaptation for Class-Incremental Unsupervised Domain Adaptation](https://arxiv.org/abs/2207.10856)
  - Prototype continual domain adaptation 基于原型的类增量domain adaptation

- [Transferability-Guided Cross-Domain Cross-Task Transfer Learning](https://arxiv.org/abs/2207.05510)
  - Cross-domain cross-task transfer learning 用迁移性指标指导跨领域跨任务迁移

- [A Data-Based Perspective on Transfer Learning](https://arxiv.org/abs/2207.05739)
  - Analyze the data numbers in transfer learning 分析迁移学习中数据的重要性

- [Few-Max: Few-Shot Domain Adaptation for Unsupervised Contrastive Representation Learning](https://arxiv.org/abs/2206.10137)
  - Few-shot DA for unsupervised constrastive learning 小样本DA用于无监督对比学习

- ICPR-22 [OTAdapt: Optimal Transport-based Approach For Unsupervised Domain Adaptation](https://arxiv.org/abs/2205.10738)
  - Optimal transport-based domain adaptation 利用最优传输进行领域自适应

- CVPR-22 [Safe Self-Refinement for Transformer-based Domain Adaptation](https://arxiv.org/abs/2204.07683)
  - Transformer-based domain adaptation 基于transformer的domain adaptation

- ISPASS-22 [Benchmarking Test-Time Unsupervised Deep Neural Network Adaptation on Edge Devices](https://arxiv.org/abs/2203.11295)
  - Benchmarking test-time adaptation for edge devices
  - 在端设备上评测test-time adaptation算法

- [Multi-Source Domain Adaptation Based on Federated Knowledge Alignment](https://arxiv.org/abs/2203.11635)
  - Multi-source domain adaptation
  - 多源域自适应

- [A Broad Study of Pre-training for Domain Generalization and Adaptation](https://arxiv.org/abs/2203.11819)
  - A broad study of pre-training models for DA and DG
  - 大量的实验进行DA和DG

- [Open Set Domain Adaptation By Novel Class Discovery](https://arxiv.org/abs/2203.03329)
  - Open set DA by novel class discovery
  - 基于新类发现的open set da

- ICML-21 workshop [Domain Adaptation with Factorizable Joint Shift](https://arxiv.org/abs/2203.02902)
  - Domain adaptation with factorizable joint shift
  - 基于可分解的联合漂移的领域自适应

- [Causal Domain Adaptation with Copula Entropy based Conditional Independence Test](https://arxiv.org/abs/2202.13482)
  - Use copula entropy based conditional independence test for csusal domain adaptation
  - 使用基于copula entopy的条件独立测试进行causal domain adaptation

- ICLR-22 [Graph-Relational Domain Adaptation](https://arxiv.org/abs/2202.03628)
  - Graph-relational domain adapttion using topological structures
  - 图级别的domain adaptation，使用拓扑结构

- [UMAD: Universal Model Adaptation under Domain and Category Shift](https://arxiv.org/abs/2112.08553)
    - Model adaptation under domain and category shift
    - 在domain和class都有shift的前提下进行模型适配

- [A Survey of Unsupervised Domain Adaptation for Visual Recognition](http://arxiv.org/abs/2112.06745)
    - A new survey article of domain adaptation
    - 对UDA的一个综述文章，来自作者博士论文

- [Unsupervised Domain Adaptation: A Reality Check](https://arxiv.org/abs/2111.15672)
    - Doing experiments to show the progress of DA methods over the years
    - 用大量的实验来验证近几年来DA方法的进展
  
- [Hierarchical Optimal Transport for Unsupervised Domain Adaptation](https://arxiv.org/abs/2112.02073)
    - hierarchical optimal transport for UDA
    - 层次性的最优传输用于domain adaptation

- [Boosting Unsupervised Domain Adaptation with Soft Pseudo-label and Curriculum Learning](https://arxiv.org/abs/2112.01948)
    - Using soft pseudo-label and curriculum learning to boost UDA
    - 用软的伪标签和课程学习增强UDA方法

- WACV-22 [Semi-supervised Domain Adaptation via Sample-to-Sample Self-Distillation](https://arxiv.org/abs/2111.14353)
    - Sample-level self-distillation for semi-supervised DA
    - 样本层次的自蒸馏用于半监督DA

- [C-MADA: Unsupervised Cross-Modality Adversarial Domain Adaptation framework for medical Image Segmentation](https://arxiv.org/abs/2110.15823)
    - Cross-modality domain adaptation for medical image segmentation
    - 跨模态的DA用于医学图像分割

- [Domain Adaptation for Rare Classes Augmented with Synthetic Samples](https://arxiv.org/abs/2110.12216)
    - Domain adaptation for rare class
    - 稀疏类的domain adaptation

- BMVC-21 [Dynamic Feature Alignment for Semi-supervised Domain Adaptation](https://arxiv.org/abs/2110.09641)
    - Dynamic feature alignment for semi-supervised DA
    - 动态特征对齐用于半监督DA

- IEEE TIP-21 [Joint Clustering and Discriminative Feature Alignment for Unsupervised Domain Adaptation](https://ieeexplore.ieee.org/abstract/document/9535218)
  - Clustering and discriminative alignment for DA
  - 聚类与判定式对齐用于DA

- IEEE TNNLS-21 [Entropy Minimization Versus Diversity Maximization for Domain Adaptation](https://ieeexplore.ieee.org/abstract/document/9537640)
  - Entropy minimization versus diversity max for DA
  - 熵最小化与diversity最大化

- [Cross-Region Domain Adaptation for Class-level Alignment](https://arxiv.org/abs/2109.06422)
  - Cross-region domain adaptation for class-level alignment
  - 跨区域的领域自适应用于类级别的对齐

- EMNLP-21 [Non-Parametric Unsupervised Domain Adaptation for Neural Machine Translation](https://arxiv.org/abs/2109.06604)
  - UDA for machine translation
  - 用领域自适应进行机器翻译

- [Unsupervised domain adaptation for cross-modality liver segmentation via joint adversarial learning and self-learning](https://arxiv.org/abs/2109.05664)
  - Domain adaptation for cross-modality liver segmentation
  - 使用domain adaptation进行肝脏的跨模态分割

- [CDTrans: Cross-domain Transformer for Unsupervised Domain Adaptation](https://arxiv.org/abs/2109.06165)
  - Cross-domain transformer for domain adaptation
  - 基于transformer进行domain adaptation

- [Robust Ensembling Network for Unsupervised Domain Adaptation](https://arxiv.org/abs/2108.09473)
  - Ensembling network for domain adaptation
  - 集成嵌入网络用于domain adaptation

- [TVT: Transferable Vision Transformer for Unsupervised Domain Adaptation](https://arxiv.org/abs/2108.05988)
  - Vision transformer for domain adaptation
  - 用视觉transformer进行DA

- [Learning Transferable Parameters for Unsupervised Domain Adaptation](https://arxiv.org/abs/2108.06129)
  - Learning partial transfer parameters for DA
  - 学习适用于迁移部分的参数做UDA任务

- ICCV-21 [BiMaL: Bijective Maximum Likelihood Approach to Domain Adaptation in Semantic Scene Segmentation](https://arxiv.org/abs/2108.03267)
  - Bijective MMD for domain adaptation
  - 双射MMD用于语义分割

- MM-21 [Few-shot Unsupervised Domain Adaptation with Image-to-class Sparse Similarity Encoding](https://arxiv.org/abs/2108.02953)
  - Few-shot DA with image-to-class sparse similarity encoding
  - 小样本的领域自适应

- [Dual-Tuning: Joint Prototype Transfer and Structure Regularization for Compatible Feature Learning](https://arxiv.org/abs/2108.02959)
  - Prototype transfer and structure regularization
  - 原型的迁移学习

- CVPR-21 [Conditional Bures Metric for Domain Adaptation](https://openaccess.thecvf.com/content/CVPR2021/html/Luo_Conditional_Bures_Metric_for_Domain_Adaptation_CVPR_2021_paper.html)
    - A new metric for domain adaptation
    - 提出一个新的metric用于domain adaptation

- CVPR-21 [Generalized Domain Adaptation](https://openaccess.thecvf.com/content/CVPR2021/html/Mitsuzumi_Generalized_Domain_Adaptation_CVPR_2021_paper.html)
  - A general definition for domain adaptation
  - 一个更抽象更一般的domain adaptation定义

- CVPR-21 [Reducing Domain Gap by Reducing Style Bias](https://openaccess.thecvf.com/content/CVPR2021/html/Nam_Reducing_Domain_Gap_by_Reducing_Style_Bias_CVPR_2021_paper.html)
  - Syle-invariant training for adaptation and generalization
  - 通过训练图像对style无法辨别来进行DA和DG

- 20210706 CVPR-21 [Instance Level Affinity-Based Transfer for Unsupervised Domain Adaptation](https://openaccess.thecvf.com/content/CVPR2021/html/Sharma_Instance_Level_Affinity-Based_Transfer_for_Unsupervised_Domain_Adaptation_CVPR_2021_paper.html)
    - Instance affinity learning for domain adaptation
    - 样本间相似度学习，用于DA

- 20210716 BMCV-extend [Exploring Dropout Discriminator for Domain Adaptation](https://arxiv.org/abs/2107.04231)
  - Using multiple discriminators for domain adaptation
  - 用分布估计代替点估计来做domain adaptation


- 20201208 TIP [Effective Label Propagation for Discriminative Semi-Supervised Domain Adaptation](http://arxiv.org/abs/2012.02621)
    - 用label propagation做半监督domain adaptation

- 20201203 [Unpaired Image-to-Image Translation via Latent Energy Transport](https://arxiv.org/abs/2012.00649)
    - 用能量模型做图像翻译



- 20200927 [Privacy-preserving Transfer Learning via Secure Maximum Mean Discrepancy](https://arxiv.org/abs/2009.11680)
    - 加密情况下的MMD用于迁移学习

- 20200914 [A First Step Towards Distribution Invariant Regression Metrics](https://arxiv.org/abs/2009.05176)
    - 分布无关的回归评价


- 20200813 ECCV-20 [Learning to Cluster under Domain Shift](https://arxiv.org/abs/2008.04646)
    - Learning to cluster under domain shift
    - 在domain shift的情况下进行聚类
- 20200706 [Learn Faster and Forget Slower via Fast and Stable Task Adaptation](https://arxiv.org/abs/2007.01388)


- 20200629 [ICML-20] [Graph Optimal Transport for Cross-Domain Alignment](https://arxiv.org/abs/2006.14744)
	- Graph OT for cross-domain alignment


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



- 20191029 [Reducing Domain Gap via Style-Agnostic Networks](https://arxiv.org/abs/1910.11645)
  	- Use style-agnostic networks to avoid domain gap
  	- 通过风格无关的网络来避免领域的gap







- 20191008 arXiv [DIVA: Domain Invariant Variational Autoencoders](https://arxiv.org/abs/1905.10427)
  	- Domain invariant variational autoencoders
  	- 领域不变的变分自编码器

- 20190821 arXiv [Transfer Learning-Based Label Proportions Method with Data of Uncertainty](https://arxiv.org/abs/1908.06603)
  	- Transfer learning with source and target having uncertainty
  	- 当source和target都有不确定label时进行迁移



- 20190703 arXiv [Inferred successor maps for better transfer learning](https://arxiv.org/abs/1906.07663)
  	- Inferred successor maps for better transfer learning



- 20190531 IJCAI-19 [Adversarial Imitation Learning from Incomplete Demonstrations](https://arxiv.org/abs/1905.12310)
  	- Adversarial imitation learning from imcomplete demonstrations
  	- 基于不完整实例的对抗模仿学习

- 20190517 arXiv [Budget-Aware Adapters for Multi-Domain Learning](https://arxiv.org/abs/1905.06242)
    - Budget-Aware Adapters for Multi-Domain Learning

- 20190301 SysML-19 [FixyNN: Efficient Hardware for Mobile Computer Vision via Transfer Learning](https://arxiv.org/abs/1902.11128)
    - An efficient hardware for mobile computer vision applications using transfer learning
    - 提出一个高效的用于移动计算机视觉应用的硬件

- 20190118 arXiv [Domain Adaptation for Structured Output via Discriminative Patch Representations](https://arxiv.org/abs/1901.05427)
    - Domain adaptation for structured output
    - Domain adaptation用于结构化输出





- 20181217 arXiv [When Semi-Supervised Learning Meets Transfer Learning: Training Strategies, Models and Datasets](https://arxiv.org/abs/1812.05313)
    - Combining semi-supervised learning and transfer learning
    - 将半监督方法应用于迁移学习

- 20181127 arXiv [Privacy-preserving Transfer Learning for Knowledge Sharing](https://arxiv.org/abs/1811.09491)
	- First work on privacy preserving in transfer learning
	- 探讨迁移学习中隐私保护的文章



- 20181121 arXiv [Not just a matter of semantics: the relationship between visual similarity and semantic similarity](https://arxiv.org/abs/1811.07120)
    -  Interpreting relationships between visual similarity and semantic similarity
    - 解释了视觉相似性和语义相似性的不同

- 20181008 arXiv [Unsupervised Learning via Meta-Learning](https://arxiv.org/abs/1810.02334)
	- Meta-learning for unsupervised learning
	- 用于无监督学习的元学习

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

- 20180604 arXiv 在Open set domain adaptation中，用共享和私有部分重建进行问题的解决：[Learning Factorized Representations for Open-set Domain Adaptation](https://arxiv.org/abs/1805.12277)

- 20210706 CVPR-21 [Multi-Target Domain Adaptation With Collaborative Consistency Learning](https://openaccess.thecvf.com/content/CVPR2021/html/Isobe_Multi-Target_Domain_Adaptation_With_Collaborative_Consistency_Learning_CVPR_2021_paper.html)
    - Using collaborative consistency training for multi-target DA
    - 用多个模型做集成一致性训练进行多目标DA

- 20210625 CVPR-21 [Generalized Domain Adaptation](http://arxiv.org/abs/2106.01656)
  - Generalized domain adaptation
  - 更通用更一般的domain adaptation

- 20210625 CVPR-21 [A Fourier-based Framework for Domain Generalization](http://arxiv.org/abs/2105.11120)
  - Fourier based domain generalization
  - 基于傅里叶特征的DG

- 20210329 ICLR-21 [Tent: Fully Test-Time Adaptation by Entropy Minimization](https://openreview.net/forum?id=uXl3bZLkr3c)
    - Test time adaptation by entropy minimization
    - 测试时通过熵最小化进行adaptation

- 20210329 [Adversarial Branch Architecture Search for Unsupervised Domain Adaptation](https://arxiv.org/abs/2102.06679v2)
    - NAS for domain adaptation
    - 用神经网络结构搜索做领域自适应

- 20210312 [Discrepancy-Based Active Learning for Domain Adaptation](https://arxiv.org/abs/2103.03757v1)
    - Discrepancy and active learning for DA
    - 基于主动学习的DA

- 20210312 [Unbalanced minibatch Optimal Transport; applications to Domain Adaptation](https://arxiv.org/abs/2103.03606v1)
    - Unbalanced minibatch OT for DA
    - 非均衡的OT用于DA问题

- 20210127 [Hierarchical Domain Invariant Variational Auto-Encoding with weak domain supervision](http://arxiv.org/abs/2101.09436)
    - 利用VAE和解耦去做domain generalization
    - Using VAE and disentanglement for domain generalization

- 20201214 WWW-20 [Domain Adaptation with Category Attention Network for Deep Sentiment Analysis](https://dl.acm.org/doi/abs/10.1145/3366423.3380088?casa_token=W6UxRKT4pDQAAAAA%3ACTbuFKp72M88OdbcURQSSua5XaM0GI2Y90795GGFv6ZiEx584ZGj8HT3x8nBSAUhvr2-DhQbnmVY1YM)
	- Unify pivots and non-pivots, and provide interpretability for domain adaptation in sentiment analysis
	- 统一pivots和non-pivots，并提供可解释性进行DA情感分析

- 20201208 NIPS-20 [Heuristic Domain Adaptation](https://proceedings.neurips.cc/paper/2020/file/555d6702c950ecb729a966504af0a635-Paper.pdf)
    - 启发式domain adaptation

- 20200804 ECCV-20 spotlight [Side-Tuning: A Baseline for Network Adaptation via Additive Side Networks](https://arxiv.org/abs/1912.13503)
    - 将现有的finetune机制进行扩展
    - Extending finetune mechanism
  - 20200804 ACMMM-20 [Adversarial Bipartite Graph Learning for Video Domain Adaptation](https://arxiv.org/abs/2007.15829)
    - Video domain adaptation
    - 视频的领域自适应
  - 20200804 MICCAI-20 [Whole MILC: generalizing learned dynamics across tasks, datasets, and populations](https://arxiv.org/abs/2007.16041)
    - Generalizing across tasks, datasets, populations
    - 在任务、数据集、人群之间做泛化

- 20200724 [Learning to Match Distributions for Domain Adaptation](https://arxiv.org/abs/2007.10791)
  	- 自动深度迁移学习
  	- Automatic domain adaptation

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

- 20210420 arXiv [On Universal Black-Box Domain Adaptation](https://arxiv.org/abs/2104.04665)
    - Universal black-box domain adaptation
    - 黑盒情况下的universal domain adaptation

- 20210319 [Learning Invariant Representations across Domains and Tasks](https://arxiv.org/abs/2103.05114)
    - Automatically learn to match distributions
    - 自动适配分布的任务适配网络

- 20191222 arXiv [Dreaming to Distill: Data-free Knowledge Transfer via DeepInversion](https://arxiv.org/abs/1912.08795)
   - Generate data without priors for transfer learning based on deep dream
   - 只用网络架构不用原来数据，生成新数据用于迁移

- 20191201 arXiv [A Unified Framework for Lifelong Learning in Deep Neural Networks](https://arxiv.org/abs/1911.09704)
     - A unified framework for life-long learing in DNN

- 20191201 arXiv [ReMixMatch: Semi-Supervised Learning with Distribution Alignment and Augmentation Anchoring](https://arxiv.org/abs/1911.09785)
     - Semi-Supervised Learning with Distribution Alignment and Augmentation Anchoring

- 20191119 NIPS-19 workshop [Collaborative Unsupervised Domain Adaptation for Medical Image Diagnosis](https://arxiv.org/abs/1911.07293)
  	- Ensemble DA using noise labels
  	- 在ensemble中出现noise label时如何处理

- 20191029 KBS [Semi-supervised representation learning via dual autoencoders for domain adaptation](https://arxiv.org/abs/1908.01342)
  	- Semi-supervised domain adaptation with autoencoders
  	- 用自动编码器进行半监督的DA

- 20190926 arXiv [Learning a Domain-Invariant Embedding for Unsupervised Domain Adaptation Using Class-Conditioned Distribution Alignment](https://arxiv.org/abs/1907.02271)
  	- Use class-conditional DA for domain adaptation
  	- 使用类条件对齐进行domain adaptation

- 20190926 arXiv [A Deep Learning-Based Approach for Measuring the Domain Similarity of Persian Texts](https://arxiv.org/abs/1909.09690)
  	- Deep learning based domain similarity learning
  	- 利用深度学习进行领域相似度的学习

- 20190926 arXiv [FEED: Feature-level Ensemble for Knowledge Distillation](https://arxiv.org/abs/1909.10754)
  	- Feature-level knowledge distillation
  	- 特征层面的知识蒸馏

- 20190926 ICCV-19 [Self-similarity Grouping: A Simple Unsupervised Cross Domain Adaptation Approach for Person Re-identification](https://arxiv.org/abs/1811.10144)
  	- A simple approach for domain adaptation
  	- 一个很简单的DA方法

- 20190910 BMVC-19 [Curriculum based Dropout Discriminator for Domain Adaptation](https://arxiv.org/abs/1907.10628)
  	- Curriculum dropout for domain adaptation
  	- 基于课程学习的dropout用于DA

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

- 20181113 NIPS-18 [Generalized Zero-Shot Learning with Deep Calibration Network](http://ise.thss.tsinghua.edu.cn/~mlong/doc/deep-calibration-network-nips18.pdf)
	-  Deep calibration network for zero-shot learning
	- 提出deep calibration network进行zero-shot learning

- 20181110 AAAI-19 [Knowledge Transfer via Distillation of Activation Boundaries Formed by Hidden Neurons](https://arxiv.org/abs/1811.03233)
	-  Transfer learning for bounding neuron activation boundaries
	- 使用迁移学习进行神经元激活边界判定

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



- [深度联合适配网络](http://proceedings.mlr.press/v70/long17a.html)（Joint Adaptation Network, JAN）
	- Deep Transfer Learning with Joint Adaptation Networks
	- 延续了之前的DAN工作，这次考虑联合适配

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

- 20190408 arXiv [DeceptionNet: Network-Driven Domain Randomization](https://arxiv.org/abs/1904.02750)
    - Using only source data for domain randomization
    - 仅利用源域数据进行domain randomization

- 20190220 arXiv [Unsupervised Domain Adaptation using Deep Networks with Cross-Grafted Stacks](https://arxiv.org/abs/1902.06328)
    - Domain adaptation using deep learning with cross-grafted stacks
    - 用跨领域嫁接栈进行domain adaptation

- 20181217 arXiv [DLOW: Domain Flow for Adaptation and Generalization](https://arxiv.org/abs/1812.05418)
    - Domain flow for adaptation and generalization
    - 域流方法应用于领域自适应和扩展

- 20181212 arXiv [Learning Transferable Adversarial Examples via Ghost Networks](https://arxiv.org/abs/1812.03413)
    - Use ghost networks to learn transferrable adversarial examples
    - 使用ghost网络来学习可迁移的对抗样本

- 20181205 arXiv [Unsupervised Domain Adaptation using Generative Models and Self-ensembling](https://arxiv.org/abs/1812.00479)
	-  UDA using CycleGAN
	- 基于CycleGAN的domain adaptation

- 20181205 arXiv [VADRA: Visual Adversarial Domain Randomization and Augmentation](https://arxiv.org/abs/1812.00491)
	- Domain randomization and augmentation
	- Domain randomization和增强

- 20181128 arXiv [Geometry-Consistent Generative Adversarial Networks for One-Sided Unsupervised Domain Mapping](https://arxiv.org/abs/1809.05852)
  - CycleGAN for domain adaptation

- 20181127 arXiv [Distorting Neural Representations to Generate Highly Transferable Adversarial Examples](https://arxiv.org/abs/1811.09020)
	-  Generate transferrable examples to fool networks
	- 生成一些可迁移的对抗样本来迷惑神经网络，在各个网络上都表现好

- 20181123 arXiv [Progressive Feature Alignment for Unsupervised Domain Adaptation](https://arxiv.org/abs/1811.08585)
    - Progressively selecting confident pseudo labeled samples for transfer
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
	- Embed an autoencoder in GAN to improve its stability in training and propose two distances
	- 将autoencoder集成到GAN中，提出相应的两种距离进行度量，提高了GAN的稳定性

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

- 20180403 CVPR-18 将样本权重应用于对抗partial transfer中：[Importance Weighted Adversarial Nets for Partial Domain Adaptation](https://arxiv.org/abs/1803.09210

- 20180326 MLSP-17 把domain separation network和对抗结合起来，提出了一个对抗网络进行迁移学习：[Adversarial domain separation and adaptation](http://ieeexplore.ieee.org/abstract/document/8168121/)

- 20180326 ICIP-17 类似于domain separation network，加入了对抗判别训练，同时优化分类、判别、相似度三个loss：[Semi-supervised domain adaptation via convolutional neural network](http://ieeexplore.ieee.org/abstract/document/8296801/)

- 20180116 ICLR-18 用对偶的形式替代对抗训练中原始问题的表达，从而进行分布对齐 [Stable Distribution Alignment using the Dual of the Adversarial Distance](https://arxiv.org/abs/1707.04046)

- 20180111 arXiv 在GAN中用原始问题的对偶问题替换max问题，使得梯度更好收敛 [Stable Distribution Alignment Using the Dual of the Adversarial Distance](https://arxiv.org/abs/1707.04046)

- 20180110 AAAI-18 将Wasserstein GAN用到domain adaptaiton中 [Wasserstein Distance Guided Representation Learning for Domain Adaptation](https://arxiv.org/abs/1707.01217)

- 201707 CVPR-17 [Adversarial Representation Learning For Domain Adaptation](https://arxiv.org/abs/1707.01217)

- AAAI-18 [Multi-Adversarial Domain Adaptation](http://ise.thss.tsinghua.edu.cn/~mlong/doc/multi-adversarial-domain-adaptation-aaai18.pdf)

- ICCV-17 [CycleGAN: Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/pdf/1703.10593.pdf)

- ICCV-17 [DualGAN: DualGAN: Unsupervised Dual Learning for Image-to-Image Translation](https://arxiv.org/pdf/1704.02510.pdf)

- CVPR-17 [Asymmetric Tri-training for Unsupervised Domain Adaptation](https://arxiv.org/abs/1702.08400.pdf)

- ICML-17 [DiscoGAN: Learning to Discover Cross-Domain Relations with Generative Adversarial Networks](https://arxiv.org/abs/1703.05192)

- - -

## Domain generalization

### Survey

- TKDE-22 [Generalizing to Unseen Domains: A Survey on Domain Generalization](https://arxiv.org/abs/2103.03097) | [知乎文章](https://zhuanlan.zhihu.com/p/354740610) | [微信公众号](https://mp.weixin.qq.com/s/DsoVDYqLB1N7gj9X5UnYqw) | [Code](https://github.com/jindongwang/transferlearning/tree/master/code/DeepDG)
    - First survey on domain generalization
    - 第一篇对Domain generalization (领域泛化)的综述

- Federated Domain Generalization: A Survey [[arxiv](http://arxiv.org/abs/2306.01334)]
  - A survey on federated domain generalization 一篇关于联邦域泛化的综述

### Tutorial

- KDD 2023 tutorial: trustworthy machine learning: robustness, generalization, and interpretability [[link](https://mltrust.github.io/)]

- WSDM-23 and IJCAI-22 A tutorial on domain generalization [[link](https://dl.acm.org/doi/10.1145/3539597.3572722)] | [[website](https://dgresearch.github.io/)]
  - A tutorial on domain generalization

### Papers

- Out-of-Distribution Detection & Applications With Ablated Learned Temperature Energy [[arxiv](https://arxiv.org/abs/2401.12129)]
  - OOD detection for ablated learned temperature energy

- ICLR'24 Supervised Knowledge Makes Large Language Models Better In-context Learners [[arxiv](https://arxiv.org/abs/2312.15918)]
  - Small models help large language models for better OOD 用小模型帮助大模型进行更好的OOD

- NeurIPS'23 Test-Time Distribution Normalization for Contrastively Learned Visual-language Models [[paper](https://openreview.net/forum?id=VKbEO2eh5w)]
  - Test-time distribution normalization for contrastively learned VLM

- NeurIPS'23 A Closer Look at the Robustness of Contrastive Language-Image Pre-Training (CLIP) [[paper](https://openreview.net/forum?id=wMNpMe0vp3)]
  - A fine-gained analysis of CLIP robustness

- NeurIPS'23 CODA: Generalizing to Open and Unseen Domains with Compaction and Disambiguation [[arxiv](https://openreview.net/forum?id=Jw0KRTjsGA)]
  - Open set domain generalization using extra classes

- CPAL'24 FIXED: Frustratingly Easy Domain Generalization with Mixup [[arxiv](https://arxiv.org/abs/2211.05228)]
  - Easy domain generalization with mixup

- SDM'24 Towards Optimization and Model Selection for Domain Generalization: A Mixup-guided Solution [[arxiv](https://arxiv.org/abs/2209.00652)]
  - Optimization and model selection for domain generalization

- Leveraging SAM for Single-Source Domain Generalization in Medical Image Segmentation [[arxiv](https://arxiv.org/abs/2401.02076)]
  - SAM for single-source domain generalization

- Open Domain Generalization with a Single Network by Regularization Exploiting Pre-trained Features [[arxiv](http://arxiv.org/abs/2312.05141)]
  - Open domain generalization with a single network 用单一网络结构进行开放式domain generalizaition

- Stronger, Fewer, & Superior: Harnessing Vision Foundation Models for Domain Generalized Semantic Segmentation [[arxiv](http://arxiv.org/abs/2312.04265)]
  - Using vision foundation models for domain genealized semantic segmentation 用视觉基础模型进行域泛化语义分割

- On the Out-Of-Distribution Robustness of Self-Supervised Representation Learning for Phonocardiogram Signals [[arxiv](http://arxiv.org/abs/2312.00502)]
  - OOD robustness for self-supervised learning for phonocardiogram 心音图信号自监督的OOD鲁棒性

- A2XP: Towards Private Domain Generalization [[arxiv](https://arxiv.org/abs/2311.10339)]
  - Private domain generalization 隐私保护的domain generalization

- Layer-wise Auto-Weighting for Non-Stationary Test-Time Adaptation [[arxiv](http://arxiv.org/abs/2311.05858)]
  - Auto-weighting for test-time adaptation 自动权重的TTA

- Domain Generalization by Learning from Privileged Medical Imaging Information [[arxiv](http://arxiv.org/abs/2311.05861)]
  - Domain generalizaiton by learning from privileged medical imageing inforamtion

- SSL-DG: Rethinking and Fusing Semi-supervised Learning and Domain Generalization in Medical Image Segmentation [[arxiv](https://arxiv.org/abs/2311.02583)]
  - Semi-supervised learning + domain generalization 把半监督和领域泛化结合在一起

- WACV'24 Learning Class and Domain Augmentations for Single-Source Open-Domain Generalization [[arxiv](https://arxiv.org/abs/2311.02599)]
  - Class and domain augmentation for single-source open-domain DG 结合类和domain增强做单源DG

- Robust Fine-Tuning of Vision-Language Models for Domain Generalization [[arxiv](https://arxiv.org/abs/2311.02236)]
  - Robust fine-tuning for domain generalization 用于领域泛化的鲁棒微调

- NeurIPS 2023 Distilling Out-of-Distribution Robustness from Vision-Language Foundation Models [[arxiv](https://arxiv.org/abs/2311.01441)]
  - Distill OOD robustness from vision-language foundational models 从VLM模型中蒸馏出OOD鲁棒性

- UbiComp 2024 Optimization-Free Test-Time Adaptation for Cross-Person Activity Recognition [[arxiv](https://arxiv.org/abs/2310.18562)]
  - Test-time adaptation for activity recognition 测试时adaptation用于行为识别

- Prompting-based Efficient Temporal Domain Generalization [[arxiv](http://arxiv.org/abs/2310.02473)]
  - Prompt based temporal domain generalization 基于prompt的时间域domain generalization

- Domain Generalization with Fourier Transform and Soft Thresholding [[arxiv](http://arxiv.org/abs/2309.09866)]
  - Domain generalization with Fourier transform 基于傅里叶变换和软阈值进行domain generalization

- Multi-Scale and Multi-Layer Contrastive Learning for Domain Generalization [[arxiv](http://arxiv.org/abs/2308.14418)]
  - Multi-scale and multi-layer contrastive learning for DG 多尺度和多层对比学习用于DG

- Exploring the Transfer Learning Capabilities of CLIP in Domain Generalization for Diabetic Retinopathy [[arxiv](http://arxiv.org/abs/2308.14212)]
  - Domain generalization for diabetic retinopathy 用领域泛化进行糖尿病视网膜病

- NormAUG: Normalization-guided Augmentation for Domain Generalization [[arxiv](http://arxiv.org/abs/2307.13492)]
  - Normalization augmentation for domain generalization

- Benchmarking Algorithms for Federated Domain Generalization [[arxiv](http://arxiv.org/abs/2307.04942)]
  - Benchmark algorthms for federated domain generalization 对联邦域泛化算法进行的benchmark

- DISPEL: Domain Generalization via Domain-Specific Liberating [[arxiv](http://arxiv.org/abs/2307.07181)]
  - Domain generalization via domain-specific liberating

- Intra- & Extra-Source Exemplar-Based Style Synthesis for Improved Domain Generalization [[arxiv](https://arxiv.org/abs/2307.00648)]
  - Exemplar-based style synthesis for domain generalization 样例格式合成用于DG

- Pruning for Better Domain Generalizability [[arxiv](http://arxiv.org/abs/2306.13237)]
  - Using pruning for better domain generalization 使用剪枝操作进行domain generalization

- TMLR'23 Generalizability of Adversarial Robustness Under Distribution Shifts [[openreview](https://openreview.net/forum?id=XNFo3dQiCJ)]
  - Evaluate the OOD perormance of adversarial training 评测对抗训练模型的OOD鲁棒性

- Domain Generalization for Domain-Linked Classes [[arxiv](http://arxiv.org/abs/2306.00879)]
  - Domain generalization for domain-linked classes

- Selective Mixup Helps with Distribution Shifts, But Not (Only) because of Mixup [[arxiv](https://arxiv.org/abs/2305.16817)]
  - Why mixup works for domain generalization? 系统性研究为啥mixup对OOD很work

- Improved Test-Time Adaptation for Domain Generalization [[arxiv](http://arxiv.org/abs/2304.04494)]
  - Improved test-time adaptation for domain generalization

- Reweighted Mixup for Subpopulation Shift [[arxiv](http://arxiv.org/abs/2304.04148)]
  - Reweighted mixup for subpopulation shift

- Domain Generalization with Adversarial Intensity Attack for Medical Image Segmentation [[arxiv](http://arxiv.org/abs/2304.02720)]
  - Domain generalization for medical segmentation 用domain generalization进行医学分割

- CVPR'23 Meta-causal Learning for Single Domain Generalization [[arxiv](http://arxiv.org/abs/2304.03709)]
  - Meta-causal learning for domain generalization

- Domain Generalization In Robust Invariant Representation [[arxiv](http://arxiv.org/abs/2304.03431)]
  - Domain generalization in robust invariant representation

- Beyond Empirical Risk Minimization: Local Structure Preserving Regularization for Improving Adversarial Robustness [[arxiv](http://arxiv.org/abs/2303.16861)]
  - Local structure preserving for adversarial robustness 通过保留局部结构来进行对抗鲁棒性

- TFS-ViT: Token-Level Feature Stylization for Domain Generalization [[arxiv](http://arxiv.org/abs/2303.15698)]
  - Token-level feature stylization for domain generalization 用token-level特征变换进行domain generalization

- Are Data-driven Explanations Robust against Out-of-distribution Data? [[arxiv](http://arxiv.org/abs/2303.16390)]
  - Data-driven explanations robust? 探索数据驱动的解释是否是OOD鲁棒的

- ERM++: An Improved Baseline for Domain Generalization [[arxiv](http://arxiv.org/abs/2304.01973)]
  - Improved ERM for domain generalization 提高的ERM用于domain generalization

- Complementary Domain Adaptation and Generalization for Unsupervised Continual Domain Shift Learning [[arxiv](http://arxiv.org/abs/2303.15833)]
  - Continual domain shift learning using adaptation and generalization 使用 adaptation和DG进行持续分布变化的学习

- CVPR'23 TWINS: A Fine-Tuning Framework for Improved Transferability of Adversarial Robustness and Generalization [[arxiv](http://arxiv.org/abs/2303.11135)]
  - Improve generalization and adversarial robustness 同时提高鲁棒性和泛化性

- Finding Competence Regions in Domain Generalization [[arxiv](http://arxiv.org/abs/2303.09989)]
  - Finding competence regions in domain generalization 在DG中发现能力区域

- CVPR'23 ALOFT: A Lightweight MLP-like Architecture with Dynamic Low-frequency Transform for Domain Generalization [[arxiv](http://arxiv.org/abs/2303.11674)]
  - A lightweight module for domain generalization 一个用于DG的轻量级模块

- CVPR'23 Sharpness-Aware Gradient Matching for Domain Generalization [[arxiv](http://arxiv.org/abs/2303.10353)]
  - Sharpness-aware gradient matching for DG 利用梯度匹配进行domain generalization

- Domain Generalization via Nuclear Norm Regularization [[arxiv](https://arxiv.org/abs/2303.07527)]
  - Domain generalization via nuclear norm regularization 使用核归一化进行domain generalization

- Imbalanced Domain Generalization for Robust Single Cell Classification in Hematological Cytomorphology [[arxiv](https://arxiv.org/abs/2303.07771)]
  - Imbalanced domain generalization for single cell classification 不平衡的DG用于单细胞分类

- FedCLIP: Fast Generalization and Personalization for CLIP in Federated Learning [[arxiv](https://arxiv.org/abs/2302.13485v1)]
  - Fast generalization for federated CLIP 在联邦中进行快速的CLIP训练

- Robust Representation Learning with Self-Distillation for Domain Generalization [[arxiv](http://arxiv.org/abs/2302.06874)]
  - Robust representation learning with self-distillation

- ICLR-23 Temporal Coherent Test-Time Optimization for Robust Video Classification [[arxiv](http://arxiv.org/abs/2302.14309)]
  - Temporal distribution shift in video classification

- On the Robustness of ChatGPT: An Adversarial and Out-of-distribution Perspective [[arxiv](https://arxiv.org/abs/2302.12095)] | [[code](https://github.com/microsoft/robustlearn)]
  - Adversarial and OOD evaluation of ChatGPT 对ChatGPT鲁棒性的评测

- How Reliable is Your Regression Model's Uncertainty Under Real-World Distribution Shifts? [[arxiv](https://arxiv.org/abs/2302.03679)]
  - Regression models uncertainty for distribution shift 回归模型对于分布漂移的不确定性

- ICLR'23 SoftMatch: Addressing the Quantity-Quality Tradeoff in Semi-supervised Learning [[arxiv](https://arxiv.org/abs/2301.10921)]
  - Semi-supervised learning algorithm 解决标签质量问题的半监督学习方法

- Empirical Study on Optimizer Selection for Out-of-Distribution Generalization [[arxiv](http://arxiv.org/abs/2211.08583)]
  - Opimizer selection for OOD generalization OOD泛化中的学习器选择

- ICML'22 Understanding the failure modes of out-of-distribution generalization [[arxiv](https://openreview.net/forum?id=fSTD6NFIW_b)]
  - Understand the failure modes of OOD generalization 探索OOD泛化中的失败现象

- ICLR'23 Out-of-distribution Representation Learning for Time Series Classification [[arxiv](https://arxiv.org/abs/2209.07027)]
  - OOD for time series classification 时间序列分类的OOD算法
- CLIP the Gap: A Single Domain Generalization Approach for Object Detection [[arxiv](https://arxiv.org/abs/2301.05499)]
  - Using CLIP for domain generalization object detection 使用CLIP进行域泛化的目标检测

- TMLR'22 A Unified Survey on Anomaly, Novelty, Open-Set, and Out of-Distribution Detection: Solutions and Future Challenges [[openreview](https://openreview.net/forum?id=aRtjVZvbpK)]
  - A recent survey on OOD/anomaly detection 一篇最新的关于OOD/anomaly detection的综述

- NeurIPS'18 A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks [[paper](https://proceedings.neurips.cc/paper/2018/hash/abdeb6f575ac5c6676b747bca8d09cc2-Abstract.html)]
  - Using class-conditional distribution for OOD detection 使用类条件概率进行OOD检测

- ICLR'22 Discrete Representations Strengthen Vision Transformer Robustness [[arxiv](http://arxiv.org/abs/2111.10493)]
  - Embed discrete representation for OOD generalization 在ViT中加入离散表征增强OOD性能

- Learning to Learn Domain-invariant Parameters for Domain Generalization [[arxiv](Learning to Learn Domain-invariant Parameters for Domain Generalization)]
  - Learning to learn domain-invariant parameters for DG 元学习进行domain generalization

- HMOE: Hypernetwork-based Mixture of Experts for Domain Generalization [[arxiv](https://arxiv.org/abs/2211.08253)]
  - Hypernetwork-based ensembling for domain generalization 超网络构成的集成学习用于domain generalization

- The Evolution of Out-of-Distribution Robustness Throughout Fine-Tuning [[arxiv](https://arxiv.org/abs/2106.15831)]
  - OOD using fine-tuning 系统总结了基于fine-tuning进行OOD的一些结果

- GLUE-X: Evaluating Natural Language Understanding Models from an Out-of-distribution Generalization Perspective [[arxiv](https://arxiv.org/abs/2211.08073)]
  - OOD for natural language processing evaluation 提出GLUE-X用于OOD在NLP数据上的评估

- CVPR'22 Delving Deep Into the Generalization of Vision Transformers Under Distribution Shifts [[arxiv](https://openaccess.thecvf.com/content/CVPR2022/html/Zhang_Delving_Deep_Into_the_Generalization_of_Vision_Transformers_Under_Distribution_CVPR_2022_paper.html)]
  - Vision transformers generalization under distribution shifts 评估ViT的分布漂移

- NeurIPS'22 Models Out of Line: A Fourier Lens on Distribution Shift Robustness [[arxiv](https://openreview.net/forum?id=YZ-N-sejjwO)]
  - A fourier lens on distribution shift robustness 通过傅里叶视角来看分布漂移的鲁棒性

- Normalization Perturbation: A Simple Domain Generalization Method for Real-World Domain Shifts [[arxiv](https://arxiv.org/abs/2211.04393)]
  - Normalization perturbation for domain generalization 通过归一化扰动来进行domain generalization

- FIXED: Frustraitingly easy domain generalization using Mixup [[arxiv](https://arxiv.org/pdf/2211.05228.pdf)]
  - 使用Mixup进行domain generalization

- Learning to Learn Domain-invariant Parameters for Domain Generalization [[arxiv](https://arxiv.org/abs/2211.04582)]
  - Learning to learn domain-invariant parameters for domain generalization

- NeurIPS'22 LOG: Active Model Adaptation for Label-Efficient OOD Generalization [[openreview](https://openreview.net/forum?id=VdQWVdT_8v)]
  - Model adaptation for label-efficient OOD generalization

- NeurIPS'22 Domain Generalization without Excess Empirical Risk [[openreview](https://openreview.net/forum?id=pluyPFTiTeJ)]
  - Domain generalization without excess empirical risk 

- NeurIPS'22 FedSR: A Simple and Effective Domain Generalization Method for Federated Learning [[openreview](https://openreview.net/forum?id=mrt90D00aQX)]
  - FedSR for federated learning domain generalization 用于联邦学习的domain generalization

- NeurIPS'22 Probable Domain Generalization via Quantile Risk Minimization [[openreview](https://openreview.net/forum?id=6FkSHynJr1)]
  - Domain generalization with quantile risk minimization 用quantile风险最小化的domain generalization

- NeurIPS'22 Your Out-of-Distribution Detection Method is Not Robust! [[openreview](https://openreview.net/forum?id=YUEP3ZmkL1)]
  - OOD models are not robust 分布外泛化模型不够鲁棒

- PhDthesis Generalizing in the Real World with Representation Learning [[arxiv](http://arxiv.org/abs/2210.09925)]
  - A phd thesis about generalization in real world 一篇关于现实世界如何做Generalization的博士论文

- The Evolution of Out-of-Distribution Robustness Throughout Fine-Tuning [[arxiv](https://openreview.net/forum?id=Qs3EfpieOh)]
  - Evolution of OOD robustness by fine-tuning 

- Out-of-Distribution Generalization in Algorithmic Reasoning Through Curriculum Learning [[arxiv](https://arxiv.org/abs/2210.03275)]
  - OOD in algorithmic reasoning 算法reasoning过程中的OOD

- Towards Out-of-Distribution Adversarial Robustness [[arxiv](https://arxiv.org/abs/2210.03150)]
  - OOD adversarial robustness OOD对抗鲁棒性

- TripleE: Easy Domain Generalization via Episodic Replay [[arxiv](https://arxiv.org/pdf/2210.01807.pdf)]
  - Easy domain generalization by episodic replay

- Deep Spatial Domain Generalization [[arxiv](https://web7.arxiv.org/pdf/2210.00729.pdf)]
  - Deep spatial domain generalization

- Assaying Out-Of-Distribution Generalization in Transfer Learning [[arXiv](http://arxiv.org/abs/2207.09239)]
  - A lot of experiments to show OOD performance 

- ICML-21 Accuracy on the Line: on the Strong Correlation Between Out-of-Distribution and In-Distribution Generalization [[arxiv](https://proceedings.mlr.press/v139/miller21b.html)]
  - Strong correlation between ID and OOD

- Generalized representations learning for time series classification[[arxiv](https://arxiv.org/abs/2209.07027)]
  - OOD for time series classification 域泛化用于时间序列分类

- Language-aware Domain Generalization Network for Cross-Scene Hyperspectral Image Classification [[arxiv](https://arxiv.org/pdf/2209.02700.pdf)]
  - Domain generalization for cross-scene hyperspectral image classification 域泛化用于高光谱图像分类

- Improving Robustness to Out-of-Distribution Data by Frequency-based Augmentation [arxiv](https://arxiv.org/abs/2209.02369)
  - OOD by frequency-based augmentation 通过基于频率的数据增强进行OOD

- Domain Generalization for Prostate Segmentation in Transrectal Ultrasound Images: A Multi-center Study [arxiv](https://arxiv.org/abs/2209.02126)
  - Domain generalizationfor prostate segmentation 领域泛化用于前列腺分割

- Domain Adaptation from Scratch [arxiv](https://arxiv.org/abs/2209.00830)
  - Domain adaptation from scratch

- Towards Optimization and Model Selection for Domain Generalization: A Mixup-guided Solution [arxiv](https://arxiv.org/abs/2209.00652)
  - Model selection for domain generalization 域泛化中的模型选择问题

- [Equivariant Disentangled Transformation for Domain Generalization under Combination Shift](https://arxiv.org/abs/2208.02011)
  - Equivariant disentangled transformation for domain generalization 新的建模domain generalization的思路

- ECCV-22 workshop [Domain-Specific Risk Minimization](https://arxiv.org/abs/2208.08661)
  - Domain-specific risk minization for OOD 领域特异性风险最小化用于域泛化

- IJCAI-22 [Domain Generalization through the Lens of Angular Invariance](https://www.ijcai.org/proceedings/2022/0139.pdf)
  - Using angular invariance for domain generalization 使用角度不变性进行domain generalization

- [Adaptive Domain Generalization via Online Disagreement Minimization](https://arxiv.org/abs/2208.01996)
  - Online domain generalization via disagreement minimization 在线DG

- [Self-Distilled Vision Transformer for Domain Generalization](http://arxiv.org/abs/2207.12392)
  - Vision transformer for domain generalization 用ViT做domain generalization

- TMLR-22 [Domain-invariant Feature Exploration for Domain Generalization](https://arxiv.org/abs/2207.12020)
  - Exploring domain-invariant feature for domain generalization 探索领域不变特征在领域泛化中的应用

- TIST-22 [Domain Generalization for Activity Recognition via Adaptive Feature Fusion](https://arxiv.org/abs/2207.11221)
  - Domain generalization for activity recognition 领域泛化用于行为识别

- [The Importance of Background Information for Out of Distribution Generalization](https://arxiv.org/abs/2206.08794)
  - Background information for OOD generalization 背景信息对于OOD泛化的重要性

- [Causal Balancing for Domain Generalization](https://arxiv.org/abs/2206.05263)
  - Causal balancing for domain generalization 因果平衡用于领域泛化

- [Temporal Domain Generalization with Drift-Aware Dynamic Neural Network](https://arxiv.org/abs/2205.10664)
  - Temporal domain generalization with drift-aware dynamic neural network 时序域泛化

- [Multiple Domain Causal Networks](https://arxiv.org/abs/2205.06791)
  - Mlutiple domain causal networks 多领域的因果网络

- IJCAI-21 [Test-time Fourier Style Calibration for Domain Generalization](https://arxiv.org/abs/2205.06427)
  - Test-time calibration for domain generalization 用傅立叶变化进行域泛化的测试时矫正

- [Out-Of-Distribution Detection In Unsupervised Continual Learning](https://arxiv.org/abs/2204.05462)
  - OOD detection in unsupervised continual learning 无监督持续学习中进行OOD检测

- ICLR-22 [Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution](https://openreview.net/forum?id=UYneFzXSJWh)
  - Fin-tuning and linear probing for ood generalization
  - 先linear probing最后一层再finetune对OOD任务最好

- ICLR-22 [Asymmetry Learning for Counterfactually-invariant Classification in OOD Tasks](https://openreview.net/forum?id=avgclFZ221l)
  - Asymmetry learning for OOD tasks
  - 非对称学习用于OOD任务

- [Improving Generalization in Federated Learning by Seeking Flat Minima](https://arxiv.org/abs/2203.11834)
  - Seeking flat minima for domain generalization in federated learning
  - 通过寻找平坦值进行联邦学习领域泛化

- [Gated Domain-Invariant Feature Disentanglement for Domain Generalizable Object Detection](https://arxiv.org/abs/2203.11432)
  - Channel masking for domain generalization object detection
  - 通过一个gate控制channel masking进行object detection DG

- [A Broad Study of Pre-training for Domain Generalization and Adaptation](https://arxiv.org/abs/2203.11819)
  - A broad study of pre-training models for DA and DG
  - 大量的实验进行DA和DG

- [Learning Semantic Segmentation from Multiple Datasets with Label Shifts](https://arxiv.org/abs/2202.14030)
  - Learning semantic segmentation from many datasets with label shifts
  - 在有标签漂移的情况下从多个数据集中学习语义分割

- PAKDD-22 [Layer Adaptive Deep Neural Networks for Out-of-distribution Detection](https://arxiv.org/abs/2203.00192)
  - Layer adaptive network for OOD detection
  - 层自适应的网络进行OOD检测

- ICLR-22 oral [A Fine-Grained Analysis on Distribution Shift](https://openreview.net/forum?id=Dl4LetuLdyK)
  - Extensive experiments on distribution shift for OOD
  - 大量的实验进行OOD验证

- ICLR-22 oral [Fine-Tuning Distorts Pretrained Features and Underperforms Out-of-Distribution](https://openreview.net/forum?id=UYneFzXSJWh)
  - Fine-tuning with linear probing for OOD
  - 微调加上linear probing用于OOD

- ICLR-22 [Uncertainty Modeling for Out-of-Distribution Generalization](https://arxiv.org/abs/2202.03958)
  - Uncertainty modeling for OOD generalization
  - 用于分布外泛化的不确定性建模

- TKDE-22 [Adaptive Memory Networks with Self-supervised Learning for Unsupervised Anomaly Detection](https://arxiv.org/abs/2201.00464)
  - Adaptiev memory network for anomaly detection
  - 自适应的记忆网络用于异常检测

- ICIP-22 [Meta-Learned Feature Critics for Domain Generalized Semantic Segmentation](https://arxiv.org/abs/2112.13538)
  - Meta-learning for domain generalization
  - 元学习用于domain generalization

- ICIP-22 [Few-Shot Classification in Unseen Domains by Episodic Meta-Learning Across Visual Domains](https://arxiv.org/abs/2112.13539)
  - Few-shot generalization using meta-learning
  - 用元学习进行小样本的泛化

- [More is Better: A Novel Multi-view Framework for Domain Generalization](https://arxiv.org/abs/2112.12329)
    - Multi-view learning for domain generalization
    - 使用多视图学习来进行domain generalization

- [Unsupervised Domain Generalization by Learning a Bridge Across Domains](https://arxiv.org/abs/2112.02300)
    - Unsupervised domain generalization
    - 无监督的domain generalization

- [ROBIN : A Benchmark for Robustness to Individual Nuisancesin Real-World Out-of-Distribution Shifts](https://arxiv.org/abs/2111.14341)
    - A benchmark for robustness to individual OOD
    - 一个OOD的benchmark

- ICML-21 workshop [Towards Principled Disentanglement for Domain Generalization](https://arxiv.org/abs/2111.13839)
    - Principled disentanglement for domain generalization
    - Principled解耦用于domain generalization

- [Federated Learning with Domain Generalization](https://arxiv.org/abs/2111.10487)
    - Federated domain generalization
    - 联邦学习+domain generalization

  - [Semi-Supervised Domain Generalization in Real World:New Benchmark and Strong Baseline](https://arxiv.org/abs/2111.10221)
    - Semi-supervised domain generalization
    - 半监督+domain generalization

  - MICCAI-21 [Domain Generalization for Mammography Detection via Multi-style and Multi-view Contrastive Learning](https://arxiv.org/abs/2111.10827)
    - Domain generalization for mammography detection
    - 领域泛化用于乳房X射线检查

- WACV-21 [Domain Generalization through Audio-Visual Relative Norm Alignment in First Person Action Recognition](https://arxiv.org/abs/2110.10101)
    - Domain generalization by audio-visual alignment
    - 通过音频-视频对齐进行domain generalization

- [Dynamically Decoding Source Domain Knowledge For Unseen Domain Generalization](http://arxiv.org/abs/2110.03027)
  - Ensemble learning for domain generalization
  - 用集成学习进行domain generalization

- [Scale Invariant Domain Generalization Image Recapture Detection](http://arxiv.org/abs/2110.03496)
  - Scale invariant domain generalizaiton
  - 尺度不变的domain generalization

- ICCV-21 [Shape-Biased Domain Generalization via Shock Graph Embeddings](https://arxiv.org/abs/2109.05671)
  - Domain generalization based on shape information
  - 基于形状进行domain generalization

- [Domain and Content Adaptive Convolution for Domain Generalization in Medical Image Segmentation](https://arxiv.org/abs/2109.05676)
  - Domain generalization for medical image segmentation
  - 领域泛化用于医学图像分割

- [Fishr: Invariant Gradient Variances for Out-of-distribution Generalization](https://arxiv.org/abs/2109.02934)
    - Invariant gradient variances for OOD generalization
    - 不变梯度方差，用于OOD

- [Class-conditioned Domain Generalization via Wasserstein Distributional Robust Optimization](https://arxiv.org/abs/2109.03676)
    - Domain generalization with wasserstein DRO
    - 使用Wasserstein DRO进行domain generalization


- CIKM-21 [AdaRNN: Adaptive Learning and Forecasting of Time Series](https://arxiv.org/abs/2108.04443) [Code](https://github.com/jindongwang/transferlearning/tree/master/code/deep/adarnn) [知乎文章](https://zhuanlan.zhihu.com/p/398036372) [Video](https://www.bilibili.com/video/BV1Gh411B7rj/)
    - A new perspective to using transfer learning for time series analysis
    - 一种新的建模时间序列的迁移学习视角

- 20190531 arXiv [Image Alignment in Unseen Domains via Domain Deep Generalization](https://arxiv.org/abs/1905.12028)
  	- Deep domain generalization for image alignment
  	- 深度领域泛化用于图像对齐

- 20200821 ECCV-20 [Towards Recognizing Unseen Categories in Unseen Domains](https://arxiv.org/abs/2007.12256)
    - Recognizing unseen classes in unseen domains
    - 对未知领域识别未知类

- 20200706 ICLR-21 [In Search of Lost Domain Generalization](https://arxiv.org/abs/2007.01434)

- 20201016 [Energy-based Out-of-distribution Detection](https://arxiv.org/abs/2010.03759)
    - Energy-based OOD

- 20201222 AAAI-21 [DecAug: Out-of-Distribution Generalization via Decomposed Feature Representation and Semantic Augmentation](http://arxiv.org/abs/2012.09382)
    - OOD generalization
    - 用特征分解和语义增强做OOD泛化

- 20210106 [Style Normalization and Restitution for Domain Generalization and Adaptation](http://arxiv.org/abs/2101.00588)
    - Style normalization and restitution for DA and DG
    - 风格归一化用于DA和DG任务

- CVPR-21 [Uncertainty-Guided Model Generalization to Unseen Domains](https://openaccess.thecvf.com/content/CVPR2021/html/Qiao_Uncertainty-Guided_Model_Generalization_to_Unseen_Domains_CVPR_2021_paper.html)
  - Uncertainty-guided generalization
  - 基于不确定性的domain generalization

- CVPR-21 [Adaptive Methods for Real-World Domain Generalization](https://openaccess.thecvf.com/content/CVPR2021/html/Dubey_Adaptive_Methods_for_Real-World_Domain_Generalization_CVPR_2021_paper.html)
  - Adaptive methods for domain generalization
  - 动态算法，用于domain generalization

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

## Source-free domain adaptation

- NeurIPS'23 When Visual Prompt Tuning Meets Source-Free Domain Adaptive Semantic Segmentation [[paper](https://openreview.net/forum?id=ChGGbmTNgE)]
  - Source-free domain adaptation using visual prompt tuning

- PromptStyler: Prompt-driven Style Generation for Source-free Domain Generalization [[arxiv](https://arxiv.org/abs/2307.15199)]
  - Prompt-driven style generation for source-free domain generalization

- Source-Free Collaborative Domain Adaptation via Multi-Perspective Feature Enrichment for Functional MRI Analysis [[arxiv](http://arxiv.org/abs/2308.12495)]
  - Source-free domain adaptation for MRI analysis

- ICCV'23 Domain-Specificity Inducing Transformers for Source-Free Domain Adaptation [[arxiv](https://arxiv.org/abs/2308.14023)]
  - Domain-specificity for source-free DA 用领域特异性驱动的source-free DA

- Visual Prompt Tuning for Test-time Domain Adaptation [[arxiv](http://arxiv.org/abs/2210.04831)]
  - VPT for test-time adaptation 用prompt tuning进行test-time DA

- [Active Source Free Domain Adaptation](https://arxiv.org/abs/2205.10711)
  - Active source-free DA 主动学习-无源域DA

- NeurIPS-21 [Model Adaptation: Historical Contrastive Learning for Unsupervised Domain Adaptation without Source Data](http://arxiv.org/abs/2110.03374)
    - Source-free domain adaptation using constrastive learning
    - 无源域数据的DA，利用对比学习

- 20200706 [Domain Adaptation without Source Data](https://arxiv.org/abs/2007.01524)

- 20200629 ICML-20 [Do We Really Need to Access the Source Data? Source Hypothesis Transfer for Unsupervised Domain Adaptation](https://arxiv.org/abs/2002.08546)
    	- Source-free adaptation
    	- 在adaptation过程中不访问source data

- - -

## Multi-source domain adaptation

- [Open-Set Crowdsourcing using Multiple-Source Transfer Learning](https://arxiv.org/abs/2111.04073)
    - Open-set crowdsourcing using multiple-source transfer learning
    - 使用多源迁移进行开放集的crowdsourcing

- BMVC-21 [Domain Attention Consistency for Multi-Source Domain Adaptation](https://arxiv.org/abs/2111.03911)
    - Multi-source domain adaptation using attention consistency
    - 用attention一致性进行多源的domain adaptation

- CVPR-21 [Wasserstein Barycenter for Multi-Source Domain Adaptation](https://openaccess.thecvf.com/content/CVPR2021/html/Montesuma_Wasserstein_Barycenter_for_Multi-Source_Domain_Adaptation_CVPR_2021_paper.html)
  - Use Wasserstein Barycenter for multi-source domain adaptation
  - 利用Wasserstein Barycenter进行DA

- 20210430 [Graphical Modeling for Multi-Source Domain Adaptation](http://arxiv.org/abs/2104.13057)
    - Graphical models for multi-source DA
    - 用概率图模型进行多源领域自适应
- 20210430 [Unsupervised Multi-Source Domain Adaptation for Person Re-Identification](http://arxiv.org/abs/2104.12961)
    - ReID using multi-source DA
    - 用多源领域自适应进行ReID任务

- 20200427 [TriGAN: Image-to-Image Translation for Multi-Source Domain Adaptation](https://arxiv.org/abs/2004.08769)
  	- A cycle-gan style multi-source DA
  	- 类似于cyclegan的多源领域适应

- 20190902 AAAI-19 [Aligning Domain-Specific Distribution and Classifier for Cross-Domain Classification from Multiple Sources](https://www.aaai.org/ojs/index.php/AAAI/article/download/4551/4429)
  	- Multi-source domain adaptation using both features and classifier adaptation
  	- 利用特征和分类器同时适配进行多源迁移，效果很好

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

## Heterogeneous transfer learning

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

## Online transfer learning

- CVPR-22 workshop [Online Unsupervised Domain Adaptation for Person Re-identification](https://arxiv.org/abs/2205.04383)
  - Online domain adaptation for REID 在线adaptation

- [Mixture of basis for interpretable continual learning with distribution shifts](https://arxiv.org/abs/2201.01853)
  - Incremental learning with mixture of basis
  - 用mixture of domains进行增量学习

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

## Zero-shot / few-shot learning

- [Few-Max: Few-Shot Domain Adaptation for Unsupervised Contrastive Representation Learning](https://arxiv.org/abs/2206.10137)
  - Few-shot DA for unsupervised constrastive learning 小样本DA用于无监督对比学习

- [Interpretable Concept-based Prototypical Networks for Few-Shot Learning](https://arxiv.org/abs/2202.13474)
  - Concept-based prototypical network for few-shot learning
  - 基于概念的原型网络用于小样本学习

- [How Well Do Self-Supervised Methods Perform in Cross-Domain Few-Shot Learning?](https://arxiv.org/abs/2202.09014)
  - Self-supervised learning for cross-domain few-shot
  - 自监督用于跨领域小样本

- 20181128 arXiv [One Shot Domain Adaptation for Person Re-Identification](https://arxiv.org/abs/1811.10144)
	- One shot learning for REID
	- One shot for再识别

- 20210426 [Few-shot Continual Learning: a Brain-inspired Approach](http://arxiv.org/abs/2104.09034)
    - Few-shot continual learning
    - 小样本持续学习

- 20201203 [How to fine-tune deep neural networks in few-shot learning?](https://arxiv.org/abs/2012.00204)
    - 对few-shot任务如何fine-tune深度网络？

- 20201116 [Filter Pre-Pruning for Improved Fine-tuning of Quantized Deep Neural Networks](https://arxiv.org/abs/2011.06751)
    - 量子神经网络中的finetune

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

- 20191204 arXiv [MetAdapt: Meta-Learned Task-Adaptive Architecture for Few-Shot Classification](https://arxiv.org/abs/1912.00412)
     - Task adaptive structure for few-shot learning
     - 目标自适应的结构用于小样本学习

- 20190409 ICLR-19 [A Closer Look at Few-shot Classification](https://arxiv.org/abs/1904.04232)
    - Give some important conclusions on few-shot classification
    - 在few-shot上给了一些有用的结论

- 20190401 IJCNN-19 [Zero-shot Image Recognition Using Relational Matching, Adaptation and Calibration](https://arxiv.org/abs/1903.11701)
    - Zero-shot image recognition
    - 零次学习的图像识别



- 20171022 ICCVW-17 [Zero-shot learning posed as a missing data problem](http://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w38/Zhao_Zero-Shot_Learning_Posed_ICCV_2017_paper.pdf)
    - 算法首先学习 semantic embeddings 的结构性知识，利用学习到的知识和已知类的 image features 合成未知类的 image features。再利用无标记的未知类数据对合成数据进行修正。 算法假设未知类数据呈混合高斯分布，用 GMM-EM 算法进行无监督修正。
    
- 20180516 arXiv-18 [A Large-scale Attribute Dataset for Zero-shot Learning](https://arxiv.org/pdf/1804.04314v2.pdf)
    - 传统 ZSL 数据集（如 AwA, CUB）存在规模小，属性标注不丰富等问题。本文提出一个新的属性数据集 LAD 用于测试零样本学习算法。新数据集包含 230 类， 78,017 张图片，标注了 359 种属性。基于此数据集举办了 AI Challenger 零样本学习竞赛。 110+ 支来自海内外的参赛队伍提交了成绩。
    
- 20180710 ICML-18 [MSplit LBI: Realizing Feature Selection and Dense Estimation Simultaneously in Few-shot and Zero-shot Learning](https://arxiv.org/pdf/1806.04360.pdf)
    - 针对 L1 （欠拟合） 和 L2 （无特征选择、有偏） 正则项存在的问题，提出 MSplit LBI 用于同时实现特征选择和密集估计。在 Few-shot Learning 和 Zero-shot Learning 两个问题上进行了实验。实验表明 MSplit LBI 由优于 L1 和 L2。针对 ZSL 进行了特征可视化实验。
    
- 20190108 WACV-19 [Zero-shot Learning via Recurrent Knowledge Transfer](https://drive.google.com/open?id=1cUsQWX80zeCxTyVSCcYlqEWZP-Hq0KzR)
    - 基于样本合成的零样本学习算法通常将 semantic embeddings 的知识迁移到 image features 以实现 ZSL。然而，这种 training 和 testing space 的不一致，会导致这种迁移失效。因此，本文提出 Space Shift Problem，并针对此问题，提出一种（在 image feature space 和 semantic embedding space 之间）递归传递知识的解决方案。

- - -


## Multi-task learning

- [Gap Minimization for Knowledge Sharing and Transfer](https://arxiv.org/abs/2201.11231)
  - Multitask learning with gap minimization
  - 用于多任务学习的gap minimization方法

- 20190806 KDD-19 [Relation Extraction via Domain-aware Transfer Learning](https://dl.acm.org/citation.cfm?id=3330890)
    - Relation extraction using transfer learning for knowledge base construction
    - 利用迁移学习进行关系抽取

- 20190531 arXiv [Multi-task Learning in Deep Gaussian Processes with Multi-kernel Layers](https://arxiv.org/abs/1905.12407)
  	- Multi-task learning in deep Gaussian process
  	- 深度高斯过程中的多任务学习

- 20200927 [Knowledge Distillation for Multi-task Learning](https://arxiv.org/abs/2007.06889)
    - 针对多任务学习的知识蒸馏

- 20200914 ECML-PKDD-20 [Towards Interpretable Multi-Task Learning Using Bilevel Programming](https://arxiv.org/abs/2009.05483)
    - 用bilevel programming解释多任务学习

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
	- Multi-task clustering
	- 多任务聚类

- 20180622 arXiv 探索了多任务迁移学习中的不确定性：[Uncertainty in Multitask Transfer Learning](https://arxiv.org/abs/1806.07528)

- 20180524 arXiv 杨强团队、与之前的learning to learning类似，这里提供了一个从经验中学习的learning to multitask框架：[Learning to Multitask](https://arxiv.org/abs/1805.07541)

- - -

## Transfer reinforcement learning

- [Multi-Agent Transfer Learning in Reinforcement Learning-Based Ride-Sharing Systems](https://arxiv.org/abs/2112.00424)
    - Multi-agent transfer in RL
    - 在RL中的多智能体迁移

- NeurIPS-21 workshop [Component Transfer Learning for Deep RL Based on Abstract Representations](https://arxiv.org/abs/2111.11525)
    - Deep transfer learning for RL
    - 深度迁移学习用于强化学习

- [Xi-Learning: Successor Feature Transfer Learning for General Reward Functions](https://arxiv.org/abs/2110.15701)
    - General reward function transfer learning in RL
    - 在强化学习中general reward function的迁移学习

- NeurIPS-21 [Unsupervised Domain Adaptation with Dynamics-Aware Rewards in Reinforcement Learning](https://arxiv.org/abs/2110.12997)
    - Domain adaptation in reinforcement learning
    - 在强化学习中应用domain adaptation

- [Understanding Domain Randomization for Sim-to-real Transfer](http://arxiv.org/abs/2110.03239)
    - Understanding domain randomizationfor sim-to-real transfer
    - 对强化学习中的sim-to-real transfer进行理论上的分析

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
    - Reinforcement transfer learning with latent models
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

## Transfer metric learning

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

- - -

## Federated transfer learning

- ZooPFL: Exploring Black-box Foundation Models for Personalized Federated Learning [[arxiv](https://arxiv.org/abs/2310.05143)]
  - Black-box foundation models for personalized federated learning 黑盒的blackbox模型进行个性化迁移学习

- Benchmarking Algorithms for Federated Domain Generalization [[arxiv](http://arxiv.org/abs/2307.04942)]
  - Benchmark algorthms for federated domain generalization 对联邦域泛化算法进行的benchmark

- IEEE'23 FedCLIP: Fast Generalization and Personalization for CLIP in Federated Learning [[arxiv](https://arxiv.org/abs/2302.13485v1)]
  - Fast generalization for federated CLIP 在联邦中进行快速的CLIP训练

- [Federated Semi-Supervised Domain Adaptation via Knowledge Transfer](https://arxiv.org/abs/2207.10727)
  - Federated semi-supervised DA 联邦半监督DA

- FL-IJCAI-22 [MetaFed: Federated Learning among Federations with Cyclic Knowledge Distillation for Personalized Healthcare](https://arxiv.org/abs/2206.08516)
  - MetaFed: a new form of federated learning 
  - 联邦之联邦学习、新范式
- Interspeech-22 [Decoupled Federated Learning for ASR with Non-IID Data](https://jd92.wang/assets/files/DecoupleFL-IS22.pdf)
  - Decoupled federated learning for non IID 
  - 解耦的联邦架构用于Non-IID语音识别
- [Test-Time Robust Personalization for Federated Learning](https://arxiv.org/abs/2205.10920)
  - Test-time robust personalization for FL 
  - 测试时鲁棒联邦学习
- IEEE TNNLS-22 [Towards Personalized Federated Learning](http://arxiv.org/abs/2103.00710)
  - A survey on personalized federated learning 
  - 一个关于个性化联邦学习的综述
- [Improving Generalization in Federated Learning by Seeking Flat Minima](https://arxiv.org/abs/2203.11834)
  - Seeking flat minima for domain generalization in federated learning
  - 通过寻找平坦值进行联邦学习领域泛化
- [SemiPFL: Personalized Semi-Supervised Federated Learning Framework for Edge Intelligence](https://arxiv.org/abs/2203.08176)
  - Personalized federated learning
  - 个性化联邦学习
- NeurIPS-21 [Parameterized Knowledge Transfer for Personalized Federated Learning](https://proceedings.neurips.cc/paper/2021/hash/5383c7318a3158b9bc261d0b6996f7c2-Abstract.html)
    - personalized group knowledge transfer training
    - 个性化群体知识迁移

- ICML-21 [Federated Continual Learning with Weighted Inter-client Transfer](https://proceedings.mlr.press/v139/yoon21b.html)
    - Federated Weighted Inter-client Transfer (FedWeIT) for Federated Continual Learning
    - 联邦加权客户端间传输方法，用于联邦持续学习

- SIGIR-21 [FedCT: Federated Collaborative Transfer for Recommendation](https://doi.org/10.1145/3404835.3462825)
    - Federated learning for cross-domain recommendation 
    - 使用联邦迁移学习执行跨域推荐任务

- KDD-21 [Federated Adversarial Debiasing for Fair and Transferable Representations](https://doi.org/10.1145/3447548.3467281)
    - Federated Adversarial DEbiasing (FADE)
    - 通过对抗性学习对联邦学习过程去除偏见

- [Federated Learning with Adaptive Batchnorm for Personalized Healthcare](https://arxiv.org/abs/2112.00734)
    - Federated learning with adaptive batchnorm
    - 用自适应BN进行个性化联邦学习
- [FedZKT: Zero-Shot Knowledge Transfer towards Heterogeneous On-Device Models in Federated Learning](https://arxiv.org/abs/2109.03775)
    - Zero-shot transfer in heterogeneous federated learning
    - 零次迁移用于联邦学习
- [Federated Multi-Task Learning under a Mixture of Distributions](https://arxiv.org/abs/2108.10252)
    - Federated multi-task learning
    - 联邦多任务学习
- NeurIPS-20 [Group Knowledge Transfer: Federated Learning of Large CNNs at the Edge](https://proceedings.neurips.cc/paper/2020/hash/a1d4c20b182ad7137ab3606f0e3fc8a4-Abstract.html)
    - Group knowledge transfer training
    - 群体知识迁移

- [Fine-tuning is Fine in Federated Learning](http://arxiv.org/abs/2108.07313)
    - Finetuning in federated learning
    - 在联邦学习中进行finetune
- [Federated Multi-Target Domain Adaptation](http://arxiv.org/abs/2108.07792)
    - Federated multi-target DA
    - 联邦学习场景下的多目标DA
- 20190909 IJCAI-FML-19 [FedHealth: A Federated Transfer Learning Framework for Wearable Healthcare](http://jd92.wang/assets/files/a15_ijcai19.pdf)
  	- The first work on federated transfer learning for wearable healthcare
  	- 第一个将联邦迁移学习用于可穿戴健康监护的工作
- 20180605 arXiv 解决federated learning中的数据不同分布的问题：[Federated Learning with Non-IID Data](https://arxiv.org/abs/1806.00582)
- 20190301 NeurIPS-18 workshp [One-Shot Federated Learning](https://arxiv.org/abs/1902.11175)
    - One-shot federated learning

- - -

## Lifelong transfer learning

- Complementary Domain Adaptation and Generalization for Unsupervised Continual Domain Shift Learning [[arxiv](http://arxiv.org/abs/2303.15833)]
  - Continual domain shift learning using adaptation and generalization 使用 adaptation和DG进行持续分布变化的学习

- TMLR'23 Learn, Unlearn and Relearn: An Online Learning Paradigm for Deep Neural Networks [[arxiv](http://arxiv.org/abs/2303.10455)]
  - A framework for online learning 一个在线学习的框架

- NeurIPS'22 Beyond Not-Forgetting: Continual Learning with Backward Knowledge Transfer [[arxiv](http://arxiv.org/abs/2211.00789)]
  - Continual learning with backward knowledge transfer 反向知识迁移的持续学习

- [Mixture of basis for interpretable continual learning with distribution shifts](https://arxiv.org/abs/2201.01853)
  - Incremental learning with mixture of basis
  - 用mixture of domains进行增量学习

- 20101008 arXiv [Concept-drifting Data Streams are Time Series; The Case for Continuous Adaptation](https://arxiv.org/abs/1810.02266)
	- Continuous adaptation for time series data
	- 对时间序列进行连续adaptation

- 20191011 arXiv [Learning to Remember from a Multi-Task Teacher](https://arxiv.org/abs/1910.04650)
  	- Dealing with the catastrophic forgetting during sequential learning
  	- 在序列学习时处理灾难遗忘

- 20191029 [Adversarial Feature Alignment: Avoid Catastrophic Forgetting in Incremental Task Lifelong Learning](https://arxiv.org/abs/1910.10986)
  	- Avoid catastrophic forgeeting in incremental task lifelong learning
  	- 在终身学习中避免灾难遗忘

- 20200706 [ICML-20] [Continuously Indexed Domain Adaptation](https://arxiv.org/abs/2007.01807)

- 20210716 TPAMI-21 [Lifelong Teacher-Student Network Learning](https://arxiv.org/abs/2107.04689)
  - Lifelong distillation
  - 持续的知识蒸馏

- 20210716 ICML-21 [Continual Learning in the Teacher-Student Setup: Impact of Task Similarity](https://arxiv.org/abs/2107.04384)
    - Investigating task similarity in teacher-student learning
    - 调研在continual learning下teacher-student learning问题的任务相似度

- 20190912 NeurIPS-19 [Meta-Learning with Implicit Gradients](https://arxiv.org/abs/1909.04630)
  - Meta-learning with implicit gradients
  - 隐式梯度的元学习

- 20180323 arXiv 终身迁移学习与增量学习结合：[Incremental Learning-to-Learn with Statistical Guarantees](https://arxiv.org/abs/1803.08089)

- 20180111 arXiv 一种新的终身学习框架，与L2T的思路有一些类似 [Lifelong Learning for Sentiment Classification](https://arxiv.org/abs/1801.02808)

- - -


## Safe transfer learning

- ICSE-22 [ReMoS: Reducing Defect Inheritance in Transfer Learning via Relevant Model Slicing](https://link.zhihu.com/?target=https%3A//jd92.wang/assets/files/icse22-remos.pdf) | [Code](https://github.com/ziqi-zhang/ReMoS_artifact) | [Blog](https://zhuanlan.zhihu.com/p/446453487) | [Video](https://www.bilibili.com/video/BV1mi4y1C7bP)
  - Safe transfer learning by reducing defect inheritance
  - 安全迁移学习的最新工作

- CVPR workshop-21 [Renofeation: A Simple Transfer Learning Method for Improved Adversarial Robustness](https://openaccess.thecvf.com/content/CVPR2021W/TCV/html/Chin_Renofeation_A_Simple_Transfer_Learning_Method_for_Improved_Adversarial_Robustness_CVPRW_2021_paper.html)
  - Improve adversarial robustness of transfer learning models
  - 提高迁移学习对于adversarial robustness的鲁棒性

- ICLR-20 [A Target-Agnostic Attack on Deep Models: Exploiting Security Vulnerabilities of Transfer Learning](https://openreview.net/forum?id=BylVcTNtDS)
  - Softmax layer is easy to get attacked
  - 设计实验来攻击迁移学习的softmax layer

- RAID'18 [Fine-Pruning: Defending Against Backdooring Attacks on Deep Neural Networks](https://link.springer.com/chapter/10.1007/978-3-030-00470-5_13)
  - Finetune and prune the weights against backdoor attack
  - 在finetune过程中剪枝来预防后门攻击

- ACM CCS-18 [Model-Reuse Attacks on Deep Learning Systems](https://dl.acm.org/doi/10.1145/3243734.3243757)
  - Model-resuse attack on transfer learning models
  - 设计实验来攻击迁移学习的预训练模型

- USENIX Security-18 [With Great Training Comes Great Vulnerability: Practical Attacks against Transfer Learning](https://www.usenix.org/system/files/conference/usenixsecurity18/sec18-wang.pdf)
  - First work to design experiments to attack pretrained models
  - 第一个设计实验来攻击预训练模型的工作

- - -

## Transfer learning applications

See [HERE](https://github.com/jindongwang/transferlearning/blob/master/doc/transfer_learning_application.md) for a full list of transfer learning applications.