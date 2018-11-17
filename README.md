# 迁移学习 Transfer Learning

[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT) 

关于迁移学习的所有资料，包括：介绍、综述文章、最新文章、代表工作及其代码、常用数据集、硕博士论文、比赛等等。(可能是**目前最全**的迁移学习资料库？) *欢迎一起贡献！*

Everything about Transfer Learning (Probably the **most complete** repository?). *Your contribution is highly valued!*

如果认为本仓库有用，请在你的论文和其他出版物中进行引用！ If you find this repo helpful, please cite it as follows:

```
@Misc{transferlearning.xyz,
howpublished = {\url{http://transferlearning.xyz}},   
title = {Everything about Transfer Learning and Domain Adapation},  
author = {Wang, Jindong and others}  
}  
```

_ _ _

## 目录 Table of contents

* [最新文章 Latest](#0latest)

* [迁移学习简介 Introduction to transfer learning](#1Introduction-and-Tutorials)

* [研究领域与相关文章 Research articles by area](#2Transfer-Learning-Areas-and-Papers) 

* [理论与综述文章 Theoretical and survey papers](#3Theory-and-Survey)

* [相关代码 Available codes](#4Code)

* [迁移学习代表性研究学者 Scholars](#5Transfer-Learning-Scholars)

* [相关的硕博士论文 Thesis](#6Transfer-Learning-Thesis)

* [常用公开数据集及算法结果 Datasets and benchmark](#7Datasets-and-Benchmarks)

* [比赛 Challenges and competitions](#8Transfer-Learning-Challenges)

* [迁移学习应用 Transfer learning applications](#Applications)

* [其他 Other resources](#Other-Resources)

* [Contributing](#contributing)

> 关于机器学习和行为识别的资料，请参考：[行为识别](https://github.com/jindongwang/activityrecognition)｜[机器学习](https://github.com/jindongwang/MachineLearning)

- - -

## 0.Latest

[迁移学习文章汇总 Awesome transfer learning papers](https://github.com/jindongwang/transferlearning/tree/master/doc/awesome_paper.md)

- **Latest publications**

	- 20181117 AAAI-19 [GaitSet: Regarding Gait as a Set for Cross-View Gait Recognition](https://arxiv.org/abs/1811.06186)
		- English: Cross-view gait recognition
		- 中文：跨视图的步态识别

	- 20181117 AAAI-19 [Robust Optimization over Multiple Domains](https://arxiv.org/abs/1805.07588)
		- English: Optimization on multi domains
		- 中文：针对多个domain建模并优化

	- 20181115 AAAI-19 [Unsupervised Transfer Learning for Spoken Language Understanding in Intelligent Agents](https://arxiv.org/abs/1811.05232)
		- English: Transfer learning for spoken language understanding
		- 中文：无监督迁移学习用于语言理解

	- 20181115 AAAI-19 [Exploiting Local Feature Patterns for Unsupervised Domain Adaptation](https://arxiv.org/abs/1811.05042)
		- English: Local domain alignment for domain adaptation
		- 中文：局部领域自适应

	 - 20181115 NIPS-18 [Co-regularized Alignment for Unsupervised Domain Adaptation](https://arxiv.org/abs/1811.05443)
		- English: The idea is the same with the above one...
		- 中文：仍然是局部对齐。。。

- **Preprints on arXiv** (Not peer-reviewed)

	- 20181117 arXiv [AdapterNet - learning input transformation for domain adaptation](https://arxiv.org/abs/1805.11601)
		- English: Learning input transformation for domain adaptation
		- 中文：对domain adaptation任务学习输入的自适应

	- 20181117 arXiv [Distance Measure Machines](https://arxiv.org/abs/1803.00250)
		- English: Machines that measures distances
		- 中文：衡量距离的算法

	- 20181117 arXiv [Performance Estimation of Synthesis Flows cross Technologies using LSTMs and Transfer Learning](https://arxiv.org/abs/1811.06017)
		- English: Performance Estimation of Synthesis Flows cross Technologies using LSTMs and Transfer Learning
		- 中文：利用迁移学习进行合成flow评价

	- 20181117 arXiv [Unsupervised domain adaptation for medical imaging segmentation with self-ensembling](https://arxiv.org/abs/1811.06042)
		- English: Medical imaging using transfer learning
		- 中文：使用迁移学习进行医学图像分割

	- 20181117 arXiv [Theoretical Perspective of Deep Domain Adaptation](https://arxiv.org/abs/1811.06199)
		- English: Providing some theory analysis on deep domain adaptation
		- 中文：对deep domain adaptaiton做出了一些理论上的分析


[**更多 More...**](https://github.com/jindongwang/transferlearning/tree/master/doc/awesome_paper.md)

- - -

## 1.Introduction and Tutorials

- 简介文字资料
	- [简单的中文简介 Chinese introduction](https://github.com/jindongwang/transferlearning/blob/master/doc/%E8%BF%81%E7%A7%BB%E5%AD%A6%E4%B9%A0%E7%AE%80%E4%BB%8B.md)
	- [PPT(English)](http://jd92.wang/assets/files/l03_transferlearning.pdf)
	- [PPT(中文)](http://jd92.wang/assets/files/l08_tl_zh.pdf)
	- 迁移学习中的领域自适应方法 Domain adaptation: [PDF](http://jd92.wang/assets/files/l12_da.pdf) ｜ [Video](http://mp.weixin.qq.com/s?__biz=MzI5MDUyMDIxNA==&mid=2247484940&idx=2&sn=35e64e07fde9a96afbb65dbf40a945eb&chksm=ec1febf5db6862e38d5e02ff3278c61b376932a46c5628c7d9cb1769c572bfd31819c13dd468&mpshare=1&scene=1&srcid=1219JpTNZFiNDCHsTUrUxwqy#rd)
	- 清华大学龙明盛老师的深度迁移学习报告 Transfer learning report by Mingsheng Long @ THU：[PPT(Samsung)](http://ise.thss.tsinghua.edu.cn/~mlong/doc/transfer-learning-talk.pdf)、[PPT(Google China)](http://ise.thss.tsinghua.edu.cn/~mlong/doc/deep-transfer-learning-talk.pdf)

- 入门教程
	- [**《迁移学习简明手册》Transfer Learning Tutorial**](https://zhuanlan.zhihu.com/p/35352154) [开发维护地址](https://github.com/jindongwang/transferlearning-tutorial)

- 视频教程
	- [台湾大学李宏毅的视频讲解(中文视频)](https://www.youtube.com/watch?v=qD6iD4TFsdQ)
	- [迁移学习中的领域自适应方法(中文)](http://mp.weixin.qq.com/s?__biz=MzI5MDUyMDIxNA==&mid=2247484940&idx=2&sn=35e64e07fde9a96afbb65dbf40a945eb&chksm=ec1febf5db6862e38d5e02ff3278c61b376932a46c5628c7d9cb1769c572bfd31819c13dd468&mpshare=1&scene=1&srcid=1219JpTNZFiNDCHsTUrUxwqy#rd)

- [迁移学习领域的著名学者、代表工作及实验室介绍 Transfer Learning Scholars and Labs](https://github.com/jindongwang/transferlearning/blob/master/doc/scholar_TL.md)

- 什么是[负迁移(negative transfer)](https://www.zhihu.com/question/66492194/answer/242870418)？

- 动手教程、代码、数据 Hands-on Codes
	- [基于深度学习和迁移学习的识花实践 Using Transfer Learning for Flower Recognition](https://cosx.org/2017/10/transfer-learning/)
	- [基于Pytorch的图像分类 Using Transfer Learning for Image Classification](https://github.com/miguelgfierro/sciblog_support/blob/master/A_Gentle_Introduction_to_Transfer_Learning/Intro_Transfer_Learning.ipynb)
	- [使用Pytorch进行finetune Using Pytorch for Fine-tune](https://github.com/Spandan-Madan/Pytorch_fine_tuning_Tutorial)
	- [基于AlexNet和ResNet的finetune Fine-tune based on Alexnet and Resnet](https://github.com/jindongwang/transferlearning/tree/master/code/AlexNet_ResNet)

- - -

## 2.Transfer Learning Areas and Papers

Related articles by research areas:

- [领域自适应(非深度) Domain Adaptation (Shallow)](https://github.com/jindongwang/transferlearning/blob/master/doc/awesome_paper.md#领域自适应)
	- Domain adaptation介绍：[Domain adaptation](https://github.com/jindongwang/transferlearning/blob/master/doc/domain_adaptation.md)
	- 这个目录主要是非深度方法，深度方法在下面。

- [在线迁移学习 Online transfer learning](https://github.com/jindongwang/transferlearning/blob/master/doc/awesome_paper.md#在线迁移学习)

- [终身迁移学习 Lifelong transfer learning](https://github.com/jindongwang/transferlearning/blob/master/doc/awesome_paper.md#终身迁移学习)

- [异构迁移学习 Heterogeneous Transfer Learning](https://github.com/jindongwang/transferlearning/blob/master/doc/awesome_paper.md#异构迁移学习)

- [深度迁移学习 Deep Transfer Learning](https://github.com/jindongwang/transferlearning/blob/master/doc/awesome_paper.md#深度迁移学习)
  
    - [深度对抗迁移迁移学习 Deep Adversarial transfer learning](https://github.com/jindongwang/transferlearning/blob/master/doc/awesome_paper.md#对抗迁移学习)

- [传递迁移学习 Transitive Transfer Learning](https://github.com/jindongwang/transferlearning/blob/master/doc/awesome_paper.md#传递迁移学习)

- [强化迁移学习 Transfer Learning with Reinforcement Learning](https://github.com/jindongwang/transferlearning/blob/master/doc/awesome_paper.md#强化迁移学习)

- [应用 Applications](https://github.com/jindongwang/transferlearning/blob/master/doc/awesome_paper.md#应用)
	- [迁移学习用于行为识别 Transfer learning for activity recognition](https://github.com/jindongwang/activityrecognition/blob/master/notes/%E8%BF%81%E7%A7%BB%E5%AD%A6%E4%B9%A0%E7%94%A8%E4%BA%8E%E8%A1%8C%E4%B8%BA%E8%AF%86%E5%88%AB.md)

一个推荐、分享论文的网站比较好，我在上面会持续整理相关的文章并分享阅读笔记。详情请见[paperweekly](http://www.paperweekly.site/collections/231/papers)。

- - -

## 3.Theory and Survey

Here are some articles on transfer learning theory and survey.

- 迁移学习领域最具代表性的综述是[A survey on transfer learning](http://ieeexplore.ieee.org/abstract/document/5288526/)，发表于2010年，对迁移学习进行了比较权威的定义。 -- The most influential survey on transfer learning.

- 迁移学习的**理论分析** Transfer Learning Theory：

	- 迁移学习方面一直以来都比较缺乏理论分析与证明的文章，以下三篇连贯式的理论文章成为了经典 Transfer learning theory：
		- NIPS-06 [Analysis of Representations for Domain Adaptation](https://dl.acm.org/citation.cfm?id=2976474)
		- ML-10 [A Theory of Learning from Different Domains](https://link.springer.com/article/10.1007/s10994-009-5152-4)
		- NIPS-08 [Learning Bounds for Domain Adaptation](http://papers.nips.cc/paper/3212-learning-bounds-for-domain-adaptation)

	- 许多研究者在迁移学习的研究中会应用MMD(Maximum Mean Discrepancy)这个最大均值差异来衡量不同domain之间的距离。MMD的理论文章是：
		- MMD的提出：[A Hilbert Space Embedding for Distributions](https://link.springer.com/chapter/10.1007/978-3-540-75225-7_5) 以及 [A Kernel Two-Sample Test](http://www.jmlr.org/papers/v13/gretton12a.html)
		- 多核MMD(MK-MMD)：[Optimal kernel choice for large-scale two-sample tests](http://papers.nips.cc/paper/4727-optimal-kernel-choice-for-large-scale-two-sample-tests)
		- MMD及多核MMD代码：[Matlab](https://github.com/lopezpaz/classifier_tests/tree/master/code/unit_test_mmd) | [Python](https://github.com/jindongwang/transferlearning/tree/master/code/basic/mmd.py)
	- 理论研究方面，重点关注Alex Smola、Ben-David、Bernhard Schölkopf、Arthur Gretton等人的研究即可。

- 较新的综述 Latest survey：

	- 2018 一篇迁移度量学习的综述: [Transfer Metric Learning: Algorithms, Applications and Outlooks](https://arxiv.org/abs/1810.03944)
	- 2018 一篇最近的非对称情况下的异构迁移学习综述：[Asymmetric Heterogeneous Transfer Learning: A Survey](https://arxiv.org/abs/1804.10834)
	- 2018 Neural style transfer的一个survey：[Neural Style Transfer: A Review](https://arxiv.org/abs/1705.04058)
	- 2018 深度domain adaptation的一个综述：[Deep Visual Domain Adaptation: A Survey](https://www.sciencedirect.com/science/article/pii/S0925231218306684)
	- 2017 多任务学习的综述，来自香港科技大学杨强团队：[A survey on multi-task learning](https://arxiv.org/abs/1707.08114)
	- 2017 异构迁移学习的综述：[A survey on heterogeneous transfer learning](https://link.springer.com/article/10.1186/s40537-017-0089-0)
	- 2017 跨领域数据识别的综述：[Cross-dataset recognition: a survey](https://arxiv.org/abs/1705.04396)
	- 2016 [A survey of transfer learning](https://pan.baidu.com/s/1gfgXLXT)。其中交代了一些比较经典的如同构、异构等学习方法代表性文章。
	- 2015 中文综述：[迁移学习研究进展](https://pan.baidu.com/s/1bpautob)

- 迁移学习的应用
	- 视觉domain adaptation综述：[Visual Domain Adaptation: A Survey of Recent Advances](https://pan.baidu.com/s/1o8BR7Vc)
	- 迁移学习应用于行为识别综述：[Transfer Learning for Activity Recognition: A Survey](https://pan.baidu.com/s/1kVABOYr)
	- 迁移学习与增强学习：[Transfer Learning for Reinforcement Learning Domains: A Survey](https://pan.baidu.com/s/1slfr0w1)
	- 多个源域进行迁移的综述：[A Survey of Multi-source Domain Adaptation](https://pan.baidu.com/s/1eSGREF4)。

_ _ _

## 4.Code

请见[这里](https://github.com/jindongwang/transferlearning/tree/master/code) | Please see [HERE](https://github.com/jindongwang/transferlearning/tree/master/code) for some popular transfer learning codes.

_ _ _

## 5.Transfer Learning Scholars

Here are some transfer learning scholars and labs.

**全部列表以及代表工作性见[这里](https://github.com/jindongwang/transferlearning/blob/master/doc/scholar_TL.md)**

Please refer to [here](https://github.com/jindongwang/transferlearning/blob/master/doc/scholar_TL.md) to see a complete list.

- [Qiang Yang](http://www.cs.ust.hk/~qyang/)：中文名杨强。香港科技大学计算机系讲座教授，迁移学习领域世界性专家。IEEE/ACM/AAAI/IAPR/AAAS fellow。[[Google scholar](https://scholar.google.com/citations?user=1LxWZLQAAAAJ&hl=zh-CN)]

- [Sinno Jialin Pan](http://www.ntu.edu.sg/home/sinnopan/)：杨强的学生，香港科技大学博士，现任新加坡南洋理工大学助理教授。迁移学习领域代表性综述A survey on transfer learning的第一作者（Qiang Yang是二作）。[[Google scholar](https://scholar.google.com/citations?user=P6WcnfkAAAAJ&hl=zh-CN)]

- [Wenyuan Dai](https://scholar.google.com.sg/citations?user=AGR9pP0AAAAJ&hl=zh-CN)：中文名戴文渊，上海交通大学硕士，现任第四范式人工智能创业公司CEO。迁移学习领域著名的牛人，在顶级会议上发表多篇高水平文章，每篇论文引用量巨大。[[Google scholar](https://scholar.google.com.hk/citations?hl=zh-CN&user=AGR9pP0AAAAJ)]

- [Lixin Duan](http://www.lxduan.info/)：中文名段立新，新加坡南洋理工大学博士，现就职于电子科技大学，教授。[[Google scholar](https://scholar.google.com.hk/citations?user=inRIcS0AAAAJ&hl=zh-CN&oi=ao)]

- [Fuzhen Zhuang](http://www.intsci.ac.cn/users/zhuangfuzhen/)：中文名庄福振，中科院计算所博士，现任中科院计算所副研究员。[[Google scholar](https://scholar.google.com/citations?user=klJBYrAAAAAJ&hl=zh-CN&oi=ao)]

- [Mingsheng Long](http://ise.thss.tsinghua.edu.cn/~mlong/)：中文名龙明盛，清华大学博士，现任清华大学助理教授、博士生导师。[[Google scholar](https://scholar.google.com/citations?view_op=search_authors&mauthors=mingsheng+long&hl=zh-CN&oi=ao)]

- [Qingyao Wu](https://sites.google.com/site/qysite/)：中文名吴庆耀，现任华南理工大学副教授。主要做在线迁移学习、异构迁移学习方面的研究。[[Google scholar](https://scholar.google.com.hk/citations?user=n6e_2IgAAAAJ&hl=zh-CN&oi=ao)]

- [Weike Pan](https://sites.google.com/site/weikep/)：中文名潘微科，杨强的学生，现任深圳大学副教授，香港科技大学博士毕业。主要做迁移学习在推荐系统方面的一些工作。 [[Google Scholar](https://scholar.google.com/citations?user=pC5Q26MAAAAJ&hl=en)]

- [Tongliang Liu](http://ieeexplore.ieee.org/abstract/document/8259375/)：中文名刘同亮，现任悉尼大学助理教授。主要做迁移学习的一些理论方面的工作。[[Google scholar](https://scholar.google.com.hk/citations?hl=zh-CN&user=EiLdZ_YAAAAJ)]

- [Tatiana Tommasi](http://tatianatommasi.wixsite.com/tatianatommasi/3)：Researcher at the Italian Institute of Technology.
_ _ _

## 6.Transfer Learning Thesis

Here are some popular thesis on transfer learning.

硕博士论文可以让我们很快地对迁移学习的相关领域做一些了解，同时，也能很快地了解概括相关研究者的工作。其中，比较有名的有

- 2016 Baochen Sun的[Correlation Alignment for Domain Adaptation](http://www.cs.uml.edu/~bsun/papers/baochen_phd_thesis.pdf)

- 2014 南加州大学的Boqing Gong的[Kernel Methods for Unsupervised Domain Adaptation](https://pan.baidu.com/s/1bpbawv9)

- 2014 清华大学龙明盛的[迁移学习问题与方法研究](http://ise.thss.tsinghua.edu.cn/~mlong/doc/phd-thesis-mingsheng-long.pdf)

- 2014 中科院计算所赵中堂的[自适应行为识别中的迁移学习方法研究](https://pan.baidu.com/s/1kVqYXnh)

- 2012 杨强的学生Hao Hu的[Learning based Activity Recognition](https://pan.baidu.com/s/1bp2K9HX)

- 2012 杨强的学生Wencheng Zheng的[Learning with Limited Data in Sensor-based Human Behavior Prediction](https://pan.baidu.com/s/1o8MbbBk)

- 2010 杨强的学生Sinno Jialin Pan的[Feature-based Transfer Learning and Its Applications](https://pan.baidu.com/s/1bUqMfW)

- 2009 上海交通大学戴文渊的[基于实例和特征的迁移学习算法研究](https://pan.baidu.com/s/1i4Vyygd)

其他的文章，请见[完整版](https://pan.baidu.com/s/1bqXEASn)。

- - -

## 7.Datasets and Benchmarks

迁移学习领域有一些公开的数据集，用来对比算法的表现。

Please see [HERE](https://github.com/jindongwang/transferlearning/blob/master/data) for the popular transfer learning **datasets and certain benchmark** results.

[这里](https://github.com/jindongwang/transferlearning/blob/master/data)整理了常用的公开数据集和一些已发表的文章在这些数据集上的实验结果。

- - -

## 8.Transfer Learning Challenges

Here are some popular challenges on transfer learning. 一些关于迁移学习的国际比赛。

- [Visual Domain Adaptation Challenge (VisDA)](http://ai.bu.edu/visda-2018/)

- - -

## Applications

See [HERE](https://github.com/jindongwang/transferlearning/blob/master/doc/transfer_learning_application.md) for transfer learning applications.

迁移学习应用请见[这里](https://github.com/jindongwang/transferlearning/blob/master/doc/transfer_learning_application.md)。

- - -
  
## Other Resources

Related projects:

- Salad: [A semi-supervised domain adaptation library](https://domainadaptation.org)


- - -

## Contributing

If you are interested in contributing, please refer to [HERE](https://github.com/jindongwang/transferlearning/blob/master/CONTRIBUTING.md) for instructions in contribution.

> ***[文章版权声明]这个仓库是我开源到Github上的，可以遵守相关的开源协议进行使用。这个仓库中包含有很多研究者的论文、硕博士论文等，都来源于在网上的下载，仅作为学术研究使用。我对其中一些文章都写了自己的浅见，希望能很好地帮助理解。这些文章的版权属于相应的出版社。如果作者或出版社有异议，请联系我进行删除。一切都是为了更好地学术！***
