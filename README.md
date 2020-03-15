# 迁移学习 Transfer Learning  

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re) [![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![LICENSE](https://img.shields.io/badge/license-Anti%20996-blue.svg)](https://github.com/996icu/996.ICU/blob/master/LICENSE) [![996.icu](https://img.shields.io/badge/link-996.icu-red.svg)](https://996.icu)


Everything about Transfer Learning (Probably the **most complete** repository?). *Your contribution is highly valued!* If you find this repo helpful, please cite it as follows:

关于迁移学习的所有资料，包括：介绍、综述文章、最新文章、代表工作及其代码、常用数据集、硕博士论文、比赛等等。(可能是**目前最全**的迁移学习资料库？) *欢迎一起贡献！* 如果认为本仓库有用，请在你的论文和其他出版物中进行引用！ 

```
@Misc{transferlearning.xyz,
howpublished = {\url{http://transferlearning.xyz}},   
title = {Everything about Transfer Learning and Domain Adapation},  
author = {Wang, Jindong and others}  
}  
```

- [迁移学习 Transfer Learning](#%e8%bf%81%e7%a7%bb%e5%ad%a6%e4%b9%a0-transfer-learning)
	- [0.Latest Publications (最新论文)](#0latest-publications-%e6%9c%80%e6%96%b0%e8%ae%ba%e6%96%87)
	- [1.Introduction and Tutorials (简介与教程)](#1introduction-and-tutorials-%e7%ae%80%e4%bb%8b%e4%b8%8e%e6%95%99%e7%a8%8b)
	- [2.Transfer Learning Areas and Papers (研究领域与相关论文)](#2transfer-learning-areas-and-papers-%e7%a0%94%e7%a9%b6%e9%a2%86%e5%9f%9f%e4%b8%8e%e7%9b%b8%e5%85%b3%e8%ae%ba%e6%96%87)
	- [3.Theory and Survey (理论与综述)](#3theory-and-survey-%e7%90%86%e8%ae%ba%e4%b8%8e%e7%bb%bc%e8%bf%b0)
	- [4.Code (代码)](#4code-%e4%bb%a3%e7%a0%81)
	- [5.Transfer Learning Scholars (著名学者)](#5transfer-learning-scholars-%e8%91%97%e5%90%8d%e5%ad%a6%e8%80%85)
	- [6.Transfer Learning Thesis (硕博士论文)](#6transfer-learning-thesis-%e7%a1%95%e5%8d%9a%e5%a3%ab%e8%ae%ba%e6%96%87)
	- [7.Datasets and Benchmarks (数据集与评测结果)](#7datasets-and-benchmarks-%e6%95%b0%e6%8d%ae%e9%9b%86%e4%b8%8e%e8%af%84%e6%b5%8b%e7%bb%93%e6%9e%9c)
	- [8.Transfer Learning Challenges (迁移学习比赛)](#8transfer-learning-challenges-%e8%bf%81%e7%a7%bb%e5%ad%a6%e4%b9%a0%e6%af%94%e8%b5%9b)
	- [Applications (迁移学习应用)](#applications-%e8%bf%81%e7%a7%bb%e5%ad%a6%e4%b9%a0%e5%ba%94%e7%94%a8)
	- [Other Resources (其他资源)](#other-resources-%e5%85%b6%e4%bb%96%e8%b5%84%e6%ba%90)
	- [Contributing (欢迎参与贡献)](#contributing-%e6%ac%a2%e8%bf%8e%e5%8f%82%e4%b8%8e%e8%b4%a1%e7%8c%ae)


> 关于机器学习和行为识别的资料，请参考：[行为识别](https://github.com/jindongwang/activityrecognition)｜[机器学习](https://github.com/jindongwang/MachineLearning)

- - -

## 0.Latest Publications (最新论文)

**A good website to see the latest arXiv preprints by search: [Transfer learning](http://arxitics.com/search?q=transfer%20learning&sort=updated#1904.01376/abstract), [Domain adaptation](http://arxitics.com/search?q=domain%20adaptation&sort=updated)**

**一个很好的网站，可以直接看到最新的arXiv文章: [Transfer learning](http://arxitics.com/search?q=transfer%20learning&sort=updated#1904.01376/abstract), [Domain adaptation](http://arxitics.com/search?q=domain%20adaptation&sort=updated)**

[迁移学习文章汇总 Awesome transfer learning papers](https://github.com/jindongwang/transferlearning/tree/master/doc/awesome_paper.md)

- **Weekly latest papers**

	- 20200210 WACVW-20 [Impact of ImageNet Model Selection on Domain Adaptation](https://arxiv.org/abs/2002.02559)
    	- A good experiment paper to indicate the power of representations
    	- 一篇很好的实验paper，揭示了深度特征+传统方法的有效性
  	- 20200210 AAAI-20 [Bi-Directional Generation for Unsupervised Domain Adaptation](https://arxiv.org/abs/2002.04869)
    	- Bidirectional GANs for domain adaptation
    	- 双向的GAN用来做DA
    

[**更多 More...**](https://github.com/jindongwang/transferlearning/tree/master/doc/awesome_paper.md)

- - -

## 1.Introduction and Tutorials (简介与教程)

- 简介文字资料
	- [简单的中文简介 Chinese introduction](https://github.com/jindongwang/transferlearning/blob/master/doc/%E8%BF%81%E7%A7%BB%E5%AD%A6%E4%B9%A0%E7%AE%80%E4%BB%8B.md)
	- [PPT(English)](http://jd92.wang/assets/files/l03_transferlearning.pdf)
	- [PPT(中文)](http://jd92.wang/assets/files/l08_tl_zh.pdf)
	- 迁移学习中的领域自适应方法 Domain adaptation: [PDF](http://jd92.wang/assets/files/l12_da.pdf) ｜ [Video](http://mp.weixin.qq.com/s?__biz=MzI5MDUyMDIxNA==&mid=2247484940&idx=2&sn=35e64e07fde9a96afbb65dbf40a945eb&chksm=ec1febf5db6862e38d5e02ff3278c61b376932a46c5628c7d9cb1769c572bfd31819c13dd468&mpshare=1&scene=1&srcid=1219JpTNZFiNDCHsTUrUxwqy#rd)
	- 清华大学龙明盛老师的深度迁移学习报告 Transfer learning report by Mingsheng Long @ THU：[PPT(Samsung)](http://ise.thss.tsinghua.edu.cn/~mlong/doc/transfer-learning-talk.pdf)、[PPT(Google China)](http://ise.thss.tsinghua.edu.cn/~mlong/doc/deep-transfer-learning-talk.pdf)

- 入门教程
	- [**《迁移学习简明手册》Transfer Learning Tutorial**](https://zhuanlan.zhihu.com/p/35352154) [在线阅读](https://tutorial.transferlearning.xyz/) [PDF](http://jd92.wang/assets/files/transfer_learning_tutorial_wjd.pdf)

- 视频教程
	- [台湾大学李宏毅的视频讲解(中文视频)](https://www.youtube.com/watch?v=qD6iD4TFsdQ)
	- [迁移学习中的领域自适应方法(中文)](http://mp.weixin.qq.com/s?__biz=MzI5MDUyMDIxNA==&mid=2247484940&idx=2&sn=35e64e07fde9a96afbb65dbf40a945eb&chksm=ec1febf5db6862e38d5e02ff3278c61b376932a46c5628c7d9cb1769c572bfd31819c13dd468&mpshare=1&scene=1&srcid=1219JpTNZFiNDCHsTUrUxwqy#rd)
	- [2020斯坦福大学多任务与元学习教程CS330 by Chelsea finn](https://www.bilibili.com/video/av91772677?p=12)

- [迁移学习领域的著名学者、代表工作及实验室介绍 Transfer Learning Scholars and Labs](https://github.com/jindongwang/transferlearning/blob/master/doc/scholar_TL.md)

- 什么是[负迁移(negative transfer)](https://www.zhihu.com/question/66492194/answer/242870418)？

- 动手教程、代码、数据 Hands-on Codes
    - [Pytorch官方迁移学习示意代码](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
	- [Pytorch的finetune Fine-tune based on Alexnet and Resnet](https://github.com/jindongwang/transferlearning/tree/master/code/AlexNet_ResNet)
	- [用Pytorch进行深度特征提取](https://github.com/jindongwang/transferlearning/tree/master/code/feature_extractor)
	- [基于Tensorflow的几个深度adaptation实现](https://github.com/asahi417/DeepDomainAdaptation)
	- [更多 More...](https://github.com/jindongwang/transferlearning/tree/master/code)

- - -

## 2.Transfer Learning Areas and Papers (研究领域与相关论文)

Related articles by research areas:

- [General Transfer Learning (普通迁移学习)](https://github.com/jindongwang/transferlearning/blob/master/doc/awesome_paper.md#general-transfer-learning-%E6%99%AE%E9%80%9A%E8%BF%81%E7%A7%BB%E5%AD%A6%E4%B9%A0)
  - [Theory (理论)](https://github.com/jindongwang/transferlearning/blob/master/doc/awesome_paper.md#theory-%E7%90%86%E8%AE%BA)
  - [Others (其他)](https://github.com/jindongwang/transferlearning/blob/master/doc/awesome_paper.md#others-%E5%85%B6%E4%BB%96)
- [Domain Adaptation (领域自适应)](https://github.com/jindongwang/transferlearning/blob/master/doc/awesome_paper.md#domain-adaptation-%E9%A2%86%E5%9F%9F%E8%87%AA%E9%80%82%E5%BA%94)
  - [Traditional Methods (传统迁移方法)](https://github.com/jindongwang/transferlearning/blob/master/doc/awesome_paper.md#traditional-methods-%E4%BC%A0%E7%BB%9F%E8%BF%81%E7%A7%BB%E6%96%B9%E6%B3%95)
  - [Deep / Adversarial Methods (深度/对抗迁移方法)](https://github.com/jindongwang/transferlearning/blob/master/doc/awesome_paper.md#deep--adversarial-methods-%E6%B7%B1%E5%BA%A6%E5%AF%B9%E6%8A%97%E8%BF%81%E7%A7%BB%E6%96%B9%E6%B3%95)
- [Domain Generalization](https://github.com/jindongwang/transferlearning/blob/master/doc/awesome_paper.md#domain-generalization)
- [Multi-source Transfer Learning (多源迁移学习)](https://github.com/jindongwang/transferlearning/blob/master/doc/awesome_paper.md#multi-source-transfer-learning-%E5%A4%9A%E6%BA%90%E8%BF%81%E7%A7%BB%E5%AD%A6%E4%B9%A0)
- [Heterogeneous Transfer Learning (异构迁移学习)](https://github.com/jindongwang/transferlearning/blob/master/doc/awesome_paper.md#heterogeneous-transfer-learning-%E5%BC%82%E6%9E%84%E8%BF%81%E7%A7%BB%E5%AD%A6%E4%B9%A0)
- [Online Transfer Learning (在线迁移学习)](https://github.com/jindongwang/transferlearning/blob/master/doc/awesome_paper.md#online-transfer-learning-%E5%9C%A8%E7%BA%BF%E8%BF%81%E7%A7%BB%E5%AD%A6%E4%B9%A0)
- [Zero-shot / Few-shot Learning](https://github.com/jindongwang/transferlearning/blob/master/doc/awesome_paper.md#zero-shot--few-shot-learning)
- [Deep Transfer Learning (深度迁移学习)](https://github.com/jindongwang/transferlearning/blob/master/doc/awesome_paper.md#deep-transfer-learning-%E6%B7%B1%E5%BA%A6%E8%BF%81%E7%A7%BB%E5%AD%A6%E4%B9%A0)
  - [Non-Adversarial Transfer Learning (非对抗深度迁移)](https://github.com/jindongwang/transferlearning/blob/master/doc/awesome_paper.md#non-adversarial-transfer-learning-%E9%9D%9E%E5%AF%B9%E6%8A%97%E6%B7%B1%E5%BA%A6%E8%BF%81%E7%A7%BB)
  - [Deep Adversarial Transfer Learning (对抗迁移学习)](https://github.com/jindongwang/transferlearning/blob/master/doc/awesome_paper.md#deep-adversarial-transfer-learning-%E5%AF%B9%E6%8A%97%E8%BF%81%E7%A7%BB%E5%AD%A6%E4%B9%A0)
- [Multi-task Learning (多任务学习)](https://github.com/jindongwang/transferlearning/blob/master/doc/awesome_paper.md#multi-task-learning-%E5%A4%9A%E4%BB%BB%E5%8A%A1%E5%AD%A6%E4%B9%A0)
- [Transfer Reinforcement Learning (强化迁移学习)](https://github.com/jindongwang/transferlearning/blob/master/doc/awesome_paper.md#transfer-reinforcement-learning-%E5%BC%BA%E5%8C%96%E8%BF%81%E7%A7%BB%E5%AD%A6%E4%B9%A0)
- [Transfer Metric Learning (迁移度量学习)](https://github.com/jindongwang/transferlearning/blob/master/doc/awesome_paper.md#transfer-metric-learning-%E8%BF%81%E7%A7%BB%E5%BA%A6%E9%87%8F%E5%AD%A6%E4%B9%A0)
- [Transitive Transfer Learning (传递迁移学习)](https://github.com/jindongwang/transferlearning/blob/master/doc/awesome_paper.md#transitive-transfer-learning-%E4%BC%A0%E9%80%92%E8%BF%81%E7%A7%BB%E5%AD%A6%E4%B9%A0)
- [Lifelong Learning (终身迁移学习)](https://github.com/jindongwang/transferlearning/blob/master/doc/awesome_paper.md#lifelong-learning-%E7%BB%88%E8%BA%AB%E8%BF%81%E7%A7%BB%E5%AD%A6%E4%B9%A0)
- [Negative Transfer (负迁移)](https://github.com/jindongwang/transferlearning/blob/master/doc/awesome_paper.md#negative-transfer-%E8%B4%9F%E8%BF%81%E7%A7%BB)
- [Transfer Learning Applications (应用)](https://github.com/jindongwang/transferlearning/blob/master/doc/transfer_learning_application.md)

[Paperweekly](http://www.paperweekly.site/collections/231/papers): 一个推荐、分享论文的网站比较好，上面会持续整理相关的文章并分享阅读笔记。

- - -

## 3.Theory and Survey (理论与综述)

Here are some articles on transfer learning theory and survey.

- 迁移学习领域最具代表性的综述是[A survey on transfer learning](http://ieeexplore.ieee.org/abstract/document/5288526/)，发表于2010年，对迁移学习进行了比较权威的定义。 -- The most influential survey on transfer learning.

- 迁移学习的**理论分析** Transfer Learning Theory：

	- 迁移学习方面一直以来都比较缺乏理论分析与证明的文章，以下几篇连贯式的理论文章成为了经典：
		- NIPS-06 [Analysis of Representations for Domain Adaptation](https://dl.acm.org/citation.cfm?id=2976474)
		- ML-10 [A Theory of Learning from Different Domains](https://link.springer.com/article/10.1007/s10994-009-5152-4)
		- NIPS-08 [Learning Bounds for Domain Adaptation](http://papers.nips.cc/paper/3212-learning-bounds-for-domain-adaptation)
		- COLT-09 [Domain adaptation: Learning bounds and algorithms](https://arxiv.org/abs/0902.3430)

	- 最近几年在ICML、NIPS、COLT、ALT上出现了一些理论分析的文章，以domain adaptation为关键字可以搜索到。绝大多数都是对上述的扩展和补充。

	- 许多研究者在迁移学习的研究中会应用MMD(Maximum Mean Discrepancy)这个最大均值差异来衡量不同domain之间的距离。MMD的理论文章是：
		- MMD的提出：[A Hilbert Space Embedding for Distributions](https://link.springer.com/chapter/10.1007/978-3-540-75225-7_5) 以及 [A Kernel Two-Sample Test](http://www.jmlr.org/papers/v13/gretton12a.html)
		- 多核MMD(MK-MMD)：[Optimal kernel choice for large-scale two-sample tests](http://papers.nips.cc/paper/4727-optimal-kernel-choice-for-large-scale-two-sample-tests)
		- MMD及多核MMD代码：[Matlab](https://github.com/lopezpaz/classifier_tests/tree/master/code/unit_test_mmd) | [Python](https://github.com/jindongwang/transferlearning/tree/master/code/basic/mmd.py)
	- 理论研究方面，重点关注Alex Smola、Ben-David、Bernhard Schölkopf、Arthur Gretton等人的研究。

- 较新的综述 Latest survey：

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

- 迁移学习的应用
	- 视觉domain adaptation综述：[Visual Domain Adaptation: A Survey of Recent Advances](https://pan.baidu.com/s/1o8BR7Vc)
	- 迁移学习应用于行为识别综述：[Transfer Learning for Activity Recognition: A Survey](https://pan.baidu.com/s/1kVABOYr)
	- 迁移学习与增强学习：[Transfer Learning for Reinforcement Learning Domains: A Survey](https://pan.baidu.com/s/1slfr0w1)
	- 多个源域进行迁移的综述：[A Survey of Multi-source Domain Adaptation](https://pan.baidu.com/s/1eSGREF4)。

_ _ _

## 4.Code (代码)

请见[这里](https://github.com/jindongwang/transferlearning/tree/master/code) | Please see [HERE](https://github.com/jindongwang/transferlearning/tree/master/code) for some popular transfer learning codes.

_ _ _

## 5.Transfer Learning Scholars (著名学者)

Here are some transfer learning scholars and labs.

**全部列表以及代表工作性见[这里](https://github.com/jindongwang/transferlearning/blob/master/doc/scholar_TL.md)**

Please refer to [here](https://github.com/jindongwang/transferlearning/blob/master/doc/scholar_TL.md) to see a complete list.

- [Qiang Yang](http://www.cs.ust.hk/~qyang/)：中文名杨强。香港科技大学计算机系讲座教授，迁移学习领域世界性专家。IEEE/ACM/AAAI/IAPR/AAAS fellow。[[Google scholar](https://scholar.google.com/citations?user=1LxWZLQAAAAJ&hl=zh-CN)]

- [Sinno Jialin Pan](http://www.ntu.edu.sg/home/sinnopan/)：杨强的学生，香港科技大学博士，现任新加坡南洋理工大学助理教授。迁移学习领域代表性综述A survey on transfer learning的第一作者（Qiang Yang是二作）。[[Google scholar](https://scholar.google.com/citations?user=P6WcnfkAAAAJ&hl=zh-CN)]

- [Wenyuan Dai](https://scholar.google.com.sg/citations?user=AGR9pP0AAAAJ&hl=zh-CN)：中文名戴文渊，上海交通大学硕士，现任第四范式人工智能创业公司CEO。迁移学习领域著名的牛人，在顶级会议上发表多篇高水平文章，每篇论文引用量巨大。[[Google scholar](https://scholar.google.com.hk/citations?hl=zh-CN&user=AGR9pP0AAAAJ)]

- [Lixin Duan](http://www.lxduan.info/)：中文名段立新，新加坡南洋理工大学博士，现就职于电子科技大学，教授。[[Google scholar](https://scholar.google.com.hk/citations?user=inRIcS0AAAAJ&hl=zh-CN&oi=ao)]

- [Boqing Gong](http://boqinggong.info/index.html)：南加州大学博士，现就职于腾讯AI Lab(西雅图)。曾任中佛罗里达大学助理教授。[[Google scholar](https://scholar.google.com/citations?user=lv9ZeVUAAAAJ&hl=en)]

- [Fuzhen Zhuang](http://www.intsci.ac.cn/users/zhuangfuzhen/)：中文名庄福振，中科院计算所博士，现任中科院计算所副研究员。[[Google scholar](https://scholar.google.com/citations?user=klJBYrAAAAAJ&hl=zh-CN&oi=ao)]

- [Mingsheng Long](http://ise.thss.tsinghua.edu.cn/~mlong/)：中文名龙明盛，清华大学博士，现任清华大学助理教授、博士生导师。[[Google scholar](https://scholar.google.com/citations?view_op=search_authors&mauthors=mingsheng+long&hl=zh-CN&oi=ao)]

- [Qingyao Wu](https://sites.google.com/site/qysite/)：中文名吴庆耀，现任华南理工大学副教授。主要做在线迁移学习、异构迁移学习方面的研究。[[Google scholar](https://scholar.google.com.hk/citations?user=n6e_2IgAAAAJ&hl=zh-CN&oi=ao)]

- [Weike Pan](https://sites.google.com/site/weikep/)：中文名潘微科，杨强的学生，现任深圳大学副教授，香港科技大学博士毕业。主要做迁移学习在推荐系统方面的一些工作。 [[Google Scholar](https://scholar.google.com/citations?user=pC5Q26MAAAAJ&hl=en)]

- [Tongliang Liu](http://ieeexplore.ieee.org/abstract/document/8259375/)：中文名刘同亮，现任悉尼大学助理教授。主要做迁移学习的一些理论方面的工作。[[Google scholar](https://scholar.google.com.hk/citations?hl=zh-CN&user=EiLdZ_YAAAAJ)]

- [Tatiana Tommasi](http://tatianatommasi.wixsite.com/tatianatommasi/3)：Researcher at the Italian Institute of Technology.

- [Vinod K Kurmi](https://github.com/vinodkkurmi)[[home page](https://github.com/vinodkkurmi)]: Researcher at the Indian Institute of Technology Kanpur(India)
_ _ _

## 6.Transfer Learning Thesis (硕博士论文)

Here are some popular thesis on transfer learning.

硕博士论文可以让我们很快地对迁移学习的相关领域做一些了解，同时，也能很快地了解概括相关研究者的工作。其中，比较有名的有

- 2016 Baochen Sun的[Correlation Alignment for Domain Adaptation](http://www.cs.uml.edu/~bsun/papers/baochen_phd_thesis.pdf)

- 2015 南加州大学的Boqing Gong的[Kernel Methods for Unsupervised Domain Adaptation](https://pan.baidu.com/s/1bpbawv9)

- 2014 清华大学龙明盛的[迁移学习问题与方法研究](http://ise.thss.tsinghua.edu.cn/~mlong/doc/phd-thesis-mingsheng-long.pdf)

- 2014 中科院计算所赵中堂的[自适应行为识别中的迁移学习方法研究](https://pan.baidu.com/s/1kVqYXnh)

- 2012 杨强的学生Hao Hu的[Learning based Activity Recognition](https://pan.baidu.com/s/1bp2K9HX)

- 2012 杨强的学生Wencheng Zheng的[Learning with Limited Data in Sensor-based Human Behavior Prediction](https://pan.baidu.com/s/1o8MbbBk)

- 2010 杨强的学生Sinno Jialin Pan的[Feature-based Transfer Learning and Its Applications](https://pan.baidu.com/s/1bUqMfW)

- 2009 上海交通大学戴文渊的[基于实例和特征的迁移学习算法研究](https://pan.baidu.com/s/1i4Vyygd)

其他的文章，请见[完整版](https://pan.baidu.com/s/1bqXEASn)。

- - -

## 7.Datasets and Benchmarks (数据集与评测结果)

Please see [HERE](https://github.com/jindongwang/transferlearning/blob/master/data) for the popular transfer learning **datasets and certain benchmark** results.

[这里](https://github.com/jindongwang/transferlearning/blob/master/data)整理了常用的公开数据集和一些已发表的文章在这些数据集上的实验结果。

- - -

## 8.Transfer Learning Challenges (迁移学习比赛)

一些关于迁移学习的国际比赛。

- [Visual Domain Adaptation Challenge (VisDA)](http://ai.bu.edu/visda-2018/)

- - -

## Applications (迁移学习应用)

See [HERE](https://github.com/jindongwang/transferlearning/blob/master/doc/transfer_learning_application.md) for transfer learning applications.

迁移学习应用请见[这里](https://github.com/jindongwang/transferlearning/blob/master/doc/transfer_learning_application.md)。

- - -
  
## Other Resources (其他资源)

Call for papers about transfer learning:

- [Transfer Learning for Multimedia Applications(A Special Issue on Multimedia Tools and Applications (MTAP))](https://lijin118.github.io/mtap/)

Related projects:

- Salad: [A semi-supervised domain adaptation library](https://domainadaptation.org)


- - -

## Contributing (欢迎参与贡献)

If you are interested in contributing, please refer to [HERE](https://github.com/jindongwang/transferlearning/blob/master/CONTRIBUTING.md) for instructions in contribution.

> ***[文章版权声明]这个仓库是我开源到Github上的，可以遵守相关的开源协议进行使用。这个仓库中包含有很多研究者的论文、硕博士论文等，都来源于在网上的下载，仅作为学术研究使用。我对其中一些文章都写了自己的浅见，希望能很好地帮助理解。这些文章的版权属于相应的出版社。如果作者或出版社有异议，请联系我进行删除。一切都是为了更好地学术！***
