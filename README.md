# 迁移学习 Transfer Learning

关于迁移学习的所有资料，包括：介绍、综述文章、最新文章、代表工作及其代码、常用数据集、硕博士论文等等。(可能是**目前最全**的迁移学习资料库？) *欢迎一起贡献！*

Everything about Transfer Learning (Probably the **most complete** repository?). *Your contribution is highly valued!*

_ _ _

## 目录 Table of contents

* [最新文章 Latest](#0latest)

* [迁移学习简介 Introduction to transfer learning](#1迁移学习简介)

* [研究领域与相关文章 Research articles by area](#2迁移学习研究领域与相关文章) 

* [理论与综述文章 Theoretical and survey papers](#3理论与综述文章)

* [相关代码 Available codes](#4代码)

* [迁移学习代表性研究学者 Scholars](#5迁移学习代表性研究学者)

* [相关的硕博士论文 Thesis](#6迁移学习相关的硕博士论文)

* [常用公开数据集及算法结果 Datasets and benchmark](#7公开数据集及实验结果)

* [其他 Miscellaneous](#其他)

* [Contributing](#contributing)

> 关于机器学习和行为识别的资料，请参考：[行为识别](https://github.com/jindongwang/activityrecognition)｜[机器学习](https://github.com/jindongwang/MachineLearning)

- - -

### 0.Latest

[迁移学习文章汇总 Awesome transfer learning papers](https://github.com/jindongwang/transferlearning/tree/master/doc/awesome_paper.md)


- **最新发表论文**

	- 20180530 ICML-18 强化迁移学习：[Importance Weighted Transfer of Samples in Reinforcement Learning](https://arxiv.org/abs/1805.10886)

	- 20180530 MNRAS 用迁移学习检测银河星系兼并：[Using transfer learning to detect galaxy mergers](https://arxiv.org/abs/1805.10289)

	- 20180524 KDD-18 用迁移学习方法进行人们的ID迁移：[Learning and Transferring IDs Representation in E-commerce](https://arxiv.org/abs/1712.08289)

	- 20180516 ACL-18 将对抗迁移学习用于危机状态下的舆情分析：[Domain Adaptation with Adversarial Training and Graph Embeddings](https://arxiv.org/abs/1805.05151)

	- 20180510 IEEE Trans. Cybernetics 提出一个通用的迁移学习框架，对不同的domain进行不同的特征变换：[Transfer Independently Together: A Generalized Framework for Domain Adaptation](https://ieeexplore-ieee-org.lib.ezproxy.ust.hk/abstract/document/8337102/)

	- 20180428 CVPR-18 相互协同学习：[Deep Mutual Learning](https://github.com/YingZhangDUT/Deep-Mutual-Learning)

- **arXiv专区** (arXiv上的文章较新，但未经过peer-review，慎重看待)

	- 20180530 arXiv 用于深度网络的鲁棒性domain adaptation方法：[Robust Unsupervised Domain Adaptation for Neural Networks via Moment Alignment](https://arxiv.org/abs/1711.06114)

	- 20180529 arXiv 迁移学习用于表情识别：[Meta Transfer Learning for Facial Emotion Recognition](https://arxiv.org/abs/1805.09946)

	- 20180524 arXiv 探索了Multi-source迁移学习的一些理论：[Algorithms and Theory for Multiple-Source Adaptation](https://arxiv.org/abs/1805.08727)

	- 20180524 arXiv 用深度强化学习的方法学习domain adaptation中的采样策略：[Learning Sampling Policies for Domain Adaptation](https://arxiv.org/abs/1805.07641)

	- 20180524 arXiv 杨强团队、与之前的learning to learning类似，这里提供了一个从经验中学习的learning to multitask框架：[Learning to Multitask](https://arxiv.org/abs/1805.07541)

	- 20180522 arXiv 用CNN进行跨领域的属性学习：[Cross-domain attribute representation based on convolutional neural network](https://arxiv.org/abs/1805.07295)

	- 20180519 arXiv 用迁移学习进行物体检测，200帧/秒：[Object detection at 200 Frames Per Second](https://arxiv.org/abs/1804.04775)

	- 20180519 arXiv 用迁移学习进行肢体语言识别：[Optimization of Transfer Learning for Sign Language Recognition Targeting Mobile Platform](https://arxiv.org/abs/1805.06618)


[**更多...**](https://github.com/jindongwang/transferlearning/tree/master/doc/awesome_paper.md)

- - -

### 1.迁移学习简介

- 简介文字资料
	- [简单的中文简介](https://github.com/jindongwang/transferlearning/blob/master/doc/%E8%BF%81%E7%A7%BB%E5%AD%A6%E4%B9%A0%E7%AE%80%E4%BB%8B.md)
	- [英文PPT](http://jd92.wang/assets/files/l03_transferlearning.pdf)
	- [中文PPT](http://jd92.wang/assets/files/l08_tl_zh.pdf)
	- 迁移学习中的领域自适应方法: [PDF](http://jd92.wang/assets/files/l12_da.pdf) ｜ [Video](http://mp.weixin.qq.com/s?__biz=MzI5MDUyMDIxNA==&mid=2247484940&idx=2&sn=35e64e07fde9a96afbb65dbf40a945eb&chksm=ec1febf5db6862e38d5e02ff3278c61b376932a46c5628c7d9cb1769c572bfd31819c13dd468&mpshare=1&scene=1&srcid=1219JpTNZFiNDCHsTUrUxwqy#rd)
	- 清华大学龙明盛老师的深度迁移学习报告：[PPT(三星研究院)](http://ise.thss.tsinghua.edu.cn/~mlong/doc/transfer-learning-talk.pdf)、[PPT(Google China)](http://ise.thss.tsinghua.edu.cn/~mlong/doc/deep-transfer-learning-talk.pdf)

- 视频教程
	- [台湾大学李宏毅的视频讲解(中文视频)](https://www.youtube.com/watch?v=qD6iD4TFsdQ)
	- [迁移学习中的领域自适应方法(中文)](http://mp.weixin.qq.com/s?__biz=MzI5MDUyMDIxNA==&mid=2247484940&idx=2&sn=35e64e07fde9a96afbb65dbf40a945eb&chksm=ec1febf5db6862e38d5e02ff3278c61b376932a46c5628c7d9cb1769c572bfd31819c13dd468&mpshare=1&scene=1&srcid=1219JpTNZFiNDCHsTUrUxwqy#rd)

- [**《迁移学习简明手册》**](https://zhuanlan.zhihu.com/p/35352154) [开发维护地址](https://github.com/jindongwang/transferlearning-tutorial)

- [迁移学习领域的著名学者、代表工作及实验室介绍](https://github.com/jindongwang/transferlearning/blob/master/doc/scholar_TL.md)

- 什么是[负迁移(negative transfer)](https://www.zhihu.com/question/66492194/answer/242870418)？

- 动手教程、代码、数据
	- [基于深度学习和迁移学习的识花实践(Tensorflow)](https://cosx.org/2017/10/transfer-learning/)
	- [基于Pytorch的图像分类](https://github.com/miguelgfierro/sciblog_support/blob/master/A_Gentle_Introduction_to_Transfer_Learning/Intro_Transfer_Learning.ipynb)
	- [使用Pytorch进行finetune](https://github.com/Spandan-Madan/Pytorch_fine_tuning_Tutorial)

- - -

### 2.迁移学习研究领域与相关文章

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

### 3.理论与综述文章

- 迁移学习领域最具代表性的综述是[A survey on transfer learning](http://ieeexplore.ieee.org/abstract/document/5288526/)，发表于2010年，对迁移学习进行了比较权威的定义。

- 迁移学习的**理论分析**：

	- 迁移学习方面一直以来都比较缺乏理论分析与证明的文章，以下三篇连贯式的理论文章成为了经典：
		- NIPS-06 [Analysis of Representations for Domain Adaptation](https://dl.acm.org/citation.cfm?id=2976474)
		- ML-10 [A Theory of Learning from Different Domains](https://link.springer.com/article/10.1007/s10994-009-5152-4)
		- NIPS-08 [Learning Bounds for Domain Adaptation](http://papers.nips.cc/paper/3212-learning-bounds-for-domain-adaptation)

	- 许多研究者在迁移学习的研究中会应用MMD(Maximum Mean Discrepancy)这个最大均值差异来衡量不同domain之间的距离。MMD的理论文章是：
		- MMD的提出：[A Hilbert Space Embedding for Distributions](https://link.springer.com/chapter/10.1007/978-3-540-75225-7_5) 以及 [A Kernel Two-Sample Test](http://www.jmlr.org/papers/v13/gretton12a.html)
		- 多核MMD(MK-MMD)：[Optimal kernel choice for large-scale two-sample tests](http://papers.nips.cc/paper/4727-optimal-kernel-choice-for-large-scale-two-sample-tests)
		- MMD及多核MMD代码：[Matlab](https://github.com/lopezpaz/classifier_tests/tree/master/code/unit_test_mmd) | [Python](https://github.com/jindongwang/transferlearning/tree/master/code/basic/mmd.py)
	- 理论研究方面，重点关注Alex Smola、Ben-David、Bernhard Schölkopf、Arthur Gretton等人的研究即可。

- 较新的综述：
	- 20180419 arXiv Neural style transfer的一个survey：[Neural Style Transfer: A Review](https://arxiv.org/abs/1705.04058)
	- 2018 arXiv 最新发表在arXiv上的深度domain adaptation的综述：[Deep Visual Domain Adaptation: A Survey](https://arxiv.org/abs/1802.03601)
	- 2017 多任务学习的综述，来自香港科技大学杨强团队：[A survey on multi-task learning](https://arxiv.org/abs/1707.08114)
	- 2017 异构迁移学习的综述：[A survey on heterogeneous transfer learning](https://link.springer.com/article/10.1186/s40537-017-0089-0)
	- 2017 跨领域数据识别的综述：[Cross-dataset recognition: a survey](https://arxiv.org/abs/1705.04396)
	- 2016 [A survey of transfer learning](https://pan.baidu.com/s/1gfgXLXT)。其中交代了一些比较经典的如同构、异构等学习方法代表性文章。包括了很多方法介绍，值得一看。
	- 2015 中文综述：[迁移学习研究进展](https://pan.baidu.com/s/1bpautob)

- 迁移学习与应用领域结合
	- 视觉domain adaptation综述：[Visual Domain Adaptation: A Survey of Recent Advances](https://pan.baidu.com/s/1o8BR7Vc)
	- 迁移学习应用于行为识别综述：[Transfer Learning for Activity Recognition: A Survey](https://pan.baidu.com/s/1kVABOYr)
	- 迁移学习与增强学习：[Transfer Learning for Reinforcement Learning Domains: A Survey](https://pan.baidu.com/s/1slfr0w1)
	- 多个源域进行迁移的综述：[A Survey of Multi-source Domain Adaptation](https://pan.baidu.com/s/1eSGREF4)。

_ _ _

### 4.代码

请见[这里](https://github.com/jindongwang/transferlearning/tree/master/code) | Please see [HERE](https://github.com/jindongwang/transferlearning/tree/master/code) for some popular transfer learning codes.

_ _ _

### 5.迁移学习代表性研究学者

**全部列表以及代表工作性见[这里](https://github.com/jindongwang/transferlearning/blob/master/doc/scholar_TL.md)**

- [Qiang Yang](http://www.cs.ust.hk/~qyang/)：中文名杨强。香港科技大学计算机系讲座教授，大数据中心主任。迁移学习领域世界性专家。IEEE/AAAI/IAPR/AAAS fellow。[[Google scholar](https://scholar.google.com/citations?user=1LxWZLQAAAAJ&hl=zh-CN)]

- [Sinno Jialin Pan](http://www.ntu.edu.sg/home/sinnopan/)：杨强的学生，香港科技大学博士，现任新加坡南洋理工大学助理教授。迁移学习领域代表性综述A survey on transfer learning的第一作者（Qiang Yang是二作）。[[Google scholar](https://scholar.google.com/citations?user=P6WcnfkAAAAJ&hl=zh-CN)]

- [Wenyuan Dai](https://scholar.google.com.sg/citations?user=AGR9pP0AAAAJ&hl=zh-CN)：中文名戴文渊，上海交通大学硕士，现任第四范式人工智能创业公司CEO。迁移学习领域著名的牛人，在顶级会议上发表多篇高水平文章，每篇论文引用量巨大。[[Google scholar](https://scholar.google.com.hk/citations?hl=zh-CN&user=AGR9pP0AAAAJ)]

- [Lixin Duan](http://www.lxduan.info/)：中文名段立新，新加坡南洋理工大学博士，现就职于电子科技大学，教授。[[Google scholar](https://scholar.google.com.hk/citations?user=inRIcS0AAAAJ&hl=zh-CN&oi=ao)]

- [Fuzhen Zhuang](http://www.intsci.ac.cn/users/zhuangfuzhen/)：中文名庄福振，中科院计算所博士，现任中科院计算所副研究员。[[Google scholar](https://scholar.google.com/citations?user=klJBYrAAAAAJ&hl=zh-CN&oi=ao)]

- [Mingsheng Long](http://ise.thss.tsinghua.edu.cn/~mlong/)：中文名龙明盛，清华大学博士，现任清华大学助理教授、博士生导师。[[Google scholar](https://scholar.google.com/citations?view_op=search_authors&mauthors=mingsheng+long&hl=zh-CN&oi=ao)]

- [Qingyao Wu](https://sites.google.com/site/qysite/)：中文名吴庆耀，现任华南理工大学副教授。主要做在线迁移学习、异构迁移学习方面的研究。[[Google scholar](https://scholar.google.com.hk/citations?user=n6e_2IgAAAAJ&hl=zh-CN&oi=ao)]

- [Weike Pan](https://sites.google.com/site/weikep/)：中文名潘微科，杨强的学生，现任深圳大学副教授，香港科技大学博士毕业。主要做迁移学习在推荐系统方面的一些工作。 [[Google Scholar](https://scholar.google.com/citations?user=pC5Q26MAAAAJ&hl=en)]

- [Tongliang Liu](http://ieeexplore.ieee.org/abstract/document/8259375/)：中文名刘同亮，现任悉尼大学助理教授。主要做迁移学习的一些理论方面的工作。[[Google scholar](https://scholar.google.com.hk/citations?hl=zh-CN&user=EiLdZ_YAAAAJ)]
_ _ _

### 6.迁移学习相关的硕博士论文

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

### 7.公开数据集及实验结果

迁移学习领域有一些公开的数据集，用来对比算法的表现。

- [这里](https://github.com/jindongwang/transferlearning/blob/master/doc/dataset.md)整理了常用的公开数据集
- [这里](https://github.com/jindongwang/transferlearning/blob/master/doc/benchmark.md)整理汇总了一些已发表的文章在这些数据集上的实验结果。

Please see [HERE](https://github.com/jindongwang/transferlearning/blob/master/doc/dataset.md) for the popular transfer learning datasets and [HERE](https://github.com/jindongwang/transferlearning/blob/master/doc/benchmark.md) for some benchmark results.

- - -

### 其他

[记与迁移学习大牛杨强教授的第二次会面](https://zhuanlan.zhihu.com/p/26260083)

- - -

### Contributing

如果你对本项目感兴趣，非常欢迎你加入！

- 正常参与：请直接fork、pull request，或者与我联系我把你加入合作者中
- 如果要上传文件：请**不要**直接上传到项目中，否则会造成git版本库过大。正确的方法是上传它的**超链接**。请按照如下方式上传：
	- (墙内)目前没有找到比较好的方式，只能通过链接，或者自己网盘的链接来做。
	- (墙外)首先在[UPLOAD](https://my.pcloud.com/#page=puplink&code=4e9Z0Vwpmfzvx0y2OqTTTMzkrRUz8q9V)直接上传（**不**需要注册账号）；上传成功后，在[DOWNLOAD](https://my.pcloud.com/publink/show?code=kZWtboZbDDVguCHGV49QkmlLliNPJRMHrFX)里找到你刚上传的文件，共享链接即可。

Welcome!

> ***[文章版权声明]这个仓库是我开源到Github上的，可以遵守相关的开源协议进行使用。这个仓库中包含有很多研究者的论文、硕博士论文等，都来源于在网上的下载，仅作为学术研究使用。我对其中一些文章都写了自己的浅见，希望能很好地帮助理解。这些文章的版权属于相应的出版社。如果作者或出版社有异议，请联系我进行删除。一切都是为了更好地学术！***
