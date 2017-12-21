# 迁移学习 Transfer Learning

关于迁移学习资料汇总，包括：介绍，综述文章，代表工作及其代码，常用数据集，硕博士论文等等。欢迎一起贡献！

> 关于机器学习和行为识别的资料，请参考：[行为识别](https://github.com/jindongwang/activityrecognition)｜[机器学习](https://github.com/jindongwang/MachineLearning)

_ _ _

#### 目录 Table of contents

* [最新 Latest](#0latest)
* [迁移学习简介 Introduction to transfer learning](#1迁移学习简介)
* [迁移学习的综述文章 Survey papers](#2迁移学习的综述文章)
* [迁移学习文章 Awesome transfer learning papers](#迁移学习文章) 
* [迁移学习相关代码 Available codes](https://github.com/jindongwang/transferlearning/tree/master/code)
* [迁移学习代表性研究学者 Scholars](#4迁移学习代表性研究学者)
* [迁移学习相关的硕博士论文 Thesis](#5迁移学习相关的硕博士论文)
* [代表性文章阅读 Paper reading](#6代表性文章阅读)
* [迁移学习用于行为识别 Transfer learning for activity recognition](https://github.com/jindongwang/activityrecognition/blob/master/notes/%E8%BF%81%E7%A7%BB%E5%AD%A6%E4%B9%A0%E7%94%A8%E4%BA%8E%E8%A1%8C%E4%B8%BA%E8%AF%86%E5%88%AB.md)
* [常用数据集 Datasets](https://github.com/jindongwang/transferlearning/blob/master/doc/dataset.md)
* [Contributing](#contributing)

- - -

### 0.Latest

[迁移学习文章 Awesome transfer learning papers](https://github.com/jindongwang/transferlearning/tree/master/doc/awesome_paper.md)

- 20171218 假设target domain中的class是包含在source domain中，然后进行选择性的对抗学习：[Partial Transfer Learning with Selective Adversarial Networks](https://arxiv.org/abs/1707.07901)

- 20171216 当target domain的数据不可用时，如何用相关domain的数据进行辅助学习？[Zero-Shot Deep Domain Adaptation](https://arxiv.org/abs/1707.01922)

- 20171201 第一篇将Tensor与domain adaptation结合的文章：[When Unsupervised Domain Adaptation Meets Tensor Representations](http://openaccess.thecvf.com/content_iccv_2017/html/Lu_When_Unsupervised_Domain_ICCV_2017_paper.html) | [我的解读](https://zhuanlan.zhihu.com/p/31834244)
- 201712 将Meta-Learning与domain generalization结合的文章：[Learning to Generalize: Meta-Learning for Domain Generalization](https://arxiv.org/pdf/1710.03463.pdf) 
- 201711 运用贝叶斯网络建模label和domain的依赖关系：[GENERALIZING ACROSS DOMAINS VIA CROSS-GRADIENT TRAINING](https://openreview.net/pdf?id=r1Dx7fbCW)
- 20171126 李飞飞小组发在NIPS 2017的文章：[Label Efficient Learning of Transferable Representations acrosss Domains and Tasks](http://papers.nips.cc/paper/6621-label-efficient-learning-of-transferable-representations-acrosss-domains-and-tasks)。针对不同的domain、不同的feature、不同的label space，统一学习一个深度网络进行表征。
- 201711 ICCV 2017发表的文章：[Open set domain adaptation](http://openaccess.thecvf.com/content_iccv_2017/html/Busto_Open_Set_Domain_ICCV_2017_paper.html)。当source和target只共享某一些类别时，怎么处理？这个文章获得了ICCV 2017的Marr Prize Honorable Mention，值得好好研究。[我的解读](https://zhuanlan.zhihu.com/p/31230331)


[更多...](https://github.com/jindongwang/transferlearning/tree/master/doc/awesome_paper.md)

- - -


### 1.迁移学习简介

[文档](https://github.com/jindongwang/transferlearning/blob/master/doc/%E8%BF%81%E7%A7%BB%E5%AD%A6%E4%B9%A0%E7%AE%80%E4%BB%8B.md)   ||   [PPT(英文)](http://jd92.wang/assets/files/l03_transferlearning.pdf)   ||  [PPT(中文)](http://jd92.wang/assets/files/l08_tl_zh.pdf)

台湾大学李宏毅的视频讲解，非常不错：https://www.youtube.com/watch?v=qD6iD4TFsdQ

[迁移学习领域的著名学者、代表工作及实验室介绍](https://github.com/jindongwang/transferlearning/blob/master/doc/scholar_TL.md)

什么是[负迁移(negative transfer)](https://www.zhihu.com/question/66492194/answer/242870418)？

[迁移学习中的领域自适应方法](http://jd92.wang/assets/files/l12_da.pdf)

动手教程：很好的深度学习+迁移学习的实践教程，有代码有数据，可以直接上手：[基于深度学习和迁移学习的识花实践(Tensorflow)](https://cosx.org/2017/10/transfer-learning/)  |  [基于Pytorch的图像分类](https://github.com/miguelgfierro/sciblog_support/blob/master/A_Gentle_Introduction_to_Transfer_Learning/Intro_Transfer_Learning.ipynb)

- - -

### 2.迁移学习的综述文章

[一些迁移学习的综述文章](https://pan.baidu.com/s/1c19TTfU)，中文英文都有。

- 其中，最具代表性的综述是[A survey on transfer learning](https://pan.baidu.com/s/1i5AFO7z)，对迁移学习进行了比较权威的定义。

- 最新的综述是[Cross-dataset recognition: a survey](https://arxiv.org/abs/1705.04396)，目前刚发在arXiv上，作者是澳大利亚卧龙岗大学的在读博士生，迁移学习领域做的不错。

- 来自香港科技大学Qiang Yang老师团队的最新综述[A survey on multi-task learning](https://arxiv.org/abs/1707.08114)

- 还有一篇较新的综述是[A survey of transfer learning](https://pan.baidu.com/s/1gfgXLXT)，写于2015-2016年。其中交代了一些比较经典的如同构、异构等学习方法代表性文章。包括了很多方法介绍，值得一看。

- 此外，还包括[迁移学习应用于行为识别](https://pan.baidu.com/s/1kVABOYr)、[迁移学习与增强学习](https://pan.baidu.com/s/1slfr0w1)结合等。
- 关于[多个源域进行迁移的综述](https://pan.baidu.com/s/1eSGREF4)、[视觉domain adaptation综述](https://pan.baidu.com/s/1o8BR7Vc)也十分有用。
- 中文方面，[迁移学习研究进展](https://pan.baidu.com/s/1bpautob)是一篇不错的中文综述。
- 关于迁移学习的**理论**方面，有三篇连贯式的理论分析文章连续发表在NIPS和Machine Learning上：[理论分析](https://pan.baidu.com/s/1jIotlTc)

_ _ _

### [迁移学习文章](https://github.com/jindongwang/transferlearning/tree/master/doc/awesome_paper.md)

### 3.代码

请见[这里](https://github.com/jindongwang/transferlearning/tree/master/code)

_ _ _

### 4.迁移学习代表性研究学者

**全部列表以及代表工作性见[这里](https://github.com/jindongwang/transferlearning/blob/master/doc/scholar_TL.md)**

- [Qiang Yang](http://www.cs.ust.hk/~qyang/)：中文名杨强。香港科技大学计算机系主任，教授，大数据中心主任。迁移学习领域世界性专家。IEEE/AAAI/IAPR/AAAS fellow。[[Google scholar](https://scholar.google.com/citations?user=1LxWZLQAAAAJ&hl=zh-CN)]
- [Sinno Jialin Pan](http://www.ntu.edu.sg/home/sinnopan/)：杨强的学生，香港科技大学博士，现任新加坡南洋理工大学助理教授。迁移学习领域代表性综述A survey on transfer learning的第一作者（Qiang Yang是二作）。[[Google scholar](https://scholar.google.com/citations?user=P6WcnfkAAAAJ&hl=zh-CN)]
- [Wenyuan Dai](https://scholar.google.com.sg/citations?user=AGR9pP0AAAAJ&hl=zh-CN)：中文名戴文渊，上海交通大学硕士，现任第四范式人工智能创业公司CEO。迁移学习领域著名的牛人，在顶级会议上发表多篇高水平文章，每篇论文引用量巨大。
- [Lixin Duan](http://www.lxduan.info/)：中文名段立新，新加坡南洋理工大学博士，现就职于电子科技大学，教授。
- [Fuzhen Zhuang](http://www.intsci.ac.cn/users/zhuangfuzhen/)：中文名庄福振，中科院计算所博士，现任中科院计算所副研究员。[[Google scholar](https://scholar.google.com/citations?user=klJBYrAAAAAJ&hl=zh-CN&oi=ao)]
- [Mingsheng Long](http://ise.thss.tsinghua.edu.cn/~mlong/)：中文名龙明盛，清华大学博士，现任清华大学助理教授、博士生导师。[[Google scholar](https://scholar.google.com/citations?view_op=search_authors&mauthors=mingsheng+long&hl=zh-CN&oi=ao)]
_ _ _

### 5.迁移学习相关的硕博士论文

硕博士论文可以让我们很快地对迁移学习的相关领域做一些了解，同时，也能很快地了解概括相关研究者的工作。其中，比较有名的有

- 杨强的学生Sinno Jialin Pan的[Feature-based Transfer Learning and Its Applications](https://pan.baidu.com/s/1bUqMfW)
- 南加州大学的Boqing Gong的[Kernel Methods for Unsupervised Domain Adaptation](https://pan.baidu.com/s/1bpbawv9)
- 杨强的学生Hao Hu的[Learning based Activity Recognition](https://pan.baidu.com/s/1bp2K9HX)
- 杨强的学生Wencheng Zheng的[Learning with Limited Data in Sensor-based Human Behavior Prediction](https://pan.baidu.com/s/1o8MbbBk)
- 清华大学龙明盛的[迁移学习问题与方法研究](https://pan.baidu.com/s/1pLiJzIV)
- 上海交通大学戴文渊的[基于实例和特征的迁移学习算法研究](https://pan.baidu.com/s/1i4Vyygd)
- 中科院计算所赵中堂的[自适应行为识别中的迁移学习方法研究](https://pan.baidu.com/s/1kVqYXnh)
- Baochen Sun的[Correlation Alignment for Domain Adaptation](http://www.cs.uml.edu/~bsun/papers/baochen_phd_thesis.pdf)

其他的文章，请见[完整版](https://pan.baidu.com/s/1mizKzna)。

- - -

### 6.代表性文章阅读

Domain adaptation是迁移学习领域比较热的研究方向，在这里整理了一些经典的文章和说明：[Domain adaptation](https://github.com/jindongwang/transferlearning/blob/master/doc/domain_adaptation.md)

最近一个推荐、分享论文的网站比较好，我在上面会持续整理相关的文章并分享阅读笔记。详情请见[paperweekly](http://www.paperweekly.site/collections/231/papers)。

#### 代表性的方法及文章

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
- 深度适配网络（Deep Adaptation Network, DAN）
	- 发表在ICML-15上：learning transferable features with deep adaptation networks
	- [我的解读](https://zhuanlan.zhihu.com/p/27657910)

- [深度联合适配网络](http://proceedings.mlr.press/v70/long17a.html)（Joint Adaptation Network, JAN）
	- Deep Transfer Learning with Joint Adaptation Networks
	- 发表在ICML 2017上，作者也是龙明盛
	- 延续了之前的DAN工作，这次考虑联合适配

- [学习迁移](https://arxiv.org/abs/1708.05629)(Learning to Transfer, L2T)
	- 迁移学习领域的新方向：与在线、增量学习结合
	- [我的解读](https://zhuanlan.zhihu.com/p/28888554)

- [Simultaneous Deep Transfer Across Domains and Tasks](https://www.cv-foundation.org/openaccess/content_iccv_2015/html/Tzeng_Simultaneous_Deep_Transfer_ICCV_2015_paper.html)
	- 发表在ICCV-15上，在传统深度迁移方法上又加了新东西
	- [我的解读](https://zhuanlan.zhihu.com/p/30621691)
_ _ _

### [迁移学习用于行为识别的文章总结](https://github.com/jindongwang/activityrecognition/blob/master/notes/%E8%BF%81%E7%A7%BB%E5%AD%A6%E4%B9%A0%E7%94%A8%E4%BA%8E%E8%A1%8C%E4%B8%BA%E8%AF%86%E5%88%AB.md)

我写的迁移学习应用于行为识别领域的文章小总结。

_ _ _

### [记与迁移学习大牛杨强教授的第二次会面](https://zhuanlan.zhihu.com/p/26260083)

- - -

### Contributing

如果你对本项目感兴趣，非常欢迎你加入！

- 正常参与：请直接fork、pull都可以
- 如果要上传文件：请**不要**直接上传到项目中，否则会造成git版本库过大。正确的方法是上传它的**超链接**。如果你要上传的文件本身就在网络中（如paper都会有链接），直接上传即可；如果是自己想分享的一些文件、数据等，鉴于国内网盘的情况，请按照如下方式上传：
	- (墙内)目前没有找到比较好的方式，只能通过链接，或者自己网盘的链接来做。
	- (墙外)首先在[UPLOAD](https://my.pcloud.com/#page=puplink&code=4e9Z0Vwpmfzvx0y2OqTTTMzkrRUz8q9V)直接上传（**不**需要注册账号）；上传成功后，在[DOWNLOAD](https://my.pcloud.com/publink/show?code=kZWtboZbDDVguCHGV49QkmlLliNPJRMHrFX)里找到你刚上传的文件，共享链接即可。

Welcome!

> ***[文章版权声明]这个仓库是我开源到Github上的，可以遵守相关的开源协议进行使用。这个仓库中包含有很多研究者的论文、硕博士论文等，都来源于在网上的下载，仅作为学术研究使用。我对其中一些文章都写了自己的浅见，希望能很好地帮助理解。这些文章的版权属于相应的出版社。如果作者或出版社有异议，请联系我进行删除（本来应该只放文章链接的，但是由于时间关系来不及）。一切都是为了更好地学术！***
