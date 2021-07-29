### Domain adaptation介绍及代表性文章梳理

Domain adaptation，DA，中文可翻译为域适配、域匹配、域适应，是迁移学习中的一类非常重要的问题，也是一个持续的研究热点。Domain adaptation可用于计算机视觉、物体识别、文本分类、声音识别等常见应用中。这个问题的基本定义是，假设源域和目标域的类别空间一样，特征空间也一样，但是数据的分布不一样，如何利用有标定的源域数据，来学习目标域数据的标定？

事实上，根据目标域中是否有少量的标定可用，可以将domain adaptation大致分为无监督（目标域中完全无label）和半监督（目标域中有少量label）两大类。我们这里偏重介绍无监督。

[视觉domain adaptation综述](https://mega.nz/#!hWQ3HLhJ!GTCIUTVDcmnn3f7-Ulhjs_MxGv6xnFyp1nayemt9Nis)

关于迁移学习的理论方面，有三篇连贯式的理论分析文章连续发表在NIPS和Machine Learning上：[理论分析](https://mega.nz/#F!ULoGFYDK!O3TQRZwrNeqTncNMIfXNTg)
- - -

#### 形式化

给定：有标定的![](https://latex.codecogs.com/png.latex?\mathcal{D}_{S}=\{X_{S_i},Y_{S_i}\}^{n}_{i=1})，以及无标定的![](https://latex.codecogs.com/png.latex?\mathcal{D}_{T}=\{X_{T_i},?\}^{m}_{i=1})

求：![](https://latex.codecogs.com/png.latex?\mathcal{D}_{T})的标定![](https://latex.codecogs.com/png.latex?Y_{T}) （在实验环境中，![](https://latex.codecogs.com/png.latex?\mathcal{D}_{T})是有标定的，仅用来测试算法精度）

条件：
- ![](https://latex.codecogs.com/png.latex?X_{S},X_{T}&space;\in&space;\mathbf{R}^{p&space;\times&space;d})，即源域和目标域的特征空间相同（都是![](https://latex.codecogs.com/png.latex?d)维）
- ![](https://latex.codecogs.com/png.latex?\{Y_{S}\}=\{Y_{T}\})，即源域和目标域的类别空间相同
- ![](https://latex.codecogs.com/png.latex?P(X_{S}))![](https://latex.codecogs.com/png.latex?\ne) ![](https://latex.codecogs.com/png.latex?P(X_T))，即源域和目标域的数据分布不同

- - -

#### 例子

比如说，同样都是一台电脑，在不同角度，不同光照，以及不同背景下拍照，图像的数据具有不同的分布，但是从根本上来说，都是一台电脑的图像。Domain adaptation要做的就是，如何根据这些不同分布的数据，很好地学习缺失的标定。

![Domain adaptation](https://raw.githubusercontent.com/jindongwang/transferlearning/master/png/domain%20_adaptation.png)

代表方法和文章请见这里：https://github.com/jindongwang/transferlearning/blob/master/doc/awesome_paper.md#deep-domain-adaptation
