## Datasets for domain adaptation and transfer learning

- *How many times have you been* **struggling to find** the useful datasets?
- *How much time have you been wasting to* **preprocess datasets**?
- *How burdersome is it to compare with other methods*? Will you re-run their code? or there is **No** code?

**Datasets are critical to machine learning**, but *You should focus on* **YOUR** work! So we want to save your time by:

**JUST GIVE THEM TO YOU** so you can **DIRECTLY** use them!

- - -

**If you are tired of repeating the experiments of other methods, you can directly use the [benchmark](https://github.com/jindongwang/transferlearning/blob/master/data/benchmark.md).**

Most datasets are image datasets:

|     Dataset    |        Area        | #Sample |       #Feature      | #Class |   Subdomain  | Reference |
|:--------------:|:------------------:|:-------:|:-------------------:|:------:|:------------:|:--------:|
| [Office+Caltech](#office+caltech) | Object recognition |   2533  | SURF:800 DeCAF:4096 |   10   |  C, A, W, D  |   [1]       |
| [Office-31](#office-31) | Object recognition |   4110  | SURF:800 DeCAF:4096 |   31   |  A, W, D  |   [1]       |
|   [MNIST+USPS](#mnist+usps)   |  Digit recognition |   3800  |         256         |   10   |  USPS, MNIST |    [4]      |
|     [COIL20](#coil20)     | Object recognition |   1440  |         1024        |   20   | COIL1, COIL2 |    [4]      |
|       [PIE](#pie)      |  Face recognition  |  11554  |         1024        |   68   |   PIE1~PIE5  |     [6]     |
|     [VOC2007](#vlsc)    |       Object recognition      |   3376  |      DeCAF:4096     |    5   |       V      |    [8]      |
|     [LabelMe](#vlsc)    |       Object recognition      |   2656  |      DeCAF:4096     |    5   |       L      |    [2]      |
|      [SUN09](#vlsc)     |       Object recognition      |   3282  |      DeCAF:4096     |    5   |       S      |    [9]      |
|   [Caltech101](#vlsc)   |       Object recognition      |   1415  |      DeCAF:4096     |    5   |       C      |    [3]      |
|    [IMAGENET](#imagenet)    |       Object recognition      |   7341  |      DeCAF:4096     |    5   |       I      |     [7]     |
|    [AWA](#animals-with-attributes)    |       Animal recognition      |   30475  |      DeCAF:4096 SIFT/SURF:2000    |    50   |       I      |    [5]      |
|    [Office-Home](#office-home)    |       Object recognition      |   30475  |      Original Images    |    65   |       4 domains      |    [10]      |
|    [Cross-dataset Testbed](#testbed)    |       Image Classification      |   *  |      Decaf7    |    40   |       3 domains     |    [15]      |
|    [ImageCLEF](#imageclef)    |       Image Classification      |   *  |      raw    |    12   |       3 domains     |       [17]  |
|    [VisDA](#VisDA)    |       Image Classification / segmentation      |   280157  |      raw    |    12   |       3 domains/3 domain     |       [18]  |
|    [LSDAC](#LSDAC)    |       Image Classification      |   569010  |      raw    |    345   |       6 domains     |       [19]  |


**NEW** A even larger dataset called [DomainNet](http://ai.bu.edu/M3SDA/#refs) is released by BU, with half a million images, 6 domains, and 345 classes!

**NEW** A new dataset released by Stanford and UC Berkeley: [Syn2Real: A New Benchmark forSynthetic-to-Real Visual Domain Adaptation](https://arxiv.org/abs/1806.09755)

## Text datasets:

Amazon review for sentiment classification

You can download the datasets [here](https://pan.baidu.com/s/1F69rBEF9_DwrVcEjHXk96w) with code `a82t`.

- - -


### Office+Caltech

#### Area

Visual object recognition

#### Introduction

Perhaps it is the **most popular** dataset for domain adaptation. Four domains are included: C(Caltech), A(Amazon), W(Webcam) and D(DSLR). In fact, this dataset is constructed from two datasets: Office-31 (which contains 31 classes of A, W and D) and Caltech-256 (which contains 256 classes of C). There are just 10 common classes in both, so the Office+Caltech dataset is formed.

Even for the same category, the data distribution of different domains is exactly different. The following picture [1] indicates this fact by the monitor images from 4 domains.

![](https://raw.githubusercontent.com/jindongwang/transferlearning/master/png/domain%20_adaptation.png)

#### Features

There are ususlly two kinds of features: SURF and DeCAF6. They are with the same number of samples per domain, resulting 2533 samples in total:

- C: 1123
- A: 958
- W: 295
- D: 157

And the dimension of features is:

- For SURF: 800
- For DeCAF6: 4096

#### Copyright

This dataset was first introduced by Gong et al. [1]. I got the SURF features from the website of [1], while DeCAF features from [10].

See benchmark results of many popular methods [here(SURF)](https://github.com/jindongwang/transferlearning/blob/master/doc/benchmark.md#officecaltech-surf) and [here(DeCaf)](https://github.com/jindongwang/transferlearning/blob/master/doc/benchmark.md#officecaltech10-decaf6).

#### Download

Download Office+Caltech original images [[Baiduyun](https://pan.baidu.com/s/14JEGQ56LJX7LMbd6GLtxCw)]

Download Office+Caltech SURF dataset [[MEGA](https://mega.nz/#F!AaJTGIzD!XHM2XMsSd9V-ljVi0EtvFg)|[Baiduyun](https://pan.baidu.com/s/1bp4g7Av)]

Download Office+Caltech DeCAF dataset [[MEGA](https://mega.nz/#F!QDxBBC4J!LizxWbE1_JEwPSrA2mrrrw)|[Baiduyun](https://pan.baidu.com/s/1nvn7eUd)]


- - -

### Office-31

This is the full Office dataset, which contains 31 categories from Amazon, webcam and DSLR.

See benchmarks on Office-31 datasets [here](https://github.com/jindongwang/transferlearning/blob/master/doc/benchmark.md#office-31).

#### Download

[Download Office-31 raw images](https://pan.baidu.com/s/1o8igXT4) | [Download from MEGA](https://mega.nz/file/dSpjyCwR#9ctB4q1RIE65a4NoJy0ox3gngh15cJqKq1XpOILJt9s)

[Download Office-31 DeCAF6 and DeCAF7 features](https://pan.baidu.com/s/1o7XrAzw)

[Download Office-31 DeCAF features by Frame](https://pan.baidu.com/s/1i5KkNxb)

[Download Office-31 SURF features](https://pan.baidu.com/s/1kU6tv4F)

[Download Office-31 ResNet-50 features](https://pan.baidu.com/s/1UoyJSqoCKCda-NcP-zraVg) | [Download from MEGA](https://mega.nz/#F!laI2lKoJ!nSmVQXrpu1Ov794sy2wFKg)

- - -

### MNIST+USPS

**Area** Handwritten digit recognition

It is also popular. It contains randomly selected samples from MNIST and USPS. Then the source and target domains are constructed using each other.

Download MNIST+USPS SURF dataset [[MEGA](https://mega.nz/#F!oHJ2UCoK!r62nRoZ0gH8NXIcgmyWReA)|[Baiduyun](https://pan.baidu.com/s/1c8mwdo)]


- - -


### COIL20

**Area** Object recognition

It contains 20 classes. There are two datasets extracted: COIL1 and COIL2.

Download COIL20 SURF dataset [[MEGA](https://mega.nz/#F!xWxyTDaZ!MWamSH17Uu065XbDNYymkQ)|[Baiduyun](https://pan.baidu.com/s/1pKM1VCn)]


- - -


### PIE

**Area** Facial recognition

It is a relatively large dataset with many classes.

Download PIE SURF dataset [[MEGA](https://mega.nz/#F!lPgmkZQB!z2QuBEmCzj2XR5AAQaIj7Q)|[Baiduyun](https://pan.baidu.com/s/1o8KFgtO)]


- - -



### VLSC

**Area** Image classification

It contains four domains: V(VOC2007), L(LabelMe), S(SUN09) and C(Caltech). There are 5 classes: 'bird', 'car', 'chair', 'dog', 'person'.

Download the VLSC DeCAF dataset [[MEGA](https://mega.nz/#F!gTJxGTJK!w9UJjZVq3ClqGj4mBDmT4A)|[Baiduyun](https://pan.baidu.com/s/1nuNiJ0l)]


- - -



### IMAGENET

It is selected from imagenet challange.

Download the IMAGENET DeCAF dataset [[MEGA](https://mega.nz/#F!gTJxGTJK!w9UJjZVq3ClqGj4mBDmT4A)|[Baiduyun](https://pan.baidu.com/s/1nuNiJ0l)]


- - -


### Animals-with-Attributes

Download the SURF/SIFT/DeCAF features [[MEGA](https://mega.nz/#F!Na5jyLiC!LT29_gyoPsd_eEoym3CgMg)|[Baiduyun](https://pan.baidu.com/s/1mi7RYQW)]

- - -

### Office-Home

This is a **new** dataset released at CVPR 2017. It contains 65 kinds of objects crawled from the web. The main research goal is for domain adatpation algorithms benchmark.

The project home page is: http://hemanthdv.org/OfficeHome-Dataset/, the dataset can be downloaded there.

Currently, there are no feature files, which is bad for traditional (no-deep) methods. I extracted features using ResNet-50 for you to use:

[Download Office-Home ResNet-50 pretrained features](https://pan.baidu.com/s/1qvcWJCXVG8JkZnoM4BVoGg) | [Download from MEGA](https://mega.nz/#F!pGIkjIxC!MDD3ps6RzTXWobMfHh0Slw)

- - -

### Cross-dataset Testbed

This is a Decaf7 based cross-dataset image classification dataset. It contains 40 categories of images from 3 domains: 3,847 images in Caltech256(C), 4,000 images in ImageNet(I), and 2,626 images for SUN(S).

[Download the Cross-dataset testbed](https://pan.baidu.com/s/1o8MeVUi)

- - -

### ImageCLEF

This is a dataset from ImageCLEF 2014 challenge.

[Download the ImageCLEF dataset](https://pan.baidu.com/s/1lx2u1SMlSamsHnAPWrAHWA)

[Download Image-CLEF ResNet-50 pretrained features](https://pan.baidu.com/s/16wBgDJI6drA0oYq537h4FQ) | [Download from MEGA](https://mega.nz/#F!QPJCzShS!b6qQUXWnCCGBMVs0m6MdQw)

- - -

### VisDA

This is a dataset from VisDA 2017 challenge. It contains two subdatasets, one for image classification tasks and the other for image segmentation tasks.

[Download the VisDA-classification dataset](http://csr.bu.edu/ftp/visda17/clf/)

[Download the VisDA-segmentation dataset](http://csr.bu.edu/ftp/visda17/seg/)

[Download VisDA classification dataset features by ResNet-50](https://pan.baidu.com/s/1sbuDqWWzwLyB1fFIpo5BdQ) | [Download from MEGA](https://mega.nz/#F!ZDY2jShR!r_M2sR7MBi_9JPsRUXXy0g)

- - -

### LSDAC

This is probably the **largest and latest** domain adaptation datasets ever! It is collected by Boston U, which contains 6 domains from 345 categories, leading to 600K images. Dataset download link will be available soon once the authors release their datasets. You can refer to [19] for more information.

- - -

### Amazon review

[Download Amazon review dataset](https://mega.nz/#F!RS43DADD!4pWwFA0CBJP1oLhAR23bTA)

For more image datasets, please refer to https://sites.google.com/site/crossdataset/home/files




### References

[1] Gong B, Shi Y, Sha F, et al. Geodesic flow kernel for unsupervised domain adaptation[C]//Computer Vision and Pattern Recognition (CVPR), 2012 IEEE Conference on. IEEE, 2012: 2066-2073.

[2] Russell B C, Torralba A, Murphy K P, et al. LabelMe: a database and web-based tool for image annotation[J]. International journal of computer vision, 2008, 77(1): 157-173.

[3] Griffin G, Holub A, Perona P. Caltech-256 object category dataset[J]. 2007.

[4] Long M, Wang J, Ding G, et al. Transfer feature learning with joint distribution adaptation[C]//Proceedings of the IEEE International Conference on Computer Vision. 2013: 2200-2207.

[5] http://attributes.kyb.tuebingen.mpg.de/

[6] http://www.cs.cmu.edu/afs/cs/project/PIE/MultiPie/Multi-Pie/Home.html

[7] http://www.cs.dartmouth.edu/~chenfang/proj_page/FXR_iccv13/

[8] M. Everingham, L. Van-Gool, C. K. Williams, J. Winn, and A. Zisserman, “The pascal visual object classes (VOC) challenge,” Int. J. Comput. Vis., vol. 88, no. 2, pp. 303–338, 2010.

[9] M. J. Choi, J. J. Lim, A. Torralba, and A. S. Willsky, “Exploiting hierarchical context on a large database of object categories,” in Proc. IEEE Conf. Comput. Vis. Pattern Recogit., 2010, pp. 129–136

[10] http://www.uow.edu.au/~jz960/

[11] Zhang J, Li W, Ogunbona P. Joint Geometrical and Statistical Alignment for Visual Domain Adaptation[C]. CVPR 2017.

[12] Tahmoresnezhad J, Hashemi S. Visual domain adaptation via transfer feature learning[J]. Knowledge and Information Systems, 2017, 50(2): 585-605.

[13] Long M, Wang J, Sun J, et al. Domain invariant transfer kernel learning[J]. IEEE Transactions on Knowledge and Data Engineering, 2015, 27(6): 1519-1532.

[14] Venkateswara H, Eusebio J, Chakraborty S, et al. Deep hashing network for unsupervised domain adaptation[C]. CVPR 2017.

[15] Daumé III H. Frustratingly easy domain adaptation[J]. arXiv preprint arXiv:0907.1815, 2009.

[16] Luo L, Chen L, Hu S. Discriminative Label Consistent Domain Adaptation[J]. arXiv preprint arXiv:1802.08077, 2018.

[17] http://imageclef.org/2014/adaptation

[18] Peng X. VisDA: The Visual Domain Adaptation Challenge. arXiv preprint arXiv:1710.06924.

[19] Xingchao Peng, et al. Moment Matching for Multi-Source Domain Adaptation. arXiv 1812.01754.