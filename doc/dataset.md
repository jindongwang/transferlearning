## Datasets for domain adaptation and transfer learning

*[I understand the fact that some of the users may fail to download the datasets since the cloud service is overseas, so I tried to make a copy to some Chinese cloud services. Please wait]*

- *How many times have you been* ++struggling to find++ the useful datasets?
- *How much time have you been wasting to* ++preprocess datasets++?
- *How burdersome is it to compare with other methods*? Will you re-run their code? or there is no code?

**Datasets are critical to machine learning**, but *You should focus on* **YOUR** work! So we want to save your time by:

**JUST GIVE THEM TO YOU** so you can **DIRECTLY** use them!

- - -

**Some widely used datasets for domain adaptation and transfer learning are listed here. See [benchmark](#benchmark) of several classical algorithms.**

*Only image datasets are listed, text datasets are to be added*

|     Dataset    |        Area        | #Sample |       #Feature      | #Class |   Subdomain  | Reference |
|:--------------:|:------------------:|:-------:|:-------------------:|:------:|:------------:|:--------:|
| [Office+Caltech](#office+caltech) | Object recognition |   2533  | SURF:800 DeCAF:4096 |   10   |  C, A, W, D  |   [1]       |
|   [MNIST+USPS](#mnist+usps)   |  Digit recognition |   3800  |         256         |   10   |  USPS, MNIST |    [4]      |
|     [COIL20](#coil20)     | Object recognition |   1440  |         1024        |   20   | COIL1, COIL2 |    [4]      |
|       [PIE](#pie)      |  Face recognition  |  11554  |         1024        |   68   |   PIE1~PIE5  |     [6]     |
|     [VOC2007](#vlsc)    |       Object recognition      |   3376  |      DeCAF:4096     |    5   |       V      |    [8]      |
|     [LabelMe](#vlsc)    |       Object recognition      |   2656  |      DeCAF:4096     |    5   |       L      |    [2]      |
|      [SUN09](#vlsc)     |       Object recognition      |   3282  |      DeCAF:4096     |    5   |       S      |    [9]      |
|   [Caltech101](#vlsc)   |       Object recognition      |   1415  |      DeCAF:4096     |    5   |       C      |    [3]      |
|    [IMAGENET](#imagenet)    |       Object recognition      |   7341  |      DeCAF:4096     |    5   |       I      |     [7]     |
|    [AWA](#animals-with-attributes)    |       Animal recognition      |   30475  |      DeCAF:4096 SIFT/SURF:2000    |    50   |       I      |    [5]      |



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

#### Download

Download Office+Caltech SURF dataset [[pCloud](https://my.pcloud.com/publink/show?code=kZFrXk7ZifluxAGy3gjdstJBIcJv3fORIkHk)]

Download Office+Caltech DeCAF dataset [[pCloud](https://my.pcloud.com/publink/show?code=kZprXk7Z1OmGWUuYioSJbWx3jWeCAhom5FPy)]


- - -


### MNIST+USPS

**Area** Handwritten digit recognition

It is also popular. It contains randomly selected samples from MNIST and USPS. Then the source and target domains are constructed using each other.

Download MNIST+USPS SURF dataset [[pCloud](https://my.pcloud.com/publink/show?code=kZHrXk7ZdIfYsBuRVtkPoAqvxL87qhgNw10V)]


- - -


### COIL20

**Area** Object recognition

It contains 20 classes. There are two datasets extracted: COIL1 and COIL2.

Download COIL20 SURF dataset [[pCloud](https://my.pcloud.com/publink/show?code=kZzrXk7ZQw37wqJsNSJVzN1DzH0FH7e3tOYV)]


- - -


### PIE

**Area** Facial recognition

It is a relatively large dataset with many classes.

Download PIE SURF dataset [[pCloud](https://my.pcloud.com/publink/show?code=kZRrXk7ZwvHA9LPSyqSz7WSlECK5A0hNMR6X)]


- - -



### VLSC

**Area** Image classification

It contains four domains: V(VOC2007), L(LabelMe), S(SUN09) and C(Caltech). There are 5 classes: 'bird', 'car', 'chair', 'dog', 'person'.

Download the VLSC DeCAF dataset [[pCloud](https://my.pcloud.com/publink/show?code=kZ8rXk7ZORx8jhQ6CzyItAp2qQHmMbFiyRW7)]


- - -



### IMAGENET

It is selected from imagenet challange.

Download the IMAGENET DeCAF dataset [[pCloud](https://my.pcloud.com/publink/show?code=kZ8rXk7ZORx8jhQ6CzyItAp2qQHmMbFiyRW7)]


- - -


### Animals-with-Attributes

Download the SURF/SIFT/DeCAF features [[pCloud](https://my.pcloud.com/publink/show?code=kZbrXk7ZYAgHIKa0Qa5W1Gi9VXbnMhzIo9GX)]


- - -


## Benchmark

TOCOMPLETE

### Office+Caltech SURF

| ID | Dim | Method | C->A | C->W | C->D | A->C | A->W | A->D | W->C | W->A | W->D | D->C | D->A | D->W |
|----|-----|----------------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| 1 | 100 | PCA+1NN | 36.95 | 32.54 | 38.22 | 34.73 | 35.59 | 27.39 | 26.36 | 31 | 77.07 | 29.65 | 32.05 | 75.93 |
| 2 | 100 | GFK+1NN | 41.02 | 40.68 | 38.85 | 40.25 | 38.98 | 36.31 | 30.72 | 29.75 | 80.89 | 30.28 | 32.05 | 75.59 |
| 3 | 100 | TCA+1NN | 38.2 | 38.64 | 41.4 | 37.76 | 37.63 | 33.12 | 29.3 | 30.06 | 87.26 | 31.7 | 32.15 | 86.1 |
| 4 | 100 | TSL+1NN | 44.47 | 34.24 | 43.31 | 37.58 | 33.9 | 26.11 | 29.83 | 30.27 | 87.26 | 28.5 | 27.56 | 85.42 |
| 5 | 100 | JDA+1NN | 44.78 | 41.69 | 45.22 | 39.36 | 37.97 | 39.49 | 31.17 | 32.78 | 89.17 | 31.52 | 33.09 | 89.49 |
| 6 | 100 | UDA+1NN | 47.39 | 46.56 | 48.41 | 41.41 | 43.05 | 42.04 | 32.41 | 34.45 | 91.08 | 34.19 | 34.24 | 90.85 |
| 7 | 30 | SA+1NN | 49.27 | 40 | 39.49 | 39.98 | 33.22 | 33.76 | 35.17 | 39.25 | 75.16 | 34.55 | 39.87 | 76.95 |
| 8 | 30 | SDA+1NN | 49.69 | 38.98 | 40.13 | 39.54 | 30.85 | 33.76 | 34.73 | 39.25 | 75.8 | 35.89 | 38.73 | 76.95 |
| 9 | 30 | GFK+1NN | 46.03 | 36.95 | 40.76 | 40.69 | 36.95 | 40.13 | 24.76 | 27.56 | 85.35 | 29.3 | 28.71 | 80.34 |
| 10 | 30 | TCA+1NN | 45.82 | 31.19 | 34.39 | 42.39 | 36.27 | 33.76 | 29.39 | 28.91 | 89.17 | 30.72 | 31 | 86.1 |
| 11 | 30 | JDA+1NN | 45.62 | 41.69 | 45.22 | 39.36 | 37.97 | 39.49 | 31.17 | 32.78 | 89.17 | 31.52 | 33.09 | 89.49 |
| 12 | 30 | TJM+1NN | 46.76 | 38.98 | 44.59 | 39.45 | 42.03 | 45.22 | 30.19 | 29.96 | 89.17 | 31.43 | 32.78 | 85.42 |
| 13 | 30 | SCA+1NN | 45.62 | 40 | 47.13 | 39.72 | 34.92 | 39.49 | 31.08 | 29.96 | 87.26 | 30.72 | 31.63 | 84.41 |
| 14 | 30 | JGSAprimal+1NN | 51.46 | 45.42 | 45.86 | 41.5 | 45.76 | 47.13 | 33.21 | 39.87 | 90.45 | 29.92 | 38 | 91.86 |
| 15 | 30 | JGSAlinear+1NN | 52.3 | 45.76 | 48.41 | 38.11 | 49.49 | 45.86 | 32.68 | 41.02 | 90.45 | 30.19 | 36.01 | 91.86 |
| 16 | 30 | JGSArbf+1NN | 53.13 | 48.47 | 48.41 | 41.5 | 45.08 | 45.22 | 33.57 | 40.81 | 88.54 | 30.28 | 38.73 | 93.22 |
| 17 | 20 | PCA+1NN | 36.95 | 32.54 | 38.22 | 34.73 | 35.59 | 27.39 | 26.36 | 29.35 | 77.07 | 29.65 | 32.05 | 75.93 |
| 18 | 20 | FSSL+1NN | 35.88 | 32.32 | 37.53 | 33.91 | 34.35 | 26.37 | 25.85 | 29.53 | 76.79 | 27.89 | 30.61 | 74.99 |
| 19 | 20 | TCA+1NN | 45.82 | 30.51 | 35.67 | 40.07 | 35.25 | 34.39 | 29.92 | 28.81 | 85.99 | 32.06 | 31.42 | 86.44 |
| 20 | 20 | GFK+1NN | 41.02 | 40.68 | 38.85 | 40.25 | 38.98 | 36.31 | 30.72 | 29.75 | 80.89 | 30.28 | 32.05 | 75.59 |
| 21 | 20 | TJM+1NN | 46.76 | 38.98 | 44.59 | 39.45 | 42.03 | 45.22 | 30.19 | 29.96 | 89.17 | 31.43 | 32.78 | 85.42 |
| 22 | 20 | VDA+1NN | 46.14 | 46.1 | 51.59 | 42.21 | 51.19 | 48.41 | 27.6 | 26.1 | 89.18 | 31.26 | 37.68 | 90.85 |
| 23 | no | 1NN | 23.7 | 25.76 | 25.48 | 26 | 29.83 | 25.48 | 19.86 | 22.96 | 59.24 | 26.27 | 28.5 | 63.39 |
| 24 | no | SVM | 55.64 | 45.22 | 43.73 | 45.77 | 42.04 | 39.66 | 31.43 | 34.76 | 82.8 | 29.39 | 26.62 | 63.39 |
| 25 | no | LapSVM | 56.27 | 45.8 | 43.73 | 44.23 | 42.74 | 39.79 | 31.99 | 34.77 | 83.43 | 29.49 | 27.37 | 64.31 |
| 26 | no | TKL+SVM | 54.28 | 46.5 | 51.19 | 45.59 | 49.04 | 46.44 | 34.82 | 40.92 | 83.44 | 35.8 | 40.71 | 84.75 |
| 27 | no | KMM+SVM | 48.32 | 45.78 | 53.53 | 42.21 | 42.38 | 42.72 | 29.01 | 31.94 | 71.98 | 31.61 | 32.2 | 72.88 |
| 28 | no | DTMKL+SVM | 54.33 | 42.04 | 44.74 | 45.01 | 36.94 | 40.85 | 32.5 | 36.53 | 88.85 | 32.1 | 34.03 | 81.69 |
| 29 | no | SKM+SVM | 53.97 | 43.31 | 43.05 | 44.7 | 37.58 | 42.37 | 31.34 | 35.07 | 89.81 | 30.37 | 30.27 | 81.02 |

**Results are coming from:**

- 1~5：[4]
- 6~15: [11]
- 16~21: [12]
- 22~28: [13]

- - -


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
