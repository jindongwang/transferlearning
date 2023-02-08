# Transfer learning applications

By reverse chronological order.

迁移学习的应用，按照时间顺序倒序排列。

- [Transfer learning applications](#transfer-learning-applications)
  - [Computer vision](#computer-vision)
  - [Medical and healthcare](#medical-and-healthcare)
  - [Natural language processing](#natural-language-processing)
  - [Time series](#time-series)
  - [Speech](#speech)
  - [Multimedia](#multimedia)
  - [Recommendation](#recommendation)
  - [Human activity recognition](#human-activity-recognition)
  - [Autonomous driving](#autonomous-driving)
  - [Others](#others)

## Computer vision

- CLIP the Gap: A Single Domain Generalization Approach for Object Detection [[arxiv](https://arxiv.org/abs/2301.05499)]
  - Using CLIP for domain generalization object detection 使用CLIP进行域泛化的目标检测

- ECCV-22 DecoupleNet: Decoupled Network for Domain Adaptive Semantic Segmentation [[arXiv](https://arxiv.org/pdf/2207.09988.pdf)] [[Code](https://github.com/dvlab-research/DecoupleNet)]
  - Domain adaptation in semantic segmentation 语义分割域适应

- Unsupervised Domain Adaptation for COVID-19 Information Service with Contrastive Adversarial Domain Mixup [[arxiv](https://arxiv.org/abs/2210.03250)]
  - Domain adaptation for COVID-19 用DA进行COVID-19预测

- Deep Domain Adaptation for Detecting Bomb Craters in Aerial Images [[arxiv](https://arxiv.org/abs/2209.11299)]
  - Bomb craters detection using domain adaptation 用DA检测遥感图像中的炮弹弹坑

- Language-aware Domain Generalization Network for Cross-Scene Hyperspectral Image Classification [[arxiv](https://arxiv.org/pdf/2209.02700.pdf)]
  - Domain generalization for cross-scene hyperspectral image classification 域泛化用于高光谱图像分类

- MM-22 [Source-Free Domain Adaptation for Real-world Image Dehazing](https://arxiv.org/abs/2207.06644)
  - Source-free DA for image dehazing 无需源域的迁移用于图像去雾

- CVPR-22 [Segmenting Across Places: The Need for Fair Transfer Learning With Satellite Imagery](https://openaccess.thecvf.com/content/CVPR2022W/FaDE-TCV/html/Zhang_Segmenting_Across_Places_The_Need_for_Fair_Transfer_Learning_With_CVPRW_2022_paper.html)
  - Fair transfer learning with satellite imagery 公平迁移学习

- [FiT: Parameter Efficient Few-shot Transfer Learning for Personalized and Federated Image Classification](https://arxiv.org/abs/2206.08671)
  - Few-shot transfer learning for image classification 小样本迁移学习用于图像分类

- [COVID-19 Detection using Transfer Learning with Convolutional Neural Network](https://arxiv.org/abs/2206.08557)
  - COVID-19 using transfer learning 用迁移学习进行COVID-19检测

- [Toward Certified Robustness Against Real-World Distribution Shifts](https://arxiv.org/abs/2206.03669)
  - Certified robustness against real-world distribution shifts 真实世界中的distribution shift

- [One Ring to Bring Them All: Towards Open-Set Recognition under Domain Shift](https://arxiv.org/abs/2206.03600)
  - Open set recognition with domain shift 开放集+domain shift

- [ConFUDA: Contrastive Fewshot Unsupervised Domain Adaptation for Medical Image Segmentation](https://arxiv.org/abs/2206.03888)
  - Fewshot UDA for medical image segmentation 小样本域自适应用于医疗图像分割

- ICME-22 [Unsupervised Domain Adaptation Learning for Hierarchical Infant Pose Recognition with Synthetic Data](https://arxiv.org/abs/2205.01892)
  - Unsupervised domain adaptation for infant pose recognition 用领域自适应进行婴儿姿势识别

- CVPR-22 [MM-TTA: Multi-Modal Test-Time Adaptation for 3D Semantic Segmentation](https://arxiv.org/abs/2204.12667)
  - Multi-modal test-time adaptation for 3D semantic segmentation 多模态测试时adaptation用于3D语义分割

- [Undoing the Damage of Label Shift for Cross-domain Semantic Segmentation](https://arxiv.org/abs/2204.05546)
  - Handle the label shift in cross-domain semantic segmentation  在跨域语义分割时考虑label shift

- [Gated Domain-Invariant Feature Disentanglement for Domain Generalizable Object Detection](https://arxiv.org/abs/2203.11432)
  - Channel masking for domain generalization object detection
  - 通过一个gate控制channel masking进行object detection DG

- [Domain generalization in deep learning-based mass detection in mammography: A large-scale multi-center study](https://arxiv.org/abs/2201.11620)
  - Domain generalization in mass detection in mammography
  - Domain generalization进行胸部射线检测

- [Continual Coarse-to-Fine Domain Adaptation in Semantic Segmentation](https://arxiv.org/abs/2201.06974)
  - Domain adaptation in semantic segmentation
  - 领域自适应在语义分割的应用

- [Transfer Learning for Scene Text Recognition in Indian Languages](2201.03180)
  - Transfer learning for scene text recognition in Indian languages
  - 用迁移学习进行印度语的场景文字识别

- IEEE TMM-22 [Decompose to Adapt: Cross-domain Object Detection via Feature Disentanglement](https://arxiv.org/abs/2201.01929)
  - Invariant and shared components for Faster RCNN detection
  - 解耦公共和私有表征进行目标检测

- [Transfer learning of phase transitions in percolation and directed percolation](https://arxiv.org/abs/2112.15516)
  - Transfer learning of phase transitions in percolation and directed percolation
  - 迁移学习用于precolation

- [Transfer learning for cancer diagnosis in histopathological images](https://arxiv.org/abs/2112.15523)
  - Transfer learning for cancer diagnosis
  - 迁移学习用于癌症诊断

- [Subtask-dominated Transfer Learning for Long-tail Person Search](https://arxiv.org/abs/2112.00527)
    - Subtask-dominated transfer for long-tail person search
    - 子任务驱动的长尾人物搜索

- NeurIPS-21 workshop [CytoImageNet: A large-scale pretraining dataset for bioimage transfer learning](https://arxiv.org/abs/2111.11646)
    - A large-scale dataset for bioimage transfer learning
    - 一个大规模的生物图像数据集用于迁移学习

- MICCAI-21 [Domain Generalization for Mammography Detection via Multi-style and Multi-view Contrastive Learning](https://arxiv.org/abs/2111.10827)
    - Domain generalization for mammography detection
    - 领域泛化用于乳房X射线检查

- [Action Recognition using Transfer Learning and Majority Voting for CSGO](https://arxiv.org/abs/2111.03882)
    - Using transfer learning and majority voting for action recognition
    - 使用迁移学习和多数投票进行动作识别

- [C-MADA: Unsupervised Cross-Modality Adversarial Domain Adaptation framework for medical Image Segmentation](https://arxiv.org/abs/2110.15823)
    - Cross-modality domain adaptation for medical image segmentation
    - 跨模态的DA用于医学图像分割

- BMVC-21 [SILT: Self-supervised Lighting Transfer Using Implicit Image Decomposition](https://arxiv.org/abs/2110.12914)
    - Lighting transfer using implicit image decomposition
    - 用隐式图像分解进行光照迁移

- [Domain Adaptation in Multi-View Embedding for Cross-Modal Video Retrieval](https://arxiv.org/abs/2110.12812)
  - Domain adaptation for cross-modal video retrieval
  - 用领域自适应进行跨模态的视频检索

- [Age and Gender Prediction using Deep CNNs and Transfer Learning](https://arxiv.org/abs/2110.12633)
  - Age and gender prediction using transfer learning
  - 用迁移学习进行年龄和性别预测

- WACV-22 [AuxAdapt: Stable and Efficient Test-Time Adaptation for Temporally Consistent Video Semantic Segmentation](https://arxiv.org/abs/2110.12369)
    - Test-time adaptation for video semantic segmentation
    - 测试时adaptation用于视频语义分割

- NeurIPS-21 [FlexMatch: Boosting Semi-Supervised Learning with Curriculum Pseudo Labeling](https://arxiv.org/abs/2110.08263) [知乎解读](https://zhuanlan.zhihu.com/p/422930830) [code](https://github.com/TorchSSL/TorchSSL)
    - Curriculum pseudo label with a unified codebase TorchSSL
    - 半监督方法FlexMatch和统一算法库TorchSSL

- ICCV-21 [BiMaL: Bijective Maximum Likelihood Approach to Domain Adaptation in Semantic Scene Segmentation](https://arxiv.org/abs/2108.03267)
    - Bijective MMD for domain adaptation
    - 双射MMD用于语义分割

- CVPR-21 [Ego-Exo: Transferring Visual Representations From Third-Person to First-Person Videos](https://openaccess.thecvf.com/content/CVPR2021/html/Li_Ego-Exo_Transferring_Visual_Representations_From_Third-Person_to_First-Person_Videos_CVPR_2021_paper.html)
    - Transfer learning from third-person to first-person video
    - 从第三人称视频迁移到第一人称

- 20210511 [Adaptive Domain-Specific Normalization for Generalizable Person Re-Identification](https://arxiv.org/abs/2105.03042)
    - Adaptive domain-specific normalization for generalizable ReID
    - 自适应的领域特异归一化用于ReID

- 20210220 [DRIV100: In-The-Wild Multi-Domain Dataset and Evaluation for Real-World Domain Adaptation of Semantic Segmentation](http://arxiv.org/abs/2102.00150)
    - A new dataset for domain adaptation on semantic segmentation
    - 一个用于domain adaptation做语义分割的新数据集

- 20210127 [Transferable Interactiveness Knowledge for Human-Object Interaction Detection](http://arxiv.org/abs/2101.10292)
    - A transferable HOI model
    - 一个可迁移的人-物交互检测模型

- 20201208 [Domain Adaptation of Aerial Semantic Segmentation](http://arxiv.org/abs/2012.02264)
    - 用domain adaptation做航空图像分割

- 20200420 AAAI-20 [Generative Adversarial Networks for Video-to-Video Domain Adaptation](https://arxiv.org/abs/2004.08058)
  	- Using GAN for video domain adaptation
  	- GAN用于视频到视频的adaptation

- 20191029 WACV-20 [Progressive Domain Adaptation for Object Detection](https://arxiv.org/abs/1910.11319)
  	- Progressive domain adaptation for object recognition
  	- 渐进式的DA用于物体检测

- 20191011 ICIP-19 [Cross-modal knowledge distillation for action recognition](https://arxiv.org/abs/1910.04641)
  	- Cross-modal knowledge distillation for action recognition
  	- 跨模态的知识蒸馏并用于动作识别

- 20191008 arXiv, ICCV-19 demo [Cross-Domain Complementary Learning with Synthetic Data for Multi-Person Part Segmentation](https://arxiv.org/abs/1907.05193)
  	- Learning human body part segmentation without human labeling 
  	- 基於合成數據的跨域互補學習人體部位分割

- 20191008 ICONIP-19 [Semi-Supervised Domain Adaptation with Representation Learning for Semantic Segmentation across Time](https://arxiv.org/abs/1805.04141)
  	- Semi-supervised domain adaptation with representation learning for semantic segmentation
  	- 半监督DA用于语义分割

- 20190926 arXiv [Restyling Data: Application to Unsupervised Domain Adaptation](https://arxiv.org/abs/1909.10900)
  	- Restyle data using domain adaptation
  	- 使用domain adaptation进行风格迁移

- 20180828 ICCV-19 workshop [Unsupervised Deep Feature Transfer for Low Resolution Image Classification](https://arxiv.org/abs/1908.10012)
  	- Deep feature transfer for low resolution image classification
  	- 深度特征迁移用于低分辨率图像分类

- 20190809 IJCAI-19 [Progressive Transfer Learning for Person Re-identification](https://arxiv.org/abs/1908.02492)
  	- Progressive transfer learning for RE_ID
  	- 渐进式迁移学习用于RE_ID

- 20190703 arXiv [Disentangled Makeup Transfer with Generative Adversarial Network](https://arxiv.org/abs/1907.01144)
  	- Makeup transfer with GAN
  	- 用GAN进行化妆的迁移

- 20190509 arXiv [Unsupervised Domain Adaptation using Generative Adversarial Networks for Semantic Segmentation of Aerial Images](https://arxiv.org/abs/1905.03198)
  	- Domain adaptation for semantic segmentation in aerial images
  	- DA应用于鸟瞰图像语义分割

- 20190415 PAKDD-19 [Adaptively Transfer Category-Classifier for Handwritten Chinese Character Recognition](https://link.springer.com/chapter/10.1007/978-3-030-16148-4_9)
  	- Transfer learning for handwritten Chinese character recognition
  	- 用迁移学习进行中文手写体识别

- 20190409 arXiv [Unsupervised Domain Adaptation for Multispectral Pedestrian Detection](https://arxiv.org/abs/1904.03692)
    - Domain adaptation for pedestrian detection
    - 无监督领域自适应用于多模态行人检测

- 20190305 arXiv [Unsupervised Domain Adaptation Learning Algorithm for RGB-D Staircase Recognition](https://arxiv.org/abs/1903.01212)
    - Domain adaptation for RGB-D staircase recognition
    - Domain adaptation进行深度和RGB楼梯识别

- 20190123 arXiv [Adapting Convolutional Neural Networks for Geographical Domain Shift](https://arxiv.org/abs/1901.06345)
  	- Convolutional neural network for geographical domain shift
  	- 将卷积网络用于地理学上的domain shift问题

- 20190115 IJAERS [Weightless Neural Network with Transfer Learning to Detect Distress in Asphalt](https://arxiv.org/abs/1901.03660)
    - Transfer learning to detect distress in asphalt
    - 用迁移学习检测路面情况

- 20190102 arXiv [High Quality Monocular Depth Estimation via Transfer Learning](https://arxiv.org/abs/1812.11941)
    - Monocular depth estimation using transfer learning
    - 用迁移学习进行单眼深度估计

- 20181213 arXiv [Multichannel Semantic Segmentation with Unsupervised Domain Adaptation](https://arxiv.org/abs/1812.04351)
    - Robot vision semantic segmentation with domain adaptation
    - 用于机器视觉中语义分割的domain adaptation

- 20181212 arXiv [3D Scene Parsing via Class-Wise Adaptation](https://arxiv.org/abs/1812.03622)
    - Class-wise adaptation for 3D scene parsing
    - 类别的适配用于3D场景分析

- 20181130 arXiv [Identity Preserving Generative Adversarial Network for Cross-Domain Person Re-identification](https://arxiv.org/abs/1811.11510)
	- Cross-domain reID
	- 跨领域的行人再识别

- 20181128 arXiv [Cross-domain Deep Feature Combination for Bird Species Classification with Audio-visual Data](https://arxiv.org/abs/1811.10199)
	- Cross-domain deep feature combination for bird species classification
	- 跨领域的鸟分类

- 20181128 WACV-19 [CNN based dense underwater 3D scene reconstruction by transfer learning using bubble database](https://arxiv.org/abs/1811.09675)
	- Transfer learning for underwater 3D scene reconstruction
	- 用迁移学习进行水下3D场景重建

- 20181128 arXiv [Low-resolution Face Recognition in the Wild via Selective Knowledge Distillation](https://arxiv.org/abs/1811.09998)
	-  Knowledge distilation for low-resolution face recognition
	- 将知识蒸馏应用于低分辨率的人脸识别

- 20181121 arXiv [Distribution Discrepancy Maximization for Image Privacy Preserving](https://arxiv.org/abs/1811.07335)
    - Distribution Discrepancy Maximization for Image Privacy Preserving
    - 通过最大化分布差异来进行图片隐私保护

- 20181114 arXiv [A Framework of Transfer Learning in Object Detection for Embedded Systems](https://arxiv.org/abs/1811.04863)
	- Transfer learning in embedded system for object detection
	- 在嵌入式系统中进行针对目标检测的迁移学习

- 20181012 arXiv [Bird Species Classification using Transfer Learning with Multistage Training](https://arxiv.org/abs/1810.04250)
	- Using transfer learning for bird species classification
	- 用迁移学习进行鸟类分类

- 20180912 ICIP-18 [Adversarial Domain Adaptation with a Domain Similarity Discriminator for Semantic Segmentation of Urban Areas](https://ieeexplore.ieee.org/abstract/document/8451010/)
    - Semantic segmentation using transfer learning
    - 用迁移学习进行语义分割

- 20180912 arXiv [Tensor Alignment Based Domain Adaptation for Hyperspectral Image Classification](https://arxiv.org/abs/1808.09769)
    - Hyperspectral image classification using domain adaptation
    - 用domain adaptation进行图像分类

- 20180904 ICPR-18 [Document Image Classification with Intra-Domain Transfer Learning and Stacked Generalization of Deep Convolutional Neural Networks](https://arxiv.org/abs/1801.09321)
	- Document image classification using transfer learning
	- 使用迁移学习进行文档图像的分类

- 20180826 ISPRS journal [Deep multi-task learning for a geographically-regularized semantic segmentation of aerial images](https://arxiv.org/abs/1808.07675)
	- a multi-task learning network for remote sensing
	- 提出一个多任务的深度网络用于遥感图像检测

- 20180819 arXiv [Transfer Learning and Organic Computing for Autonomous Vehicles](https://arxiv.org/abs/1808.05443)
  - Propose different transfer learning methods to adapt the situation of autonomous driving
  - 提出一些不同的迁移学习方法应用于自动驾驶的环境适配

- 20180801 ECCV-18 [DOCK: Detecting Objects by transferring Common-sense Knowledge](https://arxiv.org/abs/1804.01077)
  - A method called DOCK for object detection using transfer learning
  - 提出一个叫做DOCK的方法进行基于迁移学习的目标检测

- 20180801 ECCV-18 [A Zero-Shot Framework for Sketch-based Image Retrieval](https://arxiv.org/abs/1807.11724)
  - A Zero-Shot Framework for Sketch-based Image Retrieval
  - 一个针对于简笔画图像检索的zero-shot框架

- 20180731 ICANN-18 [Metric Embedding Autoencoders for Unsupervised Cross-Dataset Transfer Learning](https://arxiv.org/abs/1807.10591)
  - Deep transfer learning for Re-ID
  - 将深度迁移学习用于Re-ID

- 20180627 arXiv 生成模型用于姿态迁移：[Generative Models for Pose Transfer](https://arxiv.org/abs/1806.09070)

- 20180622 arXiv 跨领域的人脸识别用于银行认证系统：[Cross-Domain Deep Face Matching for Real Banking Security Systems](https://arxiv.org/abs/1806.07644)

- 20180614 arXiv 跨数据集的person reid：[Cross-dataset Person Re-Identification Using Similarity Preserved Generative Adversarial Networks](https://arxiv.org/abs/1806.04533)

- 20180610 CEIG-17 将迁移学习用于插图分类：[Transfer Learning for Illustration Classification](https://arxiv.org/abs/1806.02682)

- 20180519 arXiv 用迁移学习进行物体检测，200帧/秒：[Object detection at 200 Frames Per Second](https://arxiv.org/abs/1804.04775)

- 20180519 arXiv 用迁移学习进行肢体语言识别：[Optimization of Transfer Learning for Sign Language Recognition Targeting Mobile Platform](https://arxiv.org/abs/1805.06618)

- 20180427 CVPR-18(workshop) 将深度迁移学习用于Person-reidentification： [Adaptation and Re-Identification Network: An Unsupervised Deep Transfer Learning Approach to Person Re-Identification](https://arxiv.org/abs/1804.09347)

- 20180410 arXiv 用迁移学习进行犯罪现场的图像匹配：[Cross-Domain Image Matching with Deep Feature Maps](https://arxiv.org/abs/1804.02367)

- 20180408 arXiv 小数据集上的迁移学习手写体识别：[Boosting Handwriting Text Recognition in Small Databases with Transfer Learning](https://arxiv.org/abs/1804.01527)

- 20180404 arXiv 用迁移学习进行物体检测：[Transferring Common-Sense Knowledge for Object Detection](https://arxiv.org/abs/1804.01077)


## Medical and healthcare

- Unsupervised Domain Adaptation for COVID-19 Information Service with Contrastive Adversarial Domain Mixup [[arxiv](https://arxiv.org/abs/2210.03250)]
  - Domain adaptation for COVID-19 用DA进行COVID-19预测

- FL-IJCAI-22 [MetaFed: Federated Learning among Federations with Cyclic Knowledge Distillation for Personalized Healthcare](https://arxiv.org/abs/2206.08516)
  - MetaFed: a new form of federated learning 联邦之联邦学习、新范式

- [Parkinson's disease diagnostics using AI and natural language knowledge transfer](https://arxiv.org/abs/2204.12559)
  - Transfer learning for Parkinson's disease diagnostics 迁移学习用于帕金森诊断

- [Federated Learning with Adaptive Batchnorm for Personalized Healthcare](https://arxiv.org/abs/2112.00734)
    - Federated learning with adaptive batchnorm
    - 用自适应BN进行个性化联邦学习

- [Adversarial Domain Feature Adaptation for Bronchoscopic Depth Estimation](https://arxiv.org/abs/2109.11798)
    - Adversarial domain adaptation for bronchoscopic depth estimation
    - 用对抗领域自适应进行支气管镜的深度估计

- [Domain and Content Adaptive Convolution for Domain Generalization in Medical Image Segmentation](https://arxiv.org/abs/2109.05676)
    - Domain generalization for medical image segmentation
    - 领域泛化用于医学图像分割

- [Unsupervised domain adaptation for cross-modality liver segmentation via joint adversarial learning and self-learning](https://arxiv.org/abs/2109.05664)
    - Domain adaptation for cross-modality liver segmentation
    - 使用domain adaptation进行肝脏的跨模态分割

- MICCAI-21 [A Systematic Benchmarking Analysis of Transfer Learning for Medical Image Analysis](https://arxiv.org/abs/2108.05930)
    - A benchmark of transfer learning for medical image
    - 一个详细的迁移学习用于医学图像的benchmark

- [A Data Augmented Approach to Transfer Learning for Covid-19 Detection](https://arxiv.org/abs/2108.02870)
    - Data augmentation to transfer learning for COVID
    - 迁移学习使用数据增强，用于COVID-19

- [Transfer Learning in Electronic Health Records through Clinical Concept Embedding](https://arxiv.org/abs/2107.12919)
  - Transfer learning in electronic health record
  - 迁移学习用于医疗记录管理

- 20210607 [FedHealth 2: Weighted Federated Transfer Learning via Batch Normalization for Personalized Healthcare](https://arxiv.org/abs/2106.01009)
    - Federated transfer learning framework 2
    - FedHealth联邦迁移学习框架第二代

- 20201222 [Transfer Learning Through Weighted Loss Function and Group Normalization for Vessel Segmentation from Retinal Images](http://arxiv.org/abs/2012.09250)
  - Transfer learning for vessel segmentation from retinal images
  - 迁移学习用于视网膜血管分割

- 20201215 [Distant Domain Transfer Learning for Medical Imaging](https://arxiv.org/abs/2012.06346)
    - Distant domain transfer learning for medical imaging
    - 用于COVID检测的远领域迁移学习

- 20201215 AAAI-21 [Transfer Graph Neural Networks for Pandemic Forecasting](https://arxiv.org/abs/2009.08388)
    - GNN and transfer learning for pandemic forecasting
    - 用基于GNN的迁移学习进行流行病预测

- 20201116 [A Study of Domain Generalization on Ultrasound-based Multi-Class Segmentation of Arteries, Veins, Ligaments, and Nerves Using Transfer Learning](https://arxiv.org/abs/2011.07019)
    - Domain generalization用于医学分类

- 20200927 [Transfer Learning by Cascaded Network to identify and classify lung nodules for cancer detection](https://arxiv.org/abs/2009.11587)
    - 迁移学习用于肺癌检测

- 20200813 [Transfer Learning for Protein Structure Classification and Function Inference at Low Resolution](https://arxiv.org/abs/2008.04757)
    - 迁移学习用于低分辨率下的蛋白质结构分类

- 20191111 NIPS-19 workshop [Transfer Learning in 4D for Breast Cancer Diagnosis using Dynamic Contrast-Enhanced Magnetic Resonance Imaging](https://arxiv.org/abs/1911.03022)
  	- Transfer learning in 4D for breast cancer diagnosis

- 20191029 arXiv [NER Models Using Pre-training and Transfer Learning for Healthcare](https://arxiv.org/abs/1910.11241)
  	- Pretraining NER models for healthcare
  	- 预训练的NER模型用于健康监护

- 20191008 arXiv [Transfer Brain MRI Tumor Segmentation Models Across Modalities with Adversarial Networks](https://arxiv.org/abs/1910.02717)
  	- Transfer learning for multi-modal brain MRI tumor segmentation
  	- 用迁移学习进行多模态的MRI肿瘤分割

- 20191008 arXiv [Noise as Domain Shift: Denoising Medical Images by Unpaired Image Translation](https://arxiv.org/abs/1910.02702)
  	- Noise as domain shift for medical images
  	- 医学图像中的噪声进行adaptation

- 20190912 MICCAI workshop [Multi-Domain Adaptation in Brain MRI through Paired Consistency and Adversarial Learning](https://arxiv.org/abs/1908.05959)
  	- Multi-domain adaptation for brain MRI
  	- 多领域的adaptation用于大脑MRI识别

- 20190909 IJCAI-FML-19 [FedHealth: A Federated Transfer Learning Framework for Wearable Healthcare](http://jd92.wang/assets/files/a15_ijcai19.pdf)
  	- The first work on federated transfer learning for wearable healthcare
  	- 第一个将联邦迁移学习用于可穿戴健康监护的工作

- 20190828 MICCAI-19 workshop [Cross-modality Knowledge Transfer for Prostate Segmentation from CT Scans](https://arxiv.org/abs/1908.10208)
  	- Cross-modality transfer for prostate segmentation
  	- 跨模态的迁移用于前列腺分割

- 20190802 arXiv [Towards More Accurate Automatic Sleep Staging via Deep Transfer Learning](https://arxiv.org/abs/1907.13177)
  	- Accurate Sleep Staging with deep transfer learning
  	- 用深度迁移学习进行精准的睡眠阶段估计

- 20190729 MICCAI-19 [Annotation-Free Cardiac Vessel Segmentation via Knowledge Transfer from Retinal Images](https://arxiv.org/abs/1907.11483)
  	- Cardiac vessel segmentation using transfer learning from Retinal Images
  	- 用视网膜图片进行迁移学习用于心脏血管分割

- 20190703 arXiv [Applying Transfer Learning To Deep Learned Models For EEG Analysis](https://arxiv.org/abs/1907.01332)
  	- Apply transfer learning to EEG
  	- 用深度迁移学习进行EEG分析

- 20190626 arXiv [A Novel Deep Transfer Learning Method for Detection of Myocardial Infarction](https://arxiv.org/abs/1906.09358)
  	- A deep transfer learning method for detecting myocardial infarction
  	- 一种用于监测心肌梗塞的深度迁移方法

- 20190416 arXiv [Deep Transfer Learning for Single-Channel Automatic Sleep Staging with Channel Mismatch](https://arxiv.org/abs/1904.05945)
  	- Using deep transfer learning for sleep stage recognition
  	- 用深度迁移学习进行睡眠阶段的检测

- 20190403 arXiv [Transfer Learning for Clinical Time Series Analysis using Deep Neural Networks](https://arxiv.org/abs/1904.00655)
    - Using transfer learning for multivariate clinical data
    - 使用迁移学习进行多元医疗数据迁移

- 20190403 arXiv [Med3D: Transfer Learning for 3D Medical Image Analysis](https://arxiv.org/abs/1904.00625)
    - Transfer learning for 3D medical image analysis
    - 迁移学习用于3D医疗图像分析

- 20190221 arXiv [Transfusion: Understanding Transfer Learning with Applications to Medical Imaging](https://arxiv.org/abs/1902.07208)
    - Analyzing the influence of transfer learning in medical imaging
    - 在医疗图像中分析迁移学习作用

- 20190117 NeurIPS-18 workshop [Transfer Learning for Prosthetics Using Imitation Learning](https://arxiv.org/abs/1901.04772)
    - Using transfer learning for prosthetics
    - 用迁移学习进行义肢的模仿学习

- 20190115 arXiv [Disease Knowledge Transfer across Neurodegenerative Diseases](https://arxiv.org/abs/1901.03517)
    - Transfer learning for neurodegenerative disease
    - 迁移学习用于神经退行性疾病

- 20181225 arXiv [An Integrated Transfer Learning and Multitask Learning Approach for Pharmacokinetic Parameter Prediction](https://arxiv.org/abs/1812.09073)
    - Using transfer learning for Pharmacokinetic Parameter Prediction
    - 用迁移学习进行药代动力学参数估计

- 20181221 arXiv [PnP-AdaNet: Plug-and-Play Adversarial Domain Adaptation Network with a Benchmark at Cross-modality Cardiac Segmentation](https://arxiv.org/abs/1812.07907)
    - Adversarial transfer learning for medical images
    - 对抗迁移学习用于医学图像分割

- 20181214 BioCAS-19 [ECG Arrhythmia Classification Using Transfer Learning from 2-Dimensional Deep CNN Features](https://arxiv.org/abs/1812.04693)
    - Deep transfer learning for EEG Arrhythmia Classification
    - 深度迁移学习用于心率不齐分类

- 20181206 NeurIPS-18 workshop [Towards Continuous Domain adaptation for Healthcare](https://arxiv.org/abs/1812.01281)
	- Continuous domain adaptation for healthcare
	- 连续的domain adaptation用于健康监护

- 20181206 NeurIPS-18 workshop [A Hybrid Instance-based Transfer Learning Method](https://arxiv.org/abs/1812.01063)
	- Instance-based transfer learning for healthcare
	- 基于样本的迁移学习用于健康监护

- 20181205 arXiv [Learning from a tiny dataset of manual annotations: a teacher/student approach for surgical phase recognition](https://arxiv.org/abs/1812.00033)
	- Transfer learning for surgical phase recognition
	- 迁移学习用于外科手术阶段识别

- 20181128 NeurIPS-18 workshop [Multi-Task Generative Adversarial Network for Handling Imbalanced Clinical Data](https://arxiv.org/abs/1811.10419)
	- Multi-task learning for imbalanced clinical data
	- 多任务学习用于不平衡的就诊数据

- 20181127 NeurIPS-18 workshop [Predicting Diabetes Disease Evolution Using Financial Records and Recurrent Neural Networks](https://arxiv.org/abs/1811.09350)
    - Predicting diabetes using financial records
    - 用财务记录预测糖尿病

- 20181123 NIPS-18 workshop [Population-aware Hierarchical Bayesian Domain Adaptation](https://arxiv.org/abs/1811.08579)
	- Applying domain adaptation to health
	- 将domain adaptation应用于健康

- 20181121 arXiv [Transferrable End-to-End Learning for Protein Interface Prediction](https://arxiv.org/abs/1807.01297)
    - Transfer learning for protein interface prediction
    - 用迁移学习进行蛋白质接口预测

- 20181117 arXiv [Unsupervised domain adaptation for medical imaging segmentation with self-ensembling](https://arxiv.org/abs/1811.06042)
	- Medical imaging using transfer learning
	- 使用迁移学习进行医学图像分割

- 20181117 AAAI-19 [GaitSet: Regarding Gait as a Set for Cross-View Gait Recognition](https://arxiv.org/abs/1811.06186)
	- Cross-view gait recognition
	- 跨视图的步态识别

- 20181012 arXiv [Survival prediction using ensemble tumor segmentation and transfer learning](https://arxiv.org/abs/1810.04274)
	- Predicting the survival of the tumor patient using transfer learning
	- 用迁移学习估计肿瘤病人存活时间

- 20180904 EMBC-18 [Multi-Cell Multi-Task Convolutional Neural Networks for Diabetic Retinopathy Grading Kang](https://arxiv.org/abs/1808.10564)
	- Use multi-task CNN for Diabetic Retinopathy Grading Kang
	- 用多任务的CNN进行糖尿病的视网膜粒度检查

- 20180823 ICPR-18 [Multi-task multiple kernel machines for personalized pain recognition from functional near-infrared spectroscopy brain signals](https://arxiv.org/abs/1808.06774)
  - A multi-task method to recognize pains
  - 提出一个multi-task框架来检测pain

- 20180801 MICCAI-18 [Leveraging Unlabeled Whole-Slide-Images for Mitosis Detection](https://arxiv.org/abs/1807.11677)
  - Use unlabeled images for mitosis detection
  - 用未标记的图片进行细胞有丝分裂的检测

- 20180627 arXiv 用迁移学习进行感染预测：[Domain Adaptation for Infection Prediction from Symptoms Based on Data from Different Study Designs and Contexts](https://arxiv.org/abs/1806.08835)

- 20180621 arXiv 迁移学习用于角膜组织的分类：[Transfer Learning with Human Corneal Tissues: An Analysis of Optimal Cut-Off Layer](https://arxiv.org/abs/1806.07073)

- 20180612 KDD-18 多任务学习用于ICU病人数据挖掘：[Learning Tasks for Multitask Learning: Heterogenous Patient Populations in the ICU](https://arxiv.org/abs/1806.02878)

- 20180610 BioNLP-18 将迁移学习用于病人实体分类：[Embedding Transfer for Low-Resource Medical Named Entity Recognition: A Case Study on Patient Mobility](https://arxiv.org/abs/1806.02814)

- 20180610 MICCAI-18 将迁移学习用于前列腺图分类：[Adversarial Domain Adaptation for Classification of Prostate Histopathology Whole-Slide Images](https://arxiv.org/abs/1806.01357)

- 20180605 arXiv 迁移学习应用于胸X光片分割：[Semantic-Aware Generative Adversarial Nets for Unsupervised Domain Adaptation in Chest X-ray Segmentation](https://arxiv.org/abs/1806.00600)

- 20180604 arXiv 用CNN迁移学习进行硬化症检测：[One-shot domain adaptation in multiple sclerosis lesion segmentation using convolutional neural networks](https://arxiv.org/abs/1805.12415)

- 20180504 arXiv 用迁移学习进行心脏病检测分类：[ECG Heartbeat Classification: A Deep Transferable Representation](https://arxiv.org/abs/1805.00794)

- 20180426 arXiv 迁移学习用于医学名字实体检测；[Label-aware Double Transfer Learning for Cross-Specialty Medical Named Entity Recognition](https://arxiv.org/abs/1804.09021)

- 20180402 arXiv 将迁移学习用于癌症检测：[Improve the performance of transfer learning without fine-tuning using dissimilarity-based multi-view learning for breast cancer histology images](https://arxiv.org/abs/1803.11241)

## Natural language processing

- GLUE-X: Evaluating Natural Language Understanding Models from an Out-of-distribution Generalization Perspective [[arxiv](https://arxiv.org/abs/2211.08073)]
  - OOD for natural language processing evaluation 提出GLUE-X用于OOD在NLP数据上的评估

- Robust Domain Adaptation for Machine Reading Comprehension [[arxiv](https://arxiv.org/abs/2209.11615)]
  - Domain adaptation for machine reading comprehension 机器阅读理解的domain adaptation

- NAACL-22 [Modularized Transfer Learning with Multiple Knowledge Graphs for Zero-shot Commonsense Reasoning](https://arxiv.org/abs/2206.03715)
  - Transfer learning for zero-shot reasoning 迁移学习用于零次常识推理

- ICLR-22 [Enhancing Cross-lingual Transfer by Manifold Mixup](https://arxiv.org/abs/2205.04182)
  - Cross-lingual transfer using manifold mixup 用Mixup进行cross-lingual transfer

- NAACL-22 [Efficient Few-Shot Fine-Tuning for Opinion Summarization](https://arxiv.org/abs/2205.02170)
  - Few-shot fine-tuning for opinion summarization 小样本微调技术用于评论总结

- NAACL-22 [GRAM: Fast Fine-tuning of Pre-trained Language Models for Content-based Collaborative Filtering](https://arxiv.org/abs/2204.04179)
  - Fast fine-tuning for content-based collaborative filtering
  - 快速的适用于协同过滤的微调

- [One Model, Multiple Tasks: Pathways for Natural Language Understanding](https://arxiv.org/abs/2203.03312)
  - Pathways for natural language understanding
  - 使用一个model用于所有NLP任务

- ACL-22 [Investigating Selective Prediction Approaches Across Several Tasks in IID, OOD, and Adversarial Settings](https://arxiv.org/abs/2203.00211)
  - Investigate selective prediction approaches in IID, OOD, and ADV settings
  - 在独立同分布、分布外、对抗情境中调研选择性预测方法

- [IGLUE: A Benchmark for Transfer Learning across Modalities, Tasks, and Languages](https://arxiv.org/abs/2201.11732)
  - A benchmark for transfer learning in NLP
  - 一个用于NLP跨模态、任务、语言的benchmark

- [Deep Transfer Learning for Multi-source Entity Linkage via Domain Adaptation](https://arxiv.org/abs/2110.14509)
    - Domain adaptation for multi-source entiry linkage
    - 用DA进行多源的实体链接

- EMNLP-21 [Non-Parametric Unsupervised Domain Adaptation for Neural Machine Translation](https://arxiv.org/abs/2109.06604)
  - UDA for machine translation
  - 用领域自适应进行机器翻译

- EMNLP-21 [Few-Shot Intent Detection via Contrastive Pre-Training and Fine-Tuning](https://arxiv.org/abs/2109.06349)
    - Few-shot intent detection using pretrain and finetune
    - 用迁移学习进行少样本意图检测

- [Contrastive Domain Adaptation for Question Answering using Limited Text Corpora](https://arxiv.org/abs/2108.13854)
    - Contrastive domain adaptation for QA
    - QA任务中应用对比domain adaptation

- SemDIAL-21 [Generating Personalized Dialogue via Multi-Task Meta-Learning](https://arxiv.org/abs/2108.03377)
    - Generate personalized dialogue using multi-task meta-learning
    - 用多任务元学习生成个性化的对话

- 20210607 [Bilingual Alignment Pre-training for Zero-shot Cross-lingual Transfer](http://arxiv.org/abs/2106.01732)
  - Zero-shot cross-lingual transfer using bilingual alignment pretraining
  - 通过双语言进行对齐预训练进行零资源的跨语言迁移

- 20210607 [Pre-training Universal Language Representation](http://arxiv.org/abs/2105.14478)
  - Pretraining for universal language representation
  - 用统一的预训练进行语言表征建模

- 20210516 [A cost-benefit analysis of cross-lingual transfer methods](https://arxiv.org/abs/2105.06813)
  - Analysis of the running time of cross-lingual transfer
  - 分析了跨语言迁移方法的时间


- 20210420 arXiv [Domain Adaptation and Multi-Domain Adaptation for Neural Machine Translation: A Survey](https://arxiv.org/abs/2104.06951)
    - A survey on domain adaptation for machine translation
    - 关于用领域自适应进行神经机器翻译的综述

- 20210202 [Transfer Learning Approach for Detecting Psychological Distress in Brexit Tweets](https://arxiv.org/abs/2102.00912)
    - 检测英国脱欧twitter中的心理压力

- 20210106 [Decoding Time Lexical Domain Adaptation for Neural Machine Translation](http://arxiv.org/abs/2101.00421)
    - DA for NMT
    - DA用于机器翻译任务上

- 20210104 [A Closer Look at Few-Shot Crosslingual Transfer: Variance, Benchmarks and Baselines](http://arxiv.org/abs/2012.15682)
    - A closer look at few-shot crosslingual transfer

- 20201215 AAAI-21 [Multilingual Transfer Learning for QA Using Translation as Data Augmentation](https://arxiv.org/abs/2012.05958)
    - Multilingual transfer learning for QA
    - 用于QA任务的多语言迁移学习

- 20201208 [Fine-tuning BERT for Low-Resource Natural Language Understanding via Active Learning](http://arxiv.org/abs/2012.02462)
    - 用BERT结合主动学习进行低资源的NLP任务

- 20200927 EMNLP-20 [Feature Adaptation of Pre-Trained Language Models across Languages and Domains for Text Classification](https://arxiv.org/abs/2009.11538)
    - 跨语言和领域的预训练模型用于文本分类

- 20200420 ACL-20 [Geometry-aware Domain Adaptation for Unsupervised Alignment of Word Embeddings](https://arxiv.org/abs/2004.08243)
  	- DA for unsupervised word embeddings alignment
  	- 领域自适应用于word embedding对齐

- 20191214 arXiv [Unsupervised Transfer Learning via BERT Neuron Selection](https://arxiv.org/abs/1912.05308)
     - Unsupervised transfer learning via BERT neuron selection

- 20191201 arXiv [A Transfer Learning Method for Goal Recognition Exploiting Cross-Domain Spatial Features](https://arxiv.org/abs/1911.10134)
     - A transfer learning method for goal recognition 
     - 用迁移学习分析语言中的目标

- 20191201 AAAI-20 [Zero-Resource Cross-Lingual Named Entity Recognition](https://arxiv.org/abs/1911.09812)
   - Zero-resource cross-lingual NER
   - 零资源的跨语言NER

- 20191115 arXiv [Instance-based Transfer Learning for Multilingual Deep Retrieval](https://arxiv.org/abs/1911.06111)
  	- Instance based transfer learning for multilingual deep retrieval
  	- 基于实例的迁移学习用于多语言的retrieval

- 20191115 arXiv [Unsupervised Pre-training for Natural Language Generation: A Literature Review](https://arxiv.org/abs/1911.06171)
  	- Unsupervised pre-training for natural language generation survey
  	- 一篇无监督预训练用于自然语言生成的综述

- 20191115 AAAI-20 [Unsupervised Domain Adaptation on Reading Comprehension](https://arxiv.org/abs/1911.06137)
  	- 无监督DA用于阅读理解
  	- Unsupervised DA for reading comprehension

- 20191113 arXiv [Open-Ended Visual Question Answering by Multi-Modal Domain Adaptation](https://arxiv.org/abs/1911.04058)
  	- Supervised multi-modal domain adaptation in VQA
  	- 有监督的多模态DA用于VQA任务

- 20191113 AAAI-20 [TANDA: Transfer and Adapt Pre-Trained Transformer Models for Answer Sentence Selection](https://arxiv.org/abs/1911.04118)
  	- Finetune twice for answer sentence selection
  	- 两次finetune用于answer sentence selection

- 20191113 arXiv [NegBERT: A Transfer Learning Approach for Negation Detection and Scope Resolution](https://arxiv.org/abs/1911.04211)
  	- Transfer learning for negation detection and scope resolution
  	- 迁移学习用于否定检测

- 20191111 arXiv [Unsupervised Domain Adaptation of Contextual Embeddings for Low-Resource Duplicate Question Detection](https://arxiv.org/abs/1911.02645)
  	- Unsupervised DA for low-resource duplicate question detection

- 20191111 arXiv [SMART: Robust and Efficient Fine-Tuning for Pre-trained Natural Language Models through Principled Regularized Optimization](https://arxiv.org/abs/1911.03437)
  	- Fine-tuning for pre-trained language model

- 20191111 arXiv [Towards Domain Adaptation from Limited Data for Question Answering Using Deep Neural Networks](https://arxiv.org/abs/1911.02655)
  	- DA for question answering using DNN

- 20191101 arXiv [Transferable End-to-End Aspect-based Sentiment Analysis with Selective Adversarial Learning](https://arxiv.org/abs/1910.14192)
  	- Adversarial transfer learning for aspect-based sentement analysis
  	- 对抗迁移用于aspect层级的情感分析

- 20191101 [Transfer Learning from Transformers to Fake News Challenge Stance Detection (FNC-1) Task](https://arxiv.org/abs/1910.14353)
  	- A fake news challenges based on transformers
  	- 一个基于transformer的假新闻检测挑战

- 20191029 WSDM-20 [Meta-Learning with Dynamic-Memory-Based Prototypical Network for Few-Shot Event Detection](https://arxiv.org/abs/1910.11621)
  	- Meta learning with dynamic memory based prototypical network for few-shot event detection

- 20191017 arXiv [Evolution of transfer learning in natural language processing](https://arxiv.org/abs/1910.07370)
  	- Survey transfer learning works in NLP
  	- 综述了最近迁移学习在NLP的一些进展

- 20191015 arXiv [Emotion Recognition in Conversations with Transfer Learning from Generative Conversation Modeling](https://arxiv.org/abs/1910.04980)
  	- Emotion recognition in conversations with transfer learning
  	- 用迁移学习进行对话中的情绪识别

- 20191011 NeurIPS-19 [Unified Language Model Pre-training for Natural Language Understanding and Generation](https://arxiv.org/abs/1905.03197)
  	- Unified language model pre-training for understanding and generation
  	- 统一的语言模型预训练用于自然语言理解和生成

- 20191011 NeurIPS-19 workshop [Language Transfer for Early Warning of Epidemics from Social Media](https://arxiv.org/abs/1910.04519)
  	- Language transfer to predict epidemics from social media
  	- 通过社交网络数据预测传染病并进行语言模型的迁移

- 20190829 EMNLP-19 [Investigating Meta-Learning Algorithms for Low-Resource Natural Language Understanding Tasks](https://arxiv.org/abs/1908.10423)
  	- Investigating MAML for low-resource NMT
  	- 调查了MAML方法用于低资源的NMT问题的表现

- 20190829 EMNLP-19 [Unsupervised Domain Adaptation for Neural Machine Translation with Domain-Aware Feature Embeddings](https://arxiv.org/abs/1908.10430)
  	- Domain adaptation for NMT

- 20190821 arXiv [Shallow Domain Adaptive Embeddings for Sentiment Analysis](https://arxiv.org/abs/1908.06082)
  	- Domain adaptative embedding for sentiment analysis
  	- 迁移学习用于情感分类

- 20190809 ICCASP-19 [Cross-lingual Text-independent Speaker Verification using Unsupervised Adversarial Discriminative Domain Adaptation](https://arxiv.org/abs/1908.01447)
  	- Text independent speaker verification using adversarial DA
  	- 文本无关的speaker verification用DA

- 20190809 NeurIPS-18 [MacNet: Transferring Knowledge from Machine Comprehension to Sequence-to-Sequence Models](https://arxiv.org/abs/1908.01816)
  	- Transfer learning from machine comprehension to sequence to senquence Models
  	- 从机器理解到序列模型迁移

- 20190515 ACL-19 [Effective Cross-lingual Transfer of Neural Machine Translation Models without Shared Vocabularies](https://arxiv.org/abs/1905.05475)
  	- Cross-lingual transfer of NMT
  	- 跨语言的NMT模型迁移

- 20190508 arXiv [On Transfer Learning For Chatter Detection in Turning Using Wavelet Packet Transform and Empirical Mode Decomposition](https://arxiv.org/abs/1905.01982)
  	- Transfer learning for chatter detection
  	- 用迁移学习进行叽叽喳喳聊天识别

- 20190415 PAKDD-19 [Multi-task Learning for Target-Dependent Sentiment Classification](https://link.springer.com/chapter/10.1007/978-3-030-16148-4_15)
  	- Multi-task learning for sentiment classification
  	- 用多任务学习进行任务依赖的情感分析

- 20190408 arXiv [Unsupervised Domain Adaptation of Contextualized Embeddings: A Case Study in Early Modern English](https://arxiv.org/abs/1904.02817)
    - Domain adaptation in early modern english
    - Case study: 在英文中的domain adaptation

- 20190111 ICMLA-18 [Supervised Transfer Learning for Product Information Question Answering](https://arxiv.org/abs/1901.02539)
    - Transfer learning for product information question answering
    - 利用迁移学习进行产品信息的对话

- 20181221 arXiv [Deep Transfer Learning for Static Malware Classification](https://arxiv.org/abs/1812.07606)
	- Deep Transfer Learning for Static Malware Classification
	- 用深度迁移学习进行恶意软件分类

- 20181204 arXiv [From Known to the Unknown: Transferring Knowledge to Answer Questions about Novel Visual and Semantic Concepts](https://arxiv.org/abs/1811.12772)
	- Transfer learning for VQA
	- 用迁移学习进行VQA任务

- 20181129 AAAI-19 [Exploiting Coarse-to-Fine Task Transfer for Aspect-level Sentiment Classification](https://arxiv.org/abs/1811.10999)
	-  Aspect-level sentiment classification
	- 迁移学习用于情感分类

- 20181115 AAAI-19 [Unsupervised Transfer Learning for Spoken Language Understanding in Intelligent Agents](https://arxiv.org/abs/1811.05232)
	- Transfer learning for spoken language understanding
	- 无监督迁移学习用于语言理解

- 20181107 ICONIP-18 [Transductive Learning with String Kernels for Cross-Domain Text Classification](https://arxiv.org/abs/1811.01734)
	- String kernel for cross-domain text classification using transfer learning
	- 用string kernel进行迁移学习跨领域文本分类

- 20180613 CVPR-18 跨数据集的VQA：[Cross-Dataset Adaptation for Visual Question Answering](https://arxiv.org/abs/1806.03726)

- 20180612 ICASSP-18 迁移学习用于资源少的情感分类：[Semi-supervised and Transfer learning approaches for low resource sentiment classification](https://arxiv.org/abs/1806.02863)

- 20180516 ACL-18 将对抗迁移学习用于危机状态下的舆情分析：[Domain Adaptation with Adversarial Training and Graph Embeddings](https://arxiv.org/abs/1805.05151)

- 20180425 arXiv 迁移学习应用于自然语言任务：[Dropping Networks for Transfer Learning](https://arxiv.org/abs/1804.08501)

- 20191214 NIPS-19 workshop [Cross-Language Aphasia Detection using Optimal Transport Domain Adaptation](https://arxiv.org/abs/1912.04370)
    - Optimal transport domain adaptation

## Time series

- ICLR'23 Out-of-distribution Representation Learning for Time Series Classification [[arxiv](https://arxiv.org/abs/2209.07027)]
  - OOD for time series classification 时间序列分类的OOD算法

- Domain Adaptation for Time Series Under Feature and Label Shifts [[arxiv](https://arxiv.org/abs/2302.03133)]
  - Domain adaptation for time series 用于时间序列的domain adaptation

- StyleTime: Style Transfer for Synthetic Time Series Generation [[arxiv](https://arxiv.org/abs/2209.11306)]
  - Style transfer for time series generation 时间序列生成的风格迁移

- Generalized representations learning for time series classification [[arxiv](https://arxiv.org/abs/2209.07027)]
  - OOD for time series classification 域泛化用于时间序列分类

- [Time-Series Domain Adaptation via Sparse Associative Structure Alignment: Learning Invariance and Variance](https://arxiv.org/abs/2205.03554)
  - Time series domain adaptation 时间序列domain adaptation

- [Domain Adversarial Spatial-Temporal Network: A Transferable Framework for Short-term Traffic Forecasting across Cities](https://arxiv.org/abs/2202.03630)
  - Transfer learning for traffic forecasting across cities
  - 用迁移学习进行跨城市的交通流量预测

- [Domain-Invariant Representation Learning from EEG with Private Encoders](https://arxiv.org/abs/2201.11613)
  - Domain-invariant learning from EEG
  - 用于EEG信号的领域不变特征研究

- KBS-22 [Intra-domain and cross-domain transfer learning for time series data -- How transferable are the features](https://arxiv.org/abs/2201.04449)
  - An overview of transfer learning for time series data
  - 一个用迁移学习进行时间序列分析的小综述

- CIKM-21 [AdaRNN: Adaptive Learning and Forecasting of Time Series](https://arxiv.org/abs/2108.04443) [Code](https://github.com/jindongwang/transferlearning/tree/master/code/deep/adarnn) [知乎文章](https://zhuanlan.zhihu.com/p/398036372) [Video](https://www.bilibili.com/video/BV1Gh411B7rj/)
    - A new perspective to using transfer learning for time series analysis
    - 一种新的建模时间序列的迁移学习视角

- TKDE-21 [Unsupervised Deep Anomaly Detection for Multi-Sensor Time-Series Signals](https://arxiv.org/abs/2107.12626)
    - Anomaly detection using semi-supervised and transfer learning
    - 半监督学习用于无监督异常检测

- 20190703 arXiv [Applying Transfer Learning To Deep Learned Models For EEG Analysis](https://arxiv.org/abs/1907.01332)
  	- Apply transfer learning to EEG
  	- 用深度迁移学习进行EEG分析

## Speech

- Interspeech-22 [Decoupled Federated Learning for ASR with Non-IID Data](https://jd92.wang/assets/files/DecoupleFL-IS22.pdf)
  - Decoupled federated learning for non IID 解耦的联邦架构用于Non-IID语音识别

- [A Likelihood Ratio based Domain Adaptation Method for E2E Models](2201.03655)
  - Domain adaptation for speech recognition
  - 用domain adaptation进行语音识别

- [Domain Prompts: Towards memory and compute efficient domain adaptation of ASR systems](https://arxiv.org/abs/2112.08718)
    - Prompt for domain adaptation in speech recognition
    - 用Prompt在语音识别中进行domain adaptation

- IEEE TASLP-22 [Exploiting Adapters for Cross-lingual Low-resource Speech Recognition](https://arxiv.org/abs/2105.11905)
    - Cross-lingual speech recogntion using meta-learning and transfer learning
    - 用元学习和迁移学习进行跨语言的低资源语音识别

- [Temporal Knowledge Distillation for On-device Audio Classification](https://arxiv.org/abs/2110.14131)
    - Temporal knowledge distillation for on-device ASR
    - 时序知识蒸馏用于设备端的语音识别

- [Music Sentiment Transfer](https://arxiv.org/abs/2110.05765)
    - Music sentiment transfer learning
    - 迁移学习用于音乐sentiment

- 20210716 InterSpeech-21 [Speech2Video: Cross-Modal Distillation for Speech to Video Generation](https://arxiv.org/abs/2107.04806)
  - Cross-model distillation for video generation
  - 跨模态蒸馏用于语音到video的生成

- 20210607 Interspeech-21 [Cross-domain Speech Recognition with Unsupervised Character-level Distribution Matching](https://arxiv.org/abs/2104.07491)
    - Domain adaptation for speech recognition
    - 用domain adaptation进行跨领域的语音识别

- 20201116 [Arabic Dialect Identification Using BERT-Based Domain Adaptation](https://arxiv.org/abs/2011.06977)
    - 用基于BERT的domain adaptation进行阿拉伯方言识别

- 20191124 [Cantonese Automatic Speech Recognition Using Transfer Learning from Mandarin](https://arxiv.org/abs/1911.09271)
  	- Cantonese speech recognition using transfer learning from mandarin
  	- 普通话语音识别迁移到广东话识别

- 20191111 arXiv [Teacher-Student Training for Robust Tacotron-based TTS](https://arxiv.org/abs/1911.02839)
  	- Teacher-student network for robust TTS

- 20191111 arXiv [Change your singer: a transfer learning generative adversarial framework for song to song conversion](https://arxiv.org/abs/1911.02933)
  	- Adversarial transfer learning for song-to-song conversion

- 20190828 arXiv [VAE-based Domain Adaptation for Speaker Verification](https://arxiv.org/abs/1908.10092)
  	- Speaker verification using VAE domain adaptation
  	- 基于VAE的speaker verification

- 20181230 arXiv [The CORAL+ Algorithm for Unsupervised Domain Adaptation of PLDA](https://arxiv.org/abs/1812.10260)
    - Use CORAL for speaker recognition
    - 用CORAL改进版进行speaker识别

- 20180821 arXiv [Unsupervised adversarial domain adaptation for acoustic scene classification](https://arxiv.org/abs/1808.05777)
  - Using transfer learning for acoustic classification
  - 迁移学习用于声音场景分类

- 20180615 Interspeech-18 很全面地探索了很多类方法在语音识别上的应用：[A Study of Enhancement, Augmentation, and Autoencoder Methods for Domain Adaptation in Distant Speech Recognition](https://arxiv.org/abs/1806.04841)

- 20180615 Interspeech-18 对话中的语音识别：[Unsupervised Adaptation with Interpretable Disentangled Representations for Distant Conversational Speech Recognition](https://arxiv.org/abs/1806.04872)


- 20180614 arXiv 将迁移学习应用于多个speaker的文字到语音：[Transfer Learning from Speaker Verification to Multispeaker Text-To-Speech Synthesis](https://arxiv.org/abs/1806.04558)

- 20180408 ASRU-18 用迁移学习中的domain separation network进行speech recognition：[Unsupervised Adaptation with Domain Separation Networks for Robust Speech Recognition](https://arxiv.org/abs/1711.08010)

## Multimedia

- [SLIP: Self-supervision meets Language-Image Pre-training](https://arxiv.org/abs/2112.12750)
    - Self-supervised learning + language image pretraining
    - 用自监督学习用于语言到图像的预训练

- [Domain Adaptation on Point Clouds via Geometry-Aware Implicits](https://arxiv.org/abs/2112.09343)
    - Domain adaptation for point cloud
    - 针对点云的domain adaptation

- [VL-Adapter: Parameter-Efficient Transfer Learning for Vision-and-Language Tasks](http://arxiv.org/abs/2112.06825)
    - Vision-language efficient transfer learning
    - 参数高校的vision-language任务迁移

- [TimeMatch: Unsupervised Cross-Region Adaptation by Temporal Shift Estimation](https://arxiv.org/abs/2111.02682)
    - Temporal domain adaptation

- [Transferring Domain-Agnostic Knowledge in Video Question Answering](https://arxiv.org/abs/2110.13395)
    - Domain-agnostic learning for VQA
    - 在VQA任务中进行迁移学习

- WACV-21 [Domain Generalization through Audio-Visual Relative Norm Alignment in First Person Action Recognition](https://arxiv.org/abs/2110.10101)
    - Domain generalization by audio-visual alignment
    - 通过音频-视频对齐进行domain generalization

- 20210716 MICCAI-21 [Few-Shot Domain Adaptation with Polymorphic Transformers](https://arxiv.org/abs/2107.04805)
    - Few-shot domain adaptation with polymorphic transformer
    - 用多模态transformer做少样本的domain adaptation

- 20210716 InterSpeech-21 [Speech2Video: Cross-Modal Distillation for Speech to Video Generation](https://arxiv.org/abs/2107.04806)
  - Cross-model distillation for video generation
  - 跨模态蒸馏用于语音到video的生成

- 20190409 arXiv [Unsupervised Domain Adaptation for Multispectral Pedestrian Detection](https://arxiv.org/abs/1904.03692)
    - Domain adaptation for pedestrian detection
    - 无监督领域自适应用于多模态行人检测

- 20181117 arXiv [Performance Estimation of Synthesis Flows cross Technologies using LSTMs and Transfer Learning](https://arxiv.org/abs/1811.06017)
	- Performance Estimation of Synthesis Flows cross Technologies using LSTMs and Transfer Learning
	- 利用迁移学习进行合成flow评价

- 20180801 arXiv [Multimodal Deep Domain Adaptation](https://arxiv.org/abs/1807.11697)
  - Use multi-modal DA in robotic vision
  - 在机器人视觉中使用多模态的domain adaptation

- 20180413 arXiv 跨模态检索：[Cross-Modal Retrieval with Implicit Concept Association](https://arxiv.org/abs/1804.04318)

## Recommendation

- WSDM-22 [Personalized Transfer of User Preferences for Cross-domain Recommendation](https://arxiv.org/pdf/2110.11154.pdf) [code](https://github.com/easezyc/WSDM2022-PTUPCDR) 
    - Personalized Transfer of User Preferences by meta learner for cross-domain recommendation.
    - 使用元学习器个性化迁移用户兴趣偏好，用于跨领域推荐

- SIGIR-21 [Transfer-Meta Framework for Cross-domain Recommendation to Cold-Start Users](https://arxiv.org/abs/2105.04785)
    - A Transfer-Meta Training Framework for cross-domain recommendation
    - 一种新的迁移-元学习训练框架用于跨领域推荐

- [A Survey on Cross-domain Recommendation: Taxonomies, Methods, and Future Directions](https://arxiv.org/abs/2108.03357)
    - A survey on cross-domain recommendation
    - 跨领域的推荐的综述

- 20191017 arXiv [Unsupervised Domain Adaptation Meets Offline Recommender Learning](https://arxiv.org/abs/1910.07295)
  	- Unsupervised DA meets offline recommender learning
  	- 无监督DA用于离线推荐系统

- 20191017 [Transfer Learning for Algorithm Recommendation](https://arxiv.org/abs/1910.07012)
  	- Transfer learning for algorithm recommendation
  	- 迁移学习用于算法推荐
  
- 20191015 WSDM-20 [DDTCDR: Deep Dual Transfer Cross Domain Recommendation](https://arxiv.org/abs/1910.05189)
  	- Cross-modal recommendation using dual transfer learning
  	- 用对偶迁移进行跨模态推荐

- 20190123 arXiv [Cold-start Playlist Recommendation with Multitask Learning](https://arxiv.org/abs/1901.06125)
  	- Cold-start playlist recommendation with multitask learning
  	- 用多任务学习进行冷启动状态下的播放列表推荐

- 20180801 arXiv [Rank and Rate: Multi-task Learning for Recommender Systems](https://arxiv.org/abs/1807.11698)
  - A multi-task system for recommendation
  - 一个针对于推荐系统的多任务学习

- 20180613 SIGIR-18 多任务学习用于推荐系统：[Explainable Recommendation via Multi-Task Learning in Opinionated Text Data](https://arxiv.org/abs/1806.03568)

- 20180419 arXiv 跨领域的推荐系统：[CoNet: Collaborative Cross Networks for Cross-Domain Recommendation](https://arxiv.org/abs/1804.06769)



## Human activity recognition

- TKDE-22 [Adaptive Memory Networks with Self-supervised Learning for Unsupervised Anomaly Detection](https://arxiv.org/abs/2201.00464)
  - Adaptiev memory network for anomaly detection
  - 自适应的记忆网络用于异常检测

- [迁移学习用于行为识别 Transfer learning for activity recognition](https://github.com/jindongwang/activityrecognition/tree/master/notes)

- 20200405 arXiv [Joint Deep Cross-Domain Transfer Learning for Emotion Recognition](https://arxiv.org/abs/2003.11136)
  	- Transfer learning for emotion recognition
  	- 迁移学习用于情绪识别

- 20190916 ISWC-19 [Cross-dataset deep transfer learning for activity recognition](https://dl.acm.org/citation.cfm?id=3344865)
  	- Cross-dataset transfer learning for activity recognition
  	- 跨数据集的深度迁移学习用于行为识别

- 20190401 arXiv [Cross-Subject Transfer Learning in Human Activity Recognition Systems using Generative Adversarial Networks](https://arxiv.org/abs/1903.12489)
    - Cross-subject transfer learning using GAN
    - 用对抗网络进行跨用户的行为识别

- 20181225 arXiv [A Multi-task Neural Approach for Emotion Attribution, Classification and Summarization](https://arxiv.org/abs/1812.09041)
    - A multi-task approach for emotion attribution, classification, and summarization
    - 一个多任务方法同时用于情绪归属、分类和总结

- 20181220 arXiv [Deep UL2DL: Channel Knowledge Transfer from Uplink to Downlink](https://arxiv.org/abs/1812.07518)
    - Channel knowledge transfer in CSI
    - Wifi定位中的知识迁移

- 20180912 PervasiveHealth-18 [Transfer Learning and Data Fusion Approach to Recognize Activities of Daily Life](https://dl.acm.org/citation.cfm?id=3240949)
    - Transfer learning to perform activity recognition using multi-model sensors
    - 用多模态传感器进行迁移学习，用于行为识别

- 20180819 arXiv [Transfer Learning for Brain-Computer Interfaces: An Euclidean Space Data Alignment Approach](https://arxiv.org/abs/1808.05464)
  - Propose to align the different distributions of EEG signals using transfer learning
  - 针对EEG信号不同人分布不一样的问题提出迁移学习和数据增强的方式加以解决

- 20180529 arXiv 迁移学习用于表情识别：[Meta Transfer Learning for Facial Emotion Recognition](https://arxiv.org/abs/1805.09946)


## Autonomous driving

- CONDA: Continual Unsupervised Domain Adaptation Learning in Visual Perception for Self-Driving Cars [[arxiv](https://arxiv.org/abs/2212.00621)]
  - Continual DA for self-driving cars 连续的domain adaptation用于自动驾驶

- 20180909 arXiv [Driving Experience Transfer Method for End-to-End Control of Self-Driving Cars](https://arxiv.org/abs/1809.01822)
	- Driving experience transfer on self-driving cars
	- 自动驾驶车上的驾驶经验迁移

- 20180705 arXiv 将迁移学习应用于自动驾驶中的不同天气适配：[Modular Vehicle Control for Transferring Semantic Information to Unseen Weather Conditions using GANs](https://arxiv.org/abs/1807.01001)

- 20181219 ICCPS-19 [Simulation to scaled city: zero-shot policy transfer for traffic control via autonomous vehicles](https://arxiv.org/abs/1812.06120)
    - Transfer learning in autonomous vehicles
    - 迁移学习用于自动驾驶车辆的策略迁移

## Others

- Transfer learning for process design with reinforcement learning [[arxiv](https://arxiv.org/abs/2302.03375)]
  - Transfer learning for process design with reinforcement learning 使用强化迁移学习进行过程设计

- Language-Informed Transfer Learning for Embodied Household Activities [[arxiv](https://arxiv.org/abs/2301.05318)]
  - Transfer learning for robust control in household 在家居机器人上使用强化迁移学习

- FL-IJCAI-22 [MetaFed: Federated Learning among Federations with Cyclic Knowledge Distillation for Personalized Healthcare](https://arxiv.org/abs/2206.08516)
  - MetaFed: a new form of federated learning 联邦之联邦学习、新范式

- [On Transfer Learning in Functional Linear Regression](https://arxiv.org/abs/2206.04277)
  - Transfer learning in functional linear regression 迁移学习用于函数式线性回归

- [Transfer Learning for Autonomous Chatter Detection in Machining](https://arxiv.org/abs/2204.05400)
  - Transfer learning for autonomous chatter detection

- ISPASS-22 [Benchmarking Test-Time Unsupervised Deep Neural Network Adaptation on Edge Devices](https://arxiv.org/abs/2203.11295)
  - Benchmarking test-time adaptation for edge devices
  - 在端设备上评测test-time adaptation算法

- ICC-22 [Knowledge Transfer in Deep Reinforcement Learning for Slice-Aware Mobility Robustness Optimization](https://arxiv.org/abs/2203.03227)
  - Knowledge transfer in RL
  - 强化迁移学习

- [Deep Transfer Learning on Satellite Imagery Improves Air Quality Estimates in Developing Nations](https://arxiv.org/abs/2202.08890)
  - Deep transfer learning for air quality estimate
  - 深度迁移学习用于卫星图到空气质量预测

- [DROPO: Sim-to-Real Transfer with Offline Domain Randomization](https://arxiv.org/abs/2201.08434)
  - Sim-to-real transfer with domain randomization
  - 用domain randomization进行sim-to-real transfer

- AAAI-22 [Knowledge Sharing via Domain Adaptation in Customs Fraud Detection](https://arxiv.org/abs/2201.06759)
  - Domain adaptation for fraud detection
  - 用领域自适应进行欺诈检测

- [Transfer-learning-based Surrogate Model for Thermal Conductivity of Nanofluids](https://arxiv.org/abs/2201.00435)
  - Transfer learning for thermal conductivity
  - 迁移学习用于热传导

- [Toward Co-creative Dungeon Generation via Transfer Learning](http://arxiv.org/abs/2107.12533)
    - Game scene generation with transfer learning
    - 用迁移学习生成游戏场景

- 20210716 ICML-21 workshop [Leveraging Domain Adaptation for Low-Resource Geospatial Machine Learning](https://arxiv.org/abs/2107.04983)
  - Using domain adaptation for geospatial ML
  - 用domain adaptation进行地理空间的机器学习

- 20210202 [Admix: Enhancing the Transferability of Adversarial Attacks](https://arxiv.org/abs/2102.00436)
    - Enhancing the transferability of adversarial attacks
    - 增强对抗攻击的可迁移性

- 20201116 [Cross-Domain Learning for Classifying Propaganda in Online Contents](https://arxiv.org/abs/2011.06844)
    - 跨领域学习用于检测在线推广

- 20201116 [Filter Pre-Pruning for Improved Fine-tuning of Quantized Deep Neural Networks](https://arxiv.org/abs/2011.06751)
    - 量子神经网络中的finetune

- 20200914 [Transfer Learning of Graph Neural Networks with Ego-graph Information Maximization](https://arxiv.org/abs/2009.05204)
    - 迁移学习用于GNN

- 20200529 WWW-20 [Modeling Users’ Behavior Sequences with Hierarchical Explainable Network for Cross-domain Fraud Detection](https://dl.acm.org/doi/abs/10.1145/3366423.3380172)
  	- Transfer learning for cross-domain fraud detection
  	- 迁移学习用于跨领域欺诈检测

- 20200420 arXiv [Transfer Learning with Graph Neural Networks for Short-Term Highway Traffic Forecasting](https://arxiv.org/abs/2004.08038)
  	- Transfer learning with GNN for highway traffic forecasting
  	- 迁移学习+GNN用于交通流量预测

- 20191222 NIPS-19 workshop [Sim-to-Real Domain Adaptation For High Energy Physics](https://arxiv.org/abs/1912.08001)
     - Transfer learning for high energy physics
     - 迁移学习用于高能物理

- 20191222 arXiv [Transfer learning in hybrid classical-quantum neural networks](https://arxiv.org/abs/1912.08278)
     - Transfer learning for quantum neural networks
  
- 20191214 arXiv [Transfer Learning-Based Outdoor Position Recovery with Telco Data](https://arxiv.org/abs/1912.04521)
     - Outdoor position recorvey with Telco data using transfer learning

- 20191111 BigData-19 [Deep Transfer Learning for Thermal Dynamics Modeling in Smart Buildings](https://arxiv.org/abs/1911.03318)
  	- Transfer learning for thermal dynamics modeling

- 20191111 arXiv [Transfer Learning in Spatial-Temporal Forecasting of the Solar Magnetic Field](https://arxiv.org/abs/1911.03193)
  	- Transfer learning for solar magnetic field

- 20191111 arXiv [Deep geometric knowledge distillation with graphs](https://arxiv.org/abs/1911.03080)
  	- Deep geometric knowledge distillation with graphs

- 20190813 IJAIT [Transferring knowledge from monitored to unmonitored areas for forecasting parking spaces](https://arxiv.org/abs/1908.03629)
    - Transfer learning for forecasting parking spaces
    - 用迁移学习预测停车空间

- 20190517 PHM-19 [Domain Adaptive Transfer Learning for Fault Diagnosis](https://arxiv.org/abs/1905.06004)
  	- Domain adaptation for fault diagnosis
  	- 领域自适应用于错误检测

- 20190508 arXiv [Text2Node: a Cross-Domain System for Mapping Arbitrary Phrases to a Taxonomy](https://arxiv.org/abs/1905.01958)
  	- Cross-domain system for mapping arbitrary phrases to a taxonomy

- 20190415 PAKDD-19 [Targeted Knowledge Transfer for Learning Traffic Signal Plans](https://link.springer.com/chapter/10.1007/978-3-030-16145-3_14)
  	- Targeted knowledge transfer for traffic control
  	- 目标知识迁移应用于交通红绿灯

- 20190415 PAKDD-19 [Knowledge Graph Rule Mining via Transfer Learning](https://link.springer.com/chapter/10.1007/978-3-030-16142-2_38)
  	- Knowledge Graph Rule Mining via Transfer Learning
  	- 迁移学习应用于知识图谱

- 20190415 PAKDD-19 [Spatial-Temporal Multi-Task Learning for Within-Field Cotton Yield Prediction](https://link.springer.com/chapter/10.1007/978-3-030-16148-4_27)
  	- Spatial-Temporal multi-task learning for cotton yield prediction
  	- 时空依赖的多任务学习用于棉花收入预测

- 20190415 PAKDD-19 [Passenger Demand Forecasting with Multi-Task Convolutional Recurrent Neural Networks](https://link.springer.com/chapter/10.1007/978-3-030-16145-3_3)
  	- Passenger demand forecasting with multi-task CRNN
  	- 用多任务CRNN模型进行顾客需求估计

- 20190123 arXiv [Transfer Learning and Meta Classification Based Deep Churn Prediction System for Telecom Industry](https://arxiv.org/abs/1901.06091)
    - Transfer learning in telcom industry
    - 迁移学习用于电信行业

- 20181225 arXiv [A General Approach to Domain Adaptation with Applications in Astronomy](https://arxiv.org/abs/1812.08839)
    - Adopting active learning to transfer model
    - 用主动学习来进行模型迁移并应用到天文学上

- 20181220 arXiv [Domain Adaptation for Reinforcement Learning on the Atari](https://arxiv.org/abs/1812.07452)
    - Reinforcement domain adaptation on Atari games
    - 迁移强化学习用于Atari游戏

- 20181219 NER-19 [Transfer Learning in Brain-Computer Interfaces with Adversarial Variational Autoencoders](https://arxiv.org/abs/1812.06857)
    - Transfer learning in brain-computer interfaces
    - 迁移学习在脑机交互中的应用

- 20181218 arXiv [Transfer learning to model inertial confinement fusion experiments](https://arxiv.org/abs/1812.06055)
    - Using transfer learning for inertial confinement fusion
    - 用迁移学习进行惯性约束聚变

- 20181214 arXiv [Bridging the Generalization Gap: Training Robust Models on Confounded Biological Data](https://arxiv.org/abs/1812.04778)
    - Transfer learning for generalizing on biological data
    - 用迁移学习增强生物数据的泛化能力

- 20181214 LAK-19 [Transfer Learning using Representation Learning in Massive Online Open Courses](https://arxiv.org/abs/1812.05043)
    - Transfer learning in MOOCs
    - 迁移学习用于大规模在线网络课程

- 20181214 DVPBA-19 [Considering Race a Problem of Transfer Learning](https://arxiv.org/abs/1812.04751)
    - Consider race in transfer learning
    - 在迁移学习问题中考虑种族问题(跨种族迁移)

- 20181121 NSFREU-18 [Transfer Learning with Deep CNNs for Gender Recognition and Age Estimation](https://arxiv.org/abs/1811.07344)
    - Deep transfer learning for Gender Recognition and Age Estimation
    - 用深度迁移学习进行性别识别和年龄估计

- 20181120 arXiv [Spatial-temporal Multi-Task Learning for Within-field Cotton Yield Prediction](https://arxiv.org/abs/1811.06665)
	- Multi-task learning for cotton yield prediction
	- 多任务学习用于棉花产量预测
  
- 20181012 ICMLA-18 [Virtual Battery Parameter Identification using Transfer Learning based Stacked Autoencoder](https://arxiv.org/abs/1810.04642)
	- Using transfer learning for calculating the virtual battery in a thermostatics load
	- 用迁移学习进行恒温器的电量估计

- 20180909 arXiv [Deep Learning for Domain Adaption: Engagement Recognition](https://arxiv.org/abs/1808.02324)
	- deep transfer learning for engagement recognition
	- 用深度迁移学习进行人机交互中的engagement识别

- 20180621 arXiv 迁移学习用于强化学习中的图像翻译：[Transfer Learning for Related Reinforcement Learning Tasks via Image-to-Image Translation](https://arxiv.org/abs/1806.07377)

- 20180610 arXiv 迁移学习用于Coffee crop分类：[A Comparative Study on Unsupervised Domain Adaptation Approaches for Coffee Crop Mapping](https://arxiv.org/abs/1806.02400)

- 20180530 MNRAS 用迁移学习检测银河星系兼并：[Using transfer learning to detect galaxy mergers](https://arxiv.org/abs/1805.10289)

- 20180524 KDD-18 用迁移学习方法进行人们的ID迁移：[Learning and Transferring IDs Representation in E-commerce](https://arxiv.org/abs/1712.08289)

- 20180421 arXiv 采用联合分布适配的深度迁移网络用于工业生产中的错误诊断：[Deep Transfer Network with Joint Distribution Adaptation: A New Intelligent Fault Diagnosis Framework for Industry Application](https://arxiv.org/abs/1804.07265)