
- [Requirements](#requirements)
- [Prepare environment, dataset, and trained models](#prepare-environment-dataset-and-trained-models)
  - [Prepare the environment](#prepare-the-environment)
    - [Prepare the validation environment](#prepare-the-validation-environment)
    - [Prepare the training environment](#prepare-the-training-environment)
  - [Prepare the dataset and trained models](#prepare-the-dataset-and-trained-models)
- [Validate the results in the paper](#validate-the-results-in-the-paper)
  - [Figure 5: accuracy and DIR of penultimate-layer-guided adversarial attacks](#figure-5-accuracy-and-dir-of-penultimate-layer-guided-adversarial-attacks)
  - [Figure 6: DIR of neuron-coverage guided adversarial attacks](#figure-6-dir-of-neuron-coverage-guided-adversarial-attacks)
  - [Figure 7: DIR of backdoor defects](#figure-7-dir-of-backdoor-defects)
  - [Figure 8: convergence process](#figure-8-convergence-process)
  - [Table 2: DIR of NLP backdoors](#table-2-dir-of-nlp-backdoors)
- [Guideline to train the models](#guideline-to-train-the-models)
  - [Models for CV adversarial samples](#models-for-cv-adversarial-samples)
  - [Models for CV backdoor](#models-for-cv-backdoor)
  - [Models for NLP backdoor](#models-for-nlp-backdoor)

## Requirements

To use this artifact, you should have a server with Ubuntu 18.04. To validate the main results, you can follow the intructions to setup python environment. To train the model and validate the neuron-coverage results, you need a GPU and CUDA 11.4 installed on the server.

## Prepare environment, dataset, and trained models

### Prepare the environment

We recommand you to use [Anaconda](https://www.anaconda.com/) to prepare the environments. You should first follow the official website to install Anaconda in your computer. ReMoS was developed with PyTorch 1.7 and CUDA 11.4. To train the models you should have GPU and CUDA 11.4 installed. However, if you only need to validate the results, you do not need to install PyTorch because we have prepared the validation code for you. We will first introduce how to prepare the validation environment, then we will give the instructions to prepare the model-training environment with CUDA and PyTorch.

#### Prepare the validation environment
- First, you should create a new conda environment
```
conda create -n validation python=3.6
```
- Then, you should activate the conda environment
```
conda activate validation
```
- Install necessary packages

```
conda install numba matplotlib pandas==0.24.1 scipy
```
Now you can go ahead to prepare the dataset and models before check the results in the paper.

#### Prepare the training environment
- First, you should have GPU on your computer and install CUDA 11.4 following the official (website)[https://developer.nvidia.com/cuda-11-4-0-download-archive]
- Then, you should install PyTorch 1.7
```
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=11.3 -c pytorch
```
- Then, you should install advertorch

```
pip install advertorch
```
Now you have setup the training environment.

### Prepare the dataset and trained models

We have prepared the dataset and the trained models in a zip file. You can directly download zip file from [Google Drive](https://drive.google.com/file/d/1ps4fbbaGsHrONZiPjWdzK1UC53IK0Kwc/view?usp=sharing), put the downloaded file under ``ReMoS`` and run the ``unpack_downloads.sh`` script.
```
bash unpack_downloads.sh
```
After unpacking the downloaded file, the added files should be aranged as follows:
```
├── CV_adv
│   ├── data
│   │   ├── CUB_200_2011
│   │   ├── Flower_102
│   │   ├── MIT_67
│   │   ├── stanford_40
│   │   └── stanford_dog
│   └── results
│       ├── nc_adv_eval_test
│       ├── res18_models
│       ├── res50_models
├── CV_backdoor
│   └── results
│       └── remos
├── NLP_backdoor
│   └── RIPPLe
```

## Validate the results in the paper

In this section, we prepared several scripts to validate the results in the paper. 

### Figure 5: accuracy and DIR of penultimate-layer-guided adversarial attacks

Figure 5 shows the accuracy and the DIR of penultimate-layer-guided adversarial attacks, including two architectures: ResNet18 and ResNet50. The trained models of ResNet18 are stored in ``CV_adv/results/res18_models``, and the ResNet50 models are stored in ``CV_adv/results/res50_models``. Each directory has several sub-directries that store the models of one dataset and one baseline technique. For example, ``CV_adv/results/res18_models/finetune/resnet18_mit67`` stores the finetuned baseline on MIT67 dataset.

To validate the results in Figure 5, we prepared a script ``CV_adv/examples/penul_guided_defect.py`` to summarize the results in ``CV_adv/results/res18_models`` and ``CV_adv/results/res50_models``. You can check the result by typing the following commands:
```
cd CV_adv
python examples/penul_guided_defect.py
```
You will get the results like:
```
==============================   ResNet18   ==============================
Dataset Scenes:
Techniques  Finetune  DELTA  Magprune  Retrain  DELTA-R  Renofeation  ReMoS
Acc            75.30  77.69     72.84    41.04    60.52        53.81  74.40
DIR            61.55  79.25     58.40     5.45    17.51        11.23  19.66
Dataset Birds:
Techniques  Finetune  DELTA  Magprune  Retrain  DELTA-R  Renofeation  ReMoS
Acc            78.27  75.93     72.73    51.73    59.22        55.32  75.70
DIR            50.87  83.78     58.47     9.21     7.94         4.59  15.25
Dataset Flowers:
Techniques  Finetune  DELTA  Magprune  Retrain  DELTA-R  Renofeation  ReMoS
Acc            94.88  95.36     93.09    74.26    89.98        90.35  93.12
DIR            37.97  70.34     51.10     3.08     7.63         3.56  12.84
Dataset Dogs:
Techniques  Finetune  DELTA  Magprune  Retrain  DELTA-R  Renofeation  ReMoS
Acc            81.95  83.60     78.68    40.66    62.93        52.17  75.06
DIR            89.01  96.89     85.08     9.59    14.06        14.00  18.71
Dataset Actions:
Techniques  Finetune  DELTA  Magprune  Retrain  DELTA-R  Renofeation  ReMoS
Acc            76.84  77.66     75.52    30.38    63.37        65.28  75.56
DIR            73.39  82.31     75.75     5.70     5.27         3.98  15.23
==============================   ResNet50   ==============================
Dataset Scenes:
Techniques  Finetune  DELTA  Magprune  Retrain  DELTA-R  Renofeation  ReMoS
Acc            77.91  78.21     75.45    34.78    72.01        35.52  77.31
DIR            55.27  63.36     39.27     5.79     5.08         4.20  13.71
Dataset Birds:
Techniques  Finetune  DELTA  Magprune  Retrain  DELTA-R  Renofeation  ReMoS
Acc            79.08  80.57     76.49    39.98    76.49        38.30  79.76
DIR            39.83  29.72     28.16    11.46     5.26         3.88  12.02
Dataset Flowers:
Techniques  Finetune  DELTA  Magprune  Retrain  DELTA-R  Renofeation  ReMoS
Acc            95.98  96.84     96.23    68.69    93.94        77.48  97.03
DIR            19.99  31.63     19.29     3.15     2.24         1.68   2.93
Dataset Dogs:
Techniques  Finetune  DELTA  Magprune  Retrain  DELTA-R  Renofeation  ReMoS
Acc            87.56  87.50     83.54    30.14    79.83        19.55  84.78
DIR            86.57  68.39     62.15    13.55     5.35        11.39  29.64
Dataset Actions:
Techniques  Finetune  DELTA  Magprune  Retrain  DELTA-R  Renofeation  ReMoS
Acc            83.79  80.88     79.89    36.94    72.52        43.32  82.55
DIR            66.91  78.04     33.00     6.70     2.82         3.71   7.50
```

The results under ``ResNet18`` includes the results of the two left figures in Figure 5. The top-left sub-figure shows the Acc and the bottom-left sub-figures shows the DIR.
The results under ``ResNet50`` includes the results of the two right figures in Figure 5. The top-right sub-figure shows the Acc and the bottom-right sub-figures shows the DIR.

### Figure 6: DIR of neuron-coverage guided adversarial attacks

Figure 6 shows the DIR of neuron-coverage guided adversarial attacks on ResNet18 and two datasets: Scenes and Actions. You can see the results on more datasets and more models by slightly changing the scripts. The models are the same as penultimate-layer-guided adversarial attacks. To validate the results in Figure 6, you should first use the neuron-coveraged guided attacks to attack the model and summarize the results. However, due to the generation of adversarial samples is pretty slow, we prepared the log and the results of each experimental setting. You can easily validate Figure 6 by summarizing the results. Besides, we also provide the scripts to generate the NC guided samples.

To summarize the results, we prepared the NC sample generation log and results in ``CV_adv/results/nc_adv_eval_test``. Each directory in ``nc_adv_eval_test`` includes the log of NC attack of one coverage technique and strategy. You can check the results of Figure 6 by following commands:
```
cd CV_adv
python examples/nc_guided_defect.py
```
You will get the result:
```
Results for Scenes
Strategy   DeepXplore                      DLFuzzRR                        DLFuzz
Techniques   Finetune DELTA Magprune ReMoS Finetune DELTA Magprune ReMoS Finetune DELTA Magprune ReMoS
NC               50.8  64.2     52.1  19.4     49.1  60.2     50.8  21.0     64.9  73.4     60.3  28.3
TKNC             52.1  67.3     53.7  20.6     34.0  42.7     33.1  14.3     36.7  41.5     35.4  18.7
SNAC             56.9  71.5     60.4  22.8     50.9  61.4     52.2  22.6     63.9  74.8     60.3  26.9

Results for Actions
Strategy   DeepXplore                      DLFuzzRR                        DLFuzz
Techniques   Finetune DELTA Magprune ReMoS Finetune DELTA Magprune ReMoS Finetune DELTA Magprune ReMoS
NC               43.3  60.1     44.4  13.7     50.1  59.1     50.0  15.3     76.6  87.4     71.6  23.0
TKNC             48.4  61.1     46.7  15.8     31.7  41.1     33.7  11.1     36.4  39.6     33.3  13.0
SNAC             52.4  64.8     50.0  17.4     52.2  61.1     53.7  15.1     77.8  87.0     70.4  23.4
```
The table under "Results for Scenes" corresponds to the left three columns in Figure 6, and the table under "Results for Actions" shows the right three columns in Figure 6. 

We also provide the scripts to generate NC guided adversarial samples from trained models in ``CV_adv/examples/nc_guided_defect.sh``. You can generate the samples and check the results by typing
```
cd CV_adv
bash examples/nc_guided_defect.sh $gpu
```
where ``$gpu`` is the gpu id you want to use. You can also change ``nc_guided_defect.sh`` to see the results on other datasets.

### Figure 7: DIR of backdoor defects

Figure 7 shows the DIR of backdoor defects on ResNet50 and three datasets. The code and results of backdoor experiments are in ``CV_backdoor``. We provide the saved model in ``CV_backdoor/remos``, in which each directory includes the models trained by one baseline. The script to validate the backdoor results is in ``CV_backdoor/examples/backdoor.py``. You can check the results by typeing
```
cd CV_backdoor
python examples/backdoor.py
```
You will get the result:
```
Dataset      Scenes                                        Birds                                      Actions
Techniques Finetune Magprune Retrain Renofeation  ReMoS Finetune Magprune Retrain Renofeation  ReMoS Finetune Magprune Retrain Renofeation  ReMoS
Acc           78.21    77.69   53.66       55.37  76.19    81.19    81.36   63.63       60.49  82.00    82.39    82.45   42.37       44.58  81.65
DIR           77.96    57.83    5.15        4.45   8.23    50.62    40.35    5.56        7.85  13.07    90.15    89.65    7.85        9.94   9.05
```
The row of Acc shows the results in the left figure of Figure 7, and the row of DIR represents the right one. 


### Figure 8: convergence process 

Figure 8 shows the comparison in terms of the model convergence of different techniques, including finetuning, retraining, and ReMoS. The code and relevant material is in the directory ``efficiency``. By typing following commands, you can generate Figure 8:
```
cd efficiency
python plot.py
```
The figure is saved in ``efficiency/speed.pdf``

### Table 2: DIR of NLP backdoors

Table 2 shows the DIR of NLP backdoors from the models trained by various techniques. The relevant code and models are in ``NLP_backdoor/RIPPLe``. Table 2 includes two models, BERT and RoBERTa. The trained BERT models are stored in ``NLP_backdoor/RIPPLe/bert_weights`` and the trained RoBERTa models are stored in ``NLP_backdoor/RIPPLe/roberta_weights``. You can use the script ``NLP_backdoor/summarize.py`` to validate Table 2
```
cd NLP_backdoor
python summarize.py
```
You will get the results as follows:
```
Results for BERT
Dataset    Data Poison                                         Weight Poison
Techniques   Fine-tune         Mag-prune          ReMoS            Fine-tune         Mag-prune          ReMoS
Metrics            ACC     DIR       ACC     DIR    ACC    DIR           ACC     DIR       ACC     DIR    ACC    DIR
SST2-SST2        92.70  100.00     92.36  100.00  91.27  39.10         92.29  100.00     92.45  100.00  90.93  29.82
IMDB-IMDB        87.97   96.12     88.25   96.15  85.53  61.73         89.34   96.15     89.48   96.10  87.00  37.73
SST2-IMDB        90.81  100.00     91.26  100.00  90.04  74.67         91.68  100.00     91.16  100.00  87.42  61.49
IMDB-SST2        93.21   96.18     92.46   96.18  91.15  27.72         92.81   96.22     92.58   96.03  91.95  21.56

Results for RoBERTa
Dataset    Data Poison                                         Weight Poison
Techniques   Fine-tune         Mag-prune          ReMoS            Fine-tune         Mag-prune         ReMoS
Metrics            ACC     DIR       ACC     DIR    ACC    DIR           ACC     DIR       ACC    DIR    ACC    DIR
SST2-SST2        94.20  100.00     93.70  100.00  91.17  29.82         93.38  100.00     93.20  98.94  90.71  24.95
IMDB-IMDB        90.60   96.17     89.55   95.25  85.74  70.19         89.06   96.53     88.77  92.06  86.35  85.92
SST2-IMDB        92.12   99.88     92.27  100.00  88.74  61.26         91.85  100.00     90.83  99.53  88.72  30.83
IMDB-SST2        93.53   88.16     92.65   85.27  90.32  24.14         93.85   93.93     93.57  91.22  89.96  18.08
```


## Guideline to train the models

### Models for CV adversarial samples
The code to train CV backdoor models and relevant baseline models are in ``CV_adv``. To train the models, you can use following instructions:

- Train the finetuned student. The script is ``CV_adv/examples/finetune.sh`` and you can specify the dataset by slightly changing the configuration. The trained model will be saved in ``CV_adv/results/baseline/finetune/resnet18_${DATASET}``. The commands are
```
cd CV_adv
bash examples/finetune.sh $gpu
```
where ``$gpu``` is the gpu id you want to use.

- Train the mag-pruned student. The script is ``CV_adv/examples/weight.sh`` and the trained model will be saved in ``CV_adv/results/baseline/weight``. The commands are
```
cd CV_adv
bash examples/weight.sh $gpu
```

- Profile the teacher model (the Coverage Frequency Profiling step in the paper). The script is ``CV_adv/examples/nc_profile.sh`` and the profiling results will be saved in ``CV_adv/results/nc_profiling/${COVERAGE}_${DATASET}_${MODEL}``. The commands to profile are
```
cd CV_adv
bash examples/nc_profile.sh $gpu
```
- Train with the relevant model slice. The script is ``CV_adv/examples/remos.sh`` and the results will be saved in ``CV_adv/results/remos_$COVERAGE``. The commands to train with the relevant model slice is 
```
cd CV_adv
bash examples/remos.sh $gpu
```
After training, you can check the file ``posttrain_eval.txt`` to see the accuracy and DIR. 

- Other baselines. We also provide the scripts to train other baselines in Section 5.2. The scripts are in ``CV_adv/examples``, including ``renofeation.sh``, ``delta.sh``, and ``delta-r.sh``.

### Models for CV backdoor

The code to train CV backdoor models and relevant baseline models are in ``CV_backdoor``. To train the models, you can use following instructions:

- Train the backdoored teacher model. The script is ``CV_backdoor/examples/r50_poison.sh`` and you can specify the dataset by slightly changing the configuration. The trained model will be saved in ``results/r50_backdoor/backdoor/res50_$DATASET``. The command to train the poisoned teacher model is 
```
cd CV_backdoor
bash examples/r50_poison.sh $gpu
```
where ``$gpu``` is the gpu id you want to use.

- Train the finetuned student model. The script is ``CV_backdoor/examples/r50_baseline.sh`` and the models will be saved in ``results/r50_backdoor/finetune``. The command to train the finetuned student model is 
```
cd CV_backdoor
bash examples/r50_baseline.sh $gpu
```
The accuracy and DIR are saved in the ``test.tsv`` in the result directory.

- Trained the mag-pruned model. The script is ``CV_backdoor/examples/r50_magprune.sh`` and the models will be saved in ``results/r50_backdoor/finetune``.
- Profile the teacher model (the Coverage Frequency Profiling step in the paper). The script is ``CV_backdoor/examples/remos/profile.sh`` and the profiled results will be saved in ``CV_backdoor/results/nc_profiling/${COVERAGE}_${DATASET}_resnet50``. The command to profile is 
```
bash examples/remos/profile.sh $gpu
```

- Train with the relevant model slice. The script to train the relevant model slice is ``CV_backdoor/examples/remos/remos.sh`` and the trained model will be saved in ``CV_backdoor/results/r50_backdoor/remos_res50/$DATASET``. The command to train the relevant model slice is 
```
bash examples/remos/remos.sh $gpu
```

### Models for NLP backdoor

The code for NLP backdoor are saved in ``NLP_backdoor/RIPPLE``, which are developed based on [RIPPLe](https://github.com/neulab/RIPPLe). The configuration scripts for BERT and RoBERTa are saved in ``NLP_backdoor/RIPPLE/bert_mani`` and ``NLP_backdoor/RIPPLE/roberta_mani``, respectively. The script to run the configurations is saved in ``NLP_backdoor/examples/bert_slim.sh``.

