#!/bin/bash

iter=10000
mmt=0

DATASETS=(MIT_67 CUB_200_2011 stanford_40 )
DATASET_NAMES=(MIT67Data CUB200Data Stanford40Data)
DATASET_ABBRS=(mit67 cub200 stanford40)
LEARNING_RATE=5e-3
WEIGHT_DECAY=1e-4
portion=0.2

for i in 0
do

    DATASET=${DATASETS[i]}
    DATASET_NAME=${DATASET_NAMES[i]}
    DATASET_ABBR=${DATASET_ABBRS[i]}
    lr=${LEARNING_RATE}
    wd=${WEIGHT_DECAY}

    NAME=${DATASET_ABBR}
    DIR=results/r50_backdoor/finetune/

    CUDA_VISIBLE_DEVICES=$1 python -u finetune.py --teacher_datapath ../CV_adv/data/${DATASET} --teacher_dataset ${DATASET_NAME} --student_datapath ../CV_adv/data/${DATASET} --student_dataset ${DATASET_NAME} --iterations ${iter} --name ${NAME} --batch_size 64 --lr ${lr} --network resnet50 --weight_decay ${wd} --test_interval $iter --momentum ${mmt} --output_dir ${DIR} --argportion ${portion} --teacher_method backdoor_finetune --fixed_pic --checkpoint results/r50_backdoor/backdoor/res50_${DATASET_ABBR}/teacher_ckpt.pth 


done


