#!/bin/bash


iter=30000
lr=1e-3
wd=1e-4
mmt=0

DATASETS=(MIT_67 CUB_200_2011 Flower_102 stanford_40 stanford_dog)
DATASET_NAMES=(MIT67Data CUB200Data Flower102Data Stanford40Data SDog120Data)
DATASET_ABBRS=(mit67 cub200 flower102 stanford40 sdog120)

for i in 0
do
    DATASET=${DATASETS[i]}
    DATASET_NAME=${DATASET_NAMES[i]}
    DATASET_ABBR=${DATASET_ABBRS[i]}

    DIR=results/baseline/finetune
    NAME=resnet18_${DATASET_ABBR}

    CUDA_VISIBLE_DEVICES=$1 python -u finetune.py --iterations ${iter} --datapath data/${DATASET}/ --dataset ${DATASET_NAME} --name $NAME --batch_size 64 --lr ${lr} --network resnet18 --weight_decay ${wd}  --momentum ${mmt} --output_dir $DIR 

done