#!/bin/bash
export PYTHONPATH=../..:$PYTHONPATH

iter=30000
lr=5e-3
wd=1e-4
mmt=0

DATASETS=(MIT_67 CUB_200_2011 Flower_102 stanford_40 stanford_dog)
DATASET_NAMES=(MIT67Data CUB200Data Flower102Data Stanford40Data SDog120Data)
DATASET_ABBRS=(mit67 cub200 flower102 stanford40 sdog120)


for i in 0 
do
    for ratio in 0.1
    do
        for COVERAGE in neuron_coverage
        do

        total_ratio=${ratio}

        DATASET=${DATASETS[i]}
        DATASET_NAME=${DATASET_NAMES[i]}
        DATASET_ABBR=${DATASET_ABBRS[i]}

        NAME=resnet18_${DATASET_ABBR}_total${total_ratio}_lr${lr}_iter${iter}_wd${wd}_mmt${mmt}
        DIR=results/remos_$COVERAGE

        CUDA_VISIBLE_DEVICES=$1 python finetune.py --iterations ${iter} --datapath data/${DATASET}/ --dataset ${DATASET_NAME} --name ${NAME} --batch_size 64 --lr ${lr} --network resnet18 --weight_decay ${wd} --momentum ${mmt} --output_dir ${DIR} --method remos --weight_total_ratio $total_ratio --weight_init_prune_ratio $total_ratio --prune_interval $iter --weight_ratio_per_prune 0 --nc_info_dir results/nc_profiling/${COVERAGE}_${DATASET_ABBR}_resnet18 

        done
    done
done