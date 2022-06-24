#!/bin/bash

iter=10000
mmt=0

DATASETS=(MIT_67 CUB_200_2011 stanford_40 )
DATASET_NAMES=(MIT67Data CUB200Data Stanford40Data)
DATASET_ABBRS=(mit67 cub200 stanford40)

LEARNING_RATE=5e-3
WEIGHT_DECAY=1e-4

COVERAGE=neuron_coverage
portion=0.2

for nc_ratio in 0.03
do
for i in 0
do
    

    DATASET=${DATASETS[i]}
    DATASET_NAME=${DATASET_NAMES[i]}
    DATASET_ABBR=${DATASET_ABBRS[i]}
    lr=${LEARNING_RATE}
    wd=${WEIGHT_DECAY}
    
    NAME=${DATASET_ABBR}
    DIR=results/r50_backdoor/remos_res50

    CUDA_VISIBLE_DEVICES=$1 \
    python -u finetune.py \
    --teacher_datapath ../CV_adv/data/${DATASET} \
    --teacher_dataset ${DATASET_NAME} \
    --student_datapath ../CV_adv/data/${DATASET} \
    --student_dataset ${DATASET_NAME} \
    --iterations ${iter} \
    --name ${NAME} \
    --batch_size 64 \
    --lr ${lr} \
    --network resnet50 \
    --weight_decay ${wd} \
    --test_interval 10000 \
    --adv_test_interval -1 \
    --momentum ${mmt} \
    --output_dir ${DIR} \
    --argportion ${portion} \
    --teacher_method backdoor_finetune \
    --checkpoint results/r50_backdoor/backdoor/res50_${DATASET_ABBR}/teacher_ckpt.pth \
    --student_method nc_weight_rank_prune \
    --fixed_pic \
    --prune_interval 1000000 \
    --weight_total_ratio ${nc_ratio} \
    --weight_ratio_per_prune 0 \
    --weight_init_prune_ratio ${nc_ratio} \
    --dropout 1e-1 \
    --nc_info_dir results/nc_profiling/${COVERAGE}_${DATASET_ABBR}_resnet50 \
    # &


    done
done

