#!/bin/bash

iter=10000
id=1
splmda=0
lmda=0
layer=1234
mmt=0

DATASETS=(MIT_67 CUB_200_2011 Flower_102 stanford_40 stanford_dog VisDA)
DATASET_NAMES=(MIT67Data CUB200Data Flower102Data Stanford40Data SDog120Data VisDaDATA)
DATASET_ABBRS=(mit67 cub200 flower102 stanford40 sdog120 visda)

LEARNING_RATE=5e-3
WEIGHT_DECAY=1e-4
PORTION=(0.2 0.5 0.7 0.9)
RATIO=(0.0 0.5 0.7 0.9)

COVERAGE=neuron_coverage
for j in 0 
do
for nc_ratio in 0.005 0.01 0.03 0.05 0.07 0.1
do
# 0 1 3
for i in 2
do
    

    DATASET=${DATASETS[i]}
    DATASET_NAME=${DATASET_NAMES[i]}
    DATASET_ABBR=${DATASET_ABBRS[i]}
    lr=${LEARNING_RATE}
    wd=${WEIGHT_DECAY}
    portion=0.2
    ratio=${RATIO[j]}
    NAME=fixed_${DATASET_ABBR}_${ratio}_nc${nc_ratio}
    #NAME=random_${DATASET_ABBR}_${ratio}
    newDIR=results/backdoor_ablation/r50_ratio
    # newDIR=results/test
    teacher_dir=results/qianyi_res50/fixed_${DATASET_ABBR}_${ratio}

    CUDA_VISIBLE_DEVICES=$1 \
    python -u py_qianyi.py \
    --teacher_datapath ../data/${DATASET} \
    --teacher_dataset ${DATASET_NAME} \
    --student_datapath ../data/${DATASET} \
    --student_dataset ${DATASET_NAME} \
    --iterations ${iter} \
    --name ${NAME} \
    --batch_size 64 \
    --feat_lmda ${lmda} \
    --lr ${lr} \
    --network resnet50 \
    --weight_decay ${wd} \
    --beta 1e-2 \
    --test_interval 10000 \
    --adv_test_interval -1 \
    --feat_layers ${layer} \
    --momentum ${mmt} \
    --output_dir ${newDIR} \
    --argportion ${portion} \
    --backdoor_update_ratio ${ratio} \
    --teacher_method backdoor_finetune \
    --checkpoint $teacher_dir/teacher_ckpt.pth \
    --student_method nc_weight_rank_prune \
    --fixed_pic \
    --train_all \
    --prune_interval 1000000 \
    --weight_total_ratio ${nc_ratio} \
    --weight_ratio_per_prune 0 \
    --weight_init_prune_ratio ${nc_ratio} \
    --nc_info_dir results/nc_profiling/${COVERAGE}_${DATASET_ABBR}_resnet50 \
    # &

    done
    done
done

