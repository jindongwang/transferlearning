

#!/bin/bash
export PYTHONPATH=../..:$PYTHONPATH

iter=30000
id=1
splmda=0
lmda=0
layer=1234
lr=5e-3
wd=1e-4
mmt=0

DATASETS=(MIT_67 CUB_200_2011 Flower_102 stanford_40 stanford_dog VisDA)
DATASET_NAMES=(MIT67Data CUB200Data Flower102Data Stanford40Data SDog120Data VisDaDATA)
DATASET_ABBRS=(mit67 cub200 flower102 stanford40 sdog120 visda)


COVERAGE=strong_coverage
STRATEGY=deepxplore

# neuron_coverage top_k_coverage strong_coverage
# random deepxplore dlfuzz dlfuzzfirst

for i in 3
do

for COVERAGE in neuron_coverage top_k_coverage strong_coverage
do
for STRATEGY in deepxplore dlfuzz dlfuzzfirst
do

    DATASET=${DATASETS[i]}
    DATASET_NAME=${DATASET_NAMES[i]}
    DATASET_ABBR=${DATASET_ABBRS[i]}

    DIR=results/baseline/finetune
    NAME=resnet18_${DATASET_ABBR}_lr${lr}_iter${iter}_feat${lmda}_wd${wd}_mmt${mmt}_${id}

    FINETUNE_DIR=results/res18_ncprune_sum/finetune/resnet18_${DATASET_ABBR}_lr5e-3_iter30000_feat0_wd1e-4_mmt0_1
    DELTA_DIR=results/res18_ncprune_sum/delta/resnet18_${DATASET_ABBR}_lr1e-2_iter10000_feat5e-1_wd1e-4_mmt0_1
    WEIGHT_DIR=results/res18_ncprune_sum/weight/resnet18_${DATASET_ABBR}_total0.8_init0.8_per0.1_int10000_lr5e-3_iter10000_feat0_wd1e-4_mmt0_1
    NCPRUNE_DIR=results/res18_ncprune_sum/ncprune/resnet18_${DATASET_ABBR}_do_total0.1_trainall_lr5e-3_iter30000_feat0_wd1e-4_mmt0_1

    CUDA_VISIBLE_DEVICES=$1 python -u DNNtest/eval_nc.py --datapath data/${DATASET}/ --dataset ${DATASET_NAME} --name $NAME --network resnet18 --finetune_ckpt $FINETUNE_DIR/ckpt.pth --delta_ckpt $DELTA_DIR/ckpt.pth --weight_ckpt $WEIGHT_DIR/ckpt.pth --ncprune_ckpt $NCPRUNE_DIR/ckpt.pth --output_dir results/nc_adv_eval_test/${STRATEGY}_${COVERAGE}_${DATASET_ABBR}_resnet18 --batch_size 32 --coverage $COVERAGE --strategy $STRATEGY 

done
done
done