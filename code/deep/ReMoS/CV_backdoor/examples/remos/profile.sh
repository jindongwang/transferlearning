

#!/bin/bash
export PYTHONPATH=../..:$PYTHONPATH

iter=30000
lr=5e-3
wd=1e-4
mmt=0

DATASETS=(MIT_67 CUB_200_2011 stanford_40 )
DATASET_NAMES=(MIT67Data CUB200Data Stanford40Data)
DATASET_ABBRS=(mit67 cub200 stanford40)

COVERAGE=strong_coverage
STRATEGY=deepxplore

# neuron_coverage top_k_coverage strong_coverage
# random deepxplore dlfuzz dlfuzzfirst

for i in 0
do

for COVERAGE in neuron_coverage 
do

    DATASET=${DATASETS[i]}
    DATASET_NAME=${DATASET_NAMES[i]}
    DATASET_ABBR=${DATASET_ABBRS[i]}

    NAME=resnet18_${DATASET_ABBR}_lr${lr}_iter${iter}_wd${wd}_mmt${mmt}

    CUDA_VISIBLE_DEVICES=$1 \
    python -u remos/my_profile.py \
    --datapath ../CV_adv/data/${DATASET}/ \
    --dataset ${DATASET_NAME} \
    --name $NAME \
    --network resnet50 \
    --output_dir results/nc_profiling/${COVERAGE}_${DATASET_ABBR}_resnet50 \
    --batch_size 32 \
    --coverage $COVERAGE \
    --strategy $STRATEGY \
    --checkpoint results/r50_backdoor/backdoor/res50_${DATASET_ABBR}/teacher_ckpt.pth \
    # &


done
done