#!/bin/bash
export PYTHONPATH=../..:$PYTHONPATH


DATASETS=(MIT_67 CUB_200_2011 stanford_40 )
DATASET_NAMES=(MIT67Data CUB200Data Stanford40Data )
DATASET_ABBRS=(mit67 cub200 stanford40 )

for METHOD in finetune
do
for i in 0
do

    DATASET=${DATASETS[i]}
    DATASET_NAME=${DATASET_NAMES[i]}
    DATASET_ABBR=${DATASET_ABBRS[i]}

    python eval.py --datapath ../CV_adv/data/${DATASET}/ --dataset ${DATASET_NAME} --network resnet50 --output_dir remos/$METHOD/$DATASET_ABBR/ --is_poison --fixed_pic 

done
done