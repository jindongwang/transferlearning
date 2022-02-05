#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

PROJ_ROOT="/home/userxxx/DAAN"
PROJ_NAME="tmp"
LOG_FILE="${PROJ_ROOT}/log/${PROJ_NAME}-`date +'%Y-%m-%d-%H-%M-%S'`.log"

#echo "GPU: $CUDA_VISIBLE_DEVICES" > ${LOG_FILE}
#python ${PROJ_ROOT}/train.py  >> ${LOG_FILE}  2>&1

python ${PROJ_ROOT}/train.py 2>&1 | tee -a ${LOG_FILE}
