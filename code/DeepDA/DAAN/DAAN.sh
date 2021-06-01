#!/usr/bin/env bash
GPU_ID=3
data_dir=/home/houwx/tl/datasets/office31
# Office31
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config DAAN/DAAN.yaml --data_dir $data_dir --src_domain dslr --tgt_domain amazon | tee DAAN_D2A.log
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config DAAN/DAAN.yaml --data_dir $data_dir --src_domain dslr --tgt_domain webcam | tee DAAN_D2W.log

CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config DAAN/DAAN.yaml --data_dir $data_dir --src_domain amazon --tgt_domain dslr | tee DAAN_A2D.log
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config DAAN/DAAN.yaml --data_dir $data_dir --src_domain amazon --tgt_domain webcam | tee DAAN_A2W.log

CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config DAAN/DAAN.yaml --data_dir $data_dir --src_domain webcam --tgt_domain amazon | tee DAAN_W2A.log
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config DAAN/DAAN.yaml --data_dir $data_dir --src_domain webcam --tgt_domain dslr | tee DAAN_W2D.log


data_dir=/home/houwx/tl/datasets/office-home
# Office-Home
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config DAAN/DAAN.yaml --data_dir $data_dir --src_domain Art --tgt_domain Clipart | tee DAAN_A2C.log
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config DAAN/DAAN.yaml --data_dir $data_dir --src_domain Art --tgt_domain Real_World | tee DAAN_A2R.log
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config DAAN/DAAN.yaml --data_dir $data_dir --src_domain Art --tgt_domain Product | tee DAAN_A2P.log

CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config DAAN/DAAN.yaml --data_dir $data_dir --src_domain Clipart --tgt_domain Art | tee DAAN_C2A.log
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config DAAN/DAAN.yaml --data_dir $data_dir --src_domain Clipart --tgt_domain Real_World | tee DAAN_C2R.log
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config DAAN/DAAN.yaml --data_dir $data_dir --src_domain Clipart --tgt_domain Product | tee DAAN_C2P.log

CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config DAAN/DAAN.yaml --data_dir $data_dir --src_domain Product --tgt_domain Art | tee DAAN_P2A.log
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config DAAN/DAAN.yaml --data_dir $data_dir --src_domain Product --tgt_domain Real_World | tee DAAN_P2R.log
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config DAAN/DAAN.yaml --data_dir $data_dir --src_domain Product --tgt_domain Clipart | tee DAAN_P2C.log

CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config DAAN/DAAN.yaml --data_dir $data_dir --src_domain Real_World --tgt_domain Art | tee DAAN_R2A.log
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config DAAN/DAAN.yaml --data_dir $data_dir --src_domain Real_World --tgt_domain Product | tee DAAN_R2P.log
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config DAAN/DAAN.yaml --data_dir $data_dir --src_domain Real_World --tgt_domain Clipart | tee DAAN_R2C.log