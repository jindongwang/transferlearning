#!/usr/bin/env bash
GPU_ID=3
data_dir=/home/houwx/tl/datasets/office31
# Office31
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config DSAN/DSAN.yaml --data_dir $data_dir --src_domain dslr --tgt_domain amazon | tee DSAN_D2A.log
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config DSAN/DSAN.yaml --data_dir $data_dir --src_domain dslr --tgt_domain webcam | tee DSAN_D2W.log

CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config DSAN/DSAN.yaml --data_dir $data_dir --src_domain amazon --tgt_domain dslr | tee DSAN_A2D.log
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config DSAN/DSAN.yaml --data_dir $data_dir --src_domain amazon --tgt_domain webcam | tee DSAN_A2W.log

CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config DSAN/DSAN.yaml --data_dir $data_dir --src_domain webcam --tgt_domain amazon | tee DSAN_W2A.log
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config DSAN/DSAN.yaml --data_dir $data_dir --src_domain webcam --tgt_domain dslr | tee DSAN_W2D.log


data_dir=/home/houwx/tl/datasets/office-home
# Office-Home
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config DSAN/DSAN.yaml --data_dir $data_dir --src_domain Art --tgt_domain Clipart | tee DSAN_A2C.log
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config DSAN/DSAN.yaml --data_dir $data_dir --src_domain Art --tgt_domain Real_World | tee DSAN_A2R.log
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config DSAN/DSAN.yaml --data_dir $data_dir --src_domain Art --tgt_domain Product | tee DSAN_A2P.log

CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config DSAN/DSAN.yaml --data_dir $data_dir --src_domain Clipart --tgt_domain Art | tee DSAN_C2A.log
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config DSAN/DSAN.yaml --data_dir $data_dir --src_domain Clipart --tgt_domain Real_World | tee DSAN_C2R.log
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config DSAN/DSAN.yaml --data_dir $data_dir --src_domain Clipart --tgt_domain Product | tee DSAN_C2P.log

CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config DSAN/DSAN.yaml --data_dir $data_dir --src_domain Product --tgt_domain Art | tee DSAN_P2A.log
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config DSAN/DSAN.yaml --data_dir $data_dir --src_domain Product --tgt_domain Real_World | tee DSAN_P2R.log
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config DSAN/DSAN.yaml --data_dir $data_dir --src_domain Product --tgt_domain Clipart | tee DSAN_P2C.log

CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config DSAN/DSAN.yaml --data_dir $data_dir --src_domain Real_World --tgt_domain Art | tee DSAN_R2A.log
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config DSAN/DSAN.yaml --data_dir $data_dir --src_domain Real_World --tgt_domain Product | tee DSAN_R2P.log
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config DSAN/DSAN.yaml --data_dir $data_dir --src_domain Real_World --tgt_domain Clipart | tee DSAN_R2C.log