#!/usr/bin/env bash
GPU_ID=2
data_dir=/home/houwx/tl/datasets/office31
# Office31
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config DANN/DANN.yaml --data_dir $data_dir --src_domain dslr --tgt_domain amazon | tee DANN_D2A.log
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config DANN/DANN.yaml --data_dir $data_dir --src_domain dslr --tgt_domain webcam | tee DANN_D2W.log

CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config DANN/DANN.yaml --data_dir $data_dir --src_domain amazon --tgt_domain dslr | tee DANN_A2D.log
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config DANN/DANN.yaml --data_dir $data_dir --src_domain amazon --tgt_domain webcam | tee DANN_A2W.log

CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config DANN/DANN.yaml --data_dir $data_dir --src_domain webcam --tgt_domain amazon | tee DANN_W2A.log
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config DANN/DANN.yaml --data_dir $data_dir --src_domain webcam --tgt_domain dslr | tee DANN_W2D.log


data_dir=/home/houwx/tl/datasets/office-home
# Office-Home
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config DANN/DANN.yaml --data_dir $data_dir --src_domain Art --tgt_domain Clipart | tee DANN_A2C.log
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config DANN/DANN.yaml --data_dir $data_dir --src_domain Art --tgt_domain Real_World | tee DANN_A2R.log
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config DANN/DANN.yaml --data_dir $data_dir --src_domain Art --tgt_domain Product | tee DANN_A2P.log

CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config DANN/DANN.yaml --data_dir $data_dir --src_domain Clipart --tgt_domain Art | tee DANN_C2A.log
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config DANN/DANN.yaml --data_dir $data_dir --src_domain Clipart --tgt_domain Real_World | tee DANN_C2R.log
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config DANN/DANN.yaml --data_dir $data_dir --src_domain Clipart --tgt_domain Product | tee DANN_C2P.log

CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config DANN/DANN.yaml --data_dir $data_dir --src_domain Product --tgt_domain Art | tee DANN_P2A.log
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config DANN/DANN.yaml --data_dir $data_dir --src_domain Product --tgt_domain Real_World | tee DANN_P2R.log
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config DANN/DANN.yaml --data_dir $data_dir --src_domain Product --tgt_domain Clipart | tee DANN_P2C.log

CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config DANN/DANN.yaml --data_dir $data_dir --src_domain Real_World --tgt_domain Art | tee DANN_R2A.log
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config DANN/DANN.yaml --data_dir $data_dir --src_domain Real_World --tgt_domain Product | tee DANN_R2P.log
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config DANN/DANN.yaml --data_dir $data_dir --src_domain Real_World --tgt_domain Clipart | tee DANN_R2C.log
