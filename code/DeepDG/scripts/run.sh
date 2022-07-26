dataset='PACS'
algorithm=('MLDG' 'ERM' 'DANN' 'RSC' 'Mixup' 'MMD' 'CORAL' 'VREx') 
test_envs=2
gpu_ids=0
data_dir='/home/lw/lw/data/PACS/'
max_epoch=2
net='resnet18'
task='img_dg'
output='/home/lw/lw/data/train_output/test'

i=0

# MLDG 
python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net --task $task --output $output \
--test_envs $test_envs --dataset $dataset --algorithm ${algorithm[i]} --mldg_beta 10

# Group_DRO
python train.py --data_dir ~/myexp30609/data/PACS/ --max_epoch 3 --net resnet18 --task img_dg --output ~/tmp/test00 \
--test_envs 0 --dataset PACS --algorithm GroupDRO --groupdro_eta 1 

# ANDMask
python train.py --data_dir /home/lw/lw/data/PACS/ --max_epoch 3 --net resnet18 --task img_dg --output /home/lw/lw/test00 \
--test_envs 0 --dataset PACS --algorithm ANDMask --tau 1 

# The following experiments are running on the singularity cluster of MSRA.The environment are shown in the following file.
# CUDA version, wget https://dgresearchredmond.blob.core.windows.net/amulet/projects/lw/singularity/env/env1.txt
# GPU information, wget https://dgresearchredmond.blob.core.windows.net/amulet/projects/lw/singularity/env/env2.txt
# python package information, wget https://dgresearchredmond.blob.core.windows.net/amulet/projects/lw/singularity/env/env.txt
python train.py --data_dir /home/lw/lw/data/PACS/ --max_epoch 120 --net resnet18 --checkpoint_freq 1 --task img_dg --output /home/lw/lw/data/train_output/difexpacs/2-0 \
--test_envs 0 --dataset PACS --algorithm DIFEX --alpha 0.1 --beta 0.01 --lam 0.1 --disttype 2-norm
python train.py --data_dir /home/lw/lw/data/PACS/ --max_epoch 120 --net resnet18 --checkpoint_freq 1 --task img_dg --output /home/lw/lw/data/train_output/difexpacs/2-1 \
--test_envs 1 --dataset PACS --algorithm DIFEX --alpha 0.001 --beta 0 --lam 0.01 --disttype 2-norm
python train.py --data_dir /home/lw/lw/data/PACS/ --max_epoch 120 --net resnet18 --checkpoint_freq 1 --task img_dg --output /home/lw/lw/data/train_output/difexpacs/2-2 \
--test_envs 2 --dataset PACS --algorithm DIFEX --alpha 0.001 --beta 0.01 --lam 0.01 --disttype 2-norm
python train.py --data_dir /home/lw/lw/data/PACS/ --max_epoch 120 --net resnet18 --checkpoint_freq 1 --task img_dg --output /home/lw/lw/data/train_output/difexpacs/2-3 \
--test_envs 3 --dataset PACS --algorithm DIFEX --alpha 0.01 --beta 1 --lam 0 --disttype 2-norm

python train.py --data_dir /home/lw/lw/data/PACS/ --max_epoch 120 --net resnet18 --checkpoint_freq 1 --task img_dg --output /home/lw/lw/data/train_output/difexpacs/n1n-0 \
--test_envs 0 --dataset PACS --algorithm DIFEX --alpha 0.1 --beta 0.1 --lam 0 --disttype norm-1-norm
python train.py --data_dir /home/lw/lw/data/PACS/ --max_epoch 120 --net resnet18 --checkpoint_freq 1 --task img_dg --output /home/lw/lw/data/train_output/difexpacs/n1n-1 \
--test_envs 1 --dataset PACS --algorithm DIFEX --alpha 0.001 --beta 1 --lam 0.01 --disttype norm-1-norm
python train.py --data_dir /home/lw/lw/data/PACS/ --max_epoch 120 --net resnet18 --checkpoint_freq 1 --task img_dg --output /home/lw/lw/data/train_output/difexpacs/n1n-2 \
--test_envs 2 --dataset PACS --algorithm DIFEX --alpha 0.001 --beta 0.5 --lam 0.1 --disttype norm-1-norm
python train.py --data_dir /home/lw/lw/data/PACS/ --max_epoch 120 --net resnet18 --checkpoint_freq 1 --task img_dg --output /home/lw/lw/data/train_output/difexpacs/n1n-3 \
--test_envs 3 --dataset PACS --algorithm DIFEX --alpha 0.01 --beta 10 --lam 1 --disttype norm-1-norm
