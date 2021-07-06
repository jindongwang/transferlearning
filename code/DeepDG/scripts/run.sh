dataset='PACS'
algorithm=('MLDG' 'ERM' 'DANN' 'RSC' 'Mixup' 'MMD' 'CORAL') 
test_envs=2
gpu_ids=0
data_dir='../../data/PACS/'
max_epoch=2
net='resnet18'
task='img_dg'
output='../../data/train_output/test'

i=0

python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net --task $task --output $output \
--test_envs $test_envs --dataset $dataset --algorithm ${algorithm[i]} --mldg_beta 10
