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

# MLDG 
python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net --task $task --output $output \
--test_envs $test_envs --dataset $dataset --algorithm ${algorithm[i]} --mldg_beta 10

# Group_DRO
python train.py --data_dir ~/myexp30609/data/PACS/ --max_epoch 3 --net resnet18 --task img_dg --output ~/tmp/test00 \
--test_envs 0 --dataset PACS --algorithm GroupDRO --groupdro_eta 1 

# ANDMask
python train.py --data_dir ~/myexp30609/data/PACS/ --max_epoch 3 --net resnet18 --task img_dg --output ~/tmp/test00 \
--test_envs 0 --dataset PACS --algorithm ANDMask --tau 1 