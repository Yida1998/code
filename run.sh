#!/bin/bash
module load anaconda/2020.11
source activate torch17
export PYTHONUNBUFFERED=1
# python trainBatchNew.py --KFold=3  --root_path='./DataSet/dataset/CrossVal/dataset_1' --test_path='./DataSet/TrainSet1_224_112' --epoches=10 --train_times=3 --batchSize=512 --num_workers=2 --leastEpoch=0


# 实验二
# python train.py --train_path='./DataSet/论文/实验二/train/TrainSet2_224_dropm0.0_adaption_0.5/train' --test_path='./DataSet/论文/实验二/test/TrainSet2_224_dropm0.0_adaption_0.5' --epoches=120 --batchSize=1200 --num_workers=4 --model_name='resnet' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验二/train/TrainSet2_224_dropm0.0_adaption_0.5/val'

#python train.py --train_path='./DataSet/论文/实验二/train/TrainSet2_224_dropm0.0_adaption_1.0/train' --test_path='./DataSet/论文/实验二/test/TrainSet2_224_dropm0.0_adaption_1.0' --epoches=120 --batchSize=1200 --num_workers=4 --model_name='resnet' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验二/train/TrainSet2_224_dropm0.0_adaption_1.0/val'

#python train.py --train_path='./DataSet/论文/实验二/train/TrainSet2_224_dropm0.0_adaption_1.5/train' --test_path='./DataSet/论文/实验二/test/TrainSet2_224_dropm0.0_adaption_1.5' --epoches=120 --batchSize=1200 --num_workers=4 --model_name='resnet' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验二/train/TrainSet2_224_dropm0.0_adaption_1.5/val'

#python train.py --train_path='./DataSet/论文/实验二/train/TrainSet2_224_dropm0.0_adaption_2.0/train' --test_path='./DataSet/论文/实验二/test/TrainSet2_224_dropm0.0_adaption_2.0' --epoches=120 --batchSize=1200 --num_workers=4 --model_name='resnet' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验二/train/TrainSet2_224_dropm0.0_adaption_2.0/val'

# python train.py --train_path='./DataSet/论文/实验二/train/TrainSet2_224_dropm0.0_adaption_2.5/train' --test_path='./DataSet/论文/实验二/test/TrainSet2_224_dropm0.0_adaption_2.5' --epoches=120 --batchSize=1200 --num_workers=4 --model_name='resnet' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验二/train/TrainSet2_224_dropm0.0_adaption_2.5/val'


# 实验一
# python train.py --train_path='./DataSet/论文/实验一/train' --test_path='./DataSet/论文/实验一/test' --epoches=120 --batchSize=20 --num_workers=4 --model_name='resnet' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验一/val'  --raw_image --raw_size=1200
# 
# python train.py --train_path='./DataSet/论文/实验一/train' --test_path='./DataSet/论文/实验一/test' --epoches=120 --batchSize=20 --num_workers=4 --model_name='resnet' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验一/val'  --raw_image --raw_size=600
# 
# python train.py --train_path='./DataSet/论文/实验一/train' --test_path='./DataSet/论文/实验一/test' --epoches=120 --batchSize=20 --num_workers=4 --model_name='resnet' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验一/val'  --raw_image --raw_size=448
# 
# python train.py --train_path='./DataSet/论文/实验一/train' --test_path='./DataSet/论文/实验一/test' --epoches=120 --batchSize=20 --num_workers=4 --model_name='resnet' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验一/val'  --raw_image --raw_size=224
# 
# python train.py --train_path='./DataSet/论文/实验一/train' --test_path='./DataSet/论文/实验一/test' --epoches=120 --batchSize=20 --num_workers=4 --model_name='resnet' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验一/val'  --raw_image --raw_size=112


# 实验三  batch：300 300 800 1200 1600 2400 4800	batch：全部batch保持相同
# python train.py --train_path='./DataSet/论文/实验三/train/600/train' --test_path='./DataSet/论文/实验三/test/600' --epoches=120 --batchSize=300 --num_workers=4 --model_name='resnet' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验三/train/600/val'
# 
# python train.py --train_path='./DataSet/论文/实验三/train/448/train' --test_path='./DataSet/论文/实验三/test/448' --epoches=120 --batchSize=300 --num_workers=4 --model_name='resnet' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验三/train/448/val'
# 
# python train.py --train_path='./DataSet/论文/实验三/train/336/train' --test_path='./DataSet/论文/实验三/test/336' --epoches=120 --batchSize=300 --num_workers=4 --model_name='resnet' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验三/train/336/val'
# 
# python train.py --train_path='./DataSet/论文/实验三/train/224/train' --test_path='./DataSet/论文/实验三/test/224' --epoches=120 --batchSize=300 --num_workers=4 --model_name='resnet' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验三/train/224/val'
# 
# python train.py --train_path='./DataSet/论文/实验三/train/168/train' --test_path='./DataSet/论文/实验三/test/168' --epoches=120 --batchSize=300 --num_workers=4 --model_name='resnet' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验三/train/168/val'
# 
# python train.py --train_path='./DataSet/论文/实验三/train/112/train' --test_path='./DataSet/论文/实验三/test/112' --epoches=120 --batchSize=300 --num_workers=4 --model_name='resnet' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验三/train/112/val'
# 
# python train.py --train_path='./DataSet/论文/实验三/train/64/train' --test_path='./DataSet/论文/实验三/test/64' --epoches=120 --batchSize=300 --num_workers=4 --model_name='resnet' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验三/train/64/val'



# 重做实验一

# python train.py --train_path='./DataSet/论文/实验一/实验一New/train/D_600' --test_path='./DataSet/论文/实验一/实验一New/test/D_600' --epoches=120 --batchSize=20 --num_workers=4 --model_name='resnet' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验一/实验一New/val/D_600' --testWith_raw
# 
# python train.py --train_path='./DataSet/论文/实验一/实验一New/train/D_448' --test_path='./DataSet/论文/实验一/实验一New/test/D_448' --epoches=120 --batchSize=20 --num_workers=4 --model_name='resnet' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验一/实验一New/val/D_448' --testWith_raw
# 
# python train.py --train_path='./DataSet/论文/实验一/实验一New/train/D_336' --test_path='./DataSet/论文/实验一/实验一New/test/D_336' --epoches=120 --batchSize=20 --num_workers=4 --model_name='resnet' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验一/实验一New/val/D_336' --testWith_raw
# 
# python train.py --train_path='./DataSet/论文/实验一/实验一New/train/D_224' --test_path='./DataSet/论文/实验一/实验一New/test/D_224' --epoches=120 --batchSize=20 --num_workers=4 --model_name='resnet' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验一/实验一New/val/D_224' --testWith_raw
# 
# python train.py --train_path='./DataSet/论文/实验一/实验一New/train/D_168' --test_path='./DataSet/论文/实验一/实验一New/test/D_168' --epoches=120 --batchSize=20 --num_workers=4 --model_name='resnet' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验一/实验一New/val/D_168' --testWith_raw
# 
# python train.py --train_path='./DataSet/论文/实验一/实验一New/train/D_112' --test_path='./DataSet/论文/实验一/实验一New/test/D_112' --epoches=120 --batchSize=20 --num_workers=4 --model_name='resnet' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验一/实验一New/val/D_112' --testWith_raw
# 
# python train.py --train_path='./DataSet/论文/实验一/实验一New/train/D_64' --test_path='./DataSet/论文/实验一/实验一New/test/D_64' --epoches=120 --batchSize=20 --num_workers=4 --model_name='resnet' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验一/实验一New/val/D_64' --testWith_raw


# 实验四  test1
# python train.py --train_path='./DataSet/论文/实验四/TrainRandom/train/TrainSet2_600_dropm0.0_adaption_ramdom_1.0/train' --test_path='./DataSet/论文/实验四/TrainRandom/test/TrainSet2_600_dropm0.0_adaption_ramdom_1.0' --epoches=120 --batchSize=300 --num_workers=4 --model_name='resnet' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验四/TrainRandom/train/TrainSet2_600_dropm0.0_adaption_ramdom_1.0/val'
# 
# python train.py --train_path='./DataSet/论文/实验四/TrainRandom/train/TrainSet2_600_dropm0.0_adaption_ramdom_1.0/train' --test_path='./DataSet/论文/实验四/TrainSet/test/600' --epoches=120 --batchSize=300 --num_workers=4 --model_name='resnet' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验四/TrainSet/train/600/val'
# 
# python train.py --train_path='./DataSet/论文/实验四/TrainMake/train/TrainSet2_600_dropm0.0_adaption_1.0_broke/train' --test_path='./DataSet/论文/实验四/TrainMake/test/TrainSet2_600_dropm0.0_adaption_1.0_broke' --epoches=120 --batchSize=300 --num_workers=4 --model_name='resnet' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验四/TrainMake/train/TrainSet2_600_dropm0.0_adaption_1.0_broke/val'
# 
# python train.py --train_path='./DataSet/论文/实验四/TrainMake/train/TrainSet2_600_dropm0.0_adaption_1.0_broke/train' --test_path='./DataSet/论文/实验四/TrainSet/test/600' --epoches=120 --batchSize=300 --num_workers=4 --model_name='resnet' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验四/TrainSet/train/600/val'

# 实验四，测试模型是否可正常运行

# 实验四，数据集1 TrainSet
# python train.py --train_path='./DataSet/论文/实验四/TrainSet/train/600/train' --test_path='./DataSet/论文/实验四/TrainSet/test/600' --epoches=120 --batchSize=80 --num_workers=4 --model_name='resnet' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验四/TrainSet/train/600/val'
# 
# python train.py --train_path='./DataSet/论文/实验四/TrainSet/train/600/train' --test_path='./DataSet/论文/实验四/TrainSet/test/600' --epoches=120 --batchSize=80 --num_workers=4 --model_name='resnet50' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验四/TrainSet/train/600/val'
# 
# python train.py --train_path='./DataSet/论文/实验四/TrainSet/train/600/train' --test_path='./DataSet/论文/实验四/TrainSet/test/600' --epoches=120 --batchSize=80 --num_workers=4 --model_name='mobilenetv3' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验四/TrainSet/train/600/val'
# 
# python train.py --train_path='./DataSet/论文/实验四/TrainSet/train/600/train' --test_path='./DataSet/论文/实验四/TrainSet/test/600' --epoches=120 --batchSize=80 --num_workers=4 --model_name='efficientnetb0' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验四/TrainSet/train/600/val'
# 
# python train.py --train_path='./DataSet/论文/实验四/TrainSet/train/600/train' --test_path='./DataSet/论文/实验四/TrainSet/test/600' --epoches=120 --batchSize=80 --num_workers=4 --model_name='vit' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验四/TrainSet/train/600/val' --enhance


# 0907探究模型的batch

# python train.py --train_path='./DataSet/论文/实验三/train/224/train' --test_path='./DataSet/论文/实验三/test/224' --epoches=1 --batchSize=600 --num_workers=4 --model_name='resnet18' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验三/train/224/val'
# 
# python train.py --train_path='./DataSet/论文/实验三/train/224/train' --test_path='./DataSet/论文/实验三/test/224' --epoches=1 --batchSize=1200 --num_workers=4 --model_name='resnet34' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验三/train/224/val'

# python train.py --train_path='./DataSet/论文/实验三/train/224/train' --test_path='./DataSet/论文/实验三/test/224' --epoches=1 --batchSize=700 --num_workers=4 --model_name='resnet50' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验三/train/224/val'


# python train.py --train_path='./DataSet/论文/实验三/train/600/train' --test_path='./DataSet/论文/实验三/test/600' --epoches=1 --batchSize=150 --num_workers=4 --model_name='resnet50' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验三/train/600/val'

#python train.py --train_path='./DataSet/论文/实验三/train/448/train' --test_path='./DataSet/论文/实验三/test/448' --epoches=1 --batchSize=150 --num_workers=4 --model_name='resnet50' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验三/train/448/val'

#python train.py --train_path='./DataSet/论文/实验三/train/336/train' --test_path='./DataSet/论文/实验三/test/336' --epoches=1 --batchSize=300 --num_workers=4 --model_name='resnet50' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验三/train/336/val'

# python train.py --train_path='./DataSet/论文/实验三/train/224/train' --test_path='./DataSet/论文/实验三/test/224' --epoches=1 --batchSize=600 --num_workers=4 --model_name='resnet50' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验三/train/224/val'
# 
# python train.py --train_path='./DataSet/论文/实验三/train/168/train' --test_path='./DataSet/论文/实验三/test/168' --epoches=1 --batchSize=900 --num_workers=4 --model_name='resnet50' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验三/train/168/val'
# 
# python train.py --train_path='./DataSet/论文/实验三/train/112/train' --test_path='./DataSet/论文/实验三/test/112' --epoches=1 --batchSize=1200 --num_workers=4 --model_name='resnet50' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验三/train/112/val'
# 
# python train.py --train_path='./DataSet/论文/实验三/train/64/train' --test_path='./DataSet/论文/实验三/test/64' --epoches=1 --batchSize=1500 --num_workers=4 --model_name='resnet50' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验三/train/64/val'

# 0908重做实验一
# python train.py --train_path='./DataSet/论文/实验一/实验一New/train/D_560' --test_path='./DataSet/论文/实验一/实验一New/test/D_560' --epoches=120 --batchSize=40 --num_workers=4 --model_name='resnet18' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验一/实验一New/val/D_560' --testWith_raw
 
# python train.py --train_path='./DataSet/论文/实验一/实验一New/train/D_448' --test_path='./DataSet/论文/实验一/实验一New/test/D_448' --epoches=120 --batchSize=40 --num_workers=4 --model_name='resnet18' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验一/实验一New/val/D_448' --testWith_raw
 
# python train.py --train_path='./DataSet/论文/实验一/实验一New/train/D_336' --test_path='./DataSet/论文/实验一/实验一New/test/D_336' --epoches=120 --batchSize=40 --num_workers=4 --model_name='resnet18' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验一/实验一New/val/D_336' --testWith_raw
 
# python train.py --train_path='./DataSet/论文/实验一/实验一New/train/D_224' --test_path='./DataSet/论文/实验一/实验一New/test/D_224' --epoches=120 --batchSize=40 --num_workers=4 --model_name='resnet18' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验一/实验一New/val/D_224' --testWith_raw
 
# python train.py --train_path='./DataSet/论文/实验一/实验一New/train/D_168' --test_path='./DataSet/论文/实验一/实验一New/test/D_168' --epoches=120 --batchSize=40 --num_workers=4 --model_name='resnet18' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验一/实验一New/val/D_168' --testWith_raw
 
# python train.py --train_path='./DataSet/论文/实验一/实验一New/train/D_112' --test_path='./DataSet/论文/实验一/实验一New/test/D_112' --epoches=120 --batchSize=40 --num_workers=4 --model_name='resnet18' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验一/实验一New/val/D_112' --testWith_raw

# python train.py --train_path='./DataSet/论文/实验一/实验一New/train/D_56' --test_path='./DataSet/论文/实验一/实验一New/test/D_56' --epoches=120 --batchSize=40 --num_workers=4 --model_name='resnet18' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验一/实验一New/val/D_56' --testWith_raw


# python train.py --train_path='./DataSet/论文/实验一/实验一New/train/D_560' --test_path='./DataSet/论文/实验一/实验一New/test/D_560' --epoches=120 --batchSize=40 --num_workers=4 --model_name='resnet34' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验一/实验一New/val/D_560' --testWith_raw
 
# python train.py --train_path='./DataSet/论文/实验一/实验一New/train/D_448' --test_path='./DataSet/论文/实验一/实验一New/test/D_448' --epoches=120 --batchSize=40 --num_workers=4 --model_name='resnet34' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验一/实验一New/val/D_448' --testWith_raw
 
# python train.py --train_path='./DataSet/论文/实验一/实验一New/train/D_336' --test_path='./DataSet/论文/实验一/实验一New/test/D_336' --epoches=120 --batchSize=40 --num_workers=4 --model_name='resnet34' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验一/实验一New/val/D_336' --testWith_raw
 
# python train.py --train_path='./DataSet/论文/实验一/实验一New/train/D_224' --test_path='./DataSet/论文/实验一/实验一New/test/D_224' --epoches=120 --batchSize=40 --num_workers=4 --model_name='resnet34' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验一/实验一New/val/D_224' --testWith_raw
 
# python train.py --train_path='./DataSet/论文/实验一/实验一New/train/D_168' --test_path='./DataSet/论文/实验一/实验一New/test/D_168' --epoches=120 --batchSize=40 --num_workers=4 --model_name='resnet34' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验一/实验一New/val/D_168' --testWith_raw
 
# python train.py --train_path='./DataSet/论文/实验一/实验一New/train/D_112' --test_path='./DataSet/论文/实验一/实验一New/test/D_112' --epoches=120 --batchSize=40 --num_workers=4 --model_name='resnet34' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验一/实验一New/val/D_112' --testWith_raw

# python train.py --train_path='./DataSet/论文/实验一/实验一New/train/D_56' --test_path='./DataSet/论文/实验一/实验一New/test/D_56' --epoches=120 --batchSize=40 --num_workers=4 --model_name='resnet34' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验一/实验一New/val/D_56' --testWith_raw


# python train.py --train_path='./DataSet/论文/实验一/实验一New/train/D_560' --test_path='./DataSet/论文/实验一/实验一New/test/D_560' --epoches=120 --batchSize=40 --num_workers=4 --model_name='resnet50' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验一/实验一New/val/D_560' --testWith_raw
 
# python train.py --train_path='./DataSet/论文/实验一/实验一New/train/D_448' --test_path='./DataSet/论文/实验一/实验一New/test/D_448' --epoches=120 --batchSize=40 --num_workers=4 --model_name='resnet50' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验一/实验一New/val/D_448' --testWith_raw
 
# python train.py --train_path='./DataSet/论文/实验一/实验一New/train/D_336' --test_path='./DataSet/论文/实验一/实验一New/test/D_336' --epoches=120 --batchSize=40 --num_workers=4 --model_name='resnet50' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验一/实验一New/val/D_336' --testWith_raw
 
# python train.py --train_path='./DataSet/论文/实验一/实验一New/train/D_224' --test_path='./DataSet/论文/实验一/实验一New/test/D_224' --epoches=120 --batchSize=40 --num_workers=4 --model_name='resnet50' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验一/实验一New/val/D_224' --testWith_raw
 
# python train.py --train_path='./DataSet/论文/实验一/实验一New/train/D_168' --test_path='./DataSet/论文/实验一/实验一New/test/D_168' --epoches=120 --batchSize=40 --num_workers=4 --model_name='resnet50' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验一/实验一New/val/D_168' --testWith_raw
 
# python train.py --train_path='./DataSet/论文/实验一/实验一New/train/D_112' --test_path='./DataSet/论文/实验一/实验一New/test/D_112' --epoches=120 --batchSize=40 --num_workers=4 --model_name='resnet50' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验一/实验一New/val/D_112' --testWith_raw

# python train.py --train_path='./DataSet/论文/实验一/实验一New/train/D_56' --test_path='./DataSet/论文/实验一/实验一New/test/D_56' --epoches=120 --batchSize=40 --num_workers=4 --model_name='resnet50' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验一/实验一New/val/D_56' --testWith_raw


# 重做实验二
# python train.py --train_path='./DataSet/论文/实验二/train/TrainSet2_224_dropm0.0_adaption_0.5/train' --test_path='./DataSet/论文/实验二/test/TrainSet2_224_dropm0.0_adaption_0.5' --epoches=120 --batchSize=600 --num_workers=4 --model_name='resnet18' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验二/train/TrainSet2_224_dropm0.0_adaption_0.5/val'
# 
# python train.py --train_path='./DataSet/论文/实验二/train/TrainSet2_224_dropm0.0_adaption_1.0/train' --test_path='./DataSet/论文/实验二/test/TrainSet2_224_dropm0.0_adaption_1.0' --epoches=120 --batchSize=600 --num_workers=4 --model_name='resnet18' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验二/train/TrainSet2_224_dropm0.0_adaption_1.0/val'
# 
# python train.py --train_path='./DataSet/论文/实验二/train/TrainSet2_224_dropm0.0_adaption_1.5/train' --test_path='./DataSet/论文/实验二/test/TrainSet2_224_dropm0.0_adaption_1.5' --epoches=120 --batchSize=600 --num_workers=4 --model_name='resnet18' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验二/train/TrainSet2_224_dropm0.0_adaption_1.5/val'
# 
# python train.py --train_path='./DataSet/论文/实验二/train/TrainSet2_224_dropm0.0_adaption_2.0/train' --test_path='./DataSet/论文/实验二/test/TrainSet2_224_dropm0.0_adaption_2.0' --epoches=120 --batchSize=600 --num_workers=4 --model_name='resnet18' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验二/train/TrainSet2_224_dropm0.0_adaption_2.0/val'
# 
# python train.py --train_path='./DataSet/论文/实验二/train/TrainSet2_224_dropm0.0_adaption_2.5/train' --test_path='./DataSet/论文/实验二/test/TrainSet2_224_dropm0.0_adaption_2.5' --epoches=120 --batchSize=600 --num_workers=4 --model_name='resnet18' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验二/train/TrainSet2_224_dropm0.0_adaption_2.5/val'
# 
# 
# python train.py --train_path='./DataSet/论文/实验二/train/TrainSet2_224_dropm0.0_adaption_0.5/train' --test_path='./DataSet/论文/实验二/test/TrainSet2_224_dropm0.0_adaption_0.5' --epoches=120 --batchSize=600 --num_workers=4 --model_name='resnet34' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验二/train/TrainSet2_224_dropm0.0_adaption_0.5/val'
# 
# python train.py --train_path='./DataSet/论文/实验二/train/TrainSet2_224_dropm0.0_adaption_1.0/train' --test_path='./DataSet/论文/实验二/test/TrainSet2_224_dropm0.0_adaption_1.0' --epoches=120 --batchSize=600 --num_workers=4 --model_name='resnet34' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验二/train/TrainSet2_224_dropm0.0_adaption_1.0/val'
# 
# python train.py --train_path='./DataSet/论文/实验二/train/TrainSet2_224_dropm0.0_adaption_1.5/train' --test_path='./DataSet/论文/实验二/test/TrainSet2_224_dropm0.0_adaption_1.5' --epoches=120 --batchSize=600 --num_workers=4 --model_name='resnet34' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验二/train/TrainSet2_224_dropm0.0_adaption_1.5/val'
# 
# python train.py --train_path='./DataSet/论文/实验二/train/TrainSet2_224_dropm0.0_adaption_2.0/train' --test_path='./DataSet/论文/实验二/test/TrainSet2_224_dropm0.0_adaption_2.0' --epoches=120 --batchSize=600 --num_workers=4 --model_name='resnet34' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验二/train/TrainSet2_224_dropm0.0_adaption_2.0/val'
# 
# python train.py --train_path='./DataSet/论文/实验二/train/TrainSet2_224_dropm0.0_adaption_2.5/train' --test_path='./DataSet/论文/实验二/test/TrainSet2_224_dropm0.0_adaption_2.5' --epoches=120 --batchSize=600 --num_workers=4 --model_name='resnet34' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验二/train/TrainSet2_224_dropm0.0_adaption_2.5/val'
# 
# 
# python train.py --train_path='./DataSet/论文/实验二/train/TrainSet2_224_dropm0.0_adaption_0.5/train' --test_path='./DataSet/论文/实验二/test/TrainSet2_224_dropm0.0_adaption_0.5' --epoches=120 --batchSize=600 --num_workers=4 --model_name='resnet50' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验二/train/TrainSet2_224_dropm0.0_adaption_0.5/val'
# 
# python train.py --train_path='./DataSet/论文/实验二/train/TrainSet2_224_dropm0.0_adaption_1.0/train' --test_path='./DataSet/论文/实验二/test/TrainSet2_224_dropm0.0_adaption_1.0' --epoches=120 --batchSize=600 --num_workers=4 --model_name='resnet50' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验二/train/TrainSet2_224_dropm0.0_adaption_1.0/val'
# 
# python train.py --train_path='./DataSet/论文/实验二/train/TrainSet2_224_dropm0.0_adaption_1.5/train' --test_path='./DataSet/论文/实验二/test/TrainSet2_224_dropm0.0_adaption_1.5' --epoches=120 --batchSize=600 --num_workers=4 --model_name='resnet50' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验二/train/TrainSet2_224_dropm0.0_adaption_1.5/val'
# 
# python train.py --train_path='./DataSet/论文/实验二/train/TrainSet2_224_dropm0.0_adaption_2.0/train' --test_path='./DataSet/论文/实验二/test/TrainSet2_224_dropm0.0_adaption_2.0' --epoches=120 --batchSize=600 --num_workers=4 --model_name='resnet50' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验二/train/TrainSet2_224_dropm0.0_adaption_2.0/val'
# 
# python train.py --train_path='./DataSet/论文/实验二/train/TrainSet2_224_dropm0.0_adaption_2.5/train' --test_path='./DataSet/论文/实验二/test/TrainSet2_224_dropm0.0_adaption_2.5' --epoches=120 --batchSize=600 --num_workers=4 --model_name='resnet50' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验二/train/TrainSet2_224_dropm0.0_adaption_2.5/val'

# 0908 19:52 重做实验一 验证集比例设为与训练集相近似

# python train.py --train_path='./DataSet/论文/实验一/实验一35val/train/D_560' --test_path='./DataSet/论文/实验一/实验一35val/test/D_560' --epoches=120 --batchSize=60 --num_workers=4 --model_name='resnet18' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验一/实验一35val/val/D_560' --testWith_raw
 
# python train.py --train_path='./DataSet/论文/实验一/实验一35val/train/D_448' --test_path='./DataSet/论文/实验一/实验一35val/test/D_448' --epoches=120 --batchSize=60 --num_workers=4 --model_name='resnet18' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验一/实验一35val/val/D_448' --testWith_raw
 
# python train.py --train_path='./DataSet/论文/实验一/实验一35val/train/D_336' --test_path='./DataSet/论文/实验一/实验一35val/test/D_336' --epoches=120 --batchSize=60 --num_workers=4 --model_name='resnet18' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验一/实验一35val/val/D_336' --testWith_raw
 
# python train.py --train_path='./DataSet/论文/实验一/实验一35val/train/D_224' --test_path='./DataSet/论文/实验一/实验一35val/test/D_224' --epoches=120 --batchSize=60 --num_workers=4 --model_name='resnet18' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验一/实验一35val/val/D_224' --testWith_raw
 
# python train.py --train_path='./DataSet/论文/实验一/实验一35val/train/D_168' --test_path='./DataSet/论文/实验一/实验一35val/test/D_168' --epoches=120 --batchSize=60 --num_workers=4 --model_name='resnet18' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验一/实验一35val/val/D_168' --testWith_raw
 
# python train.py --train_path='./DataSet/论文/实验一/实验一35val/train/D_112' --test_path='./DataSet/论文/实验一/实验一35val/test/D_112' --epoches=120 --batchSize=60 --num_workers=4 --model_name='resnet18' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验一/实验一35val/val/D_112' --testWith_raw

# python train.py --train_path='./DataSet/论文/实验一/实验一35val/train/D_56' --test_path='./DataSet/论文/实验一/实验一35val/test/D_56' --epoches=120 --batchSize=60 --num_workers=4 --model_name='resnet18' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验一/实验一35val/val/D_56' --testWith_raw


# python train.py --train_path='./DataSet/论文/实验一/实验一35val/train/D_560' --test_path='./DataSet/论文/实验一/实验一35val/test/D_560' --epoches=120 --batchSize=60 --num_workers=4 --model_name='resnet34' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验一/实验一35val/val/D_560' --testWith_raw
 
# python train.py --train_path='./DataSet/论文/实验一/实验一35val/train/D_448' --test_path='./DataSet/论文/实验一/实验一35val/test/D_448' --epoches=120 --batchSize=60 --num_workers=4 --model_name='resnet34' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验一/实验一35val/val/D_448' --testWith_raw
 
# python train.py --train_path='./DataSet/论文/实验一/实验一35val/train/D_336' --test_path='./DataSet/论文/实验一/实验一35val/test/D_336' --epoches=120 --batchSize=60 --num_workers=4 --model_name='resnet34' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验一/实验一35val/val/D_336' --testWith_raw
 
# python train.py --train_path='./DataSet/论文/实验一/实验一35val/train/D_224' --test_path='./DataSet/论文/实验一/实验一35val/test/D_224' --epoches=120 --batchSize=60 --num_workers=4 --model_name='resnet34' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验一/实验一35val/val/D_224' --testWith_raw
 
# python train.py --train_path='./DataSet/论文/实验一/实验一35val/train/D_168' --test_path='./DataSet/论文/实验一/实验一35val/test/D_168' --epoches=120 --batchSize=60 --num_workers=4 --model_name='resnet34' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验一/实验一35val/val/D_168' --testWith_raw
 
# python train.py --train_path='./DataSet/论文/实验一/实验一35val/train/D_112' --test_path='./DataSet/论文/实验一/实验一35val/test/D_112' --epoches=120 --batchSize=60 --num_workers=4 --model_name='resnet34' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验一/实验一35val/val/D_112' --testWith_raw

# python train.py --train_path='./DataSet/论文/实验一/实验一35val/train/D_56' --test_path='./DataSet/论文/实验一/实验一35val/test/D_56' --epoches=120 --batchSize=60 --num_workers=4 --model_name='resnet34' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验一/实验一35val/val/D_56' --testWith_raw


# python train.py --train_path='./DataSet/论文/实验一/实验一35val/train/D_560' --test_path='./DataSet/论文/实验一/实验一35val/test/D_560' --epoches=120 --batchSize=60 --num_workers=4 --model_name='resnet50' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验一/实验一35val/val/D_560' --testWith_raw
 
# python train.py --train_path='./DataSet/论文/实验一/实验一35val/train/D_448' --test_path='./DataSet/论文/实验一/实验一35val/test/D_448' --epoches=120 --batchSize=60 --num_workers=4 --model_name='resnet50' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验一/实验一35val/val/D_448' --testWith_raw
 
# python train.py --train_path='./DataSet/论文/实验一/实验一35val/train/D_336' --test_path='./DataSet/论文/实验一/实验一35val/test/D_336' --epoches=120 --batchSize=60 --num_workers=4 --model_name='resnet50' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验一/实验一35val/val/D_336' --testWith_raw
 
# python train.py --train_path='./DataSet/论文/实验一/实验一35val/train/D_224' --test_path='./DataSet/论文/实验一/实验一35val/test/D_224' --epoches=120 --batchSize=60 --num_workers=4 --model_name='resnet50' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验一/实验一35val/val/D_224' --testWith_raw
 
# python train.py --train_path='./DataSet/论文/实验一/实验一35val/train/D_168' --test_path='./DataSet/论文/实验一/实验一35val/test/D_168' --epoches=120 --batchSize=60 --num_workers=4 --model_name='resnet50' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验一/实验一35val/val/D_168' --testWith_raw
 
# python train.py --train_path='./DataSet/论文/实验一/实验一35val/train/D_112' --test_path='./DataSet/论文/实验一/实验一35val/test/D_112' --epoches=120 --batchSize=60 --num_workers=4 --model_name='resnet50' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验一/实验一35val/val/D_112' --testWith_raw

# python train.py --train_path='./DataSet/论文/实验一/实验一35val/train/D_56' --test_path='./DataSet/论文/实验一/实验一35val/test/D_56' --epoches=120 --batchSize=60 --num_workers=4 --model_name='resnet50' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验一/实验一35val/val/D_56' --testWith_raw


# 0909 重做实验3
# python train.py --train_path='./DataSet/论文/实验三/train/560/train' --test_path='./DataSet/论文/实验三/test/560' --epoches=120 --batchSize=100 --num_workers=4 --model_name='resnet18' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验三/train/560/val'

# python train.py --train_path='./DataSet/论文/实验三/train/448/train' --test_path='./DataSet/论文/实验三/test/448' --epoches=120 --batchSize=150 --num_workers=4 --model_name='resnet18' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验三/train/448/val'

# python train.py --train_path='./DataSet/论文/实验三/train/336/train' --test_path='./DataSet/论文/实验三/test/336' --epoches=120 --batchSize=300 --num_workers=4 --model_name='resnet18' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验三/train/336/val'

# python train.py --train_path='./DataSet/论文/实验三/train/224/train' --test_path='./DataSet/论文/实验三/test/224' --epoches=120 --batchSize=600 --num_workers=4 --model_name='resnet18' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验三/train/224/val'

# python train.py --train_path='./DataSet/论文/实验三/train/168/train' --test_path='./DataSet/论文/实验三/test/168' --epoches=120 --batchSize=900 --num_workers=4 --model_name='resnet18' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验三/train/168/val'

# python train.py --train_path='./DataSet/论文/实验三/train/112/train' --test_path='./DataSet/论文/实验三/test/112' --epoches=120 --batchSize=1200 --num_workers=4 --model_name='resnet18' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验三/train/112/val'

# python train.py --train_path='./DataSet/论文/实验三/train/56/train' --test_path='./DataSet/论文/实验三/test/56' --epoches=120 --batchSize=1500 --num_workers=4 --model_name='resnet18' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验三/train/56/val'



# python train.py --train_path='./DataSet/论文/实验三/train/560/train' --test_path='./DataSet/论文/实验三/test/560' --epoches=120 --batchSize=100 --num_workers=4 --model_name='resnet34' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验三/train/560/val'

# python train.py --train_path='./DataSet/论文/实验三/train/448/train' --test_path='./DataSet/论文/实验三/test/448' --epoches=120 --batchSize=150 --num_workers=4 --model_name='resnet34' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验三/train/448/val'

# python train.py --train_path='./DataSet/论文/实验三/train/336/train' --test_path='./DataSet/论文/实验三/test/336' --epoches=120 --batchSize=300 --num_workers=4 --model_name='resnet34' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验三/train/336/val'

# python train.py --train_path='./DataSet/论文/实验三/train/224/train' --test_path='./DataSet/论文/实验三/test/224' --epoches=120 --batchSize=600 --num_workers=4 --model_name='resnet34' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验三/train/224/val'

# python train.py --train_path='./DataSet/论文/实验三/train/168/train' --test_path='./DataSet/论文/实验三/test/168' --epoches=120 --batchSize=900 --num_workers=4 --model_name='resnet34' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验三/train/168/val'

# python train.py --train_path='./DataSet/论文/实验三/train/112/train' --test_path='./DataSet/论文/实验三/test/112' --epoches=120 --batchSize=1200 --num_workers=4 --model_name='resnet34' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验三/train/112/val'

# python train.py --train_path='./DataSet/论文/实验三/train/56/train' --test_path='./DataSet/论文/实验三/test/56' --epoches=120 --batchSize=1500 --num_workers=4 --model_name='resnet34' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验三/train/56/val'



# python train.py --train_path='./DataSet/论文/实验三/train/560/train' --test_path='./DataSet/论文/实验三/test/560' --epoches=120 --batchSize=100 --num_workers=4 --model_name='resnet50' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验三/train/560/val'

# python train.py --train_path='./DataSet/论文/实验三/train/448/train' --test_path='./DataSet/论文/实验三/test/448' --epoches=120 --batchSize=150 --num_workers=4 --model_name='resnet50' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验三/train/448/val'

# python train.py --train_path='./DataSet/论文/实验三/train/336/train' --test_path='./DataSet/论文/实验三/test/336' --epoches=120 --batchSize=300 --num_workers=4 --model_name='resnet50' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验三/train/336/val'

# python train.py --train_path='./DataSet/论文/实验三/train/224/train' --test_path='./DataSet/论文/实验三/test/224' --epoches=120 --batchSize=600 --num_workers=4 --model_name='resnet50' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验三/train/224/val'

# python train.py --train_path='./DataSet/论文/实验三/train/168/train' --test_path='./DataSet/论文/实验三/test/168' --epoches=120 --batchSize=900 --num_workers=4 --model_name='resnet50' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验三/train/168/val'

# python train.py --train_path='./DataSet/论文/实验三/train/112/train' --test_path='./DataSet/论文/实验三/test/112' --epoches=120 --batchSize=1200 --num_workers=4 --model_name='resnet50' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验三/train/112/val'

# python train.py --train_path='./DataSet/论文/实验三/train/56/train' --test_path='./DataSet/论文/实验三/test/56' --epoches=120 --batchSize=1500 --num_workers=4 --model_name='resnet50' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验三/train/56/val'



#0909 3:04重做实验一 batch60 和batch20

# python train.py --train_path='./DataSet/论文/实验一/实验一New/train/D_560' --test_path='./DataSet/论文/实验一/实验一New/test/D_560' --epoches=120 --batchSize=60 --num_workers=4 --model_name='resnet18' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验一/实验一New/val/D_560' --testWith_raw
#  
# python train.py --train_path='./DataSet/论文/实验一/实验一New/train/D_448' --test_path='./DataSet/论文/实验一/实验一New/test/D_448' --epoches=120 --batchSize=60 --num_workers=4 --model_name='resnet18' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验一/实验一New/val/D_448' --testWith_raw
#  
# python train.py --train_path='./DataSet/论文/实验一/实验一New/train/D_336' --test_path='./DataSet/论文/实验一/实验一New/test/D_336' --epoches=120 --batchSize=60 --num_workers=4 --model_name='resnet18' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验一/实验一New/val/D_336' --testWith_raw
#  
# python train.py --train_path='./DataSet/论文/实验一/实验一New/train/D_224' --test_path='./DataSet/论文/实验一/实验一New/test/D_224' --epoches=120 --batchSize=60 --num_workers=4 --model_name='resnet18' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验一/实验一New/val/D_224' --testWith_raw
#  
# python train.py --train_path='./DataSet/论文/实验一/实验一New/train/D_168' --test_path='./DataSet/论文/实验一/实验一New/test/D_168' --epoches=120 --batchSize=60 --num_workers=4 --model_name='resnet18' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验一/实验一New/val/D_168' --testWith_raw
#  
# python train.py --train_path='./DataSet/论文/实验一/实验一New/train/D_112' --test_path='./DataSet/论文/实验一/实验一New/test/D_112' --epoches=120 --batchSize=60 --num_workers=4 --model_name='resnet18' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验一/实验一New/val/D_112' --testWith_raw
# 
# python train.py --train_path='./DataSet/论文/实验一/实验一New/train/D_56' --test_path='./DataSet/论文/实验一/实验一New/test/D_56' --epoches=120 --batchSize=60 --num_workers=4 --model_name='resnet18' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验一/实验一New/val/D_56' --testWith_raw
# 
# 
# python train.py --train_path='./DataSet/论文/实验一/实验一New/train/D_560' --test_path='./DataSet/论文/实验一/实验一New/test/D_560' --epoches=120 --batchSize=60 --num_workers=4 --model_name='resnet34' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验一/实验一New/val/D_560' --testWith_raw
#  
# python train.py --train_path='./DataSet/论文/实验一/实验一New/train/D_448' --test_path='./DataSet/论文/实验一/实验一New/test/D_448' --epoches=120 --batchSize=60 --num_workers=4 --model_name='resnet34' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验一/实验一New/val/D_448' --testWith_raw
#  
# python train.py --train_path='./DataSet/论文/实验一/实验一New/train/D_336' --test_path='./DataSet/论文/实验一/实验一New/test/D_336' --epoches=120 --batchSize=60 --num_workers=4 --model_name='resnet34' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验一/实验一New/val/D_336' --testWith_raw
#  
# python train.py --train_path='./DataSet/论文/实验一/实验一New/train/D_224' --test_path='./DataSet/论文/实验一/实验一New/test/D_224' --epoches=120 --batchSize=60 --num_workers=4 --model_name='resnet34' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验一/实验一New/val/D_224' --testWith_raw
#  
# python train.py --train_path='./DataSet/论文/实验一/实验一New/train/D_168' --test_path='./DataSet/论文/实验一/实验一New/test/D_168' --epoches=120 --batchSize=60 --num_workers=4 --model_name='resnet34' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验一/实验一New/val/D_168' --testWith_raw
#  
# python train.py --train_path='./DataSet/论文/实验一/实验一New/train/D_112' --test_path='./DataSet/论文/实验一/实验一New/test/D_112' --epoches=120 --batchSize=60 --num_workers=4 --model_name='resnet34' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验一/实验一New/val/D_112' --testWith_raw
# 
# python train.py --train_path='./DataSet/论文/实验一/实验一New/train/D_56' --test_path='./DataSet/论文/实验一/实验一New/test/D_56' --epoches=120 --batchSize=60 --num_workers=4 --model_name='resnet34' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验一/实验一New/val/D_56' --testWith_raw
# 
# 
# python train.py --train_path='./DataSet/论文/实验一/实验一New/train/D_560' --test_path='./DataSet/论文/实验一/实验一New/test/D_560' --epoches=120 --batchSize=60 --num_workers=4 --model_name='resnet50' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验一/实验一New/val/D_560' --testWith_raw
#  
# python train.py --train_path='./DataSet/论文/实验一/实验一New/train/D_448' --test_path='./DataSet/论文/实验一/实验一New/test/D_448' --epoches=120 --batchSize=60 --num_workers=4 --model_name='resnet50' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验一/实验一New/val/D_448' --testWith_raw
#  
# python train.py --train_path='./DataSet/论文/实验一/实验一New/train/D_336' --test_path='./DataSet/论文/实验一/实验一New/test/D_336' --epoches=120 --batchSize=60 --num_workers=4 --model_name='resnet50' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验一/实验一New/val/D_336' --testWith_raw
#  
# python train.py --train_path='./DataSet/论文/实验一/实验一New/train/D_224' --test_path='./DataSet/论文/实验一/实验一New/test/D_224' --epoches=120 --batchSize=60 --num_workers=4 --model_name='resnet50' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验一/实验一New/val/D_224' --testWith_raw
#  
# python train.py --train_path='./DataSet/论文/实验一/实验一New/train/D_168' --test_path='./DataSet/论文/实验一/实验一New/test/D_168' --epoches=120 --batchSize=60 --num_workers=4 --model_name='resnet50' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验一/实验一New/val/D_168' --testWith_raw
#  
# python train.py --train_path='./DataSet/论文/实验一/实验一New/train/D_112' --test_path='./DataSet/论文/实验一/实验一New/test/D_112' --epoches=120 --batchSize=60 --num_workers=4 --model_name='resnet50' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验一/实验一New/val/D_112' --testWith_raw
# 
# python train.py --train_path='./DataSet/论文/实验一/实验一New/train/D_56' --test_path='./DataSet/论文/实验一/实验一New/test/D_56' --epoches=120 --batchSize=60 --num_workers=4 --model_name='resnet50' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验一/实验一New/val/D_56' --testWith_raw
# 
# 
# 
# 
# python train.py --train_path='./DataSet/论文/实验一/实验一New/train/D_560' --test_path='./DataSet/论文/实验一/实验一New/test/D_560' --epoches=120 --batchSize=20 --num_workers=4 --model_name='resnet18' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验一/实验一New/val/D_560' --testWith_raw
#  
# python train.py --train_path='./DataSet/论文/实验一/实验一New/train/D_448' --test_path='./DataSet/论文/实验一/实验一New/test/D_448' --epoches=120 --batchSize=20 --num_workers=4 --model_name='resnet18' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验一/实验一New/val/D_448' --testWith_raw
#  
# python train.py --train_path='./DataSet/论文/实验一/实验一New/train/D_336' --test_path='./DataSet/论文/实验一/实验一New/test/D_336' --epoches=120 --batchSize=20 --num_workers=4 --model_name='resnet18' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验一/实验一New/val/D_336' --testWith_raw
#  
# python train.py --train_path='./DataSet/论文/实验一/实验一New/train/D_224' --test_path='./DataSet/论文/实验一/实验一New/test/D_224' --epoches=120 --batchSize=20 --num_workers=4 --model_name='resnet18' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验一/实验一New/val/D_224' --testWith_raw
#  
# python train.py --train_path='./DataSet/论文/实验一/实验一New/train/D_168' --test_path='./DataSet/论文/实验一/实验一New/test/D_168' --epoches=120 --batchSize=20 --num_workers=4 --model_name='resnet18' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验一/实验一New/val/D_168' --testWith_raw
#  
# python train.py --train_path='./DataSet/论文/实验一/实验一New/train/D_112' --test_path='./DataSet/论文/实验一/实验一New/test/D_112' --epoches=120 --batchSize=20 --num_workers=4 --model_name='resnet18' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验一/实验一New/val/D_112' --testWith_raw
# 
# python train.py --train_path='./DataSet/论文/实验一/实验一New/train/D_56' --test_path='./DataSet/论文/实验一/实验一New/test/D_56' --epoches=120 --batchSize=20 --num_workers=4 --model_name='resnet18' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验一/实验一New/val/D_56' --testWith_raw
# 
# 
# python train.py --train_path='./DataSet/论文/实验一/实验一New/train/D_560' --test_path='./DataSet/论文/实验一/实验一New/test/D_560' --epoches=120 --batchSize=20 --num_workers=4 --model_name='resnet34' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验一/实验一New/val/D_560' --testWith_raw
#  
# python train.py --train_path='./DataSet/论文/实验一/实验一New/train/D_448' --test_path='./DataSet/论文/实验一/实验一New/test/D_448' --epoches=120 --batchSize=20 --num_workers=4 --model_name='resnet34' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验一/实验一New/val/D_448' --testWith_raw
#  
# python train.py --train_path='./DataSet/论文/实验一/实验一New/train/D_336' --test_path='./DataSet/论文/实验一/实验一New/test/D_336' --epoches=120 --batchSize=20 --num_workers=4 --model_name='resnet34' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验一/实验一New/val/D_336' --testWith_raw
#  
# python train.py --train_path='./DataSet/论文/实验一/实验一New/train/D_224' --test_path='./DataSet/论文/实验一/实验一New/test/D_224' --epoches=120 --batchSize=20 --num_workers=4 --model_name='resnet34' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验一/实验一New/val/D_224' --testWith_raw
#  
# python train.py --train_path='./DataSet/论文/实验一/实验一New/train/D_168' --test_path='./DataSet/论文/实验一/实验一New/test/D_168' --epoches=120 --batchSize=20 --num_workers=4 --model_name='resnet34' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验一/实验一New/val/D_168' --testWith_raw
#  
# python train.py --train_path='./DataSet/论文/实验一/实验一New/train/D_112' --test_path='./DataSet/论文/实验一/实验一New/test/D_112' --epoches=120 --batchSize=20 --num_workers=4 --model_name='resnet34' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验一/实验一New/val/D_112' --testWith_raw
# 
# python train.py --train_path='./DataSet/论文/实验一/实验一New/train/D_56' --test_path='./DataSet/论文/实验一/实验一New/test/D_56' --epoches=120 --batchSize=20 --num_workers=4 --model_name='resnet34' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验一/实验一New/val/D_56' --testWith_raw
# 
# 
# python train.py --train_path='./DataSet/论文/实验一/实验一New/train/D_560' --test_path='./DataSet/论文/实验一/实验一New/test/D_560' --epoches=120 --batchSize=20 --num_workers=4 --model_name='resnet50' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验一/实验一New/val/D_560' --testWith_raw
#  
# python train.py --train_path='./DataSet/论文/实验一/实验一New/train/D_448' --test_path='./DataSet/论文/实验一/实验一New/test/D_448' --epoches=120 --batchSize=20 --num_workers=4 --model_name='resnet50' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验一/实验一New/val/D_448' --testWith_raw
#  
# python train.py --train_path='./DataSet/论文/实验一/实验一New/train/D_336' --test_path='./DataSet/论文/实验一/实验一New/test/D_336' --epoches=120 --batchSize=20 --num_workers=4 --model_name='resnet50' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验一/实验一New/val/D_336' --testWith_raw
#  
# python train.py --train_path='./DataSet/论文/实验一/实验一New/train/D_224' --test_path='./DataSet/论文/实验一/实验一New/test/D_224' --epoches=120 --batchSize=20 --num_workers=4 --model_name='resnet50' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验一/实验一New/val/D_224' --testWith_raw
#  
# python train.py --train_path='./DataSet/论文/实验一/实验一New/train/D_168' --test_path='./DataSet/论文/实验一/实验一New/test/D_168' --epoches=120 --batchSize=20 --num_workers=4 --model_name='resnet50' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验一/实验一New/val/D_168' --testWith_raw
#  
# python train.py --train_path='./DataSet/论文/实验一/实验一New/train/D_112' --test_path='./DataSet/论文/实验一/实验一New/test/D_112' --epoches=120 --batchSize=20 --num_workers=4 --model_name='resnet50' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验一/实验一New/val/D_112' --testWith_raw
# 
# python train.py --train_path='./DataSet/论文/实验一/实验一New/train/D_56' --test_path='./DataSet/论文/实验一/实验一New/test/D_56' --epoches=120 --batchSize=20 --num_workers=4 --model_name='resnet50' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验一/实验一New/val/D_56' --testWith_raw


# 0909  实验三：20-34测试一下batch， 最佳的batch没有找到啊！！！！！！！！ -> 已完成
# python train.py --train_path='./DataSet/论文/实验三/train/168/train' --test_path='./DataSet/论文/实验三/test/168' --epoches=10 --batchSize=1060 --num_workers=4 --model_name='resnet50' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验三/train/168/val'


# 0910 1-05 实验三 重新提交，待完成，已重新调整batch
# python train.py --train_path='./DataSet/论文/实验三/train/560/train' --test_path='./DataSet/论文/实验三/test/560' --epoches=120 --batchSize=96 --num_workers=4 --model_name='resnet18' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验三/train/560/val'

# python train.py --train_path='./DataSet/论文/实验三/train/448/train' --test_path='./DataSet/论文/实验三/test/448' --epoches=120 --batchSize=150 --num_workers=4 --model_name='resnet18' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验三/train/448/val'

# python train.py --train_path='./DataSet/论文/实验三/train/336/train' --test_path='./DataSet/论文/实验三/test/336' --epoches=120 --batchSize=266 --num_workers=4 --model_name='resnet18' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验三/train/336/val'

# python train.py --train_path='./DataSet/论文/实验三/train/224/train' --test_path='./DataSet/论文/实验三/test/224' --epoches=120 --batchSize=600 --num_workers=4 --model_name='resnet18' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验三/train/224/val'

# python train.py --train_path='./DataSet/论文/实验三/train/168/train' --test_path='./DataSet/论文/实验三/test/168' --epoches=120 --batchSize=1066 --num_workers=4 --model_name='resnet18' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验三/train/168/val'

# python train.py --train_path='./DataSet/论文/实验三/train/112/train' --test_path='./DataSet/论文/实验三/test/112' --epoches=120 --batchSize=2400 --num_workers=4 --model_name='resnet18' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验三/train/112/val'

# python train.py --train_path='./DataSet/论文/实验三/train/56/train' --test_path='./DataSet/论文/实验三/test/56' --epoches=120 --batchSize=9600 --num_workers=4 --model_name='resnet18' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验三/train/56/val'



# python train.py --train_path='./DataSet/论文/实验三/train/560/train' --test_path='./DataSet/论文/实验三/test/560' --epoches=120 --batchSize=96 --num_workers=4 --model_name='resnet34' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验三/train/560/val'

# python train.py --train_path='./DataSet/论文/实验三/train/448/train' --test_path='./DataSet/论文/实验三/test/448' --epoches=120 --batchSize=150 --num_workers=4 --model_name='resnet34' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验三/train/448/val'

# python train.py --train_path='./DataSet/论文/实验三/train/336/train' --test_path='./DataSet/论文/实验三/test/336' --epoches=120 --batchSize=266 --num_workers=4 --model_name='resnet34' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验三/train/336/val'

# python train.py --train_path='./DataSet/论文/实验三/train/224/train' --test_path='./DataSet/论文/实验三/test/224' --epoches=120 --batchSize=600 --num_workers=4 --model_name='resnet34' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验三/train/224/val'

# python train.py --train_path='./DataSet/论文/实验三/train/168/train' --test_path='./DataSet/论文/实验三/test/168' --epoches=120 --batchSize=1066 --num_workers=4 --model_name='resnet34' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验三/train/168/val'

# python train.py --train_path='./DataSet/论文/实验三/train/112/train' --test_path='./DataSet/论文/实验三/test/112' --epoches=120 --batchSize=2400 --num_workers=4 --model_name='resnet34' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验三/train/112/val'

# python train.py --train_path='./DataSet/论文/实验三/train/56/train' --test_path='./DataSet/论文/实验三/test/56' --epoches=120 --batchSize=9600 --num_workers=4 --model_name='resnet34' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验三/train/56/val'



# python train.py --train_path='./DataSet/论文/实验三/train/560/train' --test_path='./DataSet/论文/实验三/test/560' --epoches=120 --batchSize=96 --num_workers=4 --model_name='resnet50' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验三/train/560/val'

# python train.py --train_path='./DataSet/论文/实验三/train/448/train' --test_path='./DataSet/论文/实验三/test/448' --epoches=120 --batchSize=150 --num_workers=4 --model_name='resnet50' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验三/train/448/val'

# python train.py --train_path='./DataSet/论文/实验三/train/336/train' --test_path='./DataSet/论文/实验三/test/336' --epoches=120 --batchSize=266 --num_workers=4 --model_name='resnet50' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验三/train/336/val'

# python train.py --train_path='./DataSet/论文/实验三/train/224/train' --test_path='./DataSet/论文/实验三/test/224' --epoches=120 --batchSize=600 --num_workers=4 --model_name='resnet50' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验三/train/224/val'

# python train.py --train_path='./DataSet/论文/实验三/train/168/train' --test_path='./DataSet/论文/实验三/test/168' --epoches=120 --batchSize=1066 --num_workers=4 --model_name='resnet50' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验三/train/168/val'

# python train.py --train_path='./DataSet/论文/实验三/train/112/train' --test_path='./DataSet/论文/实验三/test/112' --epoches=120 --batchSize=2400 --num_workers=4 --model_name='resnet50' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验三/train/112/val'

# python train.py --train_path='./DataSet/论文/实验三/train/56/train' --test_path='./DataSet/论文/实验三/test/56' --epoches=120 --batchSize=9600 --num_workers=4 --model_name='resnet50' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验三/train/56/val'


# 0910 1-06 提交实验四
# python train.py --train_path='./DataSet/论文/实验四/train/Made_0.05' --test_path='./DataSet/论文/实验四/test' --epoches=120 --batchSize=600 --num_workers=4 --model_name='resnet18' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验四/val'

# python train.py --train_path='./DataSet/论文/实验四/train/Made_0.1' --test_path='./DataSet/论文/实验四/test' --epoches=120 --batchSize=600 --num_workers=4 --model_name='resnet18' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验四/val'

# python train.py --train_path='./DataSet/论文/实验四/train/Made_0.2' --test_path='./DataSet/论文/实验四/test' --epoches=120 --batchSize=600 --num_workers=4 --model_name='resnet18' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验四/val'

# python train.py --train_path='./DataSet/论文/实验四/train/Made_0.3' --test_path='./DataSet/论文/实验四/test' --epoches=120 --batchSize=600 --num_workers=4 --model_name='resnet18' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验四/val'

# python train.py --train_path='./DataSet/论文/实验四/train/Made_0.05' --test_path='./DataSet/论文/实验四/test' --epoches=120 --batchSize=600 --num_workers=4 --model_name='resnet34' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验四/val'

# python train.py --train_path='./DataSet/论文/实验四/train/Made_0.1' --test_path='./DataSet/论文/实验四/test' --epoches=120 --batchSize=600 --num_workers=4 --model_name='resnet34' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验四/val'

# python train.py --train_path='./DataSet/论文/实验四/train/Made_0.2' --test_path='./DataSet/论文/实验四/test' --epoches=120 --batchSize=600 --num_workers=4 --model_name='resnet34' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验四/val'

# python train.py --train_path='./DataSet/论文/实验四/train/Made_0.3' --test_path='./DataSet/论文/实验四/test' --epoches=120 --batchSize=600 --num_workers=4 --model_name='resnet34' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验四/val'

# python train.py --train_path='./DataSet/论文/实验四/train/Made_0.05' --test_path='./DataSet/论文/实验四/test' --epoches=120 --batchSize=600 --num_workers=4 --model_name='resnet50' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验四/val'

# python train.py --train_path='./DataSet/论文/实验四/train/Made_0.1' --test_path='./DataSet/论文/实验四/test' --epoches=120 --batchSize=600 --num_workers=4 --model_name='resnet50' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验四/val'

# python train.py --train_path='./DataSet/论文/实验四/train/Made_0.2' --test_path='./DataSet/论文/实验四/test' --epoches=120 --batchSize=600 --num_workers=4 --model_name='resnet50' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验四/val'

# python train.py --train_path='./DataSet/论文/实验四/train/Made_0.3' --test_path='./DataSet/论文/实验四/test' --epoches=120 --batchSize=600 --num_workers=4 --model_name='resnet50' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验四/val'


# 0918重做实验四
# python train.py --train_path='./DataSet/论文/实验四/train/0918re/Made_0.05' --test_path='./DataSet/论文/实验四/test' --epoches=120 --batchSize=600 --num_workers=4 --model_name='resnet18' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验四/val'
# 
# python train.py --train_path='./DataSet/论文/实验四/train/0918re/Made_0.1' --test_path='./DataSet/论文/实验四/test' --epoches=120 --batchSize=600 --num_workers=4 --model_name='resnet18' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验四/val'
# 
# python train.py --train_path='./DataSet/论文/实验四/train/0918re/Made_0.2' --test_path='./DataSet/论文/实验四/test' --epoches=120 --batchSize=600 --num_workers=4 --model_name='resnet18' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验四/val'
# 
# python train.py --train_path='./DataSet/论文/实验四/train/0918re/Made_0.3' --test_path='./DataSet/论文/实验四/test' --epoches=120 --batchSize=600 --num_workers=4 --model_name='resnet18' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验四/val'
# 
# python train.py --train_path='./DataSet/论文/实验四/train/0918re/Made_0.05' --test_path='./DataSet/论文/实验四/test' --epoches=120 --batchSize=600 --num_workers=4 --model_name='resnet34' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验四/val'
# 
# python train.py --train_path='./DataSet/论文/实验四/train/0918re/Made_0.1' --test_path='./DataSet/论文/实验四/test' --epoches=120 --batchSize=600 --num_workers=4 --model_name='resnet34' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验四/val'
# 
# python train.py --train_path='./DataSet/论文/实验四/train/0918re/Made_0.2' --test_path='./DataSet/论文/实验四/test' --epoches=120 --batchSize=600 --num_workers=4 --model_name='resnet34' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验四/val'
# 
# python train.py --train_path='./DataSet/论文/实验四/train/0918re/Made_0.3' --test_path='./DataSet/论文/实验四/test' --epoches=120 --batchSize=600 --num_workers=4 --model_name='resnet34' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验四/val'
# 
# python train.py --train_path='./DataSet/论文/实验四/train/0918re/Made_0.05' --test_path='./DataSet/论文/实验四/test' --epoches=120 --batchSize=600 --num_workers=4 --model_name='resnet50' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验四/val'
# 
# python train.py --train_path='./DataSet/论文/实验四/train/0918re/Made_0.1' --test_path='./DataSet/论文/实验四/test' --epoches=120 --batchSize=600 --num_workers=4 --model_name='resnet50' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验四/val'
# 
# python train.py --train_path='./DataSet/论文/实验四/train/0918re/Made_0.2' --test_path='./DataSet/论文/实验四/test' --epoches=120 --batchSize=600 --num_workers=4 --model_name='resnet50' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验四/val'
# 
# python train.py --train_path='./DataSet/论文/实验四/train/0918re/Made_0.3' --test_path='./DataSet/论文/实验四/test' --epoches=120 --batchSize=600 --num_workers=4 --model_name='resnet50' --learning_rate=0.0001 --lr_scheduler='None' --n_class=11 --val_path='./DataSet/论文/实验四/val'