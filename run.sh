# 后台运行命令 nohup bash -u run.sh > ./Log/0421.log 2>&1 &

#python trainBatch.py --KFold=5 --root_path='./DataSet/0413/KFold/TrainSet2_bins4' --test_path='./DataSet/TestSet1_224_112' --epoches=120 --train_times=2 --batchSize=256 --num_workers=4 --n_class=3 --leastEpoch=0
#python trainBatch.py --KFold=5 --root_path='./DataSet/0413/KFold/TrainSet2_bins8' --test_path='./DataSet/TestSet1_224_112' --epoches=120 --train_times=2 --batchSize=256 --num_workers=4 --n_class=3 --leastEpoch=0
#python trainBatch.py --KFold=5 --root_path='./DataSet/0413/KFold/TrainSet2_bins16' --test_path='./DataSet/TestSet1_224_112' --epoches=120 --train_times=2 --batchSize=256 --num_workers=4 --n_class=3 --leastEpoch=0
#python trainBatch.py --KFold=5 --root_path='./DataSet/0413/KFold/TrainSet2_004' --test_path='./DataSet/TestSet1_224_112' --epoches=120 --train_times=2 --batchSize=256 --num_workers=4 --n_class=3 --leastEpoch=0
#python trainBatch.py --KFold=5 --root_path='./DataSet/0413/KFold/TrainSet2_008' --test_path='./DataSet/TestSet1_224_112' --epoches=120 --train_times=2 --batchSize=256 --num_workers=4 --n_class=3 --leastEpoch=0
#python trainBatch.py --KFold=5 --root_path='./DataSet/0413/KFold/TrainSet2_016' --test_path='./DataSet/TestSet1_224_112' --epoches=120 --train_times=2 --batchSize=256 --num_workers=4 --n_class=3 --leastEpoch=0


#python train.py  --train_path='./DataSet/0413/TrainSet2_bins4' --test_path='./DataSet/TestSet1_224_112' --epoches=120 --train_times=2 --batchSize=256 --num_workers=4 --n_class=3 --leastEpoch=0
#python train.py  --train_path='./DataSet/0413/TrainSet2_bins8' --test_path='./DataSet/TestSet1_224_112' --epoches=120 --train_times=2 --batchSize=256 --num_workers=4 --n_class=3 --leastEpoch=0
#python train.py  --train_path='./DataSet/0413/TrainSet2_bins16' --test_path='./DataSet/TestSet1_224_112' --epoches=120 --train_times=2 --batchSize=256 --num_workers=4 --n_class=3 --leastEpoch=0
#python train.py  --train_path='./DataSet/0413/TrainSet2_004' --test_path='./DataSet/TestSet1_224_112' --epoches=120 --train_times=2 --batchSize=256 --num_workers=4 --n_class=3 --leastEpoch=0
#python train.py  --train_path='./DataSet/0413/TrainSet2_008' --test_path='./DataSet/TestSet1_224_112' --epoches=120 --train_times=2 --batchSize=256 --num_workers=4 --n_class=3 --leastEpoch=0
#python train.py  --train_path='./DataSet/0413/TrainSet2_016' --test_path='./DataSet/TestSet1_224_112' --epoches=120 --train_times=2 --batchSize=256 --num_workers=4 --n_class=3 --leastEpoch=0

# 0421训练算法1构成不同子图大小的实验, 运行3次 重新允许代码 重复1次就好 先把全部代码跑一遍
#python trainBatch.py --KFold=5 --root_path='./DataSet/0421/train/TrainSet1_600_300' --test_path='./DataSet/0421/val/TrainSet1_600_300' --epoches=120 --train_times=1 --batchSize=64 --num_workers=4 --n_class=3 --leastEpoch=0
#python trainBatch.py --KFold=5 --root_path='./DataSet/0421/train/TrainSet1_448_224' --test_path='./DataSet/0421/val/TrainSet1_448_224' --epoches=120 --train_times=1 --batchSize=128 --num_workers=4 --n_class=3 --leastEpoch=0
#python trainBatch.py --KFold=5 --root_path='./DataSet/0421/train/TrainSet1_336_168' --test_path='./DataSet/0421/val/TrainSet1_336_168' --epoches=120 --train_times=1 --batchSize=256 --num_workers=4 --n_class=3 --leastEpoch=0
#python trainBatch.py --KFold=5 --root_path='./DataSet/0421/train/TrainSet1_224_112' --test_path='./DataSet/0421/val/TrainSet1_224_112' --epoches=120 --train_times=1 --batchSize=256 --num_workers=4 --n_class=3 --leastEpoch=0
#python trainBatch.py --KFold=5 --root_path='./DataSet/0421/train/TrainSet1_168_84' --test_path='./DataSet/0421/val/TrainSet1_168_84' --epoches=120 --train_times=1 --batchSize=256 --num_workers=4 --n_class=3 --leastEpoch=0
#python trainBatch.py --KFold=5 --root_path='./DataSet/0421/train/TrainSet1_112_64' --test_path='./DataSet/0421/val/TrainSet1_112_64' --epoches=120 --train_times=1 --batchSize=256 --num_workers=4 --n_class=3 --leastEpoch=0
#python trainBatch.py --KFold=5 --root_path='./DataSet/0421/train/TrainSet1_64_32' --test_path='./DataSet/0421/val/TrainSet1_64_32' --epoches=120 --train_times=1 --batchSize=256 --num_workers=4 --n_class=3 --leastEpoch=0
#
#
#
#python trainBatch.py --KFold=5 --root_path='./DataSet/0422/train/TrainSet2_600' --test_path='./DataSet/0421/val/TrainSet1_600_300' --epoches=120 --train_times=1 --batchSize=64 --num_workers=4 --n_class=3 --leastEpoch=0
#python trainBatch.py --KFold=5 --root_path='./DataSet/0422/train/TrainSet2_448' --test_path='./DataSet/0421/val/TrainSet1_448_224' --epoches=120 --train_times=1 --batchSize=128 --num_workers=4 --n_class=3 --leastEpoch=0
#python trainBatch.py --KFold=5 --root_path='./DataSet/0422/train/TrainSet2_336' --test_path='./DataSet/0421/val/TrainSet1_336_168' --epoches=120 --train_times=1 --batchSize=256 --num_workers=4 --n_class=3 --leastEpoch=0
#python trainBatch.py --KFold=5 --root_path='./DataSet/0422/train/TrainSet2_224' --test_path='./DataSet/0421/val/TrainSet1_224_112' --epoches=120 --train_times=1 --batchSize=256 --num_workers=4 --n_class=3 --leastEpoch=0
#python trainBatch.py --KFold=5 --root_path='./DataSet/0422/train/TrainSet2_168' --test_path='./DataSet/0421/val/TrainSet1_168_84' --epoches=120 --train_times=1 --batchSize=256 --num_workers=4 --n_class=3 --leastEpoch=0
#python trainBatch.py --KFold=5 --root_path='./DataSet/0422/train/TrainSet2_112' --test_path='./DataSet/0421/val/TrainSet1_112_64' --epoches=120 --train_times=1 --batchSize=256 --num_workers=4 --n_class=3 --leastEpoch=0
#python trainBatch.py --KFold=5 --root_path='./DataSet/0422/train/TrainSet2_64' --test_path='./DataSet/0421/val/TrainSet1_64_32' --epoches=120 --train_times=1 --batchSize=256 --num_workers=4 --n_class=3 --leastEpoch=0



# 0428三组随机实验  原始数据一次 2 4 8 16 32区间舍弃各一次
#python trainBatch.py --KFold=5 --root_path='./DataSet/0428/train/TrainSet2_random1' --test_path='./DataSet/TestSet1_224_112' --epoches=120 --train_times=1 --batchSize=256 --num_workers=4 --n_class=3 --leastEpoch=0
#python trainBatch.py --KFold=5 --root_path='./DataSet/0428/train/TrainSet2_random2' --test_path='./DataSet/TestSet1_224_112' --epoches=120 --train_times=1 --batchSize=256 --num_workers=4 --n_class=3 --leastEpoch=0
#python trainBatch.py --KFold=5 --root_path='./DataSet/0428/train/TrainSet2_random3' --test_path='./DataSet/TestSet1_224_112' --epoches=120 --train_times=1 --batchSize=256 --num_workers=4 --n_class=3 --leastEpoch=0
#python trainBatch.py --KFold=5 --root_path='./DataSet/0428/train/TrainSet2_224' --test_path='./DataSet/TestSet1_224_112' --epoches=120 --train_times=1 --batchSize=256 --num_workers=4 --n_class=3 --leastEpoch=0
#python trainBatch.py --KFold=5 --root_path='./DataSet/0428/train/TrainSet2_002' --test_path='./DataSet/TestSet1_224_112' --epoches=120 --train_times=1 --batchSize=256 --num_workers=4 --n_class=3 --leastEpoch=0
#python trainBatch.py --KFold=5 --root_path='./DataSet/0428/train/TrainSet2_004' --test_path='./DataSet/TestSet1_224_112' --epoches=120 --train_times=1 --batchSize=256 --num_workers=4 --n_class=3 --leastEpoch=0
#python trainBatch.py --KFold=5 --root_path='./DataSet/0428/train/TrainSet2_008' --test_path='./DataSet/TestSet1_224_112' --epoches=120 --train_times=1 --batchSize=256 --num_workers=4 --n_class=3 --leastEpoch=0
#python trainBatch.py --KFold=5 --root_path='./DataSet/0428/train/TrainSet2_016' --test_path='./DataSet/TestSet1_224_112' --epoches=120 --train_times=1 --batchSize=256 --num_workers=4 --n_class=3 --leastEpoch=0
#python trainBatch.py --KFold=5 --root_path='./DataSet/0428/train/TrainSet2_032' --test_path='./DataSet/TestSet1_224_112' --epoches=120 --train_times=1 --batchSize=256 --num_workers=4 --n_class=3 --leastEpoch=0


# 0430 使用原图来进行实验, 分别使用数据增强及不同尺寸大小
#python trainBatch.py --KFold=5 --root_path='./DataSet/0429/train' --test_path='./DataSet/0429/test' --epoches=120 --train_times=1 --batchSize=32 --num_workers=4 --n_class=3 --leastEpoch=0 --raw_image --raw_size=224
#python trainBatch.py --KFold=5 --root_path='./DataSet/0429/train' --test_path='./DataSet/0429/test' --epoches=120 --train_times=1 --batchSize=32 --num_workers=4 --n_class=3 --leastEpoch=0 --raw_image --raw_size=224 --enhance
#python trainBatch.py --KFold=5 --root_path='./DataSet/0429/train' --test_path='./DataSet/0429/test' --epoches=120 --train_times=1 --batchSize=16 --num_workers=4 --n_class=3 --leastEpoch=0 --raw_image --raw_size=448
#python trainBatch.py --KFold=5 --root_path='./DataSet/0429/train' --test_path='./DataSet/0429/test' --epoches=120 --train_times=1 --batchSize=16 --num_workers=4 --n_class=3 --leastEpoch=0 --raw_image --raw_size=448 --enhance
#python trainBatch.py --KFold=5 --root_path='./DataSet/0429/train' --test_path='./DataSet/0429/test' --epoches=120 --train_times=1 --batchSize=8 --num_workers=4 --n_class=3 --leastEpoch=0 --raw_image --raw_size=600
#python trainBatch.py --KFold=5 --root_path='./DataSet/0429/train' --test_path='./DataSet/0429/test' --epoches=120 --train_times=1 --batchSize=8 --num_workers=4 --n_class=3 --leastEpoch=0 --raw_image --raw_size=600 --enhance


# 0502 right left 2 4 8 16 32
#python trainBatch.py --KFold=5 --root_path='./DataSet/0502/train/TrainSet2_l002' --test_path='./DataSet/TestSet1_224_112' --epoches=120 --train_times=1 --batchSize=256 --num_workers=4 --n_class=3 --leastEpoch=0
#python trainBatch.py --KFold=5 --root_path='./DataSet/0502/train/TrainSet2_l004' --test_path='./DataSet/TestSet1_224_112' --epoches=120 --train_times=1 --batchSize=256 --num_workers=4 --n_class=3 --leastEpoch=0
#python trainBatch.py --KFold=5 --root_path='./DataSet/0502/train/TrainSet2_l008' --test_path='./DataSet/TestSet1_224_112' --epoches=120 --train_times=1 --batchSize=256 --num_workers=4 --n_class=3 --leastEpoch=0
#python trainBatch.py --KFold=5 --root_path='./DataSet/0502/train/TrainSet2_l016' --test_path='./DataSet/TestSet1_224_112' --epoches=120 --train_times=1 --batchSize=256 --num_workers=4 --n_class=3 --leastEpoch=0
#python trainBatch.py --KFold=5 --root_path='./DataSet/0502/train/TrainSet2_l032' --test_path='./DataSet/TestSet1_224_112' --epoches=120 --train_times=1 --batchSize=256 --num_workers=4 --n_class=3 --leastEpoch=0
#
#python trainBatch.py --KFold=5 --root_path='./DataSet/0502/train/TrainSet2_r002' --test_path='./DataSet/TestSet1_224_112' --epoches=120 --train_times=1 --batchSize=256 --num_workers=4 --n_class=3 --leastEpoch=0
#python trainBatch.py --KFold=5 --root_path='./DataSet/0502/train/TrainSet2_r004' --test_path='./DataSet/TestSet1_224_112' --epoches=120 --train_times=1 --batchSize=256 --num_workers=4 --n_class=3 --leastEpoch=0
#python trainBatch.py --KFold=5 --root_path='./DataSet/0502/train/TrainSet2_r008' --test_path='./DataSet/TestSet1_224_112' --epoches=120 --train_times=1 --batchSize=256 --num_workers=4 --n_class=3 --leastEpoch=0
#python trainBatch.py --KFold=5 --root_path='./DataSet/0502/train/TrainSet2_r016' --test_path='./DataSet/TestSet1_224_112' --epoches=120 --train_times=1 --batchSize=256 --num_workers=4 --n_class=3 --leastEpoch=0
#python trainBatch.py --KFold=5 --root_path='./DataSet/0502/train/TrainSet2_r032' --test_path='./DataSet/TestSet1_224_112' --epoches=120 --train_times=1 --batchSize=256 --num_workers=4 --n_class=3 --leastEpoch=0

# 0505 every model to train on five dataset
#python trainBatch.py --KFold=5 --root_path='./DataSet/0505/train/TrainSet1' --test_path='./DataSet/TestSet1_224_112' --epoches=120 --train_times=1 --batchSize=256 --num_workers=4 --n_class=3 --leastEpoch=0 --model_name='shufflenetv2'
#python trainBatch.py --KFold=5 --root_path='./DataSet/0505/train/TrainSet1' --test_path='./DataSet/TestSet1_224_112' --epoches=120 --train_times=1 --batchSize=256 --num_workers=4 --n_class=3 --leastEpoch=0 --model_name='mobilenetv3'
#python trainBatch.py --KFold=5 --root_path='./DataSet/0505/train/TrainSet1' --test_path='./DataSet/TestSet1_224_112' --epoches=120 --train_times=1 --batchSize=256 --num_workers=4 --n_class=3 --leastEpoch=0 --model_name='efficientnetb0'
#python trainBatch.py --KFold=5 --root_path='./DataSet/0505/train/TrainSet1' --test_path='./DataSet/TestSet1_224_112' --epoches=120 --train_times=1 --batchSize=256 --num_workers=4 --n_class=3 --leastEpoch=0 --model_name='vit'
#
#python trainBatch.py --KFold=5 --root_path='./DataSet/0505/train/TrainSet2' --test_path='./DataSet/TestSet1_224_112' --epoches=120 --train_times=1 --batchSize=256 --num_workers=4 --n_class=3 --leastEpoch=0 --model_name='shufflenetv2'
#python trainBatch.py --KFold=5 --root_path='./DataSet/0505/train/TrainSet2' --test_path='./DataSet/TestSet1_224_112' --epoches=120 --train_times=1 --batchSize=256 --num_workers=4 --n_class=3 --leastEpoch=0 --model_name='mobilenetv3'
#python trainBatch.py --KFold=5 --root_path='./DataSet/0505/train/TrainSet2' --test_path='./DataSet/TestSet1_224_112' --epoches=120 --train_times=1 --batchSize=256 --num_workers=4 --n_class=3 --leastEpoch=0 --model_name='efficientnetb0'
#python trainBatch.py --KFold=5 --root_path='./DataSet/0505/train/TrainSet2' --test_path='./DataSet/TestSet1_224_112' --epoches=120 --train_times=1 --batchSize=256 --num_workers=4 --n_class=3 --leastEpoch=0 --model_name='vit'
#
#python trainBatch.py --KFold=5 --root_path='./DataSet/0505/train/TrainRand1' --test_path='./DataSet/TestSet1_224_112' --epoches=120 --train_times=1 --batchSize=256 --num_workers=4 --n_class=3 --leastEpoch=0 --model_name='shufflenetv2'
#python trainBatch.py --KFold=5 --root_path='./DataSet/0505/train/TrainRand1' --test_path='./DataSet/TestSet1_224_112' --epoches=120 --train_times=1 --batchSize=256 --num_workers=4 --n_class=3 --leastEpoch=0 --model_name='mobilenetv3'
#python trainBatch.py --KFold=5 --root_path='./DataSet/0505/train/TrainRand1' --test_path='./DataSet/TestSet1_224_112' --epoches=120 --train_times=1 --batchSize=256 --num_workers=4 --n_class=3 --leastEpoch=0 --model_name='efficientnetb0'
#python trainBatch.py --KFold=5 --root_path='./DataSet/0505/train/TrainRand1' --test_path='./DataSet/TestSet1_224_112' --epoches=120 --train_times=1 --batchSize=256 --num_workers=4 --n_class=3 --leastEpoch=0 --model_name='vit'
#
#python trainBatch.py --KFold=5 --root_path='./DataSet/0505/train/TrainRand2' --test_path='./DataSet/TestSet1_224_112' --epoches=120 --train_times=1 --batchSize=256 --num_workers=4 --n_class=3 --leastEpoch=0 --model_name='shufflenetv2'
#python trainBatch.py --KFold=5 --root_path='./DataSet/0505/train/TrainRand2' --test_path='./DataSet/TestSet1_224_112' --epoches=120 --train_times=1 --batchSize=256 --num_workers=4 --n_class=3 --leastEpoch=0 --model_name='mobilenetv3'
#python trainBatch.py --KFold=5 --root_path='./DataSet/0505/train/TrainRand2' --test_path='./DataSet/TestSet1_224_112' --epoches=120 --train_times=1 --batchSize=256 --num_workers=4 --n_class=3 --leastEpoch=0 --model_name='efficientnetb0'
#python trainBatch.py --KFold=5 --root_path='./DataSet/0505/train/TrainRand2' --test_path='./DataSet/TestSet1_224_112' --epoches=120 --train_times=1 --batchSize=256 --num_workers=4 --n_class=3 --leastEpoch=0 --model_name='vit'
#
#python trainBatch.py --KFold=5 --root_path='./DataSet/0505/train/TrainRand3' --test_path='./DataSet/TestSet1_224_112' --epoches=120 --train_times=1 --batchSize=256 --num_workers=4 --n_class=3 --leastEpoch=0 --model_name='shufflenetv2'
#python trainBatch.py --KFold=5 --root_path='./DataSet/0505/train/TrainRand3' --test_path='./DataSet/TestSet1_224_112' --epoches=120 --train_times=1 --batchSize=256 --num_workers=4 --n_class=3 --leastEpoch=0 --model_name='mobilenetv3'
#python trainBatch.py --KFold=5 --root_path='./DataSet/0505/train/TrainRand3' --test_path='./DataSet/TestSet1_224_112' --epoches=120 --train_times=1 --batchSize=256 --num_workers=4 --n_class=3 --leastEpoch=0 --model_name='efficientnetb0'
#python trainBatch.py --KFold=5 --root_path='./DataSet/0505/train/TrainRand3' --test_path='./DataSet/TestSet1_224_112' --epoches=120 --train_times=1 --batchSize=256 --num_workers=4 --n_class=3 --leastEpoch=0 --model_name='vit'
#
#python trainBatch.py --KFold=5 --root_path='./DataSet/0505/train/TrainSet2_004' --test_path='./DataSet/TestSet1_224_112' --epoches=120 --train_times=1 --batchSize=256 --num_workers=4 --n_class=3 --leastEpoch=0 --model_name='shufflenetv2'
#python trainBatch.py --KFold=5 --root_path='./DataSet/0505/train/TrainSet2_004' --test_path='./DataSet/TestSet1_224_112' --epoches=120 --train_times=1 --batchSize=256 --num_workers=4 --n_class=3 --leastEpoch=0 --model_name='mobilenetv3'
#python trainBatch.py --KFold=5 --root_path='./DataSet/0505/train/TrainSet2_004' --test_path='./DataSet/TestSet1_224_112' --epoches=120 --train_times=1 --batchSize=128 --num_workers=4 --n_class=3 --leastEpoch=0 --model_name='efficientnetb0'
#python trainBatch.py --KFold=5 --root_path='./DataSet/0505/train/TrainSet2_004' --test_path='./DataSet/TestSet1_224_112' --epoches=120 --train_times=1 --batchSize=256 --num_workers=4 --n_class=3 --leastEpoch=0 --model_name='vit'


# 0507 break
#python trainBatch.py --KFold=5 --root_path='./DataSet/0505/train/TrainSet1' --test_path='./DataSet/TestSet1_224_112' --epoches=120 --train_times=1 --batchSize=128 --num_workers=4 --n_class=3 --leastEpoch=0 --model_name='efficientnetb0'
#python trainBatch.py --KFold=5 --root_path='./DataSet/0505/train/TrainSet1' --test_path='./DataSet/TestSet1_224_112' --epoches=120 --train_times=1 --batchSize=128 --num_workers=4 --n_class=3 --leastEpoch=0 --model_name='vit'
#
#
#python trainBatch.py --KFold=5 --root_path='./DataSet/0505/train/TrainSet2' --test_path='./DataSet/TestSet1_224_112' --epoches=120 --train_times=1 --batchSize=128 --num_workers=4 --n_class=3 --leastEpoch=0 --model_name='efficientnetb0'
#python trainBatch.py --KFold=5 --root_path='./DataSet/0505/train/TrainSet2' --test_path='./DataSet/TestSet1_224_112' --epoches=120 --train_times=1 --batchSize=128 --num_workers=4 --n_class=3 --leastEpoch=0 --model_name='vit'
#
#
#python trainBatch.py --KFold=5 --root_path='./DataSet/0505/train/TrainRand1' --test_path='./DataSet/TestSet1_224_112' --epoches=120 --train_times=1 --batchSize=128 --num_workers=4 --n_class=3 --leastEpoch=0 --model_name='efficientnetb0'
#python trainBatch.py --KFold=5 --root_path='./DataSet/0505/train/TrainRand1' --test_path='./DataSet/TestSet1_224_112' --epoches=120 --train_times=1 --batchSize=128 --num_workers=4 --n_class=3 --leastEpoch=0 --model_name='vit'
#
#
#python trainBatch.py --KFold=5 --root_path='./DataSet/0505/train/TrainRand2' --test_path='./DataSet/TestSet1_224_112' --epoches=120 --train_times=1 --batchSize=128 --num_workers=4 --n_class=3 --leastEpoch=0 --model_name='efficientnetb0'
#python trainBatch.py --KFold=5 --root_path='./DataSet/0505/train/TrainRand2' --test_path='./DataSet/TestSet1_224_112' --epoches=120 --train_times=1 --batchSize=128 --num_workers=4 --n_class=3 --leastEpoch=0 --model_name='vit'
#
#
#python trainBatch.py --KFold=5 --root_path='./DataSet/0505/train/TrainRand3' --test_path='./DataSet/TestSet1_224_112' --epoches=120 --train_times=1 --batchSize=128 --num_workers=4 --n_class=3 --leastEpoch=0 --model_name='efficientnetb0'
#python trainBatch.py --KFold=5 --root_path='./DataSet/0505/train/TrainRand3' --test_path='./DataSet/TestSet1_224_112' --epoches=120 --train_times=1 --batchSize=128 --num_workers=4 --n_class=3 --leastEpoch=0 --model_name='vit'
#
#
#python trainBatch.py --KFold=5 --root_path='./DataSet/0505/train/TrainSet2_004' --test_path='./DataSet/TestSet1_224_112' --epoches=120 --train_times=1 --batchSize=128 --num_workers=4 --n_class=3 --leastEpoch=0 --model_name='efficientnetb0'
#python trainBatch.py --KFold=5 --root_path='./DataSet/0505/train/TrainSet2_004' --test_path='./DataSet/TestSet1_224_112' --epoches=120 --train_times=1 --batchSize=128 --num_workers=4 --n_class=3 --leastEpoch=0 --model_name='vit'

#  0508 vit
#python trainBatch.py --KFold=5 --root_path='./DataSet/0505/train/TrainSet1' --test_path='./DataSet/TestSet1_224_112' --epoches=120 --train_times=1 --batchSize=50 --num_workers=4 --n_class=3 --leastEpoch=0 --model_name='vit' --learning_rate=0.001
#
#python trainBatch.py --KFold=5 --root_path='./DataSet/0505/train/TrainSet2' --test_path='./DataSet/TestSet1_224_112' --epoches=120 --train_times=1 --batchSize=50 --num_workers=4 --n_class=3 --leastEpoch=0 --model_name='vit' --learning_rate=0.001
#
#python trainBatch.py --KFold=5 --root_path='./DataSet/0505/train/TrainRand1' --test_path='./DataSet/TestSet1_224_112' --epoches=120 --train_times=1 --batchSize=50 --num_workers=4 --n_class=3 --leastEpoch=0 --model_name='vit' --learning_rate=0.001
#
#python trainBatch.py --KFold=5 --root_path='./DataSet/0505/train/TrainRand2' --test_path='./DataSet/TestSet1_224_112' --epoches=120 --train_times=1 --batchSize=50 --num_workers=4 --n_class=3 --leastEpoch=0 --model_name='vit' --learning_rate=0.001
#
#python trainBatch.py --KFold=5 --root_path='./DataSet/0505/train/TrainRand3' --test_path='./DataSet/TestSet1_224_112' --epoches=120 --train_times=1 --batchSize=50 --num_workers=4 --n_class=3 --leastEpoch=0 --model_name='vit' --learning_rate=0.001
#
#python trainBatch.py --KFold=5 --root_path='./DataSet/0505/train/TrainSet2_004' --test_path='./DataSet/TestSet1_224_112' --epoches=120 --train_times=1 --batchSize=50 --num_workers=4 --n_class=3 --leastEpoch=0 --model_name='vit' --learning_rate=0.001

# 0519 原图 224_224训练  224_224测试
#python trainBatch.py --KFold=5 --root_path='./DataSet/0518/original/train' --test_path='./DataSet/0518/original/test' --epoches=120 --train_times=1 --batchSize=4 --num_workers=4 --n_class=3 --leastEpoch=0 --raw_image --raw_size=1200
#python trainBatch.py --KFold=5 --root_path='./DataSet/0518/original/train' --test_path='./DataSet/0518/original/test' --epoches=120 --train_times=1 --batchSize=8 --num_workers=4 --n_class=3 --leastEpoch=0 --raw_image --raw_size=600
#
#python trainBatch.py --KFold=5 --root_path='./DataSet/train/TrainSet1_224_112' --test_path='./DataSet/TestSet1_224_224' --epoches=120 --train_times=3 --batchSize=256 --num_workers=4 --n_class=3 --leastEpoch=0
#python trainBatch.py --KFold=5 --root_path='./DataSet/train/TrainSet1_224_224' --test_path='./DataSet/TestSet1_224_224' --epoches=120 --train_times=3 --batchSize=256 --num_workers=4 --n_class=3 --leastEpoch=0
#
#python trainBatch.py --KFold=5 --root_path='./DataSet/train/TrainSet2_224_224' --test_path='./DataSet/TestSet1_224_224' --epoches=120 --train_times=3 --batchSize=256 --num_workers=4 --n_class=3 --leastEpoch=0
#python trainBatch.py --KFold=5 --root_path='./DataSet/train/TrainSet2_224_112' --test_path='./DataSet/TestSet1_224_224' --epoches=120 --train_times=3 --batchSize=256 --num_workers=4 --n_class=3 --leastEpoch=0
#
#python trainBatch.py --KFold=5 --root_path='./DataSet/train/TrainSet2_004' --test_path='./DataSet/TestSet1_224_224' --epoches=120 --train_times=3 --batchSize=256 --num_workers=4 --n_class=3 --leastEpoch=0
#python trainBatch.py --KFold=5 --root_path='./DataSet/train/TrainSet2_004' --test_path='./DataSet/TestSet1_224_112' --epoches=120 --train_times=3 --batchSize=256 --num_workers=4 --n_class=3 --leastEpoch=0
#
# 0521 because of 0519's break
#python trainBatch.py --KFold=5 --root_path='./DataSet/train/TrainSet1_224_224' --test_path='./DataSet/TestSet1_224_224' --epoches=120 --train_times=3 --batchSize=256 --num_workers=4 --n_class=3 --leastEpoch=0
#python trainBatch.py --KFold=5 --root_path='./DataSet/train/TrainSet2_004' --test_path='./DataSet/TestSet1_224_224' --epoches=120 --train_times=3 --batchSize=256 --num_workers=4 --n_class=3 --leastEpoch=0

# 0522 repeat best result model
#python train.py  --train_path='./DataSet/single/TrainSet1_224_112' --test_path='./DataSet/TestSet1_224_112' --epoches=120 --train_times=3 --batchSize=256 --num_workers=4 --n_class=3 --leastEpoch=0
#python train.py  --train_path='./DataSet/single/TrainSet1_224_112' --test_path='./DataSet/TestSet1_224_224' --epoches=120 --train_times=3 --batchSize=256 --num_workers=4 --n_class=3 --leastEpoch=0
#
#python train.py  --train_path='./DataSet/single/TrainSet2_224_112' --test_path='./DataSet/TestSet1_224_112' --epoches=120 --train_times=3 --batchSize=256 --num_workers=4 --n_class=3 --leastEpoch=0
#python train.py  --train_path='./DataSet/single/TrainSet2_224_112' --test_path='./DataSet/TestSet1_224_224' --epoches=120 --train_times=3 --batchSize=256 --num_workers=4 --n_class=3 --leastEpoch=0
#
#python train.py  --train_path='./DataSet/single/TrainSet2_004' --test_path='./DataSet/TestSet1_224_112' --epoches=120 --train_times=3 --batchSize=256 --num_workers=4 --n_class=3 --leastEpoch=0
#python train.py  --train_path='./DataSet/single/TrainSet2_004' --test_path='./DataSet/TestSet1_224_224' --epoches=120 --train_times=3 --batchSize=256 --num_workers=4 --n_class=3 --leastEpoch=0

# 0523 left right 004
python train.py  --train_path='./DataSet/single/TrainSet2_r004' --test_path='./DataSet/TestSet1_224_112' --epoches=120 --train_times=3 --batchSize=256 --num_workers=4 --n_class=3 --leastEpoch=0
python train.py  --train_path='./DataSet/single/TrainSet2_r004' --test_path='./DataSet/TestSet1_224_224' --epoches=120 --train_times=3 --batchSize=256 --num_workers=4 --n_class=3 --leastEpoch=0

python train.py  --train_path='./DataSet/single/TrainSet2_l004' --test_path='./DataSet/TestSet1_224_112' --epoches=120 --train_times=3 --batchSize=256 --num_workers=4 --n_class=3 --leastEpoch=0
python train.py  --train_path='./DataSet/single/TrainSet2_l004' --test_path='./DataSet/TestSet1_224_224' --epoches=120 --train_times=3 --batchSize=256 --num_workers=4 --n_class=3 --leastEpoch=0





