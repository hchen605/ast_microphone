#!/bin/bash
# SBATCH -p gpu
# SBATCH -x sls-titan-[0-2]
# SBATCH --gres=gpu:4
# SBATCH -c 4
# SBATCH -n 1
# SBATCH --mem=48000
# SBATCH --job-name="ast-esc50"
# SBATCH --output=./log_%j.txt

set -x
# comment this line if not running on sls cluster
#. /data/sls/scratch/share-201907/slstoolchainrc
#source ../../venvast/bin/activate
export TORCH_HOME=../../pretrained_models

# +

model=ast
dataset=microphone
imagenetpretrain=True
audiosetpretrain=False
bal=none
if [ $audiosetpretrain == True ]
then
  lr=1e-5
else
  lr=1e-4
fi
freqm=24
timem=96
mixup=0
epoch=50
batch_size=20
fstride=10
tstride=10
base_exp_dir=./exp/test-${dataset}-imp$imagenetpretrain-asp$audiosetpretrain-b$batch_size-lr${lr}_5_ep100_7m_cm_3



# +
#python ./prep_esc50.py

# +

if [ -d $base_exp_dir ]; then
  echo 'exp exist'
  #exit
fi
mkdir -p $exp_dir


# +
#for((fold=1;fold<=5;fold++));
#do
echo 'now process fold'

exp_dir=${base_exp_dir}/

#tr_data=./data/datafiles/esc_train_data_${fold}.json
#te_data=./data/datafiles/esc_eval_data_${fold}.json


#train_csv=/home/hchen605/microphone_classification_add_feat/12class/data/train_full.csv
#dev_csv=/home/hchen605/microphone_classification_add_feat/12class/data/dev_full.csv
#test_csv=/home/hchen605/microphone_classification_add_feat/12class/data/test_full.csv
train_csv=/home/hchen605/microphone_class_dist/microphone_classification/12class/data/train_full_dist.csv
dev_csv=/home/hchen605/microphone_class_dist/microphone_classification/12class/data/dev_full_dis.csv
test_csv=/home/hchen605/microphone_class_dist/microphone_classification/12class/data/test_full_7m.csv
    
    
CUDA_CACHE_DISABLE=1 python -W ignore ../../src/run_mic.py --model ${model} --dataset ${dataset} \
--data-train ${train_csv} --data-val ${test_csv} --exp-dir $exp_dir \
--label-csv ./data/esc_class_labels_indices.csv --n_class 3 \
--lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model False \
--freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} \
--tstride $tstride --fstride $fstride --imagenet_pretrain $imagenetpretrain --audioset_pretrain $audiosetpretrain
#done

# +
#python ./get_esc_result.py --exp_path ${base_exp_dir}
# -


