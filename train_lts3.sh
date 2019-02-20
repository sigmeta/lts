
HOME=/var/storage/shared/sdrgvc/xuta/t-hasu/lts

export PYTHONPATH=${HOME}/tensor2tensor-baseline:${PYTHONPATH}
binFile=${HOME}/tensor2tensor-baseline/tensor2tensor/bin


PROBLEM=lts
MODEL=distillation
HPARAMS_SET=transformer_small

setting=kd
DATA_DIR=${HOME}/lts_data
TRAIN_DIR=/hdfs/sdrgvc/xuta/t-hasu/lts/${setting}
mkdir -p  $TRAIN_DIR

CUDA_VISIBLE_DEVICES=0 ${binFile}/t2t-distill \
--t2t_usr_dir=${HOME}/lts_data \
--data_dir=$DATA_DIR \
--problems=$PROBLEM \
--model=$MODEL \
--hparams_set=$HPARAMS_SET \
--output_dir=$TRAIN_DIR \
--keep_checkpoint_max=100000 \
--worker_gpu=1 \
--train_steps=20000000 \
--save_checkpoints_secs=1800 \
--schedule=train \
--worker_gpu_memory_fraction=0.95 \
--hparams="batch_size=4096"
