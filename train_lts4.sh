
HOME=/var/storage/shared/sdrgvc/xuta/t-hasu/lts

export PYTHONPATH=${HOME}/tensor2tensor-baseline:${PYTHONPATH}
binFile=${HOME}/tensor2tensor-baseline/tensor2tensor/bin


PROBLEM=lts
MODEL=transformer
HPARAMS_SET=transformer_small

setting=setting7_small
DATA_DIR=${HOME}/lts_data
TRAIN_DIR=/hdfs/sdrgvc/xuta/t-hasu/lts/${setting}
mkdir -p  $TRAIN_DIR

nohup ${binFile}/t2t-trainer \
--t2t_usr_dir=${HOME}/lts_data \
--data_dir=$DATA_DIR \
--problems=$PROBLEM \
--model=$MODEL \
--hparams_set=$HPARAMS_SET \
--output_dir=$TRAIN_DIR \
--keep_checkpoint_max=100000 \
--worker_gpu=2 \
--train_steps=20000000 \
--save_checkpoints_secs=600 \
--schedule=train \
--worker_gpu_memory_fraction=0.95 \
--hparams="batch_size=4096,num_heads=16,num_encoder_layers=3,num_decoder_layers=1,attention_dropout=0.2,relu_dropout=0.2,dropout=0.1,label_smoothing=0.0,learning_rate=0.6" > nohup.train_${setting}.log 2>&1 &
