HOME=/var/storage/shared/sdrgvc/xuta/transformer/lts

export PYTHONPATH=${HOME}/tensor2tensor-baseline:${PYTHONPATH}
binFile=${HOME}/tensor2tensor-baseline/tensor2tensor/bin

PROBLEM=lts

DATA_DIR=${HOME}/lts_data/

# Generate data
${binFile}/t2t-datagen \
  --t2t_usr_dir=$HOME/lts_data \
  --data_dir=$DATA_DIR \
  --problem=$PROBLEM \
  --tmp_dir=$DATA_DIR 
