export PYTHONPATH=/home/xutan/tensor2tensor-1.2.9:${PYTHONPATH}
binFile=/home/xutan/tensor2tensor-1.2.9/tensor2tensor/bin

#${binFile}/t2t-trainer --registry_help

PROBLEM=translate_zhen_wmt17
MODEL=transformer
HPARAMS_SET=zhen_wmt17_transformer_big_setting9
HPARAMS="batch_size=7168,shared_embedding_and_softmax_weights=0"

DATA_DIR=/home/xutan/babel/zhen_rush/test/data

ROOT_MODEL=/home/xutan/babel/zhen_rush/test/src_model/7-7

ids=$(ls ${ROOT_MODEL} | grep "model\.ckpt-[0-9]*.index" | grep -o "[0-9]*")

echo ${ids}

for ii in ${ids}; do
  tmpdir=${ROOT_MODEL}_${ii}
  
  rm -rf $tmpdir
  mkdir -p $tmpdir
  mv ${ROOT_MODEL}/model.ckpt-${ii}* $tmpdir/
  cd $tmpdir
  touch checkpoint
  echo model_checkpoint_path: \"model.ckpt-${ii}\" >> checkpoint
  echo all_model_checkpoint_paths: \"model.ckpt-${ii}\" >> checkpoint
  cd /home/xutan/babel/zhen_rush/eval
 
  python eval.py \
    --t2t_usr_dir=./zhen_wmt17 \
    --data_dir=$DATA_DIR/t2t_data_clean \
    --problems=$PROBLEM \
    --model=$MODEL \
    --hparams_set=$HPARAMS_SET \
    --output_dir=$tmpdir \
    --hparams=$HPARAMS \
    --worker_gpu=1 \
    > result/valid_loss_${ii}
   
done

: << END

END



