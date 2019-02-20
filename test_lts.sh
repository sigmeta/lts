
HOME=/var/storage/shared/sdrgvc/xuta/t-hasu/lts

export PYTHONPATH=${HOME}/tensor2tensor-baseline:${PYTHONPATH}
binFile=${HOME}/tensor2tensor-baseline/tensor2tensor/bin


PROBLEM=lts
MODEL=transformer
HPARAMS_SET=transformer_small
HPARAMS="num_heads=16,num_encoder_layers=3,num_decoder_layers=1,attention_dropout=0.2,relu_dropout=0.2,dropout=0.1,label_smoothing=0.0"
#HPARAMS=""

setting=setting6_small
ROOT_MODEL=/hdfs/sdrgvc/xuta/t-hasu/lts/${setting}


while true;
do
ids=$(ls ${ROOT_MODEL} | grep "model\.ckpt-[0-9]*.index" | grep -o "[0-9]*")
echo ${ids}

for ii in ${ids}; do

  echo $ii
  if test -s test/${setting}/${ii}/lts.s2s.test.l.transformer*; then  
    echo "pass $ii"
    #echo -e "\n\r\n\r"${ii} "\n">> test/${setting1_small}/result
    python test/calc_WER_PER.py test/lts.s2s.test.s test/${setting}/${ii}/lts.s2s.test.l.* >> test/${setting}/result
  else
    echo "test $ii"

  echo model_checkpoint_path: \"model.ckpt-${ii}\" > ${ROOT_MODEL}/checkpoint

  mkdir -p test/${setting}/${ii}
  cp lts_data/lts.s2s.test.l test/${setting}/${ii}/
  echo ${ii}
 
  ${binFile}/t2t-decoder \
    --t2t_usr_dir=./lts_data \
    --data_dir=./lts_data \
    --problems=$PROBLEM \
    --model=$MODEL \
    --hparams_set=${HPARAMS_SET} \
    --output_dir=${ROOT_MODEL} \
    --hparams=$HPARAMS \
    --decode_hparams="beam_size=10,alpha=1.1,batch_size=512" \
    --decode_from_file=test/${setting}/${ii}/lts.s2s.test.l \
    --worker_gpu=1 > inference 2>&1
  echo -e "${ii}\n\n"${ii} "\n">> test/${setting}/result
  python test/calc_WER_PER.py test/lts.s2s.test.s test/${setting}/${ii}/lts.s2s.test.l.* >> test/${setting}/result

  fi
done
sleep 600
done
