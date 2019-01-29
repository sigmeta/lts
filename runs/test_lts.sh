
HOME=/var/storage/shared/sdrgvc/xuta/transformer/lts

export PYTHONPATH=${HOME}/tensor2tensor-baseline:${PYTHONPATH}
binFile=${HOME}/tensor2tensor-baseline/tensor2tensor/bin


PROBLEM=lts
MODEL=transformer
HPARAMS_SET=transformer_small

HPARAMS=${HPARAMS}
setting=${SETTING}

ROOT_MODEL=/hdfs/sdrgvc/xuta/transformer_data/lts/${setting}


while true;
do
ids=$(ls ${ROOT_MODEL} | grep "model\.ckpt-[0-9]*.index" | grep -o "[0-9]*")
echo ${ids}

for ii in ${ids}; do

  echo $ii
  if test -s ${HOME}/test/${setting}/${ii}/lts.s2s.test.l.transformer*; then  
    echo "pass $ii"
  else
    echo "test $ii"

  echo model_checkpoint_path: \"model.ckpt-${ii}\" > ${ROOT_MODEL}/checkpoint

  mkdir -p ${HOME}/test/${setting}/${ii}
  cp ${HOME}/lts_data/lts.s2s.test.l ${HOME}/test/${setting}/${ii}/
  echo ${ii}
 
  ${binFile}/t2t-decoder \
    --t2t_usr_dir=${HOME}/lts_data \
    --data_dir=${HOME}/lts_data \
    --problems=$PROBLEM \
    --model=$MODEL \
    --hparams_set=${HPARAMS_SET} \
    --output_dir=${ROOT_MODEL} \
    --hparams=$HPARAMS \
    --decode_hparams="beam_size=1,alpha=1.1,batch_size=128" \
    --decode_from_file=${HOME}/test/${setting}/${ii}/lts.s2s.test.l \
    --worker_gpu=1
  echo -e "\n\n"${ii} "\n">> ${HOME}/test/${setting}/result
  python ${HOME}/test/calc_WER_PER.py ${HOME}/test/lts.s2s.test.s ${HOME}/test/${setting}/${ii}/lts.s2s.test.l.* >> ${HOME}/test/${setting}/result

  fi
done
sleep 600
done
