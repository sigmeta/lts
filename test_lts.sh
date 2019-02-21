
HOME=/var/storage/shared/sdrgvc/xuta/t-hasu/lts

export PYTHONPATH=${HOME}/tensor2tensor-baseline:${PYTHONPATH}
binFile=${HOME}/tensor2tensor-baseline/tensor2tensor/bin


PROBLEM=lts
MODEL=transformer
HPARAMS_SET=transformer_small
HPARAMS="num_heads=16,num_encoder_layers=3,num_decoder_layers=1,label_smoothing=0.0"
#HPARAMS=""


while true;
do

for ii in 6 8 9 11 13 14 15 16 17 18 19 20; do
  setting=setting${ii}_small
  ROOT_MODEL=/hdfs/sdrgvc/xuta/t-hasu/lts/${setting}
  echo $ii

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
    --decode_hparams="beam_size=10,alpha=1,batch_size=512" \
    --decode_from_file=lts_data/lts.s2s.test.l \
    --worker_gpu=1 > inference 2>&1
  echo -e "${ii}\n\n"${ii} "\n">> test/${setting}/result
  python test/calc_WER_PER.py test/lts.s2s.test.s lts_data/lts.s2s.test.l.* >> test/${setting}/result

done
sleep 100
done
