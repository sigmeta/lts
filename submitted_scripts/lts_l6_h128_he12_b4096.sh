#!/bin/bash
nvidia-smi
exit_status=0
cd /var/storage/shared/sdrgvc/xuta/transformer/lts
source philly/configure_philly.sh sdrgvc
HPARAMS="num_decoder_layers=6,num_encoder_layers=6,hidden_size=128,filter_size=512,num_heads=12,batch_size=4096," SETTING=lts_l6_h128_he12_b4096 bash runs/train_lts.sh --distributed=True  --exp-name=lts_l6_h128_he12_b4096
wait
exit $exit_status
