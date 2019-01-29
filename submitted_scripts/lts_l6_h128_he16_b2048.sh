#!/bin/bash
nvidia-smi
exit_status=0
cd /var/storage/shared/sdrgvc/xuta/transformer/lts
source philly/configure_philly.sh sdrgvc
HPARAMS="num_decoder_layers=6,num_encoder_layers=6,hidden_size=128,filter_size=512,num_heads=16,batch_size=2048," SETTING=lts_l6_h128_he16_b2048 bash runs/train_lts.sh --exp-name=lts_l6_h128_he16_b2048
wait
exit $exit_status
