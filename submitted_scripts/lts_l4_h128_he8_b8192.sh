#!/bin/bash
nvidia-smi
exit_status=0
cd /var/storage/shared/sdrgvc/xuta/transformer/lts
source philly/configure_philly.sh sdrgvc
HPARAMS="num_decoder_layers=4,num_encoder_layers=4,hidden_size=128,filter_size=512,num_heads=8,batch_size=8192," SETTING=lts_l4_h128_he8_b8192 bash runs/train_lts.sh --exp-name=lts_l4_h128_he8_b8192
wait
exit $exit_status
