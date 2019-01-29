#!/bin/bash
nvidia-smi
exit_status=0
cd /var/storage/shared/sdrgvc/xuta/transformer/lts
source philly/configure_philly.sh sdrgvc
HPARAMS="num_decoder_layers=6,num_encoder_layers=6,hidden_size=512,filter_size=2048,num_heads=4,batch_size=4096," SETTING=lts_l6_h512_he4_b4096 bash runs/train_lts.sh --exp-name=lts_l6_h512_he4_b4096
wait
exit $exit_status
