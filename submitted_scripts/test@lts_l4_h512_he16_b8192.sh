#!/bin/bash
nvidia-smi
exit_status=0
cd /var/storage/shared/sdrgvc/xuta/transformer/lts
source philly/configure_philly.sh sdrgvc
HPARAMS="num_decoder_layers=4,num_encoder_layers=4,hidden_size=512,filter_size=2048,num_heads=16,batch_size=8192," SETTING=lts_l4_h512_he16_b8192 bash runs/test_lts.sh --exp-name=test@lts_l4_h512_he16_b8192
wait
exit $exit_status
