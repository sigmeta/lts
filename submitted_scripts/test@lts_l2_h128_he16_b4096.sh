#!/bin/bash
nvidia-smi
exit_status=0
cd /var/storage/shared/sdrgvc/xuta/transformer/lts
source philly/configure_philly.sh sdrgvc
HPARAMS="num_decoder_layers=2,num_encoder_layers=2,hidden_size=128,filter_size=512,num_heads=16,batch_size=4096," SETTING=lts_l2_h128_he16_b4096 bash runs/test_lts.sh --exp-name=test@lts_l2_h128_he16_b4096
wait
exit $exit_status
