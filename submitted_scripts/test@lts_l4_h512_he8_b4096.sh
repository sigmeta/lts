#!/bin/bash
nvidia-smi
exit_status=0
cd /var/storage/shared/sdrgvc/xuta/transformer/lts
source philly/configure_philly.sh sdrgvc
HPARAMS="num_decoder_layers=4,num_encoder_layers=4,hidden_size=512,filter_size=2048,num_heads=8,batch_size=4096," SETTING=lts_l4_h512_he8_b4096 bash runs/test_lts.sh --exp-name=test@lts_l4_h512_he8_b4096
wait
exit $exit_status
