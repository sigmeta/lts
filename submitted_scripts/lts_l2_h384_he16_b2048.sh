#!/bin/bash
nvidia-smi
exit_status=0
cd /var/storage/shared/sdrgvc/xuta/transformer/lts
source philly/configure_philly.sh sdrgvc
HPARAMS="num_decoder_layers=2,num_encoder_layers=2,hidden_size=384,filter_size=1536,num_heads=16,batch_size=2048," SETTING=lts_l2_h384_he16_b2048 bash runs/train_lts.sh --distributed=True  --exp-name=lts_l2_h384_he16_b2048
wait
exit $exit_status
