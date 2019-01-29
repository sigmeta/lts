#!/bin/bash
nvidia-smi
exit_status=0
cd /var/storage/shared/sdrgvc/xuta/transformer/multi-trans
source philly/configure_philly.sh sdrgvc
sh train_encv.sh --exp-name=@test_14
wait
exit $exit_status
