#!/bin/bash


if [ ! -d "/var/storage/shared/sdrgvc/xuta/anaconda3" ]
then
    bash /var/storage/shared/sdrgvc/xuta/Anaconda3-5.0.1-Linux-x86_64.sh -b -p /var/storage/shared/sdrgvc/xuta/anaconda3
    export PATH="/var/storage/shared/sdrgvc/xuta/anaconda3/bin:$PATH"
    pip install -r tensor2tensor-baseline/requirements.txt
fi
export PATH="/var/storage/shared/sdrgvc/xuta/anaconda3/bin:$PATH"
pip install -r tensor2tensor-baseline/requirements.txt
export vc=$1

