#!/bin/bash
#
# training script for the substitution network of the generator on set
#
###

export PYTHONPATH=`pwd`/holophrasm
python train_pred_lm.py -b 100 -l 5e-4 \
    --log ../logs/set_gen_lm_1 \
    --output ../models/set_gen_lm_1 \
    --schedule 30 40 50 --epoch 60 --samples 6000 \
    --graph_opt adam

