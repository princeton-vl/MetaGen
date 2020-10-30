#!/bin/bash
#
# training script for the relevance network of the generator
#
###

export PYTHONPATH=`pwd`/holophrasm
python train_pred_lm.py --negative-samples 10 -b 100 -l 1e-3 \
    --log ../logs/set_pred_lm_1 \
    --output ../models/set_pred_lm_1 \
    --schedule 30 40 50 --epoch 60 --samples 6000 \
    --graph_opt adam

