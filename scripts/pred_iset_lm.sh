#!/bin/bash
#
# training script for the relevance network of the generator
#
###

export PYTHONPATH=`pwd`/holophrasm
python train_pred_lm.py --negative-samples 4 -b 100 -l 1e-3 \
    --log ../logs/iset_pred_lm \
    --output ../models/iset_pred_lm \
    --schedule 20 30 40 --epoch 50 --samples 2000 \
    --graph_opt adam --iset --data-path ../data_iset

