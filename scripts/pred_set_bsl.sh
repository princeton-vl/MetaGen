#!/bin/bash
#
# train the baseline relevance network of the prover on set.
#
###

export PYTHONPATH=`pwd`/holophrasm
python train_baseline.py -l 1e-3 --save 100 \
    --log ../logs/set_pred_bsl_1 \
    --output ../models/set_pred_bsl_1 \
    --schedule 16 24 32 --epoch 40 \
    --task pred --graph_opt adam

