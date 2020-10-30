#!/bin/bash
#
# train the baseline substitution network of the prover on set.
#
###

export PYTHONPATH=`pwd`/holophrasm
python train_baseline.py -l 5e-4 --save 100 \
    --log ../logs/set_gen_bsl_1 \
    --output ../models/set_gen_bsl_1 \
    --schedule 16 24 32 --epoch 40 \
    --task gen --graph_opt adam

