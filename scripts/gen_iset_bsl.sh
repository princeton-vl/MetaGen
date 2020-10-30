#!/bin/bash
#
# train the baseline substitution network of the prover on iset.
#
###

export PYTHONPATH=`pwd`/holophrasm
python train_baseline.py -l 5e-4 --save 100 \
    --log ../logs/iset_gen_bsl \
    --output ../models/iset_gen_bsl \
    --schedule 16 24 32 --epoch 40 \
    --task gen --graph_opt adam --iset --data-path ../data_iset

