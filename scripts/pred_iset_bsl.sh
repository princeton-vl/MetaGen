#!/bin/bash
#
# train the baseline relevance network of the prover on iset.
#
###

export PYTHONPATH=`pwd`/holophrasm
python train_baseline.py -l 1e-3 --save 100 \
    --log ../logs/iset_pred_bsl \
    --output ../models/iset_pred_bsl \
    --schedule 12 20 28 --epoch 40 \
    --task pred --graph_opt adam --iset --data-path ../data_iset

