#!/bin/bash
#
# train the substitution network of the prover on bath human data and synthetic data
#
###

export PYTHONPATH=`pwd`/holophrasm
python train_augmentations.py \
    --log ../logs/iset_gen_syn \
    --output ../models/iset_gen_syn \
    --task gen -l 5e-4 --cons_batch 70 --batch_size 30 \
    --schedule 16 24 32 --epoch 40 \
    --precalculate --exprs_pre ../exprs/iset_precompute --expr_list ../data_iset/expressions_list_1.0 \
    --cons_pre_one --iset --data-path ../data_iset --graph_opt adam
