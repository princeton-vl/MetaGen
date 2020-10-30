#!/bin/bash
#
# train the substitution network of the prover on both human data and
# synthetic data for set
#

export PYTHONPATH=`pwd`/holophrasm
python train_augmentations.py \
    --log ../logs/set_gen_syn_1 \
    --output ../models/set_gen_syn_1 \
    --task gen -l 5e-4 --cons_batch 50 --batch_size 50 \
    --schedule 10 15 20 --epoch 24 \
    --precalculate --exprs_pre ../exprs/set_1_precompute \
    --expr_list ../data/expressions_list_1.0 \
    --graph_opt adam --gen_lm
