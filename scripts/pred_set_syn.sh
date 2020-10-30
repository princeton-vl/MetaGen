#!/bin/bash
#
# train the relevance network of the prover on both human data and
# synthetic data for set
#

export PYTHONPATH=`pwd`/holophrasm
python train_augmentations.py \
    --log ../logs/set_pred_syn_1 \
    --output ../models/set_pred_syn_1 \
    --task pred -l 1e-3 --cons_batch 50 --batch_size 50 \
    --schedule 6 10 14 --epoch 20 \
    --precalculate --exprs_pre ../exprs/set_1_precompute \
    --expr_list ../data/expressions_list_1.0 \
    --graph_opt adam --gen_lm
