#!/bin/bash
#
# train the relevance network of the prover on both human data and
# synthetic data for iset
#

export PYTHONPATH=`pwd`/holophrasm
python train_augmentations.py \
    --log ../logs/iset_pred_syn \
    --output ../models/iset_pred_syn \
    --task pred -l 1e-3 --cons_batch 20 --batch_size 80 \
    --schedule 12 20 28 --epoch 40 \
    --precalculate --exprs_pre ../exprs/iset_precompute --expr_list ../data_iset/expressions_list_1.0 \
    --cons_pre_one --iset --data-path ../data_iset --graph_opt adam
