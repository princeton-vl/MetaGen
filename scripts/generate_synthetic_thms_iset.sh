#!/bin/bash
#
# generate synethtc theorems on iset
#

ID=0
NUM_PROVERS=1
export PYTHONPATH=`pwd`/holophrasm
python parallel_sampling_worker.py \
    --data_sampling 1.0 \
    --num_provers $NUM_PROVERS \
    --prover_id $ID \
    --log ../logs/sample_iset_$ID \
    --sampling_time 1000 \
    --num_cons_exprs 1000000 \
    --expr_path ../exprs/iset \
    --interface_pred_model ../models/iset_pred_lm \
    --interface_gen_model ../models/iset_gen_lm \
    --expr_list ../data_iset/expressions_list_1.0 \
    --iset --data-path ../data_iset

