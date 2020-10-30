#!/bin/bash
#
# generate synethtc theorems on set
#

ID=0
NUM_PROVERS=1
export PYTHONPATH=`pwd`/holophrasm
python parallel_sampling_worker.py \
    --data_sampling 1.0 \
    --num_provers $NUM_PROVERS \
    --prover_id $ID \
    --log ../logs/sample_set_$ID \
    --sampling_time 1000 \
    --num_cons_exprs 1000000 \
    --expr_path ../exprs/set_1 \
    --interface_pred_model ../models/set_pred_lm_1 \
    --interface_gen_model ../models/set_gen_lm_1 \
    --expr_list ../data/expressions_list_1.0 \
    --iset --data-path ../data --gen_lm

