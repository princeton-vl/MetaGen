#!/bin/bash
#
# run the Holophrasm prover
#

export PYTHONPATH=`pwd`/holophrasm

NUM_PROVERS=1 # the number of proving jobs that run in parallel
PROVER_ID=0 # ID of this proving job

RELEVANCE=../models/iset_pred_syn # path to the relevance network
SUBSTITUTION=../models/iset_gen_syn # path to the substitution network
PAYOFF=../models/iset_payout_syn # path to the payoff network
OUTPUT=../output/iset_syn_debug # path of the found proofs
NUM_WORKERS_PER_GPU=2 # number of workers used to complete this job. these workers run on the same GPU.

python run_script_ordered_multi.py \
    --num_provers $NUM_PROVERS --prover_id $PROVER_ID \
    --log ../logs/run_prover \
    --interface_pred_model $RELEVANCE \
    --interface_gen_model $SUBSTITUTION \
    --interface_payout_model $PAYOFF \
    --timeout 5 --passes 10000 \
    --output $OUTPUT --num_workers $NUM_WORKERS_PER_GPU \
    --iset --data-path ../data_iset # remove this line if test the prover on set.mm
