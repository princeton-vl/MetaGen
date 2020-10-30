#!/bin/bash
#
# training script for the substitution network of the generator
#
###

export PYTHONPATH=`pwd`/holophrasm
python train_gen_lm.py -b 100 -l 5e-4 \
    --log ../logs/iset_gen_lm \
    --output ../models/iset_gen_lm \
    --schedule 20 30 40 --epoch 50 --samples 2000 \
    --graph_opt adam --iset --data-path ../data_iset

