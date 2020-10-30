#!/bin/bash
#
# generate auxiliary data for the generator.
#
###

export PYTHONPATH=`pwd`/holophrasm

# run the following command for experiments on iset
python generate_forward_data.py --iset --data-path ../data_iset

# run the following command for experiments on set
python generate_forward_data.py --gen_lm # use 100% of human proofs
python generate_forward_data.py --gen_lm --data_sampling 0.1 # use 10% of human proofs
python generate_forward_data.py --gen_lm --data_sampling 0   # use 0% of human proofs
