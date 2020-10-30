# MetaGen

Code for reproducing the results in the following paper:

**[Learning to Prove Theorems by Learning to Generate Theorems](https://arxiv.org/abs/2002.07019)**  
Mingzhe Wang, Jia Deng  
*Neural Information Processing Systems (NeurIPS), 2020*  

## Package dependency
- Python3
- PyTorch 0.4.0
- Cuda

## Data preprocessing

We follow the same data preprocessing as [Holophrasm](https://github.com/dwhalen/holophrasm) to extract theorems and their proofs from the Metamath `set.mm` and `iset.mm` knowledge bases.

```
python data_utils.py
```

The proof steps are split into the training, validation and testing splits and stored in `data` and `data_iset` folders.

## Pretrained models

We provide our pretrained models of the theorem prover and the theorem generator on both `set.mm` and `iset.mm` benchmarks. Please refer to `models` for the usage of our pretrained models.

To run our pretrained theorem prover, please run the commands in `script/run_prover.sh`. You can find the proofs found by our provers on the test data in `output`.

## Training your own models

To train the theorem generator

```
cd src
bash ../script/forward_data.sh
bash ../script/pred_iset_lm.sh # train the relevance network of the generator on iset.mm
bash ../script/gen_iset_lm.sh # train the substitution network of the generator on iset.mm
bash ../script/pred_set_lm.sh # train the relevance network of the generator on set.mm
bash ../script/gen_set_lm.sh # train the substitution network of the generator on set.mm
```

To generate synthetic theorems using the theorem generator

```
mkdir ../exprs
bash ../script/generate_synthetic_thms_iset.sh # for iset.mm
bash ../script/generate_synthetic_thms_set.sh # for set.mm
```

To train the theorem prover on both human-written and synthetic proofs

```
bash ../script/pred_iset_syn.sh # train the relevance network of the prover on iset.mm
bash ../script/gen_iset_syn.sh # train the substitution network of the prover on iset.mm
bash ../script/pred_set_syn.sh # train the relevance network of the prover on set.mm
bash ../script/gen_set_syn.sh # train the substitution network of the prover on set.mm
```

Please contact us if you run into any issues or have any questions.

## Acknowledgement
This work is partially supported by the National Science Foundation under
Grant IIS-1903222 and the Office of Naval Research under Grant N00014-20-1-2634.
