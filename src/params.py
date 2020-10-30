import os
import argparse
import torch

def get_args():
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--log', type=str, default='../logs/debug', help='path of log file')
    parser.add_argument('--output', type=str, default='../models/debug', help='path to save models')
    parser.add_argument('--data-path', type=str, default='../data/', help='../data or ../data_iset')
    parser.add_argument('--split', type=str, default='train', help='data split')
    parser.add_argument('--gen_lm', action='store_true', help='use lm_gen; specific to the generator of set.')
    parser.add_argument('--nFeats', type=int, default=128, help='feature dimensions')
    parser.add_argument('--negative-samples', type=int, default=4,
        help='number of negative samples to train the relevance network')
    parser.add_argument('--no_use_tree', action='store_true', help='do not use tree info in input; always false in our experiment')
    parser.add_argument('--num_workers', type=int, default=1, help='number of workers for data loader or prover')
    parser.add_argument('--gru_depth', type=int, default=2, help='number of gru layers')
    parser.add_argument('--decoder_gru_depth', type=int, default=2, help='number of gru layers in the decoder of the subsitution network')
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--no_bidirectional', action='store_true', help='do not use bidirectional gru; always false')
    parser.add_argument('--schedule', type=int, nargs='+', default=[15, 20, 25],
        help='Decrease learning rate at these epochs.')
    parser.add_argument('--epoch', type=int, default=30, help='number of epochs for training')
    parser.add_argument('--learning_rate', '-l', type=float, default=1e-4)
    parser.add_argument('--samples', type=int, default=6000,
        help='number of training iterations for each epoch to train the relevance and subsitution network of the generator.')
    parser.add_argument('--batch_size', '-b', type=int, default=100)
    parser.add_argument('--pretrained', type=str, default='', help='path to a pretrained model')
    parser.add_argument('--save', type=int, default=1000, help='save models after certain iterations.')
    parser.add_argument('--lr_decay', type=float, default=0.5, help='learning rate decay')
    parser.add_argument('--evaluate', type=str, default='none', help='option: valid, train, valid_all, train_all')
    parser.add_argument('--max_gen_len', type=int, default=200, help='maximum number of tokens for a generated expression')
    parser.add_argument('--max_subs_len', type=int, default=75, help='maximum number of tokens for a generated subsitutiton')
    # proof_interface
    parser.add_argument('--beam_width', type=int, default=10, help='the width of beam search to generate subsitutions')
    parser.add_argument('--interface_pred_model', type=str, default='', help='path to the pretrained relevance network of the prover/generator')
    parser.add_argument('--interface_gen_model', type=str, default='', help='path to the pretrained subsitution network of the prover/generator')
    parser.add_argument('--interface_payout_model', type=str, default='', help='path to the pretrained payoff network of the prover')
    parser.add_argument('--timeout', type=int, default=5, help='time limit to prove each theorem')
    parser.add_argument('--passes', type=int, default=10000, help='limit of MCTS passes to prove each theorem')
    parser.add_argument('--hyp_bonus', type=float, default=0)
    parser.add_argument('--no_use_torch', action='store_true', help='use cpu')
    parser.add_argument('--threads', type=int, default=0, help='number of threads used in prover; 0 in default')
    parser.add_argument('--num_provers', type=int, default=1, help='number of provers to run on the test set')
    parser.add_argument('--prover_id', type=int, default=0, help='ID of the prover job')
    # generator
    parser.add_argument('--sample_width', type=int, default=20,
        help='number of candidate trees as the input of the relevance network of the generator.')
    parser.add_argument('--random_generate',action='store_true', help='use MetaGen-Rand')
    parser.add_argument('--cons_batch', type=int, default=20, help='number of samples from synthetic data in each training batch')
    parser.add_argument('--expr_path', type=str, default='../exprs/debug', help='path of the tmp folder for the generated synthetic theorems')
    parser.add_argument('--sampling_length', type=int, default=1000)
    parser.add_argument('--expr_list', type=str, nargs='+', default=['../data/expressions_list'],
        help='path to the folder of expressions in training data')

    parser.add_argument('--partial_lm', action='store_true', help='use a subset of the database')
    parser.add_argument('--num_props', type=int, default=30000, help='number of theorems used in a subset of the database')
    parser.add_argument('--validall', action='store_true', help='run test on training data')
    parser.add_argument('--allneg', action='store_true', help='use all negative examples')
    parser.add_argument('--cat', action='store_true',
        help='True: concatenate the embeddings of the goal and the background theorem and send to linear layers; \
              False: use a bilinear layer to combine the embeddings of the goal and the background theorem')
    parser.add_argument('--data_sampling', type=float, default='1', help='portion of human-written proofs used during training')
    parser.add_argument('--train_neg_only', action='store_true', help='train the relevance network on negative examples only.')
    parser.add_argument('--task', type=str, default='pred', help='pred: relevance network; gen: subsitutiton network')
    parser.add_argument('--num_iters', type=int, default=1000, help='number of iterations each epoch')

    parser.add_argument('--hardneg', action='store_true', help='use hard negative mining')

    parser.add_argument('--num_cons_exprs', type=int, default=3000, help='number of theorems to generate')
    parser.add_argument('--proofdir', type=str, default='proofs', help='path to the folder of proofs found on test set.')

    parser.add_argument('--num_episode', type=int, default=100, help='number of episode to train MetaGen-RL')
    parser.add_argument('--len_episode', type=int, default=40, help='number of iterations for one episode of MetaGen-RL')
    parser.add_argument('--train_rl', action='store_true', help='train MetaGen-RL')
    parser.add_argument('--debug', action='store_true', help='use debug mode')
    parser.add_argument('--lm_model', type=str, default='', help='path to the pretrained networks for the generator')
    parser.add_argument('--fake_reward', action='store_true',
        help='use the fake reward of the number of human theorems generated by MetaGen-RL.')
    parser.add_argument('--interface_model', type=str, default='', help='path to the pretrained networks of the prover')

    parser.add_argument('--max_len', type=int, default=300,
        help='maximum number of tokens for each expression; discard the rest expression')
    parser.add_argument('--short_file_list', action='store_true', help='debug on one training file')
    parser.add_argument('--repeatable', action='store_true', help='enable repeat generation of synthetic theorems')
    parser.add_argument('--num_old_exprs', type=int, default=1e6,
        help='hold a certain number of synthetic theorems; useful when we generate synthetic data and train the prover simultaneously')
    parser.add_argument('--new_expr_ratio', type=float, default=0.1,
        help='probability to sample a new theorem when we a synthetic theorem. \
              useful when we generate synthetic data and train the prover simultaneously')
    parser.add_argument('--num_gpus', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--sample_dist_ratio', type=float, default=1,
        help='probability to sample a invocable theorem according to the distribution of invocable theorems used in human proofs; \
              otherwise sample invocable theorems uniformly')
    parser.add_argument('--gpu_list', type=int, nargs='+', default=[-1], help='ID of GPUs to use.')
    parser.add_argument('--grad_clip', type=float, default=10, help='gradient clipping')
    parser.add_argument('--manual_out', action='store_true', help='print some output info separately')
    parser.add_argument('--sampling_time', type=int, default=30,
        help='time (seconds) for each worker to generate synthetic theorems separately before they communicate with each other')
    parser.add_argument('--precalculate', action='store_true',
        help='True: synthetic theorems are precalcualted; False: generate synthetic data during the training of the prover')
    parser.add_argument('--exprs_pre', type=str, default='', help='path to the precalculated synthetic theorems')

    parser.add_argument('--restart', action='store_true', help='train a pretrained network from epoch 0')
    parser.add_argument('--notrain', action='store_true', help='run the TF-IDF&LM baseline')
    parser.add_argument('--stat_model', action='store_true', help='tfidf, ngram')
    parser.add_argument('--gen_noproof', action='store_true', help='generate substitutions without human proofs')
    parser.add_argument('--ngram', action='store_true', help='use a ngram model in place of the substitution network')
    parser.add_argument('--pretraineds', type=str, nargs='+', default=[],
        help='path to the pretrained models; used only for the training of language models as the substitution network')
    parser.add_argument('--train_from_queue', action='store_true', help='train the prover on human proofs only; used in train_augmentation.py')
    parser.add_argument('--epsilon', type=float, default='1', help='probability to take a random step when we train MetaGen-RL')
    parser.add_argument('--cons_pre_one', action='store_true', help='we have a unique file for precalculated synthetic theorems')
    parser.add_argument('--thm_emb', action='store_true', help='learn a dense embedding for each background theorem')

    parser.add_argument('--gt_subs', action='store_true', help='run the prover with an oracle of the subsitution network')
    parser.add_argument('--gt_prop', action='store_true', help='run the prover with an oracle of the relevance network')
    parser.add_argument('--gt_payout', action='store_true', help='run the prover with an oracle of the payoff network')
    parser.add_argument('--gpu_id', type=float, default=0, help='which GPU to use')

    parser.add_argument('--graph_classifier', type=str, default='linear', help='[linear, bilinear], refer to torch_models.classifier_dict')
    parser.add_argument('--graph_opt', type=str, default='rmsprop', help='[rmsprop, adam], refer to data_utils.opt_dict')

    parser.add_argument('--old_vocab', action='store_true', help='use an old vocabulary for compatibility')

    parser.add_argument('--n1_jobs', type=int, default=2, help='number of subprocesses for a prover')
    parser.add_argument('--n2_jobs', type=int, default=4, help='number of subprocesses for a prover')
    parser.add_argument('--prover_settings', type=int, nargs='+', default=[0,1,2,3,4,5],
        help='the configuration to test the prover; see variable settings in run_script_ordered_multi.py')
    parser.add_argument('--save_score', action='store_true', help='save the evaluate scores for debugging')

    parser.add_argument('--iset', action='store_true', help='run experiments for iset.mm')

    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()
    args.bidirectional = not args.no_bidirectional
    args.device = torch.device("cuda:%d"%(args.gpu_id) if (not args.cpu) and torch.cuda.is_available() else "cpu")
    args.cpu = args.cpu or not torch.cuda.is_available()
    return args
