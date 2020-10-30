import write_proof
import time
import log
import os
from tree_parser import *
from data_utils5 import *
import pickle as pickle
import numpy as np
import params
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import data_utils
import proof_search_torch
import proof_search_cheat_rand

def print_args(args):
    s = '\nParameter settings:\n'
    for name, value in args.__dict__.items():
        s += str(name) + ':' + str(value) + '\n'
    return s

def run_all_props_with_current_settings(args, vprops, _logger, language_model):
    i=-1
    num_proven = 0
    num_this_proven = 0
    num_try = 0
    directory = args.output
    if args.gt_subs or args.gt_prop or args.gt_payout:
        if args.rand_prop > 0:
            proof_search = proof_search_cheat_rand
        else:
            proof_search = proof_search_cheat
    else:
        proof_search = proof_search_torch
    for prop in vprops:
        i+=1

        num_try += 1
        _logger.info('proposition {0} ({1}): {2}/{3} (with {4} steps)'.format(prop.label,prop.number,i,len(vprops), prop.num_entails_proof_steps))
        if prop.type == 'a':
            continue
        if prop.label == 'pm5.19':
            continue
        if write_proof.proof_exists(prop.label, directory=directory, proofdir=args.proofdir):
            num_proven += 1
            continue
        # set up the searcher
        searcher = proof_search.ProofSearcher(args, prop, language_model, directory=directory, timeout=args.timeout)
        # run the searcher
        t = time.time()
        searcher.run(args.passes, multi=True, threads=args.threads, print_output=False, clear_output=False)
        if searcher.passes > 0:
            _logger.info('go through %d passes, %.5f seconds per pass' % (searcher.passes, (time.time()-t)/searcher.passes))
            _logger.info('query tree size is %.5f' % (searcher.tree_size / (searcher.gen_cnt if searcher.gen_cnt > 0 else 1)))
            _logger.info('query pred network %d times in %.5f seconds, %.5f seconds per query' % (searcher.pred_cnt, searcher.pred_time, searcher.pred_time/(searcher.pred_cnt if searcher.pred_cnt > 0  else 1)))
            _logger.info('query gen network %d times in %.5f seconds, %.5f seconds per query' % (searcher.gen_cnt, searcher.gen_time, searcher.gen_time/(searcher.gen_cnt if searcher.gen_cnt > 0 else 1)))
            _logger.info('query payout network %d times in %.5f seconds, %.5f seconds per query' % (searcher.payout_cnt, searcher.payout_time, searcher.payout_time/(searcher.payout_cnt if searcher.payout_cnt > 0 else 1)))
        if searcher.proven():
            num_proven += 1
            num_this_proven += 1
            searcher.proof_object()

        _logger.warning('proven {0}/{1} this: {3}, {2:5.1f}%'.format(num_proven, num_try, 100 *num_proven/(i+1.0), num_this_proven))

def run(args, worker_id):

    args.log += '_'+str(worker_id)
    _logger = log.get_logger('prover', args)
    args.device = torch.device("cuda:0" if (not args.cpu) and torch.cuda.is_available() else "cpu")
    if args.partial_lm:
        language_model = data_utils.build_language_model(args.num_props, new=args.gen_lm, _all=False)
    else:
        language_model = data_utils.load_language_model(iset=args.iset)

    num_jobs = args.num_provers * args.num_workers
    job_id = args.prover_id*args.num_workers+worker_id
    _logger.info('num_jobs: %d job_id: %d', num_jobs, job_id)
    x = [(p.num_entails_proof_steps, p.number, p.label) for p in language_model.test_propositions
                 if p.vclass == '|-' and p.type=='p']
    x.sort()
    _, _, labels = zip(*x)

    vprops = [language_model.database.propositions[l] for l in labels]
    vprops = [vprops[i] for i in range(len(vprops)) if i % num_jobs == job_id]
    _logger.info([p.label for p in vprops])

    if args.gt_subs or args.gt_prop:
        proof_search = proof_search_cheat
    else:
        proof_search = proof_search_torch

    settings = [(True, 1), (True, 5), (False, 1), (False, 5), (True, 20), (False, 20)]
    for i in args.prover_settings:
        sets, beam_size = settings[i]
        print (sets, beam_size)
        proof_search.HYP_BONUS = args.hyp_bonus
        proof_search.BEAM_SIZE = beam_size
        proof_search.interface_torch_models.gen_model_beam_search.PERMIT_NEW_SETS = sets
        proof_search.interface_torch_models.gen_model_beam_search_torch.PERMIT_NEW_SETS = sets
        run_all_props_with_current_settings(args, vprops, _logger, language_model)

if __name__ == "__main__":

    mp.set_start_method('spawn')

    args = params.get_args()
    _logger = log.get_logger('prover', args)
    _logger.info(print_args(args))
    directory = args.output
    if args.threads == 0:
        args.threads = None
    print (args.device)

    plist = []
    args.device = None
    for i in range(args.num_workers):
        plist.append(mp.Process(target=run, args=(args, i)))
        plist[-1].start()

    for p in plist:
        p.join()
