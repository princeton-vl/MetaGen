import os,sys
import numpy as np
import math
import time
import random
import torch.multiprocessing as mp

import torch
import torch.utils.data as data
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import torch_models
import params
import log
#from data_loader_pred import *
from data_utils import *
import interface_lm
from constructor import Constructor
import trainer
import data_loader_cons
import lm_model
from scipy.stats import ks_2samp

class AverageMeterVal(AverageMeter):
    def __init__(self):
        AverageMeter.__init__(self)
        self.vals = []
    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count
        self.vals.append(val)

def train(steps):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    ppl = AverageMeter()    
    acc = AverageMeter()
    

def evaluate(loader, exprs):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    ppl_pos = AverageMeterVal()
    ppl_neg = AverageMeterVal()
    acc_pos = AverageMeterVal()
    acc_neg = AverageMeterVal()
    log_softmax = torch.nn.LogSoftmax(dim=1)
    def _eval_batch(batch, ppl, acc, _iter, num_iters):
        end = time.time()
        data_time.update(time.time() - end)
        score = model(batch)
        for j, tokens in enumerate(batch[-2]):
            if len(tokens) > args.max_len:
                continue
            gt = torch.zeros(len(tokens)).long().to(args.device)
            gt[:-1] = tokens[1:]
            gt[-1] = config.encode['END_OF_SECTION']
            _score = score[j, :len(tokens)]
            acc.update(( _score.max(dim=1)[1]==gt).sum().item()/len(gt), 1)
            _score = log_softmax(_score)
            _ppl = torch.exp(_score.gather(1, gt.view(-1,1)).sum() / len(gt))
            #print (_ppl)
            ppl.update(_ppl.item(), 1)
        batch_time.update(time.time() - end)
        end = time.time()
        print_info  = '(%d/%d) Data: %.4f | \
                      Batch: %.4f | Perplexity %.4f Acc %.4f' % (
                              _iter,
                              num_iters,
                              data_time.val,
                              batch_time.val,
                              ppl.avg,
                              acc.avg
                              )
        _logger.info(print_info)
    for i, batch in enumerate(loader):
        _eval_batch(batch, ppl_pos, acc_pos, i, len(loader))
    l = 0
    while l < len(exprs):
        r = min(l+args.batch_size, len(exprs))
        batch = model.encode_expr(exprs[l:r])
        _eval_batch(batch, ppl_neg, acc_neg, l//args.batch_size, len(exprs)//args.batch_size)
        l += args.batch_size
    _logger.warning(
            'acc_pos: %.4f ppl_pos: %.4f acc_neg: %.4f ppl_neg: %.4f',
            acc_pos.avg, ppl_pos.avg, acc_neg.avg, ppl_neg.avg)
    result = ks_2samp(ppl_pos.vals, ppl_neg.vals)
    _logger.warning(
            'statistic: %.4f pvalue: %.4f',
            result.statistic, result.pvalue)
    model.train()

if __name__ == "__main__":

    args = params.get_args()
    args.task = 'gen'

    _logger = log.get_logger(__name__, args)
    _logger.info(print_args(args))

    if args.partial_lm:
        lm = build_language_model(args.num_props, new=args.gen_lm, _all=False)
    else:
        lm = load_language_model(_all=False, new=args.gen_lm)
        load_proof_steps_into_lm(lm, ['train', 'valid'])
    config = get_config(lm)
    args.vocab_size = len(config.encode)+1
    interface = interface_lm.LMInterface(args, lm)
    args.data_sampling = 1
    generator = Constructor(args, config, interface)
    generator.initialize()
    #args.data_sampling = 0 #TODO use generator with all exprs so that we have correct indexes of hyps 
    #model = lm_model.LM_HyoAssGen(args, generator, config).to(args.device)
    if args.ngram:
        model = lm_model.LM_nGram(args, generator, config)
    else:
        model = lm_model.LM_AssGenCondHyo(args, generator, config).to(args.device)

    import data_loader_lm as data_loader
    trainloader = torch.utils.data.DataLoader(
            data_loader.MetaMath(args, config, model), collate_fn=data_loader.mycollate,
            batch_size=args.batch_size, shuffle=True,
            num_workers=0)#args.num_workers)
    valid_split = 'test' if args.evaluate else 'valid'
    validloader = torch.utils.data.DataLoader(
            data_loader.MetaMath(args, config, model, split=valid_split, evaluate=True),
            collate_fn=data_loader.mycollate,
            batch_size=args.batch_size, shuffle=False,
            num_workers=0)
    exprs = load_exprs(args.expr_path, lm)
    exprs = [exprs[i] for i in range(len(exprs)) if i % 10 == 0]
    random.shuffle(exprs)
    print ('load %d exprs as negative' % (len(exprs)))
   
    if args.ngram:
        _logger.warning ('start to fit ngram')
        model.fit(trainloader.dataset.steps)
        _logger.warning ('start to eval on valid') 
        ppl_pos = model.forward(validloader.dataset.steps) 
        _logger.warning ('ppl: %.4f', sum(ppl_pos)/len(ppl_pos))
        _logger.warning ('start to eval on samples')
        ppl_neg = model.forward(exprs)
        _logger.warning ('ppl: %.4f', sum(ppl_neg)/len(ppl_neg))
        result = ks_2samp(ppl_pos, ppl_neg)
        _logger.warning(
               'statistic: %.4f pvalue: %.4f',
               result.statistic, result.pvalue) 
        exit(0)

    if args.evaluate:
        for fp in args.pretraineds:
            _logger.warning('evaluate the model loaded from %s', fp)
            model.load(fp)
            evaluate(validloader, exprs)
            exit(0)
     
    training_object = trainer.Trainer(args, config, model, generator, _logger, validloader)
    while training_object.cur_epoch < args.epoch:
        training_object.train_from_loader(trainloader)
        if training_object.cur_epoch % 3 == 0:
            with torch.no_grad():
                evaluate(validloader, exprs)
        aux = {'epoch':training_object.cur_epoch, 'cur_iter':999}
        save_model(aux, args, training_object.model, opt=None)
        training_object.cur_epoch += 1 

