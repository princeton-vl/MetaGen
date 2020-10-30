import os,sys
import numpy as np
import math
import time
import random
import multiprocessing as mp

import torch
import torch.utils.data as data
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import torch_models
import params
import log
import data_loader_forward
from data_utils import *

def embed(tokens, trees, emb, gru, args):
    def _init_hidden():
        hidden_layers = args.gru_depth * 2 if args.bidirectional else args.gru_depth
        return torch.zeros(hidden_layers, len(tokens), args.nFeats).to(args.device)
    padded_tokens, padded_trees, perm_idx, lengths = pad_list(tokens, trees, args.device)
    padded_feats = emb(padded_tokens, padded_trees)
    packed_feats = pack_padded_sequence(padded_feats, lengths.cpu().numpy())
    packed_output, h = gru(packed_feats, _init_hidden())
    
    output = torch.cat((h[-2], h[-1]), 1) if args.bidirectional else h[-1]
    _, unperm_idx = perm_idx.sort(0)
    output = output[unperm_idx]
    return output

def step(batch, models, args):
    g_tokens, g_trees, p_tokens, p_trees = batch
    g_vec = embed(g_tokens, g_trees, models['emb'], models['g_gru'], args)
    p_vec = embed(p_tokens, p_trees, models['emb'], models['p_gru'], args)
    score = models['biln'](g_vec, p_vec).view(-1, args.negative_samples+1)
    
    return score # batch_size * (1+neg) no softmax

def step_hardneg(batch, models, args):
    g_tokens, g_trees, p_tokens, p_trees, prop_idx, true_prop = batch
    p_tokens_hard_neg = []
    p_trees_hard_neg = []
    with torch.no_grad():
        g_vec = embed(g_tokens, g_trees, models['emb'], models['g_gru'], args)
        cnt = 0
        for g, p_idx, true_pos in zip(g_vec, prop_idx, true_prop):
            l = len(p_idx) if type(p_idx) == list else p_idx
            p_emb = embed(p_tokens[cnt:cnt+l], p_trees[cnt:cnt+l], models['emb'], models['p_gru'], args)
            
            score = models['biln'](g.view(1,-1), p_emb).view(-1)
            score[true_pos] = -100
            hard_neg_idx = (-score).sort()[1][:args.negative_samples]
            p_tokens_hard_neg.append(p_tokens[cnt+true_pos])
            p_trees_hard_neg.append(p_trees[cnt+true_pos])
            for idx in hard_neg_idx:
                p_tokens_hard_neg.append(p_tokens[cnt+idx])
                p_trees_hard_neg.append(p_trees[cnt+idx])
            cnt += l
    return step([g_tokens, g_trees, p_tokens_hard_neg, p_trees_hard_neg], models, args)

def step_allneg(batch, models, args):
    g_tokens, g_trees, p_tokens, p_trees, prop_idx, true_prop = batch
    g_vec = embed(g_tokens, g_trees, models['emb'], models['g_gru'], args)
    p_vec = embed(p_tokens, p_trees, models['emb'], models['p_gru'], args)
    loss = []
    acc = 0
    cnt = 0
    for g, p_idx, true_pos in zip(g_vec, prop_idx, true_prop):
        l = len(p_idx) if type(p_idx) == list else p_idx
        p_emb = p_vec[cnt:cnt+l]
        cnt += l
        score = models['biln'](g.view(1,-1), p_emb).view(-1)
        #print (score[true_pos], score)
        idx = score[true_pos]<=score
        _pos = idx.sum().item()
        if _pos == 1:
            acc += 1
        if args.train_neg_only:
            loss.append(-score[true_pos] + score[idx].exp().sum().log())
        else:
            loss.append(-score[true_pos] + score.exp().sum().log())
    return sum(loss)/len(loss), acc


def train(models, queue, opt, args, _logger, epoch, _train=True):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acces = AverageMeter()
    acces_cons = AverageMeter()
    ppls = AverageMeter()
    # switch to train mode
    for k,v in models.items():
        if _train:
            v.train()
        else:
            v.eval()
    end = time.time()
    i = 0
    if args.cur_iter != 0:
        i = args.cur_iter
        args.cur_iter = 0
    while i < args.samples:
        batch = queue.get()
        if batch is None:
            _logger.warning('An epoch end')
            if _train==False:
                break
            batch = queue.get()

        b0 = [torch.Tensor(d).long().to(args.device) for d in batch[0]]
        b1 = [torch.Tensor(d).to(args.device) for d in batch[1]]
        b2 = [torch.Tensor(d).long().to(args.device) for d in batch[2]]
        b3 = [torch.Tensor(d).to(args.device) for d in batch[3]]
        data_time.update(time.time() - end)
        if args.allneg:
            loss, acc = step_allneg((b0,b1,b2,b3, batch[4], batch[5]), models, args)
            print (loss, acc)
        else:
            if args.hardneg:
                score = step_hardneg((b0,b1,b2,b3, batch[4], batch[5]), models, args)

            else:
                score = step((b0,b1,b2,b3), models, args)
            acc = (score.max(dim=1)[1]==0).sum().item()
            prob = torch.nn.functional.softmax(score, dim=1)[:,0]
            ppl = prob.sum().item()
            score[:, 1:] = -score[:, 1:]
            score = torch.nn.functional.logsigmoid(score)
            loss = -score.sum(dim=1).mean()
        # backward
        if _train:
            opt.zero_grad()
            loss.backward()
            opt.step()
        # update
        ppls.update(ppl, args.batch_size)
        acces.update(acc, args.batch_size)
        losses.update(loss.item(), args.batch_size)
        batch_time.update(time.time() - end)
        end = time.time()
        print_info  = 'Epoch: %d (%d/%d) Data: %.4f Batch: %.4f Loss: %.4f Acc: %.4f ppl: %.4f' % (
                epoch,
                i + 1,
                args.samples,
                data_time.val,
                batch_time.val,
                losses.avg*100,
                acces.avg*100,
                ppls.avg
                )
        _logger.info(print_info)
        if ((i+1) % args.save == 0 or (i+1) % args.samples == 0) and _train:
            aux = {'epoch':epoch, 'cur_iter':i}
            save_model(aux, args, models, opt)
            _logger.warning('save models')
        i += 1
    return losses.avg, acces.avg

def evaluate(models, loader, args, _logger):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    acces_1 = AverageMeter()
    acces_5 = AverageMeter()
    acces_20 = AverageMeter()
    rank = []
    for k,v in models.items():
        v.eval()
    prop_embs = torch.zeros(len(loader.dataset.lm_inputs), args.nFeats*2 if args.bidirectional else args.nFeats).to(args.device)
    l = 0
    while l < len(loader.dataset.lm_inputs):
        r = min(len(loader.dataset.lm_inputs), l+args.batch_size*10)
        tokens = [torch.LongTensor(loader.dataset.lm_inputs[i][0]).to(args.device) for i in range(l, r)]
        trees = [torch.Tensor(loader.dataset.lm_inputs[i][1]).to(args.device) for i in range(l, r)]
        prop_embs[l:r] = embed(tokens, trees, models['emb'], models['p_gru'], args)
        l = r
        _logger.info('Embed %d/%d props', l, len(loader.dataset.lm_inputs))

    end = time.time()
    i=0
    while loader.dataset.cur_epoch == 0:
        batch = next(iter(loader))
        data_time.update(time.time() - end)
        g_emb = embed(batch[0], batch[1], models['emb'], models['g_gru'], args)
        prop_idx = batch[-2]
        true_prop = batch[-1]
        for g, p_idx, true_pos in zip(g_emb, prop_idx, true_prop):
            #print (p_idx)
            p_emb = prop_embs[torch.Tensor(p_idx).to(device).long()]
            score = models['biln'](g.view(1,-1), p_emb)
            #print (score[true_pos], score)
            _pos = (score[true_pos]<score).sum().item()
            rank.append(_pos)
            if _pos==0:
                acces_1.update(1,1)
            else:
                acces_1.update(0,1)
            if _pos<5:
                acces_5.update(1,1)
            else:
                acces_5.update(0,1)
            if _pos<20:
                acces_20.update(1,1)
            else:
                acces_20.update(0,1)
            end = time.time()
        print_info  = '(%d/%d) Data: %.4f | \
                    Batch: %.4f | top1: %.4f | top5: %4f |\
                    top20: %.4f' % (
                    i + 1,
                    len(loader),
                    data_time.val,
                    batch_time.val,
                    acces_1.avg*100,
                    acces_5.avg*100,
                    acces_20.avg*100
                    )
        _logger.info(print_info)
        print (sum(rank)/len(rank))
        i += 1
    _logger.warning('top1: %.4f top5: %.4f top20: %.4f',
            acces_1.avg*100, acces_5.avg*100, acces_20.avg*100)

def start_loader(args, task):
    args.device = None
    q1 = mp.Queue(50)
    proc = mp.Process(
                    target=data_loader_forward.worker,
                    args=(args, 0, task, q1)
                )
    proc.daemon=True
    proc.start()
    args.device = torch.device("cuda:0" if (not args.cpu) and torch.cuda.is_available() else "cpu")
    return proc, q1

def close_loader(proc, q1):
    proc.terminate()
    q1.close()

    proc.join()

if __name__ == "__main__":
    args = params.get_args()

    _logger = log.get_logger(__name__, args)
    _logger.info(print_args(args))

    args.valid_split = 'test' if args.evaluate else 'valid'
    proc, q1 = start_loader(args, 'pred')

    if args.partial_lm:
        lm = build_language_model(args.num_props, new=args.gen_lm, _all=False)
    else:
        lm = load_language_model(new=args.gen_lm, iset=args.iset)
    config = get_config(lm)

    args.vocab_size = len(config.encode)+1
    models = {}
    models['emb'] = torch_models.Emb(args, config).to(args.device)
    in_dim = args.nFeats
    if not args.no_use_tree:
        in_dim += 4

    models['g_gru'] = torch.nn.GRU(in_dim, args.nFeats, args.gru_depth,
            dropout=args.dropout, bidirectional=args.bidirectional).to(args.device)
    models['p_gru'] = torch.nn.GRU(in_dim, args.nFeats, args.gru_depth,
            dropout=args.dropout, bidirectional=args.bidirectional).to(args.device)
    models['biln'] = torch_models.Dot(args).to(args.device)

    opt = get_opt(models, args)

    start_epoch = 0
    args.cur_iter = 0
    if args.pretrained != "":
        loaddata = torch.load(args.pretrained)
        start_epoch = loaddata['aux']['epoch']
        args.cur_iter = loaddata['aux']['cur_iter']
        for k,v in models.items():
            v.load_state_dict(loaddata['models'][k])
        opt.load_state_dict(loaddata['opt'])
        _logger.warning('load data from %s\n', args.pretrained)
    for param_group in opt.param_groups:
        param_group['lr'] = args.learning_rate

    if args.evaluate != 'none':
        _logger.warning('Evaluation begins')
        with torch.no_grad():
            evaluate(models, validloader, args, _logger)
        exit(0)

    valid_losses = []
    valid_acces = []
    for e in range(start_epoch, args.epoch):
        _logger.warning("Training Epoch %d begins.", e)
        loss, acc = train(models, q1, opt, args, _logger, e)
        if e in args.schedule:
            adjust_lr(opt, args)
            _logger.warning('Learning rate decreases')
    close_loader(proc, q1)


