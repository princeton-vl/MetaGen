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
from constructor import Constructor

end_output = None

def embed(tokens, trees, emb, args):
    padded_tokens, padded_trees, perm_idx, lengths = pad_list(tokens, trees, args.device)
    padded_feats = emb(padded_tokens, padded_trees)
    packed_feats = pack_padded_sequence(padded_feats, lengths.cpu().numpy())
    return packed_feats, perm_idx

def encoder(packed_feats, perm_idx, gru, args):
    def _init_hidden():
        hidden_layers = args.gru_depth * 2 if args.bidirectional else args.gru_depth
        return torch.zeros(hidden_layers, len(perm_idx), args.nFeats).to(args.device)
    packed_output, h = gru(packed_feats, _init_hidden())
    output, _ = pad_packed_sequence(packed_output)
    _, unperm_idx = perm_idx.sort(0)
    if args.bidirectional:
        h = h.view(-1, 2, len(perm_idx), args.nFeats)
        h = torch.cat((h[:,0,:,:], h[:,1,:,:]), dim=2)
    out_h = h[:,unperm_idx,:]
    output = output[:,unperm_idx,:]
    return output, out_h

def decoder(feats, in_h, perm_idx, gru, args):
    in_h = in_h[:, perm_idx, :]
    packed_output, h = gru(feats, in_h)


def step(batch, models, args):
    in_tokens, in_trees, out_tokens, out_trees = batch
    in_packed_feats, in_perm_idx = embed(in_tokens, in_trees, models['emb'], args)
    out_packed_feats, out_perm_idx = embed(out_tokens, out_trees, models['emb'], args)
    encoder_out, encoder_h = encoder(in_packed_feats, in_perm_idx, models['e_gru'], args)
    score = models['decoder'](out_packed_feats, encoder_h, out_perm_idx)
    return score # batch_size * (1+neg) no softmax

def train(models, queue, opt, criterion, args, _logger, epoch, _train=True):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acces = AverageMeter()
    # switch to train mode
    for k,v in models.items():
        if _train:
            v.train()
        else:
            v.eval()
    end = time.time()
    _iter = 0
    if args.cur_iter != 0:
        _iter = args.cur_iter
        args.cur_iter = 0
    for i in range(args.samples):
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
        # measure data loading time
        data_time.update(time.time() - end)
        score = step((b0, b1, b2, b3), models, args)
        loss = []
        acc = []
        for i,tokens in enumerate(b2):
            gt = torch.zeros(len(tokens)).long().to(args.device)
            gt[:-1] = tokens[1:]
            gt[-1] = end_output #ugly hack
            _score = score[i, :len(tokens)]
            loss.append(criterion(_score, gt))
            acc.append( ( _score.max(dim=1)[1]==gt).sum().item()/len(gt)  )

        loss = sum(loss)
        acc = sum(acc)
        if _train:
            opt.zero_grad()
            loss.backward()
            opt.step()
        # update
        acces.update(acc, args.batch_size)
        losses.update(loss.item(), score.shape[0])
        batch_time.update(time.time() - end)
        end = time.time()
        print_info  = 'Epoch: %d (%d/%d) Data: %.4f | Batch: %.4f | Loss: %.4f | Acc: %.4f' % (
                epoch,
                _iter + 1,
                args.samples,
                data_time.val,
                batch_time.val,
                losses.avg*100,
                acces.avg*100
                )
        _logger.info(print_info)
        if ((_iter+1) % args.save == 0 or (_iter+1)%args.samples == 0) and _train:
            aux = {'epoch':epoch, 'cur_iter':_iter+1}
            save_model(aux, args, models, opt)
            _logger.warning('save models')
        _iter += 1

    return losses.avg, acces.avg

def evaluate(models, loader, args, _logger):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    ppl = AverageMeter()
    for k,v in models.items():
        v.eval()

    end = time.time()
    i=0
    log_softmax = torch.nn.LogSoftmax(dim=1)
    while loader.dataset.cur_epoch == 0:
        batch = next(iter(loader))
        data_time.update(time.time() - end)
        score = step(batch, models, args)
        for j, tokens in enumerate(batch[2]):
            gt = torch.zeros(len(tokens)).long().to(args.device)
            gt[:-1] = tokens[1:]
            gt[-1] = 1084 #ugly hack
            _score = -log_softmax(score[j, :len(tokens)])
            _ppl = torch.exp(_score.gather(1, gt.view(-1,1)).sum() / len(gt))
            print (_ppl)
            ppl.update(_ppl.item(), 1)
        batch_time.update(time.time() - end)
        end = time.time()
        print_info  = '(%d/%d) Data: %.4f | \
                    Batch: %.4f | Perplexity %.4f' % (
                    loader.dataset.cnt,
                    len(loader),
                    data_time.val,
                    batch_time.val,
                    ppl.avg
                    )
        _logger.info(print_info)
        i += 1
    _logger.warning('Perplexity %.4f', ppl.avg)

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

if __name__ == "__main__":
    args = params.get_args()

    _logger = log.get_logger(__name__, args)
    _logger.info(print_args(args))

    args.valid_split = 'test' if args.evaluate else 'valid'
    proc, q1 = start_loader(args, 'gen')
    if args.partial_lm:
        lm = build_language_model(args.num_props, new=args.gen_lm, _all=False)
    else:
        lm = load_language_model(new=True)
    config = get_config(lm)
    end_output = config.encode['END_OF_SECTION']

    args.vocab_size = len(config.encode)+1
    models = {}
    models['emb'] = torch_models.Emb(args, config).to(args.device)
    in_dim = args.nFeats
    if not args.no_use_tree:
        in_dim += 4

    models['e_gru'] = torch.nn.GRU(in_dim, args.nFeats, args.gru_depth,
            dropout=args.dropout, bidirectional=args.bidirectional).to(args.device)
    models['decoder'] = torch_models.Decoder(args).to(args.device)

    opt = get_opt(models, args)
    criterion = torch.nn.CrossEntropyLoss().to(args.device)

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

        loss, acc = train(models, q1, opt, criterion, args, _logger, e)
        _logger.warning("Validation begins.")
        if e in args.schedule:
            adjust_lr(opt, args)
            _logger.warning('Learning rate decreases')

    close_loader(proc, q1)

