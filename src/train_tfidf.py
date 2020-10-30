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
import data_loader_pred as data_loader

args = params.get_args()
args.task = 'pred'
args.batch_size = 1
args.validall = True

_logger = log.get_logger(__name__, args)
_logger.info(print_args(args))

lm, lm_inputs = load_language_model(_all=True, new=args.gen_lm)
config = get_config(lm)
args.vocab_size = len(config.encode)+1

valid_split = 'valid' if args.evaluate=='none' else args.evaluate
validloader = torch.utils.data.DataLoader(
        data_loader.MetaMath(args, split=valid_split, evaluate=args.validall),
        collate_fn=data_loader.mycollate,
        batch_size=args.batch_size, shuffle=False,
        num_workers=0)

if args.pretrained:
    tfidf = torch.load(args.pretrained)
    tfidf.args.device = args.device
else:
    tfidf = torch_models.TFIDF(args, config)
    for p in lm.training_propositions:
        tfidf.add_doc(lm_inputs[p.number][0])
    tfidf.args.device = None
    torch.save(tfidf, args.output)
    tfidf.args.device = torch.device("cuda:0" if (not args.cpu) and torch.cuda.is_available() else "cpu")

#training_object = trainer.Trainer(args, config, tfidf, None, _logger, validloader)

prop_embs = torch.zeros(len(lm_inputs), args.vocab_size).to(args.device)
for i in range(len(lm_inputs)):
    prop_embs[i] = tfidf.embed(lm_inputs[i][0])

batch_time = AverageMeter()
data_time = AverageMeter()
acces_1 = AverageMeter()
acces_5 = AverageMeter()
acces_20 = AverageMeter()
rank = AverageMeter()
ppls = AverageMeter()
num_props = AverageMeter()

end = time.time()
for i, batch in enumerate(validloader):
    data_time.update(time.time() - end)
    g_emb = tfidf.embed(batch[0][0], batch[1], _type='g')
    p_idx = batch[-2][0]
    true_pos = batch[-1][0]
    #for g, p_idx, true_pos in zip(g_emb, prop_idx, true_prop):
    if True:
        p_emb = prop_embs[torch.Tensor(p_idx).to(args.device).long()]
        score = tfidf.biln(g_emb.view(1,-1), p_emb)
        ppl = np.exp((score[true_pos]-score.exp().sum().log()).item())
        _pos = (score[true_pos]<score).sum().item()
        ppls.update(ppl, 1)
        rank.update(1/(1+_pos), 1)
        num_props.update(len(p_idx), 1)
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
                        top20: %.4f | rank: %.4f | ppl: %.4f' % (
                        i + 1,
                        len(validloader),
                        data_time.val,
                        batch_time.val,
                        acces_1.avg*100,
                        acces_5.avg*100,
                        acces_20.avg*100,
                        rank.avg,
                        ppls.avg )
    _logger.info(print_info)
