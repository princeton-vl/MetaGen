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
from data_utils import *
import interface_lm
from constructor import Constructor
import trainer
import data_loader_gen
import data_loader_pred
import data_loader_gen_noproof

def train_baseline_model():
    args = params.get_args()
    if args.task == 'pred':
        data_loader = data_loader_pred
        data_loader_eval = data_loader 
    elif args.task == 'gen':
        data_loader_eval = data_loader_gen
        if not args.gen_noproof:
            data_loader = data_loader_gen
        else:
            data_loader = data_loader_gen_noproof
    else:
        print ('unknown task: %s' % (args.task))
        return

    _logger = log.get_logger(__name__, args)
    _logger.info(print_args(args))

    valid_split = 'valid' if args.evaluate =='none' else args.evaluate
    train_split = 'valid' if args.goal_debug else 'train'
    if args.evaluate == 'none':
        trainloader = torch.utils.data.DataLoader(
            data_loader.MetaMath(args, split=train_split), collate_fn=data_loader.mycollate,
            batch_size=args.batch_size, shuffle=True,
            num_workers=0)
    validloader = torch.utils.data.DataLoader(
        data_loader_eval.MetaMath(args, split=valid_split, evaluate=True),
        collate_fn=data_loader_eval.mycollate,
        batch_size=args.batch_size, shuffle=False,
        num_workers=0)

    if args.partial_lm:
        lm = build_language_model(args.num_props, new=args.gen_lm, _all=False)
    else:
        lm = load_language_model(_all=False, new=args.gen_lm, iset=args.iset)

    config = get_config(lm)
    args.vocab_size = len(config.encode)+1
    generator = None

    if args.task == 'pred':
        model = torch_models.PredModel(args, config).to(args.device)
    elif args.task == 'gen':
        model = torch_models.GenModel2(args, config).to(args.device)

    training_object = trainer.Trainer(args, config, model, generator, _logger, validloader)
    if args.pretrained != '':
        model.load(args.pretrained)
        data = torch.load(args.pretrained)
        if not hasattr(args, 'start_zero') or not args.start_zero:
            training_object.cur_epoch = data['aux']['epoch']+1

    if args.evaluate != 'none':
        with torch.no_grad():
            training_object.evaluate()
        return

    while training_object.cur_epoch < args.epoch:
        if training_object.cur_epoch in args.schedule:
            adjust_lr(training_object.model_opt, args)
        training_object.train_from_loader(trainloader)
        training_object.evaluate()
        aux = {'epoch':training_object.cur_epoch, 'cur_iter':9999}
        save_model(aux, args, training_object.model, opt=None)
        training_object.cur_epoch += 1

if __name__ == "__main__":
    train_baseline_model()
