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
from data_utils import *
import interface_lm
from constructor import Constructor
import trainer
import data_loader_cons

if __name__ == "__main__":

    mp.set_start_method('spawn')
    args = params.get_args()
    
    if args.gpu_list[0] == -1:
        args.gpu_list = list(range(args.num_gpus))
    else:
        args.num_gpus = len(args.gpu_list)
    
    if args.task == 'pred':
        import data_loader_pred as data_loader
    elif args.task == 'gen':
        import data_loader_gen as data_loader
    else:
        print ('unknown task: %s' % (args.task))
        exit(0)

    _logger = log.get_logger(__name__, args)
    _logger.info(print_args(args))
    if args.manual_out:
        with open(args.log+'_p', 'w') as f:
            f.write(print_args(args))

    q1 = mp.Queue(1500)
    q2 = mp.Queue(1500)

    lm = load_language_model(_all=False, new=args.gen_lm, iset=args.iset)
    config = get_config(lm)
    
    if args.notrain:
        args.device = torch.device("cuda:%d" % (args.gpu_list[0]) if (not args.cpu) and torch.cuda.is_available() else "cpu")
        args.vocab_size = len(config.encode)+1
        if args.task == 'pred':
            model = torch_models.PredModel(args, config).to(args.device)
        elif args.task == 'gen':
            model = torch_models.GenModel2(args, config).to(args.device)
        valid_split = 'valid' if args.evaluate=='none' else args.evaluate
        validloader = torch.utils.data.DataLoader(
                data_loader.MetaMath(args, split=valid_split, evaluate=True),
                collate_fn=data_loader.mycollate,
                batch_size=args.batch_size, shuffle=False,
                num_workers=0)
        training_object = trainer.Trainer(args, config, model, None, _logger, validloader)
        with torch.no_grad():
            training_object.evaluate()
        aux = {'epoch':0, 'cur_iter':999}
        save_model(aux, args, model, opt=None)
        exit(0)

    args.device = None
    if args.precalculate:
        plist = [mp.Process(target=data_loader_cons.build_worker, args=(args, q1, args.cons_batch))]
        plist[-1].start()
    elif args.evaluate=='none':
        plist = []
        total_workers = args.num_gpus * args.num_workers - 2
        if args.random_generate:
            total_workers = args.num_workers
        for i in range(total_workers):
            plist.append(mp.Process(target=data_loader_cons.worker, args=(args, q2, i)))
            plist[-1].start()
        plist.append(mp.Process(target=data_loader_cons.DataLoader, args=(args, q1, q2, args.cons_batch)))
        plist[-1].start()
    args.device = torch.device("cuda:%d" % (args.gpu_list[0]) if (not args.cpu) and torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(args.gpu_list[0])

    valid_split = 'valid' if args.evaluate == 'none' else args.evaluate
    if args.data_sampling > 0 and (not args.train_from_queue) and (args.evaluate=='none'):
        trainloader = torch.utils.data.DataLoader(
            data_loader.MetaMath(args), collate_fn=data_loader.mycollate,
            batch_size=args.batch_size, shuffle=True,
            num_workers=0)#args.num_workers)
    validloader = torch.utils.data.DataLoader(
            data_loader.MetaMath(args, split=valid_split, evaluate=True),
            collate_fn=data_loader.mycollate,
            batch_size=args.batch_size, shuffle=False,
            num_workers=0)

    args.vocab_size = len(config.encode)+1
    if args.task == 'pred':
        model = torch_models.PredModel(args, config).to(args.device)
    elif args.task == 'gen':
        model = torch_models.GenModel2(args, config).to(args.device)

    start_epoch = 0
    if args.pretrained != "":
        loaddata = torch.load(args.pretrained)
        if not args.restart:
            start_epoch = loaddata['aux']['epoch']+1
            args.cur_iter = loaddata['aux']['cur_iter']
        model.load_state_dict(loaddata['models'])
        _logger.warning('load data from %s\n', args.pretrained)

    training_object = trainer.Trainer(args, config, model, None, _logger, validloader)
    training_object.queue = q1
    training_object.loader = plist[-1] if args.evaluate=='none' else None
    training_object.cur_epoch = start_epoch

    if args.evaluate != 'none':
        with torch.no_grad():
            training_object.evaluate()
    else:
        for i in range(start_epoch, args.epoch):
            if i in args.schedule:
                adjust_lr(training_object.model_opt, args)
            if args.data_sampling > 0 and (not args.train_from_queue):
                training_object.train_from_loader_and_queue(trainloader)
            else:
                training_object.train_one_epoch()
            with torch.no_grad():
                training_object.evaluate()
            aux = {'epoch':i, 'cur_iter':999}
            save_model(aux, args, training_object.model, opt=None)
            training_object.cur_epoch += 1
            
    for p in plist:
        p.terminate()
    for p in plist:
        p.join()
    q1.close()
    q2.close()
