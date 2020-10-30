import os
import numpy as np
import json
import random
import math
import log
import argparse
import params
import torch
import time
import torch_models
import pickle
from constructor import Constructor
import torch.multiprocessing as mp
import interface_lm
import gc

from data_utils import *

def DataLoader(args, outqueue, inqueue, batch_size=None):

    if batch_size == None:
        batch_size = args.batch_size
    if args.partial_lm:
        lm = build_language_model(args.num_props, new=args.gen_lm, _all=False)
    else:
        lm = load_language_model(_all=False, new=args.gen_lm)
        if args.data_sampling > 0:
            load_proof_steps_into_lm(lm, ['train'], args.data_sampling)
    config = get_config(lm)
    args.device = torch.device('cuda:%d' % (args.gpu_list[0]))
    torch.cuda.set_device(args.gpu_list[0])
    generator = Constructor(args, config)
    generator.initialize()
    old_data = [[],[],[],[],[],[]] if args.task == 'gen' else [[],[],[],[]]
    _logger = log.get_logger('dataloader', args, append=True)
    _logger.info('dataloader is ready')
    while True:
        batch = [[],[],[],[],[],[]] if args.task == 'gen' else [[],[],[],[]]
        while len(batch[0]) < batch_size:
            if len(old_data[0]) < args.num_old_exprs or random.random() < args.new_expr_ratio:
                # use new exprs
                data = inqueue.get()
                for i in range(len(data)):
                    batch[i] += data[i]
                    if args.task == 'gen':
                        old_data[i] += data[i]
                    else:
                        old_data[i].append(data[i])
                if len(old_data[0]) > args.num_old_exprs:
                    for d in old_data:
                        d.pop(0)
            else:
                # use old exprs
                k = random.randrange(len(old_data[0]))
                for i in range(len(batch)):
                    if args.task == 'gen':
                        batch[i].append( old_data[i][k] )
                    else:
                        batch[i] += old_data[i][k]
            #_logger.info('%d %d %d', outqueue.qsize(), inqueue.qsize(), len(batch[0]))
            #print (len(batch))
        #if args.task == 'pred':
        #    data = generator.encode_pred_tasks(exprs)
        #else:
        #    data = generator.encode_gen_tasks(exprs)
        outqueue.put(batch)

def worker(args, queue, worker_id):
    #lm = config.lm
    if args.partial_lm:
        lm = build_language_model(args.num_props, new=args.gen_lm, _all=False)
    else:
        lm = load_language_model(_all=False, new=args.gen_lm)
        if args.data_sampling > 0:
            load_proof_steps_into_lm(lm, ['train'], args.data_sampling)
    #lm = load_language_model(_all=False, new=args.gen_lm)
    #load_proof_steps_into_lm(lm, ['train'], args.data_sampling)
    config = get_config(lm)
    device_idx = args.num_gpus - 1 - (worker_id // args.num_workers)
    if args.random_generate:
        device_idx = 0
    device_id = args.gpu_list[device_idx]
    print ('build a worker on gpu %d' % (device_id))
    torch.cuda.set_device(device_id)
    args.device = torch.device('cuda:'+str(device_id))
    interface = None if args.random_generate else interface_lm.LMInterface(args, lm)
    generator = Constructor(args, config, interface)
    generator.initialize()
    _logger = log.get_logger('worker%d'%(worker_id), args, append=True)
    _logger.info('worker %d initialize', worker_id)
    tt = 0
    cnt = 0
    while True:
        t = time.time()
        if args.random_generate:
            expr = generator.random_generate()
        else:
            expr = generator.parameterized_generate()
        tt += time.time() - t
        if expr is not None:
            if args.task == 'pred' and len(expr.prop.e) > 0:
                queue.put(generator.encode_pred_tasks([expr]))
            if args.task == 'gen' and len(expr.unconstrained) > 0:
                queue.put(generator.encode_gen_tasks([expr]))
            #if not (args.task == 'gen' and len(expr.unconstrained) == 0):
            #    if args.task == 'pred':
            #        queue.put(generator.encode_pred_tasks([expr]))
            #    else:
            #        queue.put(generator.encode_gen_tasks([expr]))
            if len(generator.expressions_list) > args.num_cons_exprs+generator.num_initial_expr:
                generator.reinitialize_expressions()
                _logger.info('worker %d initialize', worker_id)
            if cnt == 5000:
                _logger.info('worker %d generate time per expr %s seconds', worker_id, tt/cnt)
                cnt = 0
                tt = 0
            cnt += 1

def worker_pre(args, queue, batch_size, ii):
    _logger = log.get_logger('worker_pre', args, append=True)
    _logger.info('worker_pre initialize')
    if args.partial_lm:
        lm = build_language_model(args.num_props, new=args.gen_lm, _all=False)
    else:
        lm = load_language_model(_all=False, new=args.gen_lm, iset=args.iset)
    config = get_config(lm)
    #exprs = load_exprs(args.expr_list[0], lm)
    #interface = interface_lm.LMInterface(args, lm)
    #generator = Constructor(args, config)
    #generator.initialize_prop()
    #generator.expressions_list = exprs
    #for e in exprs:
    #    generator.expressions[e.id] = e
    #generator.num_initial_expr = len(generator.expressions)
    #generator.initialize_searcher()
    #_logger.info('initialize generator with %d exprs', generator.num_initial_expr)

    fl = os.listdir(args.exprs_pre)
    if True:
        if args.data_sampling > 0:
            exprs = load_exprs(args.expr_list[0], lm)
            generator = Constructor(args, config)
            generator.initialize_prop()
            generator.expressions_list = exprs
            for e in exprs:
                generator.expressions[e.id] = e
            generator.num_initial_expr = len(generator.expressions)
            generator.initialize_searcher()
        else:
            generator = Constructor(args, config)
            generator.initialize()
        #generator.reinitialize_expressions()
        print ('--loading pre exprs--')
        exprs_pre = load_exprs(os.path.join(args.exprs_pre, fl[ii]), lm)
        print ('--done--')
        #if args.train_from_queue:
        #    exprs_pre = exprs_pre[:300000]
        generator.expressions_list += exprs_pre
        for e in exprs_pre:
            generator.expressions[e.id] = e
        _logger.info('load %d exprs' % (len(generator.expressions)))
        i = 0
        while True:
        #for i in range(len(exprs_pre)//batch_size//5):
            _exprs = []
            while len(_exprs) < batch_size:
                expr = random.choice(exprs_pre)
                if args.task == 'gen' and len(expr.unconstrained) == 0:
                    continue
                if args.task == 'pred' and len(expr.prop.e) == 0:
                    continue
                _exprs.append(expr)
            if args.task == 'pred':
                data = generator.encode_pred_tasks(_exprs)
            else:
                data = generator.encode_gen_tasks(_exprs)
            #print (i, data)
            queue.put(data)
            i += 1
            if i >= len(exprs_pre)//batch_size//5 and (not args.train_from_queue) and (not args.cons_pre_one):
                break
        print ('finish processing current exprs')

def build_worker(args, queue, batch_size):
    fl = os.listdir(args.exprs_pre)
    if args.train_from_queue:
        fl = fl[:1]
    while True:
        for i in range(len(fl)):
            worker_pre(args, queue, batch_size, i)

def f(i):
    print (torch.rand(10).to(torch.device('cuda:'+str(i))))

if __name__ == "__main__":
    mp.set_start_method('spawn')
    q1 = mp.Queue(1000)
    q2 = mp.Queue(1000)
    args = params.get_args()
    #args.partial_lm = True
    #args.num_props = 4248
    args.gen_lm = True
    args.task = 'gen'
    #args.num_cons_exprs = 50000
    #args.num_workers = 5
    #args.num_gpus = 4
    #args.num_old_exprs = 10000
    #args.new_expr_ratio = 0.1
    args.expr_list = ['../data/expressions_list_1.0']
    args.exprs_pre = '../exprs/all_1_pre'
    #lm = build_language_model(args.num_props, new=args.gen_lm, _all=False)
    lm = load_language_model(_all=False, new=args.gen_lm)
    #load_proof_steps_into_lm(lm, ['train'], args.data_sampling)
    config = get_config(lm)
    plist = []
    args.device = None
    _logger = log.get_logger(__name__, args)
    #total_workers = args.num_workers * args.num_gpus - 2
    #for i in range(total_workers):
    #    plist.append(mp.Process(target=worker, args=(args, q2, i)))
    #    plist[-1].start()
    #plist.append(mp.Process(target=DataLoader, args=(args, q1, q2)))
    plist.append(mp.Process(target=worker_pre, args=(args, q1, 50)))
    plist[-1].start()
    print ([p.pid for p in plist])
    t = time.time()
    args.device = torch.device('cuda:0')
    data = q1.get()
    args.vocab_size = 117
    model = torch_models.GenModel2(args, config).to(args.device)
    for i in range(10000):
        os.system('nvidia-smi')
        print (q1.qsize(), q2.qsize())
        t = time.time()
        data = q1.get()
        with torch.no_grad():
            score = model(data)
        print (score.max())
        print ((time.time()-t))

    for p in plist:
        p.terminate()
    for p in plist:
        p.join()
