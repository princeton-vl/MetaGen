'''
This builds up the interface for the lm module.
'''

import gen_model_beam_search
import gen_model_beam_search_torch
import gen_model_beam_search_rl
import pred_model as pred_model_run
import payout_model_5_train as payout_model_run
from models import *
from beam_search import *
import os
import sys
import numpy as np
import pickle as pickle
import data_utils5 as data_utils
import nnlibrary as nn
import data_utils as data_utils_new
import constructor_list

import torch
import torch_models

NUM_ALLOWED_CONSTRUCTORS = None
DISALLOWED_PROPS = ['idi', 'dummylink', 'dtrucor']
if NUM_ALLOWED_CONSTRUCTORS is None:
    ALLOWED_CONSTRUCTORS = None
else:
    ALLOWED_CONSTRUCTORS = set(constructor_list.order[:NUM_ALLOWED_CONSTRUCTORS])


class LMInterface:
    def __init__(self, args, lm, recalculate_props=False):#, directory='searcher'):
        self.lm = lm
        self.args = args
        #self.holo_directory = 'searcher'
        self.config = data_utils_new.get_config(lm)
        self.args.vocab_size = len(self.config.encode)+1
        self.pred_logs = []
        self.gen_logs = []
        self.pred_rewards = []
        self.gen_rewards = []
        self.expr_pos = []
        if args.interface_model:
            data = torch.load(self.args.interface_model)
            print ('load from %s' % (self.args.interface_model))
            args_model = data['args']
            args_model.device = args.device
            device = torch.device("cuda:0")
            self.pred_model = torch_models.PredModel(args_model, self.config).to(device)
            self.gen_model = torch_models.GenModel(args_model, self.config).to(device)
            self.pred_model.load_state_dict(data['models']['pred'])
            self.gen_model.load_state_dict(data['models']['gen'])
            self.pred_model = self.pred_model.to(args.device)
            self.gen_model = self.gen_model.to(args.device)
        else:
            if self.args.interface_pred_model != '':
                args_pred = torch.load(self.args.interface_pred_model)['args']
                args_pred.device = args.device
                args_pred.cat = False
                self.pred_model = torch_models.PredModel(args_pred, self.config).to(args.device)
                self.pred_model.load(self.args.interface_pred_model)
            else:
                self.pred_model = torch_models.PredModel(args, self.config).to(args.device)
            if self.args.interface_gen_model != '':
                args_gen = torch.load(self.args.interface_gen_model)['args']
                args_gen.device = args.device
                self.gen_model = torch_models.GenModel(args_gen, self.config).to(args.device)
                self.gen_model.load(self.args.interface_gen_model)
            else:
                self.gen_model = torch_models.GenModel(args, self.config).to(args.device)
        self.bsi = gen_model_beam_search_rl.BeamSearchInterface(self.args, self.gen_model, self.gen_logs, self.gen_rewards)
        #TODO bsi database

    def get_pred(self, batch):
        g_tokens = [torch.Tensor(d).long().to(self.args.device) for d in batch[0]]
        g_trees = [torch.Tensor(d).to(self.args.device) for d in batch[1]]
        p_tokens = [torch.Tensor(d).long().to(self.args.device) for d in batch[2]]
        p_trees = [torch.Tensor(d).to(self.args.device) for d in batch[3]]
        if self.args.train_rl:
            g_vec = self.pred_model.embed(g_tokens, g_trees, _type='g')
            p_vec = self.pred_model.embed(p_tokens, p_trees, _type='p')
            score = self.pred_model.biln(g_vec, p_vec).view(-1)
            score = torch.nn.functional.softmax(score, dim=0)
        else:
            with torch.no_grad():
                g_vec = self.pred_model.embed(g_tokens, g_trees, _type='g')
                p_vec = self.pred_model.embed(p_tokens, p_trees, _type='p')
                score = self.pred_model.biln(g_vec, p_vec).view(-1)
                score = torch.nn.functional.softmax(score, dim=0)
        return score # score after softmax

    def apply_prop(self, tree, context, prop_name, n=10, return_replacement_dict=False, step=None):
        # shortcut if the unconstrainer arity is 0
        prop = self.lm.database.propositions[prop_name]
        #if prop.unconstrained_arity() == 0:
        #    return [(0.0, self.lm.simple_apply_prop(tree, prop, context, vclass='|-'))]

        ''' in this case, params = tree, context, prop_name '''
        beam_searcher = BeamSearcher(self.bsi, (tree, context, prop_name, ALLOWED_CONSTRUCTORS, return_replacement_dict, step))
        out = beam_searcher.sample()#best(n, n, n) #(width, k, num_out)  See notes regarding accuracy
        #print 'out', out
        return out

