import os
import numpy as np
import json
import random
import math
import argparse
import params
import torch
import time
import torch.utils.data as data

from data_utils import *

def mycollate(batch):
    data = []
    for i in range(len(batch[0])):
        data.append([])
        for b in batch:
            data[-1] += b[i]
    return data

class MetaMath(data.Dataset):
    def __init__(self, args, config, model=None, split='train', evaluate=False):
        self.args = args
        self.split = split
        self.evaluate = evaluate if evaluate is not None else args.evaluate
        self.lm = config.lm
        self.config = config
        self.model = model
        self.steps = []
        if self.split == 'train':
            self.propositions = self.lm.training_propositions
        elif self.split == 'valid':
            self.propositions = self.lm.validation_propositions
        else:
            self.propositions = self.lm.test_propositions
        self.all_f = {}
        if self.args.data_sampling > 0:
            num = int(10*self.args.data_sampling)
        for i,p in enumerate(self.propositions):
            for l,f in p.f.items():
                if l not in self.all_f:
                    self.all_f[l] = f
            if self.evaluate or (self.args.data_sampling > 0 and self.split == 'train' and i % 10 < num):
                # training proof tasks with proofs
                self.steps += [step for step in p.entails_proof_steps
                                        if not (step.prop.type=='f' or step.prop.type == 'e')]
            else:
                # proof tasks without proofs
                self.steps.append(p)
        self.replacement_dict = self.lm.deterministic_replacement_dict_f(f=self.all_f)
        self.idxes = list(range(len(self.steps)))
        self.cnt = 0
        self.length = len(self.idxes)
        self.cur_epoch = 0
        if not self.args.evaluate:
            random.shuffle(self.idxes)
        print ('load %s steps for LM' % (self.length))

    def load_steps(self):
        self.cnt = 0
        self.cur_epoch += 1
        if not self.args.evaluate:
            random.shuffle(self.idxes)
    
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.model is None:
            return self.__getitem_old__(idx)
        else:
            return self.__getitem_new__(idx)

    def __getitem_new__(self, idx):
        if self.cnt == self.length:
            self.load_steps()
        step = self.steps[self.idxes[self.cnt]]
        self.cnt += 1
        return self.model.encode_expr([step]) 

    def __getitem_old__(self, idx):
        if self.cnt == self.length:
            self.load_steps()
        step = self.steps[self.idxes[self.cnt]]
        self.cnt += 1
        statement = step.tree.copy().replace_values(self.replacement_dict)
        if type(step) == proof_step:
            prop = step.context
        else:
            prop = step
        hyps = [e.tree.copy().replace_values(self.replacement_dict) for e in prop.hyps if e.type == 'e']
        statement_graph_structure = TreeInformation([statement],
                start_symbol=None, intermediate_symbol='END_OF_HYP',
                end_symbol='END_OF_SECTION')
        hyps_graph_structure = TreeInformation(hyps,
                start_symbol=None, intermediate_symbol='END_OF_HYP',
                end_symbol='END_OF_SECTION')
        in_string, structure_data = merge_graph_structures_new([statement_graph_structure, hyps_graph_structure])
        tokens = torch.Tensor([self.config.encode[t] for t in in_string]).to(self.args.device).long()
        return tokens, torch.Tensor(structure_data).to(self.args.device)
