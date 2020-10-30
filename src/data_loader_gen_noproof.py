import os,sys
import numpy as np
import math
import time
import random

import torch
import torch.utils.data as data
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import torch_models
import params
import log
import data_loader_gen2 as data_loader
from data_utils import *

def mycollate(batch):
    data = [[],[],[],[],[],[]]
    for i in range(6):
        for b in batch:
            data[i].append(b[i])
    return data

class MetaMath(data.Dataset):
    def __init__(self, args, split=None, evaluate=None):
        self.args = args
        self.evaluate = evaluate if evaluate is not None else args.evaluate
        self.split = split if split is not None else args.split
        self.lm = load_language_model(_all=False)
        self.config = get_config(self.lm)
        self.props = self.lm.training_propositions
        self.plist = []
        self.idxes = []
        for i,p in enumerate(self.props):
            self.plist.append([])
            self.tolist(p.tree, self.plist[-1], p.f)
            if len(self.plist[-1]) > 0:
                self.idxes.append(i)

        self.cnt = 0
        self.cur_epoch = 0
        self.length = len(self.idxes)
        random.shuffle(self.idxes)
    
    def tolist(self, tree, l, f):
        if tree.value not in f:
            l.append(tree)
        for c in tree.leaves:
            self.tolist(c, l, f)

    def extract_from_graph(self, graph):
        string = graph.string
        structure_data = list(zip(graph.depth, graph.parent_arity, graph.leaf_position, graph.arity))
        tokens = [self.config.encode[t] for t in string]
        return tokens, structure_data

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        idx = self.idxes[self.cnt]
        prop = self.props[idx]
        self.cnt += 1

        random_replacement_dict = self.lm.random_replacement_dict_f(f=prop.f)
        tree = prop.tree.copy()
        plist = []
        self.tolist(tree, plist, prop.f)
        subtree = random.choice(plist)
        outtree = subtree.copy()
        subtree.value = 'TARGET_UC'
        subtree.leaves = []

        known_trees = [hyp.tree.copy().replace_values(random_replacement_dict)
                    for hyp in prop.hyps if hyp.type == 'e']
        to_prove_trees = [tree.replace_values(random_replacement_dict)]
        out_trees = [outtree.replace_values(random_replacement_dict)]

        known_graph_structure = TreeInformation(known_trees,
                start_symbol=None, intermediate_symbol='END_OF_HYP',
                end_symbol='END_OF_SECTION')
        to_prove_graph_structure = TreeInformation(to_prove_trees,
                start_symbol=None, intermediate_symbol='END_OF_HYP',
                end_symbol='END_OF_SECTION')
        out_graph_structure = TreeInformation(out_trees, start_symbol='START_OUTPUT',
                intermediate_symbol=None, end_symbol=None)

        known_tokens, known_trees = self.extract_from_graph(known_graph_structure)
        to_prove_tokens, to_prove_trees = self.extract_from_graph(to_prove_graph_structure)
        out_tokens, out_trees = self.extract_from_graph(out_graph_structure) 
                
        if self.cnt == len(self.idxes):
            self.cnt = 0
            self.cur_epoch += 1
            random.shuffle(self.idxes)

        return known_tokens, known_trees, to_prove_tokens, to_prove_trees, out_tokens, out_trees

