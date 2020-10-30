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
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from data_utils import *

global global_file_list
global global_cur_file
global global_cur_epoch
global_cur_file = 0
global_cur_epoch = 0
global_file_list = []

def worker_init(dataset, idx):
    print ('hello world')
    #dataset.file_list = [dataset.file_list[idx]]
    l = len(dataset.file_list)
    per = l // dataset.args.num_workers + 1
    dataset.file_list = dataset.file_list[per*idx:min(l, per*(idx+1))]
    print (dataset.file_list)
'''
def mycollate(batch):
    goal_tokens = [b[0] for b in batch]
    goal_trees = [b[1] for b in batch]
    if len(batch[0]) == 6:
        prop_idx = [b[-2] for b in batch]
        true_idx = [b[-1] for b in batch]
        return goal_tokens, goal_trees, prop_idx, true_idx
    prop_tokens = []
    prop_trees = []
    for b in batch:
        prop_tokens += b[2]
        prop_trees += b[3]
    return goal_tokens, goal_trees, prop_tokens, prop_trees
    # [tensor of strings] in b
    # [s*4] in b
    # [str] in b* (1+neg)
    # [s*4] in b* (1+neg)
'''
def mycollate(batch):
    data = [[],[],[],[],[],[]]
    for i in range(6):
        for b in batch:
            data[i] += b[i]
    return data
    #in_tokens = [b[0] for b in batch]
    #in_trees = [b[1] for b in batch]
    #out_tokens = [b[2] for b in batch]
    #out_trees = [b[3] for b in batch]
    #return in_tokens, in_trees, out_tokens, out_trees

class MetaMath(data.Dataset):
    def __init__(self, args, split=None, evaluate=False):
        self.args = args
        self.evaluate = evaluate
        self.split = split if split is not None else args.split
        if self.args.partial_lm:
            self.lm = build_language_model(self.args.num_props, new=self.args.gen_lm, _all=False)
            self.config = get_config(self.lm)
            if self.split == 'train':
                self.propositions = self.lm.training_propositions
            elif self.split == 'valid':
                self.propositions = self.lm.validation_propositions
            else:
                self.propositions = self.lm.test_propositions
            self.steps = []
            if self.args.data_sampling > 0:
                num = int(1/self.args.data_sampling)
            for i,p in enumerate(self.propositions):
                if self.split != 'train' or (self.args.data_sampling > 0 and i % num == 0):
                    self.steps += [step for step in p.entails_proof_steps
                        if not (step.prop.type=='f' or step.prop.type == 'e')]
            print (len(self.steps))
            self.steps = [step for step in self.steps if len(step.unconstrained) > 0]
            print (len(self.steps))
            self.idxes = []
            for i,step in enumerate(self.steps):
                if self.evaluate:
                    self.idxes.append(i)
                else:
                    for j in range(len(step.unconstrained)):
                        self.idxes.append((i, j))
            print ('load %d gen vars in %d steps' % (len(self.idxes), len(self.steps)))
            if not self.evaluate:
                random.shuffle(self.idxes)
            self.length = len(self.idxes)
            self.file_list = []
            self.cur_epoch = 0
            self.cnt = 0
            return
        self.lm = load_language_model(_all=False, iset=args.iset)
        self.config = get_config(self.lm)
        self.split = split if split is not None else args.split
        global_file_list = [s for s in os.listdir(os.path.join(args.data_path, self.split))
                        if s.isalnum()]
        self.file_list = global_file_list
        self.cnt = 0
        self.cur_file = global_cur_file
        self.cur_epoch = global_cur_epoch
        self.cur_target_idx = 0
        self.steps = None
        if self.split == 'train':
            self.propositions = self.lm.training_propositions
        elif self.split == 'valid':
            self.propositions = self.lm.validation_propositions
        else:
            self.propositions = self.lm.test_propositions
        self.props = []
        num = int(self.args.data_sampling * 10)
        #num = int(1/self.args.data_sampling)
        for i,p in enumerate(self.propositions):
            if self.split != 'train' or (self.args.data_sampling > 0 and i % 10 < num):
                self.props.append(p.label)
        print ('%d props after filtering' % (len(self.props)))
        #self.inputs = None
        #self.props = None
        #self.length = len(self.file_list) * 50000
        if not self.evaluate:
            random.shuffle(self.file_list)
        #self.file_list = self.file_list[:1]
        self.load_steps_all()
        self.length = len(self.idxes)

    def load_steps_all(self):
        if self.args.partial_lm:
            if not self.evaluate:
                random.shuffle(self.idxes)
            return
        self.steps = []
        print (self.file_list)
        for fn in self.file_list:
            print ('load from %s' % (fn))
            _steps = load_gen_proof_steps(
                    os.path.join(self.args.data_path, self.split, fn),
                    self.lm, _all=False)
            for i in range(len(_steps)):
                #print (_steps[i].context_label in self.props, len(_steps[i].unconstrained))
                if _steps[i].context_label in self.props and len(_steps[i].unconstrained) > 0:
                    self.steps.append(_steps[i])
            print ('load %d steps' % (len(self.steps)))
        self.idxes = []
        #self.idxes = list(range(len(self.steps)))
        for i,step in enumerate(self.steps):
            if self.evaluate:
                self.idxes.append(i)
            else:
                for j in range(len(step.unconstrained)):
                    self.idxes.append((i, j))
        print ('load %d gen vars in %d steps' % (len(self.idxes), len(self.steps)))
        if not self.evaluate:
            random.shuffle(self.idxes)

    def load_steps(self, fn=None):
        if self.args.partial_lm:
            if self.evaluate == False:
                random.shuffle(self.idxes)
            self.cnt = 0
            self.cur_epoch += 1
            return
        if fn is None:
            if self.cur_file == len(self.file_list):
                self.cur_file = 0
                self.cur_epoch += 1
            fn = self.file_list[self.cur_file]
            self.cur_file += 1
        print (self.cur_file)
        self.cnt = 0
        print ('load from', fn)
        self.steps = load_gen_proof_steps(
                os.path.join(self.args.data_path, self.split, fn),
                self.lm, _all=False)
        self.steps = [step for step in self.steps if len(step.unconstrained) > 0]
        self.steps = [step for step in self.steps if step.context_label in self.props]
        self.idxes = []
        for i,step in enumerate(self.steps):
            if self.evaluate:
                self.idxes.append(i)
            else:
                for j in range(len(step.unconstrained)):
                    self.idxes.append((i, j))
        print ('load %d gen vars in %d steps' % (len(self.idxes), len(self.steps)))
        if not self.evaluate:
            random.shuffle(self.idxes)

    def extract_from_graph(self, graph):
        string = graph.string
        structure_data = list(zip(graph.depth, graph.parent_arity, graph.leaf_position, graph.arity))
        #tokens = [self.config.encode[t] for t in string]
        return string, structure_data

    def to_tensor(self, string, trees):
        tokens = [self.config.encode[t] for t in string]
        return torch.Tensor(tokens).long().to(self.args.device), torch.Tensor(trees).to(self.args.device)

    def encode_step(self, idx, target_idx):
        step = self.steps[idx]
        tree = step.tree
        prop = step.prop
        context = step.context

        unconstrained_variables = prop.unconstrained_variables
        uv_dict = {var:'UC'+var for var in unconstrained_variables}
        fit = prop_applies_to_statement(tree, prop, context)
        random_replacement_dict = self.lm.random_replacement_dict_f(f=context.f)

        to_prove_trees = [hyp.tree.copy().replace(fit).replace_values(uv_dict).replace_values(random_replacement_dict)
               for hyp in prop.hyps if hyp.type == 'e']
        known_trees = [hyp.tree.copy().replace_values(random_replacement_dict)
                for hyp in step.context.hyps if hyp.type == 'e']
        if target_idx != -1:
            out_trees = [step.unconstrained[target_idx].copy().replace_values(random_replacement_dict)]
        else:
            out_trees = [tree.copy().replace_values(random_replacement_dict) for tree in step.unconstrained]

        known_graph_structure = TreeInformation(known_trees,
                start_symbol=None, intermediate_symbol='END_OF_HYP',
                end_symbol='END_OF_SECTION')
        to_prove_graph_structure = TreeInformation(to_prove_trees,
                start_symbol=None, intermediate_symbol='END_OF_HYP',
                end_symbol='END_OF_SECTION')
        out_graph_structure = [TreeInformation([tree], start_symbol='START_OUTPUT',
                intermediate_symbol=None, end_symbol=None) for tree in out_trees]

        known_tokens, known_trees = self.extract_from_graph(known_graph_structure)
        to_prove_tokens, to_prove_trees = self.extract_from_graph(to_prove_graph_structure)

        uv_pos = {}
        for i in range(len(unconstrained_variables)):
            uv_pos[i] = []
            for j in range(len(to_prove_tokens)):
                if to_prove_tokens[j] == uv_dict[unconstrained_variables[i]]:
                    uv_pos[i].append(j)
                    to_prove_tokens[j] = 'UC'

        known_tokens, known_trees = self.to_tensor(known_tokens, known_trees)
        to_prove_tokens, to_prove_trees = self.to_tensor(to_prove_tokens, to_prove_trees)

        out_tokens = []
        out_trees = []
        for graph in out_graph_structure:
            tokens, trees = self.extract_from_graph(graph)
            tokens, trees = self.to_tensor(tokens, trees)
            out_tokens.append(tokens)
            out_trees.append(trees)
        if target_idx != -1:
            known_tokens = [known_tokens]
            known_trees = [known_trees]
            for i in uv_pos[target_idx]:
                to_prove_tokens[i] = self.config.encode['TARGET_UC']
            to_prove_tokens = [to_prove_tokens]
            to_prove_trees = [to_prove_trees]
        else:
            l = len(unconstrained_variables)
            known_tokens = [known_tokens]*l
            known_trees = [known_trees]*l
            _to_prove_tokens = []
            for i in range(l):
                tokens = to_prove_tokens.clone()
                for j in uv_pos[i]:
                    tokens[i] = self.config.encode['TARGET_UC']
                _to_prove_tokens.append(tokens.clone())
            to_prove_tokens = _to_prove_tokens
            to_prove_trees = [to_prove_trees]*l
        return known_tokens, known_trees, to_prove_tokens, to_prove_trees, out_tokens, out_trees

    def check_end(self):
        if self.cnt == len(self.idxes):
            self.cnt = 0
            self.cur_epoch += 1

    def __getitem__(self, idx):
        #if self.steps is None or self.cnt == len(self.idxes):
        #    if self.steps is None and not self.args.evaluate:
        #        random.shuffle(self.file_list)
        #    self.load_steps()
        if self.evaluate:
            target_idx = -1
            idx = self.idxes[self.cnt]
        else:
            idx, target_idx = self.idxes[self.cnt]
        self.cnt += 1
        sample = self.encode_step(idx, target_idx)
        self.check_end()
        return sample

    def __len__(self):
        return self.length

if __name__ == "__main__":
    #parser = argparse.ArgumentParser(description='loader')
    #parser.add_argument('--data-path', type=str, default='../data/')
    #parser.add_argument('--split', type=str, default='train')
    #parser.add_argument('--negative-samples', type=int, default=4)
    #parser.add_argument('--num_workers')
    #args = parser.parse_args()
    args = params.get_args()
    args.device = torch.device('cpu')
    loader = torch.utils.data.DataLoader(
            MetaMath(args, split='valid'), collate_fn=mycollate,
            #worker_init_fn=worker_init,
            batch_size=100, shuffle=False, num_workers=2)#args.num_workers)
    t = time.time()
    i= loader.dataset.cur_epoch
    j = 0
    for _,batch in enumerate(loader):
        print (time.time(), j)
        j += 1
        if i!= loader.dataset.cur_epoch:
            break
    #print (data)
