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

def worker_init(dataset, idx):
    print ('hello world')
    #dataset.file_list = [dataset.file_list[idx]]
    l = len(dataset.file_list)
    per = l // dataset.args.num_workers + 1
    dataset.file_list = dataset.file_list[per*idx:min(l, per*(idx+1))]
    print (dataset.file_list)

def mycollate(batch):
    goal_tokens = [b[0] for b in batch]
    goal_trees = [b[1] for b in batch]
    if len(batch[0]) == 6:
        prop_idx = [b[-2] for b in batch]
        true_idx = [b[-1] for b in batch]
        if batch[0][2] is not None:
            prop_tokens = []
            prop_trees = []
            for b in batch:
                prop_tokens += b[2]
                prop_trees += b[3]
            return goal_tokens, goal_trees, prop_tokens, prop_trees, prop_idx, true_idx
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

class MetaMath(data.Dataset):
    def __init__(self, args, split='train', evaluate=False):
        self.args = args
        self.evaluate = evaluate
        self.split = split if split is not None else args.split
        if self.args.partial_lm:
            self.lm = build_language_model(self.args.num_props, new=self.args.gen_lm, _all=False)
            self.config = get_config(self.lm)
            self.lm_inputs = []
            for p in self.lm.database.propositions_list:
                self.lm_inputs.append(encode_proof_step(p.tree, p.f, p.hyps, self.lm, self.config))
            self.cnt = 0
            if self.split == 'train':
                self.propositions = self.lm.training_propositions
                #self.steps = self.lm.training_proof_steps
            elif self.split == 'valid':
                self.propositions = self.lm.validation_propositions
                #self.steps = self.lm.validation_proof_steps
            else:
                self.propositions = self.lm.test_propositions
                #self.steps = self.lm.test_proof_steps
            self.steps = []
            if self.args.data_sampling > 0:
                #num = int(1/self.args.data_sampling)
                num = int(self.args.data_sampling * 10)
            for i,p in enumerate(self.propositions):
                if self.split != 'train' or (self.args.data_sampling > 0  and i % 10 < num):
                    self.steps += [step for step in p.entails_proof_steps
                        if not (step.prop.type=='f' or step.prop.type == 'e')]
            self.props = []
            self.inputs = []
            for step in self.steps:
                labels = self.lm.searcher.search(step.tree, step.context,
                        max_proposition=step.context.number, vclass='|-')
                self.props.append(labels)
                self.inputs.append(encode_proof_step(step.tree, step.context.f, step.context.hyps, self.lm, self.config)) 
            self.idxes = list(range(len(self.steps)))
            random.shuffle(self.idxes)
            #if self.args.data_sampling < 1 and self.split=='train':
            #    #random.shuffle(self.idxes)
            #    num = int(1/self.args.data_sampling)
            #    self.idxes = [i for i in self.idxes if i % num == 0]
            #    #self.idxes = self.idxes[:int(len(self.idxes)*self.args.data_sampling)]
            self.length = len(self.idxes)
            self.cur_epoch = 0
            print ('get %d steps in loader' % (len(self.idxes)))
            return 
        self.lm, self.lm_inputs = load_language_model(_all=True, new=args.gen_lm, iset=args.iset)
        self.config = get_config(self.lm)
        self.file_list = [s for s in os.listdir(os.path.join(args.data_path, self.split))
                if s.isalnum()]
        self.cnt = 0
        self.cur_file = 0
        self.cur_epoch = 0
        self.steps = None
        self.inputs = None
        self.props = None
        if self.split == 'train':
            self.propositions = self.lm.training_propositions
        elif self.split == 'valid':
            self.propositions = self.lm.validation_propositions
        else:
            self.propositions = self.lm.test_propositions
        self.props_filter = []
        num = int(self.args.data_sampling * 10)
        for i,p in enumerate(self.propositions):
            if self.split != 'train' or (self.args.data_sampling > 0 and i % 10 < num):
                self.props_filter.append(p.label)
        print ('%d props after filtering' % (len(self.props_filter)))
        #self.length = len(self.file_list) * 50000
        if self.args.short_file_list:
            self.file_list = self.file_list[:1]
        random.shuffle(self.file_list)
        self.load_steps_all()
        #self.load_steps()
        self.length = len(self.idxes)

    def load_steps_all(self):
        if self.args.partial_lm:
            if not self.evaluate:
                random.shuffle(self.idxes)
            return
        self.steps = []
        self.inputs = []
        self.props = []
        if self.args.debug:
            self.file_list = self.file_list[:1]
        print (self.file_list)
        for fn in self.file_list:
            print ('load from %s' % (fn))
            _steps, _inputs, _props = load_proof_steps(
                    os.path.join(self.args.data_path, self.split, fn),
                    self.lm, _all=True)   
            for i, step in enumerate(_steps):
                lc = step.context_label
                lp = step.prop_label
                if lc != 'quartfull' and lp != 'quartfull' and lc in self.props_filter and len(step.prop.e) > 0:
                    self.steps.append(_steps[i])
                    self.inputs.append(_inputs[i])
                    self.props.append(_props[i])
            print ('load %d steps' % (len(self.steps)))
        #self.idxes = []
        #for i, step in enumerate(self.steps):
        #    lc = step.context_label
        #    lp = step.prop_label
        #    if lc != 'quartfull' and lp != 'quartfull' and lc in self.props_filter and len(self.steps[i].prop.e) > 0:
        #        self.idxes.append(i)
        #print ('load %d steps' % (len(self.idxes)))
        self.idxes = list(range(len(self.steps)))
        if not self.evaluate:
            random.shuffle(self.idxes)

    def load_steps(self, fn=None):
        if self.args.partial_lm:
            self.cnt = 0
            if not self.evaluate:
                random.shuffle(self.idxes)
            self.cur_epoch += 1
            return 
        if fn is None:
            if self.cur_file == len(self.file_list):
                self.cur_file = 0
                self.cur_epoch += 1
            fn = self.file_list[self.cur_file]
            self.cnt = 0
            self.cur_file += 1
            #if self.cur_file == len(self.file_list):
            #    self.cur_file = 0
            #    self.cur_epoch += 1
        print ('load from', fn)
        self.steps, self.inputs, self.props = load_proof_steps(
                os.path.join(self.args.data_path, self.split, fn),
                self.lm, _all=True)
        self.idxes = []
        for i,step in enumerate(self.steps):
            lc = step.context_label
            lp = step.prop_label
            if lc != 'quartfull' and lp != 'quartfull' and lc in self.props_filter and len(self.steps[i].prop.e) > 0:
                self.idxes.append(i)
        print ('load %d steps' % (len(self.idxes)))
        #self.idxes = list(range(len(self.steps)))
        if not self.evaluate:
            random.shuffle(self.idxes)
    
    def getitem_partial(self, idx):
        pass
    
    def check_end(self):
        if self.cnt == len(self.idxes):
            self.cnt = 0
            self.cur_epoch += 1
            #self.load_steps()
                
    def __getitem__(self, idx):
        #if self.steps is None or self.cnt == len(self.idxes):
        #    if self.steps is None and not self.evaluate:
        #        random.shuffle(self.file_list)
        #    self.load_steps()
            
        idx = self.idxes[self.cnt]#idx % len(self.steps)
        self.cnt += 1

        #if self.args.partial_lm:
        #    return self.getitem_partial(idx)

        goal_tokens = torch.Tensor(self.inputs[idx][0]).long().to(self.args.device)
        goal_tree = torch.Tensor(self.inputs[idx][1]).to(self.args.device)
        _prop_num = [self.steps[idx].prop.number]

        if self.evaluate:
            prop_idx = [self.lm.database.propositions[p].number for p in self.props[idx] if p != 'quartfull' and len(self.lm.database.propositions[p].e) > 0]
            #prop_idx = [self.lm.database.propositions[p].number for p in self.props[idx] ]
            self.check_end()
            return goal_tokens, goal_tree, None, None, prop_idx, prop_idx.index(_prop_num[0])

        if self.args.allneg:
            prop_num = [self.lm.database.propositions[p].number for p in self.props[idx] if p != 'quartfull' and len(self.lm.database.propositions[p].e) > 0]
            #[self.lm.database.propositions[i].number for i in self.props[idx]]
            pos_idx = prop_num.index(self.steps[idx].prop.number)
            if self.args.thm_emb:
                props_tokens = prop_num
                props_tree = []
            else:
                props_tokens = [torch.Tensor(self.lm_inputs[i][0]).long().to(self.args.device) for i in prop_num]
                props_tree = [torch.Tensor(self.lm_inputs[i][1]).to(self.args.device) for i in prop_num]
            self.check_end()
            return goal_tokens, goal_tree, props_tokens, props_tree, prop_num, pos_idx

        labels = [p for p in self.props[idx] if p != 'quartfull' and len(self.lm.database.propositions[p].e) > 0]
        labels.remove(self.steps[idx].prop_label)
        if len(labels) < self.args.negative_samples:
            labels = labels * self.args.negative_samples
        wrong_num = min(self.args.negative_samples, len(labels))
        rand = np.random.choice(len(labels), wrong_num, replace=False)
        _prop_num += [ self.lm.database.propositions[labels[i]].number for i in rand]
        if self.args.thm_emb:
            props_tokens = _prop_num
            props_tree = []
        else:
            props_tokens = [torch.Tensor(self.lm_inputs[i][0]).long().to(self.args.device) for i in _prop_num]
            props_tree = [torch.Tensor(self.lm_inputs[i][1]).to(self.args.device) for i in _prop_num]
        
        self.check_end()
        #print (self.cnt)
        return goal_tokens, goal_tree, props_tokens, props_tree

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
    loader = torch.utils.data.DataLoader(
            MetaMath(args), collate_fn=mycollate,
            worker_init_fn=worker_init,
            batch_size=10, shuffle=False, num_workers=0)#args.num_workers)
    t = time.time()
    for i in range(100):
        data = next(iter(loader))
        print ((time.time()-t)/(i+1))
    #print (data)
