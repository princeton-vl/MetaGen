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
    return goal_tokens, goal_trees, torch.Tensor([b[2] for b in batch])
    # [tensor of strings] in b
    # [s*4] in b
    # [str] in b* (1+neg)
    # [s*4] in b* (1+neg)

class MetaMath(data.Dataset):
    def __init__(self, args, split=None):
        self.args = args
        self.split = split if split is not None else args.split
        if self.args.partial_lm:
            if self.split == 'train':
                self.file_list = list(range(4,20))
            elif self.split == 'valid':
                self.file_list = [0,1]
            else:
                self.file_list = [2,3]
            self.file_list = [str(i) for i in self.file_list]
            self.lm = build_language_model(self.args.num_props, new=self.args.gen_lm, _all=False)
            self.config = get_config(self.lm)
            self.steps = []
            for fn in self.file_list:
                with open(os.path.join(self.args.data_path+fn), 'rb') as f:
                    data = pickle.load(f)
                for i,_pd in enumerate(data):
                    for tree in _pd.correct:
                        _to, _tr = encode_payout_step(tree, _pd.hyps, self.lm, self.config)
                        self.steps.append((_to, _tr, 1))
                    for tree in _pd.wrong:
                        _to, _tr = encode_payout_step(tree, _pd.hyps, self.lm, self.config)
                        self.steps.append((_to, _tr, 0))
            print ('load %d payout steps' % (len(self.steps)))
            self.idxes = list(range(len(self.steps)))
            if not self.args.evaluate:
                random.shuffle(self.idxes)
            self.cnt = 0
            self.length = len(self.idxes)
            self.cur_epoch = 0
            return
        self.lm = load_language_model(_all=False, new=args.gen_lm, iset=args.iset)
        self.config = get_config(self.lm)
        if self.split == 'train':
            self.file_list = list(range(4,20))
        elif self.split == 'valid':
            self.file_list = [0,1]
        else:
            self.file_list = [2,3]
        self.file_list = [str(i) for i in self.file_list]
        self.steps = []
        name = self.args.data_path.split('/')[-2]
        for fn in self.file_list:
            with open(os.path.join(self.args.data_path+fn), 'rb') as f:
                data = pickle.load(f)
            for i,_pd in enumerate(data):
                for tree in _pd.correct:
                    _to, _tr = encode_payout_step(tree, _pd.hyps, self.lm, self.config)
                    self.steps.append((_to, _tr, 1))
                for tree in _pd.wrong:
                    _to, _tr = encode_payout_step(tree, _pd.hyps, self.lm, self.config)
                    self.steps.append((_to, _tr, 0))
        self.idxes = list(range(len(self.steps)))
        print ('load %d payout steps' % (len(self.steps)))
        #self.file_list = [s for s in os.listdir(os.path.join(args.data_path, self.split))
        #        if s[-1] == 't']
        self.cnt = 0
        self.cur_file = 0
        self.cur_epoch = 0
        #self.steps = None
        #self.inputs = None
        #self.props = None
        self.length = len(self.idxes)
        if not self.args.evaluate:
            random.shuffle(self.idxes)
        #random.shuffle(self.file_list)

    def load_steps(self, fn=None):
        if self.args.partial_lm:
            if not self.args.evaluate:
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
        self.cnt = 0
        print ('load from', fn)
        with open(os.path.join(self.args.data_path, fn), 'rb') as f:
            self.steps = pickle.load(f)
        self.idxes = list(range(len(self.steps)))
        if not self.args.evaluate:
            random.shuffle(self.idxes)

    def __getitem__(self, idx):
        if self.steps is None or self.cnt == len(self.steps):
            #if self.steps is None and not self.args.evaluate:
            #    random.shuffle(self.file_list)
            #self.load_steps()
            self.cnt = 0
            self.cur_epoch += 1
        idx = self.idxes[self.cnt]#idx % len(self.steps)
        self.cnt += 1
        tokens = [self.config.encode[i] for i in self.steps[idx][0]]
        goal_tokens = torch.Tensor(tokens).long().to(self.args.device)
        goal_tree = torch.Tensor(self.steps[idx][1]).to(self.args.device)
        #print (self.cnt)
        return goal_tokens, goal_tree, self.steps[idx][2]

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
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loader = torch.utils.data.DataLoader(
            MetaMath(args), collate_fn=mycollate,
            batch_size=10, shuffle=False, num_workers=0)#args.num_workers)
    t = time.time()
    for i in range(100):
        data = next(iter(loader))
        print ((time.time()-t)/(i+1))
    print (data)
    #print (data)
