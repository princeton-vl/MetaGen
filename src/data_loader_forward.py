import os
import numpy as np
import json
import random
import math
import argparse
import params
import torch
import time
import pickle
from constructor import Constructor

from data_utils import *

class dataloader:
    def __init__(self, args, worker_id, split='train', task='pred'):
        self.args = args
        self.split = split
        self.task = task
        #self.generator = generator
        self.file_list = [s for s in os.listdir(os.path.join(args.data_path, split))
        if s.isdigit()]
        per_worker = len(self.file_list)//args.num_workers+1
        self.file_list = self.file_list[worker_id*per_worker:min((worker_id+1)*per_worker,len(self.file_list))]
        self.length = 0
        self.steps = None
        self.step_hyps = None
        if self.args.partial_lm:
            self.lm = build_language_model(self.args.num_props, new=self.args.gen_lm, _all=False)
            self.cnt = 0
            self.cur_file = 0
            self.cur_epoch = 0
            self.config = get_config(self.lm)
            #self.exprs = load_exprs(self.lm)
            generator = Constructor(args, self.config)
            generator.initialize()
            self.exprs = generator.expressions_list
            self.step_expr_pos = generator.step_expr_pos
            self.prop_hyps_pos = generator.prop_hyps_pos
            self.all_f = generator.all_f
            if self.split == 'train':
                self.propositions = self.lm.training_propositions
                #self.steps = self.lm.training_proof_steps
            elif self.split == 'valid':
                self.propositions = self.lm.validation_propositions
                #self.steps = self.lm.validation_proof_steps
            else:
                self.propositions = self.lm.test_propositions
                #self.steps = self.lm.test_proof_steps
            self.load_forward_info()
            self.steps = []
            steps = []
            if self.args.data_sampling > 0:
                num = int(self.args.data_sampling * 10)
            for i,p in enumerate(self.propositions):
                if self.split != 'train' or (self.args.data_sampling>0 and i % 10 < num):
                    steps += [step for step in p.entails_proof_steps
                        if not (step.prop.type=='f' or step.prop.type == 'e')]
                    for j, step in enumerate(p.entails_proof_steps):
                        step.pos_in_context = j
            for step in steps:
                if self.task == 'pred':
                    if len(self.lm.database.propositions[step.prop_label].e) > 0:
                        self.steps.append(step)
                else:
                    if len(self.prop_goal_var[step.prop_label]) > 0:
                        self.steps.append(step)
            if self.task == 'pred':
                self.idxes = list(range(len(self.steps)))
            else:
                self.idxes = [(i,j) for i in range(len(self.steps)) for j in range(len(self.prop_goal_var[self.steps[i].prop_label]))]
            if self.split == 'train':
                random.shuffle(self.idxes)
            return
        else:
            self.lm = load_language_model(new=args.gen_lm, iset=args.iset)
        self.exprs = load_exprs(os.path.join(args.data_path, 'expressions_list_%.1f' % (self.args.data_sampling)), self.lm)
        self.config = get_config(self.lm)
        self.all_f = {}
        for l,p in self.lm.database.propositions.items():
            for f in p.f:
                if f not in self.all_f:
                    self.all_f[f] = p.f[f]
        self.load_forward_info()
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
        self.load_steps_all()
        self.length = len(self.idxes)
        self.cur_file = 0
        self.cur_epoch = 0
        self.cnt = 0

    def load_forward_info(self):
        args = self.args
        if self.args.partial_lm:
            data = torch.load(os.path.join(args.data_path, 'forward_step_info_'+str(self.args.num_props)+('' if self.args.data_sampling==1 else '01')))
        else:
            data = torch.load(os.path.join(args.data_path, 'forward_step_info_%.1f' % (self.args.data_sampling)))
        self.step_hyps = data['step_hyps']
        self.step_prior_idx = data['step_prior_idx']
        self.goal_unconstrained = data['goal_unconstrained']
        self.prop_goal_var = data['prop_goal_var']
        if self.args.partial_lm:
            self.visible_expr = torch.load(os.path.join(args.data_path, 'visible_expr_'+str(self.args.num_props)+('' if self.args.data_sampling==1 else '01')))
        else:
            self.visible_expr = torch.load(os.path.join(args.data_path, 'visible_expr_%.1f' % (self.args.data_sampling)))
            self.step_expr_pos = torch.load(os.path.join(args.data_path, 'step_expr_pos_%.1f' % (self.args.data_sampling)))
            self.prop_hyps_pos = torch.load(os.path.join(args.data_path, 'prop_hyps_pos_%.1f' % (self.args.data_sampling)))

    def load_steps_all(self):
        if self.args.partial_lm:
            if self.split == 'train':
                random.shuffle(self.idxes)
            return
        self.steps = []
        print (self.file_list)
        for fn in self.file_list:
            print ('load from %s' % (fn))
            full_fn = os.path.join(self.args.data_path, self.split, fn)
            with open(full_fn, 'rb') as f:
                _steps = pickle.load(f)
            for step in _steps:
                if self.task == 'pred':
                    if step.context_label in self.props_filter and len(self.lm.database.propositions[step.prop_label].e) > 0:
                        self.steps.append(step)
                if self.task == 'gen':
                    if step.context_label in self.props_filter and len(self.prop_goal_var[step.prop_label]) > 0:
                        self.steps.append(step)
            print ('load %d steps' % (len(self.steps)))
        if self.task == 'pred':
            self.idxes = list(range(len(self.steps)))
        else:
            self.idxes = [(i,j) for i in range(len(self.steps)) for j in range(len(self.prop_goal_var[self.steps[i].prop_label]))]
        if self.split == 'train':
            random.shuffle(self.idxes)

    def load_steps(self, fn=None):
        if self.args.partial_lm:
            self.cnt = 0
            if self.split == 'train':
                random.shuffle(self.idxes)
            self.cur_epoch += 1
            return
        if fn is None:
            fn = self.file_list[self.cur_file]
            self.cnt = 0
            self.cur_file += 1
            if self.cur_file == len(self.file_list):
                self.cur_file = 0
                self.cur_epoch += 1
        print ('load from', fn)
        full_fn = os.path.join(self.args.data_path, self.split, fn)
        with open(full_fn, 'rb') as f:
            data = pickle.load(f)
        self.steps = []
        for step in data:
            if self.task == 'pred':
                if len(self.lm.database.propositions[step.prop_label].e) > 0:
                    self.steps.append(step)
            else:
                if len(self.prop_goal_var[step.prop_label]) > 0:
                    self.steps.append(step)
        if self.task == 'pred':
            self.idxes = list(range(len(self.steps)))
        else:
            self.idxes = [(i,j) for i in range(len(self.steps)) for j in range(len(self.prop_goal_var[self.steps[i].prop_label]))]
        if self.split == 'train':
            random.shuffle(self.idxes)

    def encode_expr(self, tree, hyps, replacement_dict):
        statement = tree.copy().replace_values(replacement_dict)
        hyps = [h.copy().replace_values(replacement_dict) for h in hyps]
        statement_graph_structure = TreeInformation([statement],
            start_symbol=None, intermediate_symbol='END_OF_HYP',
            end_symbol='END_OF_SECTION')
        hyps_graph_structure = TreeInformation(hyps,
            start_symbol=None, intermediate_symbol='END_OF_HYP',
            end_symbol='END_OF_SECTION')
        in_string, structure_data = merge_graph_structures_new([statement_graph_structure, hyps_graph_structure])
        tokens = [self.config.encode[t] for t in in_string]
        return tokens, structure_data

    def encode_pred(self, idx):
        step = self.steps[idx]
        prop = self.lm.database.propositions[step.prop_label]
        context = self.lm.database.propositions[step.context_label]
        fit = step.tree.fit(prop.tree, prop.f)
        for i,var in enumerate(prop.unconstrained_variables):
            fit[var] = step.unconstrained[i]
        replacement_dict = self.lm.deterministic_replacement_dict_f(f=self.all_f)#random_replacement_dict_f(f=self.all_f)
        num_e = len(prop.e)
        num_picked = random.randint(1, num_e)
        picked = random.sample(range(num_e), num_picked)
        target = picked[0]
        used = picked[1:]
        # prop.tree prop.hyps context.hyps exprs
        prop_hyps = [hyps.tree for hyps in prop.hyps if hyps.type=='e']
        prop_hyps_label = [hyps.label for hyps in prop.hyps if hyps.type=='e']
        used_fit = {}
        for idx in used:
            s = prop_hyps[idx].list()
            for t in s:
                if t in fit and t not in used_fit:
                    used_fit[t] = fit[t]
        prop_tree = prop.tree.copy().replace(used_fit).replace_values(replacement_dict)
        prop_hyps = [h.copy().replace(used_fit).replace_values(replacement_dict) for h in prop_hyps]

        prop_hyps[target] = Tree('TARGET_UC')
        context_hyps_labels = set()
        for idx in used:
            step_idx = self.step_prior_idx[context.label][step.pos_in_context][idx]
            if type(step_idx) == type(''):
                context_hyps_labels.add(step_idx)
            else:
                context_hyps_labels.update(self.step_hyps[context.label][step_idx])
        context_hyps = [context.e[l].tree.copy().replace_values(replacement_dict) for l in context_hyps_labels]

        prop_tree_structure = TreeInformation([prop_tree],
            start_symbol=None, intermediate_symbol='END_OF_HYP',
            end_symbol='END_OF_SECTION')
        prop_hyps_structure = TreeInformation(prop_hyps,
            start_symbol=None, intermediate_symbol='END_OF_HYP',
            end_symbol='END_OF_SECTION')
        context_hyps_structure = TreeInformation(context_hyps,
            start_symbol=None, intermediate_symbol='END_OF_HYP',
            end_symbol='END_OF_SECTION')
        in_string, in_tree = merge_graph_structures_new([
            prop_tree_structure, prop_hyps_structure, context_hyps_structure])
        in_tokens = [self.config.encode[t] for t in in_string]

        labels = self.visible_expr[prop.label][prop_hyps_label[target]].copy()
        target_step_idx = self.step_prior_idx[context.label][step.pos_in_context][target]
        if type(target_step_idx) == type(''):
            pos = self.prop_hyps_pos[context.label][target_step_idx]
        else:
            pos = self.step_expr_pos[context.label][target_step_idx]
        if pos in labels:
            labels.remove(pos)
        if len(labels) < self.args.negative_samples:
            labels += random.sample(range(len(self.exprs)), self.args.negative_samples-len(labels))
        if self.args.allneg or self.args.hardneg:
            neg = labels
        else:
            neg = random.sample(labels, self.args.negative_samples)

        pos_tokens, pos_tree = self.encode_expr(self.exprs[pos].tree,
            [self.exprs[i].tree for i in self.exprs[pos].hyps],
            replacement_dict)
        expr_tokens = [pos_tokens]
        expr_trees = [pos_tree]
        for idx in neg:
            tokens, tree = self.encode_expr(self.exprs[idx].tree,
            [self.exprs[i].tree for i in self.exprs[idx].hyps],
            replacement_dict)
            expr_tokens.append(tokens)
            expr_trees.append(tree)
        if self.args.allneg or self.args.hardneg:
            return in_tokens, in_tree, expr_tokens, expr_trees, len(expr_tokens), 0
        return in_tokens, in_tree, expr_tokens, expr_trees

    def encode_gen(self, idx):
        step = self.steps[idx[0]]
        target = idx[1]
        prop = self.lm.database.propositions[step.prop_label]
        context = self.lm.database.propositions[step.context_label]

        unconstrained_variables = self.prop_goal_var[prop.label]
        uv_dict = {var:'UC'+var for var in unconstrained_variables}

        fit = step.tree.fit(prop.tree, prop.f)
        for i,var in enumerate(prop.unconstrained_variables):
            fit[var] = step.unconstrained[i]
        cur_fit = {l:fit[l] for l in fit if l not in unconstrained_variables}

        replacement_dict = self.lm.deterministic_replacement_dict_f(f=self.all_f)#random_replacement_dict_f(f=self.all_f)
        for var in uv_dict:
            replacement_dict[uv_dict[var]] = 'UC'
        #target = random.choice(unconstrained_variables)
        replacement_dict['UC'+unconstrained_variables[target]] = 'TARGET_UC'
        out_tree = fit[unconstrained_variables[target]].copy().replace_values(replacement_dict)
        tree = prop.tree.copy().replace_values(uv_dict).replace(cur_fit)#.replace_values(replacement_dict)
        hyps = [context.e[l].tree#.copy().replace_values(replacement_dict)
            for l in self.step_hyps[context.label][step.pos_in_context]]
        in_tokens, in_tree = self.encode_expr(tree, hyps, replacement_dict)

        out_graph_structure = TreeInformation([out_tree], start_symbol='START_OUTPUT',
            intermediate_symbol=None, end_symbol=None)
        out_tokens, out_tree = merge_graph_structures_new([out_graph_structure])
        out_tokens = [self.config.encode[t] for t in out_tokens]
        return in_tokens, in_tree, out_tokens, out_tree
        #statement_graph_structure = TreeInformation([statement],
        #    start_symbol=None, intermediate_symbol='END_OF_HYP',
        #    end_symbol='END_OF_SECTION')
        #hyps_graph_structure = TreeInformation(hyps,
        #    start_symbol=None, intermediate_symbol='END_OF_HYP',
        #    end_symbol='END_OF_SECTION')
        #in_string, in_tree = merge_graph_structures_new([statement_graph_structure, hyps_graph_structure])

        #uv_pos = []
        '''
        target = random.choice(range(len(unconstrained_variables)))
        for i in range(len(unconstrained_variables)):
            #uv_pos.append([])
            if i == target:
                _token = 'TARGET_UC'
            else:
                _token = 'UC'
            for j in range(_pos, len(in_string)):
                if in_string[j] == 'END_OF_SECTION':
                    break
                if in_string[j] == uv_dict[unconstrained_variables[i]]:
                    #uv_pos[i].append(j)
                    in_string[j] = _token
        in_tokens = []
        for t in
        '''
    def check_end(self):
        if self.cnt == len(self.idxes):
            self.cnt = 0
            self.cur_epoch += 1

    def get(self):
        #if self.steps is None or self.cnt == len(self.steps):
        #    if self.steps is None and self.split == 'train':
        #        random.shuffle(self.file_list)
        #    self.load_steps()
        idx = self.idxes[self.cnt]
        self.cnt += 1
        #step = self.steps[idx]
        self.check_end()
        if self.task == 'pred':
            return self.encode_pred(idx)
        else:
            return self.encode_gen(idx)


def worker(args, workerid, task, queue):
    #if args.partial_lm:
    #lm = build_language_model(args.num_props, new=args.gen_lm, _all=False)
    #config = get_config(lm)
    #generator = constructor3_allneg.Constructor(args, config)
    #generator.initialize()

    #else:

    #exprs = load_exprs()
    loader = dataloader(args, workerid, 'train', task)
    if args.partial_lm:
        print ('build loader with %d training samples' %(len(loader.idxes)))
    #validloader = dataloader(args, workerid, exprs, args.valid_split, task)
    cur_epoch = 0
    #sig = None
    while True:
        #if sig is None:
        #    print ('no signal from q3, wait')
        #    sig = q3.get()
        #    print ('get new signal', sig)
        #if q3.qsize() > 0:
        #    sig = q3.get()
        #    print ('get new signal', sig)
        #if sig == 'train':
        #    loader = trainloader
        #    queue = q1
        #else:
        #    loader = validloader
        #    queue = q2
        data = [[], [], [], []]
        if task == 'pred' and (args.allneg or args.hardneg):
            data = [[], [], [], [], [], []]
        for i in range(args.batch_size):
            batch = loader.get()
            if task == 'pred':
                data[0].append(batch[0])
                data[1].append(batch[1])
                data[2] += batch[2]
                data[3] += batch[3]
                if args.allneg or args.hardneg:
                    data[4].append(batch[4])
                    data[5].append(batch[5])
            else:
                data[0].append(batch[0])
                data[1].append(batch[1])
                data[2].append(batch[2])
                data[3].append(batch[3])
        queue.put(data)
        if loader.cur_epoch != cur_epoch:
            # epoch end
            cur_epoch = loader.cur_epoch
            if workerid == 0:
                queue.put(None)


def get_expr(generator, step):
    exprs = []
    if len(step.prop.e) == 0:
        return None
    fit = step.tree.fit(step.prop.tree, step.prop.f)
    for i,var in enumerate(step.prop.unconstrained_variables):
        fit[var] = step.unconstrained[i]
    cur_fit = {}
    hyps = [h for h in step.prop.hyps if h.type == 'e']
    hyps = sorted(hyps, key=lambda x: len(x.tree.list()), reverse=True)
    for h in hyps:
        tree = h.tree.copy().replace(cur_fit)
        idxes = generator.search(tree, k=200)
        #tree = hyps.tree.copy().replace(fit)
        exprs.append(idxes)
        for s in h.tree.list():
            if s in fit and s not in cur_fit:
                cur_fit[s] = fit[s]
    return exprs
