import os,sys
import numpy as np
import math
import time
import random
import multiprocessing as mp

import torch
import torch.utils.data as data
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.distributions import Categorical

import torch_models
import params
import log
from data_utils import *
import interface_rl
from constructor import Constructor
import trainer

class TrieTree:
    def __init__(self, strings):
        self.root = {}
        for s in strings:
            self.insert(s)
    def insert(self, s):
        current = self.root
        for c in s:
            if c not in current:
                current[c] = {} 
            current = current[c]
        current['END'] = 0
    def search(self, s):
        current = self.root
        for c in s:
            if c not in current:
                return False
            current =  current[c]
        if 'END' not in current:
            return False
        return True
    def count(self, node):
        l = 0
        if len(node) == 0:
            return 0
        if 'END' in node:
            l = 1
        for c in node:
            if c != 'END':
                l += self.count(node[c])
        return l

class FakeReward:
    def __init__(self, args, config):
        tmp = args.data_sampling
        args.data_sampling = 1
        generator = Constructor(args, config)
        generator.initialize()
        args.data_sampling = tmp
        replacement_dict = config.lm.deterministic_replacement_dict_f(f=generator.all_f)
        self.searcher = TrieTree([])
        for e in generator.expressions_list:
            if e.is_hyps != 2:
                t, _ = generator.encode_expr(e.tree, 
                        [generator.expressions_list[i].tree for i in e.hyps],
                        replacement_dict)
                self.searcher.insert(t)
        return
    def simple_reward(self, expr):
        if expr.tree.value == 'wi':
            return 1
        else:
            return 0
    def reward(self, tokens, trees):
        r = []
        for token, tree in zip(tokens, trees):
            if self.searcher.search(token):
                r.append(1)
            else:
                r.append(0)
        return r

class LMReward:
    def __init__(self, args, config):
        self.args = args
        self.config = config
        if args.lm_model != '':
            data = torch.load(args.lm_model)
            data['args'].device = args.device
            device = torch.device("cuda:0")
            self.model = torch_models.LModel(data['args']).to(device)
            self.model.load_state_dict(data['models']) 
            self.model = self.model.to(args.device) 
        else:
            self.model = torch_models.LModel(args).to(args.device)
        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.log_softmax = torch.nn.LogSoftmax(dim=1)
    def reward(self, tokens, trees):
        ppl = []
        with torch.no_grad():
            score = self.model((tokens, trees))
            for i,token in enumerate(tokens):
                _score = score[i, :len(token)]
                loss = self.cross_entropy(_score, token).item()
                ppl.append(-loss)
        return ppl


def sample_expr(args, generator, reward_fn, fake_reward_fn):
    expr = []
    pred_rewards = generator.interface.pred_rewards
    gen_rewards = generator.interface.gen_rewards
    rewards = []
    rewards_fake = []
    num_pos = 0
    replacement_dict = config.lm.deterministic_replacement_dict_f(f=generator.all_f)
    while len(expr) < args.len_episode:
        if args.random_generate:
            e = generator.random_generate()    
        else:
            e = generator.parameterized_generate()
        if e is not None:
            expr.append(e)
            token, tree = generator.encode_expr(e.tree, [generator.expressions_list[i].tree for i in e.hyps], replacement_dict)
            r_fake = fake_reward_fn.reward([token], [tree])[0]
            if args.fake_reward:
                r = r_fake
            else:
                r = reward_fn.reward([token], [tree])[0]
            if r_fake == 1:
                num_pos += 1
            rewards.append(r)
            rewards_fake.append(r_fake)
            if args.train_rl:
                generator.interface.expr_pos.append((len(pred_rewards)-1, len(gen_rewards)-1))
                for i in range(len(pred_rewards[-1])):
                    pred_rewards[-1][i] = r
                for i in range(len(gen_rewards[-1])):
                    if gen_rewards[-1][i] == 1:
                        gen_rewards[-1][i] = r
    print (num_pos, len(expr))
    return expr, rewards, rewards_fake

def finish_eposide(args, generator, opt, e, _logger):
    # count rewards
    loss = []
    log_probs_all = []
    rewards_all = []
    interface = generator.interface
    assert len(interface.pred_logs) == len(interface.pred_rewards)
    assert len(interface.gen_logs) == len(interface.gen_rewards)
    for i1, i2 in interface.expr_pos:
        for log_prob, reward in zip(interface.pred_logs[i1], interface.pred_rewards[i1]):
            if reward != None:
                log_probs_all.append(log_prob)
                rewards_all.append(reward)
        for log_prob, reward in zip(interface.gen_logs[i2], interface.gen_rewards[i2]):
            if reward != None:
                log_probs_all.append(log_prob)
                rewards_all.append(reward)

    rewards = torch.Tensor(rewards_all).to(args.device)
    rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    for log_prob, reward in zip(log_probs_all, rewards):
        loss.append(-log_prob*reward)
    loss = sum(loss)
    opt.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_value_(interface.pred_model.parameters(), args.grad_clip)
    torch.nn.utils.clip_grad_value_(interface.gen_model.parameters(), args.grad_clip)
    opt.step()
    interface.pred_logs.clear()
    interface.pred_rewards.clear()
    interface.gen_logs.clear()
    interface.gen_rewards.clear()
    interface.expr_pos.clear()

def train(args, generator, reward_fn, fake_reward_fn, opt, e, _logger):
    generator.reinitialize_expressions()
    rewards_all = AverageMeter()
    rewards_fake_all = AverageMeter()
    for i in range(args.num_episode):
        expr, rewards, rewards_fake = sample_expr(args, generator, reward_fn, fake_reward_fn)
        if args.train_rl:
            finish_eposide(args, generator, opt, e, _logger)
        rewards_all.update(sum(rewards), len(expr))
        rewards_fake_all.update(sum(rewards_fake), len(expr))
        _logger.info('Epoch %d Iter %d NumExpr %d AvgRwd %.5f AvgFkRwd %.5f', 
                e,  i, len(expr), rewards_all.sum, rewards_fake_all.sum)
    return rewards_all.sum

def evaluate(l,  fn):
    exprs = []
    rewards = []
    replacement_dict = config.lm.deterministic_replacement_dict_f(f=generator.all_f)
    while len(exprs) < l:
        if args.random_generate:
            e = generator.random_generate()
        else:
            e = generator.parameterized_generate()
        exprs.append(e)
        token, tree = generator.encode_expr(e.tree, [generator.expressions_list[i].tree for i in e.hyps], replacement_dict)
        r = fn.reward([token], [tree])[0] 
        rewards.append(r)
        if len(exprs) % 100 == 0:
            print (len(exprs), sum(rewards))
    return exprs, rewards

if __name__ == "__main__":
    args = params.get_args()
    _logger = log.get_logger(__name__, args)
    _logger.info(print_args(args))
    
    if args.partial_lm:
        lm = build_language_model(args.num_props, new=args.gen_lm, _all=False)
    else:
        lm = load_language_model(_all=False, new=args.gen_lm)
        load_proof_steps_into_lm(lm, ['train']) 
    config = get_config(lm)
    args.vocab_size = len(config.encode)+1
    
    interface = interface_rl.LMInterface(args, lm)
    generator = Constructor(args, config, interface)
    generator.initialize()
    print (generator.prop_dist)
    fake_reward_fn = FakeReward(args, config)
    reward_fn = LMReward(args, config)
    if args.fake_reward:
        reward_fn = fake_reward_fn
    opt = get_opt({
            'pred':interface.pred_model, 
            'gen':interface.gen_model
            }, args)
    rewards_all = 0 
    for e in range(args.epoch):
        rewards_all += train(args, generator, reward_fn, fake_reward_fn, opt, e, _logger)
        print ('acc rewards', rewards_all) 
        aux = {'epoch':e, 'cur_iter':1}
        save_model(aux, args, {'pred':interface.pred_model, 'gen':interface.gen_model}, opt=None)
    