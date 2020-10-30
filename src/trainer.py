import os,sys
import numpy as np
import math
import time
import random
import multiprocessing as mp

import torch
import torch.utils.data as data
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import torch_models
import params
import log
from data_loader_pred import *
from data_utils import *
import interface_lm
from constructor import Constructor

def worker(generator, queue, expressions, task='pred', batch_size=None):
    args = generator.args
    if batch_size is None:
         batch_size = args.batch_size
    while True:
        exprs = []
        while len(exprs) < batch_size:
            expr = random.choice(expressions)
            if task == 'gen' and len(expr.unconstrained) == 0:
                continue
            exprs.append(expr)
        if task == 'pred':
            queue.put(generator.encode_pred_tasks(exprs))
        else:
            queue.put(generator.encode_gen_tasks(exprs))


class Trainer:
    def __init__(self, args, config, model, generator, _logger, validloader):
        self.args = args
        self.config = config
        self.lm = config.lm
        self.model = model
        self._logger = _logger
        if generator is not None:
            self.generator = generator
            self.cons_params = list(generator.interface.pred_model.parameters()) + list(generator.interface.gen_model.parameters()) # 1.8 M in default settinngs
            for data in self.cons_params:
                data.requires_grad = False
        self.noise = []
        self.exprs = []
        self.accs = []
        self.model_params = None
        self.cur_epoch = 0
        self.cur_iter = 0
        self.cur_noise = []
        self.loader = None
        self.validloader = validloader
        self.queue = None
        self.batch_size = None
        if args.task == 'gen':
            self.criterion = torch.nn.CrossEntropyLoss().to(args.device)
        if not args.stat_model:
            self.model_opt = torch.optim.Adam(params=model.parameters(), lr=self.args.learning_rate, weight_decay=1e-4)

    def step(self, batch):
        score = self.model(batch)
        if self.batch_size is not None:
            bs = self.batch_size#
        else:
            bs = self.args.batch_size
        acc = (score[:bs].max(dim=1)[1]==0).sum().item()
        if score.shape[0] > bs:
            acc_cons = (score[bs:].max(dim=1)[1]==0).sum().item()
        prob = torch.nn.functional.softmax(score, dim=1)[:,0].view(-1)
        ppl = prob[:bs].sum().item()
        if score.shape[0] > bs:
            ppl_cons = prob[bs:].sum().item()
        score[:, 1:] = -score[:, 1:]
        score = torch.nn.functional.logsigmoid(score)
        loss = -score.sum(dim=1).mean()
        if score.shape[0] > bs:
            return loss, acc, acc_cons, ppl, ppl_cons
        return loss, acc, ppl
        
    def step_gen(self, batch):
        score = self.model(batch)
        loss = []
        acc = []
        acc_cons = []
        ppl = []
        ppl_cons = []
        log_softmax = torch.nn.LogSoftmax(dim=1)
        for i,tokens in enumerate(batch[-2]):
            if len(tokens) > self.args.max_len:
                continue
            gt = torch.zeros(len(tokens)).long().to(self.args.device)
            gt[:-1] = tokens[1:]
            gt[-1] = self.config.encode['END_OF_SECTION']
            _score = score[i, :len(tokens)]
            loss.append(self.criterion(_score, gt))
            _ppl = torch.exp(log_softmax(_score).gather(1, gt.view(-1,1)).sum() / len(gt)).item()
            if i < self.args.batch_size:
                num_p = (_score.max(dim=1)[1]==gt).sum().item()
                acc.append( num_p/len(gt)  )
                ppl.append(_ppl)
            else:
                acc_cons.append( ( _score.max(dim=1)[1]==gt).sum().item()/len(gt)  )
                ppl_cons.append(_ppl)
        loss = sum(loss)
        acc = sum(acc)
        acc_cons = sum(acc_cons)
        ppl = sum(ppl)
        ppl_cons = sum(ppl_cons)
        if len(batch[0]) > self.args.batch_size:
            return loss, acc, ppl, acc_cons, ppl_cons
        return loss, acc, ppl

    def step_allneg(self, batch):
        g_tokens, g_trees, p_tokens, p_trees, prop_idx, true_prop = batch
        g_vec = self.model.embed(g_tokens, g_trees, _type='g')
        p_vec = self.model.embed(p_tokens, p_trees, _type='p')
        loss = []
        acc = 0
        cnt = 0
        acc_cons = 0
        ppl = 0
        ppl_cons = 0
        num = 0
        for g, p_idx, true_pos in zip(g_vec, prop_idx, true_prop):
            l = len(p_idx) if type(p_idx) == list else p_idx
            p_emb = p_vec[cnt:cnt+l]
            cnt += l
            score = self.model.biln(g.view(1,-1), p_emb).view(-1)
            idx = score[true_pos]<=score
            _pos = idx.sum().item()
            if _pos == 1:
                if num >= self.args.batch_size:
                    acc_cons += 1
                else:
                    acc += 1
            num += 1
            if self.args.train_neg_only:
                loss.append(-score[true_pos] + score[idx].exp().sum().log())
            else:
                loss.append(-score[true_pos] + score.exp().sum().log())
            prob = np.exp(-loss[-1].item())
            if num >= self.args.batch_size:
                ppl_cons += prob
            else:
                ppl += prob
        loss = sum(loss)/len(loss)
        if len(batch[0]) > self.args.batch_size:
            return loss, acc, acc_cons, ppl, ppl_cons
        return loss, acc, ppl

    def train_one_epoch(self, _train=True):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        acces = AverageMeter()
        ppls = AverageMeter()

        if _train:
            self.model.train()
        else:
            self.model.eval()
        end = time.time()
        for i in range(self.args.num_iters):
            batch = self.queue.get()
            data_time.update(time.time() - end)
            if self.args.task == 'gen':
                loss, acc, ppl = self.step_gen(batch)
            elif self.args.allneg:
                loss, acc, ppl = self.step_allneg(batch)
            else:
                loss, acc, ppl = self.step(batch)
            if _train:
                self.model_opt.zero_grad()
                loss.backward()
                self.model_opt.step()
            acces.update(acc, len(batch[0]))
            losses.update(loss.item(), len(batch[0]))
            ppls.update(ppl, len(batch[0]))
            batch_time.update(time.time() - end)
            end = time.time()
            print_info  = 'Epoch: %d (%d/%d) Data: %.4f | \
                Batch: %.4f | Loss: %.4f | \
                Acc: %.4f ppl: %.4f' % (
                        self.cur_epoch,
                        i + 1,
                        self.args.num_iters,
                        data_time.val,
                        batch_time.val,
                        losses.avg*100,
                        acces.avg*100,
                        ppls.avg )
            self._logger.info(print_info)
            if self.args.manual_out and i % 1000 == 999:
                with open(self.args.log+'_p', 'a') as f:
                    f.write(print_info+'\n')

    def train_from_loader(self, loader, _train=True):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        dislosses = AverageMeter()
        jointlosses = AverageMeter()
        acces = AverageMeter()
        ppls = AverageMeter()
        if _train:
            self.model.train()
        else:
            self.model.eval()
        end = time.time()
        _epoch = loader.dataset.cur_epoch
        for i, batch in enumerate(loader):
            data_time.update(time.time() - end)
            if self.args.task == 'gen':
                loss, acc, ppl = self.step_gen(batch)
            elif self.args.allneg:
                loss, acc, ppl = self.step_allneg(batch)
            elif hasattr(self.args, 'goal_joint_loss') and self.args.goal_joint_loss >= 0:
                loss, dis_loss, acc, ppl = self.step_joint(batch)
                dislosses.update(dis_loss.item(), 1)
            elif hasattr(self.args, 'reconstruct_loss') and self.args.reconstruct_loss:
                loss, dis_loss, acc, ppl = self.step_inter_recons(batch)
                dislosses.update(dis_loss.item(), 1)
            elif hasattr(self.args, 'inter_props_cls') and self.args.inter_props_cls:
                loss, dis_loss, acc, ppl = self.step_inter_props_cls(batch)
                dislosses.update(dis_loss.item(), 1)
            elif hasattr(self.args, 'joint_inter_emb') and self.args.joint_inter_emb:
                loss, dis_loss, acc, ppl = self.step_joint2(batch)
                dislosses.update(dis_loss.item(), 1)
            elif hasattr(self.args, 'training_stage') and self.args.training_stage == 2:
                loss = self.step_emb(batch)
                acc = 0
                ppl = 0
            else:
                loss, acc, ppl = self.step(batch)
            if _train:
                self.model_opt.zero_grad()
                final_loss = loss
                if hasattr(self.args, 'goal_joint_loss') and self.args.goal_joint_loss >= 0:
                    final_loss = loss*(1-self.args.goal_joint_loss) + dis_loss*self.args.goal_joint_loss
                    jointlosses.update(final_loss.item(), 1)
                if hasattr(self.args, 'joint_inter_emb') and self.args.joint_inter_emb:
                    final_loss = loss*(1-0.2) + dis_loss*0.2
                    jointlosses.update(final_loss.item(), 1)
                if hasattr(self.args, 'reconstruct_loss') and self.args.reconstruct_loss:
                    final_loss = loss*(1-0.2) + dis_loss*0.2
                    jointlosses.update(final_loss.item(), 1)
                if hasattr(self.args, 'inter_props_cls') and self.args.inter_props_cls:
                    final_loss = loss*(1-0.2) + dis_loss*0.2
                    jointlosses.update(final_loss.item(), 1)

                final_loss.backward()
                self.model_opt.step()
                torch.cuda.empty_cache()
            ppls.update(ppl, len(batch[0]))
            acces.update(acc, len(batch[0]))
            losses.update(loss.item(), 1)
            batch_time.update(time.time() - end)
            end = time.time()
            print_info  = 'Epoch: %d (%d/%d) Data: %.4f | \
                Batch: %.4f | Loss: %.4f | \
                Acc: %.4f | ppl: %.4f' % (
                        self.cur_epoch,
                        i + 1,
                        self.args.num_iters,
                        data_time.val,
                        batch_time.val,
                        losses.avg*100,
                        acces.avg*100,
                        ppls.avg )
            self._logger.info(print_info)
            if hasattr(self.args, 'goal_joint_loss') and self.args.goal_joint_loss >= 0:
                print_info = 'DisLoss: %.4f | JointLoss: %.4f' % (dislosses.avg*100, jointlosses.avg*100)
                self._logger.info(print_info)
            if hasattr(self.args, 'joint_inter_emb') and self.args.joint_inter_emb:
                print_info = 'DisLoss: %.4f | JointLoss: %.4f' % (dislosses.avg*100, jointlosses.avg*100)
                self._logger.info(print_info)
            if hasattr(self.args, 'reconstruct_loss') and self.args.reconstruct_loss:
                print_info = 'DisLoss: %.4f | JointLoss: %.4f' % (dislosses.avg*100, jointlosses.avg*100)
                self._logger.info(print_info)
            if hasattr(self.args, 'inter_props_cls') and self.args.inter_props_cls:
                print_info = 'DisLoss: %.4f | JointLoss: %.4f' % (dislosses.avg*100, jointlosses.avg*100)
                self._logger.info(print_info)
            if self.args.manual_out and i % 1000 == 999:
                with open(self.args.log+'_p', 'a') as f:
                    f.write(print_info+'\n')
            if _epoch != loader.dataset.cur_epoch:
                break

    def train_from_loader_and_queue(self, loader, _train=True):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        acces = AverageMeter()
        acces_cons = AverageMeter()
        ppls = AverageMeter()
        ppls_cons = AverageMeter()

        if _train:
            self.model.train()
        else:
            self.model.eval()
        end = time.time()
        for i, real_batch in enumerate(loader):
            self.batch_size = len(real_batch[0])
            cons_batch = self.queue.get()
            data_time.update(time.time() - end)
            batch = [real_batch[j]+cons_batch[j] for j in range(len(real_batch))]
            if self.args.task == 'gen':
                loss, acc, ppl, acc_cons, ppl_cons = self.step_gen(batch)
            elif self.args.allneg:
                loss, acc, acc_cons, ppl, ppl_cons = self.step_allneg(batch)
            else:
                loss, acc, acc_cons, ppl, ppl_cons = self.step(batch)
            if _train:
                self.model_opt.zero_grad()
                loss.backward()
                self.model_opt.step()
                torch.cuda.empty_cache()
            ppls.update(ppl, len(real_batch[0]))
            ppls_cons.update(ppl_cons, len(cons_batch[0]))
            acces.update(acc, len(real_batch[0]))
            acces_cons.update(acc_cons, len(cons_batch[0]))
            losses.update(loss.item(), len(batch[0]))
            batch_time.update(time.time() - end)
            end = time.time()
            print_info  = 'Epoch: %d (%d/%d) Data: %.4f | Batch: %.4f | Loss: %.4f | \
                Acc: %.4f AccCons: %.4f ppl: %.4f pplCons: %.4f' % (
                        self.cur_epoch,
                        i + 1,
                        self.args.num_iters,
                        data_time.val,
                        batch_time.val,
                        losses.avg*100,
                        acces.avg*100,
                        acces_cons.avg*100,
                        ppls.avg,
                        ppls_cons.avg)
            self._logger.info(print_info)
            if self.args.manual_out and i % 1000 == 999:
                with open(self.args.log+'_p', 'a') as f:
                    f.write(print_info+'\n')

    def evaluate(self, loader=None):
        if self.args.task == 'pred':
            return self.evaluate_pred(loader)
        elif self.args.task == 'gen':
            return self.evaluate_gen(loader)
        else:
            print ('unknown task %s for evaluation' % (args.task))

    def evaluate_gen(self, loader=None):
        args = self.args
        if loader is None:
            loader = self.validloader
        batch_time = AverageMeter()
        data_time = AverageMeter()
        ppl = AverageMeter()
        acc = AverageMeter()
        log_softmax = torch.nn.LogSoftmax(dim=1)
        end = time.time()
        for i, batch in enumerate(loader):
            data_time.update(time.time() - end)
            score = self.model(batch)
            for j, tokens in enumerate(batch[-2]):
                if len(tokens) > self.args.max_len:
                    continue
                gt = torch.zeros(len(tokens)).long().to(args.device)
                gt[:-1] = tokens[1:]
                gt[-1] = self.config.encode['END_OF_SECTION']
                _score = score[j, :len(tokens)]
                acc.update(( _score.max(dim=1)[1]==gt).sum().item()/len(gt), 1)
                _score = log_softmax(_score)
                _ppl = torch.exp(_score.gather(1, gt.view(-1,1)).sum() / len(gt))

                ppl.update(_ppl.item(), 1)
            batch_time.update(time.time() - end)
            end = time.time()
            print_info  = '(%d/%d) Data: %.4f | \
                          Batch: %.4f | Perplexity %.4f Acc %.4f' % (
                                  loader.dataset.cnt,
                                  len(loader),
                                  data_time.val,
                                  batch_time.val,
                                  ppl.avg,
                                  acc.avg
                                  )
            self._logger.info(print_info)
            if self.args.manual_out and i % 1000 == 999:
                with open(self.args.log+'_p', 'a') as f:
                    f.write(print_info+'\n')
        self._logger.warning('Perplexity %.4f Acc %.4f', ppl.avg, acc.avg)
        if self.args.manual_out:
            with open(self.args.log+'_p', 'a') as f:
                f.write('Perplexity %.4f Acc %.4f\n' % ( ppl.avg, acc.avg))
        return ppl.avg

    def evaluate_pred(self, loader=None):
        args = self.args
        if loader is None:
            loader = self.validloader
        batch_time = AverageMeter()
        data_time = AverageMeter()
        acces_1 = AverageMeter()
        acces_5 = AverageMeter()
        acces_20 = AverageMeter()
        rank = AverageMeter()
        ppls = AverageMeter()
        num_props = AverageMeter()

        model = self.model
        if hasattr(args, 'training_stage') and args.training_stage > 2:
            model = self.model.pred_model
        if hasattr(args, 'goal_joint_loss') and args.goal_joint_loss > 0:
            model = self.model.pred_model
        if hasattr(args, 'joint_inter_emb') and args.joint_inter_emb:
            model = self.model.pred_model
        if hasattr(args, 'reconstruct_loss') and args.reconstruct_loss:
            model = self.model.pred_model
        if hasattr(args, 'inter_props_cls') and args.inter_props_cls:
            model = self.model.pred_model

        prop_embs = torch.zeros(len(loader.dataset.lm_inputs), args.nFeats*2 if args.bidirectional else args.nFeats).to(args.device)
        l = 0
        while l < len(loader.dataset.lm_inputs):
            r = min(len(loader.dataset.lm_inputs), l+args.batch_size)
            if args.thm_emb:
                tokens = list(range(l, r))
                trees = None
            else:
                tokens = [torch.LongTensor(loader.dataset.lm_inputs[i][0]).to(args.device) for i in range(l, r)]
                trees = [torch.Tensor(loader.dataset.lm_inputs[i][1]).to(args.device) for i in range(l, r)]
            prop_embs[l:r] = model.embed(tokens, trees, _type='p')
            l = r
        self._logger.info('Embed %d/%d props', l, len(loader.dataset.lm_inputs))
        end = time.time()

        for i, batch in enumerate(loader):
            data_time.update(time.time() - end)
            g_emb = model.embed(batch[0], batch[1], _type='g')

            prop_idx = batch[-2]
            true_prop = batch[-1]
            for g, p_idx, true_pos in zip(g_emb, prop_idx, true_prop):
                p_emb = prop_embs[p_idx]
                score = model.biln(g.view(1,-1), p_emb)
                ppl = np.exp((score[true_pos]-score.exp().sum().log()).item())
                _pos = (score[true_pos]<score).sum().item()
                ppls.update(ppl, 1)
                rank.update(1/(1+_pos), 1)
                num_props.update(len(p_idx), 1)
                if _pos==0:
                    acces_1.update(1,1)
                else:
                    acces_1.update(0,1)
                if _pos<5:
                    acces_5.update(1,1)
                else:
                    acces_5.update(0,1)
                if _pos<20:
                    acces_20.update(1,1)
                else:
                    acces_20.update(0,1)
                end = time.time()
            print_info  = '(%d/%d) Data: %.4f | \
                            Batch: %.4f | top1: %.4f | top5: %4f |\
                            top20: %.4f | rank: %.4f | ppl: %.4f' % (
                            i + 1,
                            len(loader),
                            data_time.val,
                            batch_time.val,
                            acces_1.avg*100,
                            acces_5.avg*100,
                            acces_20.avg*100,
                            rank.avg,
                            ppls.avg )
            self._logger.info(print_info)
            if self.args.manual_out and i % 1000 == 999:
                with open(self.args.log+'_p', 'a') as f:
                    f.write(print_info+'\n')
        self._logger.warning('top1: %.4f top5: %.4f top20: %.4f rank: %.4f ppl: %.4f',
                acces_1.avg*100, acces_5.avg*100, acces_20.avg*100, rank.avg, ppls.avg)
        return acces_5.avg

