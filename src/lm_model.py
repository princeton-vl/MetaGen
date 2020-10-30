import os
import torch
import numpy as np
import torch.nn.init as init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch_models import *
import beam
from data_utils import *
import nltk
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import MLE
from nltk.lm import Vocabulary

class LM_BaseModel(torch.nn.Module):
    def __init__(self, args, generator, config=None):
        super(LM_BaseModel, self).__init__()
        self.args = args
        self.config = config
        self.emb = Emb(args, config)
        self.generator = generator
        self.replacement_dict = self.config.lm.deterministic_replacement_dict_f(f=self.generator.all_f)

    def pad_list(self, tokens, trees):
        for i in range(len(tokens)):
            if type(tokens[i])==list:
                tokens[i] = torch.Tensor(tokens[i]).long().to(self.args.device)
                trees[i] = torch.Tensor(trees[i]).to(self.args.device)
        lengths = torch.LongTensor([len(seq) for seq in tokens]).to(self.args.device)
        tokens_tensor = torch.zeros(len(tokens), lengths.max()).to(self.args.device)
        trees_tensor = torch.zeros(len(tokens), lengths.max(), 4).to(self.args.device)
        for i, l in enumerate(lengths):
            tokens_tensor[i, :l] = tokens[i]
            trees_tensor[i, :l, :] = trees[i]
        if lengths.max() > self.args.max_len:
            tokens_tensor = tokens_tensor[:, :self.args.max_len]
            trees_tensor = trees_tensor[:, :self.args.max_len, :]
            lengths[lengths > self.args.max_len] = self.args.max_len
        lengths, perm_idx = lengths.sort(0, descending=True)
        tokens_tensor = tokens_tensor[perm_idx].transpose(0,1)
        trees_tensor = trees_tensor[perm_idx].transpose(0,1) # seq, batch, feat
        return tokens_tensor, trees_tensor, perm_idx, lengths

    def embed(self, tokens, trees):
        padded_tokens, padded_trees, perm_idx, lengths = self.pad_list(tokens, trees)
        padded_feats = self.emb(padded_tokens, padded_trees)
        packed_feats = pack_padded_sequence(padded_feats, lengths.cpu().numpy())
        return packed_feats, perm_idx

    def load(self, _path):
        data = torch.load(_path)
        self.load_state_dict(data['models'])
        print ('Load lm models from %s' % (_path))

    def get_hyps(self, expr):
        if type(expr) == proposition:
            return [h.tree for h in expr.hyps if h.type=='e']
        elif type(expr) == proof_step:
            return [h.tree for h in expr.context.hyps if h.type == 'e']
        else:
            return [self.generator.expressions_list[i].tree for i in expr.hyps]

    def encode_expr(self, exprs):
        pass

    def forward(self, batch):
        #take a batch of exprs and generator as input
        pass

class LM_HyoAssGen(LM_BaseModel):
    def __init__(self, args, generator, config=None):
        LM_BaseModel.__init__(self, args, generator, config)
        self.gru = torch.nn.GRU(
                args.nFeats if args.no_use_tree else args.nFeats+4,
                args.nFeats, args.gru_depth, dropout=args.dropout )
        self.l1 = torch.nn.Linear(args.nFeats, args.nFeats)
        self.l2 = torch.nn.Linear(args.nFeats, args.vocab_size)

    def encode_expr(self, exprs):
        # encode expressions during inference time using generator
        tokens = []
        trees = []
        for e in exprs:
            token, tree = self.generator.encode_expr(
                    e.tree, 
                    self.get_hyps(e),
                    self.replacement_dict)
            tokens.append(token)
            trees.append(tree)
        return tokens, trees    

    def forward(self, batch):
        tokens, trees = batch[-2:]
        packed_feats, perm_idx = self.embed(tokens, trees)
        packed_output, h = self.gru(packed_feats)
        padded_output, _ = pad_packed_sequence(packed_output)
        padded_output = padded_output.view(-1, self.args.nFeats)
        x = self.l1(padded_output)
        x = torch.nn.functional.relu(x)
        score = self.l2(x).view(-1, len(perm_idx), self.args.vocab_size)
        score = score.transpose(0,1)
        _, unperm_idx = perm_idx.sort(0)
        score = score[unperm_idx]
        return score.contiguous()

class LM_AssGenCondHyo(LM_BaseModel):
    def __init__(self, args, generator, config=None):
        LM_BaseModel.__init__(self, args, generator, config)
        self.gru = torch.nn.GRU(
                args.nFeats if args.no_use_tree else args.nFeats+4,
                args.nFeats, args.gru_depth, dropout=args.dropout,
                bidirectional=args.bidirectional)
        self.decoder = Decoder(args)

    def encode_expr(self, exprs):
        in_tokens, in_trees, out_tokens, out_trees = [],[],[],[]
        for e in exprs:
            statement = e.tree.copy().replace_values(self.replacement_dict) 
            hyps = self.get_hyps(e)
            hyps = [h.copy().replace_values(self.replacement_dict) for h in hyps]
            statement_graph_structure = TreeInformation([statement],
                    start_symbol=None, intermediate_symbol='END_OF_HYP',
                    end_symbol='END_OF_SECTION')
            hyps_graph_structure = TreeInformation(hyps,
                    start_symbol=None, intermediate_symbol='END_OF_HYP',
                    end_symbol='END_OF_SECTION')   
            in_string, in_tree = merge_graph_structures_new([hyps_graph_structure])
            out_string, out_tree = merge_graph_structures_new([statement_graph_structure])
            in_tokens.append([self.config.encode[t] for t in in_string])
            out_tokens.append([self.config.encode[t] for t in out_string])
            in_trees.append(in_tree)
            out_trees.append(out_tree)
        return in_tokens, in_trees, out_tokens, out_trees 

    def encode(self, packed_feats, perm_idx):
        packed_output, h = self.gru(packed_feats)
        output, _ = pad_packed_sequence(packed_output)
        _, unperm_idx = perm_idx.sort(0)
        if self.args.bidirectional:
            h = h.view(-1, 2, len(perm_idx), self.args.nFeats)
            h = torch.cat((h[:,0,:,:], h[:,1,:,:]), dim=2)
        out_h = h[:,unperm_idx,:]
        return output, out_h
            
    def forward(self, batch):
        in_tokens, in_trees, out_tokens, out_trees = batch
        in_packed_feats, in_perm_idx = self.embed(in_tokens, in_trees) 
        out_packed_feats, out_perm_idx = self.embed(out_tokens, out_trees)
        _, h = self.encode(in_packed_feats, in_perm_idx)
        score = self.decoder(out_packed_feats, h, out_perm_idx)
        return score.contiguous()

class LM_nGram(LM_BaseModel):
    def __init__(self, args, generator, config=None):
        LM_BaseModel.__init__(self, args, generator, config)
        self.ngram = MLE(2) 
    
    def fit(self, steps):
        tokens = [step.tree.list() for step in steps]
        train_data = [nltk.bigrams(t,  pad_right=True, pad_left=True, left_pad_symbol="<s>", right_pad_symbol="</s>") for t in tokens]
        words = [word for sent in tokens for word in sent]
        words.extend(["<s>", "</s>"])
        padded_vocab = Vocabulary(words)
        self.ngram.fit(train_data, padded_vocab)

    def forward(self, steps):
        probs = []
        for step in steps:
            words = step.tree.list()
            ngrams = nltk.bigrams(words,  pad_right=True, pad_left=True, left_pad_symbol="<s>", right_pad_symbol="</s>")
            prob = [self.ngram.score(ngram[-1], ngram[:-1]) for ngram in ngrams]
            probs.append(sum(prob)/len(prob))
            if len(probs) % 100 == 0:
                print (len(probs), sum(probs)/len(probs))
        return probs

