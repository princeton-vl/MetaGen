import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.init as init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from collections import OrderedDict
from torch.nn.parameter import Parameter
from torch_scatter import segment_csr

class FakeVariable(Parameter):
    data = None  # Used to override some tricks in pytorch
    grad = None  # Used to override some tricks in pytorch

    def __init__(self, var):
        self.data = var  #.share_memory_()
        self.grad = None

class FakeModule:
    def __init__(self, module, device):
        params, buffers = FakeModule.init_state_dict(module)
        self.params = params
        self.buffers = buffers
        self.device = device

    @staticmethod
    def init_state_dict(module, params=None, buffers=None, prefix=''):
        if params is None:
            params = OrderedDict()
        if buffers is None:
            buffers = OrderedDict()
        for name, param in module._parameters.items():
            if param is not None:
                params[prefix + name] = FakeVariable(param.data)
        for name, buf in module._buffers.items():
            if buf is not None:
                buffers[prefix + name] = buf
        for name, smodule in module._modules.items():
            if smodule is not None:
                FakeModule.init_state_dict(smodule, params, buffers,
                                           prefix + name + '.')
        return params, buffers

    def state_dict(self):
        params = {name: param.data for name, param in self.params.items()}
        buffers = {name: buf for name, buf in self.buffers.items()}
        params.update(buffers)
        return params

    def load_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                raise KeyError('unexpected key "{}" in state_dict'.format(name))
            if isinstance(param, FakeVariable):
                param = param.data
            own_state[name].copy_(param.to(self.device))

        missing = set(own_state.keys()) - set(state_dict.keys())
        if len(missing) > 0:
            raise KeyError('missing keys in state_dict: "{}"'.format(missing))

    def parameters(self):
        for param in self.params.values():
            yield param

    def named_parameters(self):
        for name, param in self.params.items():
            yield name, param

    def named_buffers(self):
        for name, buf in self.buffers.items():
            yield name, buf

class Emb(nn.Module):
    def __init__(self, args, config):
        super(Emb, self).__init__()
        self.in_dim = args.nFeats
        self.structure_data_scaling = 0.1
        self.use_tree = not args.no_use_tree
        self.emb = nn.Embedding(args.vocab_size, self.in_dim, sparse=False)
        init.kaiming_normal_(self.emb.weight)
        if args.thm_emb:
            self.dim = args.nFeats*2 if args.bidirectional else args.nFeats
            self.props = config.lm.database.propositions
            self.num_thms = len(config.lm.database.propositions)
            self.thm_emb = torch.zeros(self.num_thms, self.dim).to(args.device)
            init.kaiming_normal_(self.thm_emb)
            self.thm_emb.requires_grad = True
            print ('build theorem embeddings for %d props' % (self.num_thms))

    def forward(self, tokens, tree=None):
        length = [len(t) for t in tokens]
        x = self.emb(tokens.long())
        if self.use_tree:
            x = torch.cat((x, tree*self.structure_data_scaling), 2)
        return x

class Dot(nn.Module):
    def __init__(self, args, dim=None, outdim=1):
        super(Dot, self).__init__()
        self.num_neg = args.negative_samples
        if dim is None:
            if args.enlarge > 0:
                self.dim = args.nFeats*enlarge*2
            else:
                self.dim = args.nFeats*2 if args.bidirectional else args.nFeats
        else:
            if args.enlarge > 0:
                self.dim = dim*args.enlarge*2
            else:
                self.dim = dim
        self.b = nn.Bilinear(self.dim, self.dim, outdim)
        init.kaiming_normal_(self.b.weight)
    def forward(self, g_vec, p_vec, _aligned=False):
        if not _aligned:
            x = g_vec.repeat(1,p_vec.shape[0]//g_vec.shape[0]).view(len(p_vec), -1)
        else:
            x = g_vec
        x = self.b(x, p_vec)
        return x

class CatLinear(nn.Module):
    def __init__(self, args, dim=None):
        super(CatLinear, self).__init__()
        self.num_neg = args.negative_samples
        if dim is None:
            self.dim = args.nFeats*2 if args.bidirectional else args.nFeats
        else:
            self.dim = dim
        self.l1 = nn.Linear(self.dim*2, self.dim)
        self.l2 = nn.Linear(self.dim, 1)
        self.bn = nn.BatchNorm1d(self.dim)
        init.kaiming_normal_(self.l1.weight)
        init.kaiming_normal_(self.l2.weight)
    def forward(self, g_vec, p_vec):
        y = g_vec.repeat(1, p_vec.shape[0]//g_vec.shape[0]).view(-1, self.dim)
        x = torch.cat( (y , p_vec) , 1 )
        x = self.l1(x)
        x = self.bn(x)
        x = nn.functional.relu_(x)
        x = self.l2(x)
        return x

classifier_dict = {
        'bilinear' : Dot,
        'linear' : CatLinear
        }

class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        self.args = args
        self.gru = torch.nn.GRU(args.nFeats if args.no_use_tree else args.nFeats+4,
                args.nFeats, args.gru_depth, dropout=args.dropout)
        self.middle = torch.nn.Linear(args.nFeats*2 if args.bidirectional else args.nFeats,
                args.nFeats)
        init.kaiming_normal_(self.middle.weight)
        self.relu = torch.nn.functional.leaky_relu_
        self.l1 = torch.nn.Linear(args.nFeats, args.nFeats)
        self.l2 = torch.nn.Linear(args.nFeats, args.vocab_size)

    def to_middle(self, encoder_h):
        h = self.middle(encoder_h.view(-1, encoder_h.shape[2])).view(self.args.gru_depth, -1, self.args.nFeats)
        h = self.relu(h)
        return h

    def forward(self, packed_feats, encoder_h, perm_idx):
        h = self.to_middle(encoder_h)
        h = h[:, perm_idx, :]
        packed_output, h = self.gru(packed_feats, h)
        padded_output, _ = pad_packed_sequence(packed_output)
        padded_output = padded_output.view(-1, self.args.nFeats)
        x = self.l1(padded_output)
        x = self.relu(x)
        score = self.l2(x).view(-1, len(perm_idx), self.args.vocab_size)
        score = score.transpose(0,1)
        _, unperm_idx = perm_idx.sort(0)
        score = score[unperm_idx]
        return score.contiguous()

class TFIDF:
    def __init__(self, args, config):
        self.args = args
        self.config = config
        self.num_doc = 0
        self.df = torch.zeros(args.vocab_size).to(args.device)+1e-8
        self.zeros = torch.zeros(args.vocab_size).to(args.device)

    def eval(self):
        pass

    def add_doc(self, tokens):
        count = {}
        for t in tokens:
            count[t] = 1
        for t in count:
            self.df[t] += 1
        self.num_doc += 1

    def embed(self, tokens, trees=None, _type=None):
        self.zeros.fill_(0)
        for i in tokens:
            self.zeros[i] += 1
        return self.zeros*self.num_doc/self.df/len(tokens)

    def biln(self, emb1, emb2):
        return torch.nn.functional.cosine_similarity(emb1, emb2)

class PredModel(nn.Module):
    def __init__(self, args, config, models=None, outdim=1):
        super(PredModel, self).__init__()
        self.args = args
        if hasattr(args, 'thm_emb') == False:
            args.thm_emb = False
        if hasattr(args, 'enlarge') == False:
            args.enlarge = 0
        self.config = config
        if models is not None:
            self.emb = models['emb']
            self.g_encoder = models['g_gru']
            self.p_encoder = models['p_gru']
            self.biln = models['biln']
        else:
            self.emb = Emb(args, config)
            self.g_encoder = torch.nn.GRU(
                    args.nFeats if args.no_use_tree else args.nFeats+4,
                    args.nFeats, args.gru_depth, dropout=args.dropout,
                    bidirectional=args.bidirectional)
            self.p_encoder = torch.nn.GRU(
                    args.nFeats if args.no_use_tree else args.nFeats+4,
                    args.nFeats, args.gru_depth, dropout=args.dropout,
                    bidirectional=args.bidirectional)
            if hasattr(args, 'cat') and args.cat:
                self.biln = CatLinear(args)
            else:
                self.biln = Dot(args, outdim=outdim)

    def load(self, _path):
        data = torch.load(_path, map_location='cpu' if self.args.cpu else 'cuda:0')
        if type(data['models']) == type({}):
            self.emb.load_state_dict(data['models']['emb'])
            self.g_encoder.load_state_dict(data['models']['g_gru'])
            self.p_encoder.load_state_dict(data['models']['p_gru'])
            self.biln.load_state_dict(data['models']['biln'])
        else:
            self.load_state_dict(data['models'])
        print ('Load pred model from %s' % (_path))

    def pad_list(self, _tokens, _trees):
        tokens = [torch.Tensor(x).long().to(self.args.device) if type(x)==list else x for x in _tokens]
        trees = [torch.Tensor(x).to(self.args.device) if type(x)==list else x for x in _trees]

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


    def embed(self, tokens, trees, _type):
        if self.args.thm_emb and _type=='p':
            feats = self.emb.thm_emb[torch.Tensor(tokens).long().to(self.args.device), :]
            return feats
        padded_tokens, padded_trees, perm_idx, lengths = self.pad_list(tokens, trees)
        padded_feats = self.emb(padded_tokens, padded_trees)
        packed_feats = pack_padded_sequence(padded_feats, lengths.cpu().numpy())
        if _type == 'g':
            encoder = self.g_encoder
        elif _type == 'p':
            encoder = self.p_encoder
        elif _type == 'is':
            if self.args.goal_debug:
                return torch.zeros(len(tokens), 2*self.args.nFeats).to(self.args.device)
            encoder = self.is_encoder
        else:
            encoder = self.ip_encoder
        packed_output, h = encoder(packed_feats)
        output = torch.cat((h[-2], h[-1]), 1) if self.args.bidirectional else h[-1]
        _, unperm_idx = perm_idx.sort(0)
        output = output[unperm_idx]
        return output


    def forward(self, batch, reshape=True):
        g_tokens, g_trees, p_tokens, p_trees = batch
        g_vec = self.embed(g_tokens, g_trees, _type='g')
        p_vec = self.embed(p_tokens, p_trees, _type='p')
        if reshape:
            score = self.biln(g_vec, p_vec).view(-1, self.args.negative_samples+1)
        else:
            score = self.biln(g_vec, p_vec)
        return score

class LModel(nn.Module):
    def __init__(self, args, config=None):
        super(LModel, self).__init__()
        self.args = args
        self.config = config
        self.emb = Emb(args, config)
        self.gru = torch.nn.GRU(
                args.nFeats if args.no_use_tree else args.nFeats+4,
                args.nFeats, args.gru_depth, dropout=args.dropout )
        self.l1 = torch.nn.Linear(args.nFeats, args.nFeats)
        self.l2 = torch.nn.Linear(args.nFeats, args.vocab_size)

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

    def logits_and_next_state(self, h, token, tree):
        packed_feats, perm_idx = self.embed([token], [tree])
        packed_output, out_h = self.gru(packed_feats, h)
        padded_output, _ = pad_packed_sequence(packed_output)
        padded_output = padded_output.view(-1, self.args.nFeats)
        x = self.l1(padded_output)
        x = torch.nn.functional.relu(x)
        logits = self.l2(x).view(-1)
        logits =  torch.nn.functional.log_softmax(logits, dim=0)
        return out_h, logits

class GenModel2(nn.Module):
    def __init__(self, args, config):
        super(GenModel2, self).__init__()
        self.args = args
        if hasattr(args, 'thm_emb') == False:
            args.thm_emb = False
        self.config = config
        self.emb = Emb(args, config)
        self.gru1 = torch.nn.GRU(
                args.nFeats if args.no_use_tree else args.nFeats+4,
                args.nFeats, args.gru_depth, dropout=args.dropout,
                bidirectional=args.bidirectional)
        self.gru2 = torch.nn.GRU(
                args.nFeats if args.no_use_tree else args.nFeats+4,
                args.nFeats, args.gru_depth, dropout=args.dropout,
                bidirectional=args.bidirectional)
        self.decoder = Decoder(args)

    def load(self, _path):
        data = torch.load(_path, map_location='cpu' if self.args.cpu else 'cuda:0')
        if type(data['models']) == type({}):
            self.emb.load_state_dict(data['models']['emb'])
            self.gru1.load_state_dict(data['models']['e_gru'])
            self.gru2.load_state_dict(data['models']['e_gru'])
            self.decoder.load_state_dict(data['models']['decoder'])
        else:
            self.load_state_dict(data['models'])
        print ('Load gen models from %s' % (_path))

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

    def encode(self, packed_feats, perm_idx, _type='k', h_in=None, cat=True): # packed_feats is in perm_idx, h_in is in the originl order
        if _type=='k':
            encoder = self.gru1
        else:
            encoder = self.gru2
        if h_in is not None:
            packed_output, h = encoder(packed_feats, h_in[:,perm_idx,:])
        else:
            packed_output, h = encoder(packed_feats)
        output, _ = pad_packed_sequence(packed_output)
        _, unperm_idx = perm_idx.sort(0)
        if self.args.bidirectional and cat:
            h = h.view(-1, 2, len(perm_idx), self.args.nFeats)
            h = torch.cat((h[:,0,:,:], h[:,1,:,:]), dim=2)
        out_h = h[:,unperm_idx,:]
        return output, out_h

    def in_forward(self, batch):
        known_tokens, known_trees, to_prove_tokens, to_prove_trees, out_tokens, out_trees = batch
        known_packed_feats, known_perm_idx = self.embed(known_tokens, known_trees)
        to_prove_packed_feats, to_prove_perm_idx = self.embed(to_prove_tokens, to_prove_trees)
        _, known_h = self.encode(known_packed_feats, known_perm_idx, cat=False, _type='k')
        _, to_prove_h = self.encode(to_prove_packed_feats, to_prove_perm_idx, h_in=known_h, _type='p', cat=True)
        return to_prove_h

    def forward(self, batch):
        h = self.in_forward(batch)
        known_tokens, known_trees, to_prove_tokens, to_prove_trees, out_tokens, out_trees = batch
        out_packed_feats, out_perm_idx = self.embed(out_tokens, out_trees)
        score = self.decoder(out_packed_feats, h, out_perm_idx)
        return score

    def logits_and_next_state(self, h, token, tree):
        packed_feats, perm_idx = self.embed([token], [tree])
        packed_output, out_h = self.decoder.gru(packed_feats, h)
        padded_output, _ = pad_packed_sequence(packed_output)
        padded_output = padded_output.view(-1, self.args.nFeats)
        x = self.decoder.l1(padded_output)
        x = self.decoder.relu(x)
        logits = self.decoder.l2(x).view(-1)
        logits =  torch.nn.functional.log_softmax(logits, dim=0)
        return out_h, logits


class GenModel(nn.Module):
    def __init__(self, args, config, models=None):
        super(GenModel, self).__init__()
        self.args = args
        if hasattr(args, 'thm_emb') == False:
            args.thm_emb = False
        self.config = config
        if models is not None:
            self.emb = models['emb']
            self.encoder = models['e_gru']
            self.decoder = models['decoder']
        else:
            self.emb = Emb(args, config)
            self.encoder = torch.nn.GRU(
                    args.nFeats if args.no_use_tree else args.nFeats+4,
                    args.nFeats, args.gru_depth, dropout=args.dropout,
                    bidirectional=args.bidirectional)
            self.decoder = Decoder(args)
    def load(self, _path):
        data = torch.load(_path, map_location='cpu' if self.args.cpu else 'cuda:0')
        if type(data['models']) == type({}):
            self.emb.load_state_dict(data['models']['emb'])
            self.encoder.load_state_dict(data['models']['e_gru'])
            self.decoder.load_state_dict(data['models']['decoder'])
        else:
            self.load_state_dict(data['models'])
        print ('Load gen models from %s' % (_path))
        return

    def pad_list(self, tokens, trees):
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

    def encode(self, packed_feats, perm_idx):
        packed_output, h = self.encoder(packed_feats)
        output, _ = pad_packed_sequence(packed_output)
        _, unperm_idx = perm_idx.sort(0)
        if self.args.bidirectional:
            h = h.view(-1, 2, len(perm_idx), self.args.nFeats)
            h = torch.cat((h[:,0,:,:], h[:,1,:,:]), dim=2)
        out_h = h[:,unperm_idx,:]
        output = output[:,unperm_idx,:]
        return output, out_h

    def forward(self, batch):
        in_tokens, in_trees, out_tokens, out_trees = batch
        in_packed_feats, in_perm_idx = self.embed(in_tokens, in_trees)
        out_packed_feats, out_perm_idx = self.embed(out_tokens, out_trees)
        encoder_out, encoder_h = self.encode(in_packed_feats, in_perm_idx)
        score = self.decoder(out_packed_feats, encoder_h, out_perm_idx)
        return score

    def logits_and_next_state(self, h, token, tree):
        packed_feats, _ = self.embed([token], [tree])
        packed_output, out_h = self.decoder.gru(packed_feats, h)
        padded_output, _ = pad_packed_sequence(packed_output)
        padded_output = padded_output.view(-1, self.args.nFeats)
        x = self.decoder.l1(padded_output)
        x = self.decoder.relu(x)
        logits = self.decoder.l2(x).view(-1)
        logits =  torch.nn.functional.log_softmax(logits, dim=0)
        return out_h, logits

class Payout(nn.Module):
    def __init__(self, args, config):
        super(Payout, self).__init__()
        self.args = args
        if hasattr(args, 'thm_emb') == False:
            args.thm_emb = False
        self.emb = Emb(args, config)
        self.gru = torch.nn.GRU(args.nFeats if args.no_use_tree else args.nFeats+4,
                args.nFeats, args.gru_depth, dropout=args.dropout,
                bidirectional=args.bidirectional)
        self.relu = torch.nn.functional.leaky_relu_
        self.l1 = torch.nn.Linear(args.nFeats*2 if args.bidirectional else args.nFeats, args.nFeats)
        self.l2 = torch.nn.Linear(args.nFeats, 1)

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

    def load(self, _path):
        data = torch.load(_path, map_location='cpu' if self.args.cpu else 'cuda:0')
        self.load_state_dict(data['models'])
        print ('Load payout model from %s' % (_path))
        return

    def embed(self, tokens, trees):
        padded_tokens, padded_trees, perm_idx, lengths = self.pad_list(tokens, trees)
        padded_feats = self.emb(padded_tokens, padded_trees)
        packed_feats = pack_padded_sequence(padded_feats, lengths.cpu().numpy())
        return packed_feats, perm_idx

    def forward(self, batch):
        tokens, trees, _ = batch
        packed_feats, perm_idx = self.embed(tokens, trees)
        packed_output, h = self.gru(packed_feats)
        _, unperm_idx = perm_idx.sort(0)
        h = torch.cat((h[-2], h[-1]), 1) if self.args.bidirectional else h[-1]
        out_h = h[unperm_idx,:]
        x = self.l1(out_h)
        x = self.relu(x)
        x = self.l2(x)
        score = torch.nn.functional.sigmoid(x)
        return score.view(-1)

