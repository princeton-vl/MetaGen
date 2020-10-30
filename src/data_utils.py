import pickle
import os
import sys
import random
from data_utils5 import *
import numpy as np
from models import *
try:
    import torch
except:
    pass

ISET = False
datapath = '../data'

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.avg_all = 0
        self.sum_all = 0
        self.count_all = 0
    def step(self):
        self.sum_all += self.sum
        self.count_all += self.count
        self.avg_all = self.sum_all/self.count_all
        self.sum = 0
        self.count = 0
        self.avg = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count

def adjust_lr(opt, args):
    for param_group in opt.param_groups:
        param_group['lr'] *= args.lr_decay

def save_model(aux, args, models, opt=None):
    savedata = {}
    savedata['aux'] = aux
    savedata['args'] = args
    savedata['args'].device = None
    if type(models) == type({}):
        savedata['models'] = {}
        for k,v in models.items():
            if type(v) == list:
                savedata[k] = []
                for vv in v:
                    savedata[k].append(vv.state_dict())
            else:
                savedata['models'][k] = v.state_dict()
    else:
        savedata['models'] = models.state_dict()
    if opt is not None:
        savedata['opt'] = opt.state_dict()
    out_path = args.output + '_' + str(aux['epoch']) + '_' + str(aux['cur_iter'])
    torch.save(savedata, out_path)
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def save_language_model(lm, suffix=''):
    def save_steps(proof_steps, outpath):
        for ps in proof_steps:
            ps.context = None
            ps.prop = None
        cnt = 0
        if os.path.exists(outpath) == False:
            os.mkdir(outpath)
        while cnt*50000 < len(proof_steps):
            with open(os.path.join(outpath, str(cnt)), 'wb') as f:
                pickle.dump(proof_steps[cnt*50000:min(len(proof_steps), (cnt+1)*50000)], f)
            cnt += 1
        print ('save %d proof steps into %s' % (len(proof_steps), outpath))
    np.random.seed()
    np.random.shuffle(lm.training_proof_steps)
    for i, ps in enumerate(lm.training_proof_steps):
        ps.pos_in_proof_list = ('train', i)
    for i, ps in enumerate(lm.validation_proof_steps):
        ps.pos_in_proof_list = ('valid', i)
    for i, ps in enumerate(lm.test_proof_steps):
        ps.pos_in_proof_list = ('test', i)
    for p in lm.database.propositions_list:
        p.entails_proof_steps_pos = [None]*len(p.entails_proof_steps)
        for i, ps in enumerate(p.entails_proof_steps):
            p.entails_proof_steps_pos[i] = ps.pos_in_proof_list
            ps.pos_in_context = i
    save_steps(lm.training_proof_steps, '../data%s/train'%(suffix))
    save_steps(lm.validation_proof_steps, '../data%s/valid'%(suffix))
    save_steps(lm.test_proof_steps, '../data%s/test'%(suffix))

    lm.training_proof_steps = None
    lm.validation_proof_steps = None
    lm.test_proof_steps = None
    lm.all_proof_steps = None
    for p in lm.database.propositions_list:
        p.entails_proof_steps = None
    with open('../data%s/lm'%(suffix), 'wb') as f:
        pickle.dump(lm, f)

def save_exprs(exprs, lm, fpath):
    for e in exprs:
        e.prop_label = e.prop if type(e.prop) == int else e.prop.label
        e.prop = None
    torch.save(exprs, fpath)
    for e in exprs:
        e.prop = e.prop_label if type(e.prop_label) == int else lm.database.propositions[e.prop_label]

def load_exprs(fpath, lm):
    exprs = torch.load(fpath)
    for e in exprs:
        e.prop = e.prop_label if type(e.prop_label) == int else lm.database.propositions[e.prop_label]
    return exprs

def build_language_model(n=100000, new=True, _all=False, require_steps=True, iset=False):
    lm_all = load_language_model(_all=_all, new=new)
    database = lm_all.database
    n = min(n, len(database.propositions_list))
    database.propositions_list = database.propositions_list[:n]
    database.propositions = {k:v for k,v in database.propositions.items() if v.number < n}
    database.non_entails_axioms = {k:v for k,v in database.non_entails_axioms.items() if v.number < n}
    if require_steps:
        for i,p in enumerate(database.propositions_list):
            if p.type == 'p':
                database.verify(p)
            if i % 10 ==0:
                print ("\rpropositions: %d"%(i), end="")
    language_model = LanguageModel(database, _old=(not new))
    if _all:
        datapath = '../data_iset' if iset else '../data'
        with open(os.path.join('props_encode'), 'rb') as f:
            inputs = pickle.load(f)
        return language_model, inputs
    return language_model

def load_language_model(_all=False, new=False, graph=False, cpu=False, iset=False):
    lmpath = '../data/lm' if new==False else '../data/lm_gen'
    _datapath = '../data'
    if ISET or iset:
        lmpath = '../data_iset/lm'
        _datapath = '../data_iset'
    with open(lmpath, 'rb') as f:
        lm = pickle.load(f)
    if _all:
        if graph:
            if cpu:
                inputs = torch.load(os.path.join(_datapath, 'props_graph'), map_location='cpu')
            else:
                inputs = torch.load(os.path.join(_datapath, 'props_graph'))
        else:
            with open(os.path.join(_datapath, 'props_encode'), 'rb') as f:
                inputs = pickle.load(f)
        return lm, inputs
    return lm

def load_proof_steps(fpath, lm=None, _all=False, graph=False):
    if lm is None:
        lm = load_language_model()
    with open(fpath, 'rb') as f:
        proof_steps = pickle.load(f)
    for ps in proof_steps:
        ps.context = lm.database.propositions[ps.context_label]
        ps.prop = lm.database.propositions[ps.prop_label]
    if _all:
        with open(fpath+'_visible_props', 'rb') as f:
            props = pickle.load(f)
        if graph:
            inputs = torch.load(fpath+'_graph')
        else:
            with open(fpath+'_encode', 'rb') as f:
                inputs = pickle.load(f)
        return proof_steps, inputs, props
    return proof_steps

def load_proof_steps_into_lm(lm, splits=['train', 'valid', 'test'], data_sampling=1, iset=False, dpath=None):
    if dpath is None:
        dpath = '../data_iset' if iset else '../data'
    if data_sampling==1:
        props = lm.database.propositions.keys()
    else:
        props = {}
        num = int(data_sampling * 10)
        for i, p in enumerate(lm.training_propositions):
            if i % 10 < num:
                props[p.label] = p
    for p in lm.database.propositions_list:
        p.entails_proof_steps = [None]*len(p.entails_proof_steps_pos)
    for split in splits:
        fn = os.path.join(dpath, split)
        fl = [i for i in os.listdir(fn) if i.isdigit()]
        for f in fl:
            steps = load_proof_steps(os.path.join(fn, f), lm)
            print (split, f, len(steps))
            for ps in steps:
                if ps.context_label in props:
                    ps.context.entails_proof_steps[ps.pos_in_context] = ps
    return

def load_gen_proof_steps(fpath, lm=None, _all=False):
    if lm is None:
        lm = load_language_model()
    with open(fpath, 'rb') as f:
        proof_steps = pickle.load(f)
    steps = []
    for ps in proof_steps:
        ps.context = lm.database.propositions[ps.context_label]
        ps.prop = lm.database.propositions[ps.prop_label]
        if ps.prop.unconstrained_arity()>0:
            steps.append(ps)
    if _all:
        with open(fpath+'_gen_encode', 'rb') as f:
            inputs = pickle.load(f)
        return steps, inputs
    return steps

def get_props(fpath, lm = None):
    if lm is None:
        lm = load_language_model()
    proof_steps = load_proof_steps(fpath, lm)
    visible_props = []
    for proof_step in proof_steps:
        labels = lm.searcher.search(proof_step.tree, proof_step.context,
                max_proposition=proof_step.context.number, vclass='|-')
        visible_props.append(labels)
    with open(fpath+'_visible_props', 'wb') as f:
        pickle.dump(visible_props, f)

def merge_graph_structures_new(gs_list):
    out_string = []
    out_parents = []
    out_left_siblings = []
    out_right_siblings = []
    out_depth = []
    out_parent_arity = []
    out_leaf_position = []
    out_arity = []

    for gs in gs_list:
        out_string += gs.string
        out_depth += gs.depth
        out_parent_arity += gs.parent_arity
        out_leaf_position += gs.leaf_position
        out_arity += gs.arity

    return out_string, list(zip(out_depth, out_parent_arity, out_leaf_position, out_arity))

def get_config(lm, graph=False):
    config = DefaultConfig(lm)
    return config

def get_graph_vocab(lm, old_vocab=False):
    config = get_config(lm)
    vocab = {}
    for k,v in config.encode.items():
        if 'Var' not in k:
            vocab[k] = len(vocab)
    vocab['WFFVAR'] = len(vocab)
    vocab['SETVAR'] = len(vocab)
    vocab['CLASSVAR'] = len(vocab)

    if not old_vocab:
        vocab['UCWFFVAR'] = len(vocab)
        vocab['UCSETVAR'] = len(vocab)
        vocab['UCCLASSVAR'] = len(vocab)
        vocab['NOHYPS'] = len(vocab)

    return vocab

def encode_proof_step(statement, f, hyps, lm, config):
    random_replacement_dict = lm.random_replacement_dict_f(f=f)
    statement = statement.copy().replace_values(random_replacement_dict)
    hyps = [h.tree.copy().replace_values(random_replacement_dict) for h in hyps if h.type=='e']

    statement_graph_structure = TreeInformation([statement],
            start_symbol=None, intermediate_symbol='END_OF_HYP',
            end_symbol='END_OF_SECTION')
    hyps_graph_structure = TreeInformation(hyps,
            start_symbol=None, intermediate_symbol='END_OF_HYP',
            end_symbol='END_OF_SECTION')
    in_string, structure_data = merge_graph_structures_new([statement_graph_structure, hyps_graph_structure])
    tokens = [config.encode[t] for t in in_string]
    return tokens, structure_data

def encode_proof_steps(fpath):
    lm = load_language_model()
    proof_steps = load_proof_steps(fpath, lm=lm)
    config = get_config(lm)
    inputs = []
    for ps in proof_steps:
        inputs.append(encode_proof_step(ps.tree, ps.context.f, ps.context.hyps, lm, config))
    with open(fpath+'_encode', 'wb') as f:
        pickle.dump(inputs, f)

def encode_gen_step(proof_step, lm, config):
    tree = proof_step.tree
    prop = proof_step.prop
    context = proof_step.context
    unconstrained_variables = prop.unconstrained_variables
    uv_dict = {var:'UC'+var for var in unconstrained_variables}
    out_trees = [t.copy() for t in proof_step.unconstrained] # ground truth trees
    fit = prop_applies_to_statement(proof_step.tree, prop, proof_step.context)
    to_prove_trees = [hyp.tree.copy().replace(fit).replace_values(uv_dict)
            for hyp in prop.hyps if hyp.type == 'e']
    known_trees = [hyp.tree.copy()
            for hyp in proof_step.context.hyps if hyp.type == 'e']
    random_replacement_dict = lm.random_replacement_dict_f(f=context.f)
    for tree in to_prove_trees:
        tree.replace_values(random_replacement_dict)
    for tree in known_trees:
        tree.replace_values(random_replacement_dict)
    for tree in out_trees:
        tree.replace_values(random_replacement_dict)
    known_graph_structure = TreeInformation(known_trees,
            start_symbol=None, intermediate_symbol='END_OF_HYP',
            end_symbol='END_OF_SECTION')
    to_prove_graph_structure = TreeInformation(to_prove_trees,
            start_symbol=None, intermediate_symbol='END_OF_HYP',
            end_symbol='END_OF_SECTION')
    out_graph_structure = [TreeInformation([tree], start_symbol='START_OUTPUT',
            intermediate_symbol=None, end_symbol=None)
            for tree in out_trees]
    in_string, in_structured_data = merge_graph_structures_new([known_graph_structure, to_prove_graph_structure])
    uv_pos = []
    _pos = in_string.index('END_OF_SECTION')
    for i in range(len(unconstrained_variables)):
        uv_pos.append([])
        for j in range(_pos, len(in_string)):
            if in_string[j] == uv_dict[unconstrained_variables[i]]:
                uv_pos[i].append(j)
                in_string[j] = 'UC'
    in_tokens = [config.encode[t] for t in in_string]
    inputs = (in_tokens, in_structured_data)
    outputs = []
    for i, graph in enumerate(out_graph_structure):
        string, structured_data = merge_graph_structures_new([graph])
        tokens = [config.encode[t] for t in string]
        outputs.append((tokens.copy(), structured_data.copy(), uv_pos[i]))
    return inputs, outputs

def encode_gen_steps(fpath):
    lm = load_language_model()
    proof_steps = load_gen_proof_steps(fpath, lm=lm)
    print ("Found %d gen steps" % (len(proof_steps)))
    config = get_config(lm)
    inputs = []
    for ps in proof_steps:
        inputs.append(encode_gen_step(ps, lm, config))
    with open(fpath+'_gen_encode', 'wb') as f:
        pickle.dump(inputs, f)

def encode_props(fpath):
    lm = load_language_model()
    inputs = []
    config = get_config(lm)
    for p in lm.database.propositions_list:
        inputs.append(encode_proof_step(p.tree, p.f, p.hyps, lm, config))
    with open(os.path.join(fpath, 'props_encode'), 'wb') as f:
        pickle.dump(inputs, f)

def encode_payout_step(statement, hyps, lm, config):
    random_replacement_dict = config.lm.random_replacement_dict_f(f=None)
    statement = statement.copy().replace_values(random_replacement_dict)
    hyps = [h.copy().replace_values(random_replacement_dict) for h in hyps]
    statement_graph_structure = TreeInformation([statement],
            start_symbol=None, intermediate_symbol='END_OF_HYP',
            end_symbol='END_OF_SECTION')
    hyps_graph_structure = TreeInformation(hyps,
            start_symbol=None, intermediate_symbol='END_OF_HYP',
            end_symbol='END_OF_SECTION')
    in_string, structure_data = merge_graph_structures_new([statement_graph_structure, hyps_graph_structure])
    #tokens = [config.encode[t] for t in in_string]
    return in_string, structure_data # don't convert string to tokens, since python2 and python3  have different config.encode

def encode_payout_steps(fpath, outpath):
    _idx = int(fpath[fpath.rfind('w')+1:])
    lm = load_language_model()
    with open(fpath, 'rb') as f:
        data = pickle.load(f)
    config = get_config(lm)
    inputs = []
    for i, _pd in enumerate(data):
        for tree in _pd.correct:
            _to, _tr = encode_payout_step(tree, _pd.hyps, lm, config)
            inputs.append((_to, _tr, 1))
        for tree in _pd.wrong:
            _to, _tr = encode_payout_step(tree, _pd.hyps, lm, config)
            inputs.append((_to, _tr, 0))

    for i in range(2):
        fname = os.path.join(outpath, str(_idx*2+i)+'_payout')
        print (fname)
        l = len(inputs)
        _per = int(l/2+1)
        with open(fname, 'wb') as f:
            pickle.dump(inputs[_per*i:min(l, _per*(i+1))], f)

def pad_list(tokens, trees, device, max_len=300):
    lengths = torch.LongTensor([len(seq) for seq in tokens]).to(device)
    tokens_tensor = torch.zeros(len(tokens), lengths.max()).to(device)
    trees_tensor = torch.zeros(len(tokens), lengths.max(), 4).to(device)
    for i, l in enumerate(lengths):
        tokens_tensor[i, :l] = tokens[i]
        trees_tensor[i, :l, :] = trees[i]
    if lengths.max() > max_len:
        tokens_tensor = tokens_tensor[:, :max_len]
        trees_tensor = trees_tensor[:, :max_len, :]
        lengths[lengths > max_len] = max_len
    lengths, perm_idx = lengths.sort(0, descending=True)
    tokens_tensor = tokens_tensor[perm_idx].transpose(0,1)
    trees_tensor = trees_tensor[perm_idx].transpose(0,1) # seq, batch, feat
    return tokens_tensor, trees_tensor, perm_idx, lengths

def print_args(args):
    s = '\nParameter settings:\n'
    for name, value in args.__dict__.items():
        s += str(name) + ':' + str(value) + '\n'
    return s

opt_dict = {
        'adam' : torch.optim.Adam,
        'rmsprop' : torch.optim.RMSprop
        }

def get_opt(_model, args):
    params = []
    if type(_model) == type({}):
        for name, value in _model.items():
            if type(value) is list:
                for m in value:
                    params += list(m.parameters())
            else:
                params += list(value.parameters())
    else:
        params += list(_model.parameters())
    opt = opt_dict[args.graph_opt](params=params, lr=args.learning_rate, weight_decay=1e-4)
    return opt

def normalize(tree, f):
    tokens = tree.list()
    replacement_dict = {}
    num = {'wff':0, 'set':0, 'class':0}
    for t in tokens:
        if t in f and t not in replacement_dict:
            vclass = f[t].vclass
            new_var = vclass.upper()+'Var'+str(num[vclass])
            replacement_dict[t] = new_var
            num[vclass] += 1
    return tree.copy().replace_values(replacement_dict)

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            param.grad.data.clamp_(-grad_clip, grad_clip)

if __name__ == "__main__":
    ISET = True
    initial_file_name = '../data_iset/iset.mm' if ISET else '../data/set.mm'
    text = file_contents(initial_file_name)
    database = meta_math_database(text,n=100000, remember_proof_steps=True)
    lm = LanguageModel(database)
    save_language_model(lm, suffix='_iset' if ISET else '')
    fp = '../data_iset'
    for split in ['train', 'valid', 'test']:
        fl = os.listdir(os.path.join(fp, split))
        fl = [x for x in fl if x.isalnum()]
        for fn in fl:
            fpath = os.path.join(fp, split, fn)
            get_props(os.path.join(fp, split, fn))
            encode_proof_steps(fpath)
            encode_gen_steps(fpath)
    encode_props('../data_iset')
