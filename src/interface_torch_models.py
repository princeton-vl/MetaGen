'''
This builds up the interface for the proof search module.
'''

import gen_model_beam_search
import gen_model_beam_search_torch
import pred_model as pred_model_run
import payout_model_5_train as payout_model_run
from models import *
from beam_search import *
import os
import sys
import numpy as np
import pickle as pickle
import data_utils5 as data_utils
import nnlibrary as nn
import data_utils as data_utils_new
import constructor_list

import torch
import torch_models

NUM_ALLOWED_CONSTRUCTORS = None
DISALLOWED_PROPS = ['idi', 'dummylink', 'dtrucor']
PRED_ENSEMBLE = 1
PRED_CACHE_ENSEMBLE = 1
PAYOUT_ENSEMBLE = 1
GEN_ENSEMBLE = 1
PAYOUT_SCALE = 1.0 # Chosen to make the spread of payouts roughly uniform over 0.5-1.0.

if NUM_ALLOWED_CONSTRUCTORS is None:
    ALLOWED_CONSTRUCTORS = None
else:
    ALLOWED_CONSTRUCTORS = set(constructor_list.order[:NUM_ALLOWED_CONSTRUCTORS])



class ProofInterface:
    def __init__(self, args, lm, recalculate_props=True, directory='searcher'):
        self.lm = lm
        self.config = data_utils_new.get_config(lm)
        self.args = args
        self.holo_directory = 'searcher'
        # load all the variables and parameters
        # I'm fixing the file locations by hand because lazy.
        loc = 'cpu' if args.cpu else 'cuda:0'
        if self.args.no_use_torch:
            self.gen_config = gen_model_beam_search.Config(lm)
            self.gen_config.load(self.holo_directory+'/gen.parameters')
            self.gen_var = gen_model_beam_search.Variables(self.gen_config)
            self.gen_var.load(self.holo_directory+'/gen.weights')

            self.pred_config = pred_model_run.Config(lm)
            self.pred_config.load(self.holo_directory+'/pred.parameters')
            self.pred_var = pred_model_run.Variables(self.pred_config)
            self.pred_var.load(self.holo_directory+'/pred.weights')

            self.payout_config = payout_model_run.Config(lm)
            self.payout_config.load(self.holo_directory+'/payout.parameters')
            self.payout_var = payout_model_run.Variables(self.payout_config)
            self.payout_var.load(self.holo_directory+'/payout.weights')
        print (args.device)
        # Load model.
        self.args.vocab_size = len(self.config.encode)+1
        if self.args.interface_pred_model != '':
            if self.args.stat_model:
                self.pred_model = torch.load(self.args.interface_pred_model)
            else:
                self.args_pred = torch.load(self.args.interface_pred_model, map_location=loc)['args']
                self.args_pred.device = args.device
                self.args_pred.cpu = args.cpu
                #self.args_pred.vocab_size = 1189 if hasattr(self.args_pred, 'gen_lm') and self.args_pred.gen_lm else 1089
                self.args_pred.max_len  = self.args.max_len
                self.pred_model = torch_models.PredModel(self.args_pred, self.config).cuda()
                self.pred_model.load(self.args.interface_pred_model)
                self.pred_model.to(args.device)
            self.pred_model.args.device = args.device
        else:
            self.pred_model = torch_models.PredModel(args, self.config).to(args.device)
            self.args_pred = args
        if self.args.interface_gen_model != '':
            if self.args.stat_model:
                data = torch.load(self.args.interface_gen_model)
                self.gen_model = torch_models.LModel(data['args'], self.config).cuda()
                self.gen_model.load_state_dict(data['models'])
            else:
                self.args_gen = torch.load(self.args.interface_gen_model, map_location=loc)['args']
                self.args_gen.device = args.device
                self.args_gen.cpu = args.cpu
                #self.args_gen.vocab_size = 1189 if hasattr(self.args_gen, 'gen_lm') and self.args_gen.gen_lm else 1089
                self.args_gen.max_len  = self.args.max_len
                self.gen_model = torch_models.GenModel2(self.args_gen, self.config).cuda()
                self.gen_model.load(self.args.interface_gen_model)
                self.gen_model.to(args.device)
            self.gen_model.args.device = args.device
        else:
            self.gen_model = torch_models.GenModel2(args, self.config).to(args.device)
            self.args_gen = args
        if self.args.interface_payout_model != '':
            self.args_payout = torch.load(self.args.interface_payout_model)['args']
            #self.args_payout.vocab_size = 1189 if hasattr(self.args_payout, 'gen_lm') and self.args_payout.gen_lm else 1089
            self.args_payout.max_len  = self.args.max_len
            self.payout_model = torch_models.Payout(self.args_payout, self.config).cuda()
            self.payout_model.load(self.args.interface_payout_model)
            self.payout_model.to(args.device)
            self.payout_model.args.device = args.device
        else:
            self.payout_model = torch_models.Payout(args, self.config).to(args.device)
            self.args_payout = args
        self.pred_model.eval()
        self.gen_model.eval()
        self.payout_model.eval()
        #self.args.vocab_size = len(self.config.encode)+1
        #self.pred_model = torch_models.PredModel(args, self.config).to(args.device)
        #self.gen_model = torch_models.GenModel(args, self.config).to(args.device)
        #self.payout_model = torch_models.Payout(args).to(args.device)
        #if self.args.interface_pred_model != '':
        #    self.pred_model.cuda()
        #    self.pred_model.load(self.args.interface_pred_model)
        #    self.pred_model.to(args.device)
        #if self.args.interface_gen_model != '':
        #    self.gen_model.cuda()
        #    self.gen_model.load(self.args.interface_gen_model)
        #    self.gen_model.to(args.device)
        #if self.args.interface_payout_model != '':
        #    self.payout_model.load(self.args.interface_payout_model)
        #self.pred_model.cpu()
        #self.gen_model.cpu()
        #self.payout_model.cpu()
        #torch.save({'models':self.pred_model.state_dict()}, '../models/pred_default_cpu')
        #torch.save({'models':self.gen_model.state_dict()}, '../models/gen_default_cpu')
        #torch.save({'models':self.payout_model.state_dict()}, '../models/payout_default_cpu')
        # beam search interface
        if self.args.no_use_torch:
            self.bsi = gen_model_beam_search.BeamSearchInterface([self.gen_var]*GEN_ENSEMBLE)
        else:
            self.bsi = gen_model_beam_search_torch.BeamSearchInterface([None]*GEN_ENSEMBLE, self.args, self.gen_model)
        # remember the answer so that we don't need to constantly recalculate it
        file_path = directory+'/pred_database'
        if self.args.cpu:
            file_path += '_cpu'
        if os.path.isfile(file_path) and not recalculate_props:
            print ('loading proposition vectors')
            if self.args.no_use_torch:
                with open(file_path, 'rb') as handle:
                    self.pred_database = pickle.load(handle, encoding='latin1')
            else:
                self.pred_database = torch.load(file_path)#pickle.load(handle)
        else:
            print ('using proposition vectors at '+file_path)
            if self.args.stat_model:
                self.initialize_pred_tfidf()
            else:
                self.initialize_pred(file_path)
        print ('pred_database', self.pred_database.shape)

    def initialize_pred_tfidf(self):
        with open('../data/props_encode', 'rb') as f:
            prop_inputs = pickle.load(f)
        prop_embs = torch.zeros(len(prop_inputs), len(self.config.encode)+1).to(self.args.device)
        for i in range(len(prop_inputs)):
            prop_embs[i] = self.pred_model.embed(prop_inputs[i][0])
        self.pred_database = prop_embs
        print ('\rdone adding propositions')

    def initialize_pred(self, file_path):
        args = self.args
        if args.partial_lm:
            prop_inputs = []
            for prop in self.lm.database.propositions_list:
                prop_inputs.append(data_utils_new.encode_proof_step(prop.tree, prop.f, prop.hyps, self.lm, self.config))
        else:
            with open(os.path.join(self.args.data_path, 'props_encode'), 'rb') as f:
                prop_inputs = pickle.load(f)
        self.pred_database = torch.zeros(len(prop_inputs), self.args_pred.nFeats*2 if self.args_pred.bidirectional else self.args_pred.nFeats).to(args.device)
        l = 0
        while l < len(prop_inputs):
            #os.system('nvidia-smi')
            r = min(len(prop_inputs), l+args.batch_size)
            tokens = [torch.LongTensor(prop_inputs[i][0]).to(args.device) for i in range(l, r)]
            trees = [torch.Tensor(prop_inputs[i][1]).to(args.device) for i in range(l, r)]
            with torch.no_grad():
                self.pred_database[l:r] = self.pred_model.embed(tokens, trees, _type='p')
            l = r
        print ('\rdone adding propositions')
        # save the database
        #if self.args.no_use_torch:
        #    with open(file_path, 'wb') as handle:
        #        pickle.dump(self.pred_database, handle)
        #else:
        #    torch.save(self.pred_database, file_path)
    '''
    def initialize_pred(self, file_path):
        # this initializes all of the proposition vectors in database,
        # so that we can call them quickly when we need to.
        # this should include the multiplication
        #self.pred_database = [pred_model_run.get_prop_vector([self.pred_var]*ENSEMBLE, prop) for prop in self.lm.database.propositions_list)]
        self.pred_database = []
        for i, prop in enumerate(self.lm.database.propositions_list):
            sys.stdout.write('\rvectorizing proposition '+str(i))
            sys.stdout.flush()
            self.pred_database.append(pred_model_run.get_prop_vector([self.pred_var]*PRED_CACHE_ENSEMBLE, prop))
        print ('\rdone adding propositions')
        self.pred_database = np.stack(self.pred_database, axis=0)

        # save the database
        with open(file_path, 'wb') as handle:
            pickle.dump(self.pred_database, handle)
    '''
    def rename_var(self, statement, hyps, f, config):
        random_replacement_dict = config.lm.random_replacement_dict_f(f=f)
        statement = statement.copy().replace_values(random_replacement_dict)
        hyps = [h.tree.copy().replace_values(random_replacement_dict) for h in hyps if h.type=='e']
        statement_graph_structure = TreeInformation([statement],
                start_symbol=None, intermediate_symbol='END_OF_HYP',
                end_symbol='END_OF_SECTION')
        hyps_graph_structure = TreeInformation(hyps,
                start_symbol=None, intermediate_symbol='END_OF_HYP',
                end_symbol='END_OF_SECTION')
        in_string, structured_data = data_utils_new.merge_graph_structures_new([statement_graph_structure, hyps_graph_structure])
        tokens = [config.encode[t] for t in in_string]
        trees = torch.Tensor(structured_data).to(self.args.device)
        tokens = torch.LongTensor(tokens).to(self.args.device)
        return tokens, trees

    def payout(self, tree, context):
        hyps = context.hyps #[h.tree for h in context.hyps if h.type=='e']
        #f = None
        #statement = tree
        #random_replacement_dict = self.payout_config.lm.random_replacement_dict_f(f=f)
        #statement = statement.copy().replace_values(random_replacement_dict)
        #hyps = [h.copy().replace_values(random_replacement_dict) for h in hyps]
        #statement_graph_structure = TreeInformation([statement],
        #        start_symbol=None, intermediate_symbol='END_OF_HYP',
        #        end_symbol='END_OF_SECTION')
        #hyps_graph_structure = TreeInformation(hyps,
        #        start_symbol=None, intermediate_symbol='END_OF_HYP',
        #        end_symbol='END_OF_SECTION')
        #in_string, structured_data = data_utils_new.merge_graph_structures_new([statement_graph_structure, hyps_graph_structure])
        #tokens = [self.payout_config.encode[t] for t in in_string]
        #tokens, trees = payout_model_run.get_input(tree, context)
        #print (tokens)
        #print (structured_data)
        #trees = torch.Tensor(structured_data).to(self.args.device)
        #tokens = torch.LongTensor(tokens).to(self.args.device)
        #print (tokens)
        #print (trees)
        tokens, trees = self.rename_var(tree, hyps, None, self.config)
        with torch.no_grad():
            score = self.payout_model.forward(([tokens], [trees], None))
        #print ('payout', tokens.shape, trees.shape, score.shape)
        return score.item()
        #return payout_model_run.get_payout([self.payout_var]*PAYOUT_ENSEMBLE, tree, context)

    def initialize_payout(self, context):
        #context.difficulty = self.payout(context.tree, context)
        pass

    def get_payout(self, tree, context):
        ''' note: the test dataset had the following histogram for delta,
        [ 0.05543478,  0.01594203,  0.01376812,  0.00797101,  0.00398551,
        0.00144928,  0.00144928] using bin sizes of 0.5, i.e. 0-0.5, 0.5-1,...
        '''
        # TODO tokenization
        if self.args.no_use_torch:
            score = payout_model_run.get_payout([self.payout_var]*PAYOUT_ENSEMBLE, tree, context)
            score = np.exp(score)/(1.0+np.exp(score))
        else:
            score = self.payout(tree, context)
        #print ('payout', score)
        return score
        #print 'getting payout'
        # return difficulty
        # delta = (context.difficulty - difficulty) * PAYOUT_SCALE
        # delta = (difficulty) * PAYOUT_SCALE
        # return delta
        #return np.exp(include_score)/(1.0+np.exp(include_score))

    def props_torch(self, tree, context):
        # TODO tokenization
        #print ('props_torch')
        hyps = context.hyps
        #statement = tree
        #f = None
        #random_replacement_dict = self.pred_config.lm.random_replacement_dict_f(f=f)
        #statement = statement.copy().replace_values(random_replacement_dict)
        #hyps = [h.tree.copy().replace_values(random_replacement_dict) for h in hyps if h.type=='e']
        tokens, trees = self.rename_var(tree, hyps, None, self.config)
        #print ('props', tokens.shape, trees.shape)
        with torch.no_grad():
            if self.args.stat_model:
                g_vec = self.pred_model.embed(tokens).view(1,-1)
            else:
                g_vec = self.pred_model.embed([tokens], [trees], _type='g')
        #print (g_vec.shape)
        # get visible props
        labels = self.lm.searcher.search(tree, context, max_proposition=context.number, vclass='|-')
        for label in DISALLOWED_PROPS:
            if label in labels:
                labels.remove(label)
        prop_nums = torch.LongTensor([self.config.lm.database.propositions[label].number for label in labels]).to(self.args.device)
        #print(prop_nums.shape)
        p_vec = self.pred_database[prop_nums]
        #print (p_vec.shape)
        # score
        with torch.no_grad():
            score = self.pred_model.biln(g_vec, p_vec).view(-1)
        #print (score.shape)
        score -= score.max()
        #print (score)
        return labels, score.cpu().numpy()

    def props_holophrasm(self, tree, context):
        # returns the sorted list of propositions.
        vec = pred_model_run.get_main_vector([self.pred_var]*PRED_ENSEMBLE, tree, context)
        labels = self.lm.searcher.search(tree, context, max_proposition=context.number, vclass='|-')
        # we disallow these two particular propositions
        for label in DISALLOWED_PROPS:
            if label in labels:
                labels.remove(label)
        prop_nums = [self.lm.database.propositions[label].number for label in labels]
        submatrix =  self.pred_database[np.array(prop_nums), :]
        logits = np.dot(submatrix, vec)

        # print labels, nn.log_softmax(logits)
        # input("Press Enter to continue...")
        return labels, logits - np.max(logits)  # rescaled log-probability
        #return labels, nn.log_softmax(logits)

        # # we don't need to do the sorting here
        # prop_indices = np.argsort(logits)[::-1]
        # sorted_labels = [labels[index] for index in prop_indices]
        # probs = nn.log_softmax(logits)
        # probs = probs[prop_indices]
        # return sorted_labels, probs  # highest to lowest
    def props(self, tree, context):
        if self.args.no_use_torch:
            labels, scores = self.props_holophrasm(tree, context)
            #print ('props_holophrasm')
            #print (labels)
            #print (scores)
        else:
            labels, scores = self.props_torch(tree, context)
            #print ('props_torch')
            #print (labels)
            #print (scores)
        return labels, scores

    def apply_prop(self, tree, context, prop_name, n=10, return_replacement_dict=False, step=None):
        # shortcut if the unconstrainer arity is 0
        prop = self.lm.database.propositions[prop_name]
        if prop.unconstrained_arity() == 0:
            return [(0.0, self.lm.simple_apply_prop(tree, prop, context, vclass='|-'))]

        ''' in this case, params = tree, context, prop_name '''
        beam_searcher = BeamSearcher(self.bsi, (tree, context, prop_name, ALLOWED_CONSTRUCTORS, return_replacement_dict, step))
        out = beam_searcher.best(n, n, n) #(width, k, num_out)  See notes regarding accuracy
        #print 'out', out
        return out

    def is_tautology(self, tree, context):
        '''
        check to see wether the tree is tautologically true.
        We can do this *really* quickly, so we might as well.

        There's a little redundency in that we calculate the
        viable props twice, but it's a pretty quick process.

        Returns None if not a tautology, otherwise returns a
        label for a proposition that proves it immediately.
        '''
        labels = self.lm.searcher.search(tree, context, max_proposition=context.number, vclass='|-')
        tauts = set(labels).intersection(self.lm.tautologies)
        if len(tauts)==0:
            return None
        else:
            return tauts.pop()

