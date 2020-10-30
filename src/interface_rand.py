
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

    def get_payout(self, tree, context):
        return 0.5

    def props(self, tree, context):
        labels = self.lm.searcher.search(tree, context, max_proposition=context.number, vclass='|-')
        for label in DISALLOWED_PROPS:
            if label in labels:
                labels.remove(label)
        score = np.random.normal(size=(len(labels)))
        return labels, score 

    def apply_prop(self, tree, context, prop_name, n=10, step=None):
        prop = self.lm.database.propositions[prop_name]
        if prop.unconstrained_arity() == 0:
            return [(0.0, self.lm.simple_apply_prop(tree, prop, context, vclass='|-'))]

        fit = tree.fit(prop.tree, prop.f)
        for f in prop.unconstrained_f:
            fit[f] = Tree(context.replacement_dict[f])
        return [(0.0, [h.tree.copy().replace(fit) for h in prop.hyps if h.type=='e'])]

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