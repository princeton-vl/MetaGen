from data_utils import *
import constructor_list
from tree_parser import *
import random
import params
import torch
import time
from torch.distributions.categorical import *
import torch.nn as nn
import torch_models
import torch.multiprocessing as mp
DISALLOWED_PROPS = ['idi', 'dummylink']

class Expression:
    is_hyps = 0 # 2 initial hyps 1 derived from some hyps 0 known facts
    hyps = set()
    proofs = [] # list of proof
    number = -1
    tree = None
    prior = []
    context_label = ''
    unconstrained = []
    def __init__(self, tree, prior_expr, prop, _type, unconstrained, context=None, f=None, d=None):
        self.tree = tree
        self.cnt = 1
        self.unconstrained = unconstrained
        self.hyps = set()
        self.proofs = []
        self.f = {}
        self.d = set()
        if context is not None:
            self.context_label = context.label
            for t in self.tree.list():
                if t in context.f and t not in self.f:
                    self.f[t] = context.f[t]
            var = {v.variable:k for k,v in context.f.items()}
            for x,y in context.d:
                if x not in var or y not in var:
                    continue
                if var[x] in self.f and var[y] in self.f:
                    self.d.add((var[x], var[y]))
        else:
            self.context_label = 'new'
            for t in self.tree.list():
                if t in f and t not in self.f:
                    self.f[t] = f[t]
            self.d = d

        if _type == 'e':
            # is a hyps
            self.is_hyps = 2
        elif _type == 'p':
            self.prop = prop
            for e in prior_expr:
                self.hyps.update(e.hyps)
                self.proofs += e.proofs
            self.proofs.append(prop.label)
            if len(self.hyps) > 0:
                self.is_hyps = 1
            else:
                self.is_hyps = 0
        self.prior = [e.number for e in prior_expr]
        self.id = '_'.join([str(i) for i in self.hyps])+str(self.tree)+str(sorted(self.d))

    def add_number(self, number):
        self.number = number
        if self.is_hyps == 2:
            self.hyps = set([number])
            self.proofs = [number]
            self.prop = number
    def print(self, generator):
        print ('hyps')
        for i in self.hyps:
            print (generator.expressions_list[i].tree)
        print ('goal')
        print (self.tree)
        if self.prop is not None:
            print ('prop', self.prop.label)
            print ([e.tree for e in self.prop.hyps if e.type == 'e'])
            print (self.prop.tree)
        print ('prior')
        for i in self.prior:
            print (generator.expressions_list[i].tree)
        if len(self.unconstrained) > 0:
            print ('unconstrained')
            for i in range(len(self.unconstrained)):
                print (self.prop.unconstrained_variables[i], self.unconstrained[i])

class Constructor:
    def __init__(self, args, config, interface=None):
        self.args = args
        #self.forward_data = forward_data
        if self.args.random_generate:
            self.args.sample_width = 1
        self.config = config
        self.lm = config.lm
        self.construction_axioms = [l for l in self.lm.constructor_labels if l not in self.lm.new_names]
        self.expressions = {}
        self.expressions_list = []
        self.step_expr_pos = {}
        self.prop_hyps_pos = {}
        self.substitutions = {'wff':{}, 'set':{}, 'class':{}}
        self.training_propositions = {}
        self.training_propositions_all = {}
        self.propositions = {}
        self.prop_goal_var = {}
        self.prop_input = {} # for the use of encoding generated tasks
        self.all_f = {}
        self.replacement_dict = None
        self.searcher = None
        self.prop_searcher = None
        self.num_initial_expr = None
        self.prop_labels = []
        self.gen_prop_labels = []
        self.non_gen_prop_labels = []
        self.prop_dist = {}
        self.prop_dist_list = []
        self.prop_samples = []
        self.prop_dist_samples = []
        self.interface = interface
        self.prop_next_step = {}
        self.step_prior_idx = None
        self.num_new_expr = 0
        self.num_gt_new_expr = 0
        self.logs = {}
        self.pred_request_idx = None

    def initialize_searcher(self):
        self.searcher = SearchProblem_new(self.expressions_list)
        self.prop_searcher = SearchProblem(self.config.lm.database)

    def initialize(self):
        self.initialize_prop()
        print ('Find %d constructors' % (len(self.propositions)))
        print ('Find %d training propositions' % (len(self.training_propositions)))
        self.initialize_expr(self.training_propositions)
        self.num_initial_expr = len(self.expressions)
        print ('Find %d expressions' % (self.num_initial_expr))
        for k,v in self.substitutions.items():
            print ('Find %d %s substitutions' % (len(v), k))
        self.initialize_searcher()

    def initialize_prop(self):
        lm = self.lm
        ths = []
        num = len(lm.database.propositions_list)
        cnt = 0
        
        if self.args.data_sampling > 0:
            num = int(self.args.data_sampling * 10)
        for i,p in enumerate(lm.training_propositions):
            if p.vclass == '|-' and p.label not in DISALLOWED_PROPS:
                if self.args.data_sampling > 0 and i % 10 < num:
                    self.training_propositions[p.label] = p
                self.training_propositions_all[p.label] = p

        for l in self.training_propositions_all:
            if l not in self.training_propositions:
                self.prop_next_step[l] = 0

        for p in lm.database.propositions_list:
            if p.vclass == '|-' and p.label not in DISALLOWED_PROPS:
                self.propositions[p.label] = p

        with open(os.path.join(self.args.data_path, 'props_encode'), 'rb') as f:
            prop_encode = pickle.load(f)

        for l,p in self.propositions.items():
            for f in p.f:
                if f not in self.all_f:
                    self.all_f[f] = p.f[f]

        for l,p in lm.database.propositions.items():
            inputs = prop_encode[p.number]

            self.prop_input[l] = inputs
            if self.args.partial_lm:
                self.prop_input[l] = encode_proof_step(p.tree, p.f, p.hyps, self.lm, self.config)

            s = []
            self.prop_goal_var[l] = []
            for _,e in p.e.items():
                s += e.tree.list()
            g = p.tree.list()
            for f in p.f:
                if f in g and f not in s:
                    self.prop_goal_var[l].append(f)

        self.prop_labels = list(self.propositions.keys())
        self.gen_prop_labels = [l for l in self.propositions if len(self.propositions[l].unconstrained_variables)>0]
        self.non_gen_prop_labels = [l for l in self.propositions if len(self.propositions[l].unconstrained_variables) == 0]

        self.replacement_dict = self.config.lm.deterministic_replacement_dict_f(self.all_f)
        for _,p in self.training_propositions.items():
            if p.proof is not None:
                for s in p.proof:
                    if s in self.propositions and self.propositions[s].vclass == '|-':
                        if s not in self.prop_dist:
                            self.prop_dist[s] = 0
                        self.prop_dist[s] += 1
        self.prop_dist_list = [0] * len(self.prop_labels)
        for i, l in enumerate(self.prop_labels):
            if l in self.prop_dist:
                self.prop_dist_list[i] = self.prop_dist[l]
            else:
                self.prop_dist_list[i] = 0

    def initialize_expr(self, training_propositions=None):
        if training_propositions is None:
            training_propositions = self.training_propositions
        for i,p in enumerate(self.lm.database.propositions_list):
            if p.label not in self.training_propositions_all:
                # we only consider the props among training proof tasks
                continue
            if p.label in training_propositions:
                self.verify(p)
            else: 
                prior_expr = []
                self.prop_hyps_pos[p.label] = {}
                self.step_expr_pos[p.label] = []
                for hyps in p.hyps:
                    if hyps.type == 'e':
                        expr = Expression(hyps.tree, [], None, 'e', [], p)
                        number = self.add_expression(expr, add2searcher=False)
                        self.prop_hyps_pos[p.label][hyps.label] = number
                        prior_expr.append(self.expressions_list[number])

                self.add_expression(Expression(p.tree, prior_expr, p, 'p', [], p), add2searcher=False)
            if i % 10000 == 0:
                print (i, p.label, len(self.expressions))


    def verify(self, prop):
        assert prop.entails_proof_steps is not None
        tmp_dict = {}
        self.step_expr_pos[prop.label] = []
        self.prop_hyps_pos[prop.label] = {}
        for l,e in prop.e.items():
            expr = Expression(e.tree, [], None, 'e', [], prop)
            number = self.add_expression(expr, add2searcher=False)
            tmp_dict[str(e.tree)] = number
            self.prop_hyps_pos[prop.label][l] = number
        for num,step in enumerate(prop.entails_proof_steps):
            fit = step.tree.fit(step.prop.tree, step.prop.f)
            for i,var in enumerate(step.prop.unconstrained_variables):
                fit[var] = step.unconstrained[i]
            for hyps in step.prop.hyps:
                if hyps.type == 'f':
                    assert hyps.label in fit

            for f in self.prop_goal_var[step.prop.label]:
                vclass = self.all_f[f].vclass
                _id = str(fit[f])
                if _id not in self.substitutions[vclass]:
                    self.substitutions[vclass][_id] = fit[f]
            prior = []
            for hyps in step.prop.hyps:
                if hyps.type == 'e':
                    tree = hyps.tree.copy().replace(fit)
                    assert str(tree) in tmp_dict
                    prior.append(self.expressions_list[tmp_dict[str(tree)]])

            expr = Expression(step.tree, prior, step.prop, 'p', step.unconstrained, prop)
            number = self.add_expression(expr, add2searcher=False)
            self.step_expr_pos[prop.label].append(number)

            tmp_dict[str(step.tree)] = number

    def reinitialize_expressions(self):
        self.expressions_list = self.expressions_list[:self.num_initial_expr]
        self.expressions = {}
        for e in self.expressions_list:
            self.expressions[e.id] = e
        self.searcher = SearchProblem_new(self.expressions_list)
        print ('reinitialize_expressions %d' % (len(self.expressions_list)))

    def add_expression(self, expr, add2searcher=True):
        if expr.id not in self.expressions:
            expr.add_number(len(self.expressions_list))
            self.expressions[expr.id] = expr
            self.expressions_list.append(expr)
            if add2searcher:
                self.searcher.add(expr, len(self.expressions_list)-1)
        else:
            self.expressions[expr.id].cnt += 1
            l = len(self.expressions[expr.id].proofs)
            if expr.is_hyps < 2 and len(expr.proofs) < l:
                self.expressions[expr.id].proofs = expr.proofs.copy()

        prop = self.expressions[expr.id].context_label
        if prop in self.propositions and self.expressions[expr.id].tree == self.propositions[prop].tree:
            self.expressions[expr.id].proofs = [prop]
        return self.expressions[expr.id].number

    def verify_d(self, prop, prior, fit):
        new_d = set()
        vars_that_appear = {v:fit[v].set().intersection(self.all_f) for v in fit}
        for (xvar, yvar) in prop.d_labels:
            if xvar not in fit or yvar not in fit: continue
            for x in vars_that_appear[xvar]:
                if x in vars_that_appear[yvar]:
                    return None
                for y in vars_that_appear[yvar]:
                    new_d.add((x,y))
        for e in prior:
            new_d.update(e.d)
        return new_d

    def sample_subs_random(self, vclass):
        label = random.choice(list(self.substitutions[vclass].keys()))
        return self.substitutions[vclass][label]

    def random_generate(self, gen=False, add_to_list=True):
        labels = self.prop_labels
        if gen:
            if random.random() < 0.9:
                labels = self.gen_prop_labels
            else:
                labels = self.non_gen_prop_labels
        num_try = 0
        while True:
            num_try += 1
            prop = self.propositions[random.choice(labels)]
            fit = {}
            prior = []
            fail = False
            for hyps in prop.hyps:
                if hyps.type == 'f':
                    continue
                tree = hyps.tree.copy().replace(fit)
                out = self.search(tree)
                if len(out) == 0:
                    fail = True
                    break
                idx, _fit = out[0]
                prior.append(idx)
                fit.update(_fit)
            if fail == False:
                # we find a prop and its subs
                if self.args.data_sampling > 0:
                    for f in self.prop_goal_var[prop.label]:
                        fit[f] = self.sample_subs_random(self.all_f[f].vclass)
                    goal = prop.tree.copy().replace(fit)
                else:
                    # generate a new expression
                    unconstrained_variables = self.prop_goal_var[prop.label]
                    uv_dict = {var:'UC'+var for var in unconstrained_variables}
                    goal = prop.tree.copy().replace_values(uv_dict).replace(fit)
                    context_hyps = []
                    context = self.standardize_context(goal, context_hyps, fit, uv_dict)
                    goal = goal.replace_values(context.replacement_dict)
                    if len(unconstrained_variables) > 0:
                        out = self.interface.apply_prop(goal, context, prop.label, n=1) # return value and tree
                        if out is None:
                            continue
                        goal = out[1]
                        
                    reverse_replacement = {v:k for k,v in context.replacement_dict.items()}
                    goal = goal.replace_values(reverse_replacement)

                prior = [self.expressions_list[i] for i in prior]
                new_d = self.verify_d(prop, prior, fit)
                if new_d is None: continue
                unconstrained = [fit[f] for f in prop.unconstrained_variables]
                expr = Expression(goal, prior, prop, 'p', unconstrained, f = self.all_f, d = new_d)
                if self.args.repeatable:
                    # repeatable, but we still want expr to be different from existing ones from gt proofs.
                    if expr.id not in self.expressions:
                        self.num_new_expr += 1
                        if add_to_list:
                            number = self.add_expression(expr)
                            return self.expressions_list[number]
                        else:
                            return expr
                    else:
                        number = self.expressions[expr.id].number
                        if number >= self.num_initial_expr:
                            return self.expressions_list[number]
                else:
                    # not repeatable, then we need every expr to be unique
                    if expr.id not in self.expressions:
                        self.num_new_expr += 1
                        if add_to_list:
                            number = self.add_expression(expr)
                            return self.expressions_list[number]
                        else:
                            return expr

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

    def get_pred(self, prop, fit, context_hyps_idx, target, candidates, require_score=False):
        if len(candidates) == 1:
            return candidates[0]
        replacement_dict = self.lm.deterministic_replacement_dict_f(f=self.all_f)#random_replacement_dict_f(f=self.all_f)
        prop_hyps = [hyps.tree for hyps in prop.hyps if hyps.type=='e']
        prop_hyps_label = [hyps.label for hyps in prop.hyps if hyps.type=='e']
        prop_hyps = [h.copy().replace(fit).replace_values(replacement_dict) for h in prop_hyps]
        prop_hyps[target] = Tree('TARGET_UC')

        context_hyps = [self.expressions_list[idx].tree.copy().replace_values(replacement_dict) for idx in context_hyps_idx]
        prop_tree = prop.tree.copy().replace(fit).replace_values(replacement_dict)
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
        idxes = [i[0] for i in candidates]

        expr_tokens = []
        expr_trees = []
        for idx in idxes:
            tokens, tree = self.encode_expr(self.expressions_list[idx].tree,
            [self.expressions_list[i].tree for i in self.expressions_list[idx].hyps],
            replacement_dict)
            expr_tokens.append(tokens)
            expr_trees.append(tree)

        _scores = self.interface.get_pred(([in_tokens], [in_tree], expr_tokens, expr_trees)) # score after softmax
        if self.args.train_rl:
            m = Categorical(_scores)
            if random.random() > self.args.epsilon:
                action = torch.randint(len(m.probs), (1,)).long().to(self.args.device)
            else:
                action = m.sample()
            self.interface.pred_logs[-1].append(m.log_prob(action))
            idx = action.item()
        else:
            scores = _scores.cpu().numpy()
            idx = random.choices(range(len(idxes)), scores)[0]
        if require_score:
            return candidates[idx], _scores.view(-1)
        else:
            return candidates[idx]

    def standardize_context(self, tree, hyps, fit, uv_dict):
        context = Container()
        replacement_dict = self.lm.deterministic_replacement_dict_f(f=self.all_f)#random_replacement_dict_f(f=self.all_f)
        context.tree = tree.copy().replace_values(replacement_dict)
        context.hyps = [h.copy().replace_values(replacement_dict) for h in hyps]
        context.replacement_dict = replacement_dict
        context.fit = {l: v.copy().replace_values(replacement_dict) for l,v in fit.items()}
        context.hyp_symbols = set()
        context.uv_dict = {l:v for l,v in uv_dict.items()}
        for h in context.hyps:
            context.hyp_symbols |= set(h.list())
        context.hyp_symbols &= set(self.lm.new_names)
        return context

    def parameterized_generate(self, fixed_prop=None, add_to_list=True):
        num_try = 0
        while True:
            if self.args.data_sampling > 0 and random.random() < self.args.sample_dist_ratio:
                if len(self.prop_dist_samples) == 0:
                    self.prop_dist_samples = random.choices(self.prop_labels, self.prop_dist_list, k=10**6)
                prop_samples = self.prop_dist_samples
            else:
                if len(self.prop_samples) == 0:
                    self.prop_samples = random.choices(self.prop_labels, k=10**6)
                prop_samples = self.prop_samples
            self.pred_request_num = 0
            if self.args.train_rl:
                self.interface.pred_logs.append([])
                self.interface.pred_rewards.append([])
            num_try += 1
            if fixed_prop is None:
                prop = self.propositions[prop_samples.pop()]
            else:
                prop = self.propositions[fixed_prop]
            uv_dict = {f.label:'UC'+f.label for f in prop.hyps if f.type=='f'}
            uv_var = list(uv_dict.values())
            goal = prop.tree.copy().replace_values(uv_dict)
            fit = {}
            prior = []
            context_hyps_idx = set()
            fail = False
            target = 0 # the number of hyps we are working on
            for hyps in prop.hyps:
                if hyps.type == 'f':
                    continue
                # we replace var in hyps.tree as UCvar, to avoid confusing with other vars.
                tree = hyps.tree.copy().replace_values(uv_dict).replace(fit)
                var = tree.set().intersection(self.all_f) # find var but uc.
                out = self.search(tree, uv_var, var) # f in tree is always uv_dict
                if len(out) == 0:
                    fail = True
                    break
                fit_old_var = {k[2:]:v for k,v in fit.items()}
                idx, _fit = self.get_pred(prop, fit_old_var, context_hyps_idx, target, out)
                fit.update(_fit)

                prior.append(idx)
                context_hyps_idx.update(self.expressions_list[idx].hyps)

                target += 1

            if fail == True:
                if self.args.train_rl:
                    for idx in self.interface.pred_logs[-1]:
                        self.interface.pred_rewards[-1].append(None)
                    assert len(self.interface.pred_rewards) == len(self.interface.pred_logs)
            else:
                if self.args.train_rl:
                    for idx in self.interface.pred_logs[-1]:
                        self.interface.pred_rewards[-1].append(1)
                    assert len(self.interface.pred_rewards) == len(self.interface.pred_logs)
                    self.interface.gen_logs.append([])
                    self.interface.gen_rewards.append([])

                fit = {k[2:]:v for k,v in fit.items()}
                unconstrained_variables = self.prop_goal_var[prop.label]
                uv_dict = {var:'UC'+var for var in unconstrained_variables}
                goal = prop.tree.copy().replace_values(uv_dict).replace(fit)

                context_hyps = [self.expressions_list[i].tree for i in context_hyps_idx]
                context = self.standardize_context(goal, context_hyps, fit, uv_dict)
                goal = goal.replace_values(context.replacement_dict)

                if len(unconstrained_variables) > 0:
                    out = self.interface.apply_prop(goal, context, prop.label, n=1) # return value and tree
                    if out is None:
                        if self.args.train_rl:
                            for ii in range(len(self.interface.gen_logs[-1])):
                                if self.interface.gen_rewards[-1][ii] == 1:
                                    self.interface.gen_rewards[-1][ii] = None
                            for ii in range(len(self.interface.pred_logs[-1])):
                                self.interface.pred_rewards[-1][ii] = None
                        continue
                    if self.args.train_rl:
                        assert len(self.interface.gen_rewards[-1]) == len(self.interface.gen_logs[-1])
                    goal = out[1]

                reverse_replacement = {v:k for k,v in context.replacement_dict.items()}
                goal = goal.replace_values(reverse_replacement)

                # TODO d
                prior = [self.expressions_list[i] for i in prior]
                new_d = self.verify_d(prop, prior, fit)
                if new_d is None:
                    if self.args.train_rl:
                        for ii in range(len(self.interface.gen_logs[-1])):
                            self.interface.gen_rewards[-1][ii] = None
                        for ii in range(len(self.interface.pred_logs[-1])):
                            self.interface.pred_rewards[-1][ii] = None
                    continue

                unconstrained = [fit[f] for f in prop.unconstrained_variables]
                expr = Expression(goal, prior, prop, 'p', unconstrained, f = self.all_f, d = new_d)
                if self.args.repeatable:
                    # repeatable, but we still want expr to be different from existing ones from gt proofs.
                    if expr.id not in self.expressions:
                        self.num_new_expr += 1
                        if add_to_list:
                            number = self.add_expression(expr)
                            return self.expressions_list[number]
                        else:
                            return expr
                    else:
                        number = self.expressions[expr.id].number
                        if number >= self.num_initial_expr:
                            return self.expressions_list[number]
                else:
                    # not repeatable, then we need every expr to be unique
                    if expr.id not in self.expressions:
                        self.num_new_expr += 1
                        if add_to_list:
                            number = self.add_expression(expr)
                            return self.expressions_list[number]
                        else:
                            return expr

                if self.args.train_rl:
                    for ii in range(len(self.interface.gen_logs[-1])):
                        self.interface.gen_rewards[-1][ii] = 0
                    for ii in range(len(self.interface.pred_logs[-1])):
                        self.interface.pred_rewards[-1][ii] = 0

    def compute_step_prop(self, step):
        prop = step.prop
        prior = []
        fit = {}
        context_hyps_idx = set()
        prob = 1
        uv_dict = {f.label:'UC'+f.label for f in prop.hyps if f.type=='f'}
        uv_var = list(uv_dict.values())
        goal = prop.tree.copy().replace_values(uv_dict)
        target = 0
        i = 0
        for hyps in prop.hyps:
            if type(i) == str:
                idx = self.prop_hyps_pos[context_label][i]
            else:
                idx = self.step_expr_pos[context_label][i]
            # idx is the gt expr
            if hyps.type == 'f':
                continue
            # we replace var in hyps.tree as UCvar, to avoid confusing with other vars.
            tree = hyps.tree.copy().replace_values(uv_dict).replace(fit)
            var = tree.set().intersection(self.all_f) # find var but uc.
            out = self.search(tree, uv_var, var)
            true_pos = -1
            for j, c in enumerate(out):
                if idx == c[0]:
                    true_pos = j
                    break
            if true_pos == -1:
                print ('idx not found')
                return
            if len(out) == 0:
                fail = True
                print('pred failed')
                return
            fit_old_var = {k[2:]:v for k,v in fit.items()}
            _, scores = self.get_pred(prop, fit_old_var, context_hyps_idx, target, out, True)
            prob *= scores[true_pos].item()
            _idx, _fit = out[true_pos]
            fit.update(_fit)
            prior.append(idx)
            context_hyps_idx.update(self.expressions_list[idx].hyps)
            target += 1
        context_hyps = [self.expressions_list[i].tree for i in context_hyps_idx]
        fit = {k[2:]:v for k,v in fit.items()}

    def generate_by_cheating(self, pred_cheat=True, gen_cheat=True, prop_cheat=True):
        if prop_cheat:
            context_label = random.choice(list(self.prop_next_step.keys()))
            context = self.propositions[context_label]
            step_id = self.prop_next_step[context.label]
            step = context.entails_proof_steps[step_id] # the step to be generated
            prop = step.prop
        else:
            if random.random() < self.args.sample_dist_ratio:
                if len(self.prop_dist_samples) == 0:
                    self.prop_dist_samples = random.choices(self.prop_labels, self.prop_dist_list, k=10**6)
                prop_samples = self.prop_dist_samples
            else:
                if len(self.prop_samples) == 0:
                    self.prop_samples = random.choices(self.prop_labels, k=10**6)
                prop_samples = self.prop_samples
            prop = self.propositions[prop_samples.pop()]
        prior = []
        fit = {}
        context_hyps_idx = set()

        if pred_cheat:
            for i in self.step_prior_idx[context_label][step_id]:
                if type(i) == str:
                    idx = self.prop_hyps_pos[context_label][i]
                else:
                    idx = self.step_expr_pos[context_label][i]
                expr = self.expressions_list[idx]
                prior.append(idx)
                context_hyps_idx.update(expr.hyps)
            fit_gt = step.tree.fit(prop.tree, prop.f)
            for i,var in enumerate(prop.unconstrained_variables):
                fit_gt[var] = step.unconstrained[i]
            unconstrained_variables = self.prop_goal_var[prop.label]
            fit = {k:v for k,v in fit_gt.items() if k not in unconstrained_variables}
            context_hyps = [h.tree for h in context.hyps if h.type == 'e']
            if self.args.train_rl:
                self.interface.pred_logs.append([])
                self.interface.pred_rewards.append([])
        else:
            fail = False
            uv_dict = {f.label:'UC'+f.label for f in prop.hyps if f.type=='f'}
            uv_var = list(uv_dict.values())
            goal = prop.tree.copy().replace_values(uv_dict)
            target = 0
            for hyps in prop.hyps:
                if hyps.type == 'f':
                    continue
                # we replace var in hyps.tree as UCvar, to avoid being confused with other vars.
                tree = hyps.tree.copy().replace_values(uv_dict).replace(fit)
                var = tree.set().intersection(self.all_f) # find var but uc.
                out = self.search(tree, uv_var, var) # f in tree is always uv_dict
                if len(out) == 0:
                    fail = True

                    return
                fit_old_var = {k[2:]:v for k,v in fit.items()}
                idx, _fit = self.get_pred(prop, fit_old_var, context_hyps_idx, target, out)

                fit.update(_fit)

                prior.append(idx)
                context_hyps_idx.update(self.expressions_list[idx].hyps)
                target += 1
            context_hyps = [self.expressions_list[i].tree for i in context_hyps_idx]
            fit = {k[2:]:v for k,v in fit.items()}

        unconstrained_variables = self.prop_goal_var[prop.label]

        uv_dict = {var:'UC'+var for var in unconstrained_variables}

        goal = prop.tree.copy().replace_values(uv_dict).replace(fit)
        std_context = self.standardize_context(goal, context_hyps, fit, uv_dict)
        goal = goal.replace_values(std_context.replacement_dict)
        if len(unconstrained_variables) > 0:
            if gen_cheat:
                goal = prop.tree.copy().replace(fit_gt)
            else:
                out = self.interface.apply_prop(goal, std_context, prop.label, n=1) # return value and tree
                if out is None:
                    if self.args.train_rl:
                        for ii in range(len(self.interface.gen_logs[-1])):
                            if self.interface.gen_rewards[-1][ii] == 1:
                                self.interface.gen_rewards[-1][ii] = -1
                        for ii in range(len(self.interface.pred_logs[-1])):
                            self.interface.pred_rewards[-1][ii] = -1
                    return
                goal = out[1]
        reverse_replacement_dict = {v:k for k,v in self.replacement_dict.items()}
        goal = goal.replace_values(reverse_replacement_dict)

        prior = [self.expressions_list[i] for i in prior]
        new_d = self.verify_d(prop, prior, fit)
        if new_d is None:
            if self.args.train_rl:
                for ii in range(len(self.interface.gen_logs[-1])):
                    if self.interface.gen_rewards[-1][ii] == 1:
                        self.interface.gen_rewards[-1][ii] = -1
                for ii in range(len(self.interface.pred_logs[-1])):
                    self.interface.pred_rewards[-1][ii] = -1
            return
        unconstrained = [fit[f] for f in prop.unconstrained_variables]
        expr = Expression(goal, prior, prop, 'p', unconstrained, f = self.all_f, d = new_d)
        if expr.id not in self.expressions:

            number = self.add_expression(expr)
            self.num_new_expr += 1
        if prop_cheat and goal == step.tree:
            self.logs[self.expressions[expr.id].number] = (context_label, self.prop_next_step[context_label])
            self.step_expr_pos[context_label].append(self.expressions[expr.id].number)
            self.prop_next_step[context_label] += 1
            self.num_gt_new_expr += 1
            if self.prop_next_step[context_label] == len(self.propositions[context_label].entails_proof_steps):
                del self.prop_next_step[context_label]
        return self.expressions[expr.id].number

    def generate(self):
        if self.args.random_generate:
            return self.random_generate()
        else:
            return self.parameterized_generate()

    def search(self, tree, f=None, var=[], k=None):
        if f is None: f = self.all_f
        if k == None: k = self.args.sample_width
        idx = self.searcher.tree_match_l(tree, f, var) # we find trees which are larger than tree for all vars.
        random.shuffle(idx)                                # if the true f is part of all_f, the other vars should remain the same.
        result = []                                        # we use tree.fit to filter the trees out of this requirement.
        for i in idx:
            fit = self.expressions_list[i].tree.fit(tree, f)
            if fit is not None:
                result.append((i, fit))
            if len(result) == k:
                break
        return result

    def savedata(self):
        per_file = 10000
        num_files = len(self.expressions)//per_file + 1
        for i in range(num_files):
            fpath = os.path.join(os.path.join(self.args.data_path, 'expressions'), str(i))
            torch.save(self.expressions_list[i*per_file::min((i+1)*per_file, len(self.expressions))])
            print (i, fpath)

    def loaddata(self):
        pass

    def encode_pred_step(self, statement, f, hyps, random_replacement_dict=None):
        if statement is None:
            return [self.config.encode['UC']], np.zeros((1,4))
        if random_replacement_dict is None:
            random_replacement_dict = self.config.lm.random_replacement_dict_f(f=f)
        statement = statement.copy().replace_values(random_replacement_dict)
        hyps = [h.copy().replace_values(random_replacement_dict) for h in hyps]
        statement_graph_structure = TreeInformation([statement],
                start_symbol=None, intermediate_symbol='END_OF_HYP',
                end_symbol='END_OF_SECTION')
        hyps_graph_structure = TreeInformation(hyps,
                start_symbol=None, intermediate_symbol='END_OF_HYP',
                end_symbol='END_OF_SECTION')
        in_string, structure_data = merge_graph_structures_new([statement_graph_structure, hyps_graph_structure])
        tokens = [self.config.encode[t] for t in in_string]
        return tokens, structure_data
    def extract_from_graph(self, graph):
        string = graph.string
        structure_data = list(zip(graph.depth, graph.parent_arity, graph.leaf_position, graph.arity))
        return string, structure_data

    def encode_gen_step(self, tree, unconstrained, prop, f, hyps, _all_var=False):
        unconstrained_variables = prop.unconstrained_variables if tree is not None else [l.label for l in prop.hyps if l.type=='f']
        uv_dict = {var:'UC'+var for var in unconstrained_variables}
        out_trees = [t.copy() for t in unconstrained] # ground truth trees
        if tree is not None:
            fit = prop_applies_to_statement(tree, prop, None)
        else:
            # we have no goal to fit now; use to generate the root
            fit = {}
        to_prove_trees = [hyp.tree.copy().replace(fit).replace_values(uv_dict)
            for hyp in prop.hyps if hyp.type == 'e']
        random_replacement_dict = self.config.lm.random_replacement_dict_f(f=f)
        known_trees = [h.copy() for h in hyps]
        for tree in known_trees: # empty for the root
            tree.replace_values(random_replacement_dict)
        for tree in to_prove_trees: # the same as prop.e for the root
            tree.replace_values(random_replacement_dict)
        for tree in out_trees: # empty for the root
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
        known_tokens, known_trees = self.extract_from_graph(known_graph_structure)
        to_prove_tokens, to_prove_trees = self.extract_from_graph(to_prove_graph_structure)
        if _all_var:
            uv_pos = {}
            _pos = in_string.index('END_OF_SECTION')
            for i in range(len(unconstrained_variables)):
                uv_pos[unconstrained_variables[i]] = []
                for j in range(_pos, len(in_string)):
                    if in_string[j] == uv_dict[unconstrained_variables[i]]:
                        uv_pos[unconstrained_variables[i]].append(j)
                        in_string[j] = 'UC'
            in_tokens = [self.config.encode[t] for t in in_string]
            subs_tokens = []
            subs_trees = []
            for i, graph in enumerate(out_graph_structure):
                string, structured_data = merge_graph_structures_new([graph])
                tokens = [self.config.encode[t] for t in string]
                subs_tokens.append(tokens)
                subs_trees.append(structured_data)
            return in_tokens, in_structured_data, subs_tokens, subs_trees, uv_pos
        uv_pos = []
        idx = random.randrange(len(out_trees))
        in_string = to_prove_tokens
        for i in range(len(unconstrained_variables)):
            for j in range(len(in_string)):
                if in_string[j] == uv_dict[unconstrained_variables[i]]:
                    if i == idx:
                        uv_pos.append(j)
                    in_string[j] = 'UC'
        known_tokens = [self.config.encode[t] for t in known_tokens]
        to_prove_tokens = [self.config.encode[t] for t in to_prove_tokens]
        string, structured_data = merge_graph_structures_new([out_graph_structure[idx]])
        tokens = [self.config.encode[t] for t in string]
        outputs = (tokens, structured_data, uv_pos)
        return (known_tokens, known_trees), (to_prove_tokens, to_prove_trees), outputs

    def encode_pred_tasks(self, exprs):
        # take proof tasks, return tensor as input of networks
        goals1 = []
        goals2 = []
        props1 = []
        props2 = []
        lengths = []
        pos_idxes = []
        for expr in exprs:
            hyps = [self.expressions_list[i].tree for i in expr.hyps]
            f = expr.f
            for idx in expr.hyps:
                f.update(self.expressions_list[idx].f)
            tokens, structure_data = self.encode_pred_step(expr.tree, f, hyps)
            goal_tokens = tokens
            goal_tree = structure_data
            props = [expr.prop.label]
            labels = self.prop_searcher.search(expr.tree, None, vclass='|-')
            if len(labels) < 5:
                labels += random.sample(self.propositions.keys(), 5-len(labels))
            if self.args.allneg:
                pos_idx = labels.index(expr.prop.label)
                if self.args.thm_emb:
                    props_tokens = [self.propositions[l].number for  l in labels if l in self.propositions]
                    props_tree = []
                else:
                    props_tokens = [self.prop_input[l][0] for l in labels]
                    props_tree = [self.prop_input[l][1] for l in labels]
            else:
                labels.remove(props[0])
                labels = [l for l in labels if l in self.propositions]
                if len(labels) < self.args.negative_samples:
                    labels_all = list(self.lm.database.propositions.keys())
                    labels = labels + random.sample(labels_all, self.args.negative_samples-len(labels))

                assert len(labels) >= self.args.negative_samples
                props += random.sample(labels, self.args.negative_samples)

                if self.args.thm_emb:
                    props_tokens = [self.propositions[l].number for l in props]
                    props_tree = []
                else:
                    props_tokens = [self.prop_input[l][0] for l in props]
                    props_tree = [self.prop_input[l][1] for l in props]
            goals1.append(goal_tokens)
            goals2.append(goal_tree)
            props1 += props_tokens
            props2 += props_tree
            if self.args.allneg:
                lengths.append(len(labels))
                pos_idxes.append(pos_idx)
        if self.args.allneg:
            return goals1, goals2, props1, props2, lengths, pos_idxes
        return goals1, goals2, props1, props2

    def encode_gen_tasks(self, exprs):
        k1 = []
        k2 = []
        t1 = []
        t2 = []
        out1 = []
        out2 = []
        for expr in exprs:
            hyps = [self.expressions_list[i].tree for i in expr.hyps]
            f = expr.f
            for idx in expr.hyps:
                f.update(self.expressions_list[idx].f)
            for idx in expr.prior:
                f.update(self.expressions_list[idx].f)
            k, t, outputs = self.encode_gen_step(expr.tree, expr.unconstrained, expr.prop, f, hyps)

            for i in outputs[2]:
                t[0][i] = self.config.encode['TARGET_UC']
            k1.append(k[0])
            k2.append(k[1])
            t1.append(t[0])
            t2.append(t[1])

            out1.append(outputs[0])
            out2.append(outputs[1])
        return k1, k2, t1, t2, out1, out2

def worker(generator, queue, task='pred'):
    args = generator.args
    num = args.cons_batch
    if not args.random_generate:
        expressions = generator.expressions_list[generator.num_initial_expr:]
        if task == 'gen':
            expressions = [e for e in expressions if len(e.unconstrained) > 0]
    while True:
        exprs = []
        while len(exprs) < num:
            if args.random_generate:
                expr = generator.random_generate(task=='gen')
            else:
                expr = random.choice(expressions)
            if task == 'gen' and len(expr.unconstrained) == 0:
                continue
            exprs.append(expr)
        if task == 'pred':
            queue.put(generator.encode_pred_tasks(exprs))
        else:
            queue.put(generator.encode_gen_tasks(exprs))
