import data_utils
import constructor
import params
import random
import torch
import pickle
import time
import os

args = params.get_args()
#args.partial_lm = True
#args.num_props = 4248
#args.gen_lm = True

if args.partial_lm:
    lm = data_utils.build_language_model(args.num_props, new=args.gen_lm, _all=False)
    config = data_utils.get_config(lm)
else:
    lm = data_utils.load_language_model(new=args.gen_lm, iset=args.iset)
    config = data_utils.get_config(lm)
    data_utils.load_proof_steps_into_lm(lm, iset=args.iset)
#labels = list(lm.database.propositions.keys())

step_hyps = {}
prop_goal_var = {}
goal_unconstrained = {}
step_prior_idx = {}
for k,p in lm.database.propositions.items():
    prop_goal_var[k] = []
    for hyps in p.hyps:
        if hyps.type == 'f':
            flag = True
            for _,e in p.e.items():
                s = e.tree.list()
                if hyps.label in s:
                    flag = False
                    break
            if flag:
                prop_goal_var[k].append(hyps.label)

for k,p in lm.database.propositions.items():
    if p.type != 'p' or p.vclass != '|-':
        continue
    print (p.number)
    step_hyps[k] = []
    step_prior_idx[k] = []
    trees = {}
    for l,e in p.e.items():
        _id = str(e.tree)
        trees[_id] = [set([l]), e.label]
    for num,step in enumerate(p.entails_proof_steps):
        fit = step.tree.fit(step.prop.tree, step.prop.f)
        for i,var in enumerate(step.prop.unconstrained_variables):
            fit[var] = step.unconstrained[i]
        for hyps in step.prop.hyps:
            if hyps.type == 'f':
                assert hyps.label in fit
        es = set([])
        idxes = []
        for hyps in step.prop.hyps:
            if hyps.type == 'e':
                tree = hyps.tree.copy().replace(fit)
                if str(tree) not in trees:
                    print (l, trees, tree)
                es.update(trees[str(tree)][0])
                idxes.append(trees[str(tree)][1])
        trees[str(step.tree)] = [es.copy(), num]
        step_hyps[k].append(es.copy())
        step_prior_idx[k].append(idxes.copy())

for k,p in lm.database.propositions.items():
    if p.type != 'p' or p.vclass != '|-':
        continue
    print (p.number)
    goal_unconstrained[k] = []
    for step in p.entails_proof_steps:
        fit = step.tree.fit(step.prop.tree, step.prop.f)
        unconstrained = []
        for f in prop_goal_var[step.prop.label]:
            unconstrained.append(fit[f])
        goal_unconstrained[k].append(unconstrained)

data = {
    'step_hyps':step_hyps,
    'step_prior_idx': step_prior_idx,
    'goal_unconstrained': goal_unconstrained,
    'prop_goal_var': prop_goal_var
    }

torch.save(data, os.path.join(args.data_path, 'forward_step_info_%.1f' % (args.data_sampling)))

generator = constructor.Constructor(args, config)#, models=models)
generator.initialize()

torch.save(generator.substitutions, os.path.join(args.data_path, 'substitutions_%.1f'%(args.data_sampling)))

def find_visible_expr(generator, worker_id, num_workers):
    hyps_visible_expr = {}
    t = time.time()
    trees = {}
    props = [p for p in lm.database.propositions_list if p.vclass == '|-']
    print (len(props))
    for i,p in enumerate(props):
        if i % num_workers != worker_id:
            continue
        print (p.label, len(hyps_visible_expr), time.time()-t)
        hyps_visible_expr[p.label] = {}
        for l,e in p.e.items():
            _id = str(e.tree)
            if _id not in trees:
                result = generator.search(e.tree, k=2000)
                trees[_id] = [s[0] for s in result]
            hyps_visible_expr[p.label][l] = trees[_id]
    torch.save(hyps_visible_expr, os.path.join(args.data_path, 'visible_expr_%d_%.1f' % (worker_id, args.data_sampling)))

import multiprocessing as mp

plist = []
num_workers = args.num_provers
for i in range(num_workers):
    plist.append(
            mp.Process(target=find_visible_expr, args=(generator, i, num_workers))
            )
    plist[-1].start()

for p in plist:
    p.join()

hyps_visible_expr = {}
for i in range(num_workers):
    data = torch.load(os.path.join(args.data_path, 'visible_expr_%d_%.1f' % (i, args.data_sampling)))
    hyps_visible_expr.update(data)
torch.save(hyps_visible_expr, os.path.join(args.data_path, 'visible_expr_%.1f' % (args.data_sampling)))

print ('save visible exprs')
data_utils.save_exprs(generator.expressions_list, lm, os.path.join(args.data_path, 'expressions_list_%.1f' % (args.data_sampling)))
torch.save(generator.step_expr_pos, os.path.join(args.data_path, 'step_expr_pos_%.1f' % (args.data_sampling)))
torch.save(generator.prop_hyps_pos, os.path.join(args.data_path, 'prop_hyps_pos_%.1f' % (args.data_sampling)))
print ('save expressions_list')
'''
splits=['train', 'valid', 'test']
for split in splits:
    fn = os.path.join('../data', split)
    fl = [i for i in os.listdir(fn) if i.isdigit()]
    for f in fl:
        #steps = load_proof_steps(os.path.join(fn, f), lm)
        with open(os.path.join(fn, f), 'rb') as fin:
            steps = pickle.load(fin)
        print (split, f, len(steps))
        data = []
        for step in steps:
            data.append(
                    (step_hyps[step.context_label][step.pos_in_context],
                     goal_unconstrained[step.context_label][step.pos_in_context])
                    )
        torch.save(data, os.path.join(fn, f+'_hyps'))
'''
