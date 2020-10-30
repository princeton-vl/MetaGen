import os, sys
import log
import params
import constructor
import data_utils
import interface_lm
import time
import random
import torch
sys.setrecursionlimit(10000)

args = params.get_args()
_logger = log.get_logger(__name__, args)
_logger.info(data_utils.print_args(args))

lm = data_utils.load_language_model(new=True, iset=args.iset)
config = data_utils.get_config(lm)
_logger.info('load lm')

try:
    os.mkdir(args.expr_path)
except:
    pass
fl = os.listdir(args.expr_path)
for s in fl:
    if s.find('finish_%d'%(args.prover_id)) >= 0:
        os.remove(os.path.join(args.expr_path, s))

_logger.info('remove old logs')

if args.evaluate == 'none':
    interface = interface_lm.LMInterface(args, lm)
else:
    interface = None
_logger.info('build interface')

if args.data_sampling > 0:
    exprs = []
    for fp in args.expr_list:
        exprs += data_utils.load_exprs(fp, lm)
    _logger.info('load initial exprs')

    generator = constructor.Constructor(args, config, interface)
    generator.initialize_prop()
    generator.expressions_list = exprs

    for e in exprs:
        generator.expressions[e.id] = e
    generator.num_initial_expr = len(generator.expressions)
    generator.substitutions = torch.load(os.path.join(args.data_path, 'substitutions_%.1f'%(args.data_sampling)))
    generator.initialize_searcher()
else:
    generator = constructor.Constructor(args, config, interface)
    generator.initialize()
_logger.info('initialize generator with %d exprs', generator.num_initial_expr)

def load(_iter, worker_id):
    fp = os.path.join(args.expr_path, 'finish_%d_%d'%(worker_id, _iter))
    while os.path.isfile(fp) == False:
        time.sleep(2)
    return data_utils.load_exprs(os.path.join(args.expr_path, 'exprs_%d_%d'%(worker_id, _iter)), lm)

if args.evaluate != 'none':
    for j in range(args.epoch):
        exprs_new = []
        for i in range(args.num_provers):
            exprs_new += data_utils.load_exprs(os.path.join(args.expr_path, 'exprs_%d_%d'%(i, j)), lm)
            print (j, i)
        for e in exprs_new:
            generator.add_expression(e)
        _logger.warning('add all new exprs, %d expressions in total', len(generator.expressions_list))
    data_utils.save_exprs(generator.expressions_list[generator.num_initial_expr:], generator.lm, os.path.join(args.expr_path, 'exprs_final'))
    exit(0)
j = 0
while len(generator.expressions_list) - generator.num_initial_expr < args.num_cons_exprs:
    _logger.info('iter %d starts', j)
    t = time.time()
    exprs = []
    i = 0
    while time.time()-t < args.sampling_time:
        if args.random_generate:
            exprs.append( generator.random_generate(add_to_list=False))
        else:
            exprs.append( generator.parameterized_generate(add_to_list=False))
        if i % 200 == 0:
            _logger.info('%d exprs', i)
        i += 1
    _logger.info('generate expressions for iter %d', j)
    data_utils.save_exprs(exprs, generator.lm, os.path.join(args.expr_path, 'exprs_%d_%d'%(args.prover_id, j)))
    with open(os.path.join(args.expr_path, 'finish_%d_%d'%(args.prover_id, j)), 'w') as f:
        f.write ('finish %d\n' % (args.prover_id))
    _logger.info('save expressions for iter %d', j)
    exprs_new = []
    # load expressions from other worker
    for i in range(args.num_provers):
        exprs_new += load(j, i)
        _logger.info('load new exprs from worker %d', i)
    for e in exprs_new:
        generator.add_expression(e)
    _logger.warning('add all new exprs, %d expressions in total', len(generator.expressions_list))
    j += 1

suffix = '_precompute'
if args.prover_id == 0:
    if os.path.isdir(args.expr_path+suffix) == False:
        os.mkdir(args.expr_path+suffix)
    data_utils.save_exprs(generator.expressions_list[generator.num_initial_expr:], generator.lm, os.path.join(args.expr_path+suffix, 'syn_thms%d'%(len(os.listdir(args.expr_path+suffix)))))

