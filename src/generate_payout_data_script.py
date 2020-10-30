import time
from tree_parser import *
import data_utils as my_utils
from data_utils5 import *
import os
import withpool
import params

args = params.get_args()
i = args.prover_id
filename = args.output+str(i)
with open(filename, 'wb') as handle:
    pickle.dump(None, handle)


#text = file_contents()
#database = meta_math_database(text,n=40000, remember_proof_steps=True)
if args.partial_lm:
    lm = my_utils.build_language_model(args.num_props, new=args.gen_lm, _all=False)
else:
    lm = my_utils.load_language_model(new=args.gen_lm, iset=args.iset)
    my_utils.load_proof_steps_into_lm(lm, iset=args.iset)
    #lm.database.remember_proof_steps = True
    #lm.database.propositions_list = [p for p in lm.database.propositions_list if p.entails_proof_steps is not None and (len(p.entails_proof_steps)==0 or p.entails_proof_steps[0] is not None)]
    #print (len(lm.database.propositions_list))
    #lm.database.propositions = {}
    #for p in lm.database.propositions_list:
    #    lm.database.propositions[p.label] = p
    #print ()
    #lm = LanguageModel(lm.database, _old=True)

saved_interface = None



# import
import build_payout_data_set as pd
pd.initialize_interface(args, lm, args.output)


valp = lm.validation_propositions
testp = lm.test_propositions

trainp = []
num = int(args.data_sampling * 10)
for ii,p in enumerate(lm.training_propositions):
    if ii % 10 < num:
        trainp.append(p)
#else:
#    trainp = lm.training_propositions
print (len(trainp))
#assert args.num_provers % 10 == 0
#num_train_chunks = args.num_provers // 10 * 8
#num_valid_chunks = args.num_provers // 10
#num_test_chunks = args.num_provers // 10
#if args.prover_id < num_train_chunks:

chunk_size = len(trainp)//16
chunk_size1 = len(valp)//2
chunk_size2 = len(testp)//2
chunks = [
valp[:chunk_size1],
valp[chunk_size1:],
testp[:chunk_size2],
testp[chunk_size2:],
trainp[:chunk_size],
trainp[chunk_size:2*chunk_size],
trainp[2*chunk_size:3*chunk_size],
trainp[3*chunk_size:4*chunk_size],
trainp[4*chunk_size:5*chunk_size],
trainp[5*chunk_size:6*chunk_size],
trainp[6*chunk_size:7*chunk_size],
trainp[7*chunk_size:8*chunk_size],
trainp[8*chunk_size:9*chunk_size],
trainp[9*chunk_size:10*chunk_size],
trainp[10*chunk_size:11*chunk_size],
trainp[11*chunk_size:12*chunk_size],
trainp[12*chunk_size:13*chunk_size],
trainp[13*chunk_size:14*chunk_size],
trainp[14*chunk_size:15*chunk_size],
trainp[15*chunk_size:]
]
print (len(chunks), i)
chunk = chunks[i]

def process_chunk(x):
    i,j = x
    print ('on chunk', i, 'item', j, '/', len(chunks[i]))
    return pd.PropositionsData(chunks[i][j])

if True:
    allpds = []
    for j in range(len(chunk)):
        allpds.append(process_chunk((i,j)))
    # do the stuff.
    #with withpool.Pool(None) as pool:
    #    allpds = pool.map(process_chunk, [(i,j) for j in range(len(chunk))], chunksize=1)

    print ('saving '+filename)
    with open(filename, 'wb') as handle:
        pickle.dump(allpds, handle)

#
# def validation_data(n):
#     print 'starting item',n
#     return pd.PropositionsData(valp[n])

''' let's do this in chunks


import withpool
with withpool.Pool(8) as pool:
    start = time.time()
    allpds = {}
    allpds['validation'] = pool.map(validation_data, range(len(valp)), chunksize=1)
    print (time.time()-start), (time.time()-start)/len(valp)
    allpds['test'] = pool.map(test_data, range(len(testp)), chunksize=1)
    print (time.time()-start), (time.time()-start)/(len(valp)+len(testp))
    allpds['training'] = pool.map(training_data, range(len(trainp)), chunksize=1)
    print (time.time()-start), (time.time()-start)/(len(valp)+len(testp)+len(trainp))


print 'saving database'

import cPickle as pickle
with open('payout_data','wb') as handle:
    pickle.dump(allpds, handle)
    '''
