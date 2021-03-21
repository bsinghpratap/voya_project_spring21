import pickle
import sys
from gensim.models.ldamulticore import LdaMulticore


use1a = (sys.argv[1] == "1a")

path = "/mnt/nfs/scratch1/hshukla/"
year_start = 2012
year_end = 2016

params = {
    'num_topics': 30,
    'chunksize': 2000,
    'passes': 20,
    'iterations': 400,
    'eval_every': None,
    'alpha': 'asymmetric',
    'eta': 'auto'
}



base = path + str(year_start) + "_" + str(year_end) + "_"

if use1a:
    with open(base + "item1a_corpus.pkl", 'rb') as pickle_file:
        corpus = pickle.load(pickle_file)
    with open(base + "item1a_id2word.pkl", 'rb') as pickle_file:
        id2word = pickle.load(pickle_file)
else:
    with open(base + "item7_corpus.pkl", 'rb') as pickle_file:
        corpus = pickle.load(pickle_file)
    with open(base + "item7_id2word.pkl", 'rb') as pickle_file:
        id2word = pickle.load(pickle_file)




lda = LdaMulticore(
        corpus=corpus,
        id2word=id2word,
        workers=8,
        **params
    )

if use1a:
    lda.save(path +"item1a")
else:
    lda.save(path +"item7")