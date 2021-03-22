#!/usr/bin/env python
# coding: utf-8

# In[8]:

import pandas as pd
import numpy as np
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.models.ldamulticore import LdaMulticore
from gensim.models.ldamodel import LdaModel

from gensim.models import Phrases
from gensim.parsing.preprocessing import preprocess_string, remove_stopwords, strip_numeric, strip_punctuation, strip_short, stem_text
from gensim.test.utils import common_corpus
import matplotlib.pyplot as plt

import getopt
from multiprocessing import Pool

# In[9]:
import sys
# sys.path.append('C:\\Users\\Alan\\Documents\\Projects School\\696DS\\voya_project_spring21')

from util import load_data, load_gensim_data, save_models
from util import SECTORS


# In[12]:
TEST_RUN = False
WORKERS = 16


# YEARS = (2012, 2015)
YEARS = (2012, 2015)
data = load_gensim_data(YEARS, path='../Files/gensim/')



# In[13]:


PARAMS = {
    'num_topics': 30,
    'chunksize': 2000,
    'passes': 20,
    'iterations': 400,
    'eval_every': None,
    'alpha': 'symmetric',
    'eta': 'auto'
}

#%%

# TODO: Change back
def train_model(corpus, id2word, workers=16, model_class=LdaMulticore):
    return model_class(
        corpus=corpus,
        id2word=id2word,
        workers=workers,
        **PARAMS
    )

# In[ ]:


if __name__ == '__main__':

    argv = sys.argv[1:]
    opts, args = getopt.getopt(argv, '', ['workers=', 'test'])

    ARG_MAP = {
        'workers': WORKERS,
        'test': TEST_RUN
    }
    for opt, val in opts:
        ARG_MAP[opt[2:]] = val if val else True
    TEST_RUN = ARG_MAP['test']
    WORKERS = int(ARG_MAP['workers'])

    if TEST_RUN:
        print("STARTING TEST RUN")
    print('Starting training with', WORKERS, 'workers')

    models = dict()

    for sector in data:
        models[sector] = dict()
        for item in data[sector]:

            corpus = data[sector][item]['corpus']
            id2word = data[sector][item]['id2word']

            model = train_model(corpus, id2word, workers=WORKERS)

            models[sector][item] = model

            print(f'Completed {sector}_{item}')

            if TEST_RUN:
                break
        if TEST_RUN:
            break

    # pool = Pool()
    # pool_dict = dict()

    # for sector in data:
    #     pool_dict[sector] = dict()
    #     for item in data[sector]:
    #         call_args = (data[sector][item]['corpus'], data[sector][item]['id2word'])
    #         # call_dict = dict(model_class=LdaModel)
    #         res = pool.apply_async(train_model, call_args)
    #         pool_dict[sector][item] = res
    #
    #         break  # TODO: Remove
    #     break
    #
    # models = dict()
    #
    # for sector in data:
    #     models[sector] = dict()
    #     for item in data[sector]:
    #         model = pool_dict[sector][item].get()
    #         models[sector][item] = model

    save_models(models, YEARS)


# In[7]:


# models['item1a'].print_topics(num_topics=5, num_words=5)
# models['item7'].print_topics(num_topics=5, num_words=5)


# Look at things your throwing out in filtering
# Look at strange occurences "duke"
# Look at total perecent of words that are made up
# Split by sector


# In[ ]:


# for model_name in models:
#     models[model_name].save(f'./{YEARS[0]}-{YEARS[1]}_{model_name}.gnsm')


# In[96]:


# results = []
# for doc_idx in range(item1a.shape[0]):
#     result_doc = {}
#     for item in items:
#         lda_model = models[item]
#         scores = []
#         topics = []
#         for index, score in sorted(lda_model[corpus[item][doc_idx]], key=lambda tup: -1*tup[1]):
# #             print ("Score: {}\t Topic ID: {} Topic: {}".format(score, index, lda_model.print_topic(index, 10)))
#             topics.append(dictionaries[item][index])
#             scores.append(score)
#         result_doc[item] = (topics, scores)
#     results.append(result_doc)


# In[150]:


# for item in items:
#     lda_model = models[item]
#     print(f'{item}:')
#     print(lda_model.alpha)
#     print(lda_model.eta)

