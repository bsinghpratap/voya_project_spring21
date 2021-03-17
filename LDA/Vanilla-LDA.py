#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import numpy as np
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.models.ldamulticore import LdaMulticore

from gensim.models import Phrases
from gensim.parsing.preprocessing import preprocess_string, remove_stopwords, strip_numeric, strip_punctuation, strip_short, stem_text
from gensim.test.utils import common_corpus
import matplotlib.pyplot as plt


# In[9]:


import sys
sys.path.append('../')
from util import load_data, load_gensim_data


# In[12]:


YEARS = (2012, 2016)
corpus, id2word = load_gensim_data(YEARS, path='../../Files/gensim/')


# In[13]:


params = {
    'num_topics': 30,
    'chunksize': 2000,
    'passes': 20,
    'iterations': 400,
    'eval_every': None,
    'alpha': 'symmetric',
    'eta': 'auto'
}


# In[ ]:

if __name__ == '__main__':
    models = {}
    for item in corpus:
        models[item] = LdaMulticore(
            corpus=corpus[item],
            id2word=id2word[item],
            workers=16,
            **params
        )
        print(f'Finished training {item} model')
        models[item].save(f'./{YEARS[0]}-{YEARS[1]}_{item}.gnsm')

        print(models['1a'].show_topics(num_topics=5, num_words=5, formatted=True))


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

