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
sys.path.append('C:\\Users\\Alan\\Projects School\\696DS\\voya_project_spring21')

from util import load_data, load_gensim_data, save_models
from util import SECTORS

# SECTORS = ['all', *SECTORS]
# In[12]:
TEST_RUN = False
WORKERS = 16


YEARS = (2017, 2017)
# YEARS = (2016, 2016)
data = load_gensim_data(YEARS)

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

    # for sector in data:
    for sector in SECTORS:
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

    save_models(models, YEARS)
