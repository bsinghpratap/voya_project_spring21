#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.models.ldamulticore import LdaMulticore
import os

from gensim.models import Phrases
from gensim.parsing.preprocessing import preprocess_string, remove_stopwords, strip_numeric, strip_punctuation, strip_short, stem_text
import matplotlib.pyplot as plt


# In[2]:


from gensim import corpora
from gensim.utils import simple_preprocess
from gensim.corpora import Dictionary

'''
Run file in shell with arguments start and end.
    --start [start year]
    --end [end year]
'''

# ## Custom Year Definition
# Modify these variables for your needs

# In[4]:

# In[16]:
import sys, getopt

FILE_PATH = '../Files/gensim/'

# Add bigrams and trigrams
ADD_BIGRAMS = True

# Only add bigrams that appear BIGRAMS_MIN_COUNT times or more
BIGRAMS_MIN_COUNT = 20

# Filter out words that occur in less than FILTER_NO_ABOVE documents
FILTER_NO_BELOW = 20


# In[17]:

from util import load_data


from gensim.parsing.preprocessing import STOPWORDS
from nltk.corpus import stopwords
import nltk
nltk.download("stopwords")
cachedWords = stopwords.words('english')

stopwords_total = set(list(STOPWORDS) + cachedWords)
# display(stopwords_total) # No financial terms


def process(docs, sentence=False):
    
    tokenizer = RegexpTokenizer(r'\w+')
    for idx in range(len(docs)):
        docs.iloc[idx] = docs.iloc[idx].lower()  # Convert to lowercase.
        docs.iloc[idx] = tokenizer.tokenize(docs.iloc[idx])  # Split into words.
        docs.iloc[idx] = docs.iloc[idx][4:] # Remove first 4 words

    # Remove numbers, but not words that contain numbers.
    docs = pd.Series([[token for token in doc if not token.isnumeric()] for doc in docs])

    # Remove words that are only one character.
    docs = pd.Series([[token for token in doc if len(token) > 1] for doc in docs])

    lemmatizer = WordNetLemmatizer()
    docs = pd.Series([[lemmatizer.lemmatize(token) for token in doc] for doc in docs])

    # Add bigrams and trigrams to docs (only ones that appear BIGRAMS_MIN_COUNT times or more).
    if ADD_BIGRAMS:
        bigram = Phrases(docs, min_count=BIGRAMS_MIN_COUNT)
        for idx in range(len(docs)):
            for token in bigram[docs[idx]]:
                if '_' in token:
                    # Token is a bigram, add to document.
                    docs[idx].append(token)

    dictionary = Dictionary(docs)

    # Filter out words that occur in less than FILTER_NO_BELOW documents.
    dictionary.filter_extremes(no_below=FILTER_NO_BELOW)

    corpus = [dictionary.doc2bow(doc) for doc in docs]

    print('\tNumber of unique tokens: %d' % len(dictionary))
    print('\tNumber of documents: %d' % len(corpus))



    _temp = dictionary[0] # Initialize id2token mappings
    id2word = dictionary.id2token
    
    return corpus, id2word


def write(corpus, id2word, years, name='data', path='../Files/gensim/'):
    
    import pickle

    base_name = f'{str(years[0])}-{str(years[-1])}'
#     if len(years) > 1: base_name += f'-{years[-1]}'
    path_full = path+base_name+'/'
    
    base_name += '_{}_{}.pkl'

    str_mapping = {
        'corpus': corpus,
        'id2word': id2word
    }
    
    if not os.path.exists(path_full):
        os.makedirs(path_full)

    for obj in str_mapping:
        with open(path_full+base_name.format(name, obj), 'wb') as file:
            pickle.dump(str_mapping[obj], file)

            
if __name__ == '__main__':
    
    argv = sys.argv[1:]

    opts, args = getopt.getopt(argv, '', ['start=', 'end=', 'sectors', 'sentence'])

    ARG_MAP = {
        'start': 2009,
        'end': 2020,
        'sectors': False,
        'sentence': False
    #     'bigrams': True,
    #     'bigram-min-count': 20,
    #     'filter-no-below': 20
    }

    for opt, val in opts:
        ARG_MAP[opt[2:]] = val if val else True
    
    # Selected years
    SELECTED_YEARS = list(range(int(ARG_MAP['start']), int(ARG_MAP['end'])+1))
    print('Processing data from years:')
    print(SELECTED_YEARS)
    
    SPLIT_BY_SECTORS = ARG_MAP['sectors']
    print(f'Split by sectors: {SPLIT_BY_SECTORS}')
    
    SENTENCE = ARG_MAP['sentence']
    print(f'Processing for sentence LDA: {SENTENCE}')
    
    data, X, y = load_data()
    data.query('year_x in @SELECTED_YEARS', inplace=True)
    del X
    del y
    
    
    items = {
        'item1a': 'item1a_risk',
        'item7': 'item7_mda'
    }
    
    sectors = pd.unique(data['sector'])
    
    for item in items:
        if SPLIT_BY_SECTORS:
            for sector in sectors:
                print('Processing sector: ', sector)
                data_slice = data[(data.sector == sector)][items[item]]
                if len(data_slice) > 1 and 'Unavailable' not in str(sector): # Sector unavailable
                    corpus, id2word = process(data[(data.sector == sector)][items[item]], sentence=SENTENCE)
                    write(corpus, id2word, SELECTED_YEARS, name=sector+'_'+item, path=FILE_PATH)
                else:
                    print(f"Skipping sector {sector} with {len(data_slice)} document(s)")
        else:
            corpus, id2word = process(data[items[item]], sentence=SENTENCE)
            write(corpus, id2word, SELECTED_YEARS, name='all_'+item, path=FILE_PATH)


    print(f'Got {data.shape[0]} documents')
