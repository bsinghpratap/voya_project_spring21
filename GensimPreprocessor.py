#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.models.ldamulticore import LdaMulticore
from time import sleep
from gensim.models import Phrases
from gensim.parsing.preprocessing import preprocess_string, remove_stopwords, strip_numeric, strip_punctuation, strip_short, stem_text
import matplotlib.pyplot as plt
import pickle

from multiprocessing import Pool

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
import os

FILE_PATH = '../Files/gensim/'
SLEEP_TIME = 100
# Add bigrams and trigrams
ADD_BIGRAMS = True

# Only add bigrams that appear BIGRAMS_MIN_COUNT times or more
BIGRAMS_MIN_COUNT = 20

# Filter out words that occur in less than FILTER_NO_ABOVE documents
FILTER_NO_BELOW = 20


# In[17]:

from util import load_data, SECTORS


from gensim.parsing.preprocessing import STOPWORDS
from nltk.corpus import stopwords
import nltk
nltk.download("stopwords")
cachedWords = stopwords.words('english')

stopwords_total = set(list(STOPWORDS) + cachedWords)
# display(stopwords_total) # No financial terms


def process(docs,
            sentence=False,
            return_dictionary=False,
            single_doc=False,
            verbose=True,
            dictionary=None,
            return_docs=False):
    if single_doc:
        docs = pd.Series([docs])
    else:
        docs = docs.copy()
    tokenizer = RegexpTokenizer(r'\w+')
    for idx in range(len(docs)):
        docs.iloc[idx] = docs.iloc[idx].lower()  # Convert to lowercase.
        docs.iloc[idx] = tokenizer.tokenize(docs.iloc[idx])  # Split into words.
        docs.iloc[idx] = docs.iloc[idx][4:] # Remove first 4 words

    # if single_doc:
    #     docs = pd.Series([docs])

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

    if return_docs:
        return docs

    if dictionary is None:
        dictionary = Dictionary(docs)

    # Filter out words that occur in less than FILTER_NO_BELOW documents.
    if not single_doc:
        dictionary.filter_extremes(no_below=FILTER_NO_BELOW)

    corpus = [dictionary.doc2bow(doc) for doc in docs]

    if verbose:
        print('\tNumber of unique tokens: %d' % len(dictionary))
        print('\tNumber of documents: %d' % len(corpus))

    if not single_doc:
        _temp = dictionary[0]  # Initialize id2token mappings
        id2word = dictionary.id2token
        if not return_dictionary:
            return corpus, id2word
        else:
            return corpus, id2word, dictionary
    else:
        return corpus[0]


def write(obj, years, name='data', path=os.getenv('VOYA_PATH_DATA_GENSIM'), dictionary=False):
    base_name = f'{str(years[0])}-{str(years[-1])}'
    path_full = path+base_name+'/'
    base_name += '_{}_{}'

    if not os.path.exists(path_full):
        os.makedirs(path_full)

    if not dictionary:
        base_name += '.pkl'
        str_mapping = {'corpus': obj[0], 'id2word': obj[1]}
        for obj in str_mapping:
            with open(path_full + base_name.format(name, obj), 'wb') as file:
                pickle.dump(str_mapping[obj], file)
    else:
        base_name += '.gnsm'
        obj.save(path_full + base_name.format(name, 'dictionary'))


def process_sector(data_slice, sector, item, selected_years, sentence, dictionary):
    print('Processing sector: ', sector)
    if len(data_slice) > 1 and 'Unavailable' not in str(sector):
        obj = process(data_slice, sentence=sentence, return_dictionary=dictionary)
        write(obj[:2], selected_years, name=sector + '_' + item, dictionary=False)
        if dictionary:
            write(obj[2], selected_years, name=sector + '_' + item, dictionary=True)
    else:  # Sector unavailable
        print(f"Skipping sector {sector} with {len(data_slice)} document(s)")
    print('Finished processing:', item, sector)


if __name__ == '__main__':
    
    argv = sys.argv[1:]

    opts, args = getopt.getopt(argv, '', ['start=', 'end=', 'sectors', 'sentence', 'dictionary'])

    ARG_MAP = {
        'start': 2009,
        'end': 2020,
        'sectors': False,
        'sentence': False,
        'dictionary': True
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

    DICTIONARY = ARG_MAP['dictionary']
    print('Giving dictionary as output:', DICTIONARY)
    
    data, X, y = load_data()
    data.query('year_x in @SELECTED_YEARS', inplace=True)
    del X
    del y

    print(f'Got {data.shape[0]} documents for processing')

    items = {
        'item1a': 'item1a_risk',
        'item7': 'item7_mda'
    }
    
    # sectors = pd.unique(data['sector']) if SPLIT_BY_SECTORS else ['all']
    sectors = ['all', *SECTORS]

    pool = Pool()
    procs = list()
    
    for item in items:
        for sector in sectors:
            if sector == 'all':
                data_slice = data[items[item]]
            else:
                data_slice = data[(data.sector == sector)][items[item]]
            call_args = (data_slice, sector, item, SELECTED_YEARS, SENTENCE, DICTIONARY)
            procs.append(pool.apply_async(process_sector, call_args))
            sleep(SLEEP_TIME)
        # else:
        #     corpus, id2word = process(data[items[item]], sentence=SENTENCE)
        #     write(corpus, id2word, SELECTED_YEARS, name='all_'+item)

    for proc in procs:
        proc.get()
