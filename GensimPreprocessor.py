#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.models.ldamulticore import LdaMulticore

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

argv = sys.argv[1:]

opts, args = getopt.getopt(argv, '', ['start=', 'end='])

ARG_MAP = {
    'start': 2009,
    'end': 2020,
#     'bigrams': True,
#     'bigram-min-count': 20,
#     'filter-no-below': 20
}

for opt, val in opts:
    ARG_MAP[opt[2:]] = val

FILE_PATH = '../Files/gensim/'

# Selected years
SELECTED_YEARS = list(range(int(ARG_MAP['start']), int(ARG_MAP['end'])+1))
print('Processing data from years:')
print(SELECTED_YEARS)

# Add bigrams and trigrams
ADD_BIGRAMS = True

# Only add bigrams that appear BIGRAMS_MIN_COUNT times or more
BIGRAMS_MIN_COUNT = 20

# Filter out words that occur in less than FILTER_NO_ABOVE documents
FILTER_NO_BELOW = 20


# In[17]:

from util import load_data
data, X, y = load_data()


from gensim.parsing.preprocessing import STOPWORDS
from nltk.corpus import stopwords
import nltk
nltk.download("stopwords")
cachedWords = stopwords.words('english')

stopwords_total = set(list(STOPWORDS) + cachedWords)
# display(stopwords_total) # No financial terms


# In[18]:


data.query('year_x in @SELECTED_YEARS', inplace=True)
del X
del y
items = {
    'item1a': data['item1a_risk'],
    'item7': data['item7_mda']
}


# In[19]:


print(f'Got {data.shape[0]} documents')


# In[20]:


tokenizer = RegexpTokenizer(r'\w+')
for item in items:
    docs = items[item]
    for idx in range(len(docs)):
        docs.iloc[idx] = docs.iloc[idx].lower()  # Convert to lowercase.
        docs.iloc[idx] = tokenizer.tokenize(docs.iloc[idx])  # Split into words.
        docs.iloc[idx] = docs.iloc[idx][4:] # Remove first 4 words
    
    # Remove numbers, but not words that contain numbers.
    docs = [[token for token in doc if not token.isnumeric()] for doc in docs]

    # Remove words that are only one character.
    docs = [[token for token in doc if len(token) > 1] for doc in docs]
    
    lemmatizer = WordNetLemmatizer()
    docs = [[lemmatizer.lemmatize(token) for token in doc] for doc in docs]
    
    # Add bigrams and trigrams to docs (only ones that appear BIGRAMS_MIN_COUNT times or more).
    if ADD_BIGRAMS:
        bigram = Phrases(docs, min_count=BIGRAMS_MIN_COUNT)
        for idx in range(len(docs)):
            for token in bigram[docs[idx]]:
                if '_' in token:
                    # Token is a bigram, add to document.
                    docs[idx].append(token)
                
    items[item] = docs


# In[21]:


dictionaries = {}
for item in items:
    dictionaries[item] = Dictionary(items[item])


# In[22]:


for dictionary in dictionaries.values():
    # Filter out words that occur in less than FILTER_NO_BELOW documents.
    dictionary.filter_extremes(no_below=FILTER_NO_BELOW)
    
#     dictionary.filter_extremes(no_below=20, no_above=0.1)


# In[23]:


corpus = {}
for item in items:
    corpus[item] = [dictionaries[item].doc2bow(doc) for doc in items[item]]


# In[24]:


for item in items:
    print(item + ':')
    print('\tNumber of unique tokens: %d' % len(dictionaries[item]))
    print('\tNumber of documents: %d' % len(corpus[item]))


# In[25]:


id2word = {}
for item in items:
    temp = dictionaries[item][0] # Initialize id2token mappings
    id2word[item] = dictionaries[item].id2token


# ### Write output to file

# In[26]:


import pickle

base_name = str(SELECTED_YEARS[0])
if len(SELECTED_YEARS) > 1: base_name += f'-{SELECTED_YEARS[-1]}'
base_name += '_{}_{}.pkl'

str_mapping = {
    'corpus': corpus,
    'id2word': id2word
}

for item in items:
    for obj in str_mapping:
        with open(FILE_PATH+base_name.format(item, obj), 'wb') as file:
            pickle.dump(str_mapping[obj][item], file)


# In[ ]:
