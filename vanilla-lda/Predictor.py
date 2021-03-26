import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from gensim.corpora import Dictionary
from gensim.models.ldamulticore import LdaMulticore

from util import load_data, load_gensim_data, get_args, load_models, SECTORS
from GensimPreprocessor import process

#%% Get command line arguments
args = {
    'start': 2012,
    'end': 2015,
    'predict': 2016
}
# get_args(arg_map=args)
years = list(range(int(args['start']), int(args['end'])+1))

#%% Load Dict
dictionaries = load_gensim_data(years, dictionary=True)
#%% Load Data
data_all, data_X, data_y = load_data(file='processed_data.csv')
#%% Load Trained Models
models = load_models(years, LdaMulticore, by_sector=True)

#%% Find subset of valid data
print("LDA on [{},{}] predicting for {}".format(years[0], years[-1], args['predict']))
data_all["is_dividend_payer"] = data_all["is_dividend_payer"].astype(bool)
data_valid = data_all[data_all["is_dividend_payer"] & data_all["is_dps_cut"].notnull()]
data_valid["is_dps_cut"] = data_valid["is_dps_cut"].astype(int)

#%% train/test data preparation
data_train = data_valid[(data_valid.year_x >= years[0]) & (data_valid.year_x <= years[-1])]
data_test = data_valid[data_valid.year_x == args['predict']]
print("# train rows: {}".format(len(data_train)))
print("# test rows: {}".format(len(data_test)))


#%% Get Weights function
def get_weights(model, corpus_list):
    weights = np.zeros(shape=(len(corpus_list), model.num_topics))
    for idx, corpus in enumerate(corpus_list):
        try:
            topics = model.get_document_topics(corpus)
        except:
            print('Got exception with:', idx, corpus)
        for topic in topics:
            weights[idx][topic[0]] = topic[1]
    return weights

#%%
def get_preds(sector, item):
    """:returns (predictions, true)"""
    # Define models
    rf = RandomForestClassifier(n_estimators=10, warm_start=False)
    dictionary = dictionaries[sector][item]
    # Get weights for training
    # X_train_docs = data_train[(data_train.year_x in years) & (data_train.sector == sector)]
    X_train_docs = data_train.query('year_x in @years' and 'sector == @sector')
    X_train_corpus = [process(doc, single_doc=True, dictionary=dictionary, verbose=False) for doc in X_train_docs['item1a_risk']]
    X_train_weights = get_weights(models[sector][item], X_train_corpus)

    # Train model
    rf.fit(X_train_weights, X_train_docs['is_dps_cut'])

    # Get weights for prediction
    X_test_docs = data_test.query('year_x in @years' and 'sector == @sector')
    X_test_corpus = [process(doc, single_doc=True, verbose=False) for doc in X_test_docs['item1a_risk']]
    X_test_weights = get_weights(models[sector][item], X_test_corpus)

    # Make predictions
    y_pred = rf.predict(X_test_weights)
    return y_pred, X_test_docs['is_dps_cut']

#%% Isolate single sector and year for development purposes TODO: Make dynamic
item = 'item1a'

#%% Train Models
preds = list()
true = list()
for sector in SECTORS:
    results = get_preds(sector, item)
    preds.append(results[0])
    true.append(results[1])
    print('Completed sector:', sector)

#%% Parse Results
preds_df = pd.DataFrame(preds[:][:])
preds_df[preds_df.isna()] = 0
# preds_df.to_csv('./predictions.csv')
true_df = pd.DataFrame(true[:][:])
true_df[true_df.isna()] = 0

#%% Get Score
f1_score(true_df, preds_df)

