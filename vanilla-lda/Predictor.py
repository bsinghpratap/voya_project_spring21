import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, cohen_kappa_score

from gensim.corpora import Dictionary
from gensim.models.ldamulticore import LdaMulticore

from imblearn.over_sampling import SMOTE

from util import load_data, load_gensim_data, get_args, load_models, SECTORS, ITEMS
from GensimPreprocessor import process

#%% Model Args
RF_PARAMS = {
    'n_estimators': 30,
    'warm_start': False,
    'max_depth': 3
}

SMOTE_PARAMS = {
    'sampling_strategy': 0.5,
    'random_state': 22
}

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
data_all, _, _ = load_data(file='processed_data.csv')
data_all.query('year_x in @years | year_x == @args.get("predict")', inplace=True)
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
    """:returns (predictions, true_values)"""
    # Define models
    rf = RandomForestClassifier(**RF_PARAMS)
    dictionary = dictionaries[sector][item]
    # Get weights for training
    # X_train_docs = data_train[(data_train.year_x in years) & (data_train.sector == sector)]
    X_train_docs = data_train.query('year_x in @years & sector == @sector')
    X_train_corpus = [process(doc, single_doc=True, dictionary=dictionary, verbose=False) for doc in X_train_docs['item1a_risk']]
    X_train_weights = get_weights(models[sector][item], X_train_corpus)

    # Oversample
    over_sampler = SMOTE(**SMOTE_PARAMS)
    X_train_weights_sm, X_train_target_sm = over_sampler.fit_resample(X_train_weights, X_train_docs['is_dps_cut'])
    # Train model
    rf.fit(X_train_weights_sm, X_train_target_sm)

    # Get weights for prediction
    X_test_docs = data_test.query('year_x == @args.get("predict") & sector == @sector')
    X_test_corpus = [process(doc, single_doc=True, verbose=False) for doc in X_test_docs['item1a_risk']]
    X_test_weights = get_weights(models[sector][item], X_test_corpus)

    # Make predictions
    y_pred = rf.predict(X_test_weights)
    return y_pred, X_test_docs['is_dps_cut'].values


#%% Isolate single sector and year for development purposes TODO: Make dynamic
item = 'item1a'

#%% Train Models
preds = dict()
true = dict()
for sector in SECTORS:
    preds[sector] = dict()
    true[sector] = dict()
    for item in ITEMS:
        try:
            results = get_preds(sector, item)
        except ValueError:
            print("Got value error:", sector, item)
            continue
        preds[sector][item] = results[0]
        true[sector][item] = results[1]
        print(sector, item,
              "\n\tF1:", f1_score(results[1], results[0]),
              "\n\tCK:", cohen_kappa_score(results[1], results[0])
        )


#%% Parse Results
# preds_df = pd.DataFrame(preds[:][:])
# preds_df[preds_df.isna()] = 0
# # preds_df.to_csv('./predictions.csv')
# true_df = pd.DataFrame(true[:][:])
# true_df[true_df.isna()] = 0

#%% Get Score
# for sector in SECTORS:
#     for item in ITEMS:
#         print(sector, item, "F1:", f1_score(true[sector][item], preds[sector][item]))
