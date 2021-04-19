import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import f1_score, cohen_kappa_score, plot_roc_curve, plot_precision_recall_curve

from gensim.corpora import Dictionary
from gensim.models.ldamulticore import LdaMulticore

from imblearn.over_sampling import SMOTE

from util import load_data, load_gensim_data, get_args, load_models, SECTORS, ITEMS
from GensimPreprocessor import process

#%% Params

# TARGET_VALUE = "is_dps_cut"
# TARGET_VALUE_TYPE = bool
TARGET_VALUE = "is_dps_cut"
TARGET_VALUE_TYPE = bool

TEST_RUN = False

RF_PARAMS = {
    'n_estimators': 30,
    'warm_start': False,
    'max_depth': 3,
    'class_weight': 'balanced'
}

SMOTE_PARAMS = {
    'sampling_strategy': 0.5,
    'random_state': 22
}

GLOBAL_WEIGHTS = True
GLOBAL_ESTIMATORS = True
INCLUDE_ALL = True
APPEND_ALL = False
PLOT_ROC = False

OVERSAMPLE = False

if INCLUDE_ALL and 'all' not in SECTORS:
    SECTORS = ['all', *SECTORS]

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
models['all'] = load_models(years, LdaMulticore, by_sector=False)['all']

#%% Find subset of valid data
print("LDA on [{},{}] predicting for {}".format(years[0], years[-1], args['predict']))

if TARGET_VALUE == "is_dps_cut":

    data_all["is_dividend_payer"] = data_all["is_dividend_payer"].astype(bool)
    data_valid = data_all[data_all["is_dividend_payer"] & data_all["is_dps_cut"].notnull()]
    data_valid["is_dps_cut"] = data_valid["is_dps_cut"].astype(int)

elif TARGET_VALUE == "z_environmental":

    data_valid = data_all[data_all["z_environmental"].notnull() & data_all["z_environmental"].notna()]
    data_valid["z_environmental"] = data_valid["z_environmental"].astype(float)

if TEST_RUN:
    data_valid = data_valid.sample(frac=0.025)

#%% train/test data preparation
data_train = data_valid[(data_valid.year_x >= years[0]) & (data_valid.year_x <= years[-1])]
data_test = data_valid[data_valid.year_x == args['predict']]
print("# train rows: {}".format(len(data_train)))
print("# test rows: {}".format(len(data_test)))


#%% Get Weights function
def get_weights(models, sector, item, corpus_list, global_weights=GLOBAL_WEIGHTS, append_all=APPEND_ALL):

    model = models[sector][item]

    if global_weights:
        idx_offset = SECTORS.index(sector) * model.num_topics
        output_shape = (len(corpus_list), model.num_topics*len(SECTORS))
    elif append_all:
        idx_offset = 0
        output_shape = (len(corpus_list), model.num_topics*2)
    else:
        idx_offset = 0
        output_shape = (len(corpus_list), model.num_topics)

    weights = np.zeros(shape=output_shape)

    for idx, corpus in enumerate(corpus_list):
        try:
            topics = model.get_document_topics(corpus)
            if append_all:
                topics_all = models['all'][item].get_document_topics(corpus)
        except:
            print('Got exception with:', idx, corpus)

        for topic in topics:
            weights[idx][topic[0]+idx_offset] = topic[1]

        if append_all:
            for topic in topics_all:
                weights[idx][topic[0]+model.num_topics] = topic[1]

    return weights


#%%
def get_preds(sector, item, aggregator=None, plot_roc=PLOT_ROC):
    """:returns (predictions, true_values)"""

    # Define models
    if aggregator is None:
        if TARGET_VALUE_TYPE is bool:
            estimator = RandomForestClassifier(**RF_PARAMS)
        else:
            estimator = RandomForestRegressor(**RF_PARAMS)

    dictionary = dictionaries[sector][item]
    # Get weights for training
    # X_train_docs = data_train[(data_train.year_x in years) & (data_train.sector == sector)]
    if sector != 'all':
        X_train_docs = data_train.query('year_x in @years & sector == @sector')
    else:
        X_train_docs = data_train.query('year_x in @years')
    X_train_corpus = [process(doc, single_doc=True, dictionary=dictionary, verbose=False) for doc in X_train_docs['item1a_risk']]
    X_train_weights = get_weights(models, sector, item, X_train_corpus)
    X_train_target = X_train_docs['is_dps_cut']

    # Oversample
    if OVERSAMPLE:
        over_sampler = SMOTE(**SMOTE_PARAMS)
        X_train_weights, X_train_target = over_sampler.fit_resample(X_train_weights, X_train_docs['is_dps_cut'])

    # Train model
    if aggregator is None:
        estimator.fit(X_train_weights, X_train_target)

    # Get weights for prediction
    if sector != 'all':
        X_test_docs = data_test.query('year_x == @args.get("predict") & sector == @sector')
    else:
        X_test_docs = data_test.query('year_x == @args.get("predict")')
    X_test_corpus = [process(doc, single_doc=True, verbose=False) for doc in X_test_docs['item1a_risk']]
    X_test_weights = get_weights(models, sector, item, X_test_corpus)

    if aggregator:
        return {
            'X_train_weights': X_train_weights,
            'X_train_target': X_train_target,
            'X_test_weights': X_test_weights,
            'X_test_target': X_test_docs['is_dps_cut'].values
        }

    if plot_roc:
        plot_precision_recall_curve(
            estimator=estimator,
            X=X_test_weights,
            y=X_test_docs['is_dps_cut'].values,
            name=f'{TARGET_VALUE} {sector}-{item} ROC'
        )
        plt.show()

    print("Train docs:", len(X_test_docs), "\tTest docs:", len(X_test_docs))

    # Make predictions
    if aggregator is None:
        y_pred = estimator.predict(X_test_weights)
    return y_pred, X_test_docs['is_dps_cut'].values


#%% Isolate single sector and year for development purposes TODO: Make dynamic
item = 'item1a'

#%% Train Models
preds = dict()
true = dict()

aggregators = {
    'X_train_weights': list(),
    'X_train_target': list(),
    'X_test_weights': list(),
    'X_test_target': list()
}

for sector in SECTORS:
    preds[sector] = dict()
    true[sector] = dict()

    if GLOBAL_ESTIMATORS:
        local_aggregators = dict()
        for item in ITEMS:
            local_aggregators[item] = get_preds(sector, item, aggregator=True)
        aggregators['X_train_weights'] += np.concatenate((local_aggregators['item1a']['X_train_weights'], local_aggregators['item7']['X_train_weights']), axis=1).tolist()
        aggregators['X_train_target'] += local_aggregators['item1a']['X_train_target'].to_list()
        aggregators['X_test_weights'] += np.concatenate((local_aggregators['item1a']['X_test_weights'], local_aggregators['item7']['X_test_weights']), axis=1).tolist()
        aggregators['X_test_target'] += local_aggregators['item1a']['X_test_target'].tolist()
    else:
        for item in ITEMS:
            results = get_preds(sector, item)
            # except ValueError:
            #     print("Got value error:", sector, item)
            #     continue

            if not GLOBAL_ESTIMATORS:
                preds[sector][item] = results[0]
                true[sector][item] = results[1]
                print(sector, item,
                      "\n\tF1:", f1_score(results[1], results[0]),
                      "\n\tCK:", cohen_kappa_score(results[1], results[0])
                )


#%% Train Global Estimators
if GLOBAL_ESTIMATORS:
    estimator = RandomForestClassifier(**RF_PARAMS)
    estimator.fit(
        aggregators['X_train_weights'],
        aggregators['X_train_target']
    )
#%%
    y_pred = estimator.predict(aggregators['X_test_weights'])
    y_true = aggregators['X_test_target']
    print("\n\tF1:", f1_score(y_true=y_true, y_pred=y_pred))
    print("\n\tCK:", cohen_kappa_score(y_true, y_pred))


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
