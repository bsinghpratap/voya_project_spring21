import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

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

# TODO: Remove

RF_PARAMS = {
    'n_estimators': 30,
    'warm_start': False,
    'max_depth': 7,
    'class_weight': 'balanced',
    'n_jobs': 8
}

SMOTE_PARAMS = {
    'sampling_strategy': 0.5,
    'random_state': 22
}

TEST_RUN = False
LOAD_FROM_FILE = False
GLOBAL_WEIGHTS = True
GLOBAL_ESTIMATORS = True
INCLUDE_ALL = False
APPEND_ALL = True
PLOT_ROC = True

OVERSAMPLE = False

if INCLUDE_ALL and 'all' not in SECTORS:
    print("Including all sector")
    SECTORS = ['all', *SECTORS]

#%% Get command line arguments
args = {
    'start': 2012,
    'end': 2015,
    'validate': 2016,
    'predict': 2017
}
# get_args(arg_map=args)
YEARS = list(range(int(args['start']), int(args['end']) + 1))

#%% Load Dict
dictionaries = load_gensim_data(YEARS, dictionary=True)
#%% Load Data
data_all, _, _ = load_data(file='processed_data.csv')
data_all.query('year_x in @YEARS | year_x == @args.get("predict")', inplace=True)
#%% Load Trained Models
models = load_models(YEARS, LdaMulticore, by_sector=True)
models['all'] = load_models(YEARS, LdaMulticore, by_sector=False)['all']

#%% Find subset of valid data
print("LDA on [{},{}] predicting for {}".format(YEARS[0], YEARS[-1], args['predict']))

if TARGET_VALUE == "is_dps_cut":

    data_all["is_dividend_payer"] = data_all["is_dividend_payer"].astype(bool)
    data_valid = data_all[data_all["is_dividend_payer"] & data_all["is_dps_cut"].notnull()]
    data_valid["is_dps_cut"] = data_valid["is_dps_cut"].astype(int)

elif TARGET_VALUE == "z_environmental":

    data_valid = data_all[data_all["z_environmental"].notnull() & data_all["z_environmental"].notna()]
    data_valid["z_environmental"] = data_valid["z_environmental"].astype(float)

if TEST_RUN:
    print("WARNING: Subsampling data for test purposes")
    data_valid = data_valid.sample(frac=0.3)

#%% train/test data preparation
data_train = data_valid[(data_valid.year_x >= YEARS[0]) & (data_valid.year_x <= YEARS[-1])]
data_validate = data_valid[data_valid.year_x == args['validate']]
data_test = data_valid[data_valid.year_x == args['predict']]
print("# train rows: {}".format(len(data_train)))
print("# test rows: {}".format(len(data_test)))


#%% Load Weights from file
def load_weights_file(sector, item, years, num_topics, all=False):
    target_path = f"{os.getenv('VOYA_PATH_MODELS')}{YEARS[0]}-{YEARS[-1]}\\weights\\"
    target_name = f'{years[0]}-{years[-1]}_{sector}_{item}_{num_topics}'
    if all: target_name += '_all'
    target_name += '.pkl'

    weights = None

    if not os.path.isdir(target_path):
        pass
    elif os.path.isfile(target_path + target_name):
        print("Loading weights from file", sector, item, years, num_topics, 'all' if all else 'sector')
        with open(target_path + target_name, mode='rb') as file:
            weights = pickle.load(file)
    return weights


def save_weights_file(sector, item, years, num_topics, weights, all=False):
    print("Saving weights to file", sector, item, years, num_topics, 'all' if all else 'sector')
    target_path = f"{os.getenv('VOYA_PATH_MODELS')}{YEARS[0]}-{YEARS[-1]}\\weights\\"
    target_name = f'{years[0]}-{years[-1]}_{sector}_{item}_{num_topics}'
    if all: target_name += '_all'
    target_name += '.pkl'

    if not os.path.isdir(target_path):
        os.mkdir(target_path)

    with open(target_path + target_name, mode='wb') as file:
        pickle.dump(weights, file)


#%% Get Weights function
def get_weights(models, sector, item, corpus_list, years=YEARS, global_weights=GLOBAL_WEIGHTS, append_all=APPEND_ALL):

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
    loaded_topics = load_weights_file(sector, item, years, model.num_topics)
    loaded_topics_all = load_weights_file(sector, item, years, model.num_topics, all=True)
    write_topics = list()
    write_topics_all = list()

    for idx, corpus in enumerate(corpus_list):
        if loaded_topics is None:
            topics = model.get_document_topics(corpus)
            write_topics.append(topics)
        else:
            topics = loaded_topics[idx]
        if append_all:
            if loaded_topics_all is None:
                topics_all = models['all'][item].get_document_topics(corpus)
                write_topics_all.append(topics_all)
            else:
                topics_all = loaded_topics_all[idx]

        for topic in topics:
            weights[idx][topic[0]+idx_offset] = topic[1]

        if append_all:
            for topic in topics_all:
                weights[idx][topic[0]+model.num_topics] = topic[1]

    if len(write_topics) > 0:
        save_weights_file(sector, item, years, model.num_topics, write_topics)
    if len(write_topics_all) > 0:
        save_weights_file(sector, item, years, model.num_topics, write_topics_all, all=True)

    return weights


#%%
def get_preds(sector, item, years=YEARS, aggregator=None, plot_roc=PLOT_ROC):
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
        X_train_docs = data_train.query('year_x in @YEARS & sector == @sector')
    else:
        X_train_docs = data_train.query('year_x in @YEARS')
    X_train_corpus = [process(doc, single_doc=True, dictionary=dictionary, verbose=False) for doc in X_train_docs['item1a_risk']]
    X_train_weights = get_weights(models, sector, item, X_train_corpus, years=YEARS)
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
    X_test_weights = get_weights(models, sector, item, X_test_corpus, years=[args['predict']])

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
                # if PLOT_ROC:
                #     plot_precision_recall_curve(
                #         estimator=estimator,
                #         X=aggregators['X_test_weights'],
                #         y=y_true,
                #         name=f'{TARGET_VALUE} ROC')
                #     plt.show()


#%% Train Global Estimators
if GLOBAL_ESTIMATORS:
    estimator = RandomForestClassifier(**RF_PARAMS)
    print("Fitting global estimator")
    estimator.fit(
        aggregators['X_train_weights'],
        aggregators['X_train_target']
    )
#%%
    y_pred = estimator.predict(aggregators['X_test_weights'])
    y_true = aggregators['X_test_target']
    print("\n\tF1:", f1_score(y_true=y_true, y_pred=y_pred))
    print("\n\tCK:", cohen_kappa_score(y_true, y_pred))

    if PLOT_ROC:
        plot_precision_recall_curve(
            estimator=estimator,
            X=aggregators['X_test_weights'],
            y=y_true,
            name=f'{TARGET_VALUE} ROC')
        plt.show()


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
