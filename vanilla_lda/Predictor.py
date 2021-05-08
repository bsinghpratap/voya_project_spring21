import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import f1_score, cohen_kappa_score, plot_roc_curve, plot_precision_recall_curve
from sklearn.model_selection import ParameterGrid

from gensim.corpora import Dictionary
from gensim.models.ldamulticore import LdaMulticore

from imblearn.over_sampling import SMOTE

sys.path.append(os.getenv('VOYA_PATH_PROJECT'))
from util import load_data, load_gensim_data, get_args, load_models, load_weights_file, save_weights_file, SECTORS, ITEMS
from GensimPreprocessor import process

#%% Params
# TARGET_VALUE = "is_dps_cut"
# TARGET_VALUE_TYPE = bool
TARGET_VALUE = "is_dps_cut"
TARGET_VALUE_TYPE = bool

# TODO: Remove

RF_PARAMS = {
    'n_estimators': [10, 30, 50],
    'max_depth': [1, 3, 4, 5, 6, 7, 8],
    'class_weight': ['balanced', None],
    'n_jobs': [8],
    'random_state': [4]
}

SMOTE_PARAMS = {
    'sampling_strategy': ['auto', 0.3, 0.5, 0.7],
    'random_state': [22]
}

CONCAT_ITEMS = True
TEST_RUN = False
LOAD_FROM_FILE = False
GLOBAL_WEIGHTS = True
GLOBAL_ESTIMATORS = True
INCLUDE_ALL = False
APPEND_ALL = False
PLOT_ROC = False

OVERSAMPLE = False

if INCLUDE_ALL and 'all' not in SECTORS:
    print("Including all sector")
    SECTORS = ['all', *SECTORS]

#%% Get command line arguments
args = {
    'start': 2017,
    'end': 2017,
    'validate': 2017,
    'predict': 2017
}
# get_args(arg_map=args)
YEARS = list(range(int(args['start']), int(args['end']) + 1))

#%% Load Dict
dictionaries = load_gensim_data(YEARS, dictionary=True)
#%% Load Data
data_all, _, _ = load_data(file='processed_data.csv')
data_all.query('year_x in @YEARS | year_x == @args.get("validate") | year_x == @args.get("predict")', inplace=True)
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
print("# validate rows: {}".format(len(data_validate)))
print("# test rows: {}".format(len(data_test)))



#%% Get Weights function
def get_weights(models, sector, item, document_list, years=YEARS, global_weights=GLOBAL_WEIGHTS, append_all=APPEND_ALL, dictionary=None):

    model = models[sector][item]

    if global_weights:
        idx_offset = SECTORS.index(sector) * model.num_topics
        output_shape = (len(document_list), model.num_topics * len(SECTORS))
    elif append_all:
        idx_offset = 0
        output_shape = (len(document_list), model.num_topics * 2)
    else:
        idx_offset = 0
        output_shape = (len(document_list), model.num_topics)

    weights = np.zeros(shape=output_shape)
    loaded_topics = load_weights_file(sector, item, years, model.num_topics)
    if INCLUDE_ALL or APPEND_ALL:
        loaded_topics_all = load_weights_file(sector, item, years, model.num_topics, all=True)
    if loaded_topics is None or ((INCLUDE_ALL or APPEND_ALL) and loaded_topics_all is None):
        corpus_list = [process(doc, single_doc=True, verbose=False) for doc in
                       document_list['item1a_risk' if item == 'item1a' else 'item7_mda']]
    else:
        corpus_list = document_list  # Should never have to access
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
def get_preds(sector, item, params, years=YEARS, test_year=args['validate'], test_data=data_validate, aggregator=None, plot_roc=PLOT_ROC):
    """:returns (predictions, true_values)"""

    # Define models
    if aggregator is None:
        if TARGET_VALUE_TYPE is bool:
            estimator = RandomForestClassifier(**params['estimator'])
        else:
            estimator = RandomForestRegressor(**params['estimator'])

    if type(item) is list:
        dictionary = dictionaries[sector]
    else:
        dictionary = dictionaries[sector][item]
    # Get weights for training
    # X_train_docs = data_train[(data_train.year_x in years) & (data_train.sector == sector)]
    if sector != 'all':
        X_train_docs = data_train.query('year_x in @years & sector == @sector')
    else:
        X_train_docs = data_train.query('year_x in @years')
    # X_train_corpus = [process(doc, single_doc=True, dictionary=dictionary, verbose=False) for doc in X_train_docs['item1a_risk']]
    if type(item) is list:
        X_train_weights = np.concatenate((
            get_weights(models, sector, item[0], X_train_docs, years=years, dictionary=dictionary[item[0]]),
            get_weights(models, sector, item[1], X_train_docs, years=years, dictionary=dictionary[item[1]])), axis=1)
    else:
        X_train_weights = get_weights(models, sector, item, X_train_docs, years=years, dictionary=dictionary)
    # X_train_target = X_train_docs['is_dps_cut']

    over_sampler = SMOTE(**params['smote'])
    X_train_weights, X_train_target = over_sampler.fit_resample(X_train_weights, X_train_docs['is_dps_cut'])

    # Train model
    if aggregator is None:
        estimator.fit(X_train_weights, X_train_target)

    # Get weights for prediction
    if sector != 'all':
        X_test_docs = test_data.query('year_x == @test_year & sector == @sector')
    else:
        X_test_docs = test_data.query('year_x == @test_year')
    # X_test_corpus = [process(doc, single_doc=True, verbose=False) for doc in X_test_docs['item1a_risk']]
    if type(item) is list:
        X_test_weights = np.concatenate((
            get_weights(models, sector, item[0], X_test_docs, years=[test_year]),
            get_weights(models, sector, item[1], X_test_docs, years=[test_year])), axis=1)
    else:
        X_test_weights = get_weights(models, sector, item, X_test_docs, years=[test_year])

    # if sector != 'all':
    #     X_test_docs = data_test.query('year_x == @args.get("predict") & sector == @sector')
    # else:
    #     X_test_docs = data_test.query('year_x == @args.get("predict")')
    # X_test_corpus = [process(doc, single_doc=True, verbose=False) for doc in X_test_docs['item1a_risk']]
    # X_test_weights = get_weights(models, sector, item, X_test_corpus, years=[args['predict']])

    if aggregator:
        return {
            'X_train_weights': X_train_weights,
            'X_train_target': X_train_target,
            'X_test_weights': X_test_weights,
            'X_test_target': X_test_docs['is_dps_cut'].values
        }

    if plot_roc:
        # return X_test_weights
        plot_precision_recall_curve(
            estimator=estimator,
            X=X_test_weights,
            y=X_test_docs['is_dps_cut'].values,
            name=f'{TARGET_VALUE} {sector}-{item}'
        )
        # plt.legend('')
        plt.ylim([0, 0.2])
        plt.show()

        plot_roc_curve(
            estimator=estimator,
            X=X_test_weights,
            y=X_test_docs['is_dps_cut'].values,
            name=f'{TARGET_VALUE} {sector}-{item}'
        )
        # plt.legend('')
        plt.show()

    print("Train docs:", len(X_train_docs), "\tTest docs:", len(X_test_docs), "Weights Shape:", X_train_weights.shape, X_test_weights.shape)
    # Make predictions
    if aggregator is None:
        y_pred = estimator.predict(X_test_weights)
    return X_test_docs['is_dps_cut'].values, y_pred, estimator


#%% Train Models
def evaluate(params, sector=None, global_estimator=False, test_year=args['validate'], test_data=data_valid):

    if global_estimator:
        aggregate_file = os.getenv('VOYA_PATH_MODELS')+f"{YEARS[0]}-{YEARS[-1]}/weights/{args['start']}-{test_year}_aggregate.pkl"
        if os.path.isfile(aggregate_file):
            with open(aggregate_file, mode='rb') as file:
                aggregators = pickle.load(file)
        else:
            aggregators = {
                'X_train_weights': list(),
                'X_train_target': list(),
                'X_test_weights': list(),
                'X_test_target': list()
            }
            for sector in SECTORS:
                local_aggregators = dict()
                for item in ITEMS:
                    local_aggregators[item] = get_preds(sector, item, params, aggregator=True, test_year=test_year, test_data=test_data)
                aggregators['X_train_weights'] += np.concatenate((local_aggregators['item1a']['X_train_weights'], local_aggregators['item7']['X_train_weights']), axis=1).tolist()
                aggregators['X_train_target'] += local_aggregators['item1a']['X_train_target'].to_list()
                aggregators['X_test_weights'] += np.concatenate((local_aggregators['item1a']['X_test_weights'], local_aggregators['item7']['X_test_weights']), axis=1).tolist()
                aggregators['X_test_target'] += local_aggregators['item1a']['X_test_target'].tolist()

            with open(aggregate_file, mode='wb') as file:
                pickle.dump(aggregators, file)

        estimator = RandomForestClassifier(**params['estimator'])
        print("Fitting global estimator")
        estimator.fit(
            aggregators['X_train_weights'],
            aggregators['X_train_target']
        )
        y_pred = estimator.predict(aggregators['X_test_weights'])
        y_true = aggregators['X_test_target']
        return y_true, y_pred, estimator
    else:
        # TODO: Loop does nothing
        for item in ITEMS if not CONCAT_ITEMS else [ITEMS]:
            return get_preds(sector, item, params, test_year=test_year, test_data=test_data)


def evaluate_sector(sector=None, global_estimator=False):
    best_params = [-100, None]
    for estimator_params in ParameterGrid(RF_PARAMS):
        for smote_params in ParameterGrid(SMOTE_PARAMS):
            # print("Evaluating", params, sector if sector is not None else 'Global')
            params = {
                'estimator': estimator_params,
                'smote': smote_params
            }
            # try:
            y_true, y_pred, _ = evaluate(params, sector=sector, global_estimator=global_estimator)
            # except ValueError:
            #     print("Got value error with", params)
            #     continue
            f1 = f1_score(y_true, y_pred)
            # print("Got score:", f1)
            if f1 > best_params[0]:
                best_params[0] = f1
                best_params[1] = params
    return best_params


def save_final_results(params, sector, plot_roc=PLOT_ROC):
    # y_true, y_pred = get_preds(sector, item, params, test_year=args['predict'], test_data=data_test, plot_roc=PLOT_ROC)
    y_true, y_pred, estimator = evaluate(params, sector=sector, test_year=args['predict'], test_data=data_test, global_estimator=GLOBAL_ESTIMATORS)
    if sector is None:
        sector = 'global'

    if APPEND_ALL:
        dir_name = 'append_all'
    else:
        dir_name = 'default'

    output_file_dir = f"../results/{dir_name}/"
    output_filename = f"{args['start']}-{args['end']}_{args['validate']}_{args['predict']}_{sector}.pkl"
    if not os.path.isdir(output_file_dir):
        os.mkdir(output_file_dir)

    with open(output_file_dir + output_filename, mode='wb') as file:
        pickle.dump({
            'sector': sector,
            'params': params,
            'y_true': y_true,
            'y_pred': y_pred,
            'estimator': estimator,
            'start': args['start'],
            'end': args['end'],
            'validate': args['validate'],
            'predict': args['predict']
        }, file)
    print("Saved results for", sector, "\tF1:", f1_score(y_true, y_pred), "\tKC:", cohen_kappa_score(y_true, y_pred))


# %%
if __name__ == "__main__":
    if GLOBAL_ESTIMATORS:
        best_params = evaluate_sector(global_estimator=True)
        print(best_params)
        save_final_results(best_params[1], sector=None)
    else:
        best_params = dict()
        for sector in SECTORS:
            best_params[sector] = evaluate_sector(sector=sector)
            save_final_results(best_params[sector][1], sector)

    # estimator = RandomForestClassifier(**best_params[1])
    # estimator.fit()

    # if PLOT_ROC:
    #     plot_precision_recall_curve(
    #         estimator=estimator,
    #         X=aggregators['X_test_weights'],
    #         y=y_true,
    #         name=f'{TARGET_VALUE} ROC')
    #     plt.show()

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
# %% Generate weights
if __name__ == '__main__':
    for sector in SECTORS:
        for item in ITEMS:
            dictionary = dictionaries[sector][item]
            if sector != 'all':
                X_train_docs = data_train.query('year_x in @YEARS & sector == @sector')
            else:
                X_train_docs = data_train.query('year_x in @YEARS')
            get_weights(models, sector, item, X_train_docs, years=YEARS, dictionary=dictionary)