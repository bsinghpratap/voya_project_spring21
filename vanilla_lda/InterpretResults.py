import pickle
import numpy as np
from gensim.models.ldamulticore import LdaMulticore
from sklearn.ensemble import RandomForestClassifier
from util import load_data, load_models, ITEMS, load_weights_file
# %%
from vanilla_lda.Predictor import load_weights_file

# %%
YEARS = (2012, 2015)
models = load_models(YEARS, LdaMulticore, by_sector=True)
models['all'] = load_models(YEARS, LdaMulticore)['all']

# %% Load Data
args = {
    'start': 2012,
    'end': 2015,
    'validate': 2016,
    'predict': 2017
}
# Tech doc_idx 7

data_all, _, _ = load_data(file='processed_data.csv')
data_all.query('year_x in @YEARS | year_x == @args.get("validate") | year_x == @args.get("predict")', inplace=True)

# # %% Format Data
data_all["is_dividend_payer"] = data_all["is_dividend_payer"].astype(bool)
data_valid = data_all[data_all["is_dividend_payer"] & data_all["is_dps_cut"].notnull()]
data_valid["is_dps_cut"] = data_valid["is_dps_cut"].astype(int)
#
# data_train = data_valid[(data_valid.year_x >= YEARS[0]) & (data_valid.year_x <= YEARS[-1])]
# data_validate = data_valid[data_valid.year_x == args['validate']]
data_test = data_valid[data_valid.year_x == args['predict']]

# target_sector = 'Tech'
# document_idx = 7

# %% Set Target sector
target_sector = 'Tech'

# %% Load Real Estate good results
with open(f'../results/append_all/2012-2015_2016_2017_{target_sector}.pkl', mode='rb') as file:
    results = pickle.load(file)

# %% Print indices of correctly classified cuts
correctly_classified_idxs = list()
for idx, y_values in enumerate(zip(results['y_true'], results['y_pred'])):
    y_true, y_pred = y_values
    if y_true == 1 and y_pred == 1:
        correctly_classified_idxs.append(idx)
print(correctly_classified_idxs)

# %% Set Target idx
document_idx = 31

# Select company from data

# print("# train rows: {}".format(len(data_train)))
# print("# validate rows: {}".format(len(data_validate)))
# print("# test rows: {}".format(len(data_test)))
pass
# %% Get weights for specific sector
weights_1a = load_weights_file(target_sector, 'item1a', YEARS, 30, YEARS=YEARS)
weights_7 = load_weights_file(target_sector, 'item7', YEARS, 30, YEARS=YEARS)
weights_1a_all = load_weights_file('all', 'item1a', YEARS, 30, YEARS=YEARS)
weights_7_all = load_weights_file('all', 'item7', YEARS, 30, YEARS=YEARS)

# %% Look at good prediction for second document
weights_sel = {
    target_sector: {
        'item1a': weights_1a[document_idx],
        'item7': weights_7[document_idx]
    },
    'all': {
        'item1a': weights_1a_all[document_idx],
        'item7': weights_7_all[document_idx]
    }
}
# weights_1a_sel = weights_1a[document_idx]
# weights_7_sel = weights_7[document_idx]
# weights_1a_all_sel = weights_1a_all[document_idx]
# weights_7_all_sel = weights_7_all[document_idx]


# %% Get topic descriptions
company_row = data_test[data_test.sector == target_sector].iloc[document_idx]
print("Ticker:", company_row['ticker_x'], '-', company_row['company_name'])

estimator = results['estimator']

feature_importances = estimator.feature_importances_
# fi_zipped = list(zip(range(len(feature_importances)), feature_importances))
# fi_zipped = np.array(sorted(fi_zipped, key=lambda x: x[1], reverse=True))
# for sector in [target_sector]:
for sector in [target_sector, 'all']:
    for item in ITEMS:
        print("Sector:", sector, '-', item)
        topic_weights = weights_sel[sector][item]
        feature_idx_offset = (30 if item == 'item7' else 0) + (60 if sector == 'all' else 0)
        zipped_weights = [(topic, feature_importances[topic[0] + feature_idx_offset]) for topic in topic_weights]
            # list(zip(topic_weights,
            #                       feature_importances[feature_idx_offset:feature_idx_offset+30]))
        # max_weight, max_weight_id = 0, 0
        # for idx, topic_weight in enumerate(topic_weights):
        #     if topic_weight[1] > max_weight:
        #         max_weight_id, max_weight = topic_weight
        # sorted_topic_weights = sorted(topic_weights, key=lambda x: x[1], reverse=True)
        # print("idx offset for", sector, item, (30 if item == 'item7' else 0) + (60 if sector == 'all' else 0))
        # print(len(list(zipped_weights)))
        for topic_zip in sorted(zipped_weights, key=lambda x: x[1], reverse=True):
            topic, feature_importance = topic_zip
            if feature_importance == 0:
                continue
            topic_terms = models[sector][item].show_topic(topic[0])
            print(round(feature_importance,2),'-',
                  round(topic[1], 2), ":",
                  [topic[0] for topic in topic_terms])
        print("\n-------\n")
print(correctly_classified_idxs)
# %%


