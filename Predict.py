#!/usr/bin/env python
# coding: utf-8
import pandas as pd
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import heapq
from matplotlib import pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
import joblib
import argparse


# Parsing
parser = argparse.ArgumentParser()
JOB_TYPES = ["vanilla_lda", "sentence_lda"]

# Required or binary
parser.add_argument("--job_type", required=True, choices=JOB_TYPES, help="Type of prediction job")
parser.add_argument("--lda_risk", required=True, help="Path to LDA model for item1a_risk")
parser.add_argument("--lda_mda",  required=True, help="Path to LDA model for item7_mda")
parser.add_argument("--input_file",  required=True, help=".csv or .pkl where the data is stored for all years")
parser.add_argument("--output_folder",  required=True, help="Folder to output model, results, and predictions")
parser.add_argument("-sy", "--start_year", type = int, required=True, help="Inclusive start range")
parser.add_argument("-ey", "--end_year", type = int, required=True, help="Inclusive end range")
parser.add_argument("-py", "--predict_year", type = int, required=True, help="Year to predict (usually end_year + 1)")

# Optional
parser.add_argument("-ws", "--window_size", type = int, required=False, help="For ppt sentence_lda, number of sentences in a document")

# Binary flags
parser.add_argument("--pickled", action='store_true', help="Use if input_file is .pkl instead of .csv")
parser.add_argument("--corpus_filter", action='store_true', help="Use if the input corpus needs to be filtered")
args = parser.parse_args()
print(args)

# Fetch settings
job_type = args.job_type
lda_risk_path = args.lda_risk
lda_mda_path = args.lda_mda
data_path = args.input_file
output_folder = args.output_folder
is_pkl = args.pickled
start_year = args.start_year
end_year = args.end_year
predict_year = args.predict_year
train_range = list(range(start_year,end_year+1))
if args.job_type == "sentence_lda":
    window_size = args.window_size
else:
    window_size = None
is_vanilla = job_type == "vanilla_lda"
is_pkl = args.pickled
is_corp_filter = args.corpus_filter



# Load data
print("Loading input data")
if is_pkl:
    data = pd.read_pickle(data_path)
else:
    data = pd.read_csv(data_path)
data = data.sort_values(by=['year_x'])
lda_risk = LdaModel.load(lda_risk_path)
lda_mda = LdaModel.load(lda_mda_path)



# Find subset of valid data
print("Using {} model for [{},{}] inclusive predicting for {}".format(job_type, start_year, end_year, predict_year))
data["is_dividend_payer"] = data["is_dividend_payer"].astype(bool)
data_valid = data[data["is_dividend_payer"] & data["is_dps_cut"].notnull()]
data_valid["is_dps_cut"] = data_valid["is_dps_cut"].astype(int)

# train/test
data_train = data_valid[(data_valid.year_x >= start_year) & (data_valid.year_x <= end_year)]
data_test = data_valid[data_valid.year_x == predict_year]
print("# train rows: {}".format(len(data_train)))
print("# test rows: {}".format(len(data_test)))


# The dictionary is defined over the entire training dataset
if is_vanilla:
    risk_docs = []
    mda_docs = []
else:
    risk_docs = [sentence_grp for doc in data_train["item1a_risk"].to_list() for sentence_grp in doc]
    mda_docs = [sentence_grp for doc in data_train["item7_mda"].to_list() for sentence_grp in doc]

risk_dict = Dictionary(risk_docs)
mda_dict = Dictionary(risk_docs)
if is_corp_filter: # Used filtering in sent-lda to speed things up
    risk_dict.filter_extremes(no_below=10)
    mda_dict.filter_extremes(no_below=10)
del risk_docs
del mda_docs



train_values = data_train["is_dps_cut"].value_counts() / sum(data_train["is_dps_cut"].value_counts())
test_values = data_test["is_dps_cut"].value_counts() / sum(data_test["is_dps_cut"].value_counts())
class_weight = {0: 1.0 / train_values[0], 1: 1.0 / train_values[1]}
print("Train class membership = (0,{:.3f}) and (1,{:.3f})".format(train_values[0], train_values[1]))
print("Test class membership = (0,{:.3f}) and (1,{:.3f})".format(test_values[0], test_values[1]))
print("Using class weights:")
print(class_weight)


rf = RandomForestClassifier(random_state=5, warm_start=False, n_jobs=-1, verbose=1, class_weight=class_weight)



"""
window = 1: Only one topics
window = 5: Two topics
window = 7: Three topics
"""
if window_size == 1:
    #Yes, it is slower to use heapify for a single max element - easier implementation tho
    num_topics = 1
elif window_size == 5:
    num_topics = 2
elif window_size == 7:
    num_topics = 3
else:
    print("ERROR")
print("Num topics per sentence: {}".format(num_topics))


def parse_weights(weights_arr, num_topics):
    result = np.zeros((30,1))
    if isinstance(weights_arr[0], list): # we have more than 1 set of weights
        for sentence_grp in weights_arr:
            top_topics = heapq.nlargest(num_topics, sentence_grp, key=lambda x: x[1])
            for (idx_topic, weight) in top_topics:
                result[idx_topic] += weight
    else:
        """
        Just a single set of weights -> Use it! Edge case for very short docs.
        If we were to only use top x and normalize,
        then it would seem like these documents strongly related to a topic
        -> This isn't actually hit ever I dont think
        """ 
        print("Single")
        result = np.array([topic_weight[1] for topic_weight in weights_arr], dtype=np.float64)[:,None] # grab only the weight
    return result / np.linalg.norm(result, ord=1) # Normalize before returning


# In[11]:


print("For training we have {} documents".format(len(data_train)))

risk_docs = [sentence_grp for doc in data_train["item1a_risk"].to_list() for sentence_grp in doc]
risk_corpus = [risk_dict.doc2bow(doc) for doc in risk_docs]
del risk_docs

mda_docs = [sentence_grp for doc in data_train["item7_mda"].to_list() for sentence_grp in doc]
mda_corpus = [mda_dict.doc2bow(doc) for doc in mda_docs]
del mda_docs


documents_weights = np.zeros((len(data_train), 60))
idx_risk = 0
idx_mda = 0
for idx_slice in range(len(data_train)):
    row = data_train.iloc[idx_slice]
    n_risk = len(row["item1a_risk"])
    n_mda = len(row["item7_mda"])

    row_risk_results = [item for item in lda_risk[risk_corpus[idx_risk:idx_risk+n_risk]]]
    row_mda_results = [item for item in lda_mda[mda_corpus[idx_mda:idx_mda+n_mda]]]

    weights_risk = parse_weights(row_risk_results, num_topics)
    weights_mda = parse_weights(row_mda_results, num_topics)
    weights = np.concatenate((weights_risk, weights_mda), axis=0)

    documents_weights[idx_slice,:] = weights.squeeze()

    idx_risk += n_risk
    idx_mda += n_mda


# In[ ]:


rf.fit(X=documents_weights,y=data_train["is_dps_cut"].to_list())


# In[12]:


print("For testing we have {} documents".format(len(data_test)))
test_risk_docs = [sentence_grp for doc in data_test["item1a_risk"].to_list() for sentence_grp in doc]
test_risk_corpus = [risk_dict.doc2bow(doc) for doc in test_risk_docs]

test_mda_docs = [sentence_grp for doc in data_test["item7_mda"].to_list() for sentence_grp in doc]
test_mda_corpus = [mda_dict.doc2bow(doc) for doc in test_mda_docs]

del test_risk_docs
del test_mda_docs


# In[13]:


# Find testing features
test_documents_weights = np.zeros((len(data_test), 60))
idx_risk = 0
idx_mda = 0
for idx_slice in range(len(data_test)):
    row = data_test.iloc[idx_slice]
    n_risk = len(row["item1a_risk"])
    n_mda = len(row["item7_mda"])
    
    row_risk_results = [item for item in lda_risk[test_risk_corpus[idx_risk:idx_risk+n_risk]]]
    row_mda_results = [item for item in lda_mda[test_mda_corpus[idx_mda:idx_mda+n_mda]]]
    
    weights_risk = parse_weights(row_risk_results, num_topics)
    weights_mda = parse_weights(row_mda_results, num_topics)
    weights = np.concatenate((weights_risk, weights_mda), axis=0)
    
    test_documents_weights[idx_slice,:] = weights.squeeze()
    
    idx_risk += n_risk
    idx_mda += n_mda

y_pred = rf.predict(test_documents_weights)
y_actual = data_test["is_dps_cut"].to_list()


accuracy = accuracy_score(y_actual, y_pred)
precision = precision_score(y_actual, y_pred)
recall = recall_score(y_actual, y_pred)
f1 = f1_score(y_actual, y_pred)
print("Accuracy: {:.4f}".format(accuracy))
print("Precision: {:.4f}".format(precision))
print("Recall: {:.4f}".format(recall))
print("F1-score: {:.4f}".format(f1))


feature_columns = ["risk_topic_" + str(i) for i in range(30)] + ["mda_topic_" + str(i) for i in range(30)]

# Copy and add training features
training_data = data_train.copy().reset_index()
training_features = pd.DataFrame(data=documents_weights, columns=feature_columns)
training_data_output = pd.concat([training_data, training_features], axis=1)

# Copy and add predictions + training features
testing_data = data_test.copy().reset_index()
testing_data["dps_cut_prediction"] = y_pred
testing_features =  pd.DataFrame(data=test_documents_weights, columns=feature_columns)
testing_data_output = pd.concat([testing_data, testing_features], axis=1)

# Write to disk
print("Writing to disk")
print("Writing training")
training_data_output.to_csv(output_folder + "training_{}_{}_{}_{}.csv".format(start_year, end_year, predict_year, window_size))
print("Writing testing")
testing_data_output.to_csv(output_folder + "testing_{}_{}_{}_{}.csv".format(start_year, end_year, predict_year, window_size))
print("Writing forest")
joblib.dump(rf, output_folder + 'rf_sentencelda_{}_{}_{}_{}.pkl'.format(start_year, end_year, predict_year, window_size))

