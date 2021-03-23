
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel
from sklearn.ensemble import RandomForestClassifier

#%% Parameters
lda_risk_path = ""
lda_mda_path = ""
input_path = ""
output_path = ""
is_pkl = True
is_vanilla = False
corp_filter = True

sentence_windows = [1,5,7]
start_year = 2012
end_year = 2015
predict_year = end_year + 1
train_range = list(range(start_year,end_year+1))

#%% Load data
if is_pkl:
	data = pd.read_pickle(input_path)
else:
	data = pd.read_csv(input_path)
lda_risk = LdaModel.load(lda_risk_path)
lda_mda = LdaModel.load(lda_mda_path)

#%% Find subset of valid data
print("LDA on [{},{}] predicting for {}".format(start_year, end_year, predict_year))
data["is_dividend_payer"] = data["is_dividend_payer"].astype(bool)
data_valid = data[data["is_dividend_payer"] & data["is_dps_cut"].notnull()]
data_valid["is_dps_cut"] = data_valid["is_dps_cut"].astype(int)

#%% train/test
data_train = data_valid[(data_valid.year_x >= start_year) & (data_valid.year_x <= end_year)]
data_test = data_valid[data_valid.year_x == predict_year]
print("# train rows: {}".format(len(data_train)))
print("# test rows: {}".format(len(data_test)))

# Re-create our risk dictionary using ENTIRE data for that range
# Could be different for vanilla-lda
if is_vanilla:
	risk_docs = []
	mda_docs = []
else:
	risk_docs = [sentence_grp for doc in data_train["item1a_risk"].to_list() for sentence_grp in doc]
	mda_docs = [sentence_grp for doc in data_train["item7_mda"].to_list() for sentence_grp in doc]

risk_dict = Dictionary(risk_docs)
mda_dict = Dictionary(risk_docs)
if corp_filter: # Used filtering in sent-lda to speed things up
	risk_dict.filter_extremes(no_below=10)
	mda_dict.filter_extremes(no_below=10)
del risk_docs
del mda_docs

risk = data_train[data_train.year_x == 2012]["item1a_risk"]
these_documents = [sentence_grp for doc in risk.to_list() for sentence_grp in doc]
this_corpus = [dictionary.doc2bow(doc) for doc in these_documents]


# Get weights for risk and mda
def get_weights(year, window_size=None, is_train=True):
	if is_train:
		data_slice = data_valid[data_valid.year_x == year]
	else:
		data_slice = data_test #Only has that year's data anyways
	print("For year {} we have {} documents".format(year, len(data_slice)))

	# Flatten the documents, convert to ID tokens
	risk_docs = [sentence_grp for doc in data_slice["item1a_risk"].to_list() for sentence_grp in doc]
	risk_corpus = [risk_dict.doc2bow(doc) for doc in risk_docs]
	mda_docs = [sentence_grp for doc in data_slice["item7_mda"].to_list() for sentence_grp in doc]
	mda_corpus = [mda_dict.doc2bow(doc) for doc in risk_docs]
	del risk_docs
	del mda_docs

	risk_results = lda_risk[risk_corpus]
	mda_results = lda_mda[mda_corpus]

	risk_idx = 0
	mda_idx = 0
	weights = []
	for row in data_slice: #Look at size of the row,
		num_risk = len(row["item1a_risk"])
		num_mda = len(row["item7_mda"])
		risk_weights = [risk_idx:risk_idx+num_risk]


	risk_res = lda_risk[]


# Train the model
rf = RandomForest() # Parameters to try?
for year in train_range:
	weights = get_weights(year)
	rf.train(x=weights, y=this_year["DPS cut"])

# Output test accuracy


# Save the rf model as well as results