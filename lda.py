data = pd.read("data.csv")
start_year = 2012
end_year = 2015
predict_year = end_year + 1
train_range = list(range(start_year,end_year+1))

data_train = data[data.year_x is in RANGE]
data_test = data[data.year_x == predict_year]

lda_risk = Lda(data_train["item1a_risk"])
lda_mda = Lda(data_train["item7_mda"])

# For each row in data_slice, returns a length 60 float vector
def get_weights(data_slice):
	return [lda_risk.get_document_topics(row) + lda_mda.get_document_topics(row) for row in data_slice]


rf = RandomForest()
for year in RANGE:
	this_year = data_train[data.year_x == year]
	weights = get_weights(this_year)
	rf.train(x=weights, y=this_year["DPS cut"])



results = rf.predict(get_weights(data_test["DPS cut"]))
print(accuracy(results, data_test["DPS cut"]))