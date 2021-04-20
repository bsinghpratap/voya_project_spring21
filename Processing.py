import pandas as pd
import numpy as np
from gensim.parsing.preprocessing import preprocess_string, strip_numeric, strip_punctuation, strip_short, stem_text
from gensim.parsing.preprocessing import STOPWORDS
from nltk.corpus import stopwords
import nltk
import string
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
import pandas as pd
import ast
from gensim.corpora import Dictionary
from gensim.models.ldamulticore import LdaMulticore
import pickle
from gensim.matutils import corpus2csc, corpus2dense
from gensim.models import TfidfModel


def join_filings_metrics(input_folder, output_file):
    data_by_year = {}
    relevant_cols = ["cik", "ticker", "filing_date", "item1a_risk", "item7_mda"]
    for year in range(2009,2021):
        data_by_year[year] = pd.read_csv(input_folder + str(year) + ".csv", usecols=relevant_cols)
        data_by_year[year] = data_by_year[year].dropna(subset=['cik', 'item1a_risk', 'item7_mda']).drop_duplicates()
        data_by_year[year]["year"] = year
        data_by_year[year]["filing_date"] = pd.to_datetime(data_by_year[year]["filing_date"])
        data_by_year[year]["filing_year"] =  pd.DatetimeIndex(data_by_year[year]["filing_date"]).year


    def collapse_cik_groups(grp):
        if len(grp) > 1:
            """ If the 1a and 7 text is the same, take the most recent (regardless of ticker)"""
            if (grp.iloc[0,3] == grp["item1a_risk"]).all() and (grp.iloc[0,3] == grp["item7_mda"]).all():
                # Seems like its sorted by filing_date originally - just take the last
                return grp.iloc[-1,:]
            else:
                """For now, just return the most recent"""
                return grp.iloc[-1,:]
        else:
            return grp.squeeze()

    for year in data_by_year.keys():
        data_by_year[year] = data_by_year[year].groupby("cik").apply(lambda grp: collapse_cik_groups(grp)).reset_index(drop=True)    


    # Concat all dataframes into a single one
    text_dfs = pd.concat(data_by_year.values(), ignore_index=True)
    del data_by_year

    # Load predictions
    relevant_cols = ["PERMID", "CIK", "Ticker", "year", "FilingDate", "company_name", "Dividend Payer", "DPS growth", "DPS cut", "zEnvironmental", "dEnvironmental", "sector"]
    predictions = pd.read_excel(input_folder + "predictions.xlsx", sheet_name="data", skiprows=32, usecols=relevant_cols)
    predictions.columns = ["perm_id", "cik", "ticker", "year", "filing_date", "company_name", "is_dividend_payer", "dps_change", "is_dps_cut", "z_environmental", "d_environmental", "sector"]
    predictions['perm_id'] = predictions['perm_id'].str.replace(r"[^0-9]",'')
    predictions["filing_date"] = pd.to_datetime(predictions["filing_date"])
    predictions["filing_year"] =  pd.DatetimeIndex(predictions["filing_date"]).year

    result = pd.merge(text_dfs, predictions, on=["cik", "filing_date"])
    result.to_csv(output_file)

    del text_dfs
    del predictions
    del result

def baseline(input_file, output_file, start, end, is_pickled):
    valid_year = end + 1
    test_year =  end + 2

    print("Reading {}".format(input_file))
    if is_pickled:
        data = pd.read_pickle(input_file)
    else:
        data = pd.read_csv(input_file)
    data.sort_values(["year_x", "sector"], axis=0, inplace=True)
    data_train =  data[(data.year_x >= start) & (data.year_x <= end)].copy(deep=True).reset_index(drop=True)
    data_test = data[(data.year_x == valid_year) | (data.year_x == test_year)].copy(deep=True).reset_index(drop=True)
    del data

    print("Num data_train = {}".format(len(data_train)))
    print("Num data_test = {}".format(len(data_test)))

    for label in ["item1a_risk", "item7_mda"]:
        train_docs = data_train[label].to_list()
        test_docs = data_test[label].to_list()

        dictionary = Dictionary(train_docs)
        dictionary.filter_extremes(no_below=10)
        _temp = dictionary[0]
        id2word = dictionary.id2token

        train_corpus = [dictionary.doc2bow(doc) for doc in train_docs]
        test_corpus = [dictionary.doc2bow(doc) for doc in test_docs]

        print("Saving train dictionary")
        with open(output_file + "baseline_dict_train_" + "_".join([label,str(start),str(end)]) + ".pkl", 'wb') as file:
            pickle.dump(dictionary, file)

        print("Saving train corpus")
        with open(output_file + "baseline_corpus_train_" +  "_".join([label,str(start),str(end)]) + ".pkl", 'wb') as file:
            pickle.dump(train_corpus, file)

        print("Saving test corpus")
        with open(output_file + "baseline_corpus_test_" +  "_".join([label,str(start),str(end)]) + ".pkl", 'wb') as file:
            pickle.dump(test_corpus, file)

        dict_terms = len(dictionary.keys())
        train_size = len(train_docs)
        test_size = len(test_docs)

        print("Len dict_terms {}".format(dict_terms))
        print("Len train_size {}".format(train_size))
        print("Len test_size {}".format(test_size))

        """ Frequency based features """
        # Train
        train_freq_features = pd.DataFrame(corpus2dense(train_corpus, num_terms=dict_terms, num_docs=train_size).T).reset_index(drop=True)
        train_freq_features.columns = ["freq_" + label + "_" + str(id2word.get(int(col))) for col in train_freq_features.columns]
        # Test
        test_freq_features = pd.DataFrame(corpus2dense(test_corpus, num_terms=dict_terms, num_docs=test_size).T).reset_index(drop=True)
        test_freq_features.columns = ["freq_" + label + "_" + str(id2word.get(int(col))) for col in test_freq_features.columns]

        print("Shape of train_freq_features: {}".format(str(train_freq_features.shape)))
        print("Shape of test_freq_features: {}".format(str(test_freq_features.shape)))

        """ TFIDF features """
        tfidf = TfidfModel(train_corpus)
        # Train
        train_tfidf_features = [feature for feature in tfidf[train_corpus]]
        train_tfidf_features = pd.DataFrame(corpus2dense(train_tfidf_features, num_terms=dict_terms, num_docs=train_size).T).reset_index(drop=True)
        train_tfidf_features.columns = ["tfidf_" + label + "_" + str(id2word.get(int(col))) for col in train_tfidf_features.columns]
        # Test
        test_tfidf_features = [feature for feature in tfidf[test_corpus]]
        test_tfidf_features = pd.DataFrame(corpus2dense(test_tfidf_features, num_terms=dict_terms, num_docs=test_size).T).reset_index(drop=True)
        test_tfidf_features.columns = ["tfidf_" + label + "_" + str(id2word.get(int(col))) for col in test_tfidf_features.columns]

        print("Shape of train_tfidf_features: {}".format(str(train_tfidf_features.shape)))
        print("Shape of test_tfidf_features: {}".format(str(test_tfidf_features.shape)))
        print("Data_train shape pre-merge: {}".format(data_train.shape))
        print("Data_test shape pre-merge: {}".format(data_test.shape))
        
        data_train = pd.concat([data_train, train_freq_features, train_tfidf_features], axis=1)
        data_test = pd.concat([data_test, test_freq_features, test_tfidf_features], axis=1)       
        print("Data_train shape post-merge: {}".format(data_train.shape))
        print("Data_test shape post-merge: {}".format(data_test.shape))
        print()


    train_postfix = "_".join(["baseline","train", str(start), str(end)]) + ".pkl"
    test_postfix = "_".join(["baseline","test", str(valid_year), str(test_year)]) + ".pkl"
    data_train.to_pickle(output_file + train_postfix, protocol=0)
    data_test.to_pickle(output_file + test_postfix, protocol=0)




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

def create_weights(data, lda_risk, lda_mda, risk_corpus, mda_corpus, ws):
    if ws == 1:
        num_topics = 1
    elif ws == 5:
        num_topics = 2
    elif ws == 7:
        num_topics = 3

    idx_risk = 0
    idx_mda = 0
    document_weights = np.zeros((len(data), 60))
    for idx_slice in range(len(data)):
        row = data.iloc[idx_slice]

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

    return document_weights


def sentence_lda_features(input_file, output_folder, start, end, ws, is_pickled):
    train_range = list(range(start_year,end_year+1))
    valid_year = end + 1
    test_year = end + 2

    print("Reading {}".format(input_file))
    if is_pickled:
        data = pd.read_pickle(input_file)
    else:
        data = pd.read_csv(input_file)
    data.sort_values(["year_x", "sector"], axis=0, inplace=True)
    data_train =  data[(data.year_x >= start) & (data.year_x <= end)].copy(deep=True).reset_index(drop=True)
    data_test = data[(data.year_x == valid_year) | (data.year_x == test_year)].copy(deep=True).reset_index(drop=True)
    del data


    print("Loading dictionary")
    risk_postfix = "_".join(["item1a_risk",str(ws),str(start),str(end)])
    mda_postfix = "_".join(["item7_mda",str(ws),str(start),str(end)])

    # LDAs
    lda_risk = LdaModel.load(output_folder + "sen_lda_" + risk_postfix + ".model")
    lda_mda = LdaModel.load(loutput_folder + "sen_lda_" + mda_postfix + ".model")

    # Dicts
    with open(output_folder + "sen_dict_" + risk_postfix + ".pkl", 'rb') as file:
        train_risk_dict = pickle.load(file)
    with open(output_folder + "sen_dict_" + mda_postfix + ".pkl", 'rb') as file:
        train_mda_dict = pickle.load(file)

    # Train corpus
    with open(output_folder + "sen_corpus_" + risk_postfix + ".pkl", 'rb') as file:
        train_risk_corpus = pickle.load(file)
    with open(output_folder + "sen_corpus_" + mda_postfix + ".pkl", 'rb') as file:
        train_mda_corpus = pickle.load(file)

    # Create valid/test corpus
    test_valid_risk_corpus = [train_risk_dict.doc2bow(doc) for doc in data_test["item1a_risk"].to_list()]
    test_valid_mda_corpus = [train_mda_dict.doc2bow(doc) for doc in data_test["item7_mda"].to_list()]

    print("Saving Test/Validcorpus")
    with open(output_file + "sen_corpus_test_valid_" + risk_postfix + ".pkl", 'wb') as file:
        pickle.dump(test_valid_risk_corpus, file)
    with open(output_file + "sen_corpus_test_valid_" + mda_postfix + ".pkl", 'wb') as file:
        pickle.dump(test_valid_mda_corpus, file)


    # Find train weights
    train_documents_weights = create_weights(data_train, lda_risk, lda_mda, train_risk_corpus, train_mda_corpus, ws)
    test_valid_documents_weights = create_weights(data_test, lda_risk, lda_mda, test_valid_risk_corpus, test_valid_mda_corpus, ws)

    # Append weights to train or valid/test
    feature_columns = ["risk_topic_" + str(i) for i in range(30)] + ["mda_topic_" + str(i) for i in range(30)]
    train_documents_weights = pd.DataFrame(data=train_documents_weights, columns=feature_columns)
    test_valid_documents_weights = pd.DataFrame(data=test_valid_documents_weights, columns=feature_columns)

    data_train = pd.concat([data_train, train_documents_weights], axis=1)
    data_test = pd.concat([data_test, test_valid_documents_weights], axis=1)

    # Write to disk
    run_specific_postfix = "_".join([str(ws),str(start),str(end)])
    data_train.to_pickle("data_train_" + run_specific_postfix + ".pkl", protocol=0)
    data_test.to_pickle("data_test_valid_" + run_specific_postfix + ".pkl", protocol=0)



def sentence_lda(input_file, output_file, start, end, ws, is_pickled, label):
    print("Reading {}".format(input_file))
    if is_pickled:
        data = pd.read_pickle(input_file)
    else:
        data = pd.read_csv(input_file)
    data.sort_values(["year_x", "sector"], axis=0, inplace=True)
    data_slice =  data[(data.year_x >= start) & (data.year_x <= end)]
    params = {
        "random_state":10,
        'num_topics': 30,
        'chunksize': 50,
        'passes': 20,
        'iterations': 400,
        'eval_every': None,
        'alpha': 'asymmetric',
        'eta': 'auto'
    }
    print(params["chunksize"])

    
    print(label)
    print("Collapsing documents")
    documents = [sentence_grp for doc in data_slice[label].to_list() for sentence_grp in doc]

    dictionary = Dictionary(documents)
    print(len(dictionary))
    dictionary.filter_extremes(no_below=10)
    print(len(dictionary))
    corpus = [dictionary.doc2bow(doc) for doc in documents]
    _temp = dictionary[0] # Initialize id2token mappings
    id2word = dictionary.id2token

    print("Saving dictionary")
    run_specific_postfix = "_".join([label,str(ws),str(start),str(end)])
    with open(output_file + "sen_dict_" + run_specific_postfix + ".pkl", 'wb') as file:
        pickle.dump(dictionary, file)

    print("Saving documents")
    with open(output_file + "sen_docs_" + run_specific_postfix + ".pkl", 'wb') as file:
        pickle.dump(documents, file)

    print("Saving corpus")
    with open(output_file + "sen_corpus_" + run_specific_postfix + ".pkl", 'wb') as file:
        pickle.dump(corpus, file)


    print("Running LDA")
    lda = LdaMulticore(corpus=corpus, id2word=id2word, workers=4,**params)

    print("Saving LDA")
    lda.save(output_file + "sen_lda_" + run_specific_postfix + ".model")


def split_period(txt, window_size, window_overlap):
    sentences = txt.split(".")
    if sentences[-1] == "": # Period at end => empty string as last ele
        sentences = sentences[:-1]

    # Make a windowed version of our sentences
    if window_size == 1: # Regular sentences
        return sentences
    else:
        step_size = window_size - window_overlap
        if len(sentences) < window_size:  # Not enough sentences for this window size -> single window
            return [" ".join(sentences)]
        else:
            startind_incl = 0
            docs = []
            while startind_incl < len(sentences):
                endind_excl = startind_incl + window_size
                if(endind_excl > len(sentences)): # Not enough space for another window
                    docs.append(" ".join(sentences[startind_incl:]))
                    break
                else:
                    docs.append(" ".join(sentences[startind_incl:endind_excl]))
                startind_incl += step_size
            return docs

trans = str.maketrans(dict.fromkeys(string.punctuation))

#Removes punctuation from a string
def remove_punc(txt):
    return txt.translate(trans)

"""
Preprocesses merged 10-k and metrics
-> output row of vanilla lda = ["word1", "word2"...]
-> output row of sentence lda = [[word1,word2,word3...], [word3,word4,word5,...], ...]
"""
def preprocess_data(input_file, output_file, process_type, window_size=1, window_overlap=1):
    data = pd.read_csv(input_file)
    data.sort_values(["year_x", "sector"], axis=0, inplace=True)

    # Useful for preprocessing
    EXTRA_STOPWORDS = ["the","thing","there","our","any","while","10k","this","day","for","date","item","use"]
    nltk.download("stopwords")
    nltk.download("wordnet")
    stopwords_total = set(list(STOPWORDS) + stopwords.words('english') + EXTRA_STOPWORDS)
    lemmatizer = WordNetLemmatizer()
    tokenizer = RegexpTokenizer(r'\w+')

    def is_valid_word(word):
        return word not in stopwords_total and len(word) > 2 and not word.isnumeric()

    if process_type == "vanilla_lda":
        """ remove punc, tokenize on spaces, remove stopwords and short words, lemmatizer """
        for label in ["item1a_risk", "item7_mda"]:
            print("Preprocessing " + label)
            # Lowercasen, remove punctuation, tokenize on spaces
            data[label] = data[label].apply(lambda txt: tokenizer.tokenize(remove_punc(txt.lower())))

            # Lemmatize and remove stopwords
            data[label] = data[label].apply(lambda doc: [lemmatizer.lemmatize(word) for word in doc if (is_valid_word(lemmatizer.lemmatize(word)) and is_valid_word(word))])

    elif process_type == "sentence_lda":
        for label in ["item1a_risk", "item7_mda"]:
            print("Preprocessing " + label)
            # Split on periods, create documents based on window properties
            data[label] = data[label].apply(lambda txt: split_period(txt.lower(), window_size, window_overlap))

            # Remove puntuation, tokenize on spaces
            data[label] = data[label].apply(lambda lst: [tokenizer.tokenize(remove_punc(txt)) for txt in lst])

            # Remove stopwords + short words + numeric-only words, lemmatizer leftovers
            # data[label][0] = List[List[String]]
            data[label] = data[label].apply(lambda documents: [[lemmatizer.lemmatize(word) for word in doc if is_valid_word(word)] for doc in documents])
    else:
        print("ERROR! invalid process_type")
    print("Writing to csv")
    data.to_pickle(output_file, protocol=0)
    print("Write finished")
        


    
import argparse
JOB_TYPES = ["preprocess_data", "join_filings_metrics", "sentence_lda", "baseline"]
PREPROCESS_TYPES = ["vanilla_lda", "sentence_lda"]
VALID_LABELS = ["item1a_risk", "item7_mda"]

parser = argparse.ArgumentParser()
# Required
parser.add_argument("-j", "--job_type", required=True, choices=JOB_TYPES, help="Type of job")
parser.add_argument("-i", "--input", required=True, help="Input file or folder")
parser.add_argument("-o", "--output_file", required=True, help="Output file")

# Optional
parser.add_argument("-ppt", "--preprocess_type", required=False, choices=PREPROCESS_TYPES, help="What type of preprocessing")
parser.add_argument("-ws", "--window_size", type = int, required=False, help="For ppt sentence_lda, number of sentences in a document")
parser.add_argument("-wo", "--window_overlap", type = int, required=False, help="For ppt sentence_lda, number of overlapping sentences b/t subsequent documents")
parser.add_argument("-sy", "--start_year", type = int, required=False, help="Inclusive start range")
parser.add_argument("-ey", "--end_year", type = int, required=False, help="Inclusive end range")
parser.add_argument('-p', "--pickled", action='store_true')
parser.add_argument("-l", "--label", required=False, choices=VALID_LABELS, help="For lda, text column label to use")
args = parser.parse_args()
print(args)   
    
if args.job_type == "join_filings_metrics":
    join_filings_metrics(args.input, args.output_file)
elif args.job_type == "preprocess_data":
    if args.preprocess_type == None:
        print("No preprocessType set")
        quit()
    if args.preprocess_type == "sentence_lda" and (args.window_size == None or args.window_overlap == None):
        print("Window Size and Window Overlap not chosen for sentence lda")
        quit()
    preprocess_data(args.input, args.output_file, args.preprocess_type, args.window_size, args.window_overlap)
elif args.job_type == "sentence_lda":
    if args.start_year == None or args.end_year == None:
        print("start/end not correctly set")
        quit()
    if args.window_size == None:
        print("No window size declared")
        quit()
    if args.label == None:
        print("No label chosen")
        quit()
    sentence_lda(args.input, args.output_file, args.start_year, args.end_year, args.window_size, args.pickled, args.label)
elif args.job_type == "baseline":
    if args.start_year == None or args.end_year == None:
        print("start/end not correctly set")
        quit()
    baseline(args.input, args.output_file, args.start_year, args.end_year, args.pickled)
elif args.job_type == "sentence_lda_features":
    print("lda features")
else:
    print("JobType Error")

