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


def sentence_lda(input_file, output_file, start, end, ws, is_pickled):
    print("Reading {}".format(input_file))
    if is_pickled:
        data = pd.read_csv(input_file)
    else:
        data = pd.read_pickle(input_file)
    data_slice =  data[(data.year_x >= start) & (data.year_x <= end)]
    params = {
        'num_topics': 30,
        'chunksize': 2000,
        'passes': 20,
        'iterations': 400,
        'eval_every': None,
        'alpha': 'asymmetric',
        'eta': 'auto'
    }

    for label in ["item1a_risk", "item7_mda"]:
        print(label)
        print("Collapsing documents")
        documents = [sentence_grp for doc in data_slice[label].to_list() for sentence_grp in doc]

        dictionary = Dictionary(documents)
        corpus = [dictionary.doc2bow(doc) for doc in documents]
        _temp = dictionary[0] # Initialize id2token mappings
        id2word = dictionary.id2token

        print("Running LDA")
        lda = LdaMulticore(corpus=corpus, id2word=id2word, workers=32,**params)

        print("Saving LDA")
        lda.save(output_file + "sen_lda_" + label + "_" + str(ws) + ".model")
        del documents
        del dictionary
        del corpus
        del _temp
        del id2word
        del lda

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

    # Useful for preprocessing
    nltk.download("stopwords")
    nltk.download("wordnet")
    stopwords_total = set(list(STOPWORDS) + stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokenizer = RegexpTokenizer(r'\w+')

    def is_valid_word(word):
        return word not in stopwords_total and len(word) > 2 and not word.isnumeric()

    if process_type == "vanilla_lda":
        """ remove punc, tokenize on spaces, remove stopwords and short words, lemmatizer """
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
    data.to_csv(output_file)
    print("Write finished")
        


    
import argparse
JOB_TYPES = ["preprocess_data", "join_filings_metrics", "sentence_lda"]
PREPROCESS_TYPES = ["vanilla_lda", "sentence_lda"]

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
parser.add_argument("-ey", "--end_year", type = int, required=False, help="Inclusive end range")
parser.add_argument('-p', "--pickled", action='store_true')
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
    sentence_lda(args.input, args.output_file, args.start_year, args.end_year, args.window_size, args.pickled)
else:
    print("JobType Error")
