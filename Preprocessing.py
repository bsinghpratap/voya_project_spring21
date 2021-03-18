import pandas as pd
import numpy as np
from gensim.parsing.preprocessing import preprocess_string, strip_numeric, strip_punctuation, strip_short, stem_text
from gensim.parsing.preprocessing import STOPWORDS
from nltk.corpus import stopwords
import nltk
nltk.download("stopwords")


import sys, getopt


def join_10ks(inputFolder, outputFolder):
    data_by_year = {}
    relevant_cols = ["cik", "ticker", "filing_date", "item1a_risk", "item7_mda"]
    for year in range(2009,2021):
        data_by_year[year] = pd.read_csv(inputFolder + str(year) + ".csv", usecols=relevant_cols)
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
    text_dfs.to_csv(outputFolder + "10k.csv")
    

def preprocess_10k(inputFile, outputFolder, process_type):
    #Stopword Removal
    cachedWords = stopwords.words('english')
    stopwords_total = set(list(STOPWORDS) + cachedWords)
    def remove_stopwords(txt):
        return " ".join([word for word in txt.split() if (word not in stopwords_total)])
    PREPROCESS_TEXT_FILTERS = [strip_punctuation, strip_numeric, remove_stopwords, stem_text, lambda x: strip_short(x, minsize=3)]
    print("Preprocessing 1a")
    for filt in PREPROCESS_TEXT_FILTERS:
        text_dfs["item1a_risk"] = text_dfs["item1a_risk"].apply(filt)
    print("Preprocessing 7")
    for filt in PREPROCESS_TEXT_FILTERS:
        text_dfs["item7_mda"] = text_dfs["item7_mda"].apply(filt)


def join_10ks_predictions(textInputPath, predictionInputPath, outputFolder):
    # Load text
    text_dfs = pd.read_csv(textInputPath)
    
    # Load predictions
    relevant_cols = ["PERMID", "CIK", "Ticker", "year", "FilingDate", "company_name", "Dividend Payer", "DPS growth", "DPS cut", "zEnvironmental", "dEnvironmental", "sector"]
    predictions = pd.read_excel(PATH + "predictions.xlsx", sheet_name="data", skiprows=32, usecols=relevant_cols)
    predictions.columns = ["perm_id", "cik", "ticker", "year", "filing_date", "company_name", "is_dividend_payer", "dps_change", "is_dps_cut", "z_environmental", "d_environmental", "sector"]
    predictions['perm_id'] = predictions['perm_id'].str.replace(r"[^0-9]",'')
    predictions["filing_date"] = pd.to_datetime(predictions["filing_date"])
    predictions["filing_year"] =  pd.DatetimeIndex(predictions["filing_date"]).year

    result = pd.merge(text_dfs, predictions, on=["cik", "filing_date"])
    result.to_csv(outputFolder + "processed_data.csv")



    
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
    
    
arguments = sys.argv[1:] #Skip filename argument
short_options = "i:o:j:"
long_options = ["input", "output", "jobtype"]

arguments = []
values = []
try:
    arguments, values = getopt.getopt(argument, short_options, long_options)
except getopt.error as err:
    # Output error, and return with an error code
    print (str(err))
    sys.exit(2)