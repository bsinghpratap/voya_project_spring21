import pandas as pd
import numpy as np
from gensim.parsing.preprocessing import preprocess_string, remove_stopwords, strip_numeric, strip_punctuation, strip_short, stem_text

data_by_year = {}
# Filing_Date is included because its (allegedly) useful for matching with the predictions. Also useful for 
# deciding what 1a/7 text is correct (by newness)
relevant_cols = ["cik", "ticker", "filing_date", "item1a_risk", "item7_mda"]

# Only load ^ columns. Drop if N/A in cik, item1a, or item7 or if duplicate. Append year
for year in range(2009,2021):
    data_by_year[year] = pd.read_csv(str(year) + ".csv", usecols=relevant_cols)
    data_by_year[year] = data_by_year[year].dropna(subset=['cik', 'item1a_risk', 'item7_mda']).drop_duplicates()
    data_by_year[year]["year"] = year
    data_by_year[year]["filing_date"] = pd.to_datetime(data_by_year[year]["filing_date"])


# Dirty logic for collapsing groups. Reformat as needed - currently pretty dumb
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



# Text cleaning

# strip_short/strip_numeric are questionable because there are lots of relevant financial terms that are short
# Consider removing words that are rare across documents (appear < 4 times)
# A single year takes about 7 minutes to process. In total = mins
PREPROCESS_TEXT_FILTERS = [remove_stopwords, stem_text, lambda x: strip_short(x, minsize=3), strip_numeric, strip_punctuation]
text_dfs["item1a_risk"] = text_dfs["item1a_risk"].apply(lambda txt: preprocess_string(txt, PREPROCESS_TEXT_FILTERS))
text_dfs["item7_mda"] = text_dfs["item7_mda"].apply(lambda txt: preprocess_string(txt, PREPROCESS_TEXT_FILTERS))


# Load predictions
relevant_cols = ["PERMID", "CIK", "Ticker", "year", "FilingDate", "company_name", "Dividend Payer", "DPS growth", "DPS cut", "zEnvironmental", "dEnvironmental", "sector"]
predictions = pd.read_excel("predictions.xlsx", sheet_name="data", skiprows=32, usecols=relevant_cols)
predictions.columns = ["perm_id", "cik", "ticker", "year", "filing_date", "company_name", "is_dividend_payer", "dps_change", "is_dps_cut", "z_environmental", "d_environmental", "sector"]
predictions['perm_id'] = predictions['perm_id'].str.replace(r"[^0-9]",'')
predictions["filing_date"] = pd.to_datetime(predictions["filing_date"])
""" Difference in filing_date and the year (ticker AA  has 2016 w/ 2017 filing)"""

result = pd.merge(text_dfs, predictions, on=["cik", "year"])

# Just taking some metrics before freeing up data
num_pred = float(len(predictions))
num_text = float(len(text_dfs))

del predictions
del text_dfs

""" Relevant statistics post merge """
num_result = float(len(result))
num_text_lost = num_result - num_text
num_pred_lost = num_result - num_pred

ticker_mismatch = result["ticker_x"] != result["ticker_y"]
filing_date_mismatch = result["filing_date_x"] != result["filing_date_y"]
ticker_and_filing_mismatch = ticker_mismatch & filing_date_mismatch
ticker_or_filing_mismatch = ticker_mismatch | filing_date_mismatch

print("# and % of 10-K filings lost: ({:n},{:.0%})".format(num_text_lost, num_text_lost/ num_text))
print("# and % of stock events lost: ({:n},{:.0%})".format(num_pred_lost, num_pred_lost / num_pred))
print("# and % of ticker mismatches: ({:n},{:.0%})".format(ticker_mismatch.sum(), float(ticker_mismatch.sum()) / num_result))
print("# and % of filing date mismatches: ({:n},{:.0%})".format(filing_date_mismatch.sum(), float(filing_date_mismatch.sum()) / num_result))
print("# and % of ticker and filing date mismatches: ({:n},{:.0%})".format(ticker_and_filing_mismatch.sum(), float(ticker_and_filing_mismatch.sum()) / num_result))
print("# and % of ticker or filing date mismatches: ({:n},{:.0%})".format(ticker_or_filing_mismatch.sum(), float(ticker_or_filing_mismatch.sum()) / num_result))


result.drop(columns=["ticker_x", "filing_date_x", "ticker_y", "filing_date_y", "cik"], inplace=True)
result.write_csv("processed_data.csv")