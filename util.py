import pandas as pd
import pickle

# Loads data
# returns: Tuple(all_data, X, y)
def load_data(path='../Files/', file='processed_data.csv'):
    proc_df = pd.read_csv(path+file, index_col=None)
    X = proc_df[['cik', 'ticker_x', 'filing_date', 'item1a_risk', 'item7_mda', 'year_x', 'filing_year_x']]
    y = proc_df[['perm_id','ticker_y','year_y','company_name','is_dividend_payer','dps_change','is_dps_cut','z_environmental','d_environmental','sector','filing_year_y']]
    
    return proc_df, X, y


def unpickle(file, path='../Files/gensim/'):
    with open(path+file, 'rb') as file:
        obj = pickle.load(file)
    return obj


def load_gensim_data(year, path='../Files/gensim'):
    if type(year) in [tuple, list]:
        year_str = str(year[0]) + '-' + str(year[1])
    else:
        year_str = str(year)
    
    corpus = {
        '1a': unpickle(f'{year_str}_item1a_corpus.pkl', path),
        '7': unpickle(f'{year_str}_item7_corpus.pkl', path)
    }

    id2word = {
        '1a': unpickle(f'{year_str}_item1a_id2word.pkl', path),
        '7': unpickle(f'{year_str}_item7_id2word.pkl', path)
    }
    
    return corpus, id2word