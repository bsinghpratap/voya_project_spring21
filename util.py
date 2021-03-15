import pandas as pd

# Loads data
# returns: Tuple(all_data, X, y)
def load_data(path='../Files/', file='processed_data.csv'):
    proc_df = pd.read_csv(path+file, index_col=None)
    X = proc_df[['cik', 'ticker_x', 'filing_date', 'item1a_risk', 'item7_mda', 'year_x', 'filing_year_x']]
    y = proc_df[['perm_id','ticker_y','year_y','company_name','is_dividend_payer','dps_change','is_dps_cut','z_environmental','d_environmental','sector','filing_year_y']]
    
    return proc_df, X, y
