import pandas as pd
path_base = "/mnt/nfs/scratch1/hshukla/sentence_results/"
model_base = "sen_lda_{}_{}.model"
data_base = "df_sen_{}_{}.pkl"
data_base_alt = "df_sen_{}_{}_tmp.pkl"

pd.read_pickle(path_base + data_base.format(1,1)).to_pickle(path_base + data_base_alt.format(1,1), protocol=0)
pd.read_pickle(path_base + data_base.format(5,2)).to_pickle(path_base + data_base_alt.format(5,2), protocol=0)
pd.read_pickle(path_base + data_base.format(7,3)).to_pickle(path_base + data_base_alt.format(7,3), protocol=0)