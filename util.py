import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

SECTORS = ['Industrials', 'Tech', 'Commodities', 'Consumer', 'Health Care', 'Real Estate', 'Utilities']
ITEMS = ['item1a', 'item7']


def load_data(path=os.getenv('VOYA_PATH_DATA'), file='processed_data.csv'):
    """Loads data
    returns: Tuple(all_data, X, y)
    """
    proc_df = pd.read_csv(path+file, index_col=None)
    sort_data_inplace(proc_df)
    X = proc_df[['cik', 'ticker_x', 'filing_date', 'item1a_risk', 'item7_mda', 'year_x', 'filing_year_x']]
    y = proc_df[['perm_id','ticker_y','year_y','company_name','is_dividend_payer','dps_change','is_dps_cut','z_environmental','d_environmental','sector','filing_year_y']]
    
    return proc_df, X, y


def sort_data_inplace(data):
    """Sort data frame by columns
    (cik, ticker) inplace
    """
    data.sort_values(by=['year_x', 'cik', 'ticker_x'], inplace=True)


def unpickle(file, path='../Files/gensim/'):
    with open(path+file, 'rb') as file:
        obj = pickle.load(file)
    return obj


def load_gensim_data(year, path=os.getenv('VOYA_PATH_DATA_GENSIM')):
    if type(year) in [tuple, list]:
        year_str = str(year[0]) + '-' + str(year[1])
    else:
        year_str = str(year) + '-' + str(year)
        
    objs = dict()
    
    for filename in os.listdir(path+year_str+'/'):
        if filename.endswith('.pkl'):   
            _, sector, item, gtype = filename.split('_')
            gtype = gtype.split('.')[0]
            
            if sector not in objs:
                objs[sector] = {
                    'item1a': dict(),
                    'item7': dict()
                }
            
            objs[sector][item][gtype] = unpickle(f'{year_str}_{sector}_{item}_{gtype}.pkl', path=path+year_str+'/')
    
    return objs


def save_models(models, years, path=os.getenv('VOYA_PATH_MODELS')):
    """Save models to disk
    models object should be dictionary formatted as follows
        models[sector][item] = model
    """
    path_full = f'{path}{str(years[0])}-{str(years[-1])}/'
    file_name = f'{str(years[0])}-{str(years[-1])}' + '_{}_{}.gnsm'

    if not os.path.exists(path_full):
        os.makedirs(path_full)

    for sector in models:
        for item in models[sector]:
            model = models[sector][item]
            file_path_full = path_full + file_name.format(sector, item)
            model.save(file_path_full)


def load_models(years, model_class, by_sector=False, path=os.getenv('VOYA_PATH_MODELS')):
    """Load models from disk
    output is a dictionary formatted as follows
        models[sector][item] = model
    """
    path_full = f'{path}{str(years[0])}-{str(years[-1])}/'
    file_name = f'{str(years[0])}-{str(years[-1])}' + '_{}_{}.gnsm'

    sectors = SECTORS if by_sector else ['all']

    models = dict()

    for sector in sectors:
        models[sector] = dict()
        for item in ITEMS:
            model = model_class.load(path_full+file_name.format(sector, item))
            models[sector][item] = model

    return models


def plot_topics(model, num_topics=10, topn=10):
#     top_topics = model.top_topics(corpus=corpus, topn=topn)
    # fig, ax_list = plt.subplots(np.ceil(num_topics/2), np.ceil(num_topics/2))
    for t in range(num_topics):
        topic = model.show_topic(t, topn=topn)
        words = [pair[0] for pair in topic]
        probs = [pair[1] for pair in topic]
        
        fig, ax = plt.subplots()
        plt.rcdefaults()
        y_pos = np.arange(len(words))
        
        ax.barh(y_pos, probs, align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(words)
        ax.invert_yaxis()
        ax.set_xlabel('Probability')
        ax.set_title(f'Topic id: {t+1}')
        
        plt.show()