import os
import sys
import getopt
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib import rcParams
from gensim.corpora import Dictionary

SECTORS = ['Industrials', 'Tech', 'Commodities', 'Consumer', 'Health Care', 'Real Estate', 'Utilities']
ITEMS = ['item1a', 'item7']

rcParams.update({'figure.autolayout': True})


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


def load_gensim_data(years, path=os.getenv('VOYA_PATH_DATA_GENSIM'), dictionary=False):
    """:returns
        dictionary=False -> dict[sector][item][corpus/id2word] = Corpus/Id2word
        dictionary=True -> dict[sector][item] = Dictionary
    """
    year_str = str(years[0]) + '-' + str(years[-1])
    objs = dict()
    for filename in os.listdir(path+year_str+'/'):
        if filename.endswith('.pkl') or filename.endswith('.gnsm'):
            _, sector, item, gtype = filename.split('_')
            gtype = gtype.split('.')[0]

            if not dictionary and gtype != 'dictionary':
                if sector not in objs:
                    objs[sector] = {
                        'item1a': dict(),
                        'item7': dict()
                    }
                objs[sector][item][gtype] = unpickle(f'{year_str}_{sector}_{item}_{gtype}.pkl', path=path+year_str+'/')
            elif dictionary and gtype == 'dictionary':
                if sector not in objs:
                    objs[sector] = dict()
                objs[sector][item] = Dictionary.load(path+year_str+'/'+f'{year_str}_{sector}_{item}_{gtype}.gnsm')

    return objs


def save_models(models, years, path=os.getenv('VOYA_PATH_MODELS')):
    """Save models to disk
    models object should be dictionary formatted as follows
        models[sector][item] = model
    """
    path_full = f'{path}{str(years[0])}-{str(years[-1])}/'
    file_name = f'{str(years[0])}-{str(years[-1])}' + '_{}_{}_{}.gnsm'

    if not os.path.exists(path_full):
        os.makedirs(path_full)

    for sector in models:
        for item in models[sector]:
            model = models[sector][item]
            file_path_full = path_full + file_name.format(sector, item, model.num_topics)
            model.save(file_path_full)


def load_models(years, model_class, by_sector=False, path=os.getenv('VOYA_PATH_MODELS')):
    """Load models from disk
    output is a dictionary formatted as follows
        models[sector][item] = model
    """
    path_full = f'{path}{str(years[0])}-{str(years[-1])}/'
    file_name = f'{str(years[0])}-{str(years[-1])}' + '_{}_{}' + '_30.gnsm'

    sectors = SECTORS if by_sector else ['all']

    models = dict()

    for sector in sectors:
        models[sector] = dict()
        for item in ITEMS:
            model = model_class.load(path_full+file_name.format(sector, item))
            models[sector][item] = model

    return models


def get_args(arg_map):
    options = list()
    for key in arg_map:
        if type(arg_map[key]) is bool:
            options.append(key)
        else:
            options.append(key+'=')

    argv = sys.argv[1:]
    opts, args = getopt.getopt(argv, '', options)

    for opt, val in opts:
        arg_map[opt[2:]] = val if val else True

    return arg_map


def plot_topics(model, num_topics=10, topn=10, title=None, path=None):
#     top_topics = model.top_topics(corpus=corpus, topn=topn)
    # fig, ax_list = plt.subplots(np.ceil(num_topics/2), np.ceil(num_topics/2))

    for t in range(num_topics):
        topic = model.show_topic(t, topn=topn)
        words = [pair[0] for pair in topic]
        probs = [pair[1] for pair in topic]
        
        fig, ax = plt.subplots()
        fig.subplots_adjust(left=0.2)
        # plt.tight_layout()
        plt.rcdefaults()
        y_pos = np.arange(len(words))
        
        ax.barh(y_pos, probs, align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(words, rotation=30)
        ax.invert_yaxis()
        ax.set_xlabel('Probability')
        # ax.set_title(f'Topic id: {t+1}')
        plt_title = title.format(t+1)
        ax.set_title(plt_title)

        if not path:
            plt.show()
        else:
            plt.savefig(f'{path}/{plt_title.replace(":", "-")}.png')
            plt.close()


def plot_coherence(coherence_scores, title="Coherence Scores"):
    labels = coherence_scores.keys()
    item1a = [s['item1a'] for s in coherence_scores.values()]
    item7 = [s['item7'] for s in coherence_scores.values()]

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, item1a, width, label='Item1a')
    rects2 = ax.bar(x + width / 2, item7, width, label='Item7')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Coherence Score')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30)
    ax.legend()

    # fig.tight_layout()

    plt.show()


#%% Load Weights from file # TODO: Change YEARS
def load_weights_file(sector, item, years, num_topics, all=False, YEARS=None):
    target_path = f"{os.getenv('VOYA_PATH_MODELS')}{YEARS[0]}-{YEARS[-1]}/weights/"
    target_name = f'{years[0]}-{years[-1]}_{sector}_{item}_{num_topics}'
    if all: target_name += '_all'
    target_name += '.pkl'

    weights = None

    if not os.path.isdir(target_path):
        pass
    elif os.path.isfile(target_path + target_name):
        # print("Loading weights from file", sector, item, years, num_topics, 'all' if all else 'sector')
        with open(target_path + target_name, mode='rb') as file:
            weights = pickle.load(file)
    return weights


def save_weights_file(sector, item, years, num_topics, weights, all=False):
    # print("Saving weights to file", sector, item, years, num_topics, 'all' if all else 'sector')
    target_path = f"{os.getenv('VOYA_PATH_MODELS')}{YEARS[0]}-{YEARS[-1]}/weights/"
    target_name = f'{years[0]}-{years[-1]}_{sector}_{item}_{num_topics}'
    if all: target_name += '_all'
    target_name += '.pkl'

    if not os.path.isdir(target_path):
        os.mkdir(target_path)

    with open(target_path + target_name, mode='wb') as file:
        pickle.dump(weights, file)