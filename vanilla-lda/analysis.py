from gensim.models.ldamulticore import LdaMulticore
from gensim.models.coherencemodel import CoherenceModel
from util import load_data, load_models, load_gensim_data, plot_topics, SECTORS, ITEMS
from GensimPreprocessor import process

from multiprocessing import Pool
#%% Parameters
FUNCTION = "coherence"
years = (2012, 2015)
coherence = 'c_v'

#%% Load Models
models = load_models(years, model_class=LdaMulticore, by_sector=True)
#%% Load Data
data, _, _ = load_data()
data.query('year_x in @years', inplace=True)


""" Topic Plotting """
#%% Plot Topics
# for item in ['item1a']:
#     for sector in SECTORS:
#         title = 'Sector:', sector, 'item:', item
#         plot_topics(models[sector][item], 5, 10, title=title)


""" Coherence """
#%% Get coherence per sector
coherence_scores = dict()
gensim_data = load_gensim_data(years)
dicts = load_gensim_data(years, dictionary=True)

pool = Pool()
procs = list()


def get_coherence(sector, item, coherence_scores):
    print("Starting coherence for:", sector, item)
    params = {
        'model': models[sector][item],
        'coherence': coherence,
        'dictionary': dicts[sector][item]
    }
    corpus = gensim_data[sector][item]['corpus']

    if coherence != 'u_mass':
        item_col = 'item1a_risk' if item == 'item1a' else 'item7_mda'
        docs = data.query('sector == @sector')[item_col]
        params['texts'] = process(docs, return_docs=True).to_list()
    else:
        params['corpus'] = corpus

    cm = CoherenceModel(**params)
    coherence_scores[sector][item] = cm.get_coherence()
    print("Completed coherence for:", sector, item)

for sector in SECTORS:
    coherence_scores[sector] = dict()
    for item in ITEMS:
        # ret = pool.apply_async(get_coherence, (sector, item, coherence_scores))
        # procs.append(ret)
        get_coherence(sector, item, coherence_scores)
        # params = {
        #     'model': models[sector][item],
        #     'coherence': coherence,
        #     'dictionary': dicts[sector][item]
        # }
        # corpus = gensim_data[sector][item]['corpus']
        #
        # if coherence != 'u_mass':
        #     item_col = 'item1a_risk' if item == 'item1a' else 'item7_mda'
        #     docs = data.query('sector == @sector')[item_col]
        #     params['texts'] = process(docs, return_docs=True).to_list()
        # else:
        #     params['corpus'] = corpus
        #
        # cm = CoherenceModel(**params)
        # coherence_scores[sector][item] = cm.get_coherence()

    #     break
    # break
