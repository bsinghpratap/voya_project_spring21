from gensim.models.ldamulticore import LdaMulticore
from gensim.models.coherencemodel import CoherenceModel
from util import load_data, load_models, load_gensim_data, plot_topics, plot_coherence, SECTORS, ITEMS
from GensimPreprocessor import process
SECTORS = ['all', *SECTORS]
# from multiprocessing import Pool
#%% Parameters
FUNCTION = "coherence"
years = (2012, 2015)
coherence = 'u_mass'
# coherence = 'c_npmi'

#%% Load Models
models = load_models(years, model_class=LdaMulticore, by_sector=True)
models['all'] = load_models(years, model_class=LdaMulticore, by_sector=False)['all']

""" Topic Plotting """
# %% Plot Topics
# for item in ['item1a', 'item7']:
#     for sector in SECTORS:
#         title = f'sector: {sector}-{item}, topic_id: ' + '{}'
#         plot_topics(models[sector][item], num_topics=3, topn=7, title=title, path='../figures/vanillalda')


""" Coherence """
#%% Load Data
data, _, _ = load_data()
data.query('year_x in @years', inplace=True)

#%% Setup
coherence_scores = dict()
gensim_data = load_gensim_data(years)
dicts = load_gensim_data(years, dictionary=True)

#%% Get coherence per sector
# pool = Pool()
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
        if sector == 'all':
            docs = data[item_col]
        else:
            docs = data.query('sector == @sector')[item_col]
        params['texts'] = process(docs, return_docs=True).to_list()
    else:
        params['corpus'] = corpus

    cm = CoherenceModel(**params)
    coherence_scores[sector][item] = cm.get_coherence()
    print("Completed coherence for:", sector, item)


# for sector in SECTORS:
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

#%% Plot Coherence
plot_coherence(coherence_scores, title=f'Coherence Scores: {coherence}')
