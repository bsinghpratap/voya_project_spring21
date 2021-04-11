import pyLDAvis.gensim_models as pyLDAvis_gensim
import pyLDAvis
from gensim.models.ldamulticore import LdaMulticore

from util import load_models, load_gensim_data, ITEMS

#%% Params
YEARS = (2012, 2015)

#%% Load Models
models = load_models(YEARS, LdaMulticore, by_sector=True)
models['all'] = load_models(YEARS, LdaMulticore, by_sector=False)['all']

#%% Load Gensim Data
model_data = load_gensim_data(YEARS)
model_dicts = load_gensim_data(YEARS, dictionary=True)

#%% Visualize
for sector in models:
    for item in ITEMS:

        print("Starting", sector, item)

        vis_data = pyLDAvis_gensim.prepare(
            topic_model=models[sector][item],
            corpus=model_data[sector][item]['corpus'],
            dictionary=model_dicts[sector][item])
        file_name = f'../pyldavis/{YEARS[0]}-{YEARS[-1]}/{sector}-{item}.html'
        pyLDAvis.save_html(vis_data, file_name)

        print("Completed", sector, item)


#%% Save
# with open(f'../pyldavis/2012-2015/{sector}-{item}') as file:
