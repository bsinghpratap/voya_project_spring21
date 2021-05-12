import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import ast

from util import ITEMS
from sklearn.metrics import f1_score, cohen_kappa_score, plot_precision_recall_curve
from vanilla_lda.Predictor import get_preds

#%%
def load_results(args=None, path='../results/', by_sector=False):
    results = dict() if by_sector else list()
    comp_keys = ['start', 'end', 'validate', 'predict']
    for root_dirname in os.listdir(path):
        if os.path.isdir(path+root_dirname):
            for file_name in os.listdir(path+root_dirname):
                if file_name.endswith('.pkl'):
                    # start, end, validate, predict, sector, _ = re.split(', |_|-|\.|\+', file_name)
                    with open(path+root_dirname+'/'+file_name, mode='rb') as file:
                        obj = pickle.load(file)
                        obj['category'] = root_dirname
                        if args is None or all(obj[key] == args[key] for key in comp_keys):
                            if by_sector:
                                results[obj['sector']] = obj
                            else:
                                results.append(obj)

    return results

#%%
results = load_results()

#%%
columns = list(results[0].keys())
df = pd.DataFrame(results)

#%% Add score columns
df['f1'] = df.apply(lambda row : f1_score(row['y_true'], row['y_pred']), axis=1)
df['ck'] = df.apply(lambda row : cohen_kappa_score(row['y_true'], row['y_pred']), axis=1)

#%%
# compiled = pd.read_csv('../results/compiled-v1.csv')
compiled = df
category = {
    2012: 'append_all',
    2013: 'append_all',
    2014: 'default'
}

for start in range(2012, 2015):
    end = start+3
    all_select = compiled[(compiled.start == start) & (compiled.end == end)]
    sector_select = all_select[(compiled.sector != 'global') & (compiled.category == category[start])]
    global_select = all_select[(compiled.sector == 'global')]
    print(start, end)
    for metric in ('f1', 'ck'):
        print(
              "\t", metric,
              round(sector_select[metric].mean(), 3), ",",
              round(sector_select[metric].median(), 3), ",",
              round(sector_select[metric].std(), 3))
        print("\tglobal", metric,
              round(global_select[metric].mean(), 3), ",",
              round(global_select[metric].median(), 3), ",",
              round(global_select[metric].std(), 3))

#%% Get PRC for best model
best = results[0]
get_preds(
    sector=best['sector'],
    item=ITEMS,
    params=(best['params']),
    plot_roc=True
)

#%% Plot Scores
labels = [s+5 for s in range(2012, 2015)]
f1 = [compiled[(compiled.start == s) & (compiled.end == s+3) &
               (compiled.sector != 'global') & (compiled.category == category[s])].f1.mean() for s in range(2012, 2015)]
ck = [compiled[(compiled.start == s) & (compiled.end == s+3) &
               (compiled.sector != 'global') & (compiled.category == category[s])].ck.mean() for s in range(2012, 2015)]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width / 2, f1, width, label='F1')
rects2 = ax.bar(x + width / 2, ck, width, label='CK')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Scores')
ax.set_title("Average scores across all sectors")
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=30)
ax.legend()

# fig.tight_layout()

plt.show()

#%% Plot Scores Append
# labels = [f'{s}-{s+3}' for s in range(2012, 2015)]
default = [compiled[(compiled.start == s) & (compiled.end == s+3) &
                    (compiled.sector != 'global') & (compiled.category == 'default')].f1.mean() for s in range(2012, 2015)]
append = [compiled[(compiled.start == s) & (compiled.end == s+3) &
                   (compiled.sector != 'global') & (compiled.category == 'append_all')].f1.mean() for s in range(2012, 2015)]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width / 2, default, width, label='Default')
rects2 = ax.bar(x + width / 2, append, width, label='Appended All')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Scores')
ax.set_title("Average F1 scores")
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=30)
ax.legend()

# fig.tight_layout()

plt.show()

#%% Count Documents
for start in range(2012, 2015):
    glob = compiled[(compiled.sector == 'global') & (compiled.start == start) & (compiled.end == start+3)]
    assert len(glob) == 1
    glob = glob.iloc[0]
    print(f"{start}-{start+3}", len(glob.y_true))
