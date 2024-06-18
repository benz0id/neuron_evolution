import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joypy import joyplot
from math import prod
from matplotlib import cm

np.random.seed(111)

dfa = pd.read_csv('/Users/ben/Desktop/neuron_project/pythonProject/simulation/cache/sim_stats/2024-06-14-19-00-15/all_neurons.csv',
                  index_col='epoch')

dfa = dfa.drop(columns=[dfa.columns[-1]])

attrs = dfa.columns
num_epochs = len(dfa.columns)
now_classes = len(dfa.index)
vals = np.array(dfa).T

means = vals.max(axis=1).reshape((13, 1))

norm_vals = vals / (means + 0.000000001)

new_df = pd.DataFrame({
    'epoch': np.tile(np.arange(now_classes), num_epochs),
    'cls': np.repeat(attrs, now_classes),
    'val': norm_vals.reshape((prod(dfa.shape)))
})

joyplot(new_df, by='cls',
        column='val',
        kind='values',
        x_range=np.arange(100),
        figsize=(7, 15),
        colormap=cm.get_cmap('gnuplot2'))

plt.savefig()

