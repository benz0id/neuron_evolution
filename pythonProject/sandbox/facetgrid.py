# importing packages
import seaborn
import pandas as pd
import matplotlib.pyplot as plt

# loading of a dataframe from seaborn
df = seaborn.load_dataset('tips')

df = pd.read_csv('/Users/ben/Desktop/neuron_project/pythonProject/simulation/cache/sim_stats/2024-06-14-19-00-15/all_neurons.csv',
                 index_col='epoch')



pass