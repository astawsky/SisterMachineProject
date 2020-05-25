from __future__ import print_function

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sys, math
import matplotlib.pyplot as plt
import random

import pickle
import scipy.stats as stats
import random


def main():
    # Import the Refined Data
    pickle_in = open("metastructdata.pickle", "rb")
    struct = pickle.load(pickle_in)
    pickle_in.close()

    # Get the population level statistics of cycle params to convert to Standard Normal variables
    sis_traces = [A for A in struct.dict_with_all_sister_traces.values()]
    non_traces = [B for B in struct.dict_with_all_non_sister_traces.values()]
    all_traces = sis_traces + non_traces
    # mean and std div for 'generationtime', 'length_birth', 'length_final', 'growth_length', 'division_ratios__f_n'
    pop_means = [np.mean(pd.concat(np.array([trace[c_p] for trace in all_traces]), ignore_index=True)) for c_p in all_traces[0].keys()]
    pop_stds = [np.std(pd.concat(np.array([trace[c_p] for trace in all_traces]), ignore_index=True)) for c_p in all_traces[0].keys()]
    phi_mean = np.mean(pd.concat(np.array([trace['generationtime'] * trace['growth_length'] for trace in all_traces]), ignore_index=True))
    phi_std = np.std(pd.concat(np.array([trace['generationtime'] * trace['growth_length'] for trace in all_traces]), ignore_index=True))
    pop_means.append(phi_mean)
    pop_stds.append(phi_std)

    columns = np.append(np.array(struct.A_dict_sis['Sister_Trace_A_0'].keys()), ['phi'])
    memory = []
    for col1 in columns:
        for col2 in columns:
            if col1 != col2 and [col2, col1] not in memory:
                x = np.array([])
                for val in struct.dict_all.values():
                    val['phi'] = val['generationtime'] * val['growth_length']
                    val = (val - pop_means) / pop_stds
                    x = np.concatenate([x, np.array(val[col1])])
                y = np.array([])
                for val in struct.dict_all.values():
                    val['phi'] = val['generationtime'] * val['growth_length']
                    val = (val - pop_means) / pop_stds
                    y = np.concatenate([y, np.array(val[col2])])
                # x = []
                # for val in struct.dict_all.values():
                #     x.append(val[col1].iloc[0])
                # y = []
                # for val in struct.dict_all.values():
                #     y.append(val[col2].iloc[0])
                # x = np.array(x)
                # y = np.array(y)
                # print(x)
                # print(y)
                m, c = np.linalg.lstsq(np.concatenate([x[:, np.newaxis], np.ones_like(x)[:, np.newaxis]], axis=1), y)[0]
                plt.scatter(x, y)
                plt.plot(np.linspace(min(x), max(x)), np.linspace(min(x),max(x)) * m + c, label=r'$\rho=${}'.format(round(stats.pearsonr(x, y)[0],
                                                                                                                          3)), color='orange')
                plt.legend()
                plt.xlabel(col1)
                plt.ylabel(col2)
                plt.show()
                memory.append([col1, col2])





if __name__=='__main__':
    main()
