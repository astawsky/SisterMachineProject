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
from sklearn.neural_network import MLPRegressor


def GetTheTrainTestSets(struct, factors, targets):
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
    print('population statistics:', pop_means, pop_stds, len(pop_means), len(pop_stds))

    # columns for the dataframe
    cols = ['generationtime_m', 'length_birth_m', 'length_final_m', 'growth_length_m', 'division_ratios__f_n_m', 'phi_m',
            'generationtime_d', 'length_birth_d', 'length_final_d', 'growth_length_d', 'division_ratios__f_n_d', 'phi_d',
            'trap_average_generationtime', 'trap_average_length_birth', 'trap_average_length_final',
            'trap_average_growth_length', 'trap_average_division_ratios__f_n', 'trap_average_phi']

    # Data Frame
    m_d_dependance = pd.DataFrame(columns=cols)

    # ind is the index to add everything to the mother daughter dataframe
    ind = 0
    A_sis = np.array([key for key in struct.A_dict_sis.keys()])
    B_sis = np.array([key for key in struct.B_dict_sis.keys()])

    # the experiment A cell data
    for key in A_sis:
        # the dataframe with cycle params
        val = struct.A_dict_sis[key]
        val['phi'] = val['generationtime'] * val['growth_length']
        # all the cycle params' means
        trap_averages = np.mean(val)
        # turn them into a standard normal value
        trap_averages = (trap_averages - pop_means) / pop_stds
        # trap_averages = (trap_averages) / pop_stds
        for row1, row2 in zip(range(len(val) - 1), range(1, len(val))):  # row1 is the previous generation of row2, ie. the mom and daughter
            mom = np.array(val.iloc[row1])
            daughter = np.array(val.iloc[row2])
            # convert them to standard normal values
            mom = (mom - pop_means) / pop_stds
            daughter = (daughter - pop_means) / pop_stds
            # mom = (mom) / pop_stds
            # daughter = (daughter) / pop_stds
            # concatenate them and add them to the dataframe
            together = pd.Series(np.append(mom, np.append(daughter, trap_averages)), index=m_d_dependance.columns)
            m_d_dependance.loc[ind] = together
            # move on to the next generation
            ind = ind + 1

    print('A sis traces finished')

    # the experiment B cell data
    for key in B_sis:
        # the dataframe with cycle params
        val = struct.B_dict_sis[key]
        val['phi'] = val['generationtime'] * val['growth_length']
        # all the cycle params' means
        trap_averages = np.mean(val)
        # turn them into a standard normal value
        trap_averages = (trap_averages - pop_means) / pop_stds
        # trap_averages = (trap_averages) / pop_stds
        for row1, row2 in zip(range(len(val) - 1), range(1, len(val))):  # row1 is the previous generation of row2, ie. the mom and daughter
            mom = np.array(val.iloc[row1])
            daughter = np.array(val.iloc[row2])
            # convert them to standard normal values
            mom = (mom - pop_means) / pop_stds
            daughter = (daughter - pop_means) / pop_stds
            # mom = (mom) / pop_stds
            # daughter = (daughter) / pop_stds
            # concatenate them and add them to the dataframe
            together = pd.Series(np.append(mom, np.append(daughter, trap_averages)), index=m_d_dependance.columns)
            m_d_dependance.loc[ind] = together
            # move on to the next generation
            ind = ind + 1

    print('B sis traces finished')

    # ind is the index to add everything to the mother daughter dataframe
    A_non_sis = np.array([key for key in struct.A_dict_non_sis.keys()])
    B_non_sis = np.array([key for key in struct.B_dict_non_sis.keys()])

    # the experiment A cell data
    for key in A_non_sis:
        # the dataframe with cycle params
        val = struct.A_dict_non_sis[key]
        val['phi'] = val['generationtime'] * val['growth_length']
        # all the cycle params' means
        trap_averages = np.mean(val)
        # turn them into a standard normal value
        trap_averages = (trap_averages - pop_means) / pop_stds
        # trap_averages = (trap_averages) / pop_stds
        for row1, row2 in zip(range(len(val) - 1), range(1, len(val))):  # row1 is the previous generation of row2, ie. the mom and daughter
            mom = np.array(val.iloc[row1])
            daughter = np.array(val.iloc[row2])
            # convert them to standard normal values
            mom = (mom - pop_means) / pop_stds
            daughter = (daughter - pop_means) / pop_stds
            # mom = (mom) / pop_stds
            # daughter = (daughter) / pop_stds
            # concatenate them and add them to the dataframe
            together = pd.Series(np.append(mom, np.append(daughter, trap_averages)), index=m_d_dependance.columns)
            m_d_dependance.loc[ind] = together
            # move on to the next generation
            ind = ind + 1

    print('A non sis traces finished')

    # the experiment B cell data
    for key in B_non_sis:
        # the dataframe with cycle params
        val = struct.B_dict_non_sis[key]
        val['phi'] = val['generationtime'] * val['growth_length']
        # all the cycle params' means
        trap_averages = np.mean(val)
        # turn them into a standard normal value
        trap_averages = (trap_averages - pop_means) / pop_stds
        # trap_averages = (trap_averages) / pop_stds
        for row1, row2 in zip(range(len(val) - 1), range(1, len(val))):  # row1 is the previous generation of row2, ie. the mom and daughter
            mom = np.array(val.iloc[row1])
            daughter = np.array(val.iloc[row2])
            # convert them to standard normal values
            mom = (mom - pop_means) / pop_stds
            daughter = (daughter - pop_means) / pop_stds
            # mom = (mom) / pop_stds
            # daughter = (daughter) / pop_stds
            # concatenate them and add them to the dataframe
            together = pd.Series(np.append(mom, np.append(daughter, trap_averages)), index=m_d_dependance.columns)
            m_d_dependance.loc[ind] = together
            # move on to the next generation
            ind = ind + 1

    print('B non sis traces finished')

    ''' We leave out final length so that length birth of daughter is not a trivial combination of the division ratio and final length of the mother
        UPDATE: We also leave out the division ratios because we are getting weird results... '''
    X_train, X_test, y_train, y_test = train_test_split(m_d_dependance[factors], m_d_dependance[targets], test_size=0.33, random_state=42)

    return X_train, X_test, y_train, y_test

def main():
    # Import the Refined Data
    pickle_in = open("metastructdata.pickle", "rb")
    struct = pickle.load(pickle_in)
    pickle_in.close()

    ''' factors are what we put in to the matrix and targets are what we get out '''
    factors = ['generationtime_m', 'length_birth_m', 'growth_length_m', 'division_ratios__f_n_m', 'phi_m']  # 'length_final_m',
    # 'trap_average_length_final',
    # 'trap_average_division_ratios__f_n', 'trap_average_generationtime', 'trap_average_length_birth',
    #             'trap_average_growth_length'
    targets = ['generationtime_d', 'length_birth_d', 'growth_length_d', 'division_ratios__f_n_d', 'phi_d']  # 'length_final_d'

    X_train, X_test, y_train, y_test = GetTheTrainTestSets(struct, factors, targets)

    ''' Activating the neural network '''
    est = MLPRegressor()
    est.fit(X_train, y_train)
    results = est.predict(X_test)
    results = pd.DataFrame(data=results, index=range(results.shape[0]), columns=y_test.columns)

    ''' see if the simulated daughters contain similar same-cell correlations as the data '''
    # memory = []
    # print(results)
    # for col1 in results.columns:
    #     for col2 in results.columns:
    #         if col1 != col2 and [col2, col1] not in memory:
    #             x = results[col1]
    #             y = results[col2]
    #
    #             m, c = np.linalg.lstsq(np.concatenate([x[:, np.newaxis], np.ones_like(x)[:, np.newaxis]], axis=1), y)[0]
    #             plt.scatter(x, y)
    #             plt.plot(np.linspace(min(x), max(x)), np.linspace(min(x), max(x)) * m + c, label=r'$\rho=${}'.format(round(stats.pearsonr(x, y)[0],
    #                                                                                                                        3)), color='orange')
    #             plt.legend()
    #             plt.xlabel(col1)
    #             plt.ylabel(col2)
    #             plt.show()
    #             memory.append([col1, col2])

    ''' the pearson correlations of the simulated and real testing set USING MATRIX'''
    fig, axs = plt.subplots(1, len(y_test.columns), sharey=True)
    # fig.text(0.04, 0.5, 'Simulated', ha='center')
    ind = 0
    for col in y_test.columns:
        print(np.ones_like(y_test[col])[:, np.newaxis])
        print(results[col])
        m, c = np.linalg.lstsq(np.concatenate([np.array(y_test[col])[:, np.newaxis], np.ones_like(y_test[col])[:, np.newaxis]], axis=1),
                               results[col])[0]
        axs[ind].scatter(y_test[col], results[col], label='_nolegend_')
        axs[ind].plot(np.linspace(min(y_test[col]), max(y_test[col])), np.linspace(min(y_test[col]),
                                                                                   max(y_test[col])) * m + c,
                      label=r'$\rho$, pval: {}, {:.2e}'.format(round(stats.pearsonr(y_test[col], results[col])[0], 3),
                                                               stats.pearsonr(y_test[col], results[col])[1]), color='orange')
        # axs[ind].hist(error_test[col], label=r'{:.2e}$\pm${:.2e}'.format(np.mean(error_test[col]), np.std(error_test[col])))
        axs[ind].set_ylim([np.mean(results[col]) - 3 * np.std(results[col]), np.mean(results[col]) + 3 * np.std(results[col])])
        axs[ind].set_xlim([np.mean(y_test[col]) - 3 * np.std(y_test[col]), np.mean(y_test[col]) + 3 * np.std(y_test[col])])
        axs[ind].legend()
        axs[ind].set_xlabel('data ' + col)
        axs[ind].set_ylabel('simulation ' + col)
        ind = ind + 1
    plt.suptitle('correlation between real and simulated daughters in testing set')
    plt.show()


if __name__ == '__main__':
    main()
