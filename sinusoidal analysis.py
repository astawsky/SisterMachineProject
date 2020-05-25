
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys, math
import glob
import pickle
import os
import scipy.stats as stats
from scipy import signal
import random
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import seaborn as sns
import itertools
import NewSisterCellClass as ssc


def log_the_gen_dict(**kwargs):
    dictionary = kwargs.get('dictionary', [])
    variables_to_log = kwargs.get('variables_to_log', [])

    # the new dictionary with the new dataframes
    new_dictionary = dict()


    print(dictionary.keys())
    # loop over all trajectories in S and NS datasets
    for key in dictionary.keys():
        # to have everything else the same
        dictionary_key = dictionary[key].copy()

        for var in variables_to_log:
            # normalizing the two length observations for all trajectories in S and NS datasets
            dictionary_key[var] = np.log(dictionary[key][var])

        # save the new dataframe to the new dictionary
        new_dictionary.update({key: dictionary_key})

    return new_dictionary


def main():
    # Import the objects we want
    pickle_in = open("NewSisterCellClass_Population.pickle", "rb")
    Population = pickle.load(pickle_in)
    pickle_in.close()

    pickle_in = open("NewSisterCellClass_Sister.pickle", "rb")
    Sister = pickle.load(pickle_in)
    pickle_in.close()

    pickle_in = open("NewSisterCellClass_Nonsister.pickle", "rb")
    Nonsister = pickle.load(pickle_in)
    pickle_in.close()

    pickle_in = open("NewSisterCellClass_Control.pickle", "rb")
    Control = pickle.load(pickle_in)
    pickle_in.close()

    for A_dict, B_dict, dataID in zip(Sister.A_dict.values(), Sister.B_dict.values(), range(len(Sister._data))):

        # var = 'growth_rate'
        # # plt.plot(Sister._data[dataID]['timeA'].iloc[0] + np.cumsum(A_dict['generationtime']), A_dict[var], label='A_{}'.format(var), marker='.',
        # #          color='blue')
        # # plt.plot(Sister._data[dataID]['timeA'].iloc[0] + np.cumsum(A_dict['generationtime']), A_dict['generationtime'], label='A_{}'.format('generationtime'), marker='.',
        # #          color='orange')
        # # plt.axhline(y=np.mean(A_dict['generationtime']), color='blue')
        # # plt.plot(Sister._data[dataID]['timeB'].iloc[0] + np.cumsum(B_dict['generationtime']), B_dict[var], label='B_{}'.format(var), marker='.',
        # #          color='orange')
        # # plt.axhline(y=np.mean(B_dict['generationtime']), color='orange')
        # plt.plot(Sister._data[dataID]['timeA'].iloc[0] + np.cumsum(A_dict['generationtime']), A_dict[var]-np.mean(pd.concat([A_dict[var], B_dict[var]], axis=0)), label='A_{}'.format(var), marker='.',
        #          color='blue')
        # plt.plot(Sister._data[dataID]['timeA'].iloc[0] + np.cumsum(A_dict['generationtime']), A_dict['generationtime']-np.mean(pd.concat([A_dict['generationtime'], B_dict['generationtime']], axis=0)),
        #          label='A_{}'.format('generationtime'), marker='.',
        #          color='orange')
        # # plt.axhline(y=np.mean(pd.concat([A_dict[var], B_dict[var]], axis=0)), color='black', label='{} trap mean'.format(var))
        # # plt.axhline(y=np.mean(pd.concat([A_dict['generationtime'], B_dict['generationtime']], axis=0)), color='black', label='generationtime trap mean')
        # plt.legend()
        # plt.show()
        # plt.close()

        var = 'generationtime'
        plt.plot(Sister._data[dataID]['timeA'].iloc[0] + np.cumsum(A_dict['generationtime']),
                 A_dict[var], label='A_{}'.format(var), marker='.',
                 color='blue')
        plt.plot(Sister._data[dataID]['timeB'].iloc[0] + np.cumsum(B_dict['generationtime']), B_dict[var], label='B_{}'.format(var), marker='.', color='orange')
        plt.axhline(y=np.mean(pd.concat([A_dict[var].copy(), B_dict[var].copy()], axis=0)), color='black', label='{} trap mean'.format(var))
        plt.legend()
        plt.show()
        plt.close()


        #
        # for var in A_dict.columns:
        #
        #     plt.plot(Sister._data[dataID]['timeA'].iloc[0]+np.cumsum(A_dict['generationtime']), A_dict[var], label='A_{}'.format(var), marker='.', color='blue')
        #     # plt.axhline(y=np.mean(A_dict['generationtime']), color='blue')
        #     plt.plot(Sister._data[dataID]['timeB'].iloc[0] + np.cumsum(B_dict['generationtime']), B_dict[var], label='B_{}'.format(var), marker='.', color='orange')
        #     # plt.axhline(y=np.mean(B_dict['generationtime']), color='orange')
        #     plt.axhline(y=np.mean(pd.concat([A_dict[var], B_dict[var]], axis=0)), color='black', label='trap mean')
        #     plt.legend()
        #     plt.show()
        #     plt.close()





    exit()

    # getting the log dictionaries
    log_A_dict = log_the_gen_dict(dictionary=Sister.A_dict, variables_to_log=['length_birth', 'length_final', 'division_ratio'])
    log_B_dict = log_the_gen_dict(dictionary=Sister.B_dict, variables_to_log=['length_birth', 'length_final', 'division_ratio'])

    # to get the global mean we copy the mother df, log the necessary columns, and get the mean of the columns we care about
    all_of_them = Population.mother_dfs[0].copy()
    for col in ['length_birth', 'length_final', 'division_ratio']:
        all_of_them[col] = np.log(all_of_them[col])
    global_mean = all_of_them.mean()[Population._variable_names]
    print('global mean', global_mean)

    sup_dict = dict(zip(['global_A', 'global_B', 'trap_A', 'trap_B', 'traj_A', 'traj_B'], [[], [], [], [], [], []]))

    # for each of the different types of averages, get the dataframe that contains the normalized data with the appropriate mean used
    for df_A, df_B in zip(log_A_dict.values(), log_B_dict.values()):
        # get the global, trap and traj means as pd.series
        trap_mean = pd.concat([df_A.copy(), df_B.copy()], axis=0).mean()
        A_traj_mean = df_A.copy().mean()
        B_traj_mean = df_B.copy().mean()
        # print(trap_mean, A_traj_mean, B_traj_mean)
        # print(df_A.columns)

        global_A = df_A.copy() - global_mean
        global_B = df_B.copy() - global_mean

        trap_A = df_A.copy() - trap_mean
        trap_B = df_B.copy() - trap_mean

        traj_A = df_A.copy() - A_traj_mean
        traj_B = df_B.copy() - B_traj_mean

        # for all mean dataframes
        for df, key in zip([global_A, global_B, trap_A, trap_B, traj_A, traj_B], ['global_A', 'global_B', 'trap_A', 'trap_B', 'traj_A', 'traj_B']):
            # normalize them
            for col in global_A.columns:
                df[col] = df[col]/(df[col].std())

            # print('df', df)
            # update the dictionary
            sup_dict[key].append(df)
            # sup_dict.update({key: sup_dict[key].append(df)})

        """

        # append them all to he avegs list
        avegs = [global_A, global_B, trap_A, trap_B, traj_A, traj_B]


        print(np.cumsum(df_A['generationtime']))
        print(global_A.var())

        for aveg in avegs:
            for col in global_A.columns:
                aveg[col] = aveg[col]/(aveg[col].std())

        print(avegs[0])




        # global_A['average'] = 'global'
        # global_B['average'] = 'global'
        # trap_A['average'] = 'trap'
        # trap_B['average'] = 'trap'
        # traj_A['average'] = 'traj'
        # traj_B['average'] = 'traj'

        # global_A['realtime'] = np.cumsum(global_A['generationtime'])
        # global_B['realtime'] = np.cumsum(global_B['generationtime'])
        # trap_A['realtime'] = np.cumsum(trap_A['generationtime'])
        # trap_B['realtime'] = np.cumsum(trap_B['generationtime'])
        # traj_A['realtime'] = np.cumsum(traj_A['generationtime'])
        # traj_B['realtime'] = np.cumsum(traj_B['generationtime'])

        print(avegs[0].columns)

        for num in range(len(avegs)):

            sns.lineplot(data=avegs[num][['generationtime', 'growth_rate', 'length_birth']], marker='.')
            # sns.lineplot(x=np.cumsum(df_B['generationtime']), y=global_B.div(global_B.var(), axis=0), label='global_B')
            plt.legend()
            plt.show()
            plt.close()

        print(global_A.columns)


        combines = pd.concat([global_A, global_B, trap_A, trap_B, traj_A, traj_B], axis=0)
        print(combines)
        print(combines.columns)

        # for avg in ['global', 'trap', 'traj']:
        #     combines[combines['average']==avg]

        g = sns.FacetGrid(data=combines, col='average') #, hue='average'
        g.map(sns.lineplot)
        g.map(plt.gca().axhline, y=0, color='black')
        plt.show()
        plt.close()
        """

    for num in range(130):

        # print(sup_dict['global_A'][num]['generationtime'])
        # print(np.cumsum(sup_dict['global_A'][num]['generationtime']))


        # fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1)

        sns.set_style('darkgrid')

        plt.plot(range(len(sup_dict['global_A'][num]['generationtime'])),  sup_dict['global_A'][num]['generationtime'], marker='.', label=r'$\tau_A$', color='blue')
        plt.plot(range(len(sup_dict['global_B'][num]['generationtime'])), sup_dict['global_B'][num]['generationtime'], marker='.', label=r'$\tau_B$', alpha=.5, color='blue')
        plt.plot(range(len(sup_dict['global_A'][num]['growth_rate'])), sup_dict['global_A'][num]['growth_rate'], marker='.', label=r'$\alpha_A$', color='orange')
        plt.plot(range(len(sup_dict['global_B'][num]['growth_rate'])), sup_dict['global_B'][num]['growth_rate'], marker='.', label=r'$\alpha_B$', alpha=.5, color='orange')
        plt.plot(range(len(sup_dict['global_A'][num]['fold_growth'])), sup_dict['global_A'][num]['fold_growth'], marker='.', label=r'$\phi_A$', color='green')
        plt.plot(range(len(sup_dict['global_B'][num]['fold_growth'])), sup_dict['global_B'][num]['fold_growth'], marker='.', label=r'$\phi_B$', alpha=.5, color='green')
        # ax2.plot(x=np.cumsum(sup_dict['global_A'][num]['generationtime']), y=sup_dict['global_A'][num]['generationtime'], label=r'$\tau$')
        # ax3.plot(x=np.cumsum(sup_dict['global_A'][num]['generationtime']), y=sup_dict['global_A'][num]['generationtime'], label=r'$\tau$')

        plt.legend()
        plt.show()
    exit()

    # for df_A, df_B in zip(Sister.A_dict.values(), Sister.B_dict.values()):
    #     trap_mean = pd.concat([df_A, df_B], axis=0).mean()
    #     A_traj_mean = df_A.mean()
    #     B_traj_mean = df_B.mean()



if __name__ == '__main__':
    main()
