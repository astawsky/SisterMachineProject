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
import sklearn
from sklearn.model_selection import train_test_split
import seaborn as sns
import itertools
import NewSisterCellClass as ssc
import pymc3 as pm
from scipy.special import kn as bessel_function_of_second_kind
from scipy.special import comb as combinations


def give_the_trap_avg_for_control(key, sis_A, sis_B, non_sis_A, non_sis_B):
    # give the key of the reference of the dictionary, for example, "nonsis_A_57"
    if key.split('_')[0] == 'sis':
        ref_A = sis_A[key.split('_')[2]]
        ref_B = sis_B[key.split('_')[2]]
        trap_mean = pd.concat([ref_A, ref_B], axis=0).reset_index(drop=True).mean()
    else:
        ref_A = non_sis_A[key.split('_')[2]]
        ref_B = non_sis_B[key.split('_')[2]]
        trap_mean = pd.concat([ref_A, ref_B], axis=0).reset_index(drop=True).mean()

    return trap_mean


def subtract_trap_averages(df, columns_names, trap_mean):
    df_new = pd.DataFrame(columns=columns_names)
    for col in columns_names:
        df_new[col] = df[col] - trap_mean[col]

    return df_new


def subtract_traj_averages(df, columns_names):
    df_new = pd.DataFrame(columns=columns_names)
    for col in columns_names:
        df_new[col] = df[col] - np.mean(df[col])

    return df_new


def subtract_trap_averages_control(df, columns_names, ref_key, sis_A, sis_B, non_sis_A, non_sis_B):

    # we have to get the original trap mean
    trap_mean = give_the_trap_avg_for_control(ref_key, sis_A, sis_B, non_sis_A, non_sis_B)

    df_new = pd.DataFrame(columns=columns_names)
    for col in columns_names:
        df_new[col] = df[col] - trap_mean[col]

    return df_new


def subtract_global_averages(df, columns_names, pop_mean):
    df_new = pd.DataFrame(columns=columns_names)
    for col in columns_names:
        df_new[col] = df[col] - pop_mean[col]

    return df_new


# here you can graph the mutual information and change the centered dictionaries (global, trap, traj, etc..)
def for_all_datasets(A_sis, B_sis, A_non_sis, B_non_sis, A_con, B_con, variable_names, graph_vars, seed):
    
    def for_all_traps(keys, variable_names, A_dict, B_dict): # this pools together all traps for one dataset

        def get_entropies(A_centered, B_centered): # this gives the conditional entropies and mutual information for two centered vectors

            A_centered = A_centered.iloc[:min(len(A_centered), len(B_centered))]
            B_centered = B_centered.iloc[:min(len(A_centered), len(B_centered))]
            joint_centered = pd.DataFrame({'A': A_centered, 'B': B_centered})

            # joint probabilities
            plus_plus_joint = len(joint_centered[(joint_centered['A'] >= 0) & (joint_centered['B'] >= 0)]) / len(joint_centered)
            plus_minus_joint = len(joint_centered[(joint_centered['A'] >= 0) & (joint_centered['B'] < 0)]) / len(joint_centered)
            minus_plus_joint = len(joint_centered[(joint_centered['A'] < 0) & (joint_centered['B'] >= 0)]) / len(joint_centered)
            minus_minus_joint = len(joint_centered[(joint_centered['A'] < 0) & (joint_centered['B'] < 0)]) / len(joint_centered)

            # marginal probabilites
            A_minus_marginal = len(A_centered.iloc[np.where(A_centered < 0)]) / len(A_centered)
            A_plus_marginal = len(A_centered.iloc[np.where(A_centered >= 0)]) / len(A_centered)
            B_minus_marginal = len(B_centered.iloc[np.where(B_centered < 0)]) / len(B_centered)
            B_plus_marginal = len(A_centered.iloc[np.where(B_centered >= 0)]) / len(B_centered)

            # to make the calculation easier
            joint_prob_list = [plus_plus_joint, minus_minus_joint, plus_minus_joint, minus_plus_joint]
            A_trace_marginal_probs = [A_plus_marginal, A_minus_marginal, A_minus_marginal, A_plus_marginal]
            B_trace_marginal_probs = [B_plus_marginal, B_minus_marginal, B_plus_marginal, B_minus_marginal]
            
            every_probability = [plus_plus_joint, minus_minus_joint, plus_minus_joint, minus_plus_joint, A_plus_marginal, A_minus_marginal, B_plus_marginal, B_minus_marginal]

            # conditioning the A trace based on the B trace
            A_conditioned_on_B_entropy = np.array(
                [- joint * np.log(joint / marg) for joint, marg in zip(joint_prob_list, A_trace_marginal_probs) if joint != 0 and marg != 0]).sum()
            B_conditioned_on_A_entropy = np.array(
                [- joint * np.log(joint / marg) for joint, marg in zip(joint_prob_list, B_trace_marginal_probs) if joint != 0 and marg != 0]).sum()

            # the mutual information between A and B trace for this variable thinking that marginal A and B came from each trace distribution
            mutual_info_trace = np.array(
                [joint * np.log(joint / (marg_A * marg_B)) for joint, marg_A, marg_B in
                 zip(joint_prob_list, A_trace_marginal_probs, B_trace_marginal_probs)
                 if
                 joint != 0 and marg_A != 0 and marg_B != 0]).sum()

            # print(joint_centered)
            # print('++:\n', joint_centered[(joint_centered['A'] >= 0) & (joint_centered['B'] >= 0)])
            # print('--:\n', joint_centered[(joint_centered['A'] < 0) & (joint_centered['B'] < 0)])
            # print('+-:\n', joint_centered[(joint_centered['A'] >= 0) & (joint_centered['B'] < 0)])
            # print('-+:\n', joint_centered[(joint_centered['A'] < 0) & (joint_centered['B'] >= 0)])
            # input()
            #
            # print(plus_plus_joint)
            # print(minus_minus_joint)
            # print(plus_minus_joint)
            # print(minus_plus_joint)
            # print('1=', plus_plus_joint+plus_minus_joint+minus_plus_joint+minus_minus_joint)
            # print('______')
            # print(A_minus_marginal)
            # print(A_plus_marginal)
            # print('1=', A_minus_marginal+A_plus_marginal)
            # print(B_minus_marginal)
            # print(B_plus_marginal)
            # print('1=', B_minus_marginal + B_plus_marginal)
            # exit()

            return A_conditioned_on_B_entropy, B_conditioned_on_A_entropy, mutual_info_trace, every_probability

        # the dictionary where the keys are the variables and the values are the entropies of all the traps in the dataset
        A_conditioned_on_B_entropy_dict = dict(zip(variable_names, [[] for l in variable_names]))
        B_conditioned_on_A_entropy_dict = dict(zip(variable_names, [[] for l in variable_names]))
        mutual_info_trace_dict = dict(zip(variable_names, [[] for l in variable_names]))
        every_probability_dict = dict(zip(variable_names, [[] for l in variable_names]))

        for variable in variable_names:
            for key in keys:
                A_conditioned_on_B_entropy, B_conditioned_on_A_entropy, mutual_info_trace, every_probability = get_entropies(A_centered=A_dict[key][variable],
                                                                                                          B_centered=B_dict[key][variable])
                A_conditioned_on_B_entropy_dict[variable].append(A_conditioned_on_B_entropy)
                B_conditioned_on_A_entropy_dict[variable].append(A_conditioned_on_B_entropy)
                mutual_info_trace_dict[variable].append(mutual_info_trace)
                every_probability_dict[variable].append(every_probability)

        return A_conditioned_on_B_entropy_dict, B_conditioned_on_A_entropy_dict, mutual_info_trace_dict, every_probability_dict
    # A/B_S/N/C can be global, trap or traj centered dictionaries

    if seed != None:
        np.random.seed(seed)

    # we do this because sis has 132 instead of 88...
    sis_keys = random.choices(list(A_sis.keys()), k=88)

    sis_A_conditioned_on_B_entropy_dict, sis_B_conditioned_on_A_entropy_dict, sis_mutual_info_trace_dict, sis_every_probability_dict = \
        for_all_traps(keys=sis_keys, variable_names=variable_names, A_dict=A_sis, B_dict=B_sis)
    non_sis_A_conditioned_on_B_entropy_dict, non_sis_B_conditioned_on_A_entropy_dict, non_sis_mutual_info_trace_dict, non_sis_every_probability_dict = \
        for_all_traps(keys=A_non_sis.keys(), variable_names=variable_names, A_dict=A_non_sis, B_dict=B_non_sis)
    con_A_conditioned_on_B_entropy_dict, con_B_conditioned_on_A_entropy_dict, con_mutual_info_trace_dict, con_every_probability_dict = \
        for_all_traps(keys=A_con.keys(), variable_names=variable_names, A_dict=A_con, B_dict=B_con)

    for var in variable_names:
        if var in graph_vars:
            sns.kdeplot(sis_mutual_info_trace_dict[var], label='Sister')
            sns.kdeplot(non_sis_mutual_info_trace_dict[var], label='Nonsister')
            sns.kdeplot(con_mutual_info_trace_dict[var], label='Control')
            plt.title(var)
            plt.xlabel('Mutual Information of A and B')
            plt.legend()
            plt.show()
            plt.close()

            # sns.kdeplot(sis_A_conditioned_on_B_entropy_dict[var], label='sis conditional A,B')
            # sns.kdeplot(non_sis_A_conditioned_on_B_entropy_dict[var], label='nonsis conditional A,B')
            # sns.kdeplot(con_A_conditioned_on_B_entropy_dict[var], label='control conditional A,B')
            # plt.title(var)
            # plt.legend()
            # plt.show()
            # plt.close()
            # 
            # sns.kdeplot(sis_B_conditioned_on_A_entropy_dict[var], label='sis conditional B,A')
            # sns.kdeplot(non_sis_B_conditioned_on_A_entropy_dict[var], label='nonsis conditional B,A')
            # sns.kdeplot(con_B_conditioned_on_A_entropy_dict[var], label='control conditional B,A')
            # plt.title(var)
            # plt.legend()
            # plt.show()
            # plt.close()

    return [sis_A_conditioned_on_B_entropy_dict, sis_B_conditioned_on_A_entropy_dict, sis_mutual_info_trace_dict, sis_every_probability_dict], \
           [non_sis_A_conditioned_on_B_entropy_dict, non_sis_B_conditioned_on_A_entropy_dict, non_sis_mutual_info_trace_dict, non_sis_every_probability_dict], \
           [con_A_conditioned_on_B_entropy_dict, con_B_conditioned_on_A_entropy_dict, con_mutual_info_trace_dict, con_every_probability_dict]


def save_the_graphs(sis_entropies, non_sis_entropies, con_entropies, variable_names, xlabel):
    for var in variable_names:
        # The histogram of the traps
        sns.kdeplot(sis_entropies[2][var], label='Sister')
        sns.kdeplot(non_sis_entropies[2][var], label='Nonsister')
        sns.kdeplot(con_entropies[2][var], label='Control')
        plt.axvline(np.mean(sis_entropies[2][var]), linestyle='--', color='blue')
        plt.axvline(np.mean(non_sis_entropies[2][var]), linestyle='--', color='orange')
        plt.axvline(np.mean(con_entropies[2][var]), linestyle='--', color='green')
        plt.title('Mutual Information of {} of the A and B traces'.format(var))
        plt.xlabel(xlabel)
        plt.ylabel('PDF')
        plt.legend()
        plt.savefig('MI, {}, {}'.format(xlabel, var), dpi=300)
        plt.close()

        # the histogram of the correct joint probabilities



def get_the_bin_count_for_pop_gentime(Population):
    mom = Population.mother_dfs[0].copy()
    daug = Population.daughter_dfs[0].copy()

    mom['generationtime'] = my_round(mom['generationtime'])
    daug['generationtime'] = my_round(daug['generationtime'])

    print(mom['generationtime'].min())
    print(mom['generationtime'].max())
    print('number of bins between the min and max of the population:', (mom['generationtime'].max() - mom['generationtime'].min())/.05)
    print(daug['generationtime'].min())
    print(daug['generationtime'].max())
    print('number of bins between the min and max of the population:', (daug['generationtime'].max() - daug['generationtime'].min()) / .05)

    print(len(mom['generationtime']), len(daug['generationtime']), len(pd.concat([mom['generationtime'], daug['generationtime']])))
    print(len(np.unique(pd.concat([mom['generationtime'], daug['generationtime']]))))
    print(np.unique(pd.concat([mom['generationtime'], daug['generationtime']])))

    return np.unique(pd.concat([mom['generationtime'], daug['generationtime']]))


def myround(x, prec=2, base=.05):
  return round(base * round(float(x)/base),prec)


def my_round(x, prec=2, base=0.05):
    return (base * (np.array(x) / base).round()).round(prec)


def trap_hists(A_sis, B_sis, A_non_sis, B_non_sis, A_con, B_con, variable_names, graph_vars, n_bins, seed, strategy):

    def for_all_traps(keys, variable_names, A_dict, B_dict, n_bins=n_bins, strategy=strategy):  # this pools together all traps for one dataset

        def get_entropies(A_centered, B_centered, n_bins=n_bins, strategy=strategy):  # this gives the conditional entropies and mutual information for two centered vectors

            A_centered = A_centered.iloc[:min(len(A_centered), len(B_centered))]
            B_centered = B_centered.iloc[:min(len(A_centered), len(B_centered))]
            # joint_centered = pd.DataFrame({'A': A_centered, 'B': B_centered})

            # print(len(A_centered), n_bins)

            # divide the trap into n_bins for the joint distribution and the trace distributions respectively
            est = sklearn.preprocessing.KBinsDiscretizer(n_bins=n_bins, strategy=strategy)
            est.fit(np.array(pd.concat([A_centered, B_centered], axis=0).reset_index(drop=True)).reshape(-1, 1))
            est_A = sklearn.preprocessing.KBinsDiscretizer(n_bins=n_bins, strategy=strategy)
            est_B = sklearn.preprocessing.KBinsDiscretizer(n_bins=n_bins, strategy=strategy)
            est_A.fit(np.array(A_centered).reshape(-1, 1))
            est_B.fit(np.array(B_centered).reshape(-1, 1))

            # check that the array of unique bin edges are the same length as the number of bins, ie. there are no duplicates
            if len(np.unique(est_A.bin_edges_[0])) != n_bins+1:
                print('There was a bin edge duplicate for A!')
                if len(np.unique(est_B.bin_edges_[0])) != n_bins+1:
                    print('There was a bin edge duplicate for B!')
                print('_________')

                return None, None, None, []

            if len(np.unique(est_B.bin_edges_[0])) != n_bins+1:
                print('There was a bin edge duplicate for B!\n_________')
                return None, None, None, []

            # # sanity check
            # if est.n_bins_[0] != n_bins+1:
            #     print('wrong number of bins!')
            #     print(est.n_bins_[0], type(est.n_bins_[0]))
            #     print(n_bins, type(n_bins))
            #     exit()

            # based on the bin edges, give each entry in the A and B series an integer that maps to the bin in the marginal distribution
            A_cut = pd.cut(A_centered, est_A.bin_edges_[0], right=True, include_lowest=True, labels=np.arange(n_bins))  # , duplicates='drop'
            B_cut = pd.cut(B_centered, est_B.bin_edges_[0], right=True, include_lowest=True, labels=np.arange(n_bins))
            joint_centered = pd.DataFrame({'A': A_cut, 'B': B_cut})

            joint_prob_list = dict([('{}_{}'.format(label_A, label_B), len(joint_centered[(joint_centered['A'] == label_A) & (joint_centered['B'] == label_B)]) /
                                         len(joint_centered)) for label_B in range(n_bins) for label_A in range(n_bins)])
            A_trace_marginal_probs = dict([('{}'.format(label_A), len(A_cut.iloc[np.where(A_cut == label_A)]) / len(A_cut))
                                                for label_A in range(n_bins)])
            B_trace_marginal_probs = dict([('{}'.format(label_B), len(B_cut.iloc[np.where(B_cut == label_B)]) / len(B_cut))
                                                for label_B in range(n_bins)])

            # conditioning the A trace based on the B trace
            A_conditioned_on_B_entropy = np.array([- joint_prob_list[key] * np.log(joint_prob_list[key] / A_trace_marginal_probs[key.split('_')[0]])
                                          for key in joint_prob_list.keys() if joint_prob_list[key] != 0 and
                                                   A_trace_marginal_probs[key.split('_')[0]] != 0]).sum()
            B_conditioned_on_A_entropy = np.array([- joint_prob_list[key] * np.log(joint_prob_list[key] / B_trace_marginal_probs[key.split('_')[1]])
                                          for key in joint_prob_list.keys() if joint_prob_list[key] != 0 and
                                                   B_trace_marginal_probs[key.split('_')[0]] != 0]).sum()

            # the mutual information between A and B trace for this variable thinking that marginal A and B came from each trace distribution
            mutual_info_trace = np.array([joint_prob_list[key] * np.log(joint_prob_list[key] / (A_trace_marginal_probs[key.split('_')[0]] *
                                                                                                B_trace_marginal_probs[key.split('_')[1]]))
                                          for key in joint_prob_list.keys() if joint_prob_list[key] != 0 and
                                          B_trace_marginal_probs[key.split('_')[1]] != 0 and
                                          A_trace_marginal_probs[key.split('_')[0]] != 0]).sum()

            # print(joint_prob_list)
            # print(A_trace_marginal_probs)
            # print(B_trace_marginal_probs)
            # print(A_conditioned_on_B_entropy)
            # print(B_conditioned_on_A_entropy)
            # print(mutual_info_trace)
            # exit()

            # print(joint_centered)
            # print('++:\n', joint_centered[(joint_centered['A'] >= 0) & (joint_centered['B'] >= 0)])
            # print('--:\n', joint_centered[(joint_centered['A'] < 0) & (joint_centered['B'] < 0)])
            # print('+-:\n', joint_centered[(joint_centered['A'] >= 0) & (joint_centered['B'] < 0)])
            # print('-+:\n', joint_centered[(joint_centered['A'] < 0) & (joint_centered['B'] >= 0)])
            # input()
            #
            # print(plus_plus_joint)
            # print(minus_minus_joint)
            # print(plus_minus_joint)
            # print(minus_plus_joint)
            # print('1=', plus_plus_joint+plus_minus_joint+minus_plus_joint+minus_minus_joint)
            # print('______')
            # print(A_minus_marginal)
            # print(A_plus_marginal)
            # print('1=', A_minus_marginal+A_plus_marginal)
            # print(B_minus_marginal)
            # print(B_plus_marginal)
            # print('1=', B_minus_marginal + B_plus_marginal)
            # exit()

            return A_conditioned_on_B_entropy, B_conditioned_on_A_entropy, mutual_info_trace, joint_prob_list

        # the dictionary where the keys are the variables and the values are the entropies of all the traps in the dataset
        A_conditioned_on_B_entropy_dict = dict(zip(variable_names, [[] for l in variable_names]))
        B_conditioned_on_A_entropy_dict = dict(zip(variable_names, [[] for l in variable_names]))
        mutual_info_trace_dict = dict(zip(variable_names, [[] for l in variable_names]))
        every_probability_dict = dict(zip(variable_names, [[] for l in variable_names]))

        for variable in variable_names:
            for key in keys:
                A_conditioned_on_B_entropy, B_conditioned_on_A_entropy, mutual_info_trace, joint_probs = get_entropies(
                    A_centered=A_dict[key][variable],
                    B_centered=B_dict[key][variable])
                A_conditioned_on_B_entropy_dict[variable].append(A_conditioned_on_B_entropy)
                B_conditioned_on_A_entropy_dict[variable].append(A_conditioned_on_B_entropy)
                mutual_info_trace_dict[variable].append(mutual_info_trace)
                every_probability_dict[variable].append(joint_probs)

        return A_conditioned_on_B_entropy_dict, B_conditioned_on_A_entropy_dict, mutual_info_trace_dict, every_probability_dict

    # A/B_S/N/C can be global, trap or traj centered dictionaries

    if seed != None:
        np.random.seed(seed)

    # we do this because sis has 132 instead of 88...
    sis_keys = random.choices(list(A_sis.keys()), k=88)

    sis_A_conditioned_on_B_entropy_dict, sis_B_conditioned_on_A_entropy_dict, sis_mutual_info_trace_dict, sis_every_probability_dict = \
        for_all_traps(keys=sis_keys, variable_names=variable_names, A_dict=A_sis, B_dict=B_sis)
    non_sis_A_conditioned_on_B_entropy_dict, non_sis_B_conditioned_on_A_entropy_dict, non_sis_mutual_info_trace_dict, non_sis_every_probability_dict = \
        for_all_traps(keys=A_non_sis.keys(), variable_names=variable_names, A_dict=A_non_sis, B_dict=B_non_sis)
    con_A_conditioned_on_B_entropy_dict, con_B_conditioned_on_A_entropy_dict, con_mutual_info_trace_dict, con_every_probability_dict = \
        for_all_traps(keys=A_con.keys(), variable_names=variable_names, A_dict=A_con, B_dict=B_con)

    print('sis MI #:', len(sis_mutual_info_trace_dict['generationtime']))
    print('non sis MI #:', len(non_sis_mutual_info_trace_dict['generationtime']))
    print('control MI #:', len(con_mutual_info_trace_dict['generationtime']))

    for var in variable_names:
        if var in graph_vars:
            sns.kdeplot(sis_mutual_info_trace_dict[var], label='sis MI')
            sns.kdeplot(non_sis_mutual_info_trace_dict[var], label='nonsis MI')
            sns.kdeplot(con_mutual_info_trace_dict[var], label='control MI')
            plt.title(var)
            plt.legend()
            plt.show()
            plt.close()

            # sns.kdeplot(sis_A_conditioned_on_B_entropy_dict[var], label='sis conditional A,B')
            # sns.kdeplot(non_sis_A_conditioned_on_B_entropy_dict[var], label='nonsis conditional A,B')
            # sns.kdeplot(con_A_conditioned_on_B_entropy_dict[var], label='control conditional A,B')
            # plt.title(var)
            # plt.legend()
            # plt.show()
            # plt.close()
            #
            # sns.kdeplot(sis_B_conditioned_on_A_entropy_dict[var], label='sis conditional B,A')
            # sns.kdeplot(non_sis_B_conditioned_on_A_entropy_dict[var], label='nonsis conditional B,A')
            # sns.kdeplot(con_B_conditioned_on_A_entropy_dict[var], label='control conditional B,A')
            # plt.title(var)
            # plt.legend()
            # plt.show()
            # plt.close()

    return [sis_A_conditioned_on_B_entropy_dict, sis_B_conditioned_on_A_entropy_dict, sis_mutual_info_trace_dict, sis_every_probability_dict], \
           [non_sis_A_conditioned_on_B_entropy_dict, non_sis_B_conditioned_on_A_entropy_dict, non_sis_mutual_info_trace_dict,
            non_sis_every_probability_dict], \
           [con_A_conditioned_on_B_entropy_dict, con_B_conditioned_on_A_entropy_dict, con_mutual_info_trace_dict, con_every_probability_dict]


# here you can graph the mutual information and change the centered dictionaries (global, trap, traj, etc..)
def for_all_datasets_1_std_distinction(A_sis, B_sis, A_non_sis, B_non_sis, A_con, B_con, variable_names, graph_vars, seed, bins_on_side):
    def for_all_traps(keys, variable_names, A_dict, B_dict, bins_on_side=bins_on_side):  # this pools together all traps for one dataset

        def get_entropies(A_centered, B_centered, bins_on_side=bins_on_side):  # this gives the conditional entropies and mutual information for two centered vectors

            A_centered = A_centered.iloc[:min(len(A_centered), len(B_centered))]
            B_centered = B_centered.iloc[:min(len(A_centered), len(B_centered))]
            joint_centered = pd.DataFrame({'A': A_centered, 'B': B_centered})
            
            # A_edges = [min(A_centered), -np.std(A_centered), 0, np.std(A_centered), max(A_centered)]
            # B_edges = [min(B_centered), -np.std(B_centered), 0, np.std(B_centered), max(B_centered)]
            # A_edges = [min(A_centered),  - np.mean(A_centered) - np.std(A_centered), np.mean(A_centered), np.mean(A_centered) + np.std(A_centered), max(A_centered)]
            # B_edges = [min(B_centered), - np.mean(B_centered) - np.std(B_centered), np.mean(B_centered), np.mean(B_centered) + np.std(B_centered),
            #            max(B_centered)]

            A_edges = [-(((bins_on_side-interval)/bins_on_side) * (np.mean(A_centered)-min(A_centered))) + np.mean(A_centered) for interval in range(bins_on_side)] + \
                      [((interval/bins_on_side) * (max(A_centered)-np.mean(A_centered))) + np.mean(A_centered) for interval in range(bins_on_side+1)]

            B_edges = [-(((bins_on_side-interval) / bins_on_side) * (np.mean(B_centered) - min(B_centered))) + np.mean(B_centered) for interval in
                       range(bins_on_side)] + \
                      [((interval / bins_on_side) * (max(B_centered) - np.mean(B_centered))) + np.mean(B_centered) for interval in
                       range(bins_on_side+1)]

            # A_edges = [min(A_centered), (min(A_centered) + np.mean(A_centered)) * (2 / 3), (min(A_centered) + np.mean(A_centered)) / 3,
            #            np.mean(A_centered), (max(A_centered) + np.mean(A_centered)) / 3, (max(A_centered) + np.mean(A_centered)) * (2 / 3),
            #            max(A_centered)]
            # B_edges = [min(B_centered), 2 * (min(B_centered) + np.mean(B_centered)) / 3, (min(B_centered) + np.mean(B_centered)) / 3,
            #            np.mean(B_centered), (max(B_centered) + np.mean(B_centered)) / 3, 2 * (max(B_centered) + np.mean(B_centered)) / 3,
            #            max(B_centered)]

            # print('A_edges:', A_edges)
            # print('B_edges:', B_edges)
            # print('_________')
            
            for ind in range(len(A_edges)-1):
                if A_edges[ind] >= A_edges[ind+1]:
                    print('A_edges:', A_edges)
                    print(A_centered)
            for ind in range(len(B_edges)-1):
                if B_edges[ind] >= B_edges[ind+1]:
                    print('B_edges:', B_edges)
                    print(B_centered)

            # based on the bin edges, give each entry in the A and B series an integer that maps to the bin in the marginal distribution
            A_cut = pd.cut(A_centered, A_edges, right=True, include_lowest=True, labels=np.arange(len(A_edges)-1))  # , duplicates='drop'
            B_cut = pd.cut(B_centered, B_edges, right=True, include_lowest=True, labels=np.arange(len(B_edges)-1))
            joint_centered = pd.DataFrame({'A': A_cut, 'B': B_cut})

            joint_prob_list = dict(
                [('{}_{}'.format(label_A, label_B), len(joint_centered[(joint_centered['A'] == label_A) & (joint_centered['B'] == label_B)]) /
                  len(joint_centered)) for label_B in np.arange(len(B_edges)-1) for label_A in np.arange(len(A_edges)-1)])
            A_trace_marginal_probs = dict([('{}'.format(label_A), len(A_cut.iloc[np.where(A_cut == label_A)]) / len(A_cut))
                                           for label_A in np.arange(len(A_edges)-1)])
            B_trace_marginal_probs = dict([('{}'.format(label_B), len(B_cut.iloc[np.where(B_cut == label_B)]) / len(B_cut))
                                           for label_B in np.arange(len(B_edges)-1)])

            # conditioning the A trace based on the B trace
            A_conditioned_on_B_entropy = np.array([- joint_prob_list[key] * np.log(joint_prob_list[key] / A_trace_marginal_probs[key.split('_')[0]])
                                                   for key in joint_prob_list.keys() if joint_prob_list[key] != 0 and
                                                   A_trace_marginal_probs[key.split('_')[0]] != 0]).sum()
            B_conditioned_on_A_entropy = np.array([- joint_prob_list[key] * np.log(joint_prob_list[key] / B_trace_marginal_probs[key.split('_')[1]])
                                                   for key in joint_prob_list.keys() if joint_prob_list[key] != 0 and
                                                   B_trace_marginal_probs[key.split('_')[1]] != 0]).sum()

            # the mutual information between A and B trace for this variable thinking that marginal A and B came from each trace distribution
            mutual_info_trace = np.array([joint_prob_list[key] * np.log(joint_prob_list[key] / (A_trace_marginal_probs[key.split('_')[0]] *
                                                                                                B_trace_marginal_probs[key.split('_')[1]]))
                                          for key in joint_prob_list.keys() if joint_prob_list[key] != 0 and
                                          B_trace_marginal_probs[key.split('_')[1]] != 0 and
                                          A_trace_marginal_probs[key.split('_')[0]] != 0]).sum()

            # print('A_cut:', A_cut)
            # print('B_cut:', B_cut)
            # print('A_edges:', A_edges)
            # print('B_edges:', B_edges)
            # print('joint_prob_list:', joint_prob_list, np.sum([val for val in joint_prob_list.values()]))
            # print('A_trace_marginal_probs:', A_trace_marginal_probs, np.sum([val for val in A_trace_marginal_probs.values()]))
            # print('B_trace_marginal_probs:', B_trace_marginal_probs, np.sum([val for val in B_trace_marginal_probs.values()]))
            
            if mutual_info_trace < 0:
                print('mutual info is negative! something is wrong...')
                print('_________')
                exit()

            # print('_________')

            # print(joint_prob_list)
            # print(A_trace_marginal_probs)
            # print(B_trace_marginal_probs)
            # print(A_conditioned_on_B_entropy)
            # print(B_conditioned_on_A_entropy)
            # print(mutual_info_trace)
            # exit()

            """
            # joint probabilities
            plus_plus_joint = len(joint_centered[(joint_centered['A'] >= 0) & (joint_centered['B'] >= 0)]) / len(joint_centered)
            plus_minus_joint = len(joint_centered[(joint_centered['A'] >= 0) & (joint_centered['B'] < 0)]) / len(joint_centered)
            minus_plus_joint = len(joint_centered[(joint_centered['A'] < 0) & (joint_centered['B'] >= 0)]) / len(joint_centered)
            minus_minus_joint = len(joint_centered[(joint_centered['A'] < 0) & (joint_centered['B'] < 0)]) / len(joint_centered)

            # marginal probabilites
            A_minus_marginal = len(A_centered.iloc[np.where(A_centered < 0)]) / len(A_centered)
            A_plus_marginal = len(A_centered.iloc[np.where(A_centered >= 0)]) / len(A_centered)
            B_minus_marginal = len(B_centered.iloc[np.where(B_centered < 0)]) / len(B_centered)
            B_plus_marginal = len(A_centered.iloc[np.where(B_centered >= 0)]) / len(B_centered)

            # to make the calculation easier
            joint_prob_list = [plus_plus_joint, minus_minus_joint, plus_minus_joint, minus_plus_joint]
            A_trace_marginal_probs = [A_plus_marginal, A_minus_marginal, A_minus_marginal, A_plus_marginal]
            B_trace_marginal_probs = [B_plus_marginal, B_minus_marginal, B_plus_marginal, B_minus_marginal]

            every_probability = [plus_plus_joint, minus_minus_joint, plus_minus_joint, minus_plus_joint, A_plus_marginal, A_minus_marginal,
                                 B_plus_marginal, B_minus_marginal]

            # conditioning the A trace based on the B trace
            A_conditioned_on_B_entropy = np.array(
                [- joint * np.log(joint / marg) for joint, marg in zip(joint_prob_list, A_trace_marginal_probs) if joint != 0 and marg != 0]).sum()
            B_conditioned_on_A_entropy = np.array(
                [- joint * np.log(joint / marg) for joint, marg in zip(joint_prob_list, B_trace_marginal_probs) if joint != 0 and marg != 0]).sum()

            # the mutual information between A and B trace for this variable thinking that marginal A and B came from each trace distribution
            mutual_info_trace = np.array(
                [joint * np.log(joint / (marg_A * marg_B)) for joint, marg_A, marg_B in
                 zip(joint_prob_list, A_trace_marginal_probs, B_trace_marginal_probs)
                 if
                 joint != 0 and marg_A != 0 and marg_B != 0]).sum()
            """

            # print(joint_centered)
            # print('++:\n', joint_centered[(joint_centered['A'] >= 0) & (joint_centered['B'] >= 0)])
            # print('--:\n', joint_centered[(joint_centered['A'] < 0) & (joint_centered['B'] < 0)])
            # print('+-:\n', joint_centered[(joint_centered['A'] >= 0) & (joint_centered['B'] < 0)])
            # print('-+:\n', joint_centered[(joint_centered['A'] < 0) & (joint_centered['B'] >= 0)])
            # input()
            #
            # print(plus_plus_joint)
            # print(minus_minus_joint)
            # print(plus_minus_joint)
            # print(minus_plus_joint)
            # print('1=', plus_plus_joint+plus_minus_joint+minus_plus_joint+minus_minus_joint)
            # print('______')
            # print(A_minus_marginal)
            # print(A_plus_marginal)
            # print('1=', A_minus_marginal+A_plus_marginal)
            # print(B_minus_marginal)
            # print(B_plus_marginal)
            # print('1=', B_minus_marginal + B_plus_marginal)
            # exit()

            return A_conditioned_on_B_entropy, B_conditioned_on_A_entropy, mutual_info_trace, joint_prob_list

        # the dictionary where the keys are the variables and the values are the entropies of all the traps in the dataset
        A_conditioned_on_B_entropy_dict = dict(zip(variable_names, [[] for l in variable_names]))
        B_conditioned_on_A_entropy_dict = dict(zip(variable_names, [[] for l in variable_names]))
        mutual_info_trace_dict = dict(zip(variable_names, [[] for l in variable_names]))
        every_probability_dict = dict(zip(variable_names, [[] for l in variable_names]))

        for variable in variable_names:
            for key in keys:
                A_conditioned_on_B_entropy, B_conditioned_on_A_entropy, mutual_info_trace, every_probability = get_entropies(
                    A_centered=A_dict[key][variable],
                    B_centered=B_dict[key][variable])
                A_conditioned_on_B_entropy_dict[variable].append(A_conditioned_on_B_entropy)
                B_conditioned_on_A_entropy_dict[variable].append(A_conditioned_on_B_entropy)
                mutual_info_trace_dict[variable].append(mutual_info_trace)
                every_probability_dict[variable].append(every_probability)

        return A_conditioned_on_B_entropy_dict, B_conditioned_on_A_entropy_dict, mutual_info_trace_dict, every_probability_dict

    # A/B_S/N/C can be global, trap or traj centered dictionaries

    if seed != None:
        np.random.seed(seed)

    # we do this because sis has 132 instead of 88...
    sis_keys = np.random.choice(list(A_sis.keys()), size=88, replace=False)

    sis_A_conditioned_on_B_entropy_dict, sis_B_conditioned_on_A_entropy_dict, sis_mutual_info_trace_dict, sis_every_probability_dict = \
        for_all_traps(keys=sis_keys, variable_names=variable_names, A_dict=A_sis, B_dict=B_sis)
    print('sis finished')
    non_sis_A_conditioned_on_B_entropy_dict, non_sis_B_conditioned_on_A_entropy_dict, non_sis_mutual_info_trace_dict, non_sis_every_probability_dict = \
        for_all_traps(keys=A_non_sis.keys(), variable_names=variable_names, A_dict=A_non_sis, B_dict=B_non_sis)
    print('non sis finished')
    con_A_conditioned_on_B_entropy_dict, con_B_conditioned_on_A_entropy_dict, con_mutual_info_trace_dict, con_every_probability_dict = \
        for_all_traps(keys=A_con.keys(), variable_names=variable_names, A_dict=A_con, B_dict=B_con)
    print('con finished')

    for var in variable_names:
        if var in graph_vars:
            sns.kdeplot(sis_mutual_info_trace_dict[var], label='Sister')
            sns.kdeplot(non_sis_mutual_info_trace_dict[var], label='Nonsister')
            sns.kdeplot(con_mutual_info_trace_dict[var], label='Control')
            plt.axvline(np.mean(sis_mutual_info_trace_dict[var]), linestyle='--', color='blue')
            plt.axvline(np.mean(non_sis_mutual_info_trace_dict[var]), linestyle='--', color='orange')
            plt.axvline(np.mean(con_mutual_info_trace_dict[var]), linestyle='--', color='green')
            plt.title(var)
            plt.xlabel('Mutual Information of A and B')
            plt.legend()
            plt.show()
            plt.close()

            # sns.kdeplot(sis_A_conditioned_on_B_entropy_dict[var], label='sis conditional A,B')
            # sns.kdeplot(non_sis_A_conditioned_on_B_entropy_dict[var], label='nonsis conditional A,B')
            # sns.kdeplot(con_A_conditioned_on_B_entropy_dict[var], label='control conditional A,B')
            # plt.title(var)
            # plt.legend()
            # plt.show()
            # plt.close()
            #
            # sns.kdeplot(sis_B_conditioned_on_A_entropy_dict[var], label='sis conditional B,A')
            # sns.kdeplot(non_sis_B_conditioned_on_A_entropy_dict[var], label='nonsis conditional B,A')
            # sns.kdeplot(con_B_conditioned_on_A_entropy_dict[var], label='control conditional B,A')
            # plt.title(var)
            # plt.legend()
            # plt.show()
            # plt.close()

    return [sis_A_conditioned_on_B_entropy_dict, sis_B_conditioned_on_A_entropy_dict, sis_mutual_info_trace_dict, sis_every_probability_dict], \
           [non_sis_A_conditioned_on_B_entropy_dict, non_sis_B_conditioned_on_A_entropy_dict, non_sis_mutual_info_trace_dict,
            non_sis_every_probability_dict], \
           [con_A_conditioned_on_B_entropy_dict, con_B_conditioned_on_A_entropy_dict, con_mutual_info_trace_dict, con_every_probability_dict]


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

    pickle_in = open("NewSisterCellClass_Env_Sister.pickle", "rb")
    Env_Sister = pickle.load(pickle_in)
    pickle_in.close()

    pickle_in = open("NewSisterCellClass_Env_Nonsister.pickle", "rb")
    Env_Nonsister = pickle.load(pickle_in)
    pickle_in.close()

    # mom = Population.mother_dfs[0].copy()
    # daug = Population.daughter_dfs[0].copy()
    #
    # mom['generationtime'] = my_round(mom['generationtime'])
    # daug['generationtime'] = my_round(daug['generationtime'])
    # gen_gen = np.unique(pd.concat([mom['generationtime'], daug['generationtime']])).reshape(1, -1)
    # growth = np.unique(pd.concat([mom['growth_rate'], daug['growth_rate']])).reshape(-1, 1)
    # print(growth)
    # # get_the_bin_count_for_pop_gentime(Population)
    # est = sklearn.preprocessing.KBinsDiscretizer(n_bins=34)
    # est.fit(growth)
    # print(est.bin_edges_)
    # print(est.n_bins_)
    #
    # exit()
    #
    #
    #
    # for col in Population._variable_names:
    #     sns.distplot(mom[col], bins=100)
    #     plt.title(col)
    #     plt.show()
    #     plt.close()
    # exit()

    # renaming
    sis_A, sis_B = Sister.A_dict.copy(), Sister.B_dict.copy()
    non_sis_A, non_sis_B = Nonsister.A_dict.copy(), Nonsister.B_dict.copy()
    con_A, con_B = Control.A_dict.copy(), Control.B_dict.copy()
    con_ref_A, con_ref_B = Control.reference_A_dict.copy(), Control.reference_B_dict.copy()

    # the population mean
    pop_mean = Population.mother_dfs[0].copy().mean()
    
    # subtracting the global averages
    sis_A_global = {key: subtract_global_averages(df=val, columns_names=val.columns, pop_mean=pop_mean) for key, val in sis_A.items()}
    sis_B_global = {key: subtract_global_averages(df=val, columns_names=val.columns, pop_mean=pop_mean) for key, val in sis_B.items()}
    non_sis_A_global = {key: subtract_global_averages(df=val, columns_names=val.columns, pop_mean=pop_mean) for key, val in non_sis_A.items()}
    non_sis_B_global = {key: subtract_global_averages(df=val, columns_names=val.columns, pop_mean=pop_mean) for key, val in non_sis_B.items()}
    con_A_global = {key: subtract_global_averages(df=val, columns_names=val.columns, pop_mean=pop_mean) for key, val in con_A.items()}
    con_B_global = {key: subtract_global_averages(df=val, columns_names=val.columns, pop_mean=pop_mean) for key, val in con_B.items()}
    

    # subtracting the trap averages
    sis_A_trap = {key: subtract_trap_averages(df=val, columns_names=val.columns, trap_mean=trap_mean.loc['mean']) for key, val, trap_mean in zip(sis_A.keys(), sis_A.values(), Sister.trap_stats_dict.values())}
    sis_B_trap = {key: subtract_trap_averages(df=val, columns_names=val.columns, trap_mean=trap_mean.loc['mean']) for key, val, trap_mean in zip(sis_B.keys(), sis_B.values(), Sister.trap_stats_dict.values())}
    non_sis_A_trap = {key: subtract_trap_averages(df=val, columns_names=val.columns, trap_mean=trap_mean.loc['mean']) for key, val, trap_mean in
                  zip(non_sis_A.keys(), non_sis_A.values(), Nonsister.trap_stats_dict.values())}
    non_sis_B_trap = {key: subtract_trap_averages(df=val, columns_names=val.columns, trap_mean=trap_mean.loc['mean']) for key, val, trap_mean in
                  zip(non_sis_B.keys(), non_sis_B.values(), Nonsister.trap_stats_dict.values())}
    con_A_trap = {key: subtract_trap_averages_control(val, val.columns, ref_key, sis_A, sis_B, non_sis_A, non_sis_B) for key, val, ref_key in
                  zip(con_A.keys(), con_A.values(), con_ref_A.values())}
    con_B_trap = {key: subtract_trap_averages_control(val, val.columns, ref_key, sis_A, sis_B, non_sis_A, non_sis_B) for key, val, ref_key in
                  zip(con_B.keys(), con_B.values(), con_ref_B.values())}

    # subtracting the trajectory averages
    sis_A_traj = {key: subtract_traj_averages(df=val, columns_names=val.columns) for key, val in zip(sis_A.keys(), sis_A.values())}
    sis_B_traj = {key: subtract_traj_averages(df=val, columns_names=val.columns) for key, val in zip(sis_B.keys(), sis_B.values())}
    non_sis_A_traj = {key: subtract_traj_averages(df=val, columns_names=val.columns) for key, val in zip(non_sis_A.keys(), non_sis_A.values())}
    non_sis_B_traj = {key: subtract_traj_averages(df=val, columns_names=val.columns) for key, val in zip(non_sis_B.keys(), non_sis_B.values())}
    con_A_traj = {key: subtract_traj_averages(df=val, columns_names=val.columns) for key, val in zip(con_A.keys(), con_A.values())}
    con_B_traj = {key: subtract_traj_averages(df=val, columns_names=val.columns) for key, val in zip(con_B.keys(), con_B.values())}

    # for strategy in ['quantile', 'uniform']:
    #     if strategy == 'quantile':

    type_of_mean = 'global' # ['global', 'trap']
    bin_num = 3 # [2, 3, 4, 5, 6, 7, 8] for uniform it can be however many you want but for quantile its only up to 3 sadly, try to make something to circumvent this
    strategy = 'quantile' # ['quantile', 'uniform']

    # sis_entropies_global_hist, non_sis_entropies_global_hist, con_entropies_global_hist = trap_hists(A_sis=sis_A_global, B_sis=sis_B_global,
    #                                                                                   A_non_sis=non_sis_A_global,
    #                                                                                   B_non_sis=non_sis_B_global, A_con=con_A_global,
    #                                                                                   B_con=con_B_global,
    #                                                                                   variable_names=Population._variable_names,
    #                                                                                   graph_vars=Population._variable_names, n_bins=bin_num,
    #                                                                                   seed=42, strategy=strategy)
    #
    # xlabel = '{} bins, {} strategy, {} centered'.format(bin_num, strategy, type_of_mean)
    # save_the_graphs(sis_entropies=sis_entropies_global_hist, non_sis_entropies=non_sis_entropies_global_hist, con_entropies=con_entropies_global_hist,
    #                 variable_names=Population._variable_names, xlabel=xlabel)

    # sis_entropies_traj, non_sis_entropies_traj, con_entropies_traj = for_all_datasets_1_std_distinction(A_sis=sis_A_traj, B_sis=sis_B_traj,
    #                                                                                                     A_non_sis=non_sis_A_traj,
    #                                                                                                     B_non_sis=non_sis_B_traj, A_con=con_A_traj,
    #                                                                                                     B_con=con_B_traj,
    #                                                                                                     variable_names=Population._variable_names,
    #                                                                                                     graph_vars=[],
    #                                                                                                     seed=42,
    #                                                                                                     bins_on_side=2)
    #
    # sis_entropies_traj, non_sis_entropies_traj, con_entropies_traj = for_all_datasets_1_std_distinction(A_sis=sis_A, B_sis=sis_B,
    #                                                                                                     A_non_sis=non_sis_A,
    #                                                                                                     B_non_sis=non_sis_B, A_con=con_A,
    #                                                                                                     B_con=con_B,
    #                                                                                                     variable_names=Population._variable_names,
    #                                                                                                     graph_vars=[],
    #                                                                                                     seed=42,
    #                                                                                                     bins_on_side=2)
    #
    # exit()

    for bins_on_side in range(1, 12):

        sis_entropies_traj, non_sis_entropies_traj, con_entropies_traj = for_all_datasets_1_std_distinction(A_sis=sis_A, B_sis=sis_B,
                                                                                                            A_non_sis=non_sis_A,
                                                                                                            B_non_sis=non_sis_B, A_con=con_A,
                                                                                                            B_con=con_B,
                                                                                                            variable_names=Population._variable_names,
                                                                                                            graph_vars=[],
                                                                                                            seed=42,
                                                                                                            bins_on_side=bins_on_side)

        xlabel = '{} bins on side, not centered but mean taken into account'.format(bins_on_side)
        save_the_graphs(sis_entropies=sis_entropies_traj, non_sis_entropies=non_sis_entropies_traj, con_entropies=con_entropies_traj,
                        variable_names=Population._variable_names, xlabel=xlabel)

    # sis_entropies_global, non_sis_entropies_global, con_entropies_global = for_all_datasets(A_sis=sis_A_global, B_sis=sis_B_global, A_non_sis=non_sis_A_global,
    #                                                                    B_non_sis=non_sis_B_global, A_con=con_A_global, B_con=con_B_global,
    #                                                                    variable_names=Population._variable_names,
    #                                                                    graph_vars=[], seed=42)

    # xlabel = 'higher or lower than {} mean'.format(type_of_mean)
    # save_the_graphs(sis_entropies=sis_entropies_global, non_sis_entropies=non_sis_entropies_global, con_entropies=con_entropies_global,
    #                 variable_names=Population._variable_names, xlabel=xlabel)

    type_of_mean = 'trap'  # ['global', 'trap']

    sis_entropies_trap_hist, non_sis_entropies_trap_hist, con_entropies_trap_hist = trap_hists(A_sis=sis_A_trap, B_sis=sis_B_trap,
                                                                                                     A_non_sis=non_sis_A_trap,
                                                                                                     B_non_sis=non_sis_B_trap, A_con=con_A_trap,
                                                                                                     B_con=con_B_trap,
                                                                                                     variable_names=Population._variable_names,
                                                                                                     graph_vars=Population._variable_names,
                                                                                               n_bins=bin_num, seed=42, strategy=strategy)

    xlabel = '{} bins, {} strategy, {} centered'.format(bin_num, strategy, type_of_mean)
    save_the_graphs(sis_entropies=sis_entropies_trap_hist, non_sis_entropies=non_sis_entropies_trap_hist, con_entropies=con_entropies_trap_hist,
                    variable_names=Population._variable_names, xlabel=xlabel)

    exit()

    # sis_entropies_trap, non_sis_entropies_trap, con_entropies_trap = for_all_datasets(A_sis=sis_A_trap, B_sis=sis_B_trap,
    #                                                                                         A_non_sis=non_sis_A_trap,
    #                                                                                         B_non_sis=non_sis_B_trap, A_con=con_A_trap,
    #                                                                                         B_con=con_B_trap,
    #                                                                                         variable_names=Population._variable_names,
    #                                                                                         graph_vars=[], seed=42)

    # xlabel = 'higher or lower than {} mean'.format(type_of_mean)
    # save_the_graphs(sis_entropies=sis_entropies_trap, non_sis_entropies=non_sis_entropies_trap, con_entropies=con_entropies_trap,
    #                 variable_names=Population._variable_names, xlabel=xlabel)

    # plus_plus_joint, minus_minus_joint, plus_minus_joint, minus_plus_joint, A_plus_marginal, A_minus_marginal, B_plus_marginal, B_minus_marginal
    for var in Population._variable_names:
        # sns.kdeplot(np.array(sis_entropies_global[-1][var])[:, 0], label='sis plus_plus_joint', color='blue')
        # sns.kdeplot(np.array(sis_entropies_global[-1][var])[:, 1], label='sis minus_minus_joint', color='blue', linestyle='--')
        # sns.kdeplot(np.array(non_sis_entropies_global[-1][var])[:, 0], label='non_sis plus_plus_joint', color='orange')
        # sns.kdeplot(np.array(non_sis_entropies_global[-1][var])[:, 1], label='non_sis minus_minus_joint', color='orange', linestyle='--')
        # sns.kdeplot(np.array(con_entropies_global[-1][var])[:, 0], label='con plus_plus_joint', color='green')
        # sns.kdeplot(np.array(con_entropies_global[-1][var])[:, 1], label='con minus_minus_joint', color='green', linestyle='--')
        sns.kdeplot(np.array(sis_entropies_global[-1][var])[:, 0]+np.array(sis_entropies_global[-1][var])[:, 1], label='sis ++ and --', color='blue')
        sns.kdeplot(np.array(sis_entropies_global[-1][var])[:, 2] + np.array(sis_entropies_global[-1][var])[:, 3], label='sis +- and -+', color='blue', linestyle='--')
        sns.kdeplot(np.array(non_sis_entropies_global[-1][var])[:, 0] + np.array(non_sis_entropies_global[-1][var])[:, 1], label='non_sis ++ and --',
                    color='orange')
        sns.kdeplot(np.array(non_sis_entropies_global[-1][var])[:, 2] + np.array(non_sis_entropies_global[-1][var])[:, 3], label='non_sis +- and -+',
                    color='orange', linestyle='--')
        sns.kdeplot(np.array(con_entropies_global[-1][var])[:, 0] + np.array(con_entropies_global[-1][var])[:, 1], label='con ++ and --',
                    color='green')
        sns.kdeplot(np.array(con_entropies_global[-1][var])[:, 2] + np.array(con_entropies_global[-1][var])[:, 3], label='con +- and -+',
                    color='green', linestyle='--')
        plt.legend()
        plt.title(var)
        # plt.show()
        plt.savefig('correct and incorrect probabilities, global {}'.format(var), dpi=300)
        plt.close()

    # plus_plus_joint, minus_minus_joint, plus_minus_joint, minus_plus_joint, A_plus_marginal, A_minus_marginal, B_plus_marginal, B_minus_marginal
    for var in Population._variable_names:
        # sns.kdeplot(np.array(sis_entropies_trap[-1][var])[:, 0], label='sis plus_plus_joint', color='blue')
        # sns.kdeplot(np.array(sis_entropies_trap[-1][var])[:, 1], label='sis minus_minus_joint', color='blue', linestyle='--')
        # sns.kdeplot(np.array(non_sis_entropies_trap[-1][var])[:, 0], label='non_sis plus_plus_joint', color='orange')
        # sns.kdeplot(np.array(non_sis_entropies_trap[-1][var])[:, 1], label='non_sis minus_minus_joint', color='orange', linestyle='--')
        # sns.kdeplot(np.array(con_entropies_trap[-1][var])[:, 0], label='con plus_plus_joint', color='green')
        # sns.kdeplot(np.array(con_entropies_trap[-1][var])[:, 1], label='con minus_minus_joint', color='green', linestyle='--')
        sns.kdeplot(np.array(sis_entropies_trap[-1][var])[:, 0] + np.array(sis_entropies_trap[-1][var])[:, 1], label='sis ++ and --',
                    color='blue')
        sns.kdeplot(np.array(sis_entropies_trap[-1][var])[:, 2] + np.array(sis_entropies_trap[-1][var])[:, 3], label='sis +- and -+',
                    color='blue', linestyle='--')
        sns.kdeplot(np.array(non_sis_entropies_trap[-1][var])[:, 0] + np.array(non_sis_entropies_trap[-1][var])[:, 1], label='non_sis ++ and --',
                    color='orange')
        sns.kdeplot(np.array(non_sis_entropies_trap[-1][var])[:, 2] + np.array(non_sis_entropies_trap[-1][var])[:, 3], label='non_sis +- and -+',
                    color='orange', linestyle='--')
        sns.kdeplot(np.array(con_entropies_trap[-1][var])[:, 0] + np.array(con_entropies_trap[-1][var])[:, 1], label='con ++ and --',
                    color='green')
        sns.kdeplot(np.array(con_entropies_trap[-1][var])[:, 2] + np.array(con_entropies_trap[-1][var])[:, 3], label='con +- and -+',
                    color='green', linestyle='--')
        plt.legend()
        plt.title(var)
        # plt.show()
        plt.savefig('correct and incorrect probabilities, trap {}'.format(var), dpi=300)
        plt.close()





if __name__ == '__main__':
    main()
