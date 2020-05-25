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


def print_full_dataframe(df):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(df)


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


def put_the_categories_global(A_df, B_df, bins_on_side, variable_names, all_bacteria):
    intra_array_A = []
    intra_array_B = []

    # get the edges for all the variables
    edges = dict()
    for var in variable_names:
        edge = [-(((bins_on_side - interval) / bins_on_side) * (np.mean(all_bacteria[var]) - min(all_bacteria[var]))) + np.mean(all_bacteria[var])
                 for interval in range(bins_on_side)] + \
                [((interval / bins_on_side) * (max(all_bacteria[var]) - np.mean(all_bacteria[var]))) + np.mean(all_bacteria[var]) for interval in
                 range(bins_on_side + 1)]

        for indexx in range(len(edge) - 1):
            if edge[indexx] >= edge[indexx + 1]:
                print('edges:', edge)
                print(all_bacteria[var])

        edges.update({var: edge})

    print('edges global:\n', edges)

    for ind in range(6):
        print(ind)
        print('_____')

        intra_A_df = dict()
        intra_B_df = dict()
        for var in variable_names:
            edge = edges[var]
            # print(var)
            var_A_series = []
            var_B_series = []
            for val_A, val_B in zip(A_df.values(), B_df.values()):
                A_centered = val_A[var].iloc[:min(len(val_A[var]), len(val_B[var]))]
                B_centered = val_B[var].iloc[:min(len(val_A[var]), len(val_B[var]))]

                # based on the bin edge, give each entry in the A and B series an integer that maps to the bin in the marginal distribution
                A_cut = pd.cut(A_centered, edge, right=True, include_lowest=True, labels=np.arange(len(edge) - 1))  # , duplicates='drop'
                B_cut = pd.cut(B_centered, edge, right=True, include_lowest=True, labels=np.arange(len(edge) - 1))
                # joint_centered = pd.DataFrame({'A': A_cut, 'B': B_cut}).iloc[ind]

                if A_cut.isna().any():
                    # print('_____')
                    # print(var)
                    # print('A')
                    # print(A_cut.isna().sum())

                    if (abs(np.array(A_centered.iloc[np.where(A_cut.isna())]) - np.array(edge)[0])[0] < .0001):
                        # print('its close to the left one!')
                        A_cut.iloc[np.where(A_cut.isna())] = 0.0
                        # print(A_cut.isna().any())
                    elif (abs(np.array(A_centered.iloc[np.where(A_cut.isna())]) - np.array(edge)[-1])[0] < .0001):
                        # print('its close to the right one!')
                        A_cut.iloc[np.where(A_cut.isna())] = np.int64(len(edge) - 2)
                        # print(A_cut.isna().any())

                if B_cut.isna().any():
                    # print('_____')
                    # print(var)
                    # print('B')
                    # print(B_cut.isna().sum())

                    if (abs(np.array(B_centered.iloc[np.where(B_cut.isna())]) - np.array(edge)[0])[0] < .0001):
                        # print('its close to the left one!')
                        B_cut.iloc[np.where(B_cut.isna())] = 0.0
                        # print(B_cut.isna().any())
                    elif (abs(np.array(B_centered.iloc[np.where(B_cut.isna())]) - np.array(edge)[-1])[0] < .0001):
                        # print('its close to the right one!')
                        B_cut.iloc[np.where(B_cut.isna())] = np.int64(len(edge) - 2)
                        # print(B_cut.isna().any())

                # append this trap's variable's labels
                var_A_series.append(A_cut.iloc[ind])
                var_B_series.append(B_cut.iloc[ind])

            # append this variable's labels for all traps
            intra_A_df.update({var: var_A_series})
            intra_B_df.update({var: var_B_series})

        # make all the variables from all traps into a dataframe
        intra_A_df = pd.DataFrame(intra_A_df)
        intra_B_df = pd.DataFrame(intra_B_df)

        # put it for this intra-generation
        intra_array_A.append(intra_A_df)
        intra_array_B.append(intra_B_df)

    return intra_array_A, intra_array_B


def put_the_categories_trap(A_df, B_df, bins_on_side, variable_names):
    intra_array_A = []
    intra_array_B = []

    for ind in range(6):
        print(ind)
        print('_____')

        intra_A_df = dict()
        intra_B_df = dict()
        for var in variable_names:
            # print(var)
            var_A_series = []
            var_B_series = []
            for val_A, val_B in zip(A_df.values(), B_df.values()):
                A_centered = val_A[var].iloc[:min(len(val_A[var]), len(val_B[var]))]
                B_centered = val_B[var].iloc[:min(len(val_A[var]), len(val_B[var]))]
                joint_centered = pd.concat([A_centered, B_centered]).reset_index(drop=True)

                edges = [-(((bins_on_side - interval) / bins_on_side) * (np.mean(joint_centered) - min(joint_centered))) + np.mean(joint_centered)
                         for interval in range(bins_on_side)] + \
                          [((interval / bins_on_side) * (max(joint_centered) - np.mean(joint_centered))) + np.mean(joint_centered) for interval in
                           range(bins_on_side + 1)]

                for indexx in range(len(edges) - 1):
                    if edges[indexx] >= edges[indexx + 1]:
                        print('edges:', edges)
                        print(joint_centered)

                # based on the bin edges, give each entry in the A and B series an integer that maps to the bin in the marginal distribution
                A_cut = pd.cut(A_centered, edges, right=True, include_lowest=True, labels=np.arange(len(edges) - 1))  # , duplicates='drop'
                B_cut = pd.cut(B_centered, edges, right=True, include_lowest=True, labels=np.arange(len(edges) - 1))
                # joint_centered = pd.DataFrame({'A': A_cut, 'B': B_cut}).iloc[ind]

                if A_cut.isna().any():
                    # print('_____')
                    # print(var)
                    # print('A')
                    # print(A_cut.isna().sum())

                    if (abs(np.array(A_centered.iloc[np.where(A_cut.isna())]) - np.array(edges)[0])[0] < .0001):
                        # print('its close to the left one!')
                        A_cut.iloc[np.where(A_cut.isna())] = 0.0
                        # print(A_cut.isna().any())
                    elif (abs(np.array(A_centered.iloc[np.where(A_cut.isna())]) - np.array(edges)[-1])[0] < .0001):
                        # print('its close to the right one!')
                        A_cut.iloc[np.where(A_cut.isna())] = np.int64(len(edges) - 2)
                        # print(A_cut.isna().any())

                if B_cut.isna().any():
                    # print('_____')
                    # print(var)
                    # print('B')
                    # print(B_cut.isna().sum())

                    if (abs(np.array(B_centered.iloc[np.where(B_cut.isna())]) - np.array(edges)[0])[0] < .0001):
                        # print('its close to the left one!')
                        B_cut.iloc[np.where(B_cut.isna())] = 0.0
                        # print(B_cut.isna().any())
                    elif (abs(np.array(B_centered.iloc[np.where(B_cut.isna())]) - np.array(edges)[-1])[0] < .0001):
                        # print('its close to the right one!')
                        B_cut.iloc[np.where(B_cut.isna())] = np.int64(len(edges) - 2)
                        # print(B_cut.isna().any())

                # append this trap's variable's labels
                var_A_series.append(A_cut.iloc[ind])
                var_B_series.append(B_cut.iloc[ind])

            # append this variable's labels for all traps
            intra_A_df.update({var: var_A_series})
            intra_B_df.update({var: var_B_series})

        # make all the variables from all traps into a dataframe
        intra_A_df = pd.DataFrame(intra_A_df)
        intra_B_df = pd.DataFrame(intra_B_df)

        # put it for this intra-generation
        intra_array_A.append(intra_A_df)
        intra_array_B.append(intra_B_df)

    return intra_array_A, intra_array_B


def put_the_categories_traj(A_df, B_df, bins_on_side, variable_names):
    intra_array_A = []
    intra_array_B = []

    for ind in range(6):
        print(ind)
        print('_____')

        intra_A_df = dict()
        intra_B_df = dict()
        for var in variable_names:
            # print(var)
            var_A_series = []
            var_B_series = []
            for val_A, val_B in zip(A_df.values(), B_df.values()):
                A_centered = val_A[var].iloc[:min(len(val_A[var]), len(val_B[var]))]
                B_centered = val_B[var].iloc[:min(len(val_A[var]), len(val_B[var]))]
                # joint_centered = pd.DataFrame({'A': A_centered, 'B': B_centered})

                A_edges = [-(((bins_on_side - interval) / bins_on_side) * (np.mean(A_centered) - min(A_centered))) + np.mean(A_centered) for interval in
                           range(bins_on_side)] + \
                          [((interval / bins_on_side) * (max(A_centered) - np.mean(A_centered))) + np.mean(A_centered) for interval in
                           range(bins_on_side + 1)]

                B_edges = [-(((bins_on_side - interval) / bins_on_side) * (np.mean(B_centered) - min(B_centered))) + np.mean(B_centered) for interval in
                           range(bins_on_side)] + \
                          [((interval / bins_on_side) * (max(B_centered) - np.mean(B_centered))) + np.mean(B_centered) for interval in
                           range(bins_on_side + 1)]

                for indexx in range(len(A_edges) - 1):
                    if A_edges[indexx] >= A_edges[indexx + 1]:
                        print('A_edges:', A_edges)
                        print(A_centered)
                for indexx in range(len(B_edges) - 1):
                    if B_edges[indexx] >= B_edges[indexx + 1]:
                        print('B_edges:', B_edges)
                        print(B_centered)

                # based on the bin edges, give each entry in the A and B series an integer that maps to the bin in the marginal distribution
                A_cut = pd.cut(A_centered, A_edges, right=True, include_lowest=True, labels=np.arange(len(A_edges) - 1))  # , duplicates='drop'
                B_cut = pd.cut(B_centered, B_edges, right=True, include_lowest=True, labels=np.arange(len(B_edges) - 1))
                # joint_centered = pd.DataFrame({'A': A_cut, 'B': B_cut}).iloc[ind]

                if A_cut.isna().any():
                    # print('_____')
                    # print(var)
                    # print('A')
                    # print(A_cut.isna().sum())

                    if (abs(np.array(A_centered.iloc[np.where(A_cut.isna())]) - np.array(A_edges)[0])[0] < .0001):
                        # print('its close to the left one!')
                        A_cut.iloc[np.where(A_cut.isna())] = 0.0
                        # print(A_cut.isna().any())
                    elif (abs(np.array(A_centered.iloc[np.where(A_cut.isna())]) - np.array(A_edges)[-1])[0] < .0001):
                        # print('its close to the right one!')
                        A_cut.iloc[np.where(A_cut.isna())] = np.int64(len(A_edges) - 2)
                        # print(A_cut.isna().any())

                if B_cut.isna().any():
                    # print('_____')
                    # print(var)
                    # print('B')
                    # print(B_cut.isna().sum())

                    if (abs(np.array(B_centered.iloc[np.where(B_cut.isna())]) - np.array(B_edges)[0])[0] < .0001):
                        # print('its close to the left one!')
                        B_cut.iloc[np.where(B_cut.isna())] = 0.0
                        # print(B_cut.isna().any())
                    elif (abs(np.array(B_centered.iloc[np.where(B_cut.isna())]) - np.array(B_edges)[-1])[0] < .0001):
                        # print('its close to the right one!')
                        B_cut.iloc[np.where(B_cut.isna())] = np.int64(len(B_edges) - 2)
                        # print(B_cut.isna().any())

                # append this trap's variable's labels
                var_A_series.append(A_cut.iloc[ind])
                var_B_series.append(B_cut.iloc[ind])

            # append this variable's labels for all traps
            intra_A_df.update({var: var_A_series})
            intra_B_df.update({var: var_B_series})

        # make all the variables from all traps into a dataframe
        intra_A_df = pd.DataFrame(intra_A_df)
        intra_B_df = pd.DataFrame(intra_B_df)

        # put it for this intra-generation
        intra_array_A.append(intra_A_df)
        intra_array_B.append(intra_B_df)

    return intra_array_A, intra_array_B


def get_categories_traj(val_A, val_B, bins_on_side, variable_names):
    mi_dict = dict()
    for var1 in variable_names:
        for var2 in variable_names:
            A_centered = val_A[var1]
            B_centered = val_B[var2]
            # joint_centered = pd.DataFrame({'A': A_centered, 'B': B_centered})
    
            A_edges = [-(((bins_on_side - interval) / bins_on_side) * (np.mean(A_centered) - min(A_centered))) + np.mean(A_centered) for interval in
                       range(bins_on_side)] + \
                      [((interval / bins_on_side) * (max(A_centered) - np.mean(A_centered))) + np.mean(A_centered) for interval in
                       range(bins_on_side + 1)]
    
            B_edges = [-(((bins_on_side - interval) / bins_on_side) * (np.mean(B_centered) - min(B_centered))) + np.mean(B_centered) for interval in
                       range(bins_on_side)] + \
                      [((interval / bins_on_side) * (max(B_centered) - np.mean(B_centered))) + np.mean(B_centered) for interval in
                       range(bins_on_side + 1)]
    
            for indexx in range(len(A_edges) - 1):
                if A_edges[indexx] >= A_edges[indexx + 1]:
                    print('A_edges:', A_edges)
                    print(A_centered)
            for indexx in range(len(B_edges) - 1):
                if B_edges[indexx] >= B_edges[indexx + 1]:
                    print('B_edges:', B_edges)
                    print(B_centered)
    
            # based on the bin edges, give each entry in the A and B series an integer that maps to the bin in the marginal distribution
            A_cut = pd.cut(A_centered, A_edges, right=True, include_lowest=True, labels=np.arange(len(A_edges) - 1))  # , duplicates='drop'
            B_cut = pd.cut(B_centered, B_edges, right=True, include_lowest=True, labels=np.arange(len(B_edges) - 1))
            # joint_centered = pd.DataFrame({'A': A_cut, 'B': B_cut}).iloc[ind]
    
            if A_cut.isna().any():
                # print('_____')
                # print(var)
                # print('A')
                # print(A_cut.isna().sum())
    
                if (abs(np.array(A_centered.iloc[np.where(A_cut.isna())]) - np.array(A_edges)[0])[0] < .0001):
                    # print('its close to the left one!')
                    A_cut.iloc[np.where(A_cut.isna())] = 0.0
                    # print(A_cut.isna().any())
                elif (abs(np.array(A_centered.iloc[np.where(A_cut.isna())]) - np.array(A_edges)[-1])[0] < .0001):
                    # print('its close to the right one!')
                    A_cut.iloc[np.where(A_cut.isna())] = np.int64(len(A_edges) - 2)
                    # print(A_cut.isna().any())
    
            if B_cut.isna().any():
                # print('_____')
                # print(var)
                # print('B')
                # print(B_cut.isna().sum())
    
                if (abs(np.array(B_centered.iloc[np.where(B_cut.isna())]) - np.array(B_edges)[0])[0] < .0001):
                    # print('its close to the left one!')
                    B_cut.iloc[np.where(B_cut.isna())] = 0.0
                    # print(B_cut.isna().any())
                elif (abs(np.array(B_centered.iloc[np.where(B_cut.isna())]) - np.array(B_edges)[-1])[0] < .0001):
                    # print('its close to the right one!')
                    B_cut.iloc[np.where(B_cut.isna())] = np.int64(len(B_edges) - 2)
                    # print(B_cut.isna().any())
            
            mi_dict.update({'{}_{}'.format(var1, var2): get_entropies_from_labeled_data(A_cut, B_cut, bins_on_side)[0]})
            
    return mi_dict


def get_categories_trap(val_A, val_B, bins_on_side, variable_names):
    mi_dict = dict()
    for var1 in variable_names:
        for var2 in variable_names:
            A_centered = val_A[var1]
            B_centered = val_B[var2]
            joint_centered = pd.concat([A_centered, B_centered]).reset_index(drop=True)

            edges = [-(((bins_on_side - interval) / bins_on_side) * (np.mean(joint_centered) - min(joint_centered))) + np.mean(joint_centered)
                     for interval in range(bins_on_side)] + \
                    [((interval / bins_on_side) * (max(joint_centered) - np.mean(joint_centered))) + np.mean(joint_centered) for interval in
                     range(bins_on_side + 1)]
            
            print(var1, var2, np.mean(joint_centered), 'should be 0')

            for indexx in range(len(edges) - 1):
                if edges[indexx] >= edges[indexx + 1]:
                    print('edges:', edges)
                    print(joint_centered)

            # based on the bin edges, give each entry in the A and B series an integer that maps to the bin in the marginal distribution
            A_cut = pd.cut(A_centered, edges, right=True, include_lowest=True, labels=np.arange(len(edges) - 1))  # , duplicates='drop'
            B_cut = pd.cut(B_centered, edges, right=True, include_lowest=True, labels=np.arange(len(edges) - 1))
            # joint_centered = pd.DataFrame({'A': A_cut, 'B': B_cut}).iloc[ind]

            if A_cut.isna().any():
                # print('_____')
                # print(var)
                # print('A')
                # print(A_cut.isna().sum())

                if (abs(np.array(A_centered.iloc[np.where(A_cut.isna())]) - np.array(edges)[0])[0] < .0001):
                    # print('its close to the left one!')
                    A_cut.iloc[np.where(A_cut.isna())] = 0.0
                    # print(A_cut.isna().any())
                elif (abs(np.array(A_centered.iloc[np.where(A_cut.isna())]) - np.array(edges)[-1])[0] < .0001):
                    # print('its close to the right one!')
                    A_cut.iloc[np.where(A_cut.isna())] = np.int64(len(edges) - 2)
                    # print(A_cut.isna().any())

            if B_cut.isna().any():
                # print('_____')
                # print(var)
                # print('B')
                # print(B_cut.isna().sum())

                if (abs(np.array(B_centered.iloc[np.where(B_cut.isna())]) - np.array(edges)[0])[0] < .0001):
                    # print('its close to the left one!')
                    B_cut.iloc[np.where(B_cut.isna())] = 0.0
                    # print(B_cut.isna().any())
                elif (abs(np.array(B_centered.iloc[np.where(B_cut.isna())]) - np.array(edges)[-1])[0] < .0001):
                    # print('its close to the right one!')
                    B_cut.iloc[np.where(B_cut.isna())] = np.int64(len(edges) - 2)
                    # print(B_cut.isna().any())

            mi_dict.update({'{}_{}'.format(var1, var2): get_entropies_from_labeled_data(A_cut, B_cut, bins_on_side)[0]})

    return mi_dict


def get_entropies_from_labeled_data(A_cut, B_cut, bins_on_side):
    joint_centered = pd.DataFrame({'A': A_cut, 'B': B_cut})

    joint_prob_list = dict(
        [('{}_{}'.format(label_A, label_B), len(joint_centered[(joint_centered['A'] == label_A) & (joint_centered['B'] == label_B)]) /
          len(joint_centered)) for label_B in np.arange(2*bins_on_side) for label_A in np.arange(2*bins_on_side)])
    A_trace_marginal_probs = dict([('{}'.format(label_A), len(A_cut.iloc[np.where(A_cut == label_A)]) / len(A_cut))
                                   for label_A in np.arange(2*bins_on_side)])
    B_trace_marginal_probs = dict([('{}'.format(label_B), len(B_cut.iloc[np.where(B_cut == label_B)]) / len(B_cut))
                                   for label_B in np.arange(2*bins_on_side)])

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

    # checking joint prob adds up to one
    if round(np.array([val for val in joint_prob_list.values()]).sum(), 4) != 1.0:
        print('joint prob does not add up to 1.0! it adds up to {}'.format(np.array([val for val in joint_prob_list.values()]).sum()))
        exit()

    # checking A marginal prob adds up to one
    if round(np.array([val for val in A_trace_marginal_probs.values()]).sum(), 4) != 1.0:
        print('A_trace_marginal_probs does not add up to 1.0! it adds up to {}'.format(np.array([val for val in A_trace_marginal_probs.values()]).sum()))
        exit()

    # checking B marginal prob adds up to one
    if round(np.array([val for val in B_trace_marginal_probs.values()]).sum(), 4) != 1.0:
        print('B_trace_marginal_probs does not add up to 1.0! it adds up to {}'.format(np.array([val for val in B_trace_marginal_probs.values()]).sum()))
        exit()

    # mutual information cannot be negative
    if mutual_info_trace < 0:
        print('mutual info is negative! something is wrong...')
        print('_________')
        exit()

    return mutual_info_trace, A_conditioned_on_B_entropy, B_conditioned_on_A_entropy, joint_prob_list


def get_entropies(A_centered, B_centered,
                  bins_on_side):  # this gives the conditional entropies and mutual information for two centered vectors

    A_centered = A_centered.iloc[:min(len(A_centered), len(B_centered))]
    B_centered = B_centered.iloc[:min(len(A_centered), len(B_centered))]
    joint_centered = pd.DataFrame({'A': A_centered, 'B': B_centered})

    # A_edges = [min(A_centered), -np.std(A_centered), 0, np.std(A_centered), max(A_centered)]
    # B_edges = [min(B_centered), -np.std(B_centered), 0, np.std(B_centered), max(B_centered)]
    # A_edges = [min(A_centered),  - np.mean(A_centered) - np.std(A_centered), np.mean(A_centered), np.mean(A_centered) + np.std(A_centered), max(A_centered)]
    # B_edges = [min(B_centered), - np.mean(B_centered) - np.std(B_centered), np.mean(B_centered), np.mean(B_centered) + np.std(B_centered),
    #            max(B_centered)]

    A_edges = [-(((bins_on_side - interval) / bins_on_side) * (np.mean(A_centered) - min(A_centered))) + np.mean(A_centered) for interval in
               range(bins_on_side)] + \
              [((interval / bins_on_side) * (max(A_centered) - np.mean(A_centered))) + np.mean(A_centered) for interval in range(bins_on_side + 1)]

    B_edges = [-(((bins_on_side - interval) / bins_on_side) * (np.mean(B_centered) - min(B_centered))) + np.mean(B_centered) for interval in
               range(bins_on_side)] + \
              [((interval / bins_on_side) * (max(B_centered) - np.mean(B_centered))) + np.mean(B_centered) for interval in
               range(bins_on_side + 1)]

    # A_edges = [-(((bins_on_side - interval) / bins_on_side) * (np.mean(A_centered) - min(A_centered))) + np.mean(A_centered) for interval in
    #            range(bins_on_side)] + \
    #           [((interval / bins_on_side) * (max(A_centered) - np.mean(A_centered))) + np.mean(A_centered) for interval in range(bins_on_side + 1)]
    #
    # B_edges = [-(((bins_on_side - interval) / bins_on_side) * (np.mean(B_centered) - min(B_centered))) + np.mean(B_centered) for interval in
    #            range(bins_on_side)] + \
    #           [((interval / bins_on_side) * (max(B_centered) - np.mean(B_centered))) + np.mean(B_centered) for interval in
    #            range(bins_on_side + 1)]

    # print('A_edges:', A_edges)
    # print('B_edges:', B_edges)
    # print('_________')

    for ind in range(len(A_edges) - 1):
        if A_edges[ind] >= A_edges[ind + 1]:
            print('A_edges:', A_edges)
            print(A_centered)
    for ind in range(len(B_edges) - 1):
        if B_edges[ind] >= B_edges[ind + 1]:
            print('B_edges:', B_edges)
            print(B_centered)

    # based on the bin edges, give each entry in the A and B series an integer that maps to the bin in the marginal distribution
    A_cut = pd.cut(A_centered, A_edges, right=True, include_lowest=True, labels=np.arange(len(A_edges) - 1))  # , duplicates='drop'
    B_cut = pd.cut(B_centered, B_edges, right=True, include_lowest=True, labels=np.arange(len(B_edges) - 1))
    joint_centered = pd.DataFrame({'A': A_cut, 'B': B_cut})

    joint_prob_list = dict(
        [('{}_{}'.format(label_A, label_B), len(joint_centered[(joint_centered['A'] == label_A) & (joint_centered['B'] == label_B)]) /
          len(joint_centered)) for label_B in np.arange(len(B_edges) - 1) for label_A in np.arange(len(A_edges) - 1)])
    A_trace_marginal_probs = dict([('{}'.format(label_A), len(A_cut.iloc[np.where(A_cut == label_A)]) / len(A_cut))
                                   for label_A in np.arange(len(A_edges) - 1)])
    B_trace_marginal_probs = dict([('{}'.format(label_B), len(B_cut.iloc[np.where(B_cut == label_B)]) / len(B_cut))
                                   for label_B in np.arange(len(B_edges) - 1)])

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

    return A_conditioned_on_B_entropy, B_conditioned_on_A_entropy, mutual_info_trace, joint_prob_list


def check_global_bin_edges(mom, daug, variable_names, pop_stats, bins_on_side):
    # getting the all bacteria dataframe
    all_bacteria = pd.concat([mom, daug], axis=0).reset_index().drop_duplicates(inplace=False, keep='first', subset=mom.columns)

    # get the edges for all the variables
    edges = dict()
    for var in variable_names:
        edge = [-(((bins_on_side - interval) / bins_on_side) * (np.mean(all_bacteria[var]) - min(all_bacteria[var]))) + np.mean(all_bacteria[var])
                for interval in range(bins_on_side)] + \
               [((interval / bins_on_side) * (max(all_bacteria[var]) - np.mean(all_bacteria[var]))) + np.mean(all_bacteria[var]) for interval in
                range(bins_on_side + 1)]

        for indexx in range(len(edge) - 1):
            if edge[indexx] >= edge[indexx + 1]:
                print('edges:', edge)
                print(all_bacteria[var])

        edges.update({var: edge})

    print('edges global:\n', edges)
    print(pop_stats.loc['mean'])


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

    # renaming
    mom, daug = Population.mother_dfs[0].copy(), Population.daughter_dfs[0].copy()
    sis_A, sis_B = Sister.A_dict.copy(), Sister.B_dict.copy()
    non_sis_A, non_sis_B = Nonsister.A_dict.copy(), Nonsister.B_dict.copy()
    con_A, con_B = Control.A_dict.copy(), Control.B_dict.copy()
    con_ref_A, con_ref_B = Control.reference_A_dict.copy(), Control.reference_B_dict.copy()
    env_sis_intra_A, env_sis_intra_B = Env_Sister.A_intra_gen_bacteria.copy(), Env_Sister.B_intra_gen_bacteria.copy()
    sis_intra_A, sis_intra_B = Sister.A_intra_gen_bacteria.copy(), Sister.B_intra_gen_bacteria.copy()
    non_sis_intra_A, non_sis_intra_B = Nonsister.A_intra_gen_bacteria.copy(), Nonsister.B_intra_gen_bacteria.copy()
    env_nonsis_intra_A, env_nonsis_intra_B = Env_Nonsister.A_intra_gen_bacteria.copy(), Env_Nonsister.B_intra_gen_bacteria.copy()
    con_intra_A, con_intra_B = Control.A_intra_gen_bacteria.copy(), Control.B_intra_gen_bacteria.copy()

    # getting the all bacteria dataframe
    all_bacteria = pd.concat([mom, daug], axis=0).reset_index().drop_duplicates(inplace=False, keep='first', subset=mom.columns)

    # so that every dataset has 88 sets
    np.random.seed(42)
    sis_keys = np.random.choice(list(sis_A.keys()), size=130, replace=False)
    new_sis_A = dict([(key, sis_A[key]) for key in sis_keys])
    new_sis_B = dict([(key, sis_B[key]) for key in sis_keys])

    end_num = 9
    start_num = 0
    necessary_num = 9

    sis_A_trap_pooled = pd.DataFrame(columns=Population._variable_names)
    sis_B_trap_pooled = pd.DataFrame(columns=Population._variable_names)
    sis_A_traj_pooled = pd.DataFrame(columns=Population._variable_names)
    sis_B_traj_pooled = pd.DataFrame(columns=Population._variable_names)
    count = 0
    for A, B in zip(new_sis_A.values(), new_sis_B.values()):
        if min(len(A), len(B)) > necessary_num and count < 88:
            count += 1
            trap = pd.concat([A, B], axis=0).reset_index(drop=True)
            sis_A_traj_pooled = sis_A_traj_pooled.append(A.iloc[start_num:end_num] - A.iloc[start_num:end_num].mean(), ignore_index=True)
            sis_B_traj_pooled = sis_B_traj_pooled.append(B.iloc[start_num:end_num] - B.iloc[start_num:end_num].mean(), ignore_index=True)
            sis_A_trap_pooled = sis_A_trap_pooled.append(A.iloc[start_num:end_num] - trap.mean(), ignore_index=True)
            sis_B_trap_pooled = sis_B_trap_pooled.append(B.iloc[start_num:end_num] - trap.mean(), ignore_index=True)

            if len(sis_A_traj_pooled) != len(sis_B_traj_pooled):
                print('sis traj size dont match')
                exit()
                
            if len(sis_A_trap_pooled) != len(sis_B_trap_pooled):
                print('sis trap size dont match')
                exit()

    non_sis_A_trap_pooled = pd.DataFrame(columns=Population._variable_names)
    non_sis_B_trap_pooled = pd.DataFrame(columns=Population._variable_names)
    non_sis_A_traj_pooled = pd.DataFrame(columns=Population._variable_names)
    non_sis_B_traj_pooled = pd.DataFrame(columns=Population._variable_names)
    for A, B in zip(non_sis_A.values(), non_sis_B.values()):
        if min(len(A), len(B)) > necessary_num:
            trap = pd.concat([A, B], axis=0).reset_index(drop=True)
            non_sis_A_traj_pooled = non_sis_A_traj_pooled.append(A.iloc[start_num:end_num] - A.iloc[start_num:end_num].mean(), ignore_index=True)
            non_sis_B_traj_pooled = non_sis_B_traj_pooled.append(B.iloc[start_num:end_num] - B.iloc[start_num:end_num].mean(), ignore_index=True)
            non_sis_A_trap_pooled = non_sis_A_trap_pooled.append(A.iloc[start_num:end_num] - trap.mean(), ignore_index=True)
            non_sis_B_trap_pooled = non_sis_B_trap_pooled.append(B.iloc[start_num:end_num] - trap.mean(), ignore_index=True)

            if len(non_sis_A_traj_pooled) != len(non_sis_B_traj_pooled):
                print('non_sis traj size dont match')
                exit()

            if len(non_sis_A_trap_pooled) != len(non_sis_B_trap_pooled):
                print('non_sis trap size dont match')
                exit()

    con_A_trap_pooled = pd.DataFrame(columns=Population._variable_names)
    con_B_trap_pooled = pd.DataFrame(columns=Population._variable_names)
    con_A_traj_pooled = pd.DataFrame(columns=Population._variable_names)
    con_B_traj_pooled = pd.DataFrame(columns=Population._variable_names)
    for A, B in zip(con_A.values(), con_B.values()):
        if min(len(A), len(B)) > necessary_num:
            trap = pd.concat([A, B], axis=0).reset_index(drop=True)
            con_A_traj_pooled = con_A_traj_pooled.append(A.iloc[start_num:end_num] - A.iloc[start_num:end_num].mean(), ignore_index=True)
            con_B_traj_pooled = con_B_traj_pooled.append(B.iloc[start_num:end_num] - B.iloc[start_num:end_num].mean(), ignore_index=True)
            con_A_trap_pooled = con_A_trap_pooled.append(A.iloc[start_num:end_num] - trap.mean(), ignore_index=True)
            con_B_trap_pooled = con_B_trap_pooled.append(B.iloc[start_num:end_num] - trap.mean(), ignore_index=True)

            if len(con_A_traj_pooled) != len(con_B_traj_pooled):
                print('con traj size dont match')
                exit()

            if len(con_A_trap_pooled) != len(con_B_trap_pooled):
                print('con trap size dont match')
                exit()

    print(len(sis_A_trap_pooled), len(sis_B_trap_pooled))
    print(len(non_sis_A_trap_pooled), len(non_sis_B_trap_pooled))
    print(len(con_A_trap_pooled), len(con_B_trap_pooled))
    print(len(sis_A_traj_pooled), len(sis_B_traj_pooled))
    print(len(non_sis_A_traj_pooled), len(non_sis_B_traj_pooled))
    print(len(con_A_traj_pooled), len(con_B_traj_pooled))
    print('____')
    
    bins_on_side = 1

    sis_trap_mi_dict = get_categories_trap(val_A=sis_A_trap_pooled, val_B=sis_B_trap_pooled, bins_on_side=bins_on_side, variable_names=Population._variable_names)
    non_sis_trap_mi_dict = get_categories_trap(val_A=non_sis_A_trap_pooled, val_B=non_sis_B_trap_pooled, bins_on_side=bins_on_side,
                                      variable_names=Population._variable_names)
    con_trap_mi_dict = get_categories_trap(val_A=con_A_trap_pooled, val_B=con_B_trap_pooled, bins_on_side=bins_on_side,
                                      variable_names=Population._variable_names)

    sis_traj_mi_dict = get_categories_traj(val_A=sis_A_traj_pooled, val_B=sis_B_traj_pooled, bins_on_side=bins_on_side,
                                      variable_names=Population._variable_names)
    non_sis_traj_mi_dict = get_categories_traj(val_A=non_sis_A_traj_pooled, val_B=non_sis_B_traj_pooled, bins_on_side=bins_on_side,
                                          variable_names=Population._variable_names)
    con_traj_mi_dict = get_categories_traj(val_A=con_A_traj_pooled, val_B=con_B_traj_pooled, bins_on_side=bins_on_side,
                                      variable_names=Population._variable_names)
    
    sis_trap_mi_df = pd.DataFrame(columns=Population._variable_names, index=Population._variable_names)
    non_sis_trap_mi_df = pd.DataFrame(columns=Population._variable_names, index=Population._variable_names)
    con_trap_mi_df = pd.DataFrame(columns=Population._variable_names, index=Population._variable_names)
    for var1 in Population._variable_names:
        for var2 in Population._variable_names:
            # A trace has var1 and B_trace has var2
            sis_trap_mi_df[var1].loc[var2] = sis_trap_mi_dict['{}_{}'.format(var1, var2)]
            non_sis_trap_mi_df[var1].loc[var2] = non_sis_trap_mi_dict['{}_{}'.format(var1, var2)]
            con_trap_mi_df[var1].loc[var2] = con_trap_mi_dict['{}_{}'.format(var1, var2)]
            print(var1, var2)
            print(sis_trap_mi_dict['{}_{}'.format(var1, var2)])
            print(non_sis_trap_mi_dict['{}_{}'.format(var1, var2)])
            print(con_trap_mi_dict['{}_{}'.format(var1, var2)])
    
    exit()

    sis_lengths = np.array([min(len(A), len(B)) for A, B in zip(sis_A.values(), sis_B.values())])
    sis_at_least_lengths = [len(sis_lengths[np.where(sis_lengths >= length)]) for length in range(max(sis_lengths))]
    non_sis_lengths = np.array([min(len(A), len(B)) for A, B in zip(non_sis_A.values(), non_sis_B.values())])
    non_sis_at_least_lengths = [len(non_sis_lengths[np.where(non_sis_lengths >= length)]) for length in range(max(non_sis_lengths))]
    con_lengths = np.array([min(len(A), len(B)) for A, B in zip(con_A.values(), con_B.values())])
    con_at_least_lengths = [len(con_lengths[np.where(con_lengths >= length)]) for length in range(max(con_lengths))]

    # every one has at least [0:9] ie. ten generations

    sns.set_style('darkgrid')
    plt.plot(np.arange(1, len(sis_at_least_lengths)+1), sis_at_least_lengths, label='sis', marker='.')
    plt.plot(np.arange(1, len(non_sis_at_least_lengths)+1), non_sis_at_least_lengths, label='non_sis', marker='.')
    plt.plot(np.arange(1, len(con_at_least_lengths)+1), con_at_least_lengths, label='con', marker='.')
    plt.legend()
    plt.xlabel('length of shortest trace')
    plt.ylabel('number of traps that have this length')
    plt.show()
    plt.close()



if __name__ == '__main__':
    main()
