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


def give_the_trap_bins_for_control(key, sis_A, sis_B, non_sis_A, non_sis_B, bins_on_side, var, log_vars):
    # give the key of the reference of the dictionary, for example, "nonsis_A_57"
    if key.split('_')[0] == 'sis':
        if var in log_vars:
            ref_A = np.log(sis_A[key.split('_')[2]][var])
            ref_B = np.log(sis_B[key.split('_')[2]][var])
        else:
            ref_A = sis_A[key.split('_')[2]][var]
            ref_B = sis_B[key.split('_')[2]][var]
        trap = pd.concat([ref_A, ref_B], axis=0).reset_index(drop=True)

    else:
        if var in log_vars:
            ref_A = np.log(non_sis_A[key.split('_')[2]][var])
            ref_B = np.log(non_sis_B[key.split('_')[2]][var])
        else:
            ref_A = non_sis_A[key.split('_')[2]][var]
            ref_B = non_sis_B[key.split('_')[2]][var]
        trap = pd.concat([ref_A, ref_B], axis=0).reset_index(drop=True)

    trap_var_bins = [-(((bins_on_side - interval) / bins_on_side) * (np.mean(trap) - min(trap))) + np.mean(trap) for interval
               in range(bins_on_side)] + \
              [((interval / bins_on_side) * (max(trap) - np.mean(trap))) + np.mean(trap) for interval in
               range(bins_on_side + 1)]

    return trap_var_bins


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


def put_the_categories_global(A_df, B_df, bins_on_side, variable_names, all_bacteria, log_vars):
    intra_array_A = []
    intra_array_B = []

    # get the global edges for all the variables
    edges = dict()
    all_bacteria_copy = all_bacteria.copy()
    for var in variable_names:
        if var in log_vars:
            all_bacteria_copy[var] = np.log(all_bacteria[var])
        else:
            all_bacteria_copy[var] = all_bacteria[var]
            
        edge = [-(((bins_on_side - interval) / bins_on_side) * (np.mean(all_bacteria_copy[var]) - min(all_bacteria_copy[var]))) + np.mean(all_bacteria_copy[var])
                 for interval in range(bins_on_side)] + \
                [((interval / bins_on_side) * (max(all_bacteria_copy[var]) - np.mean(all_bacteria_copy[var]))) + np.mean(all_bacteria_copy[var]) for interval in
                 range(bins_on_side + 1)]
    
        for indexx in range(len(edge) - 1):
            if edge[indexx] >= edge[indexx + 1]:
                print('edges:', edge)
                print(all_bacteria_copy[var])
                
        edges.update({var: edge})

    print('edges global:\n', edges)

    for ind in range(6):
        print(ind)
        print('_____')

        intra_A_df = dict()
        intra_B_df = dict()
        for var in variable_names:
            edge = edges[var]
            var_A_series = []
            var_B_series = []
            for val_A, val_B in zip(A_df.values(), B_df.values()):
                if var in log_vars:
                    A_centered = np.log(val_A[var]).iloc[:min(len(val_A[var]), len(val_B[var]))]
                    B_centered = np.log(val_B[var]).iloc[:min(len(val_A[var]), len(val_B[var]))]
                else:
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
        
        # checking there are still no nans
        if intra_A_df.isna().values.any():
            print('intra_A_df has nans still!')
            exit()
        if intra_B_df.isna().values.any():
            print('intra_B_df has nans still!')
            exit()

        # put it for this intra-generation
        intra_array_A.append(intra_A_df)
        intra_array_B.append(intra_B_df)

    return intra_array_A, intra_array_B


def put_the_categories_trap(A_df, B_df, bins_on_side, variable_names, log_vars):
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
                if var in log_vars:
                    A_centered = np.log(val_A[var]).iloc[:min(len(val_A[var]), len(val_B[var]))]
                    B_centered = np.log(val_B[var]).iloc[:min(len(val_A[var]), len(val_B[var]))]
                else:
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

        # checking there are still no nans
        if intra_A_df.isna().values.any():
            print('intra_A_df has nans still!')
            exit()
        if intra_B_df.isna().values.any():
            print('intra_B_df has nans still!')
            exit()

        # put it for this intra-generation
        intra_array_A.append(intra_A_df)
        intra_array_B.append(intra_B_df)

    return intra_array_A, intra_array_B


def put_the_categories_trap_con(A_df, B_df, bins_on_side, variable_names, ref_A, ref_B, sis_A, sis_B, non_sis_A, non_sis_B, log_vars):
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
            for key_A, key_B in zip(A_df.keys(), B_df.keys()):
                if var in log_vars:
                    A_centered = np.log(A_df[key_A][var]).iloc[:min(len(A_df[key_A][var]), len(B_df[key_B][var]))]
                    B_centered = np.log(B_df[key_B][var]).iloc[:min(len(A_df[key_A][var]), len(B_df[key_B][var]))]
                else:
                    A_centered = A_df[key_A][var].iloc[:min(len(A_df[key_A][var]), len(B_df[key_B][var]))]
                    B_centered = B_df[key_B][var].iloc[:min(len(A_df[key_A][var]), len(B_df[key_B][var]))]

                A_edges = give_the_trap_bins_for_control(key=ref_A[key_A], sis_A=sis_A, sis_B=sis_B, non_sis_A=non_sis_A, non_sis_B=non_sis_B,
                                                         bins_on_side=bins_on_side, var=var, log_vars=log_vars)
                B_edges = give_the_trap_bins_for_control(key=ref_B[key_B], sis_A=sis_A, sis_B=sis_B, non_sis_A=non_sis_A, non_sis_B=non_sis_B,
                                                         bins_on_side=bins_on_side, var=var, log_vars=log_vars)

                # print('A_edges:', A_edges)
                # print('B_edges:', B_edges)
                # exit()

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

                # print('A_cut:', A_cut)

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

            # print('var_A_series:', len(var_A_series), var_A_series)
            # append this variable's labels for all traps
            intra_A_df.update({var: var_A_series})
            intra_B_df.update({var: var_B_series})

        # make all the variables from all traps into a dataframe
        intra_A_df = pd.DataFrame(intra_A_df)
        intra_B_df = pd.DataFrame(intra_B_df)

        # checking there are still no nans
        if intra_A_df.isna().values.any():
            print('intra_A_df has nans still!')
            exit()
        if intra_B_df.isna().values.any():
            print('intra_B_df has nans still!')
            exit()

        # print('intra_A_df:', intra_A_df)
        # exit()

        # put it for this intra-generation
        intra_array_A.append(intra_A_df)
        intra_array_B.append(intra_B_df)

    return intra_array_A, intra_array_B


def put_the_categories_traj(A_df, B_df, bins_on_side, variable_names, log_vars):
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
                if var in log_vars:
                    A_centered = np.log(val_A[var]).iloc[:min(len(val_A[var]), len(val_B[var]))]
                    B_centered = np.log(val_B[var]).iloc[:min(len(val_A[var]), len(val_B[var]))]
                else:
                    A_centered = val_A[var].iloc[:min(len(val_A[var]), len(val_B[var]))]
                    B_centered = val_B[var].iloc[:min(len(val_A[var]), len(val_B[var]))]

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

        # checking there are still no nans
        if intra_A_df.isna().values.any():
            print('intra_A_df has nans still!')
            exit()
        if intra_B_df.isna().values.any():
            print('intra_B_df has nans still!')
            exit()

        # put it for this intra-generation
        intra_array_A.append(intra_A_df)
        intra_array_B.append(intra_B_df)

    return intra_array_A, intra_array_B


def get_entropies_from_labeled_data(A_cut, B_cut, bins_on_side):
    joint_centered = pd.DataFrame({'A': A_cut, 'B': B_cut})

    # print(A_cut)
    # print(B_cut)

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
    all_bacteria = pd.concat([mom, daug], axis=0).reset_index(drop=True).drop_duplicates(inplace=False, keep='first', subset=mom.columns)
    all_bacteria = all_bacteria[all_bacteria['growth_rate'] > 0].reset_index(drop=True)
    
    log_vars = ['generationtime', 'length_birth', 'length_final', 'growth_rate', 'fold_growth']

    # # the Histograms to check if variables are normal or lognormal
    # for var in Population._variable_names:
    #     print(var)
    #     normal = np.random.normal(np.mean(all_bacteria[var]), np.std(all_bacteria[var]), 500)
    #     lognormal = np.random.lognormal(np.mean(np.log(all_bacteria[var])), np.std(np.log(all_bacteria[var])), 500)
    #     normal_da_p = stats.mstats.normaltest(all_bacteria[var])
    #     log_normal_da_p = stats.mstats.normaltest(np.log(all_bacteria[var]))
    #     print('test for normality:', normal_da_p[0], normal_da_p[1])
    #     print('test for log-normality:', log_normal_da_p[0], log_normal_da_p[1])
    #     # sns.kdeplot(all_bacteria[var], label=var)
    #     # sns.kdeplot(normal, label='normal')
    #     # sns.kdeplot(lognormal, label='lognormal')
    #     # plt.legend()
    #     # plt.show()
    #     # plt.close()
    # exit()

    # so that every dataset has 88 sets
    np.random.seed(42)
    sis_keys = np.random.choice(list(sis_A.keys()), size=88, replace=False)
    new_sis_A = dict([(key, sis_A[key]) for key in sis_keys])
    new_sis_B = dict([(key, sis_B[key]) for key in sis_keys])

    for bins_on_side in range(1, 8): # starting on 4 now
        print('bins_on_side', bins_on_side)
        
        # for the global bin edges
        print('sis global')
        intra_array_A, intra_array_B = put_the_categories_global(A_df=new_sis_A, B_df=new_sis_B, bins_on_side=bins_on_side,
                                                               variable_names=Population._variable_names, all_bacteria=all_bacteria, log_vars=log_vars)
        MI_sis_global = dict([(var, [get_entropies_from_labeled_data(A_cut=A_array[var], B_cut=B_array[var], bins_on_side=bins_on_side)[0] for
                                   A_array, B_array in zip(intra_array_A, intra_array_B)]) for var in Population._variable_names])
        print('non sis global')
        intra_array_A, intra_array_B = put_the_categories_global(A_df=non_sis_A, B_df=non_sis_B, bins_on_side=bins_on_side,
                                                               variable_names=Population._variable_names, all_bacteria=all_bacteria, log_vars=log_vars)
        MI_non_sis_global = dict([(var, [get_entropies_from_labeled_data(A_cut=A_array[var], B_cut=B_array[var], bins_on_side=bins_on_side)[0] for
                                       A_array, B_array in zip(intra_array_A, intra_array_B)]) for var in Population._variable_names])
        print('control global')
        intra_array_A, intra_array_B = put_the_categories_global(A_df=con_A, B_df=con_B, bins_on_side=bins_on_side,
                                                               variable_names=Population._variable_names, all_bacteria=all_bacteria, log_vars=log_vars)
        MI_con_global = dict([(var, [get_entropies_from_labeled_data(A_cut=A_array[var], B_cut=B_array[var], bins_on_side=bins_on_side)[0] for
                                   A_array, B_array in zip(intra_array_A, intra_array_B)]) for var in Population._variable_names])
        
        # for the trap bin edges
        print('sis trap')
        intra_array_A, intra_array_B = put_the_categories_trap(A_df=new_sis_A, B_df=new_sis_B, bins_on_side=bins_on_side,
                                                               variable_names=Population._variable_names, log_vars=log_vars)
        MI_sis_trap = dict([(var, [get_entropies_from_labeled_data(A_cut=A_array[var], B_cut=B_array[var], bins_on_side=bins_on_side)[0] for
                                   A_array, B_array in zip(intra_array_A, intra_array_B)]) for var in Population._variable_names])
        print('non sis trap')
        intra_array_A, intra_array_B = put_the_categories_trap(A_df=non_sis_A, B_df=non_sis_B, bins_on_side=bins_on_side,
                                                               variable_names=Population._variable_names, log_vars=log_vars)
        MI_non_sis_trap = dict([(var, [get_entropies_from_labeled_data(A_cut=A_array[var], B_cut=B_array[var], bins_on_side=bins_on_side)[0] for
                                       A_array, B_array in zip(intra_array_A, intra_array_B)]) for var in Population._variable_names])
        print('control trap')
        intra_array_A, intra_array_B = put_the_categories_trap_con(A_df=con_A, B_df=con_B, bins_on_side=bins_on_side,
                                                                   variable_names=Population._variable_names, ref_A=con_ref_A, ref_B=con_ref_B,
                                                                   sis_A=sis_A, sis_B=sis_B, non_sis_A=non_sis_A, non_sis_B=non_sis_B, 
                                                                   log_vars=log_vars)
        MI_con_trap = dict([(var, [get_entropies_from_labeled_data(A_cut=A_array[var], B_cut=B_array[var], bins_on_side=bins_on_side)[0] for
                                   A_array, B_array in zip(intra_array_A, intra_array_B)]) for var in Population._variable_names])
        
        # for the traj bin edges
        print('sis traj')
        intra_array_A, intra_array_B = put_the_categories_traj(A_df=new_sis_A, B_df=new_sis_B, bins_on_side=bins_on_side, 
                                                               variable_names=Population._variable_names, log_vars=log_vars)
        MI_sis_traj = dict([(var, [get_entropies_from_labeled_data(A_cut=A_array[var], B_cut=B_array[var], bins_on_side=bins_on_side)[0] for
                       A_array, B_array in zip(intra_array_A, intra_array_B)]) for var in Population._variable_names])
        print('non sis traj')
        intra_array_A, intra_array_B = put_the_categories_traj(A_df=non_sis_A, B_df=non_sis_B, bins_on_side=bins_on_side,
                                                               variable_names=Population._variable_names, log_vars=log_vars)
        MI_non_sis_traj = dict([(var, [get_entropies_from_labeled_data(A_cut=A_array[var], B_cut=B_array[var], bins_on_side=bins_on_side)[0] for
                                  A_array, B_array in zip(intra_array_A, intra_array_B)]) for var in Population._variable_names])
        print('control traj')
        intra_array_A, intra_array_B = put_the_categories_traj(A_df=con_A, B_df=con_B, bins_on_side=bins_on_side,
                                                               variable_names=Population._variable_names, log_vars=log_vars)
        MI_con_traj = dict([(var, [get_entropies_from_labeled_data(A_cut=A_array[var], B_cut=B_array[var], bins_on_side=bins_on_side)[0] for
                              A_array, B_array in zip(intra_array_A, intra_array_B)]) for var in Population._variable_names])

        for var in Population._variable_names:
            # the GLOBAL
            plt.plot(np.arange(1, len(MI_sis_global) + 1), MI_sis_global[var], label='sis', marker='.')
            plt.plot(np.arange(1, len(MI_non_sis_global) + 1), MI_non_sis_global[var], label='non_sis', marker='.')
            plt.plot(np.arange(1, len(MI_con_global) + 1), MI_con_global[var], label='con', marker='.')
            plt.legend()
            plt.title('intra-gen {} with {} bins on each side of the global mean'.format(var, bins_on_side))
            plt.ylabel('Mutual Information')
            plt.xlabel('Intra-Generation')
            # plt.show()
            if var in log_vars:
                plt.savefig('log {} with {} bins on each side of the global mean'.format(var, bins_on_side), dpi=300)
            else:
                plt.savefig('{} with {} bins on each side of the global mean'.format(var, bins_on_side), dpi=300)
            plt.close()

            # the TRAP
            plt.plot(np.arange(1, len(MI_sis_trap) + 1), MI_sis_trap[var], label='sis', marker='.')
            plt.plot(np.arange(1, len(MI_non_sis_trap) + 1), MI_non_sis_trap[var], label='non_sis', marker='.')
            plt.plot(np.arange(1, len(MI_con_trap) + 1), MI_con_trap[var], label='con', marker='.')
            plt.legend()
            plt.title('intra-gen {} with {} bins on each side of the trap mean'.format(var, bins_on_side))
            plt.ylabel('Mutual Information')
            plt.xlabel('Intra-Generation')
            # plt.show()
            if var in log_vars:
                plt.savefig('NEW log {} with {} bins on each side of the trap mean'.format(var, bins_on_side), dpi=300)
            else:
                plt.savefig('NEW {} with {} bins on each side of the trap mean'.format(var, bins_on_side), dpi=300)
            plt.close()

            # the TRAJ
            plt.plot(np.arange(1, len(MI_sis_traj) + 1), MI_sis_traj[var], label='sis', marker='.')
            plt.plot(np.arange(1, len(MI_non_sis_traj) + 1), MI_non_sis_traj[var], label='non_sis', marker='.')
            plt.plot(np.arange(1, len(MI_con_traj) + 1), MI_con_traj[var], label='con', marker='.')
            plt.legend()
            plt.title('intra-gen {} with {} bins on each side of the traj mean'.format(var, bins_on_side))
            plt.ylabel('Mutual Information')
            plt.xlabel('Intra-Generation')
            # plt.show()
            if var in log_vars:
                plt.savefig('log {} with {} bins on each side of the traj mean'.format(var, bins_on_side), dpi=300)
            else:
                plt.savefig('{} with {} bins on each side of the traj mean'.format(var, bins_on_side), dpi=300)
            plt.close()
    exit()

    delta_sis_intra_A = [pd.DataFrame([val.iloc[ind] - np.mean(val) for val in sis_A.values()]) for ind in range(6)]
    delta_sis_intra_B = [pd.DataFrame([val.iloc[ind] - np.mean(val) for val in sis_B.values()]) for ind in range(6)]
    delta_non_sis_intra_A = [pd.DataFrame([val.iloc[ind] - np.mean(val) for val in non_sis_A.values()]) for ind in range(6)]
    delta_non_sis_intra_B = [pd.DataFrame([val.iloc[ind] - np.mean(val) for val in non_sis_B.values()]) for ind in range(6)]
    delta_con_intra_A = [pd.DataFrame([val.iloc[ind] - np.mean(val) for val in con_A.values()]) for ind in range(6)]
    delta_con_intra_B = [pd.DataFrame([val.iloc[ind] - np.mean(val) for val in con_B.values()]) for ind in range(6)]

    bins_on_side = 1
    for var in Population._variable_names:
        sis_MI = [get_entropies(A_centered=delta_sis_intra_A[ind][var], B_centered=delta_sis_intra_B[ind][var], bins_on_side=bins_on_side)[2] for ind in range(len(delta_sis_intra_A))]
        non_sis_MI = [get_entropies(A_centered=delta_non_sis_intra_A[ind][var], B_centered=delta_non_sis_intra_B[ind][var], bins_on_side=bins_on_side)[2] for ind in
                  range(len(delta_non_sis_intra_A))]
        con_MI = [get_entropies(A_centered=delta_con_intra_A[ind][var], B_centered=delta_con_intra_B[ind][var], bins_on_side=bins_on_side)[2] for ind in
                  range(len(delta_con_intra_A))]
        plt.plot(np.arange(1, len(delta_sis_intra_A)+1), sis_MI, label='sis')
        plt.plot(np.arange(1, len(delta_sis_intra_A)+1), non_sis_MI, label='non sis')
        plt.plot(np.arange(1, len(delta_sis_intra_A)+1), con_MI, label='control')
        plt.legend()
        plt.title('{} with {} bins on each side of the mean'.format(var, bins_on_side))
        plt.ylabel('Mutual Information')
        plt.xlabel('Intra-Generation')
        plt.show()
        plt.close()
        
    # print([(var, [get_entropies(A_centered=sis_intra_A[ind][var], B_centered=sis_intra_B[ind][var], bins_on_side=1)[2] for ind in range(len(sis_intra_A))]) for var in Population._variable_names])
    exit()

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
    sis_A_trap = {key: subtract_trap_averages(df=val, columns_names=val.columns, trap_mean=trap_mean.loc['mean']) for key, val, trap_mean in
                  zip(sis_A.keys(), sis_A.values(), Sister.trap_stats_dict.values())}
    sis_B_trap = {key: subtract_trap_averages(df=val, columns_names=val.columns, trap_mean=trap_mean.loc['mean']) for key, val, trap_mean in
                  zip(sis_B.keys(), sis_B.values(), Sister.trap_stats_dict.values())}
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













if __name__ == '__main__':
    main()
