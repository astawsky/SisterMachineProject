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


def put_the_categories_global(A_df, B_df, bins_on_side, variable_names, all_bacteria, log_vars):

    # get the global edges for all the variables
    edges = dict()
    all_bacteria_copy = all_bacteria.copy()
    for var in variable_names:
        if var in log_vars:
            all_bacteria_copy[var] = np.log(all_bacteria[var])
        else:
            all_bacteria_copy[var] = all_bacteria[var]

        edge = [-(((bins_on_side - interval) / bins_on_side) * (np.mean(all_bacteria_copy[var]) - min(all_bacteria_copy[var]))) + np.mean(
            all_bacteria_copy[var])
                for interval in range(bins_on_side)] + \
               [((interval / bins_on_side) * (max(all_bacteria_copy[var]) - np.mean(all_bacteria_copy[var]))) + np.mean(all_bacteria_copy[var]) for
                interval in
                range(bins_on_side + 1)]

        for indexx in range(len(edge) - 1):
            if edge[indexx] >= edge[indexx + 1]:
                print('edges:', edge)
                print(all_bacteria_copy[var])

        edges.update({var: edge})

    # print('edges global:\n', edges)
    
    A_cut_dict = dict()
    B_cut_dict = dict()
    for key_A, key_B in zip(A_df.keys(), B_df.keys()):
        df_A_cut = pd.DataFrame(columns=variable_names)
        df_B_cut = pd.DataFrame(columns=variable_names)
        for var in variable_names:
            
            edge = edges[var]
            
            if var in log_vars:
                A_centered = np.log(A_df[key_A][var]).iloc[:min(len(A_df[key_A][var]), len(B_df[key_B][var]))]
                B_centered = np.log(B_df[key_B][var]).iloc[:min(len(A_df[key_A][var]), len(B_df[key_B][var]))]
            else:
                A_centered = A_df[key_A][var].iloc[:min(len(A_df[key_A][var]), len(B_df[key_B][var]))]
                B_centered = B_df[key_B][var].iloc[:min(len(A_df[key_A][var]), len(B_df[key_B][var]))]

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
                print('_____')
                print(var)
                print('B')
                print(key_B)
                print(B_cut.isna().sum())
                print(B_cut)
                print(B_centered)

                if (abs(np.array(B_centered.iloc[np.where(B_cut.isna())]) - np.array(edge)[0])[0] < .0001):
                    # print('its close to the left one!')
                    B_cut.iloc[np.where(B_cut.isna())] = 0.0
                    # print(B_cut.isna().any())
                elif (abs(np.array(B_centered.iloc[np.where(B_cut.isna())]) - np.array(edge)[-1])[0] < .0001):
                    # print('its close to the right one!')
                    B_cut.iloc[np.where(B_cut.isna())] = np.int64(len(edge) - 2)
                    # print(B_cut.isna().any())

            # append this trap's variable's labels
            df_A_cut[var] = A_cut
            df_B_cut[var] = B_cut

        # checking there are still no nans
        if df_A_cut.isna().values.any():
            print('df_A_cut has nans still!')
            print(df_A_cut)
            print(df_A_cut.isna())
            exit()
        if df_B_cut.isna().values.any():
            print('df_B_cut has nans still!')
            print(df_B_cut)
            print(df_B_cut.isna())
            exit()

        # append this variable's labels for all traps
        A_cut_dict.update({key_A: df_A_cut})
        B_cut_dict.update({key_B: df_B_cut})

    return A_cut_dict, B_cut_dict


def put_the_categories_trap(A_df, B_df, bins_on_side, variable_names, log_vars):
    A_cut_dict = dict()
    B_cut_dict = dict()
    for key_A, key_B in zip(A_df.keys(), B_df.keys()):
        df_A_cut = pd.DataFrame(columns=variable_names)
        df_B_cut = pd.DataFrame(columns=variable_names)
        for var in variable_names:
            if var in log_vars:
                A_centered = np.log(A_df[key_A][var]).iloc[:min(len(A_df[key_A][var]), len(B_df[key_B][var]))]
                B_centered = np.log(B_df[key_B][var]).iloc[:min(len(A_df[key_A][var]), len(B_df[key_B][var]))]
            else:
                A_centered = A_df[key_A][var].iloc[:min(len(A_df[key_A][var]), len(B_df[key_B][var]))]
                B_centered = B_df[key_B][var].iloc[:min(len(A_df[key_A][var]), len(B_df[key_B][var]))]
            joint_centered = pd.concat([A_centered, B_centered]).reset_index(drop=True)

            edges = [-(((bins_on_side - interval) / bins_on_side) * (np.mean(joint_centered) - min(joint_centered))) + np.mean(joint_centered)
                     for interval in range(bins_on_side)] + \
                    [((interval / bins_on_side) * (max(joint_centered) - np.mean(joint_centered))) + np.mean(joint_centered) for interval in
                     range(bins_on_side + 1)]

            for indexx in range(len(edges) - 1):
                if edges[indexx] >= edges[indexx + 1]:
                    print('edges:', edges)
                    print(joint_centered)

            # based on the bin edge, give each entry in the A and B series an integer that maps to the bin in the marginal distribution
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
            df_A_cut[var] = A_cut
            df_B_cut[var] = B_cut

        # checking there are still no nans
        if df_A_cut.isna().values.any():
            print('df_A_cut has nans still!')
            exit()
        if df_B_cut.isna().values.any():
            print('df_B_cut has nans still!')
            exit()

        # append this variable's labels for all traps
        A_cut_dict.update({key_A: df_A_cut})
        B_cut_dict.update({key_B: df_B_cut})

    return A_cut_dict, B_cut_dict


def put_the_categories_trap_con(A_df, B_df, bins_on_side, variable_names, ref_A, ref_B, sis_A, sis_B, non_sis_A, non_sis_B, log_vars):
    A_cut_dict = dict()
    B_cut_dict = dict()
    for key_A, key_B in zip(A_df.keys(), B_df.keys()):
        df_A_cut = pd.DataFrame(columns=variable_names)
        df_B_cut = pd.DataFrame(columns=variable_names)
        for var in variable_names:
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

            for indexx in range(len(A_edges) - 1):
                if A_edges[indexx] >= A_edges[indexx + 1]:
                    print('A_edges:', A_edges)
                    print(A_centered)
            for indexx in range(len(B_edges) - 1):
                if B_edges[indexx] >= B_edges[indexx + 1]:
                    print('B_edges:', B_edges)
                    print(B_centered)

            # based on the bin edge, give each entry in the A and B series an integer that maps to the bin in the marginal distribution
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
            df_A_cut[var] = A_cut
            df_B_cut[var] = B_cut

        # checking there are still no nans
        if df_A_cut.isna().values.any():
            print('df_A_cut has nans still!')
            exit()
        if df_B_cut.isna().values.any():
            print('df_B_cut has nans still!')
            exit()

        # append this variable's labels for all traps
        A_cut_dict.update({key_A: df_A_cut})
        B_cut_dict.update({key_B: df_B_cut})

    return A_cut_dict, B_cut_dict


def put_the_categories_traj(A_df, B_df, bins_on_side, variable_names, log_vars):
    A_cut_dict = dict()
    B_cut_dict = dict()
    for key_A, key_B in zip(A_df.keys(), B_df.keys()):
        df_A_cut = pd.DataFrame(columns=variable_names)
        df_B_cut = pd.DataFrame(columns=variable_names)
        for var in variable_names:
            if var in log_vars:
                A_centered = np.log(A_df[key_A][var]).iloc[:min(len(A_df[key_A][var]), len(B_df[key_B][var]))]
                B_centered = np.log(B_df[key_B][var]).iloc[:min(len(A_df[key_A][var]), len(B_df[key_B][var]))]
            else:
                A_centered = A_df[key_A][var].iloc[:min(len(A_df[key_A][var]), len(B_df[key_B][var]))]
                B_centered = B_df[key_B][var].iloc[:min(len(A_df[key_A][var]), len(B_df[key_B][var]))]

            A_edges = [-(((bins_on_side - interval) / bins_on_side) * (np.mean(A_centered) - min(A_centered))) + np.mean(A_centered) for interval
                       in
                       range(bins_on_side)] + \
                      [((interval / bins_on_side) * (max(A_centered) - np.mean(A_centered))) + np.mean(A_centered) for interval in
                       range(bins_on_side + 1)]

            B_edges = [-(((bins_on_side - interval) / bins_on_side) * (np.mean(B_centered) - min(B_centered))) + np.mean(B_centered) for interval
                       in
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

            # based on the bin edge, give each entry in the A and B series an integer that maps to the bin in the marginal distribution
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
            df_A_cut[var] = A_cut
            df_B_cut[var] = B_cut

        # checking there are still no nans
        if df_A_cut.isna().values.any():
            print('df_A_cut has nans still!')
            exit()
        if df_B_cut.isna().values.any():
            print('df_B_cut has nans still!')
            exit()

        # append this variable's labels for all traps
        A_cut_dict.update({key_A: df_A_cut})
        B_cut_dict.update({key_B: df_B_cut})

    return A_cut_dict, B_cut_dict


def get_entropies_from_labeled_data(A_cut, B_cut, bins_on_side):
    joint_centered = pd.DataFrame({'A': A_cut, 'B': B_cut})

    joint_prob_list = dict(
        [('{}_{}'.format(label_A, label_B), len(joint_centered[(joint_centered['A'] == label_A) & (joint_centered['B'] == label_B)]) /
          len(joint_centered)) for label_B in np.arange(2 * bins_on_side) for label_A in np.arange(2 * bins_on_side)])
    A_trace_marginal_probs = dict([('{}'.format(label_A), len(A_cut.iloc[np.where(A_cut == label_A)]) / len(A_cut))
                                   for label_A in np.arange(2 * bins_on_side)])
    B_trace_marginal_probs = dict([('{}'.format(label_B), len(B_cut.iloc[np.where(B_cut == label_B)]) / len(B_cut))
                                   for label_B in np.arange(2 * bins_on_side)])

    # conditioning the A trace based on the B trace
    A_conditioned_on_B_entropy = np.array([- joint_prob_list[key] * np.log(joint_prob_list[key] / A_trace_marginal_probs[key.split('_')[0]])
                                           for key in joint_prob_list.keys() if joint_prob_list[key] != 0 and
                                           A_trace_marginal_probs[key.split('_')[0]] != 0]).sum()
    B_conditioned_on_A_entropy = np.array([- joint_prob_list[key] * np.log(joint_prob_list[key] / B_trace_marginal_probs[key.split('_')[1]])
                                           for key in joint_prob_list.keys() if joint_prob_list[key] != 0 and
                                           B_trace_marginal_probs[key.split('_')[1]] != 0]).sum()

    # the mutual information between A and B trace for this variable thinking that marginal A and B came from each trace distribution
    mutual_info_trace = round(np.array([joint_prob_list[key] * np.log(joint_prob_list[key] / (A_trace_marginal_probs[key.split('_')[0]] *
                                                                                        B_trace_marginal_probs[key.split('_')[1]]))
                                  for key in joint_prob_list.keys() if joint_prob_list[key] != 0 and
                                  B_trace_marginal_probs[key.split('_')[1]] != 0 and
                                  A_trace_marginal_probs[key.split('_')[0]] != 0]).sum(), 14)

    # checking joint prob adds up to one
    if round(np.array([val for val in joint_prob_list.values()]).sum(), 4) != 1.0:
        print('joint prob does not add up to 1.0! it adds up to {}'.format(np.array([val for val in joint_prob_list.values()]).sum()))
        exit()

    # checking A marginal prob adds up to one
    if round(np.array([val for val in A_trace_marginal_probs.values()]).sum(), 4) != 1.0:
        print('A_trace_marginal_probs does not add up to 1.0! it adds up to {}'.format(
            np.array([val for val in A_trace_marginal_probs.values()]).sum()))
        exit()

    # checking B marginal prob adds up to one
    if round(np.array([val for val in B_trace_marginal_probs.values()]).sum(), 4) != 1.0:
        print('B_trace_marginal_probs does not add up to 1.0! it adds up to {}'.format(
            np.array([val for val in B_trace_marginal_probs.values()]).sum()))
        exit()

    # mutual information cannot be negative
    if mutual_info_trace < 0:
        print('mutual info is negative! something is wrong...')
        print(A_cut)
        print(B_cut)
        print(joint_prob_list)
        print(A_trace_marginal_probs)
        print(B_trace_marginal_probs)
        print(mutual_info_trace)
        for key in joint_prob_list.keys():
            print('key:', key)
            if joint_prob_list[key] != 0 and B_trace_marginal_probs[key.split('_')[1]] != 0 and A_trace_marginal_probs[key.split('_')[0]] != 0:
                print(joint_prob_list[key])
                print(A_trace_marginal_probs[key.split('_')[0]])
                print(B_trace_marginal_probs[key.split('_')[1]])
                print(joint_prob_list[key] * np.log(joint_prob_list[key] / (A_trace_marginal_probs[key.split('_')[0]] *
                                                                                        B_trace_marginal_probs[key.split('_')[1]])))
        print('_________')
        exit()

    return mutual_info_trace, A_conditioned_on_B_entropy, B_conditioned_on_A_entropy, joint_prob_list


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

    log_vars = ['generationtime', 'length_birth', 'length_final', 'growth_rate', 'fold_growth']

    # so that every dataset has 88 sets
    np.random.seed(42)
    sis_keys = np.random.choice(list(sis_A.keys()), size=88, replace=False)
    new_sis_A = dict([(key, sis_A[key]) for key in sis_keys])
    new_sis_B = dict([(key, sis_B[key]) for key in sis_keys])

    for bins_on_side in range(1, 8):  # starting on 4 now
        print('bins_on_side', bins_on_side)

        # for the global bin edges
        print('sis global')
        intra_array_A, intra_array_B = put_the_categories_global(A_df=new_sis_A, B_df=new_sis_B, bins_on_side=bins_on_side,
                                                                 variable_names=Population._variable_names, all_bacteria=all_bacteria,
                                                                 log_vars=log_vars)
        MI_sis_global = dict([(var, dict([(key, get_entropies_from_labeled_data(A_cut=intra_array_A[key][var], B_cut=intra_array_B[key][var],
                                                bins_on_side=bins_on_side)[0]) for key in sis_keys])) for var in Population._variable_names])

        print('non sis global')
        intra_array_A, intra_array_B = put_the_categories_global(A_df=non_sis_A, B_df=non_sis_B, bins_on_side=bins_on_side,
                                                                 variable_names=Population._variable_names, all_bacteria=all_bacteria,
                                                                 log_vars=log_vars)
        MI_non_sis_global = dict([(var, dict([(key, get_entropies_from_labeled_data(A_cut=intra_array_A[key][var], B_cut=intra_array_B[key][var],
                                                bins_on_side=bins_on_side)[0]) for key in non_sis_A.keys()])) for var in Population._variable_names])

        print('control global')
        intra_array_A, intra_array_B = put_the_categories_global(A_df=con_A, B_df=con_B, bins_on_side=bins_on_side,
                                                                 variable_names=Population._variable_names, all_bacteria=all_bacteria,
                                                                 log_vars=log_vars)
        MI_con_global = dict([(var, dict([(key, get_entropies_from_labeled_data(A_cut=intra_array_A[key][var], B_cut=intra_array_B[key][var],
                                                                                    bins_on_side=bins_on_side)[0]) for key in con_A.keys()])) for
                                  var in Population._variable_names])

        # for the trap bin edges
        print('sis trap')
        intra_array_A, intra_array_B = put_the_categories_trap(A_df=new_sis_A, B_df=new_sis_B, bins_on_side=bins_on_side,
                                                               variable_names=Population._variable_names, log_vars=log_vars)
        MI_sis_trap = dict([(var, dict([(key, get_entropies_from_labeled_data(A_cut=intra_array_A[key][var], B_cut=intra_array_B[key][var],
                                                                                bins_on_side=bins_on_side)[-1]) for key in sis_keys])) for var in
                              Population._variable_names])

        # for var in Population._variable_names:
        #     print(var)
        #     print([np.array([val[key] for key in ['{}_{}'.format(ind, ind) for ind in range(bins_on_side+1)]]).sum() for val in MI_sis_trap[var].values()])
        # exit()

        print('non sis trap')
        intra_array_A, intra_array_B = put_the_categories_trap(A_df=non_sis_A, B_df=non_sis_B, bins_on_side=bins_on_side,
                                                               variable_names=Population._variable_names, log_vars=log_vars)
        MI_non_sis_trap = dict([(var, dict([(key, get_entropies_from_labeled_data(A_cut=intra_array_A[key][var], B_cut=intra_array_B[key][var],
                                                                                    bins_on_side=bins_on_side)[-1]) for key in non_sis_A.keys()])) for
                                  var in Population._variable_names])

        print('control trap')
        intra_array_A, intra_array_B = put_the_categories_trap_con(A_df=con_A, B_df=con_B, bins_on_side=bins_on_side,
                                                                   variable_names=Population._variable_names, ref_A=con_ref_A, ref_B=con_ref_B,
                                                                   sis_A=sis_A, sis_B=sis_B, non_sis_A=non_sis_A, non_sis_B=non_sis_B,
                                                                   log_vars=log_vars)
        MI_con_trap = dict([(var, dict([(key, get_entropies_from_labeled_data(A_cut=intra_array_A[key][var], B_cut=intra_array_B[key][var],
                                                                                    bins_on_side=bins_on_side)[-1]) for key in con_A.keys()])) for
                                  var in Population._variable_names])

        # for the traj bin edges
        print('sis traj')
        intra_array_A, intra_array_B = put_the_categories_traj(A_df=new_sis_A, B_df=new_sis_B, bins_on_side=bins_on_side,
                                                               variable_names=Population._variable_names, log_vars=log_vars)
        MI_sis_traj = dict([(var, dict([(key, get_entropies_from_labeled_data(A_cut=intra_array_A[key][var], B_cut=intra_array_B[key][var],
                                                                              bins_on_side=bins_on_side)[0]) for key in sis_keys])) for var in
                            Population._variable_names])

        print('non sis traj')
        intra_array_A, intra_array_B = put_the_categories_traj(A_df=non_sis_A, B_df=non_sis_B, bins_on_side=bins_on_side,
                                                               variable_names=Population._variable_names, log_vars=log_vars)
        MI_non_sis_traj = dict([(var, dict([(key, get_entropies_from_labeled_data(A_cut=intra_array_A[key][var], B_cut=intra_array_B[key][var],
                                                                                    bins_on_side=bins_on_side)[0]) for key in non_sis_A.keys()])) for
                                  var in Population._variable_names])

        print('control traj')
        intra_array_A, intra_array_B = put_the_categories_traj(A_df=con_A, B_df=con_B, bins_on_side=bins_on_side,
                                                               variable_names=Population._variable_names, log_vars=log_vars)
        MI_con_traj = dict([(var, dict([(key, get_entropies_from_labeled_data(A_cut=intra_array_A[key][var], B_cut=intra_array_B[key][var],
                                                                                    bins_on_side=bins_on_side)[0]) for key in con_A.keys()])) for
                                  var in Population._variable_names])

        for var in Population._variable_names:
            # the GLOBAL
            sns.kdeplot(list(MI_sis_global[var].values()), label='Sisters')
            sns.kdeplot(list(MI_non_sis_global[var].values()), label='Non-Sisters')
            sns.kdeplot(list(MI_con_global[var].values()), label='Control')
            plt.legend()
            plt.title('{} with {} bins on each side of the global mean'.format(var, bins_on_side))
            plt.ylabel('PDF')
            plt.xlabel('Mutual Information')
            # plt.show()
            if var in log_vars:
                plt.savefig('log {} with {} bins on each side of the global mean'.format(var, bins_on_side), dpi=300)
            else:
                plt.savefig('{} with {} bins on each side of the global mean'.format(var, bins_on_side), dpi=300)
            plt.close()

            # the TRAP
            # sns.kdeplot([np.array([val[key] for key in ['{}_{}'.format(ind, ind) for ind in range(bins_on_side+1)]]).sum()
            #              for val in MI_sis_trap[var].values()], label='Sisters')
            # sns.kdeplot([np.array([val[key] for key in ['{}_{}'.format(ind, ind) for ind in range(bins_on_side + 1)]]).sum()
            #              for val in MI_non_sis_trap[var].values()], label='Non-Sisters')
            # sns.kdeplot([np.array([val[key] for key in ['{}_{}'.format(ind, ind) for ind in range(bins_on_side + 1)]]).sum()
            #              for val in MI_con_trap[var].values()], label='Control')
            sns.kdeplot(list(MI_sis_trap[var].values()), label='Sisters')
            sns.kdeplot(list(MI_non_sis_trap[var].values()), label='Non-Sisters')
            sns.kdeplot(list(MI_con_trap[var].values()), label='Control')
            plt.legend()
            plt.title('{} with {} bins on each side of the trap mean'.format(var, bins_on_side))
            plt.ylabel('PDF')
            plt.xlabel('Mutual Information')
            # plt.show()
            if var in log_vars:
                plt.savefig('NEW log {} with {} bins on each side of the trap mean'.format(var, bins_on_side), dpi=300)
            else:
                plt.savefig('NEW {} with {} bins on each side of the trap mean'.format(var, bins_on_side), dpi=300)
            plt.close()

            # the TRAJ
            sns.kdeplot(list(MI_sis_traj[var].values()), label='Sisters')
            sns.kdeplot(list(MI_non_sis_traj[var].values()), label='Non-Sisters')
            sns.kdeplot(list(MI_con_traj[var].values()), label='Control')
            plt.legend()
            plt.title('{} with {} bins on each side of the traj mean'.format(var, bins_on_side))
            plt.ylabel('PDF')
            plt.xlabel('Mutual Information')
            # plt.show()
            if var in log_vars:
                plt.savefig('log {} with {} bins on each side of the traj mean'.format(var, bins_on_side), dpi=300)
            else:
                plt.savefig('{} with {} bins on each side of the traj mean'.format(var, bins_on_side), dpi=300)
            plt.close()


if __name__ == '__main__':
    main()
