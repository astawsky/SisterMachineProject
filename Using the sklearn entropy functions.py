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
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif


def print_full_dataframe(df):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(df)


def give_the_trap_avg_for_control(key, sis_A, sis_B, non_sis_A, non_sis_B, start_index=None, end_index=None):
    # give the key of the reference of the dictionary, for example, "nonsis_A_57"
    if key.split('_')[0] == 'sis':
        ref_A = sis_A[key.split('_')[2]]
        ref_B = sis_B[key.split('_')[2]]
    else:
        ref_A = non_sis_A[key.split('_')[2]]
        ref_B = non_sis_B[key.split('_')[2]]

    # decide what generations to use to determine the trap mean
    if start_index == None:
        if end_index == None:
            trap_mean = pd.concat([ref_A, ref_B], axis=0).reset_index(drop=True).mean()
        else:
            trap_mean = pd.concat([ref_A.iloc[:end_index], ref_B.iloc[:end_index]], axis=0).reset_index(drop=True).mean()
    else:
        if end_index == None:
            trap_mean = pd.concat([ref_A.iloc[start_index:], ref_B.iloc[start_index:]], axis=0).reset_index(drop=True).mean()
        else:
            trap_mean = pd.concat([ref_A.iloc[start_index:end_index], ref_B.iloc[start_index:end_index]], axis=0).reset_index(drop=True).mean()

    return trap_mean


def subtract_trap_averages(df_main, df_other, columns_names, start_index=None, end_index=None):
    df_new = pd.DataFrame(columns=columns_names)
    for col in columns_names:
        if start_index == None:
            if end_index == None:
                trap_mean = pd.concat([df_main[columns_names], df_other[columns_names]], axis=0).reset_index(drop=True).mean()
                df_new[col] = df_main[col] - trap_mean[col]
            else:
                trap_mean = pd.concat([df_main[columns_names].iloc[:end_index], df_other[columns_names].iloc[:end_index]], axis=0).reset_index(drop=True).mean()
                df_new[col] = df_main[col].iloc[:end_index] - trap_mean[col]
        else:
            if end_index == None:
                trap_mean = pd.concat([df_main[columns_names].iloc[start_index:], df_other[columns_names].iloc[start_index:]], axis=0).reset_index(drop=True).mean()
                df_new[col] = df_main[col].iloc[start_index:] - trap_mean[col]
            else:
                trap_mean = pd.concat([df_main[columns_names].iloc[start_index:end_index], df_other[columns_names].iloc[start_index:end_index]], axis=0).reset_index(drop=True).mean()
                df_new[col] = df_main[col].iloc[start_index:end_index] - trap_mean[col]

    return df_new


def subtract_traj_averages(df, columns_names, start_index=None, end_index=None):
    df_new = pd.DataFrame(columns=columns_names)
    for col in columns_names:
        if start_index == None:
            if end_index == None:
                df_new[col] = df[col] - df[col].mean()
            else:
                df_new[col] = df[col].iloc[:end_index] - df[col].iloc[:end_index].mean()
        else:
            if end_index == None:
                df_new[col] = df[col].iloc[start_index:] - df[col].iloc[start_index:].mean()
            else:
                df_new[col] = df[col].iloc[start_index:end_index] - df[col].iloc[start_index:end_index].mean()

    return df_new


def subtract_trap_averages_control(df, columns_names, ref_key, sis_A, sis_B, non_sis_A, non_sis_B, start_index=None, end_index=None):
    # we have to get the original trap mean
    trap_mean = give_the_trap_avg_for_control(ref_key, sis_A, sis_B, non_sis_A, non_sis_B, start_index, end_index)

    df_new = pd.DataFrame(columns=columns_names)
    for col in columns_names:
        if start_index == None:
            if end_index == None:
                df_new[col] = df[col] - trap_mean[col]
            else:
                df_new[col] = df[col].iloc[:end_index] - trap_mean[col]
        else:
            if end_index == None:
                df_new[col] = df[col].iloc[start_index:] - trap_mean[col]
            else:
                df_new[col] = df[col].iloc[start_index:end_index] - trap_mean[col]

    return df_new


def subtract_global_averages(df, columns_names, pop_mean):
    df_new = pd.DataFrame(columns=columns_names)
    for col in columns_names:
        df_new[col] = df[col] - pop_mean[col]

    return df_new


def add_the_added_length(dictionary):
    new_dictionary = dictionary.copy()
    for key, val in dictionary.items():
        new_dictionary[key]['added_length'] = dictionary[key]['length_final'] - dictionary[key]['length_birth']
        if pd.Series(dictionary[key]['length_final'] < dictionary[key]['length_birth']).isna().any():
            print(dictionary[key]['length_final'].iloc[pd.Series(dictionary[key]['length_final'] < dictionary[key]['length_birth']).isna()],
                  dictionary[key]['length_birth'].iloc[pd.Series(dictionary[key]['length_final'] < dictionary[key]['length_birth']).isna()])
        if len(np.where(new_dictionary[key]['added_length'] < 0)) > 1:
            print("found a declining bacteria")
            print(np.where(new_dictionary[key]['added_length'] < 0))
            print(new_dictionary[key]['added_length'].iloc[np.where(new_dictionary[key]['added_length'] < 0)])
        new_dictionary[key] = new_dictionary[key].drop(index=np.where(new_dictionary[key]['added_length'] < 0)[0])

    return new_dictionary


def get_pooled_trap_df(A_df, B_df, variable_names, A_variable_symbol, B_variable_symbol):
    together_array = pd.DataFrame(columns=A_variable_symbol + B_variable_symbol, dtype=float)
    for val_A, val_B in zip(A_df.values(), B_df.values()):
        min_len = min(len(val_A), len(val_B))
        # append this variable's labels for all traps
        together_array = together_array.append(
            pd.concat([val_A.iloc[:min_len].rename(columns=dict(zip(variable_names, A_variable_symbol))),
                       val_B.iloc[:min_len].rename(columns=dict(zip(variable_names, B_variable_symbol)))],
                      axis=1), ignore_index=True)

    return together_array


def calculate_discrete_and_continuous_MIs(df_A, df_B, variable_names_A, variable_names_B, k):
    MI_df = pd.DataFrame(columns=variable_names_A, index=variable_names_B, dtype=float)
    corr_coeff_df = pd.DataFrame(columns=variable_names_A, index=variable_names_B)
    for var1 in variable_names_A:
        for var2 in variable_names_B:
            corr_coeff = round(stats.pearsonr(df_A[var1], df_B[var2])[0], 2)
            MI = mutual_info_regression(X=np.array(df_A[var1]).reshape(-1, 1), y=np.array(df_B[var2]), discrete_features=False,
                                        n_neighbors=k, copy=True, random_state=42)[0]

            """ This is if we want to make generationtime a discrete variable. Because this changes how we estimate the MI from all the other continuous 
            variables, we decided to interpret generationtime as a continuous variable, even though it can very easily be binned discretely because of
            the .05 hour intervals in which we measure the bacteria. 
            """
            # if var1 != variable_names_A[0] and var2 != variable_names_B[0]:
            #     corr_coeff = round(stats.pearsonr(df_A[var1], df_B[var2])[0], 2)
            #     MI = mutual_info_regression(X=np.array(df_A[var1]).reshape(-1, 1), y=np.array(df_B[var2]), discrete_features=False,
            #                                 n_neighbors=k, copy=True, random_state=42)[0]
            # elif var1 != variable_names_A[0] and var2 == variable_names_B[0]:
            #     # here we discretize the generationtime since they are in units of hours and we need integers for the sklearn algorithm
            #     discrete_factor_B = [int(blah * 100-min(df_B[var2])*100) for blah in df_B[var2]]
            #     corr_coeff = round(stats.pearsonr(discrete_factor_B, df_A[var1])[0], 2)
            #     MI = mutual_info_regression(X=np.array(discrete_factor_B).reshape(-1, 1), y=np.array(df_A[var1]),
            #                                 discrete_features=True, n_neighbors=k, copy=True, random_state=42)[0]
            # elif var1 == variable_names_A[0] and var2 != variable_names_B[0]:
            #     # here we discretize the generationtime since they are in units of hours and we need integers for the sklearn algorithm
            #     discrete_factor_A = [int(blah * 100-min(df_A[var1])*100) for blah in df_A[var1]]
            #     corr_coeff = round(stats.pearsonr(discrete_factor_A, df_B[var2])[0], 2)
            #     MI = mutual_info_regression(X=np.array(discrete_factor_A).reshape(-1, 1), y=np.array(df_B[var2]),
            #                                 discrete_features=True, n_neighbors=k, copy=True, random_state=42)[0]
            # else:  # both are generationtime
            #     # here we discretize the generationtime since they are in units of hours and we need integers for the sklearn algorithm
            #     discrete_factor_A = [int(blah * 100-min(df_A[var1])*100) for blah in df_A[var1]]
            #     discrete_factor_B = [int(blah * 100-min(df_B[var2])*100) for blah in df_B[var2]]
            #     corr_coeff = round(stats.pearsonr(discrete_factor_A, discrete_factor_B)[0], 2)
            #     MI = mutual_info_classif(X=np.array(discrete_factor_A).reshape(-1, 1), y=np.array(discrete_factor_B), discrete_features=True,
            #                              n_neighbors=k, copy=True, random_state=42)[0]

            MI_df[var1].loc[var2] = MI
            corr_coeff_df[var1].loc[var2] = corr_coeff

    return MI_df, corr_coeff_df


def plot_and_save_k_selection(df_A, df_B, variable_names_A, variable_names_B, k_array, title):
    for var1 in variable_names_A:
        for var2 in variable_names_B:
            if var1 != variable_names_A[0] and var2 != variable_names_B[0]:
                corr_coeff = round(stats.pearsonr(df_A[var1], df_B[var2])[0], 2)
                array = []
                for k in k_array:
                    array.append(mutual_info_regression(X=np.array(df_A[var1]).reshape(-1, 1), y=np.array(df_B[var2]), discrete_features=False,
                                                        n_neighbors=k, copy=True, random_state=42)[0])
            elif var1 != variable_names_A[0] and var2 == variable_names_B[0]:
                # here we discretize the generationtime since they are in units of hours and we need integers for the sklearn algorithm
                discrete_factor_B = [int(blah * 100) for blah in df_B[var2]]
                corr_coeff = round(stats.pearsonr(discrete_factor_B, df_A[var1])[0], 2)
                array = []
                for k in k_array:
                    array.append(mutual_info_regression(X=np.array(discrete_factor_B).reshape(-1, 1), y=np.array(df_A[var1]),
                                                        discrete_features=True, n_neighbors=k, copy=True, random_state=42)[0])
            elif var1 == variable_names_A[0] and var2 != variable_names_B[0]:
                # here we discretize the generationtime since they are in units of hours and we need integers for the sklearn algorithm
                discrete_factor_A = [int(blah * 100) for blah in df_A[var1]]
                corr_coeff = round(stats.pearsonr(discrete_factor_A, df_B[var2])[0], 2)
                array = []
                for k in k_array:
                    array.append(
                        mutual_info_regression(X=np.array(discrete_factor_A).reshape(-1, 1), y=np.array(df_B[var2]),
                                               discrete_features=True, n_neighbors=k, copy=True, random_state=42)[0])
            else:  # both are generationtime
                # here we discretize the generationtime since they are in units of hours and we need integers for the sklearn algorithm
                discrete_factor_A = [int(blah * 100) for blah in df_A[var1]]
                discrete_factor_B = [int(blah * 100) for blah in df_B[var2]]
                corr_coeff = round(stats.pearsonr(discrete_factor_A, discrete_factor_B)[0], 2)
                array = []
                for k in k_array:
                    array.append(mutual_info_classif(X=np.array(discrete_factor_A).reshape(-1, 1),
                                                     y=np.array(discrete_factor_B), discrete_features=True,
                                                     n_neighbors=k, copy=True, random_state=42)[0])

            sns.set_style("whitegrid")
            plt.plot(k_array, array, marker='x', color='blue', label=r'$\rho={}$'.format(corr_coeff))
            plt.axhline(-.5 * (np.log(1 - (stats.pearsonr(np.array(df_A[var1]), np.array(df_B[var2]))[0] ** 2))),
                        label='exact relationship if it were a bivariate Gaussian', color='blue', ls='--', alpha=.7)
            plt.axhline(0, color='black', ls='-')
            plt.title(r'{} {}'.format(var1, var2))
            plt.ylim(bottom=0)
            plt.legend()
            plt.xlabel('k')
            plt.ylabel('MI')
            # plt.show()
            plt.savefig(title + ' {}_A {}_B'.format(var1, var2), dpi=300)
            plt.close()


def MI_heatmap_for_one_dataset(k, df_A, df_B, variable_names_A, variable_names_B, filename_title, mult_number, half_matrix, directory):
    # # This is to check the selection of k on the mutual info for same-cell
    # memory = []
    # k_array = np.arange(1, 21)
    # diction = dict()
    # for var1 in variable_names_A:
    #     for var2 in variable_names_B:
    #         if ([var1, var2] not in memory) and (var1 != var2):
    #             diction.update({'{} {}'.format(var1, var2) :[mutual_info_regression(X=np.array(df_A[var1]).reshape(-1, 1), y=np.array(df_B[var2]),
    #                                                                                 discrete_features=False, n_neighbors=k1, copy=True,
    #                                                                                 random_state=42)[0] for k1 in k_array]})
    #             memory.append([var1, var2])
    #             memory.append([var2, var1])
    #
    # for key, val in diction.items():
    #     plt.plot(k_array, val, label=str(key))
    # plt.legend()
    # plt.show()
    # plt.close()
    #
    # exit()

    MI, corr_coeff = calculate_discrete_and_continuous_MIs(df_A=df_A[variable_names_A], df_B=df_B[variable_names_B],
                                                           variable_names_A=variable_names_A, variable_names_B=variable_names_B, k=k)

    MI = MI * mult_number

    plt.figure(figsize=(12.7, 7.5))
    if half_matrix == True:
        # so that the color scale corresponds to the subdiagonal heatmap and not the whole heatmap

        # This is to plot only the subdiagonal heatmap
        mask = np.zeros_like(MI)
        mask[np.triu_indices_from(mask)] = True

        vals_array = []
        for ind1 in range(1, len(variable_names_A)):
            for ind2 in range(ind1):
                vals_array.append(MI[variable_names_A[ind1]].loc[variable_names_A[ind2]])
        vals_array = np.array(vals_array)

        vmax = np.max(vals_array)
        vmin = np.min(vals_array)

        sns.heatmap(data=MI, annot=True, fmt='.0f', mask=mask, vmax=vmax, vmin=vmin)
    else:
        sns.heatmap(data=MI, annot=True, fmt='.0f')
    plt.title(filename_title)
    plt.savefig(directory + '/' + filename_title, dpi=300)
    # plt.show()
    plt.close()


def MI_heatmaps_for_three_datasets(k, sis_df_A, sis_df_B, non_sis_df_A, non_sis_df_B, con_df_A, con_df_B, variable_names_A, variable_names_B, filename_title, mult_number, directory):
    MI_sis, corr_coeff_sis = calculate_discrete_and_continuous_MIs(df_A=sis_df_A[variable_names_A], df_B=sis_df_B[variable_names_B],
                                                                   variable_names_A=variable_names_A, variable_names_B=variable_names_B, k=k)
    MI_non_sis, corr_coeff_non_sis = calculate_discrete_and_continuous_MIs(df_A=non_sis_df_A[variable_names_A], df_B=non_sis_df_B[variable_names_B],
                                                                           variable_names_A=variable_names_A, variable_names_B=variable_names_B, k=k)
    MI_con, corr_coeff_con = calculate_discrete_and_continuous_MIs(df_A=con_df_A[variable_names_A], df_B=con_df_B[variable_names_B],
                                                                   variable_names_A=variable_names_A, variable_names_B=variable_names_B, k=k)

    MI_sis = MI_sis * mult_number
    MI_non_sis = MI_non_sis * mult_number
    MI_con = MI_con * mult_number

    vmin = np.min(np.array([np.min(MI_sis.min()), np.min(MI_non_sis.min()), np.min(MI_con.min())]))
    vmax = np.max(np.array([np.max(MI_sis.max()), np.max(MI_non_sis.max()), np.max(MI_con.max())]))

    fig, (ax_sis, ax_non_sis, ax_con) = plt.subplots(ncols=3, figsize=(12.7, 7.5))
    fig.subplots_adjust(wspace=0.01)

    sns.heatmap(data=MI_sis, annot=True, ax=ax_sis, cbar=False, vmin=vmin, vmax=vmax, yticklabels=True, fmt='.0f')  #
    ax_sis.set_title('Sister')
    sns.heatmap(data=MI_non_sis, annot=True, ax=ax_non_sis, cbar=False, vmin=vmin, vmax=vmax, yticklabels=False, fmt='.0f')
    ax_non_sis.set_title('Non-Sister')
    sns.heatmap(data=MI_con, annot=True, ax=ax_con, cbar=True, vmin=vmin, vmax=vmax, yticklabels=False, fmt='.0f')  # xticklabels=[]
    ax_con.set_title('Control')
    fig.suptitle(filename_title)
    plt.savefig(directory + '/' + filename_title, dpi=300)
    # plt.show()
    plt.close()


def meta_graphs(all_bacteria, variable_names, new_sis_A, new_sis_B, sis_A, sis_B, non_sis_A, non_sis_B, con_A, con_B, con_ref_A, con_ref_B, A_variable_symbol, B_variable_symbol,
                sis_intra_A, sis_intra_B, non_sis_intra_A, non_sis_intra_B, con_intra_A, con_intra_B, inter_pooled_mom, inter_pooled_daughter, k_same_cell, k_pooled_trap, k_intra, k_inter,
                mult_number_same_cell,
                mult_number_pooled_trap, mult_number_intra, mult_number_inter, how_many_intra_gens, how_many_inter_gens, type_of_mean, pop_mean, start_index, end_index):
    # make the directories where everything will go
    params_filename = 'meta graphs, k are {}_same {}_pooled_trap {}_intra {}_inter'.format(k_same_cell, k_pooled_trap, k_intra, k_inter)
    try:
        # Create target Directory
        os.mkdir(params_filename)
        print("Directory ", params_filename, " Created ")
    except FileExistsError:
        print("Directory ", params_filename, " already exists")

    if type_of_mean == 'global mean':
        # subtracting the global averages
        new_sis_A = {key: subtract_global_averages(df=val, columns_names=val.columns, pop_mean=pop_mean) for key, val in new_sis_A.items()}
        new_sis_B = {key: subtract_global_averages(df=val, columns_names=val.columns, pop_mean=pop_mean) for key, val in new_sis_B.items()}
        non_sis_A = {key: subtract_global_averages(df=val, columns_names=val.columns, pop_mean=pop_mean) for key, val in non_sis_A.items()}
        non_sis_B = {key: subtract_global_averages(df=val, columns_names=val.columns, pop_mean=pop_mean) for key, val in non_sis_B.items()}
        con_A = {key: subtract_global_averages(df=val, columns_names=val.columns, pop_mean=pop_mean) for key, val in con_A.items()}
        con_B = {key: subtract_global_averages(df=val, columns_names=val.columns, pop_mean=pop_mean) for key, val in con_B.items()}

        all_bacteria = all_bacteria[variable_names] - pop_mean[variable_names]

        sis_intra_A = [sis_intra_A[int(index)][variable_names] - pop_mean[variable_names] for
                       index in how_many_intra_gens]
        sis_intra_B = [sis_intra_B[int(index)][variable_names] - pop_mean[variable_names] for
                       index in how_many_intra_gens]
        non_sis_intra_A = [
            non_sis_intra_A[int(index)][variable_names] - pop_mean[variable_names] for
            index in how_many_intra_gens]
        non_sis_intra_B = [
            non_sis_intra_B[int(index)][variable_names] - pop_mean[variable_names] for
            index in how_many_intra_gens]
        con_intra_A = [con_intra_A[int(index)][variable_names] - pop_mean[variable_names] for
                       index in how_many_intra_gens]
        con_intra_B = [con_intra_B[int(index)][variable_names] - pop_mean[variable_names] for
                       index in how_many_intra_gens]

        inter_pooled_mom = [inter_pooled_mom[int(index)][variable_names] - pop_mean[variable_names] for index in how_many_inter_gens]
        inter_pooled_daughter = [inter_pooled_daughter[int(index)][variable_names] - pop_mean[variable_names] for index in how_many_inter_gens]
    elif type_of_mean == 'trap mean':
        # subtracting the trap averages
        con_A = {key: subtract_trap_averages_control(val, val.columns, ref_key, sis_A, sis_B, non_sis_A, non_sis_B, start_index=start_index, end_index=end_index) for key, val, ref_key
                 in zip(con_A.keys(), con_A.values(), con_ref_A.values())}
        con_B = {key: subtract_trap_averages_control(val, val.columns, ref_key, sis_A, sis_B, non_sis_A, non_sis_B, start_index=start_index, end_index=end_index) for key, val, ref_key
                 in zip(con_B.keys(), con_B.values(), con_ref_B.values())}
        new_sis_A = {keyA: subtract_trap_averages(df_main=valA, df_other=valB, columns_names=valA.columns, start_index=start_index, end_index=end_index) for keyA, valA, keyB, valB in
                     zip(new_sis_A.keys(), new_sis_A.values(), new_sis_B.keys(), new_sis_B.values())}
        new_sis_B = {keyB: subtract_trap_averages(df_main=valB, df_other=valA, columns_names=valB.columns, start_index=start_index, end_index=end_index) for keyB, valB, keyA, valA in
                     zip(new_sis_B.keys(), new_sis_B.values(), new_sis_A.keys(), new_sis_A.values())}
        non_sis_A = {keyA: subtract_trap_averages(df_main=valA, df_other=valB, columns_names=valA.columns, start_index=start_index, end_index=end_index) for keyA, valA, keyB, valB in
                     zip(non_sis_A.keys(), non_sis_A.values(), non_sis_B.keys(), non_sis_B.values())}
        non_sis_B = {keyB: subtract_trap_averages(df_main=valB, df_other=valA, columns_names=valB.columns, start_index=start_index, end_index=end_index) for keyB, valB, keyA, valA in
                     zip(non_sis_B.keys(), non_sis_B.values(), non_sis_A.keys(), non_sis_A.values())}

        all_bacteria_new = pd.DataFrame(columns=variable_names)
        for var in variable_names:
            all_bacteria_new[var] = all_bacteria[var] - all_bacteria['trap_avg_' + var]
        all_bacteria = all_bacteria_new

        sis_intra_A = [sis_intra_A[int(index)][variable_names] - sis_intra_A[int(index)][['trap_avg_' + var for var in variable_names]].values for
                       index in how_many_intra_gens]
        sis_intra_B = [sis_intra_B[int(index)][variable_names] - sis_intra_B[int(index)][['trap_avg_' + var for var in variable_names]].values for
                       index in how_many_intra_gens]
        non_sis_intra_A = [non_sis_intra_A[int(index)][variable_names] - non_sis_intra_A[int(index)][['trap_avg_' + var for var in variable_names]].values for
                           index in how_many_intra_gens]
        non_sis_intra_B = [non_sis_intra_B[int(index)][variable_names] - non_sis_intra_B[int(index)][['trap_avg_' + var for var in variable_names]].values for
                           index in how_many_intra_gens]
        con_intra_A = [con_intra_A[int(index)][variable_names] - con_intra_A[int(index)][['trap_avg_' + var for var in variable_names]].values for
                       index in how_many_intra_gens]
        con_intra_B = [con_intra_B[int(index)][variable_names] - con_intra_B[int(index)][['trap_avg_' + var for var in variable_names]].values for
                       index in how_many_intra_gens]

        inter_pooled_mom = [inter_pooled_mom[int(index)][variable_names] - inter_pooled_mom[int(index)][['trap_avg_' + var for var in variable_names]].values for index in how_many_inter_gens]
        inter_pooled_daughter = [inter_pooled_daughter[int(index)][variable_names] - inter_pooled_daughter[int(index)][['trap_avg_' + var for var in variable_names]].values for index in
                                 how_many_inter_gens]
    elif type_of_mean == 'traj mean':
        # subtracting the trajectory averages
        new_sis_A = {key: subtract_traj_averages(df=val, columns_names=val.columns, start_index=start_index, end_index=end_index) for key, val in zip(new_sis_A.keys(), new_sis_A.values())}
        new_sis_B = {key: subtract_traj_averages(df=val, columns_names=val.columns, start_index=start_index, end_index=end_index) for key, val in zip(new_sis_B.keys(), new_sis_B.values())}
        non_sis_A = {key: subtract_traj_averages(df=val, columns_names=val.columns, start_index=start_index, end_index=end_index) for key, val in zip(non_sis_A.keys(), non_sis_A.values())}
        non_sis_B = {key: subtract_traj_averages(df=val, columns_names=val.columns, start_index=start_index, end_index=end_index) for key, val in zip(non_sis_B.keys(), non_sis_B.values())}
        con_A = {key: subtract_traj_averages(df=val, columns_names=val.columns, start_index=start_index, end_index=end_index) for key, val in zip(con_A.keys(), con_A.values())}
        con_B = {key: subtract_traj_averages(df=val, columns_names=val.columns, start_index=start_index, end_index=end_index) for key, val in zip(con_B.keys(), con_B.values())}

        all_bacteria_new = pd.DataFrame(columns=variable_names)
        for var in variable_names:
            all_bacteria_new[var] = all_bacteria[var] - all_bacteria['traj_avg_' + var]
        all_bacteria = all_bacteria_new

        sis_intra_A = [sis_intra_A[int(index)][variable_names] - sis_intra_A[int(index)][['traj_avg_' + var for var in variable_names]].values for
                       index in how_many_intra_gens]
        sis_intra_B = [sis_intra_B[int(index)][variable_names] - sis_intra_B[int(index)][['traj_avg_' + var for var in variable_names]].values for
                       index in how_many_intra_gens]
        non_sis_intra_A = [
            non_sis_intra_A[int(index)][variable_names] - non_sis_intra_A[int(index)][['traj_avg_' + var for var in variable_names]].values for
            index in how_many_intra_gens]
        non_sis_intra_B = [
            non_sis_intra_B[int(index)][variable_names] - non_sis_intra_B[int(index)][['traj_avg_' + var for var in variable_names]].values for
            index in how_many_intra_gens]
        con_intra_A = [con_intra_A[int(index)][variable_names] - con_intra_A[int(index)][['traj_avg_' + var for var in variable_names]].values for
                       index in how_many_intra_gens]
        con_intra_B = [con_intra_B[int(index)][variable_names] - con_intra_B[int(index)][['traj_avg_' + var for var in variable_names]].values for
                       index in how_many_intra_gens]

        inter_pooled_mom = [inter_pooled_mom[int(index)][variable_names] - inter_pooled_mom[int(index)][['traj_avg_' + var for var in variable_names]].values for index in how_many_inter_gens]
        inter_pooled_daughter = [inter_pooled_daughter[int(index)][variable_names] - inter_pooled_daughter[int(index)][['traj_avg_' + var for var in variable_names]].values for index in
                                 how_many_inter_gens]
    elif type_of_mean == 'measurement':
        # we don't need to do anything since the dictionaries already come in measurement form
        pass
    else:
        print('type of mean is ', type_of_mean, ' which is not global/trap/traj mean or measurement')

    """ Where the graphs are going in
    """
    mean_filename = params_filename + '/' + type_of_mean
    os.makedirs(mean_filename)

    """ This is for the Same-Cell comparison, the diagonal is H(X) = I(X,X) >= I(X,Y). Maybe it is better to make two heatmaps: one for the entropies
         and the other for the mutual information... There's a problem with the Generationtime because it is supposed to be as strong as the other H(X)
         and supposed to be very strongly Mutual Info to with the fold_growth...
        """
    MI_heatmap_for_one_dataset(k=k_same_cell, df_A=all_bacteria, df_B=all_bacteria, variable_names_A=variable_names, variable_names_B=variable_names,
                               filename_title='Mutual Information x {} of Same-Cell with {} values and k={}'.format(mult_number_same_cell,
                                                                                                                    type_of_mean,
                                                                                                                    k_same_cell),
                               mult_number=mult_number_same_cell, half_matrix=True, directory=mean_filename)

    """ This is for the pooled trap comparison
    """
    pooled_sis = get_pooled_trap_df(A_df=new_sis_A, B_df=new_sis_B, variable_names=variable_names, A_variable_symbol=A_variable_symbol,
                                    B_variable_symbol=B_variable_symbol)
    pooled_non_sis = get_pooled_trap_df(A_df=non_sis_A, B_df=non_sis_B, variable_names=variable_names, A_variable_symbol=A_variable_symbol,
                                        B_variable_symbol=B_variable_symbol)
    pooled_con = get_pooled_trap_df(A_df=con_A, B_df=con_B, variable_names=variable_names, A_variable_symbol=A_variable_symbol,
                                    B_variable_symbol=B_variable_symbol)
    MI_heatmaps_for_three_datasets(k=k_pooled_trap, sis_df_A=pooled_sis, sis_df_B=pooled_sis, non_sis_df_A=pooled_non_sis, non_sis_df_B=pooled_non_sis,
                                   con_df_A=pooled_con, con_df_B=pooled_con, variable_names_A=A_variable_symbol, variable_names_B=B_variable_symbol,
                                   filename_title='Mutual Information x {} of pooled traps with {} values and k={}'.format(mult_number_pooled_trap,
                                                                                                                           type_of_mean,
                                                                                                                           k_pooled_trap),
                                   mult_number=mult_number_pooled_trap, directory=mean_filename)

    """ This is for the intra-generational comparison for the first 7 generations 
    """
    for index in how_many_intra_gens:
        MI_heatmaps_for_three_datasets(k=k_intra, sis_df_A=sis_intra_A[index], sis_df_B=sis_intra_B[index], non_sis_df_A=non_sis_intra_A[index],
                                       non_sis_df_B=non_sis_intra_B[index],
                                       con_df_A=con_intra_A[index], con_df_B=con_intra_B[index], variable_names_A=variable_names,
                                       variable_names_B=variable_names,
                                       filename_title='Mutual Information x {} of intra-gen {} with {} values and k={}'.format(mult_number_intra,
                                                                                                                               index, type_of_mean,
                                                                                                                               k_intra),
                                       mult_number=mult_number_intra, directory=mean_filename)

    """ This is for the inter-generational comparison for the first 7 generations 
        """
    for index in how_many_inter_gens:
        MI_heatmap_for_one_dataset(k=k_inter, df_A=inter_pooled_mom[index], df_B=inter_pooled_daughter[index], variable_names_A=variable_names,
                                   variable_names_B=variable_names,
                                   filename_title='Mutual Information x {} of inter-gen {} with {} values and k={}'.format(mult_number_inter,
                                                                                                                           index, type_of_mean,
                                                                                                                           k_inter),
                                   mult_number=mult_number_inter, half_matrix=False, directory=mean_filename)


def k_selection(df_A, df_B, variable_names_A, variable_names_B, start, end, number_of_points_in_graph, type_of_mean, mult_number, directory):
    # make the directories where everything will go
    params_filename = 'k_selection for {}'.format(type_of_mean)
    try:
        # Create target Directory
        os.mkdir(params_filename)
        print("Directory ", params_filename, " Created ")
    except FileExistsError:
        print("Directory ", params_filename, " already exists")

    k_array = np.arange(start, end, int(np.ceil((end - start) / number_of_points_in_graph)))
    print(len(k_array), number_of_points_in_graph, k_array)
    MI_corr_coeff_array = []
    for k in k_array:
        print(k)
        MI, _ = calculate_discrete_and_continuous_MIs(df_A=df_A[variable_names_A], df_B=df_B[variable_names_B], variable_names_A=variable_names_A,
                                                      variable_names_B=variable_names_B, k=k)
        print(MI)
        MI = MI * mult_number
        print(MI)
        MI_corr_coeff_array.append(MI)
    # MI_corr_coeff_array = np.array([calculate_discrete_and_continuous_MIs(df_A=df_A[variable_names_A], df_B=df_B[variable_names_B], variable_names_A=variable_names_A, variable_names_B=variable_names_B, k=k) for k in k_array])
    # print(MI_corr_coeff_array)

    plotting_array = dict()
    for varA in variable_names_A:
        for varB in variable_names_B:
            plotting_array.update({'{}_A, {}_B'.format(varA, varB): np.array([MI[varA].loc[varB] for MI in MI_corr_coeff_array])})

    plt.figure(figsize=(12.7, 7.5))
    sns.set_style('whitegrid')
    for key in plotting_array.keys():
        plt.plot(k_array, plotting_array[key], label=key)
    plt.legend()
    plt.xlabel('k')
    plt.ylabel('MI')
    plt.title('MI depending on k')
    plt.show()
    # plt.savefig(directory + '/' + 'MI depending on k', dpi=300)
    plt.close()
    exit()


def discrete_trap_number_MI(dataset, variables, k):  # do this with the all_bacteria dataset to see how each variable gets the simpsons effect
    all_bacteria = dataset.copy()
    MI_df = pd.DataFrame(columns=variables, index=['global', 'trap', 'traj', 'rand'], dtype=float)
    for var1 in variables:
        MI_df[var1].loc['global'] = mutual_info_regression(X=np.array(all_bacteria['trap_ID']).reshape(-1, 1),
                                                           y=np.array(all_bacteria[var1] - all_bacteria[var1].mean()),
                                                           discrete_features=False, n_neighbors=k, copy=True, random_state=42)[0]

        MI_df[var1].loc['trap'] = mutual_info_regression(X=np.array(all_bacteria['trap_ID']).reshape(-1, 1),
                                                         y=np.array(all_bacteria[var1] - all_bacteria['trap_avg_' + var1]),
                                                         discrete_features=False, n_neighbors=k, copy=True, random_state=42)[0]

        MI_df[var1].loc['traj'] = mutual_info_regression(X=np.array(all_bacteria['trap_ID']).reshape(-1, 1),
                                                         y=np.array(all_bacteria[var1] - all_bacteria['traj_avg_' + var1]),
                                                         discrete_features=False, n_neighbors=k, copy=True, random_state=42)[0]

        MI_df[var1].loc['rand'] = mutual_info_regression(X=np.array(all_bacteria['trap_ID']).reshape(-1, 1),
                                                         y=np.random.normal(34, 6, size=len(all_bacteria)),
                                                         discrete_features=False, n_neighbors=k, copy=True, random_state=42)[0]

    return MI_df


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
    inter_pooled_mom, inter_pooled_daughter = Population.mother_dfs.copy(), Population.daughter_dfs.copy()
    # mom = add_the_added_length(dictionary=mom)
    # daug = add_the_added_length(dictionary=daug)
    sis_A, sis_B = Sister.A_dict.copy(), Sister.B_dict.copy()
    # sis_A = add_the_added_length(dictionary=sis_A)
    # sis_B = add_the_added_length(dictionary=sis_B)
    non_sis_A, non_sis_B = Nonsister.A_dict.copy(), Nonsister.B_dict.copy()
    # non_sis_A = add_the_added_length(dictionary=non_sis_A)
    # non_sis_B = add_the_added_length(dictionary=non_sis_B)
    con_A, con_B = Control.A_dict.copy(), Control.B_dict.copy()
    # con_A = add_the_added_length(dictionary=con_A)
    # con_B = add_the_added_length(dictionary=con_B)
    con_ref_A, con_ref_B = Control.reference_A_dict.copy(), Control.reference_B_dict.copy()
    # env_sis_intra_A, env_sis_intra_B = Env_Sister.A_intra_gen_bacteria.copy(), Env_Sister.B_intra_gen_bacteria.copy()
    sis_intra_A, sis_intra_B = Sister.A_intra_gen_bacteria.copy(), Sister.B_intra_gen_bacteria.copy()
    # sis_intra_A = [
    #     pd.concat([sis_intra_A[ind], pd.Series(sis_intra_A[ind]['length_final'] - sis_intra_A[ind]['length_birth'], name='added_length')], axis=1) for
    #     ind in range(len(sis_intra_A))]
    # sis_intra_B = [
    #     pd.concat([sis_intra_B[ind], pd.Series(sis_intra_B[ind]['length_final'] - sis_intra_B[ind]['length_birth'], name='added_length')], axis=1) for
    #     ind in range(len(sis_intra_B))]
    non_sis_intra_A, non_sis_intra_B = Nonsister.A_intra_gen_bacteria.copy(), Nonsister.B_intra_gen_bacteria.copy()
    # non_sis_intra_A = [
    #     pd.concat([non_sis_intra_A[ind], pd.Series(non_sis_intra_A[ind]['length_final'] - non_sis_intra_A[ind]['length_birth'], name='added_length')],
    #               axis=1) for
    #     ind in range(len(non_sis_intra_A))]
    # non_sis_intra_B = [
    #     pd.concat([non_sis_intra_B[ind], pd.Series(non_sis_intra_B[ind]['length_final'] - non_sis_intra_B[ind]['length_birth'], name='added_length')],
    #               axis=1) for
    #     ind in range(len(non_sis_intra_B))]
    # env_nonsis_intra_A, env_nonsis_intra_B = Env_Nonsister.A_intra_gen_bacteria.copy(), Env_Nonsister.B_intra_gen_bacteria.copy()
    con_intra_A, con_intra_B = Control.A_intra_gen_bacteria.copy(), Control.B_intra_gen_bacteria.copy()
    # con_intra_A = [
    #     pd.concat([con_intra_A[ind], pd.Series(con_intra_A[ind]['length_final'] - con_intra_A[ind]['length_birth'], name='added_length')], axis=1) for
    #     ind in range(len(con_intra_A))]
    # con_intra_B = [
    #     pd.concat([con_intra_B[ind], pd.Series(con_intra_B[ind]['length_final'] - con_intra_B[ind]['length_birth'], name='added_length')], axis=1) for
    #     ind in range(len(con_intra_B))]

    # getting the all bacteria dataframe
    all_bacteria = Population.all_bacteria
    # all_bacteria['added_length'] = pd.Series(all_bacteria['length_final'] - all_bacteria['length_birth'], name='added_length')
    # all_bacteria = all_bacteria.drop(index=np.where(all_bacteria['added_length'] < 0)[0])
    pop_mean = all_bacteria.mean()

    log_vars = ['generationtime', 'length_birth', 'length_final', 'growth_rate', 'fold_growth', 'added_length']
    variable_names = ['generationtime', 'length_birth', 'length_final', 'growth_rate', 'fold_growth', 'division_ratio', 'added_length']

    # # the Histograms to check if variables are normal or lognormal
    # for var in variable_names:
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

    #### It is the log of delta addded length!

    # so that every dataset has 88 sets
    np.random.seed(42)
    sis_keys = np.random.choice(list(sis_A.keys()), size=88, replace=False)
    new_sis_A = dict([(key, sis_A[key]) for key in sis_keys])
    new_sis_B = dict([(key, sis_B[key]) for key in sis_keys])

    A_variable_symbol = [r'$\ln(\tau)_A$', r'$\ln(x(0))_A$', r'$\ln(x(\tau))_A$', r'$\ln(\alpha)_A$', r'$\ln(\phi)_A$', r'$f_A$', r'$\Delta_A$']
    B_variable_symbol = [r'$\ln(\tau)_B$', r'$\ln(x(0))_B$', r'$\ln(x(\tau))_B$', r'$\ln(\alpha)_B$', r'$\ln(\phi)_B$', r'$f_B$', r'$\Delta_B$']

    print(len(np.unique(np.around(all_bacteria['generationtime'], decimals=2))))
    print(np.unique(np.around(all_bacteria['generationtime'], decimals=2)))
    exit()

    # k_array = np.arange(1, 902, 50)
    #
    # discreteness = [mutual_info_classif(X=np.array([np.int(some) for some in all_bacteria['generationtime'] * 100]).reshape(-1, 1), y=np.array([np.int(some) for some in all_bacteria['generationtime'] * 100]), discrete_features=True, n_neighbors=k, copy=True, random_state=42)[0] for k in k_array]
    # diction = dict([(var, [mutual_info_regression(X=np.array(all_bacteria[var]).reshape(-1, 1), y=np.array(all_bacteria[var]), discrete_features=False, n_neighbors=k, copy=True, random_state=42)[0] for k in k_array]) for var in variable_names])
    #
    # print(diction)
    #
    # sns.set_style("whitegrid")
    # for key, val in diction.items():
    #     plt.plot(k_array, val, marker='x', label=str(key))
    # plt.plot(k_array, discreteness, label='discrete')
    # plt.axhline(0, color='black', ls='-')
    # plt.ylim(bottom=0)
    # plt.legend()
    # plt.xlabel('k')
    # plt.ylabel('MI')
    # plt.show()
    # plt.close()
    # exit()

    # print(all_bacteria.columns)
    # MI = discrete_trap_number_MI(dataset=all_bacteria, variables=variable_names, k=3)
    # np.random.seed(42)
    # # print(all_bacteria)
    #
    # for var in variable_names:
    #     print(var)
    #     rands = np.random.choice(all_bacteria['trap_ID'], size=10, replace=False)
    #
    #     print(rands)
    #     index=0
    #     for index in range(len(rands)):
    #         print(rands[index])
    #         indexes = np.where([ind for ind in range(len(all_bacteria['trap_ID'])) if all_bacteria['trap_ID'].iloc[ind] == rands[index]])
    #         sns.distplot(all_bacteria.iloc[indexes][var], rug=True, label=rands[index], kde=True, hist=False)
    #
    #     plt.xlabel(var)
    #     plt.ylabel('dists')
    #     plt.legend()
    #     plt.show()
    #     plt.close()
    #
    #     index = 0
    #     for index in range(len(rands)):
    #         print(rands[index])
    #         indexes = np.where([ind for ind in range(len(all_bacteria['trap_ID'])) if all_bacteria['trap_ID'].iloc[ind] == rands[index]])
    #         sns.distplot(all_bacteria.iloc[indexes][var]-all_bacteria.iloc[indexes]['trap_avg_'+var], rug=True, label=rands[index], kde=True, hist=False)
    #
    #     plt.xlabel(var)
    #     plt.ylabel('dists')
    #     plt.legend()
    #     plt.show()
    #     plt.close()
    #
    #     index = 0
    #     for index in range(len(rands)):
    #         print(rands[index])
    #         indexes = np.where([ind for ind in range(len(all_bacteria['trap_ID'])) if all_bacteria['trap_ID'].iloc[ind] == rands[index]])
    #         sns.distplot(all_bacteria.iloc[indexes][var] - all_bacteria.iloc[indexes]['traj_avg_' + var], rug=True, label=rands[index], kde=True, hist=False)
    #
    #     plt.xlabel(var)
    #     plt.ylabel('dists')
    #     plt.legend()
    #     plt.show()
    #     plt.close()
    #
    # sns.heatmap(data=MI, annot=True)
    # plt.show()
    # plt.close()
    #
    # global_MIs = [discrete_trap_number_MI(dataset=all_bacteria, variables=variable_names, k=k) for k in np.arange(1, 101, 10)]
    # # trap_MIs = [discrete_trap_number_MI(dataset=all_bacteria, variables=variable_names, k=k).loc['trap'] for k in np.arange(1, 31)]
    # # traj_MIs = [discrete_trap_number_MI(dataset=all_bacteria, variables=variable_names, k=k).loc['traj'] for k in np.arange(1, 31)]
    # plt.plot(np.arange(1, 101, 10), [blah.loc['global'] for blah in global_MIs], marker='.')
    # plt.plot(np.arange(1, 101, 10), [blah.loc['trap'] for blah in global_MIs], marker='+')
    # plt.plot(np.arange(1, 101, 10), [blah.loc['traj'] for blah in global_MIs], marker='*')
    # # plt.legend()
    # plt.xlabel('k')
    # plt.ylabel('MI')
    # plt.show()
    # # plt.plot(global_MIs, label='global')
    # # plt.plot(global_MIs, label='global')
    # exit()

    start_index = None
    end_index = None

    # k_selection(df_A=all_bacteria, df_B=all_bacteria, variable_names_A=variable_names, variable_names_B=variable_names, start=1, end=11, number_of_points_in_graph=5, type_of_mean='measurement', mult_number=100000, directory='blah')
    # exit()

    meta_graphs(all_bacteria, variable_names, new_sis_A, new_sis_B, sis_A, sis_B, non_sis_A, non_sis_B, con_A, con_B, con_ref_A, con_ref_B,
                A_variable_symbol,
                B_variable_symbol, sis_intra_A, sis_intra_B, non_sis_intra_A, non_sis_intra_B, con_intra_A, con_intra_B, inter_pooled_mom,
                inter_pooled_daughter,
                k_same_cell=3, k_pooled_trap=50, k_intra=18, k_inter=18, mult_number_same_cell=100, mult_number_pooled_trap=10000,
                mult_number_intra=10000,
                mult_number_inter=10000, how_many_intra_gens=np.arange(7), how_many_inter_gens=np.arange(6), type_of_mean='measurement',
                pop_mean=pop_mean, start_index=start_index, end_index=end_index)

    meta_graphs(all_bacteria, variable_names, new_sis_A, new_sis_B, sis_A, sis_B, non_sis_A, non_sis_B, con_A, con_B, con_ref_A, con_ref_B,
                A_variable_symbol,
                B_variable_symbol, sis_intra_A, sis_intra_B, non_sis_intra_A, non_sis_intra_B, con_intra_A, con_intra_B, inter_pooled_mom,
                inter_pooled_daughter,
                k_same_cell=3, k_pooled_trap=50, k_intra=18, k_inter=18, mult_number_same_cell=100, mult_number_pooled_trap=10000,
                mult_number_intra=10000,
                mult_number_inter=10000, how_many_intra_gens=np.arange(7), how_many_inter_gens=np.arange(6), type_of_mean='global mean',
                pop_mean=pop_mean, start_index=start_index, end_index=end_index)

    meta_graphs(all_bacteria, variable_names, new_sis_A, new_sis_B, sis_A, sis_B, non_sis_A, non_sis_B, con_A, con_B, con_ref_A, con_ref_B, A_variable_symbol,
                B_variable_symbol, sis_intra_A, sis_intra_B, non_sis_intra_A, non_sis_intra_B, con_intra_A, con_intra_B, inter_pooled_mom, inter_pooled_daughter,
                k_same_cell=3, k_pooled_trap=50, k_intra=18, k_inter=18, mult_number_same_cell=100, mult_number_pooled_trap=10000, mult_number_intra=10000,
                mult_number_inter=10000, how_many_intra_gens=np.arange(7), how_many_inter_gens=np.arange(6), type_of_mean='trap mean',
                pop_mean=pop_mean, start_index=start_index, end_index=end_index)

    meta_graphs(all_bacteria, variable_names, new_sis_A, new_sis_B, sis_A, sis_B, non_sis_A, non_sis_B, con_A, con_B, con_ref_A, con_ref_B,
                A_variable_symbol,
                B_variable_symbol, sis_intra_A, sis_intra_B, non_sis_intra_A, non_sis_intra_B, con_intra_A, con_intra_B, inter_pooled_mom,
                inter_pooled_daughter,
                k_same_cell=3, k_pooled_trap=50, k_intra=18, k_inter=18, mult_number_same_cell=100, mult_number_pooled_trap=10000,
                mult_number_intra=10000,
                mult_number_inter=10000, how_many_intra_gens=np.arange(7), how_many_inter_gens=np.arange(6), type_of_mean='traj mean',
                pop_mean=pop_mean, start_index=start_index, end_index=end_index)

    exit()

    # k_array = np.arange(1, 80)

    MI_array = []
    corr_coeff_array = []
    for intra_A, intra_B in zip(sis_intra_A, sis_intra_B):
        # plot_and_save_k_selection(df_A=intra_A, df_B=intra_B, variable_names_A=variable_names, variable_names_B=variable_names, k_array=k_array,
        #                           title='intra number {} k selection'.format(ind))
        MI, corr_coeff = calculate_discrete_and_continuous_MIs(df_A=intra_A, df_B=intra_B, variable_names_A=variable_names,
                                                               variable_names_B=variable_names, k=18)
        MI_array.append(MI)
        corr_coeff_array.append(corr_coeff)

    mult_number = 100000

    vmin = np.min([np.min(np.array(MI * mult_number)) for MI in MI_array])
    vmax = np.max([np.max(np.array(MI * mult_number)) for MI in MI_array])
    # vmin = np.min(np.array([np.min(MI_sis.min()), np.min(MI_non_sis.min()), np.min(MI_con.min())]))
    # vmax = np.max(np.array([np.max(MI_sis.max()), np.max(MI_non_sis.max()), np.max(MI_con.max())]))

    ind = 0
    for MI, corr_coeff in zip(MI_array, corr_coeff_array):
        MI = MI * mult_number

        sns.heatmap(MI, annot=True, vmin=vmin, vmax=vmax, fmt='.0f')
        plt.title('k=18, Mutual Information of sis intra-gen {} measures'.format(ind))
        # plt.show()
        plt.savefig('k=18, Mutual Information of sis intra-gen {} measures'.format(ind), dpi=300)
        plt.close()
        ind += 1

    MI_array = []
    corr_coeff_array = []
    for intra_A, intra_B in zip(non_sis_intra_A, non_sis_intra_B):
        # plot_and_save_k_selection(df_A=intra_A, df_B=intra_B, variable_names_A=variable_names, variable_names_B=variable_names, k_array=k_array,
        #                           title='intra number {} k selection'.format(ind))
        MI, corr_coeff = calculate_discrete_and_continuous_MIs(df_A=intra_A, df_B=intra_B, variable_names_A=variable_names,
                                                               variable_names_B=variable_names, k=18)
        MI_array.append(MI)
        corr_coeff_array.append(corr_coeff)

    mult_number = 100000

    vmin = np.min([np.min(np.array(MI * mult_number)) for MI in MI_array])
    vmax = np.max([np.max(np.array(MI * mult_number)) for MI in MI_array])
    # vmin = np.min(np.array([np.min(MI_non_sis.min()), np.min(MI_non_non_sis.min()), np.min(MI_con.min())]))
    # vmax = np.max(np.array([np.max(MI_non_sis.max()), np.max(MI_non_non_sis.max()), np.max(MI_con.max())]))

    ind = 0
    for MI, corr_coeff in zip(MI_array, corr_coeff_array):
        MI = MI * mult_number

        sns.heatmap(MI, annot=True, vmin=vmin, vmax=vmax, fmt='.0f')
        plt.title('k=18, Mutual Information of non_sis intra-gen {} measures'.format(ind))
        # plt.show()
        plt.savefig('k=18, Mutual Information of non_sis intra-gen {} measures'.format(ind), dpi=300)
        plt.close()
        ind += 1


if __name__ == '__main__':
    main()
