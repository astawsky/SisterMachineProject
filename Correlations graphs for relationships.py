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


def getLabels(gen_depth):

    label = []

    # sisters and cousins
    for gen in range(gen_depth):
        label.append(str(gen)+'-'+str(gen))

    # aunts and nieces
    for gen in range(gen_depth):
        label.append(str(gen)+'-'+str(gen+1))

    # great aunts and great nieces
    for gen in range(gen_depth):
        label.append(str(gen)+'-'+str(gen+2))

    return label


def getCorrelationPD1(A_dict, B_dict, struct, generation_depth):
    
    columns = struct.A_dict_sis['Sister_Trace_A_0'].keys()

    index_range = np.arange(generation_depth)
    # gen 0 to gen 0 correlation to intialize the dataframe
    A_trace = pd.DataFrame([val_A.iloc[0] for val_A in A_dict.values()], columns=struct.A_dict_sis['Sister_Trace_A_0'].keys())
    B_trace = pd.DataFrame([val_B.iloc[0] for val_B in B_dict.values()], columns=struct.B_dict_sis['Sister_Trace_B_0'].keys())
    pearson_array = dict()
    for col in columns:
        if col == 'division_ratios__f_n':
            pearson = stats.pearsonr(np.abs(A_trace[col] - .5), np.abs(B_trace[col] - .5))[0]
            pearson_array.update({col: pearson})
        else:
            pearson = stats.pearsonr(A_trace[col], B_trace[col])[0]
            pearson_array.update({col: pearson})
    pearson_pd = pd.DataFrame(pearson_array, index=[0])

    # cousin correlation, ie the same generations for generation_depth generations
    for index1, index2 in zip(index_range[1:], index_range[1:]):
        A_trace = pd.DataFrame([val_A.iloc[index1] for val_A in A_dict.values() if index1 < len(val_A)], columns=struct.A_dict_sis[
            'Sister_Trace_A_0'].keys())
        B_trace = pd.DataFrame([val_B.iloc[index2] for val_B in B_dict.values() if index2 < len(val_B)], columns=struct.B_dict_sis[
            'Sister_Trace_B_0'].keys())

        # If we have a "long" generation depth
        if len(A_trace) != len(B_trace):
            A_trace = A_trace[:min(len(A_trace), len(B_trace))]
            B_trace = B_trace[:min(len(A_trace), len(B_trace))]


        pearson_array = dict()
        for col in columns:

            if col == 'division_ratios__f_n':
                pearson = stats.pearsonr(np.abs(A_trace[col] - .5), np.abs(B_trace[col] - .5))[0]
                pearson_array.update({col: pearson})
            else:
                pearson = stats.pearsonr(A_trace[col], B_trace[col])[0]
                pearson_array.update({col: pearson})
            # pearson = stats.pearsonr(A_trace[col], B_trace[col])[0]
            # pearson_array.update({col: pearson})
        pearson_array = pd.DataFrame(pearson_array, index=[0])
        pearson_pd = pearson_pd.append(pearson_array, ignore_index=True)

    index_range = np.arange(generation_depth+1)
    # aunt and niece correlation, differ by 1 gen for generation_depth generations
    for index1, index2 in zip(index_range[:-1], index_range[1:]):
        # A_trace = pd.DataFrame([val_A.iloc[index1] for val_A in A_dict.values()], columns=struct.A_dict_sis['Sister_Trace_A_0'].keys())
        # B_trace = pd.DataFrame([val_B.iloc[index2] for val_B in B_dict.values()], columns=struct.B_dict_sis['Sister_Trace_B_0'].keys())

        # Combining the asymetric displacement on both trajectories, ie. all the examples from when A will be displaced and when B will
        A_trace1 = pd.DataFrame([val_A.iloc[index1] for val_A in A_dict.values() if index1 < len(val_A)], columns=struct.A_dict_sis[
            'Sister_Trace_A_0'].keys())
        B_trace1 = pd.DataFrame([val_B.iloc[index2] for val_B in B_dict.values() if index2 < len(val_B)], columns=struct.B_dict_sis[
            'Sister_Trace_B_0'].keys())

        A_trace2 = pd.DataFrame([val_A.iloc[index2] for val_A in A_dict.values() if index2 < len(val_A)], columns=struct.A_dict_sis[
            'Sister_Trace_A_0'].keys())
        B_trace2 = pd.DataFrame([val_B.iloc[index1] for val_B in B_dict.values() if index1 < len(val_B)], columns=struct.B_dict_sis[
            'Sister_Trace_B_0'].keys())

        A_trace = A_trace1.append(A_trace2)
        B_trace = B_trace1.append(B_trace2)

        # If we have a "long" generation depth
        if len(A_trace) != len(B_trace):
            A_trace = A_trace[:min(len(A_trace), len(B_trace))]
            B_trace = B_trace[:min(len(A_trace), len(B_trace))]

        pearson_array = dict()
        for col in columns:
            if col == 'division_ratios__f_n':
                pearson = stats.pearsonr(np.abs(A_trace[col] - .5), np.abs(B_trace[col] - .5))[0]
                pearson_array.update({col: pearson})
            else:
                pearson = stats.pearsonr(A_trace[col], B_trace[col])[0]
                pearson_array.update({col: pearson})
            # pearson = stats.pearsonr(A_trace[col], B_trace[col])[0]
            # pearson_array.update({col: pearson})
        pearson_array = pd.DataFrame(pearson_array, index=[0])
        pearson_pd = pearson_pd.append(pearson_array, ignore_index=True)

    index_range = np.arange(generation_depth + 2)
    # grand aunt and grand niece correlation, differ by 2 gen for generation_depth generations
    for index1, index2 in zip(index_range[:-2], index_range[2:]):
        # A_trace = pd.DataFrame([val_A.iloc[index1] for val_A in A_dict.values()], columns=struct.A_dict_sis['Sister_Trace_A_0'].keys())
        # B_trace = pd.DataFrame([val_B.iloc[index2] for val_B in B_dict.values()], columns=struct.B_dict_sis['Sister_Trace_B_0'].keys())

        # Combining the asymetric displacement on both trajectories, ie. all the examples from when A will be displaced and when B will
        A_trace1 = pd.DataFrame([val_A.iloc[index1] for val_A in A_dict.values() if index1 < len(val_A)], columns=struct.A_dict_sis[
            'Sister_Trace_A_0'].keys())
        B_trace1 = pd.DataFrame([val_B.iloc[index2] for val_B in B_dict.values() if index2 < len(val_B)], columns=struct.B_dict_sis[
            'Sister_Trace_B_0'].keys())

        A_trace2 = pd.DataFrame([val_A.iloc[index2] for val_A in A_dict.values() if index2 < len(val_A)], columns=struct.A_dict_sis[
            'Sister_Trace_A_0'].keys())
        B_trace2 = pd.DataFrame([val_B.iloc[index1] for val_B in B_dict.values() if index1 < len(val_B)], columns=struct.B_dict_sis[
            'Sister_Trace_B_0'].keys())

        A_trace = A_trace1.append(A_trace2)
        B_trace = B_trace1.append(B_trace2)

        # If we have a "long" generation depth
        if len(A_trace) != len(B_trace):
            A_trace = A_trace[:min(len(A_trace), len(B_trace))]
            B_trace = B_trace[:min(len(A_trace), len(B_trace))]

        pearson_array = dict()
        for col in columns:
            if col == 'division_ratios__f_n':
                pearson = stats.pearsonr(np.abs(A_trace[col] - .5), np.abs(B_trace[col] - .5))[0]
                pearson_array.update({col: pearson})
            else:
                pearson = stats.pearsonr(A_trace[col], B_trace[col])[0]
                pearson_array.update({col: pearson})
            # pearson = stats.pearsonr(A_trace[col], B_trace[col])[0]
            # pearson_array.update({col: pearson})
        pearson_array = pd.DataFrame(pearson_array, index=[0])
        pearson_pd = pearson_pd.append(pearson_array, ignore_index=True)

    return pearson_pd


# With this function we just convert it to standard normal variables, add phi, and then do everything else the same
def getCorrelationPD2(A_dict, B_dict, struct, generation_depth):

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

    # the cycle parameters
    columns = np.array(struct.A_dict_sis['Sister_Trace_A_0'].keys())
    columns = np.append(columns, ['phi'])

    index_range = np.arange(generation_depth)
    # gen 0 to gen 0 correlation to intialize the dataframe
    A_trace = pd.DataFrame([val_A.iloc[0] for val_A in A_dict.values()], columns=struct.A_dict_sis['Sister_Trace_A_0'].keys())
    B_trace = pd.DataFrame([val_B.iloc[0] for val_B in B_dict.values()], columns=struct.B_dict_sis['Sister_Trace_B_0'].keys())

    A_trace['phi'] = A_trace['generationtime'] * A_trace['growth_length']
    B_trace['phi'] = B_trace['generationtime'] * B_trace['growth_length']
    A_trace, B_trace = (A_trace - pop_means) / pop_stds, (B_trace - pop_means) / pop_stds

    pearson_array = dict()
    for col in columns:
        if col == 'division_ratios__f_n':
            pearson = stats.pearsonr(np.abs(A_trace[col] - .5), np.abs(B_trace[col] - .5))[0]
            pearson_array.update({col: pearson})
        else:
            pearson = stats.pearsonr(A_trace[col], B_trace[col])[0]
            pearson_array.update({col: pearson})
    pearson_pd = pd.DataFrame(pearson_array, index=[0])

    # cousin correlation, ie the same generations for generation_depth generations
    for index1, index2 in zip(index_range[1:], index_range[1:]):
        A_trace = pd.DataFrame([val_A.iloc[index1] for val_A in A_dict.values() if index1 < len(val_A)], columns=struct.A_dict_sis[
            'Sister_Trace_A_0'].keys())
        B_trace = pd.DataFrame([val_B.iloc[index2] for val_B in B_dict.values() if index2 < len(val_B)], columns=struct.B_dict_sis[
            'Sister_Trace_B_0'].keys())

        # If we have a "long" generation depth
        if len(A_trace) != len(B_trace):
            A_trace = A_trace[:min(len(A_trace), len(B_trace))]
            B_trace = B_trace[:min(len(A_trace), len(B_trace))]

        # Add phi and convert it to standard normal
        A_trace['phi'] = A_trace['generationtime'] * A_trace['growth_length']
        B_trace['phi'] = B_trace['generationtime'] * B_trace['growth_length']
        A_trace, B_trace = (A_trace - pop_means) / pop_stds, (B_trace - pop_means) / pop_stds

        pearson_array = dict()
        for col in columns:

            if col == 'division_ratios__f_n':
                pearson = stats.pearsonr(np.abs(A_trace[col] - .5), np.abs(B_trace[col] - .5))[0]
                pearson_array.update({col: pearson})
            else:
                pearson = stats.pearsonr(A_trace[col], B_trace[col])[0]
                pearson_array.update({col: pearson})
            # pearson = stats.pearsonr(A_trace[col], B_trace[col])[0]
            # pearson_array.update({col: pearson})
        pearson_array = pd.DataFrame(pearson_array, index=[0])
        pearson_pd = pearson_pd.append(pearson_array, ignore_index=True)

    index_range = np.arange(generation_depth + 1)
    # aunt and niece correlation, differ by 1 gen for generation_depth generations
    for index1, index2 in zip(index_range[:-1], index_range[1:]):
        # A_trace = pd.DataFrame([val_A.iloc[index1] for val_A in A_dict.values()], columns=struct.A_dict_sis['Sister_Trace_A_0'].keys())
        # B_trace = pd.DataFrame([val_B.iloc[index2] for val_B in B_dict.values()], columns=struct.B_dict_sis['Sister_Trace_B_0'].keys())

        # Combining the asymetric displacement on both trajectories, ie. all the examples from when A will be displaced and when B will
        A_trace1 = pd.DataFrame([val_A.iloc[index1] for val_A in A_dict.values() if index1 < len(val_A)], columns=struct.A_dict_sis[
            'Sister_Trace_A_0'].keys())
        B_trace1 = pd.DataFrame([val_B.iloc[index2] for val_B in B_dict.values() if index2 < len(val_B)], columns=struct.B_dict_sis[
            'Sister_Trace_B_0'].keys())

        A_trace2 = pd.DataFrame([val_A.iloc[index2] for val_A in A_dict.values() if index2 < len(val_A)], columns=struct.A_dict_sis[
            'Sister_Trace_A_0'].keys())
        B_trace2 = pd.DataFrame([val_B.iloc[index1] for val_B in B_dict.values() if index1 < len(val_B)], columns=struct.B_dict_sis[
            'Sister_Trace_B_0'].keys())

        A_trace = A_trace1.append(A_trace2)
        B_trace = B_trace1.append(B_trace2)

        # If we have a "long" generation depth
        if len(A_trace) != len(B_trace):
            A_trace = A_trace[:min(len(A_trace), len(B_trace))]
            B_trace = B_trace[:min(len(A_trace), len(B_trace))]

        # Add phi and convert it to standard normal
        A_trace['phi'] = A_trace['generationtime'] * A_trace['growth_length']
        B_trace['phi'] = B_trace['generationtime'] * B_trace['growth_length']
        A_trace, B_trace = (A_trace - pop_means) / pop_stds, (B_trace - pop_means) / pop_stds

        pearson_array = dict()
        for col in columns:
            if col == 'division_ratios__f_n':
                pearson = stats.pearsonr(np.abs(A_trace[col] - .5), np.abs(B_trace[col] - .5))[0]
                pearson_array.update({col: pearson})
            else:
                pearson = stats.pearsonr(A_trace[col], B_trace[col])[0]
                pearson_array.update({col: pearson})
            # pearson = stats.pearsonr(A_trace[col], B_trace[col])[0]
            # pearson_array.update({col: pearson})
        pearson_array = pd.DataFrame(pearson_array, index=[0])
        pearson_pd = pearson_pd.append(pearson_array, ignore_index=True)

    index_range = np.arange(generation_depth + 2)
    # grand aunt and grand niece correlation, differ by 2 gen for generation_depth generations
    for index1, index2 in zip(index_range[:-2], index_range[2:]):
        # A_trace = pd.DataFrame([val_A.iloc[index1] for val_A in A_dict.values()], columns=struct.A_dict_sis['Sister_Trace_A_0'].keys())
        # B_trace = pd.DataFrame([val_B.iloc[index2] for val_B in B_dict.values()], columns=struct.B_dict_sis['Sister_Trace_B_0'].keys())

        # Combining the asymetric displacement on both trajectories, ie. all the examples from when A will be displaced and when B will
        A_trace1 = pd.DataFrame([val_A.iloc[index1] for val_A in A_dict.values() if index1 < len(val_A)], columns=struct.A_dict_sis[
            'Sister_Trace_A_0'].keys())
        B_trace1 = pd.DataFrame([val_B.iloc[index2] for val_B in B_dict.values() if index2 < len(val_B)], columns=struct.B_dict_sis[
            'Sister_Trace_B_0'].keys())

        A_trace2 = pd.DataFrame([val_A.iloc[index2] for val_A in A_dict.values() if index2 < len(val_A)], columns=struct.A_dict_sis[
            'Sister_Trace_A_0'].keys())
        B_trace2 = pd.DataFrame([val_B.iloc[index1] for val_B in B_dict.values() if index1 < len(val_B)], columns=struct.B_dict_sis[
            'Sister_Trace_B_0'].keys())

        A_trace = A_trace1.append(A_trace2)
        B_trace = B_trace1.append(B_trace2)

        # If we have a "long" generation depth
        if len(A_trace) != len(B_trace):
            A_trace = A_trace[:min(len(A_trace), len(B_trace))]
            B_trace = B_trace[:min(len(A_trace), len(B_trace))]

        # Add phi and convert it to standard normal
        A_trace['phi'] = A_trace['generationtime'] * A_trace['growth_length']
        B_trace['phi'] = B_trace['generationtime'] * B_trace['growth_length']
        A_trace, B_trace = (A_trace - pop_means) / pop_stds, (B_trace - pop_means) / pop_stds

        pearson_array = dict()
        for col in columns:
            if col == 'division_ratios__f_n':
                pearson = stats.pearsonr(np.abs(A_trace[col] - .5), np.abs(B_trace[col] - .5))[0]
                pearson_array.update({col: pearson})
            else:
                pearson = stats.pearsonr(A_trace[col], B_trace[col])[0]
                pearson_array.update({col: pearson})
            # pearson = stats.pearsonr(A_trace[col], B_trace[col])[0]
            # pearson_array.update({col: pearson})
        pearson_array = pd.DataFrame(pearson_array, index=[0])
        pearson_pd = pearson_pd.append(pearson_array, ignore_index=True)

    return pearson_pd


# we're not using this one anymore, keeping it here just in case
def getCorrelationPD(A_dict, B_dict, struct):

    columns = struct.A_dict_sis['Sister_Trace_A_0'].keys()

    # same generation correlations
    A_trace = pd.DataFrame([val_A.iloc[0] for val_A in A_dict.values()], columns=struct.A_dict_sis['Sister_Trace_A_0'].keys())
    B_trace = pd.DataFrame([val_B.iloc[0] for val_B in B_dict.values()], columns=struct.B_dict_sis['Sister_Trace_B_0'].keys())
    pearson_array = dict()
    for col in columns:
        if col == 'division_ratios__f_n':
            pearson = stats.pearsonr(np.abs(A_trace[col] - .5), np.abs(B_trace[col]-.5))[0]
            pearson_array.update({col: pearson})
        else:
            pearson = stats.pearsonr(A_trace[col], B_trace[col])[0]
            pearson_array.update({col: pearson})
    pearson_pd = pd.DataFrame(pearson_array, index=[0])

    A_trace = pd.DataFrame([val_A.iloc[1] for val_A in A_dict.values()], columns=struct.A_dict_sis['Sister_Trace_A_0'].keys())
    B_trace = pd.DataFrame([val_B.iloc[1] for val_B in B_dict.values()], columns=struct.B_dict_sis['Sister_Trace_B_0'].keys())
    pearson_array = dict()
    for col in columns:
        if col == 'division_ratios__f_n':
            pearson = stats.pearsonr(np.abs(A_trace[col] - .5), np.abs(B_trace[col]-.5))[0]
            pearson_array.update({col: pearson})
        else:
            pearson = stats.pearsonr(A_trace[col], B_trace[col])[0]
            pearson_array.update({col: pearson})
        # pearson = stats.pearsonr(A_trace[col], B_trace[col])[0]
        # pearson_array.update({col: pearson})
    pearson_array = pd.DataFrame(pearson_array, index=[0])
    pearson_pd = pearson_pd.append(pearson_array, ignore_index=True)

    A_trace = pd.DataFrame([val_A.iloc[2] for val_A in A_dict.values()], columns=struct.A_dict_sis['Sister_Trace_A_0'].keys())
    B_trace = pd.DataFrame([val_B.iloc[2] for val_B in B_dict.values()], columns=struct.B_dict_sis['Sister_Trace_B_0'].keys())
    pearson_array = dict()
    for col in columns:
        if col == 'division_ratios__f_n':
            pearson = stats.pearsonr(np.abs(A_trace[col] - .5), np.abs(B_trace[col]-.5))[0]
            pearson_array.update({col: pearson})
        else:
            pearson = stats.pearsonr(A_trace[col], B_trace[col])[0]
            pearson_array.update({col: pearson})
        # pearson = stats.pearsonr(A_trace[col], B_trace[col])[0]
        # pearson_array.update({col: pearson})
    pearson_array = pd.DataFrame(pearson_array, index=[0])
    pearson_pd = pearson_pd.append(pearson_array, ignore_index=True)

    # 1 generation difference correlations
    A_trace = pd.DataFrame([val_A.iloc[0] for val_A in A_dict.values()], columns=struct.A_dict_sis['Sister_Trace_A_0'].keys())
    B_trace = pd.DataFrame([val_B.iloc[1] for val_B in B_dict.values()], columns=struct.B_dict_sis['Sister_Trace_B_0'].keys())
    pearson_array = dict()
    for col in columns:
        if col == 'division_ratios__f_n':
            pearson = stats.pearsonr(np.abs(A_trace[col] - .5), np.abs(B_trace[col]-.5))[0]
            pearson_array.update({col: pearson})
        else:
            pearson = stats.pearsonr(A_trace[col], B_trace[col])[0]
            pearson_array.update({col: pearson})
        # pearson = stats.pearsonr(A_trace[col], B_trace[col])[0]
        # pearson_array.update({col: pearson})
    pearson_array = pd.DataFrame(pearson_array, index=[0])
    pearson_pd = pearson_pd.append(pearson_array, ignore_index=True)

    A_trace = pd.DataFrame([val_A.iloc[1] for val_A in A_dict.values()], columns=struct.A_dict_sis['Sister_Trace_A_0'].keys())
    B_trace = pd.DataFrame([val_B.iloc[2] for val_B in B_dict.values()], columns=struct.B_dict_sis['Sister_Trace_B_0'].keys())
    pearson_array = dict()
    for col in columns:
        if col == 'division_ratios__f_n':
            pearson = stats.pearsonr(np.abs(A_trace[col] - .5), np.abs(B_trace[col]-.5))[0]
            pearson_array.update({col: pearson})
        else:
            pearson = stats.pearsonr(A_trace[col], B_trace[col])[0]
            pearson_array.update({col: pearson})
        # pearson = stats.pearsonr(A_trace[col], B_trace[col])[0]
        # pearson_array.update({col: pearson})
    pearson_array = pd.DataFrame(pearson_array, index=[0])
    pearson_pd = pearson_pd.append(pearson_array, ignore_index=True)

    A_trace = pd.DataFrame([val_A.iloc[2] for val_A in A_dict.values()], columns=struct.A_dict_sis['Sister_Trace_A_0'].keys())
    B_trace = pd.DataFrame([val_B.iloc[3] for val_B in B_dict.values()], columns=struct.B_dict_sis['Sister_Trace_B_0'].keys())
    pearson_array = dict()
    for col in columns:
        if col == 'division_ratios__f_n':
            pearson = stats.pearsonr(np.abs(A_trace[col] - .5), np.abs(B_trace[col]-.5))[0]
            pearson_array.update({col: pearson})
        else:
            pearson = stats.pearsonr(A_trace[col], B_trace[col])[0]
            pearson_array.update({col: pearson})
        # pearson = stats.pearsonr(A_trace[col], B_trace[col])[0]
        # pearson_array.update({col: pearson})
    pearson_array = pd.DataFrame(pearson_array, index=[0])
    pearson_pd = pearson_pd.append(pearson_array, ignore_index=True)

    # 2 generations difference correlations
    A_trace = pd.DataFrame([val_A.iloc[0] for val_A in A_dict.values()], columns=struct.A_dict_sis['Sister_Trace_A_0'].keys())
    B_trace = pd.DataFrame([val_B.iloc[2] for val_B in B_dict.values()], columns=struct.B_dict_sis['Sister_Trace_B_0'].keys())
    pearson_array = dict()
    for col in columns:
        if col == 'division_ratios__f_n':
            pearson = stats.pearsonr(np.abs(A_trace[col] - .5), np.abs(B_trace[col]-.5))[0]
            pearson_array.update({col: pearson})
        else:
            pearson = stats.pearsonr(A_trace[col], B_trace[col])[0]
            pearson_array.update({col: pearson})
        # pearson = stats.pearsonr(A_trace[col], B_trace[col])[0]
        # pearson_array.update({col: pearson})
    pearson_array = pd.DataFrame(pearson_array, index=[0])
    pearson_pd = pearson_pd.append(pearson_array, ignore_index=True)

    A_trace = pd.DataFrame([val_A.iloc[1] for val_A in A_dict.values()], columns=struct.A_dict_sis['Sister_Trace_A_0'].keys())
    B_trace = pd.DataFrame([val_B.iloc[3] for val_B in B_dict.values()], columns=struct.B_dict_sis['Sister_Trace_B_0'].keys())
    pearson_array = dict()
    for col in columns:
        if col == 'division_ratios__f_n':
            pearson = stats.pearsonr(np.abs(A_trace[col] - .5), np.abs(B_trace[col]-.5))[0]
            pearson_array.update({col: pearson})
        else:
            pearson = stats.pearsonr(A_trace[col], B_trace[col])[0]
            pearson_array.update({col: pearson})
        # pearson = stats.pearsonr(A_trace[col], B_trace[col])[0]
        # pearson_array.update({col: pearson})
    pearson_array = pd.DataFrame(pearson_array, index=[0])
    pearson_pd = pearson_pd.append(pearson_array, ignore_index=True)

    A_trace = pd.DataFrame([val_A.iloc[2] for val_A in A_dict.values()], columns=struct.A_dict_sis['Sister_Trace_A_0'].keys())
    B_trace = pd.DataFrame([val_B.iloc[4] for val_B in B_dict.values()], columns=struct.B_dict_sis['Sister_Trace_B_0'].keys())
    pearson_array = dict()
    for col in columns:
        if col == 'division_ratios__f_n':
            pearson = stats.pearsonr(np.abs(A_trace[col] - .5), np.abs(B_trace[col]-.5))[0]
            pearson_array.update({col: pearson})
        else:
            pearson = stats.pearsonr(A_trace[col], B_trace[col])[0]
            pearson_array.update({col: pearson})
        # pearson = stats.pearsonr(A_trace[col], B_trace[col])[0]
        # pearson_array.update({col: pearson})
    pearson_array = pd.DataFrame(pearson_array, index=[0])
    pearson_pd = pearson_pd.append(pearson_array, ignore_index=True)

    return pearson_pd


def main():
    # Import the Refined Data
    pickle_in = open("metastructdata.pickle", "rb")
    struct = pickle.load(pickle_in)
    pickle_in.close()

    # pearson_pd_sis = getCorrelationPD(struct.A_dict_sis, struct.B_dict_sis, struct)
    # 
    # pearson_pd_non_sis = getCorrelationPD(struct.A_dict_non_sis, struct.B_dict_non_sis, struct)
    # 
    # pearson_pd_both = getCorrelationPD(struct.A_dict_both, struct.B_dict_both, struct)
    #
    # labels=['s_s', 'c_c', 'c2_c2', 'niece_aunt', 'niece1_aunt1', 'niece2_aunt2', 'g-niece_g-aunt', 'g-niece1_g-aunt1', 'g-niece2_g-aunt2']
    
    gen_depth = 15

    pearson_pd_sis = getCorrelationPD2(struct.A_dict_sis, struct.B_dict_sis, struct, gen_depth)

    pearson_pd_non_sis = getCorrelationPD2(struct.A_dict_non_sis, struct.B_dict_non_sis, struct, gen_depth)

    pearson_pd_both = getCorrelationPD2(struct.A_dict_both, struct.B_dict_both, struct, gen_depth)

    # pearson_pd_sis1 = getCorrelationPD1(struct.A_dict_sis, struct.B_dict_sis, struct, gen_depth)
    #
    # pearson_pd_non_sis1 = getCorrelationPD1(struct.A_dict_non_sis, struct.B_dict_non_sis, struct, gen_depth)
    #
    # pearson_pd_both1 = getCorrelationPD1(struct.A_dict_both, struct.B_dict_both, struct, gen_depth)

    label = getLabels(gen_depth)

    # plot it all
    for col in pearson_pd_sis.columns:
        # to seperate the different types of relationships
        plt.axvline(gen_depth, color='black')
        plt.axvline(2*gen_depth, color='black')

        # To see when the correlation is really just noise, here we are using the max but we might do 1 or 2 standard deviations maybe,
        # if we assume the oscilations around the control mean make a normal distribution
        plt.axhline(max(pearson_pd_both[col]), color='black')

        # plotting the correlations
        plt.errorbar(np.arange(len(pearson_pd_sis[col])), pearson_pd_sis[col], marker='.', label='sis')
        plt.errorbar(np.arange(len(pearson_pd_non_sis[col])), pearson_pd_non_sis[col], marker='.', label='non sis')
        plt.errorbar(np.arange(len(pearson_pd_both[col])), pearson_pd_both[col], marker='.', label='Control')

        # specifying the generation relations
        plt.xticks(np.arange(len(pearson_pd_sis[col])), labels=label, rotation=45)
        plt.legend()
        plt.title(col)
        plt.show()

        # plt.close()
        #
        # # to seperate the different types of relationships
        # plt.axvline(gen_depth, color='black')
        # plt.axvline(2 * gen_depth, color='black')
        #
        # # To see when the correlation is really just noise, here we are using the max but we might do 1 or 2 standard deviations maybe,
        # # if we assume the oscilations around the control mean make a normal distribution
        # plt.axhline(max(pearson_pd_both1[col]), color='black')
        #
        # # plotting the correlations
        # plt.errorbar(np.arange(len(pearson_pd_sis1[col])), pearson_pd_sis1[col], marker='.', label='sis')
        # plt.errorbar(np.arange(len(pearson_pd_non_sis1[col])), pearson_pd_non_sis1[col], marker='.', label='non sis')
        # plt.errorbar(np.arange(len(pearson_pd_both1[col])), pearson_pd_both1[col], marker='.', label='Control')
        #
        # # specifying the generation relations
        # plt.xticks(np.arange(len(pearson_pd_sis1[col])), labels=label, rotation=45)
        # plt.legend()
        # plt.title(col)
        # plt.show()
        #
        # plt.close()


if __name__ == '__main__':
    main()
