from __future__ import print_function

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sys, math
import matplotlib.pyplot as plt
import random
import itertools

import pickle
import scipy.stats as stats
import random


def compareTwoCycleCreations(index, dset, trace):
    # Import the Refined Data
    pickle_in = open("metastructdata.pickle", "rb")
    struct = pickle.load(pickle_in)
    pickle_in.close()

    # Import the Refined Data
    pickle_in = open("metastructdata_old.pickle", "rb")
    old_struct = pickle.load(pickle_in)
    pickle_in.close()

    # either A or B
    if dset == 'Sisters':
        raw_data = struct.Sisters[index]
        print('sisters raw data')
        if trace == 'A':
            cycle_data = struct.A_dict_sis['Sister_Trace_A_' + str(index)]
            print('Sister_Trace_A_' + str(index))
        elif trace == 'B':
            cycle_data = struct.B_dict_sis['Sister_Trace_B_' + str(index)]
            print('Sister_Trace_B_' + str(index))
        else:
            IOError('Choose either trace A or B')
    elif dset == 'Non-sisters':
        raw_data = struct.Nonsisters[index]
        print('non-sisters raw data')
        if trace == 'A':
            cycle_data = struct.A_dict_non_sis['Non-sister_Trace_A_' + str(index)]
            print('Non-sister_Trace_A_' + str(index))
        elif trace == 'B':
            cycle_data = struct.B_dict_non_sis['Non-sister_Trace_B_' + str(index)]
            print('Non-sister_Trace_B_' + str(index))
        else:
            IOError('Choose either trace A or B')
    else:
        IOError('Choose either Sisters or Nonsisters data set')

    # either A or B
    if dset == 'Sisters':
        old_raw_data = old_struct.Sisters[index]
        print('sisters raw data')
        if trace == 'A':
            old_cycle_data = old_struct.A_dict_sis['Sister_Trace_A_' + str(index)]
            print('Sister_Trace_A_' + str(index))
        elif trace == 'B':
            old_cycle_data = old_struct.B_dict_sis['Sister_Trace_B_' + str(index)]
            print('Sister_Trace_B_' + str(index))
        else:
            IOError('Choose either trace A or B')
    elif dset == 'Non-sisters':
        old_raw_data = old_struct.Nonsisters[index]
        print('non-sisters raw data')
        if trace == 'A':
            old_cycle_data = old_struct.A_dict_non_sis['Non-sister_Trace_A_' + str(index)]
            print('Non-sister_Trace_A_' + str(index))
        elif trace == 'B':
            old_cycle_data = old_struct.B_dict_non_sis['Non-sister_Trace_B_' + str(index)]
            print('Non-sister_Trace_B_' + str(index))
        else:
            IOError('Choose either trace A or B OLD')
    else:
        IOError('Choose either Sisters or Nonsisters data set OLD')

    old_length_array = []
    for x, alpha, gentime_current in zip(old_cycle_data['length_birth'], old_cycle_data['growth_length'], old_cycle_data['generationtime']):
        steps = int(np.ceil(round((gentime_current - .05) / .05, 2)) + 1)
        size = x * np.exp(np.linspace(0, round(gentime_current - .05, 2), num=steps) * alpha)
        old_length_array = np.concatenate([old_length_array, size], axis=0)

    length_array = []
    for x, alpha, gentime_current in zip(cycle_data['length_birth'], cycle_data['growth_length'], cycle_data['generationtime']):
        steps = int(np.ceil(round(gentime_current / .05, 2)) + 1)
        size = x * np.exp(np.linspace(0, round(gentime_current, 2), num=steps) * alpha)
        length_array = np.concatenate([length_array, size], axis=0)

    plt.plot(raw_data['length'+trace], label='raw data', linestyle='-')
    plt.plot(length_array, label='New cycle creat. {}'.format(np.linalg.norm(length_array - raw_data['length'+trace][:len(length_array)])),
             linestyle='--')
    plt.plot(old_length_array, label='Old cycle creat. {}'.format(np.linalg.norm(old_length_array - old_raw_data['length'+trace][:len(
        old_length_array)])), linestyle='--')
    plt.legend()
    plt.show()
    plt.close()


def GetNewGrowthRates(struct, index, dset, trace): # index is the dataID of the desired trace, dset is either Sisters or Nonsisters, and trace is
    # either A or B
    if dset == 'Sisters':
        raw_data = struct.Sisters[index]
        if trace == 'A':
            cycle_data = struct.A_dict_sis['Sister_Trace_A_' + str(index)]
        elif trace == 'B':
            cycle_data = struct.B_dict_sis['Sister_Trace_B_' + str(index)]
        else:
            IOError('Choose either trace A or B')
    elif dset == 'Nonsisters':
        raw_data = struct.Nonsisters[index]
        if trace == 'A':
            cycle_data = struct.A_dict_non_sis['Non-sister_Trace_A_' + str(index)]
        elif trace == 'B':
            cycle_data = struct.B_dict_non_sis['Non-sister_Trace_B_' + str(index)]
        else:
            IOError('Choose either trace A or B')
    else:
        IOError('Choose either Sisters or Nonsisters data set')


    # Now we figure out where the division times occur, also found in SisterMachineDataProcessing, ie. the "source code"
    diffdata = np.diff(raw_data['length'+trace])

    index_div = np.where(diffdata < -1)[0].flatten()
    for ind1, ind2 in zip(index_div[:-1], index_div[1:]):
        if ind2 - ind1 <= 2:
            index_div = np.delete(index_div, np.where(index_div == ind2))
    # WE DISCARD THE LAST CYCLE BECAUSE IT MOST LIKELY WON'T BE COMPLETE
    # THESE ARE INDICES AS WELL!
    start_times = [x + 1 for x in index_div]
    end_times = [x - 1 for x in index_div]
    start_times.append(0)
    start_times.sort()
    del start_times[-1]
    end_times.sort()
    if raw_data['lengthA'].index[-1] in start_times:
        start_times = start_times.remove(raw_data['lengthA'].index[-1])
    if 0 in end_times:
        end_times = end_times.remove(0)
    for start, end in zip(start_times, end_times):
        if start >= end:
            print('start', start, 'end', end, "didn't work")

    dj = sum(np.where(np.diff(index_div) == 1)[0].flatten())

    # timepoint of division is assumed to be the average before and after the drop in signal
    time_div = np.concatenate([[1.5 * raw_data['timeA'][0] - 0.5 * raw_data
    ['timeA'][1]], 0.5 * np.array(raw_data['timeA'][index_div + 1]) + 0.5 * np.array
                               (raw_data['timeA'][index_div])])
    # GROWTH LENGTH IS THE INDIVIDUAL ALPHA
    domain = np.linspace(0, (len(raw_data['timeA'][:index_div[0]]) - 1) * .05, num=len(raw_data['timeA'][:index_div[0]]))
    # first generation
    a = np.linalg.lstsq(domain[:, np.newaxis],
                        np.log(raw_data['lengthA'][:index_div[0]]) - np.log(raw_data['lengthA'][0]))[0]
    # rest of the generations
    b = np.array([np.linalg.lstsq(
        np.linspace(0, (len(raw_data['timeA'][index_div[i] + 1:index_div[i + 1] + 1]) - 1) * .05,
                    num=len(raw_data['timeA'][index_div[i] + 1:index_div[i + 1] + 1]))[:, np.newaxis],  # x-values
        np.log(raw_data['lengthA'][index_div[i] + 1:index_div[i + 1] + 1])
        - np.log(raw_data['lengthA'][index_div[i] + 1])  # y-values
    )[0] for i in range(len(index_div) - 1)]).flatten()
    growth_array = np.concatenate([a, b], axis=0)


    # NOW FOR THE FITTED INTERCEPT AND FREE SLOPE IE. NO FIXED INTERCEPT
    c = np.linalg.lstsq(np.concatenate([domain[:, np.newaxis], np.ones_like(domain)[:, np.newaxis]], axis=1),
                        np.log(raw_data['lengthA'][:index_div[0]]))[0]

    d = np.array([np.linalg.lstsq(np.concatenate([
        np.linspace(0, (len(raw_data['timeA'][index_div[i] + 1:index_div[i + 1] + 1]) - 1) * .05,
                    num=len(raw_data['timeA'][index_div[i] + 1:index_div[i + 1] + 1]))[:, np.newaxis], np.ones_like(
            np.linspace(0, (len(raw_data['timeA'][index_div[i] + 1:index_div[i + 1] + 1]) - 1) * .05,
                        num=len(raw_data['timeA'][index_div[i] + 1:index_div[i + 1] + 1]))
        )[:, np.newaxis]], axis=1),  # x-values
        np.log(raw_data['lengthA'][index_div[i] + 1:index_div[i + 1] + 1])  # y-values
    )[0] for i in range(len(index_div) - 1)])

    # print('c', c, c.shape)
    # print('d', d, d.shape)

    growth_array1 = np.concatenate([np.array(c, ndmin=2), d], axis=0)
    # print('growth_array1', growth_array1, growth_array1.shape)


    # fitted data in absolute time using our old growth rate without fixed intercept and using data intercept
    length_array = np.array([])
    for x, alpha, gentime_current in zip(cycle_data['length_birth'][:-1], cycle_data['growth_length'][:-1], cycle_data['generationtime'][:-1]):
        steps = int(np.ceil(round(gentime_current / .05, 2))+1)
        size = np.concatenate([np.array([x]), x * np.exp(np.linspace(0.05, round(gentime_current, 2), num=steps - 1) * alpha)],
                              axis=0)
        size = x * np.exp(np.linspace(0, round(gentime_current - .05, 2), num=steps) * alpha)
        length_array = np.concatenate([length_array, size], axis=0)

    # fitted data in absolute time using our new growth rate with fixed intercept and using data intercept
    length_array1 = np.array([])
    for x, alpha, gentime_current in zip(cycle_data['length_birth'][:-1], growth_array[:-1], cycle_data['generationtime'][:-1]):
        steps = int(np.ceil(round(gentime_current / .05, 2))+1)
        size = np.concatenate([np.array([x]), x * np.exp(np.linspace(0.05, round(gentime_current, 2), num=steps - 1) * alpha)],
                              axis=0)
        size = x * np.exp(np.linspace(0, round(gentime_current - .05, 2), num=steps) * alpha)
        length_array1 = np.concatenate([length_array1, size], axis=0)

    # fitted data in absolute time using our new growth rate without fixed intercept and using fitted intercept
    length_array2 = np.array([])
    for x, alpha, gentime_current in zip(np.exp(growth_array1[:-1, 1]), growth_array1[:-1, 0], cycle_data['generationtime'][:-1]):
        steps = int(np.ceil(round(gentime_current / .05, 2))+1)
        size = np.concatenate([np.array([x]), x * np.exp(np.linspace(0.05, round(gentime_current, 2), num=steps - 1) * alpha)],
                              axis=0)
        size = x * np.exp(np.linspace(0, round(gentime_current - .05, 2), num=steps) * alpha)
        length_array2 = np.concatenate([length_array2, size], axis=0)

    plt.plot(raw_data['lengthA'], label='raw data', linestyle='-')
    print(len(length_array))
    print(len(raw_data['lengthA'][:len(length_array)]))
    print(np.linalg.norm(length_array - raw_data['lengthA'][:len(length_array)]))
    plt.plot(length_array, label='Old alphas {}'.format(np.linalg.norm(length_array - raw_data['lengthA'][:len(length_array)])), linestyle='--')
    plt.plot(length_array1, label='New alphas {}'.format(np.linalg.norm(length_array1 - raw_data['lengthA'][:len(length_array)])), linestyle='-.')
    plt.plot(length_array2, label='fitted intercept {}'.format(np.linalg.norm(length_array2 - raw_data['lengthA'][:len(length_array)])),
             linestyle='-.')
    plt.legend()
    plt.show()


def showDataAndMapTraj(struct, index, dset, trace):
    # either A or B
    if dset == 'Sisters':
        raw_data = struct.Sisters[index]
        if trace == 'A':
            cycle_data = struct.A_dict_sis['Sister_Trace_A_' + str(index)]
        elif trace == 'B':
            cycle_data = struct.B_dict_sis['Sister_Trace_B_' + str(index)]
        else:
            IOError('Choose either trace A or B')
    elif dset == 'Nonsisters':
        raw_data = struct.Nonsisters[index]
        if trace == 'A':
            cycle_data = struct.A_dict_non_sis['Non-sister_Trace_A_' + str(index)]
        elif trace == 'B':
            cycle_data = struct.B_dict_non_sis['Non-sister_Trace_B_' + str(index)]
        else:
            IOError('Choose either trace A or B')
    else:
        IOError('Choose either Sisters or Nonsisters data set')
    size_array = np.array([])
    for x, alpha, gentime_current in zip(cycle_data['length_birth'], cycle_data['growth_length'], cycle_data['generationtime']):
        points = int(np.ceil(round(gentime_current / .05, 2)) + 1) # because for n steps there are n+1 points in it, ie the step connects
        size = x*np.exp(alpha*np.linspace(0, round(gentime_current, 2), num=points))
        print(size)
        size_array = np.concatenate([size_array, size], axis=0)
    plt.plot(raw_data['lengthA'], label='raw data', linestyle='-')
    plt.plot(size_array, label='model {:.2e}'.format(np.linalg.norm(size_array - raw_data['lengthA'][:len(size_array)])), linestyle='--')
    plt.legend()
    plt.show()


def main():
    # Import the Refined Data
    pickle_in = open("metastructdata.pickle", "rb")
    struct = pickle.load(pickle_in)
    pickle_in.close()

    ''' this is visually checking all of the different alphas and intercept combinations '''
    # for index in range(len(struct.A_dict_sis)):
    #     compareTwoCycleCreations(index, 'Non-sisters', 'B')
    #
    # exit()




    ''' give the imported things the new names '''
    best_combos = struct.best_combos
    factors_for_iter = struct.factors_for_iter
    factors = struct.factors
    targets = struct.targets
    X_test = struct.X_test
    X_train = struct.X_train
    y_test = struct.y_test
    y_train = struct.y_train
    m_d_dependance = struct.m_d_dependance
    m_d_dependance_units = struct.m_d_dependance_units
    cols = struct.cols
    pop_means = struct.pop_means
    pop_stds = struct.pop_stds
    print('Factors are ', factors) # ['generationtime_m', 'length_birth_m', 'growth_length_m', 'division_ratios__f_n_m', 'length_final_m', 'phi_m']
    print('Targets are ', targets)

    # Get the population level statistics of cycle params to convert to Standard Normal variables
    sis_traces = [A for A in struct.dict_with_all_sister_traces.values()]
    non_traces = [B for B in struct.dict_with_all_non_sister_traces.values()]
    all_traces = sis_traces + non_traces
    pop_dist_of_cps = [pd.concat(np.array([trace[c_p] for trace in all_traces]), ignore_index=True) for c_p in all_traces[0].keys()]

    ''' output the population distribution graphs for powerpoint '''
    # for col, num in zip(list(all_traces[0].keys()), range(len(list(all_traces[0].keys())))):
    #     ''' the standard normal distributions '''
    #     # far_away = 2
    #     # left_range = np.mean(m_d_dependance[col]) - far_away*np.std(m_d_dependance[col])
    #     # right_range = np.mean(m_d_dependance[col]) + far_away*np.std(m_d_dependance[col])
    #     # plt.hist(m_d_dependance[col], range=[left_range, right_range], label=r'{}$\pm${}'.format(pop_means[num], pop_stds[num]))
    #     # plt.xlabel(col)
    #     # plt.legend()
    #     # plt.show()
    #
    #     ''' in the units we observed them '''
    #     far_away = 2
    #     left_range = np.mean(pop_dist_of_cps[num]) - far_away * np.std(pop_dist_of_cps[num])
    #     right_range = np.mean(pop_dist_of_cps[num]) + far_away * np.std(pop_dist_of_cps[num])
    #     plt.hist(pop_dist_of_cps[num], range=[left_range, right_range], label=r'{:.2e}$\pm${:.2e}'.format(pop_means[num], pop_stds[num]), bins=12)
    #     plt.xlabel(col)
    #     plt.legend()
    #     # plt.show()
    #     plt.savefig('population distribution of '+str(col)[:-2], dpi=300)
    #     plt.close()
    # exit()

    ''' elbow graphs for factor selection '''
    # for col in struct.y_test.columns:
    #     print(col + ' combo')
    #     with pd.option_context('display.max_rows', None, 'display.max_columns', None, ):  # more
    #         # options can be specified also
    #         print(best_combos[[col + ' combo', col + ' pearson']].sort_values(col + ' pearson').reset_index())
    #     plt.plot(range(63), best_combos[[col + ' pearson']].sort_values(col + ' pearson'), label=col, marker='.')
    #     plt.legend()
    #     plt.title('who has an elbow?')
    #     plt.show()
    #     print('----------------------------------------')


    # initialize the matrix

    ''' Same-cell cycle parameter correlations '''
    memory = []
    print(all_traces[0].keys()) # ['generationtime', 'length_birth', 'growth_length', 'length_final', 'division_ratios__f_n', 'phi']
    additional_params = 1
    for col1 in range(len(all_traces[0].keys())+additional_params):
        for col2 in range(len(all_traces[0].keys())+additional_params):
            if col1 != col2 and [col2, col1] not in memory:
                # For |f_n - .5| = F
                # x = np.abs(pop_dist_of_cps[col1]-.5)
                # y = np.abs(pop_dist_of_cps[col2]-.5)
                # Add this condition: 'and (col1 == 4 or col2 == 4)'
                # if col1 == len(all_traces[0].keys()):

                if col1 < len(all_traces[0].keys()):
                    x = pop_dist_of_cps[col1]
                    xlabel = list(all_traces[0].keys())[col1]
                else:
                    # x, xlabel = pop_dist_of_cps[1]*np.exp(pop_dist_of_cps[2]*pop_dist_of_cps[0]), 'predicted final length'
                    x, xlabel = -np.log(pop_dist_of_cps[1])/pop_dist_of_cps[2], 'predicted generationtime'

                if col2 < len(all_traces[0].keys()):
                    y = pop_dist_of_cps[col2]
                    ylabel = list(all_traces[0].keys())[col2]
                else:
                    # y, ylabel = pop_dist_of_cps[1]*np.exp(pop_dist_of_cps[2]*pop_dist_of_cps[0]), 'predicted final length'
                    y, ylabel = -np.log(pop_dist_of_cps[1])/pop_dist_of_cps[2], 'predicted generationtime'


                m, c = np.linalg.lstsq(np.concatenate([x[:, np.newaxis], np.ones_like(x)[:, np.newaxis]], axis=1), y)[0]
                print(len(x), len(y))
                exit()
                correlation = round(stats.pearsonr(x, y)[0], 3)
                print(xlabel, ylabel, correlation)
                plt.scatter(x, y, facecolors='none', edgecolors='blue')
                plt.plot(np.linspace(min(x), max(x)), np.linspace(min(x), max(x)) * m + c, label=r'$\rho=${}'.format(correlation), color='orange')
                plt.legend()
                plt.xlabel(xlabel)
                plt.ylabel(ylabel)
                plt.show()
                # plt.savefig('Same-cell correlation between '+str(list(all_traces[0].keys())[col1])+' and '+list(all_traces[0].keys())[col2], dpi=300)
                # plt.close()
                memory.append([col1, col2])

    exit()

    ''' Mother-Daughter correlations -- the pearson correlations of the targets depending on each of the factors DATA '''
    #
    # # ''' create a new variable'''
    # # m_d_dependance_units['new_one_m'] = np.log((2 * struct.x_avg) / m_d_dependance_units['length_birth_m']) / m_d_dependance_units['growth_length_m']
    # # m_d_dependance_units['new_one_d'] = np.log((2 * struct.x_avg) / m_d_dependance_units['length_birth_d']) / m_d_dependance_units['growth_length_d']
    # # factors.append('new_one_m')
    # # targets.append('new_one_d')
    # #
    # # m_d_dependance_units['new_one_m1'] = -np.log(m_d_dependance_units['length_birth_m']) / m_d_dependance_units['growth_length_m']
    # # m_d_dependance_units['new_one_d1'] = -np.log(m_d_dependance_units['length_birth_d']) / m_d_dependance_units['growth_length_d']
    # # factors.append('new_one_m1')
    # # targets.append('new_one_d1')
    # #
    # # ''' normalize the birth sizes '''
    # # m_d_dependance_units['length_birth_m'] = np.log(m_d_dependance_units['length_birth_m'] / struct.x_avg)
    # # m_d_dependance_units['length_birth_d'] = np.log(m_d_dependance_units['length_birth_d'] / struct.x_avg)
    #
    # mother_daughter_correlation_matrix = pd.DataFrame(columns=factors)
    # for factor in factors:
    #     # fig, axs = plt.subplots(1, len(targets), sharex=True)
    #     # fig.text(0.5, 0.04, factor, ha='center')
    #     ind = 0
    #     correlation_array = []
    #     for target in targets:
    #         ''' using the units, not standard variables '''
    #         m, c = np.linalg.lstsq(np.concatenate([np.array(m_d_dependance_units[factor])[:, np.newaxis], np.ones_like(m_d_dependance_units[factor])
    #         [:, np.newaxis]], axis=1), m_d_dependance_units[target])[0]
    #         correlation = round(stats.pearsonr(m_d_dependance_units[target], m_d_dependance_units[factor])[0], 3)
    #
    #         ''' using standard variables '''
    #         m, c = np.linalg.lstsq(np.concatenate([np.array(m_d_dependance[factor])[:, np.newaxis], np.ones_like(m_d_dependance[factor])
    #         [:, np.newaxis]], axis=1), m_d_dependance[target])[0]
    #         correlation = round(stats.pearsonr(m_d_dependance[target], m_d_dependance[factor])[0], 3)
    #         # axs[ind].scatter(m_d_dependance_units[factor], m_d_dependance_units[target], facecolors='none', edgecolors='blue')  # , label='_nolegend_'
    #         # axs[ind].plot(np.linspace(min(m_d_dependance_units[factor]), max(m_d_dependance_units[factor])), np.linspace(min(m_d_dependance_units[factor]),
    #         #               max(m_d_dependance_units[factor])) * m + c, label=r'$\rho=${}'.format(correlation), color='orange')
    #         # label=r'$\rho$, pval: {}, {:.2e}'.format(round(stats.pearsonr(m_d_dependance_units[target], m_d_dependance_units[factor])[0], 3),
    #         # stats.pearsonr(m_d_dependance_units[target], m_d_dependance_units[factor])[1])
    #         # axs[ind].set_ylabel(target)
    #
    #         # axs[ind].set_ylim([np.mean(m_d_dependance_units[factor]) - 5 * np.std(m_d_dependance_units[factor]), np.mean(m_d_dependance_units[factor]) + 5 * np.std(
    #         #     m_d_dependance_units[factor])])
    #         # axs[ind].set_xlim([np.mean(m_d_dependance_units[target]) - 5 * np.std(m_d_dependance_units[target]), np.mean(m_d_dependance_units[target]) + 5 * np.std(
    #         #     m_d_dependance_units[target])])
    #         # axs[ind].legend()
    #         print(factor, target, correlation, m, c)
    #         correlation_array.append(correlation)
    #         ind = ind + 1
    #     mother_daughter_correlation_matrix[factor] = correlation_array
    #     # plt.suptitle('How ' + str(factor) + ' affects the other parameters in experimental data')
    #     # plt.tight_layout()
    #     # plt.show()
    #     # plt.savefig('Mother ' + str(factor) + ' and daughter cycle parameter correlations', dpi=300, bbox_inches='tight')
    #     # plt.close()
    # mother_daughter_correlation_matrix.rename(index=dict(zip(np.arange(len(targets)), targets)), inplace=True)
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    #     print(mother_daughter_correlation_matrix)
    # exit()

    ''' the pearson correlations of Characteristics of Sisters DATA '''
    change_from_faction_to_distance_from_mean = False # If we want it
    index = 2  # because we are looking at sisters
    memory = []
    sis_correlations = []
    non_sis_correlations = []
    control_correlations = []
    for c_p1 in all_traces[0].keys():
        for c_p2 in all_traces[0].keys():
            if [c_p1, c_p2] not in memory and [c_p2, c_p1] not in memory:

                sample_number = 100

                fig, axs = plt.subplots(1, 3, sharex=True, sharey=True)
                fig.text(0.5, 0.04, c_p1, ha='center')

                index_s_1 = 0 # now we want cousins
                index_s_2 = 1
                c_p1_array = []
                c_p2_array = []
                for A_trace, B_trace in zip(struct.A_dict_sis.values(), struct.B_dict_sis.values()):
                    ''' make all f be |f-.5| '''
                    if change_from_faction_to_distance_from_mean:

                        if c_p1 == 'division_ratios__f_n':
                            # print(A_trace[c_p1])
                            A_trace[c_p1] = np.abs(np.array(A_trace[c_p1]) - .5 * np.ones_like(A_trace[c_p1]))
                            B_trace[c_p1] = np.abs(np.array(B_trace[c_p1]) - .5 * np.ones_like(B_trace[c_p1]))
                            # print(A_trace[c_p1])
                            # print('c_p1 is f_n')
                            # print(A_trace[c_p1])
                            # print(B_trace[c_p1])
                        if c_p2 == 'division_ratios__f_n':
                            # print(A_trace[c_p2])
                            B_trace[c_p2] = np.abs(np.array(B_trace[c_p2]) - .5 * np.ones_like(B_trace[c_p2]))
                            A_trace[c_p2] = np.abs(np.array(A_trace[c_p2]) - .5 * np.ones_like(A_trace[c_p2]))
                            # print(A_trace[c_p2])
                            # print('c_p2 is f_n')
                            # print('A_trace[c_p2]', A_trace[c_p2])
                            # print('B_trace[c_p2]', B_trace[c_p2])
                    c_p1_array.append(A_trace[c_p1].iloc[index_s_1])
                    c_p2_array.append(B_trace[c_p2].iloc[index_s_2])
                    c_p1_array.append(B_trace[c_p1].iloc[index_s_1])
                    c_p2_array.append(A_trace[c_p2].iloc[index_s_2])
                correlation = round(stats.pearsonr(c_p1_array, c_p2_array)[0], 3)
                sis_correlations.append(correlation)
                m, c = np.linalg.lstsq(np.concatenate([np.array(c_p1_array)[:, np.newaxis], np.ones_like(c_p1_array)
                       [:, np.newaxis]], axis=1), c_p2_array)[0]
                axs[0].scatter(c_p1_array[:sample_number], c_p2_array[:sample_number], label=r'Sisters $\rho={}$'.format(correlation), facecolors='none', edgecolors='blue')
                axs[0].plot(np.linspace(min(c_p1_array), max(c_p1_array)), np.linspace(min(c_p1_array), max(c_p1_array)) * m + c, color='orange')
                axs[0].legend()
                axs[0].set_ylabel(c_p2)
                axs[0].set_ylim([np.mean(c_p2_array) - 3 * np.std(c_p2_array), np.mean(c_p2_array) + 3 * np.std(c_p2_array)])
                axs[0].set_xlim([np.mean(c_p1_array) - 3 * np.std(c_p1_array), np.mean(c_p1_array) + 3 * np.std(c_p1_array)])
                # print('sister', c_p1, c_p2, correlation)

                ''' I put the index_ns because the first two non sisters were chosen by having similar length_birth, so we use the second cycle '''
                index_ns_1 = 0
                index_ns_2 = 1
                c_p1_array = []
                c_p2_array = []
                for A_trace, B_trace in zip(struct.A_dict_non_sis.values(), struct.B_dict_non_sis.values()):
                    ''' make all f be |f-.5| '''
                    if change_from_faction_to_distance_from_mean:

                        if c_p1 == 'division_ratios__f_n':
                            # print(A_trace[c_p1])
                            A_trace[c_p1] = np.abs(np.array(A_trace[c_p1]) - .5*np.ones_like(A_trace[c_p1]))
                            B_trace[c_p1] = np.abs(np.array(B_trace[c_p1]) - .5 * np.ones_like(B_trace[c_p1]))
                            # print(A_trace[c_p1])
                            # print('c_p1 is f_n')
                            # print(A_trace[c_p1])
                            # print(B_trace[c_p1])
                        if c_p2 == 'division_ratios__f_n':
                            # print(A_trace[c_p2])
                            B_trace[c_p2] = np.abs(np.array(B_trace[c_p2]) - .5 * np.ones_like(B_trace[c_p2]))
                            A_trace[c_p2] = np.abs(np.array(A_trace[c_p2]) - .5 * np.ones_like(A_trace[c_p2]))
                            # print(A_trace[c_p2])
                            # print('c_p2 is f_n')
                            # print('A_trace[c_p2]', A_trace[c_p2])
                            # print('B_trace[c_p2]', B_trace[c_p2])
                    c_p1_array.append(A_trace[c_p1].iloc[index_ns_1])
                    c_p2_array.append(B_trace[c_p2].iloc[index_ns_2])
                    c_p1_array.append(B_trace[c_p1].iloc[index_ns_1])
                    c_p2_array.append(A_trace[c_p2].iloc[index_ns_2])
                correlation = round(stats.pearsonr(c_p1_array, c_p2_array)[0], 3)
                non_sis_correlations.append(correlation)
                axs[1].scatter(c_p1_array[:sample_number], c_p2_array[:sample_number], label=r'Non-Sisters $\rho={}$'.format(correlation), facecolors='none', edgecolors='blue')
                axs[1].plot(np.linspace(min(c_p1_array), max(c_p1_array)), np.linspace(min(c_p1_array), max(c_p1_array)) * m + c, color='orange')
                axs[1].legend()
                axs[1].set_ylim([np.mean(c_p2_array) - 3 * np.std(c_p2_array), np.mean(c_p2_array) + 3 * np.std(c_p2_array)])
                axs[1].set_xlim([np.mean(c_p1_array) - 3 * np.std(c_p1_array), np.mean(c_p1_array) + 3 * np.std(c_p1_array)])
                print('non-sister', c_p1, c_p2, correlation)

                ''' If I want the weighted average of all the cycle correlations to check if the environmental influence holds '''
                # samples_array = []
                # total_num = len(struct.A_dict_non_sis.values())
                # max_generation = max([min(len(val_A), len(val_B)) for val_A, val_B in zip(struct.A_dict_non_sis.values(),
                #                                                                           struct.B_dict_non_sis.values())])
                # print(c_p1, c_p2)
                # weighted_correlations = []
                # for index_ns in range(3):
                #     # print('index_ns', index_ns)
                #     number_of_samples = len([0 for val_A, val_B in zip(struct.A_dict_non_sis.values(), struct.B_dict_non_sis.values()) if min(
                #         len(val_A), len(val_B)) > index_ns])
                #     samples_array.append(number_of_samples/total_num)
                #     c_p1_array = []
                #     c_p2_array = []
                #     for A_trace, B_trace in zip(struct.A_dict_non_sis.values(), struct.B_dict_non_sis.values()):
                #         if min(len(A_trace), len(B_trace)) > index_ns:
                #             c_p1_array.append(A_trace[c_p1].iloc[index_ns])
                #             c_p2_array.append(B_trace[c_p2].iloc[index_ns])
                #             c_p1_array.append(B_trace[c_p1].iloc[index_ns])
                #             c_p2_array.append(A_trace[c_p2].iloc[index_ns])
                #     correlation = round(stats.pearsonr(c_p1_array, c_p2_array)[0], 3)
                #     # print('correlation', correlation)
                #     weighted_correlations.append(correlation)
                # normalized_weights = np.array(samples_array) / np.linalg.norm(np.array(samples_array))
                # average_correlation = np.dot(normalized_weights, np.array(weighted_correlations))
                # # print('normalized_weights', normalized_weights)
                # # print('weighted_correlations', np.array(weighted_correlations))
                # non_sis_correlations.append(average_correlation / len(normalized_weights))
                # print('average correlation', average_correlation / len(normalized_weights))

                c_p1_array = []
                c_p2_array = []
                for A_trace, B_trace in zip(struct.A_dict_both.values(), struct.B_dict_both.values()):
                    c_p1_array.append(A_trace[c_p1].iloc[index])
                    c_p2_array.append(B_trace[c_p2].iloc[index])
                    c_p1_array.append(B_trace[c_p1].iloc[index])
                    c_p2_array.append(A_trace[c_p2].iloc[index])
                correlation = round(stats.pearsonr(c_p1_array, c_p2_array)[0], 3)
                control_correlations.append(correlation)
                axs[2].scatter(c_p1_array[:sample_number], c_p2_array[:sample_number], label=r'Control $\rho={}$'.format(correlation), facecolors='none', edgecolors='blue')
                axs[2].plot(np.linspace(min(c_p1_array), max(c_p1_array)), np.linspace(min(c_p1_array), max(c_p1_array)) * m + c, color='orange')
                axs[2].legend()
                axs[2].set_ylim([np.mean(c_p2_array) - 3 * np.std(c_p2_array), np.mean(c_p2_array) + 3 * np.std(c_p2_array)])
                axs[2].set_xlim([np.mean(c_p1_array) - 3 * np.std(c_p1_array), np.mean(c_p1_array) + 3 * np.std(c_p1_array)])
                # print('Control', c_p1, c_p2, correlation)

                # plt.show()
                # plt.close()

                memory.append([c_p1, c_p2])
    exit()

    ''' Getting the exponential decay of parameter correlations and how long it lasts DATA '''
    # change_from_faction_to_distance_from_mean = False  # If we want it
    # index = 0  # because we are looking at sisters
    # memory = []
    # sis_correlations = []
    # non_sis_correlations = []
    # control_correlations = []
    # for c_p1 in all_traces[0].keys():
    #     for c_p2 in all_traces[0].keys():
    #         if [c_p1, c_p2] not in memory and [c_p2, c_p1] not in memory:
    #
    #             sis_inside_correlations = []
    #             for index_s in range(9):
    #
    #                 c_p1_array = []
    #                 c_p2_array = []
    #                 for A_trace, B_trace in zip(struct.A_dict_sis.values(), struct.B_dict_sis.values()):
    #                     ''' make all f be |f-.5| '''
    #                     if change_from_faction_to_distance_from_mean:
    #
    #                         if c_p1 == 'division_ratios__f_n':
    #                             A_trace[c_p1] = np.abs(np.array(A_trace[c_p1]) - .5 * np.ones_like(A_trace[c_p1]))
    #                             B_trace[c_p1] = np.abs(np.array(B_trace[c_p1]) - .5 * np.ones_like(B_trace[c_p1]))
    #                         if c_p2 == 'division_ratios__f_n':
    #                             B_trace[c_p2] = np.abs(np.array(B_trace[c_p2]) - .5 * np.ones_like(B_trace[c_p2]))
    #                             A_trace[c_p2] = np.abs(np.array(A_trace[c_p2]) - .5 * np.ones_like(A_trace[c_p2]))
    #                     c_p1_array.append(A_trace[c_p1].iloc[index_s])
    #                     c_p2_array.append(B_trace[c_p2].iloc[index_s])
    #                     c_p1_array.append(B_trace[c_p1].iloc[index_s])
    #                     c_p2_array.append(A_trace[c_p2].iloc[index_s])
    #                 correlation = round(stats.pearsonr(c_p1_array, c_p2_array)[0], 3)
    #                 if correlation > .28 or correlation < -.28:
    #                     sis_inside_correlations.append(correlation)
    #                 else:
    #                     break
    #
    #             non_sis_inside_correlations = []
    #             for index_ns in range(9):
    #                 c_p1_array = []
    #                 c_p2_array = []
    #                 for A_trace, B_trace in zip(struct.A_dict_non_sis.values(), struct.B_dict_non_sis.values()):
    #                     ''' make all f be |f-.5| '''
    #                     if change_from_faction_to_distance_from_mean:
    #
    #                         if c_p1 == 'division_ratios__f_n':
    #                             A_trace[c_p1] = np.abs(np.array(A_trace[c_p1]) - .5 * np.ones_like(A_trace[c_p1]))
    #                             B_trace[c_p1] = np.abs(np.array(B_trace[c_p1]) - .5 * np.ones_like(B_trace[c_p1]))
    #                         if c_p2 == 'division_ratios__f_n':
    #                             B_trace[c_p2] = np.abs(np.array(B_trace[c_p2]) - .5 * np.ones_like(B_trace[c_p2]))
    #                             A_trace[c_p2] = np.abs(np.array(A_trace[c_p2]) - .5 * np.ones_like(A_trace[c_p2]))
    #                     c_p1_array.append(A_trace[c_p1].iloc[index_ns])
    #                     c_p2_array.append(B_trace[c_p2].iloc[index_ns])
    #                     c_p1_array.append(B_trace[c_p1].iloc[index_ns])
    #                     c_p2_array.append(A_trace[c_p2].iloc[index_ns])
    #                 correlation = round(stats.pearsonr(c_p1_array, c_p2_array)[0], 3)
    #                 if correlation > .28 or correlation < -.28:
    #                     non_sis_inside_correlations.append(correlation)
    #                 else:
    #                     break
    #
    #             if (len(non_sis_inside_correlations) > 1 or len(sis_inside_correlations) > 1):
    #
    #                 for sis in sis_inside_correlations:
    #                     if sis > 0:
    #                         sis_sign = True
    #                     if sis < 0:
    #                         sis_sign = False
    #
    #                 for non_sis in non_sis_inside_correlations:
    #                     if non_sis > 0:
    #                         non_sis_sign = True
    #                     if non_sis < 0:
    #                         non_sis_sign = False
    #
    #                 print(sis_inside_correlations)
    #                 print(non_sis_inside_correlations)
    #                 fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
    #                 fig.text(0.5, 0.04, 'Generation index', ha='center')
    #
    #                 if sis_sign:
    #                     m, c = np.linalg.lstsq(
    #                         np.concatenate([np.arange(len(sis_inside_correlations))[:, np.newaxis], np.ones_like(np.arange(len(sis_inside_correlations)))
    #                         [:, np.newaxis]], axis=1), np.log(np.array(sis_inside_correlations)))[0]
    #                     m = round(m, 4)
    #                     sis_correlations.append([c_p1, c_p2, len(sis_inside_correlations), sis_inside_correlations, m])
    #                     axs[0].plot(np.arange(len(sis_inside_correlations)), c + m * np.arange(len(sis_inside_correlations)), color='orange',
    #                                 label='exponential rate of {}'.format(m))
    #                     axs[0].scatter(np.arange(len(sis_inside_correlations)), np.log(np.array(sis_inside_correlations)))
    #                     axs[0].set_ylabel('log of correlation')
    #                     axs[0].legend()
    #                 else:
    #                     m, c = np.linalg.lstsq(
    #                         np.concatenate(
    #                             [np.arange(len(sis_inside_correlations))[:, np.newaxis], np.ones_like(np.arange(len(sis_inside_correlations)))
    #                             [:, np.newaxis]], axis=1), np.log(np.abs(np.array(sis_inside_correlations))))[0]
    #                     m = round(-m, 4)
    #                     sis_correlations.append([c_p1, c_p2, len(sis_inside_correlations), sis_inside_correlations, m])
    #                     axs[0].plot(np.arange(len(sis_inside_correlations)), c - m * np.arange(len(sis_inside_correlations)), color='orange',
    #                                 label='exponential rate of {}'.format(m))
    #                     axs[0].scatter(np.arange(len(sis_inside_correlations)), np.log(np.abs(np.array(sis_inside_correlations))))
    #                     axs[0].set_ylabel('log of correlation')
    #                     axs[0].legend()
    #
    #                 if non_sis_sign:
    #                     m, c = np.linalg.lstsq(
    #                         np.concatenate([np.arange(len(non_sis_inside_correlations))[:, np.newaxis], np.ones_like(np.arange(len(non_sis_inside_correlations)))
    #                         [:, np.newaxis]], axis=1), np.log(np.array(non_sis_inside_correlations)))[0]
    #                     m = round(m, 4)
    #                     non_sis_correlations.append([c_p1, c_p2, len(non_sis_inside_correlations), non_sis_inside_correlations, m])
    #                     axs[1].plot(np.arange(len(non_sis_inside_correlations)), c + m * np.arange(len(non_sis_inside_correlations)), color='orange',
    #                                 label='exponential rate of {}'.format(m))
    #                     axs[1].scatter(np.arange(len(non_sis_inside_correlations)), np.log(np.array(non_sis_inside_correlations)))
    #                     axs[1].legend()
    #                     plt.suptitle(str(c_p1) + ' ' + str(c_p2))
    #                     print(len(non_sis_inside_correlations), len(sis_inside_correlations))
    #                     plt.show()
    #                     plt.legend()
    #                     plt.close()
    #                 else:
    #                     m, c = np.linalg.lstsq(
    #                         np.concatenate(
    #                             [np.arange(len(non_sis_inside_correlations))[:, np.newaxis], np.ones_like(np.arange(len(non_sis_inside_correlations)))
    #                             [:, np.newaxis]], axis=1), np.log(np.abs(np.array(non_sis_inside_correlations))))[0]
    #                     m = round(-m, 4)
    #                     non_sis_correlations.append([c_p1, c_p2, len(non_sis_inside_correlations), non_sis_inside_correlations, m])
    #                     axs[1].plot(np.arange(len(non_sis_inside_correlations)), c - m * np.arange(len(non_sis_inside_correlations)), color='orange',
    #                                 label='exponential rate of {}'.format(m))
    #                     axs[1].scatter(np.arange(len(non_sis_inside_correlations)), np.log(np.abs(np.array(non_sis_inside_correlations))))
    #                     axs[1].legend()
    #                     plt.suptitle(str(c_p1) + ' ' + str(c_p2))
    #                     print(len(non_sis_inside_correlations), len(sis_inside_correlations))
    #                     plt.show()
    #                     plt.legend()
    #                     plt.close()
    #
    # print('Sister')
    # print(np.array(sis_correlations))
    # print('Non-Sisters')
    # print(np.array(non_sis_correlations))
    # exit()


    matrix = np.zeros([len(y_train.columns), len(X_train.columns)])
    # index to input the vector a into the matrix
    ind1 = 0
    for param_d in y_train.keys():
        # leastsquares approx for all training data
        a = np.linalg.lstsq(X_train, y_train[param_d])[0]
        # input to matrix
        matrix[ind1] = a
        # on to the next target
        ind1 = ind1+1
    print('HERE IS THE MATRIX:')
    print(matrix)

    ''' change the norm of the matrix '''
    # print('norm of the matrix', np.linalg.norm(matrix))
    #
    # matrix = matrix / (1*np.linalg.norm(matrix))
    #
    # print(matrix)




    ''' Now we will make a simulated daughter based on a mother that has data '''
    results = pd.DataFrame(columns=y_test.columns)
    randomness = pd.DataFrame(columns=y_test.columns)


    ''' make the simulated results and a random shuffle '''
    for sample in range(len(X_test)):


        # add the simulated sample
        simulated_result = np.dot(matrix, X_test.iloc[sample])
        results.loc[sample] = pd.Series(simulated_result, index=y_test.columns)

        # add the randomly sampled sample, basically the same thing between both
        # randomness.loc[sample] = pd.Series(np.array([random.choices(m_d_dependance[target], k=1) for target in targets]).flatten(), index=y_test.columns)
        randomness.loc[sample] = pd.Series(np.random.normal(loc=np.mean(m_d_dependance[targets]), scale=np.std(m_d_dependance[targets])),
                                   index=y_test.columns)


    ''' plot the simulated vs the real daughters SIGNAL '''
    # for ind in range(3): #range(len(y_test))
    #     reverted_gentime = int(round(((results['generationtime_d'].iloc[ind] * pop_stds[0]) + pop_means[0])/.05))
    #     reverted_gentime1 = int(round(((y_test['generationtime_d'].iloc[ind] * pop_stds[0]) + pop_means[0])/.05))
    #
    #     x = np.linspace(0, reverted_gentime*.05-.05, reverted_gentime)
    #     x1 = np.linspace(0, reverted_gentime1 * .05-.05, reverted_gentime1)
    #
    #     reverted_length = ((results['length_birth_d'].iloc[ind] * pop_stds[1]) + pop_means[1])
    #     reverted_length1 = ((y_test['length_birth_d'].iloc[ind] * pop_stds[1]) + pop_means[1])
    #
    #     reverted_growth = ((results['length_birth_d'].iloc[ind] * pop_stds[3]) + pop_means[3])
    #     reverted_growth1 = ((y_test['length_birth_d'].iloc[ind] * pop_stds[3]) + pop_means[3])
    #
    #
    #     plt.plot(reverted_length*np.exp(reverted_growth * x), label='simulation', marker='.')
    #     plt.plot(reverted_length1*np.exp(reverted_growth1 * x1), label='data', marker='.')
    #     plt.legend()
    #     plt.show()


    ''' see if the simulated daughters contain similar same-cell correlations as the data '''
    # memory = []
    # print(results.columns)
    # for col1 in results.columns:
    #     for col2 in results.columns:
    #         if col1 != col2 and [col2, col1] not in memory:
    #
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
    #
    # exit()

    ''' the pearson correlations of the targets depending on each of the factors SIMULATION '''
    # for factor in factors:
    #     fig, axs = plt.subplots(1, len(targets), sharex=True)
    #     fig.text(0.5, 0.04, factor, ha='center')
    #     ind = 0
    #     for target in targets:
    #         m, c = np.linalg.lstsq(np.concatenate([np.array(X_test[factor])[:, np.newaxis], np.ones_like(X_test[factor])[:, np.newaxis]], axis=1), results[target])[0]
    #         axs[ind].scatter(X_test[factor], results[target], facecolors='none', edgecolors='blue')  # , label='_nolegend_'
    #         axs[ind].plot(np.linspace(min(X_test[factor]), max(X_test[factor])), np.linspace(min(X_test[factor]),
    #                                                                                                          max(X_test[factor])) * m + c,
    #                       label=r'$\rho=${}'.format(round(stats.pearsonr(results[target], X_test[factor])[0], 3)), color='orange')
    #         # label=r'$\rho$, pval: {}, {:.2e}'.format(round(stats.pearsonr(results[target], X_test[factor])[0], 3),
    #         # stats.pearsonr(results[target], X_test[factor])[1])
    #         # axs[ind].set_ylabel(target)
    #         axs[ind].set_ylim([-5, 10])
    #         axs[ind].set_xlim([-5, 10])
    #         axs[ind].legend()
    #         print(target, m, c)
    #         ind = ind + 1
    #     plt.suptitle('How ' + str(factor) + ' affects the other parameters in simulation data')
    #     plt.show()
    #     plt.close()


    ''' the pearson correlations of the simulated and real testing set USING MATRIX '''
    # fig, axs = plt.subplots(1, len(y_test.columns), sharey=True)
    # # fig.text(0.04, 0.5, 'Simulated', ha='center')
    # ind = 0
    # for col in y_test.columns:
    #     m, c = np.linalg.lstsq(np.concatenate([np.array(y_test[col])[:, np.newaxis], np.ones_like(y_test[col])
    #     [:, np.newaxis]], axis=1), results[col])[0]
    #     axs[ind].scatter(y_test[col], results[col], label='_nolegend_')
    #     axs[ind].plot(np.linspace(min(y_test[col]), max(y_test[col])), np.linspace(min(y_test[col]),
    #                                                                                max(y_test[col])) * m + c,
    #                   label=r'$\rho$, pval: {}, {:.2e}'.format(round(stats.pearsonr(y_test[col], results[col])[0], 3),
    #                                                            stats.pearsonr(y_test[col], results[col])[1]), color='orange')
    #     # axs[ind].hist(error_test[col], label=r'{:.2e}$\pm${:.2e}'.format(np.mean(error_test[col]), np.std(error_test[col])))
    #     axs[ind].set_ylim([np.mean(results[col]) - 3 * np.std(results[col]), np.mean(results[col]) + 3 * np.std(results[col])])
    #     axs[ind].set_xlim([np.mean(y_test[col]) - 3 * np.std(y_test[col]), np.mean(y_test[col]) + 3 * np.std(y_test[col])])
    #     axs[ind].legend()
    #     axs[ind].set_xlabel('data ' + col)
    #     axs[ind].set_ylabel('simulation ' + col)
    #     ind = ind + 1
    # plt.suptitle('correlation between real and simulated daughters in testing set')
    # plt.show()

    ''' the pearson correlations of the simulated and real testing set RANDOM SET'''
    # fig, axs = plt.subplots(1, len(y_test.columns), sharey=True)
    # # fig.text(0.04, 0.5, 'Simulated', ha='center')
    # ind = 0
    # for col in y_test.columns:
    #     m, c = np.linalg.lstsq(np.concatenate([np.array(y_test[col])[:, np.newaxis], np.ones_like(y_test[col])
    #     [:, np.newaxis]], axis=1), randomness[col])[0]
    #     axs[ind].scatter(y_test[col], randomness[col], label='_nolegend_')
    #     axs[ind].plot(np.linspace(min(y_test[col]), max(y_test[col])), np.linspace(min(y_test[col]),
    #                                                                                max(y_test[col])) * m + c,
    #                   label=r'$\rho$, pval: {}, {:.2e}'.format(round(stats.pearsonr(y_test[col], randomness[col])[0], 3),
    #                                                            stats.pearsonr(y_test[col], randomness[col])[1]), color='orange')
    #     # axs[ind].hist(error_test[col], label=r'{:.2e}$\pm${:.2e}'.format(np.mean(error_test[col]), np.std(error_test[col])))
    #     axs[ind].legend()
    #     axs[ind].set_xlabel('data ' + col)
    #     axs[ind].set_ylabel('random ' + col)
    #     ind = ind + 1
    # plt.suptitle('correlation between real and simulated daughters in testing set')
    # plt.show()


    ''' the delta distributions of the simulated and real testing set USING MATRIX '''
    #
    # error_test = y_test.sub(results, fill_value=0)
    #
    # # Now we will plot them
    # fig, axs = plt.subplots(1, len(error_test.columns), sharey=True)
    # fig.text(0.04, 0.5, 'PDF', ha='center')
    # ind = 0
    # for col in error_test.columns:
    #     axs[ind].hist(error_test[col], label=r'{:.2e}$\pm${:.2e}'.format(np.mean(error_test[col]), np.std(error_test[col])))
    #     axs[ind].legend()
    #     axs[ind].set_xlabel(col)
    #     ind=ind+1
    # plt.suptitle('Difference between real and simulated daughters in testing set')
    # plt.show()

    ''' the delta distributions of the simulated and real testing set RANDOM SET '''
    #
    # error_test = y_test.sub(randomness, fill_value=0)
    #
    # # Now we will plot them
    # fig, axs = plt.subplots(1, len(error_test.columns), sharey=True)
    # fig.text(0.04, 0.5, 'PDF', ha='center')
    # ind = 0
    # for col in error_test.columns:
    #     axs[ind].hist(error_test[col], label=r'{:.2e}$\pm${:.2e}'.format(np.mean(error_test[col]), np.std(error_test[col])))
    #     axs[ind].legend()
    #     axs[ind].set_xlabel(col)
    #     ind = ind + 1
    # plt.suptitle('Difference between real and simulated daughters in testing set')
    # plt.show()



    exit()




if __name__ == '__main__':
    main()
