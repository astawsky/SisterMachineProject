from __future__ import print_function

import numpy as np
import argparse
import sys, math
import glob
import matplotlib.pyplot as plt

import pickle
import matplotlib.patches as mpatches

import sistercellclass as ssc

import CALCULATETHEBETAS
import os

import scipy.stats as stats
import random


def Distribution(diffs_sis, diffs_non_sis, diffs_both, xlabel, Nbins, abs_diff_range):
    # PoolID == 1
    sis_label = 'Sister ' + r'$\mu=$' + '{:.2e}'.format(np.mean(diffs_sis)) + r', $\sigma=$' + '{:.2e}'.format(
        np.var(diffs_sis))
    non_label = 'Non Sister ' + r'$\mu=$' + '{:.2e}'.format(
        np.mean(diffs_non_sis)) + r', $\sigma=$' + '{:.2e}'.format(np.var(diffs_non_sis))
    both_label = 'Control ' + r'$\mu=$' + '{:.2e}'.format(np.mean(diffs_both)) + r', $\sigma=$' + '{:.2e}'.format(
        np.var(diffs_both))

    arr_sis = plt.hist(x=diffs_sis, label=sis_label, weights=np.ones_like(diffs_sis) / float(len(diffs_sis)), bins=Nbins, range=[0, abs_diff_range])
    arr_non_sis = plt.hist(x=diffs_non_sis, label=non_label,
                           weights=np.ones_like(diffs_non_sis) / float(len(diffs_non_sis)), bins=Nbins, range=[0, abs_diff_range])
    arr_both = plt.hist(x=diffs_both, label=both_label, weights=np.ones_like(diffs_both) / float(len(diffs_both)), bins=Nbins, range=[0, abs_diff_range])
    plt.close()

    # print('arr_sis[0]:', arr_sis[0])
    # print('arr_non_sis[0]:', arr_non_sis[0])
    # print('arr_both[0]:', arr_both[0])

    plt.plot(np.array([(arr_sis[1][l] + arr_sis[1][l + 1]) / 2. for l in range(len(arr_sis[1]) - 1)]), arr_sis[0], label=sis_label, marker='.')
    plt.plot(np.array([(arr_non_sis[1][l] + arr_non_sis[1][l + 1]) / 2. for l in range(len(arr_non_sis[1]) - 1)]), arr_non_sis[0], label=non_label, marker='.')
    plt.plot(np.array([(arr_both[1][l] + arr_both[1][l + 1]) / 2. for l in range(len(arr_both[1]) - 1)]), arr_both[0], label=both_label, marker='.')
    plt.xlabel(xlabel)
    plt.ylabel('PDF (Weighted Histogram)')
    plt.legend()
    plt.show()


def main():
    # Import the Refined Data
    pickle_in = open("metastructdata.pickle", "rb")
    metadatastruct = pickle.load(pickle_in)
    pickle_in.close()

    x_avg = metadatastruct.x_avg

    cycles_using = 20

    differences = []

    list_of_params = ['ln(x_n/x*)', 'growth_length', 'generationtime', 'phi', 'ln(f)', 'beta', 'phi*',
                      'Pcorr of ln(f) vs phi']
    param = list_of_params[7]

    for A_dict, B_dict in zip([metadatastruct.A_dict_sis, metadatastruct.A_dict_non_sis, metadatastruct.A_dict_both],
                              [metadatastruct.B_dict_sis, metadatastruct.B_dict_non_sis, metadatastruct.B_dict_both]):

        if param == 'ln(x_n/x*)':
            A_averaged_length_array = np.array([np.mean(np.log(A_dict[idA]['length_birth'][:cycles_using]/x_avg)) for idA, idB in
                                               zip(A_dict.keys(), B_dict.keys()) if min(len(A_dict[idA]), len(B_dict[idB]))])

            B_averaged_length_array = np.array([np.mean(np.log(B_dict[idB]['length_birth'][:cycles_using] / x_avg)) for idA, idB in
                                               zip(A_dict.keys(), B_dict.keys()) if
                                               min(len(A_dict[idA]), len(B_dict[idB]))])
            xlabel = r'$|ln(\frac{x_n^A}{x^*})-ln(\frac{x_n^B}{x^*})|$'
            abs_diff_range = .4

        if param == 'growth_length':
            A_averaged_length_array = np.array(
                [np.mean(A_dict[idA]['growth_length'][:cycles_using]) for idA, idB in
                 zip(A_dict.keys(), B_dict.keys()) if min(len(A_dict[idA]), len(B_dict[idB]))])

            B_averaged_length_array = np.array(
                [np.mean(B_dict[idB]['growth_length'][:cycles_using]) for idA, idB in
                 zip(A_dict.keys(), B_dict.keys()) if min(len(A_dict[idA]), len(B_dict[idB]))])
            xlabel = r'$|<\alpha^A>_n-<\alpha^B>_n|$'
            abs_diff_range = .4

        if param == 'generationtime':
            A_averaged_length_array = np.array(
                [np.mean(A_dict[idA]['generationtime'][:cycles_using]) for idA, idB in
                 zip(A_dict.keys(), B_dict.keys()) if min(len(A_dict[idA]), len(B_dict[idB]))])

            B_averaged_length_array = np.array(
                [np.mean(B_dict[idB]['generationtime'][:cycles_using]) for idA, idB in
                 zip(A_dict.keys(), B_dict.keys()) if min(len(A_dict[idA]), len(B_dict[idB]))])
            xlabel = r'$|<T^A>_n-<T^B>_n|$'
            abs_diff_range = .2

        if param == 'phi':
            A_averaged_length_array = np.array(
                [np.mean(A_dict[idA]['generationtime'][:cycles_using] * A_dict[idA]['growth_length'][:cycles_using]) for idA, idB in
                 zip(A_dict.keys(), B_dict.keys()) if min(len(A_dict[idA]), len(B_dict[idB]))])

            B_averaged_length_array = np.array(
                [np.mean(B_dict[idB]['generationtime'][:cycles_using] * B_dict[idB]['growth_length'][:cycles_using]) for idA, idB in
                 zip(A_dict.keys(), B_dict.keys()) if min(len(A_dict[idA]), len(B_dict[idB]))])
            xlabel = r'$|<\phi^A>_n-<\phi^B>_n|$'
            abs_diff_range = .15

        if param == 'ln(f)':
            A_averaged_length_array = np.array(
                [np.mean(A_dict[idA]['division_ratios__f_n'][:cycles_using]) for idA, idB in
                 zip(A_dict.keys(), B_dict.keys()) if min(len(A_dict[idA]), len(B_dict[idB]))])

            B_averaged_length_array = np.array(
                [np.mean(B_dict[idB]['division_ratios__f_n'][:cycles_using]) for idA, idB in
                 zip(A_dict.keys(), B_dict.keys()) if min(len(A_dict[idA]), len(B_dict[idB]))])
            xlabel = r'$|<ln(f^A)>_n-<ln(f^B)>_n|$'
            abs_diff_range = .05

        if param == 'Pcorr of ln(f) vs phi': # DOESN'T WORK YET!!! NEED TO PUT THE FOR LOOP OUTSIDE THE LIST

            A_averaged_length_array = np.array([np.mean(A_dict[idA]['generationtime'][:cycles_using] * A_dict
                            [idA]['growth_length'][:cycles_using]) for idA, idB in zip(A_dict.keys(),
                            B_dict.keys()) if min(len(A_dict[idA]), len(B_dict[idB]))]), np.array([np.mean(np.log(A_dict
                            [idA]['division_ratios__f_n'][:cycles_using])) for idA, idB in zip(A_dict.keys(),
                            B_dict.keys()) if min(len(A_dict[idA]), len(B_dict[idB]))])

            B_averaged_length_array = np.array([np.mean(B_dict[idB]['generationtime'][:cycles_using] * B_dict
                            [idB]['growth_length'][:cycles_using])for idA, idB in zip(A_dict.keys(),
                            B_dict.keys()) if min(len(A_dict[idA]), len(B_dict[idB]))]), np.array([np.mean(np.log(B_dict
                            [idB]['division_ratios__f_n'][:cycles_using])) for idA, idB in zip(A_dict.keys(),
                            B_dict.keys()) if min(len(A_dict[idA]), len(B_dict[idB]))])

            xlabel = r'$|pcorr^A - pcorr^B|$, where pcorr = pearson(phi, ln(f))'
            abs_diff_range = 2
            
        if param == 'beta':

            A_slope_array = []
            B_slope_array = []

            for idA, idB in zip(A_dict.keys(), B_dict.keys()):
                if min(len(A_dict[idA]), len(B_dict[idB])):
                    slope_A, intercept_A, r_value_A, p_value_A, std_err_A = stats.linregress( np.array(np.log(A_dict
                                [idA]['length_birth'][:cycles_using]/x_avg)), np.array(A_dict[idA]['generationtime']
                                [:cycles_using] * A_dict[idA]['growth_length'][:cycles_using]) )

                    slope_B, intercept_B, r_value_B, p_value_B, std_err_B = stats.linregress( np.array(np.log(B_dict
                                [idB]['length_birth'][:cycles_using]/x_avg)), np.array(B_dict[idB]['generationtime']
                                [:cycles_using] * B_dict[idB]['growth_length'][:cycles_using]) )
                    A_slope_array.append(slope_A)
                    B_slope_array.append(slope_B)

            A_averaged_length_array = np.array(A_slope_array)
            B_averaged_length_array = np.array(B_slope_array)
            xlabel = r'$|\beta^A - \beta^B|$'
            abs_diff_range = .9

        if param == 'phi*':

            A_intercept_array = []
            B_intercept_array = []

            for idA, idB in zip(A_dict.keys(), B_dict.keys()):
                if min(len(A_dict[idA]), len(B_dict[idB])):
                    slope_A, intercept_A, r_value_A, p_value_A, std_err_A = stats.linregress( np.array(np.log(A_dict
                                [idA]['length_birth'][:cycles_using]/x_avg)), np.array(A_dict[idA]['generationtime']
                                [:cycles_using] * A_dict[idA]['growth_length'][:cycles_using]) )

                    slope_B, intercept_B, r_value_B, p_value_B, std_err_B = stats.linregress( np.array(np.log(B_dict
                                [idB]['length_birth'][:cycles_using]/x_avg)), np.array(B_dict[idB]['generationtime']
                                [:cycles_using] * B_dict[idB]['growth_length'][:cycles_using]) )
                    A_intercept_array.append(intercept_A)
                    B_intercept_array.append(intercept_B)

            A_averaged_length_array = np.array(A_intercept_array)
            B_averaged_length_array = np.array(B_intercept_array)
            xlabel = r'$|\phi*_A - \phi*_B|$'
            abs_diff_range = .3

        differences.append(np.abs(A_averaged_length_array - B_averaged_length_array))

    Nbins = 10

    Distribution(diffs_sis=differences[0], diffs_non_sis=differences[1], diffs_both=differences[2],
                 xlabel=xlabel, Nbins=Nbins, abs_diff_range=abs_diff_range)

    Nbins = 15

    Distribution(diffs_sis=differences[0], diffs_non_sis=differences[1], diffs_both=differences[2],
                 xlabel=xlabel, Nbins=Nbins, abs_diff_range=abs_diff_range)


if __name__ == '__main__':
    main()
