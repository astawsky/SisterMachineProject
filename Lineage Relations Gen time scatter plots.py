
from __future__ import print_function

import numpy as np

import matplotlib.pyplot as plt

import pickle

import scipy.stats as stats

import pandas as pd


def dependanceOnNumOfGens():
    # Import the Refined Data
    pickle_in = open("metastructdata.pickle", "rb")
    struct = pickle.load(pickle_in)
    pickle_in.close()

    # This is to see if there is a correlation between the first gen measurement and the time average
    for observable in ['generationtime', 'growth_length', 'length_birth', 'length_final']:
        sis_array = []
        non_sis_array = []
        both_array = []
        
        sis_per_trap = np.array([[len(val[observable]), val[observable]] for val in struct.dict_with_all_sister_traces.values()])
        x_sis = np.array([sis_per_trap[ind][0] for ind in range(len(sis_per_trap))])
        y_sis = np.array([np.mean(sis_per_trap[ind][1]) for ind in range(len(sis_per_trap))])
        y_err_sis = np.array([np.std(sis_per_trap[ind][1]) for ind in range(len(sis_per_trap))])
        non_sis_per_trap = np.array([[len(val[observable]), val[observable]] for val in struct.dict_with_all_non_sister_traces.values()])
        x_non_sis = np.array([non_sis_per_trap[ind][0] for ind in range(len(non_sis_per_trap))])
        y_non_sis = np.array([np.mean(non_sis_per_trap[ind][1]) for ind in range(len(non_sis_per_trap))])
        y_err_non_sis = np.array([np.std(non_sis_per_trap[ind][1]) for ind in range(len(non_sis_per_trap))])

        # The value of the means per generation of recorded trace
        plt.errorbar(x=x_sis, y=y_sis, yerr=y_err_sis, label='sis', uplims=True, lolims=True, marker='.', fmt='o')
        plt.errorbar(x=x_non_sis, y=y_non_sis, yerr=y_err_non_sis, label='non_sis', uplims=True, lolims=True, marker='.', fmt='o')
        plt.legend()
        plt.xlabel('length of recorded traces [gen]')
        plt.ylabel('distribution of mean')
        plt.title(observable)
        plt.xlim([8, 30])
        plt.show()

        # # Does the mean change with respect to the length of the trace [gen]?
        sis_means = dict()
        for val in struct.dict_with_all_sister_traces.values():
            if len(val[observable]) in sis_means.keys():
                sis_means[len(val[observable])] = np.append(sis_means[len(val[observable])], np.array([np.mean(val[observable])]))
            else:
                sis_means[len(val[observable])] = np.array([np.mean(val[observable])])
                
        sis_means_array = []
        for ind in range(min(sis_means.keys()), max(sis_means.keys())):
            if ind in sis_means.keys():
                sis_means_array.append(sis_means[ind])
            
        non_sis_means = dict()
        for val in struct.dict_with_all_non_sister_traces.values():
            if len(val[observable]) in non_sis_means.keys():
                non_sis_means[len(val[observable])] = np.append(non_sis_means[len(val[observable])], np.array([np.mean(val[observable])]))
            else:
                non_sis_means[len(val[observable])] = np.array([np.mean(val[observable])])

        non_sis_means_array = []
        for ind in range(min(non_sis_means.keys()), max(non_sis_means.keys())):
            if ind in non_sis_means.keys():
                non_sis_means_array.append(non_sis_means[ind])
            
        # sis_means = {len(val[observable]):np.mean(val[observable]) for val in struct.dict_with_all_sister_traces.values()}
        # non_sis_means = {len(val[observable]): np.mean(val[observable]) for val in struct.dict_with_all_non_sister_traces.values()}
        # sis_means = np.array([[np.mean([]), np.std()]])
            
        # sis_means = np.array([[len(val[observable]), np.mean(val[observable])] for val in struct.dict_with_all_sis_traces.values()])
        # non_sis_means = np.array([[len(val[observable]), np.mean(val[observable])] for val in struct.dict_with_all_non_sis_traces.values()])

        for min_gen in range(1, 51):

            A_mean_array_sis = []
            B_mean_array_sis = []
            A_first_meas_sis = []
            B_first_meas_sis = []
            for keyA, keyB in zip(struct.A_dict_sis.keys(), struct.B_dict_sis.keys()):
                # min_gen = min(len(struct.A_dict_sis[keyA]), len(struct.B_dict_sis[keyB]))
                if min(len(struct.A_dict_sis[keyA][observable]), len(struct.B_dict_sis[keyB][observable])) >= min_gen:
                    A_mean_array_sis.append(np.mean(struct.A_dict_sis[keyA][observable][:min_gen]))
                    B_mean_array_sis.append(np.mean(struct.B_dict_sis[keyB][observable][:min_gen]))
                    A_first_meas_sis.append(struct.A_dict_sis[keyA][observable][0])
                    B_first_meas_sis.append(struct.B_dict_sis[keyB][observable][0])

            first_meas_sis = np.array(A_first_meas_sis) + np.array(B_first_meas_sis) / 2.0

            A_first_meas_sis = np.array(A_first_meas_sis)
            B_first_meas_sis = np.array(B_first_meas_sis)
            A_mean_array_sis = np.array(A_mean_array_sis)
            B_mean_array_sis = np.array(B_mean_array_sis)

            sis_array.append([A_mean_array_sis, B_mean_array_sis, A_first_meas_sis, B_first_meas_sis, first_meas_sis, stats.pearsonr(
                A_mean_array_sis, B_mean_array_sis)[0], np.hstack([A_mean_array_sis, B_mean_array_sis])])

            A_mean_array_non_sis = []
            B_mean_array_non_sis = []
            A_first_meas_non_sis = []
            B_first_meas_non_sis = []
            for keyA, keyB in zip(struct.A_dict_non_sis.keys(), struct.B_dict_non_sis.keys()):
                # min_gen = min(len(struct.A_dict_non_sis[keyA]), len(struct.B_dict_non_sis[keyB]))
                if min(len(struct.A_dict_non_sis[keyA][observable]), len(struct.B_dict_non_sis[keyB][observable])) >= min_gen:
                    A_mean_array_non_sis.append(np.mean(struct.A_dict_non_sis[keyA][observable][:min_gen]))
                    B_mean_array_non_sis.append(np.mean(struct.B_dict_non_sis[keyB][observable][:min_gen]))
                    A_first_meas_non_sis.append(struct.A_dict_non_sis[keyA][observable][0])
                    B_first_meas_non_sis.append(struct.B_dict_non_sis[keyB][observable][0])

            first_meas_non_sis = np.array(A_first_meas_non_sis) + np.array(B_first_meas_non_sis) / 2.0

            A_first_meas_non_sis = np.array(A_first_meas_non_sis)
            B_first_meas_non_sis = np.array(B_first_meas_non_sis)
            A_mean_array_non_sis = np.array(A_mean_array_non_sis)
            B_mean_array_non_sis = np.array(B_mean_array_non_sis)

            non_sis_array.append([A_mean_array_non_sis, B_mean_array_non_sis, A_first_meas_non_sis, B_first_meas_non_sis, first_meas_non_sis,
                                  stats.pearsonr(A_mean_array_non_sis, B_mean_array_non_sis)[0], np.hstack([A_mean_array_non_sis,
                                                                                                            B_mean_array_non_sis])])

            A_mean_array_both = []
            B_mean_array_both = []
            A_first_meas_both = []
            B_first_meas_both = []
            for keyA, keyB in zip(struct.A_dict_both.keys(), struct.B_dict_both.keys()):
                # min_gen = min(len(struct.A_dict_both[keyA]), len(struct.B_dict_both[keyB]))
                if min(len(struct.A_dict_both[keyA][observable]), len(struct.B_dict_both[keyB][observable])) >= min_gen:
                    A_mean_array_both.append(np.mean(struct.A_dict_both[keyA][observable][:min_gen]))
                    B_mean_array_both.append(np.mean(struct.B_dict_both[keyB][observable][:min_gen]))
                    A_first_meas_both.append(struct.A_dict_both[keyA][observable][0])
                    B_first_meas_both.append(struct.B_dict_both[keyB][observable][0])

            first_meas_both = np.array(A_first_meas_both) + np.array(B_first_meas_both) / 2.0

            A_first_meas_both = np.array(A_first_meas_both)
            B_first_meas_both = np.array(B_first_meas_both)
            A_mean_array_both = np.array(A_mean_array_both)
            B_mean_array_both = np.array(B_mean_array_both)

            both_array.append([A_mean_array_both, B_mean_array_both, A_first_meas_both, B_first_meas_both, first_meas_both, stats.pearsonr(
                A_mean_array_both, B_mean_array_both)[0]])

        sis_array = np.array(sis_array)
        non_sis_array = np.array(non_sis_array)
        both_array = np.array(both_array)

        # Pearson Correlations
        plt.plot(range(1, 51), [sis_array[ind][5] for ind in range(sis_array.shape[0])], marker='.', label='sis')
        plt.plot(range(1, 51), [non_sis_array[ind][5] for ind in range(non_sis_array.shape[0])], marker='.', label='non_sis')
        plt.plot(range(1, 51), [both_array[ind][5] for ind in range(both_array.shape[0])], marker='.', label='both')
        plt.xlabel('length of traces [gen]')
        plt.ylabel('pearson coefficient')
        plt.legend()
        plt.title(observable)
        plt.show()

        # The Values per vector length
        plt.errorbar(x=range(1, 51), y=[np.mean(sis_array[ind][-1]) for ind in range(sis_array.shape[0])], yerr=[np.std(sis_array[ind][-1]) for
                          ind in range(sis_array.shape[0])], marker='.', label='sis', uplims=True, lolims=True)
        plt.errorbar(x=range(1, 51), y=[np.mean(non_sis_array[ind][-1]) for ind in range(non_sis_array.shape[0])], yerr=[np.std(non_sis_array[ind][
                     -1]) for ind in range(non_sis_array.shape[0])], uplims=True, lolims=True, marker='.', label='non_sis')
        plt.legend()
        plt.xlabel('length of traces [gen]')
        plt.ylabel('distribution of value')
        plt.title(observable)
        plt.show()


def main():
    # Import the Refined Data
    pickle_in = open("metastructdata.pickle", "rb")
    struct = pickle.load(pickle_in)
    pickle_in.close()

    # To control for the length of trace parameter
    min_gen = 30

    # This is to see if there is a correlation between the first gen measurement and the time average
    for observable in ['generationtime', 'growth_length', 'length_birth', 'length_final']:
        A_mean_array_sis = []
        B_mean_array_sis = []
        A_first_meas_sis = []
        B_first_meas_sis = []
        for keyA, keyB in zip(struct.A_dict_sis.keys(), struct.B_dict_sis.keys()):
            # min_gen = min(len(struct.A_dict_sis[keyA]), len(struct.B_dict_sis[keyB]))
            if min(len(struct.A_dict_sis[keyA][observable]), len(struct.B_dict_sis[keyB][observable])) >= min_gen:
                A_mean_array_sis.append(np.mean(struct.A_dict_sis[keyA][observable][:min_gen]))
                B_mean_array_sis.append(np.mean(struct.B_dict_sis[keyB][observable][:min_gen]))
                A_first_meas_sis.append(struct.A_dict_sis[keyA][observable][0])
                B_first_meas_sis.append(struct.B_dict_sis[keyB][observable][0])
            
        first_meas_sis = np.array(A_first_meas_sis) + np.array(B_first_meas_sis) / 2.0

        A_first_meas_sis = np.array(A_first_meas_sis)
        B_first_meas_sis = np.array(B_first_meas_sis)
        A_mean_array_sis = np.array(A_mean_array_sis)
        B_mean_array_sis = np.array(B_mean_array_sis)

        A_mean_array_non_sis = []
        B_mean_array_non_sis = []
        A_first_meas_non_sis = []
        B_first_meas_non_sis = []
        for keyA, keyB in zip(struct.A_dict_non_sis.keys(), struct.B_dict_non_sis.keys()):
            # min_gen = min(len(struct.A_dict_non_sis[keyA]), len(struct.B_dict_non_sis[keyB]))
            if min(len(struct.A_dict_non_sis[keyA][observable]), len(struct.B_dict_non_sis[keyB][observable])) >= min_gen:
                A_mean_array_non_sis.append(np.mean(struct.A_dict_non_sis[keyA][observable][:min_gen]))
                B_mean_array_non_sis.append(np.mean(struct.B_dict_non_sis[keyB][observable][:min_gen]))
                A_first_meas_non_sis.append(struct.A_dict_non_sis[keyA][observable][0])
                B_first_meas_non_sis.append(struct.B_dict_non_sis[keyB][observable][0])

        first_meas_non_sis = np.array(A_first_meas_non_sis) + np.array(B_first_meas_non_sis) / 2.0

        A_first_meas_non_sis = np.array(A_first_meas_non_sis)
        B_first_meas_non_sis = np.array(B_first_meas_non_sis)
        A_mean_array_non_sis = np.array(A_mean_array_non_sis)
        B_mean_array_non_sis = np.array(B_mean_array_non_sis)

        A_mean_array_both = []
        B_mean_array_both = []
        A_first_meas_both = []
        B_first_meas_both = []
        for keyA, keyB in zip(struct.A_dict_both.keys(), struct.B_dict_both.keys()):
            # min_gen = min(len(struct.A_dict_both[keyA]), len(struct.B_dict_both[keyB]))
            if min(len(struct.A_dict_both[keyA][observable]), len(struct.B_dict_both[keyB][observable])) >= min_gen:
                A_mean_array_both.append(np.mean(struct.A_dict_both[keyA][observable][:min_gen]))
                B_mean_array_both.append(np.mean(struct.B_dict_both[keyB][observable][:min_gen]))
                A_first_meas_both.append(struct.A_dict_both[keyA][observable][0])
                B_first_meas_both.append(struct.B_dict_both[keyB][observable][0])

        first_meas_both = np.array(A_first_meas_both) + np.array(B_first_meas_both) / 2.0

        A_first_meas_both = np.array(A_first_meas_both)
        B_first_meas_both = np.array(B_first_meas_both)
        A_mean_array_both = np.array(A_mean_array_both)
        B_mean_array_both = np.array(B_mean_array_both)

        plt.scatter(A_mean_array_sis, B_mean_array_sis, facecolors='none', label='Sisters, pearson coeff={:.2e}, {} samples'.format(
            stats.pearsonr(A_mean_array_sis, B_mean_array_sis)[
                0], len(A_mean_array_sis)), edgecolors='blue')
        plt.scatter(A_mean_array_non_sis, B_mean_array_non_sis, facecolors='none',
                    label='Non-Sisters, pearson coeff={:.2e}, {} samples'.format(
                        stats.pearsonr(A_mean_array_non_sis, B_mean_array_non_sis)[0], len(A_mean_array_non_sis)),
                    edgecolors='orange')
        plt.scatter(A_mean_array_both, B_mean_array_both, facecolors='none',
                    label='Control, pearson coeff={:.2e}, {} samples'.format(
                        stats.pearsonr(A_mean_array_both, B_mean_array_both)[
                            0], len(A_mean_array_both)), edgecolors='green')
        title = str(observable) + ' averaged over '+str(min_gen)+' generations in the trap'
        plt.title(title)
        plt.xlabel('Mean(A)')
        plt.ylabel('Mean(B)')
        # if observable == 'generationtime':
        #     plt.xlim([.45, .8])
        #     plt.ylim([.45, .8])
        # elif observable == 'length_birth':
        #     plt.xlim([2, 3.5])
        #     plt.ylim([2, 3.5])
        # elif observable == 'length_final':
        #     plt.xlim([4, 6.8])
        #     plt.ylim([4, 6.8])
        plt.legend()
        plt.show()
        # plt.savefig(title+'.png', dpi=300)
        # plt.close()

        plt.scatter(A_mean_array_sis - A_first_meas_sis, B_mean_array_sis - B_first_meas_sis, facecolors='none', label='Sisters, pearson coeff={'
                     ':.2e}, {} samples'.format(stats.pearsonr(A_mean_array_sis - A_first_meas_sis, B_mean_array_sis - B_first_meas_sis)[
                                                                                                                           0], len(A_mean_array_sis)),
                    edgecolors='blue')
        plt.scatter(A_mean_array_non_sis - A_first_meas_non_sis, B_mean_array_non_sis - B_first_meas_non_sis, facecolors='none',
                    label='Non-Sisters, pearson coeff={:.2e}, {} samples'.format(stats.pearsonr(A_mean_array_non_sis - A_first_meas_non_sis,
                                                                                                  B_mean_array_non_sis - B_first_meas_non_sis)[0],
                                                                                  len(A_mean_array_non_sis)
                                                                                  ), edgecolors='orange')
        plt.scatter(A_mean_array_both - A_first_meas_both, B_mean_array_both - B_first_meas_both, facecolors='none',
                    label='Control, pearson coeff={:.2e}, {} samples'.format(stats.pearsonr(A_mean_array_both - A_first_meas_both, B_mean_array_both - B_first_meas_both)[
                                                                                                                            0],
                                                                              len(A_mean_array_both)),
                    edgecolors='green')
        title = str(observable) + ' averaged over generations in the trap'
        plt.title(title)
        plt.xlabel('Mean(A) - A[0]')
        plt.ylabel('Mean(B) - B[0]')
        # if observable == 'generationtime':
        #     plt.xlim([.45, .8])
        #     plt.ylim([.45, .8])
        # elif observable == 'length_birth':
        #     plt.xlim([2, 3.5])
        #     plt.ylim([2, 3.5])
        # elif observable == 'length_final':
        #     plt.xlim([4, 6.8])
        #     plt.ylim([4, 6.8])
        plt.legend()
        plt.show()
        # plt.savefig(title+'.png', dpi=300)
        # plt.close()

        plt.scatter(A_mean_array_sis - first_meas_sis, B_mean_array_sis - first_meas_sis, facecolors='none', label='Sisters, pearson coeff={'
                      ':.2e}, {} samples'.format(stats.pearsonr(A_mean_array_sis - first_meas_sis, B_mean_array_sis - first_meas_sis)[0], len(A_mean_array_sis)),
                    edgecolors='blue')
        plt.scatter(A_mean_array_non_sis - first_meas_non_sis, B_mean_array_non_sis - first_meas_non_sis, facecolors='none',
                    label='Non-Sisters, pearson coeff={:.2e}, {} samples'.format(stats.pearsonr(A_mean_array_non_sis - first_meas_non_sis,
                                                                                                  B_mean_array_non_sis - first_meas_non_sis)[0],
                                                                                  len(A_mean_array_non_sis)), edgecolors='orange')
        plt.scatter(A_mean_array_both - first_meas_both, B_mean_array_both - first_meas_both, facecolors='none',
                    label='Control, pearson coeff={:.2e}, {} samples'.format(stats.pearsonr(A_mean_array_both - first_meas_both, B_mean_array_both - first_meas_both)[
                                                                     0], len(A_mean_array_both)), edgecolors='green')
        title = str(observable) + ' averaged over generations in the trap'
        plt.title(title)
        plt.xlabel('Mean(A) - (A[0]+B[0])/2')
        plt.ylabel('Mean(B) - (A[0]+B[0])/2')
        # if observable == 'generationtime':
        #     plt.xlim([.45, .8])
        #     plt.ylim([.45, .8])
        # elif observable == 'length_birth':
        #     plt.xlim([2, 3.5])
        #     plt.ylim([2, 3.5])
        # elif observable == 'length_final':
        #     plt.xlim([4, 6.8])
        #     plt.ylim([4, 6.8])
        plt.legend()
        plt.show()
        # plt.savefig(title+'.png', dpi=300)
        # plt.close()

    # # This is to see the correlation between the time-averaged mean of A and B
    # for observable in ['generationtime', 'growth_length', 'length_birth', 'length_final']:
    #     A_mean_array_sis = []
    #     B_mean_array_sis = []
    #     for keyA, keyB in zip(struct.A_dict_sis.keys(), struct.B_dict_sis.keys()):
    #         A_mean_array_sis.append(np.mean(struct.A_dict_sis[keyA][observable]))
    #         B_mean_array_sis.append(np.mean(struct.B_dict_sis[keyB][observable]))
    #
    #     A_mean_array_non_sis = []
    #     B_mean_array_non_sis = []
    #     for keyA, keyB in zip(struct.A_dict_non_sis.keys(), struct.B_dict_non_sis.keys()):
    #         A_mean_array_non_sis.append(np.mean(struct.A_dict_non_sis[keyA][observable]))
    #         B_mean_array_non_sis.append(np.mean(struct.B_dict_non_sis[keyB][observable]))
    #
    #     A_mean_array_both = []
    #     B_mean_array_both = []
    #     for keyA, keyB in zip(struct.A_dict_both.keys(), struct.B_dict_both.keys()):
    #         A_mean_array_both.append(np.mean(struct.A_dict_both[keyA][observable]))
    #         B_mean_array_both.append(np.mean(struct.B_dict_both[keyB][observable]))
    #
    #     plt.scatter(A_mean_array_sis, B_mean_array_sis, label='Sisters, pearson coeff={:.2e}'.format(stats.pearsonr(A_mean_array_sis,
    #                                                                                                                 B_mean_array_sis)[0]))
    #     plt.scatter(A_mean_array_non_sis, B_mean_array_non_sis, label='Non-Sisters, pearson coeff={:.2e}'.format(stats.pearsonr(A_mean_array_non_sis,
    #                                                                                                                      B_mean_array_non_sis)[0]))
    #     plt.scatter(A_mean_array_both, B_mean_array_both, label='Control, pearson coeff={:.2e}'.format(stats.pearsonr(A_mean_array_both,
    #                                                                                                                   B_mean_array_both)[0]))
    #     title = str(observable)+' averaged over generations in the trap'
    #     plt.title(title)
    #     plt.xlabel('A trace')
    #     plt.ylabel('B trace')
    #     if observable == 'generationtime':
    #         plt.xlim([.45, .8])
    #         plt.ylim([.45, .8])
    #     elif observable == 'length_birth':
    #         plt.xlim([2, 3.5])
    #         plt.ylim([2, 3.5])
    #     elif observable == 'length_final':
    #         plt.xlim([4, 6.8])
    #         plt.ylim([4, 6.8])
    #     plt.legend()
    #     # plt.show()
    #     plt.savefig(title+'.png', dpi=300)
    #     plt.close()
    #
    #
    # # THIS IS FOR PHI=GENTIME*ALPHA
    # A_mean_array_sis = []
    # B_mean_array_sis = []
    # for keyA, keyB in zip(struct.A_dict_sis.keys(), struct.B_dict_sis.keys()):
    #     A_mean_array_sis.append(np.mean(struct.A_dict_sis[keyA]['generationtime']*np.mean(struct.A_dict_sis[keyA]['growth_length'])))
    #     B_mean_array_sis.append(np.mean(struct.B_dict_sis[keyB]['generationtime']*np.mean(struct.B_dict_sis[keyB]['growth_length'])))
    #
    # A_mean_array_non_sis = []
    # B_mean_array_non_sis = []
    # for keyA, keyB in zip(struct.A_dict_non_sis.keys(), struct.B_dict_non_sis.keys()):
    #     A_mean_array_non_sis.append(np.mean(struct.A_dict_non_sis[keyA]['generationtime'] * np.mean(struct.A_dict_non_sis[keyA]['growth_length'])))
    #     B_mean_array_non_sis.append(np.mean(struct.B_dict_non_sis[keyB]['generationtime'] * np.mean(struct.B_dict_non_sis[keyB]['growth_length'])))
    #
    # A_mean_array_both = []
    # B_mean_array_both = []
    # for keyA, keyB in zip(struct.A_dict_both.keys(), struct.B_dict_both.keys()):
    #     A_mean_array_both.append(np.mean(struct.A_dict_both[keyA]['generationtime'] * np.mean(struct.A_dict_both[keyA]['growth_length'])))
    #     B_mean_array_both.append(np.mean(struct.B_dict_both[keyB]['generationtime'] * np.mean(struct.B_dict_both[keyB]['growth_length'])))
    #
    # plt.scatter(A_mean_array_sis, B_mean_array_sis, label='Sisters, pearson coeff={:.2e}'.format(stats.pearsonr(A_mean_array_sis,
    #                                                                                                             B_mean_array_sis)[0]))
    # plt.scatter(A_mean_array_non_sis, B_mean_array_non_sis, label='Non-Sisters, pearson coeff={:.2e}'.format(stats.pearsonr(A_mean_array_non_sis,
    #                                                                                                                         B_mean_array_non_sis)[0]))
    # plt.scatter(A_mean_array_both, B_mean_array_both, label='Control, pearson coeff={:.2e}'.format(stats.pearsonr(A_mean_array_both,
    #                                                                                                               B_mean_array_both)[0]))
    # title = 'phi averaged over generations in the trap'
    # plt.title(title)
    # plt.xlabel('A trace')
    # plt.ylabel('B trace')
    # plt.xlim([.65,.85])
    # plt.ylim([.65, .85])
    # plt.legend()
    # # plt.show()
    # plt.savefig(title + '.png', dpi=300)
    # plt.close()
    

    # # THIS IS FOR THE MOTHER DAUGHTER TRACES
    # sis_d = np.array([np.array([daughter for daughter, mother in zip(trace['generationtime'].iloc[:-1], trace['generationtime'].iloc[
    #                    1:])]).flatten() for trace in struct.dict_with_all_sister_traces.values()]).flatten()
    # sis_m = np.array([np.array([mother for daughter, mother in zip(trace['generationtime'].iloc[:-1], trace['generationtime'].iloc[
    #                    1:])]) for trace in struct.dict_with_all_sister_traces.values()]).ravel('K')
    # # sis_md = np.array([np.array([np.array([mother, daughter]).flatten() for daughter, mother in zip(trace['generationtime'].iloc[:-1],
    # #                    trace['generationtime'].iloc[1:])]) for trace in struct.dict_with_all_sister_traces.values()])
    #
    # # print(sis_md)
    #
    # sis_m = []
    # sis_d = []
    # for trace in struct.dict_with_all_sister_traces.values():
    #     for daughter, mother in zip(trace['generationtime'].iloc[:-1], trace['generationtime'].iloc[1:]):
    #         sis_m.append(mother)
    #         sis_d.append(daughter)
    #
    # non_sis_m = []
    # non_sis_d = []
    # for trace in struct.dict_with_all_non_sister_traces.values():
    #     for daughter, mother in zip(trace['generationtime'].iloc[:-1], trace['generationtime'].iloc[1:]):
    #         non_sis_m.append(mother)
    #         non_sis_d.append(daughter)
    #
    # both_m = []
    # both_d = []
    # for trace in struct.dict_with_all_both_traces.values():
    #     for daughter, mother in zip(trace['generationtime'].iloc[:-1], trace['generationtime'].iloc[1:]):
    #         both_m.append(mother)
    #         both_d.append(daughter)
    #
    # plt.scatter(sis_m, sis_d, label='Sis, pearson r = {:.2e}'.format(stats.pearsonr(sis_m, sis_d)[0]), color='b')
    # plt.scatter(non_sis_m, non_sis_d, label='Nonsis, pearson r = {:.2e}'.format(stats.pearsonr(sis_m, sis_d)[0]), color='orange')
    # plt.scatter(both_m, both_d, label='Control, pearson r = {:.2e}'.format(stats.pearsonr(sis_m, sis_d)[0]), color='green')
    # plt.xlabel('mother')
    # plt.ylabel('daughter')
    # plt.legend()
    # plt.show()

    # A_sis_array = []
    # for observable in ['generationtime', 'growth_length', 'length_birth', 'length_final']:
    #     # for key in struct.A_dict_sis.keys():
    #     #     max_gen = min(len(struct.A_dict_sis[key][observable]), len(struct.B_dict_sis[key][observable]))
    #     #     for gen in :
    # 
    #     max_gen = 20
    # 
    #     sis_pearsons_coeff = []
    #     sis_pearsons_pval = []
    #     non_sis_pearsons_coeff = []
    #     non_sis_pearsons_pval = []
    #     both_pearsons_coeff = []
    #     both_pearsons_pval = []
    #     # sis_pearsons_coeff = []
    #     # sis_pearsons_pval = []
    #     # non_sis_pearsons_coeff = []
    #     # non_sis_pearsons_pval = []
    #     # both_pearsons_coeff = []
    #     # both_pearsons_pval = []
    #     for gen in range(max_gen):
    # 
    #         A_sis_array = np.array([struct.A_dict_sis[keyA][observable][gen] for keyA, keyB in zip(struct.A_dict_sis.keys(),
    #                                                                                                struct.B_dict_sis.keys()) if min(len(
    #             struct.A_dict_sis[keyA][observable]),len(struct.B_dict_sis[keyB][observable])) > gen])
    #         B_sis_array = np.array([struct.B_dict_sis[keyB][observable][gen] for keyA, keyB in zip(struct.A_dict_sis.keys(),
    #                                                                                                struct.B_dict_sis.keys()) if min(len(
    #             struct.A_dict_sis[keyA][observable]), len(struct.B_dict_sis[keyB][observable])) > gen])
    #         A_non_sis_array = np.array([struct.A_dict_non_sis[keyA][observable][gen] for keyA, keyB in zip(struct.A_dict_non_sis.keys(),
    #                                                                                                struct.B_dict_non_sis.keys()) if min(len(
    #             struct.A_dict_non_sis[keyA][observable]), len(struct.B_dict_non_sis[keyB][observable])) > gen])
    #         B_non_sis_array = np.array([struct.B_dict_non_sis[keyB][observable][gen] for keyA, keyB in zip(struct.A_dict_non_sis.keys(),
    #                                                                                                struct.B_dict_non_sis.keys()) if min(len(
    #             struct.A_dict_non_sis[keyA][observable]), len(struct.B_dict_non_sis[keyB][observable])) > gen])
    #         A_both_array = np.array([struct.A_dict_both[keyA][observable][gen] for keyA, keyB in zip(struct.A_dict_both.keys(),
    #                                                                                                struct.B_dict_both.keys()) if min(len(
    #             struct.A_dict_both[keyA][observable]), len(struct.B_dict_both[keyB][observable])) > gen])
    #         B_both_array = np.array([struct.B_dict_both[keyB][observable][gen] for keyA, keyB in zip(struct.A_dict_both.keys(),
    #                                                                                                struct.B_dict_both.keys()) if min(len(
    #             struct.A_dict_both[keyA][observable]), len(struct.B_dict_both[keyB][observable])) > gen])
    # 
    #         # plt.scatter(A_sis_array, B_sis_array, label='sis, {:.2e}, {:.2e}'.format(stats.pearsonr(A_sis_array, B_sis_array)[0], stats.pearsonr(A_sis_array, B_sis_array)[0]))
    #         # plt.scatter(A_non_sis_array, B_non_sis_array, label='non_sis, {:.2e}, {:.2e}'.format(stats.pearsonr(A_non_sis_array, B_non_sis_array)[0], stats.pearsonr(A_non_sis_array, B_non_sis_array)[0]))
    #         # plt.scatter(A_both_array, B_both_array, label='both, {:.2e}, {:.2e}'.format(stats.pearsonr(A_both_array, B_both_array)[0], stats.pearsonr(A_both_array, B_both_array)[0]))
    #         # plt.legend()
    #         # plt.xlabel('A trace')
    #         # plt.ylabel('B trace')
    #         # plt.title(str(gen+1)+', '+observable)
    #         # plt.show()
    # 
    #         sis_pearsons_coeff.append(stats.pearsonr(A_sis_array, B_sis_array)[0])
    #         sis_pearsons_pval.append(stats.pearsonr(A_sis_array, B_sis_array)[1])
    #         non_sis_pearsons_coeff.append(stats.pearsonr(A_non_sis_array, B_non_sis_array)[0])
    #         non_sis_pearsons_pval.append(stats.pearsonr(A_non_sis_array, B_non_sis_array)[1])
    #         both_pearsons_coeff.append(stats.pearsonr(A_both_array, B_both_array)[0])
    #         both_pearsons_pval.append(stats.pearsonr(A_both_array, B_both_array)[1])
    #         # sis_pearsons_coeff.append(stats.pearsonr(A_sis_array, B_sis_array)[0])
    #         # sis_pearsons_pval.append(stats.pearsonr(A_sis_array, B_sis_array)[1])
    #         # non_sis_pearsons_coeff.append(stats.pearsonr(A_non_sis_array, B_non_sis_array)[0])
    #         # non_sis_pearsons_pval.append(stats.pearsonr(A_non_sis_array, B_non_sis_array)[1])
    #         # both_pearsons_coeff.append(stats.pearsonr(A_both_array, B_both_array)[0])
    #         # both_pearsons_pval.append(stats.pearsonr(A_both_array, B_both_array)[1])
    # 
    #     plt.plot(sis_pearsons_coeff, label='sis', marker='.')
    #     plt.plot(non_sis_pearsons_coeff, label='non_sis', marker='.')
    #     plt.plot(both_pearsons_coeff, label='both', marker='.')
    #     # plt.plot(sis_pearsons_coeff, label='sis, pearson', marker='.')
    #     # plt.plot(non_sis_pearsons_coeff, label='non_sis, pearson', marker='.')
    #     # plt.plot(both_pearsons_coeff, label='both, pearson', marker='.')
    #     plt.xlabel('Generation')
    #     plt.ylabel('Spearman coefficient of A and B pooled over traps')
    #     plt.title(observable)
    #     plt.legend()
    #     plt.show()
    # 
    #     plt.plot(sis_pearsons_pval, label='sis', marker='.')
    #     plt.plot(non_sis_pearsons_pval, label='non_sis', marker='.')
    #     plt.plot(both_pearsons_pval, label='both', marker='.')
    #     # plt.plot(sis_pearsons_pval, label='sis, pearson', marker='.')
    #     # plt.plot(non_sis_pearsons_pval, label='non_sis, pearson', marker='.')
    #     # plt.plot(both_pearsons_pval, label='both, pearson', marker='.')
    #     plt.xlabel('Generation')
    #     plt.ylabel('Spearman pval of A and B pooled over traps')
    #     plt.title(observable)
    #     plt.legend()
    #     plt.show()




if __name__ == '__main__':
    # main()
    dependanceOnNumOfGens()
