
from __future__ import print_function

import numpy as np

import matplotlib.pyplot as plt

import pickle


def main():

    # Import the Refined Data
    pickle_in = open("metastructdata.pickle", "rb")
    struct = pickle.load(pickle_in)
    pickle_in.close()

    ### Here we compute the fraction overlaps for each experiment and dataset and put them in s/n/b_frac arrays
    sis_frac = []
    for IDA, IDB in zip(struct.A_dict_sis.keys(), struct.B_dict_sis.keys()):
        maxgen = min(len(struct.A_dict_sis[IDA]), len(struct.B_dict_sis[IDB]))
        A_abs_time = np.cumsum(struct.A_dict_sis[IDA]['generationtime'].loc[:maxgen])
        B_abs_time = np.cumsum(struct.B_dict_sis[IDB]['generationtime'].loc[:maxgen])
        frac_overlap = [np.abs((max(A_abs_time[gen], B_abs_time[gen])-min(A_abs_time[gen+1], B_abs_time[gen+1]))/(min(A_abs_time[gen], B_abs_time[gen])-max(
            A_abs_time[gen+1], B_abs_time[gen+1]))) for gen in range(maxgen-1)]
        sis_frac.append(frac_overlap)

    non_sis_frac = []
    for IDA, IDB in zip(struct.A_dict_non_sis.keys(), struct.B_dict_non_sis.keys()):
        maxgen = min(len(struct.A_dict_non_sis[IDA]), len(struct.B_dict_non_sis[IDB]))
        A_abs_time = np.cumsum(struct.A_dict_non_sis[IDA]['generationtime'].loc[:maxgen])
        B_abs_time = np.cumsum(struct.B_dict_non_sis[IDB]['generationtime'].loc[:maxgen])
        frac_overlap = [np.abs(
            (max(A_abs_time[gen], B_abs_time[gen]) - min(A_abs_time[gen + 1], B_abs_time[gen + 1])) / (min(A_abs_time[gen], B_abs_time[gen]) - max(
                A_abs_time[gen + 1], B_abs_time[gen + 1]))) for gen in range(maxgen - 1)]
        non_sis_frac.append(frac_overlap)

    both_frac = []
    for IDA, IDB in zip(struct.A_dict_both.keys(), struct.B_dict_both.keys()):
        maxgen = min(len(struct.A_dict_both[IDA]), len(struct.B_dict_both[IDB]))
        A_abs_time = np.cumsum(struct.A_dict_both[IDA]['generationtime'].loc[:maxgen])
        B_abs_time = np.cumsum(struct.B_dict_both[IDB]['generationtime'].loc[:maxgen])
        frac_overlap = [np.abs(
            (max(A_abs_time[gen], B_abs_time[gen]) - min(A_abs_time[gen + 1], B_abs_time[gen + 1])) / (min(A_abs_time[gen], B_abs_time[gen]) - max(
                A_abs_time[gen + 1], B_abs_time[gen + 1]))) for gen in range(maxgen - 1)]
        both_frac.append(frac_overlap)

    # The x-axis, mean and variance over all experiments for each generation
    x_axis = range(1,21)
    sis_y_axis = [np.mean([sis_frac[ID][gen] for ID in range(len(sis_frac)) if len(sis_frac[ID])>=20]) for gen in range(20)]
    non_sis_y_axis = [np.mean([non_sis_frac[ID][gen] for ID in range(len(non_sis_frac)) if len(non_sis_frac[ID])>=20]) for gen in range(20)]
    both_y_axis = [np.mean([both_frac[ID][gen] for ID in range(len(both_frac)) if len(both_frac[ID])>=20]) for gen in range(20)]
    sis_y_err = [np.var([sis_frac[ID][gen] for ID in range(len(sis_frac)) if len(sis_frac[ID])>=20]) for gen in range(20)]
    non_sis_y_err = [np.var([non_sis_frac[ID][gen] for ID in range(len(non_sis_frac)) if len(non_sis_frac[ID])>=20]) for gen in range(20)]
    both_y_err = [np.var([both_frac[ID][gen] for ID in range(len(both_frac)) if len(both_frac[ID])>=20]) for gen in range(20)]

    # Plot them
    plt.errorbar(x_axis, np.array(sis_y_axis), yerr=np.array(sis_y_err).transpose(), marker='.', color='blue', label='Sister', capsize=2)
    plt.errorbar(x_axis, non_sis_y_axis, yerr=np.array(non_sis_y_err).transpose(), marker='.', color='orange', label='Non-Sister', capsize=2)
    plt.errorbar(x_axis, both_y_axis, yerr=np.array(both_y_err).transpose(), marker='.', color='green', label='Control', capsize=2)
    plt.legend()
    plt.xticks(range(1,21,2))
    plt.xlabel('Generations')
    plt.ylabel('Fraction of Generation Time overlap')
    # plt.show()
    plt.savefig('Fraction of Generation Time overlap.png', dpi=300)
    
    
if __name__ == '__main__':
    main()
