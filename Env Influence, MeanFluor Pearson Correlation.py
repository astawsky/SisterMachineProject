
from __future__ import print_function

import numpy as np

import matplotlib.pyplot as plt

import pickle

import scipy.stats as stats


def Distribution(diffs_sis, diffs_non_sis, diffs_both, xlabel, Nbins, abs_diff_range):
    # PoolID == 1
    sis_label = 'Sister ' + r'$\mu=$' + '{:.2e}'.format(np.mean(diffs_sis)) + r', $\sigma=$' + '{:.2e}'.format(
        np.std(diffs_sis))
    non_label = 'Non Sister ' + r'$\mu=$' + '{:.2e}'.format(
        np.mean(diffs_non_sis)) + r', $\sigma=$' + '{:.2e}'.format(np.std(diffs_non_sis))
    both_label = 'Control ' + r'$\mu=$' + '{:.2e}'.format(np.mean(diffs_both)) + r', $\sigma=$' + '{:.2e}'.format(
        np.std(diffs_both))

    arr_sis = plt.hist(x=diffs_sis, label=sis_label, weights=np.ones_like(diffs_sis) / float(len(diffs_sis)), bins=Nbins, range=abs_diff_range)
    arr_non_sis = plt.hist(x=diffs_non_sis, label=non_label,
                           weights=np.ones_like(diffs_non_sis) / float(len(diffs_non_sis)), bins=Nbins, range=abs_diff_range)
    arr_both = plt.hist(x=diffs_both, label=both_label, weights=np.ones_like(diffs_both) / float(len(diffs_both)), bins=Nbins, range=abs_diff_range)
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
    struct = pickle.load(pickle_in)
    pickle_in.close()

    sis_diff_per_exp = []
    for index in range(len(struct.Sisters)):
        # The division times of trace A/B in absolute time
        div_time_A = np.insert(np.array(struct.Sisters[index]['timeA'][0] + np.cumsum(np.array(
            struct.A_dict_sis['Sister_Trace_A_' + str(index)]['generationtime']))), 0, struct.Sisters[index]['timeA'][0])
        div_time_B = np.insert(np.array(struct.Sisters[index]['timeB'][0] + np.cumsum(np.array(
            struct.B_dict_sis['Sister_Trace_B_' + str(index)]['generationtime']))), 0,
                               struct.Sisters[index]['timeB'][0])

        # indices in the dataframe of the division times
        indices_A = np.concatenate([np.where(round(struct.Sisters[index]['timeA'], 2) == round(x, 2)) for x in
                                    div_time_A], axis=1)[0]
        indices_B = np.concatenate([np.where(round(struct.Sisters[index]['timeB'], 2) == round(x, 2)) for x in
                                    div_time_B], axis=1)[0]

        abs_time_start = max(indices_A[5], indices_B[5]) # start from the sixth generation
        
        stepsize = 10

        # print(np.arange(0, (len(struct.Sisters[index]['meanfluorescenceA'])-abs_time_start)/stepsize, dtype=np.dtype(np.int16)))
        # print(type(np.arange(0, (len(struct.Sisters[index]['meanfluorescenceA'])-abs_time_start)/stepsize, dtype=np.dtype(np.int16))))

        averaged_sis_A = np.concatenate(np.array([struct.Sisters[index]['meanfluorescenceA'][abs_time_start+k*stepsize:abs_time_start+(k+1)*stepsize] for k in \
                np.arange(0, (len(struct.Sisters[index]['meanfluorescenceA'])-abs_time_start)/stepsize, dtype=np.dtype(np.int16))]))

        averaged_sis_B = np.concatenate(np.array(
            [struct.Sisters[index]['meanfluorescenceB'][abs_time_start + k * stepsize:abs_time_start + (k + 1) * stepsize] for k in \
             np.arange(0, (len(struct.Sisters[index]['meanfluorescenceB']) - abs_time_start) / stepsize, dtype=np.dtype(np.int16))]))

        # sis_diff_experiment = stats.pearsonr(struct.Sisters[index]['meanfluorescenceA'][abs_time_start:], struct.Sisters[index]['meanfluorescenceB'][
        #                                                                                                 abs_time_start:])[0]

        sis_diff_experiment = stats.pearsonr(averaged_sis_A, averaged_sis_B)[0]

        sis_diff_per_exp.append(sis_diff_experiment)

    sis_diff = sis_diff_per_exp

    # plt.hist(sis_diff)
    # plt.show()

    non_sis_diff_per_exp = []
    for index in range(len(struct.Nonsisters)-1):
        # The division times of trace A/B in absolute time
        div_time_A = np.insert(np.array(struct.Nonsisters[index]['timeA'][0] + np.cumsum(np.array(
            struct.A_dict_non_sis['Non-sister_Trace_A_' + str(index)]['generationtime']))), 0, struct.Nonsisters[index]['timeA'][0])
        div_time_B = np.insert(np.array(struct.Nonsisters[index]['timeB'][0] + np.cumsum(np.array(
            struct.B_dict_non_sis['Non-sister_Trace_B_' + str(index)]['generationtime']))), 0,
                               struct.Nonsisters[index]['timeB'][0])

        # indices in the dataframe of the division times
        indices_A = np.concatenate([np.where(round(struct.Nonsisters[index]['timeA'], 2) == round(x, 2)) for x in
                                    div_time_A], axis=1)[0]
        indices_B = np.concatenate([np.where(round(struct.Nonsisters[index]['timeB'], 2) == round(x, 2)) for x in
                                    div_time_B], axis=1)[0]

        abs_time_start = max(indices_A[5], indices_B[5])  # start from the sixth generation

        stepsize = 10

        averaged_non_sis_A = np.concatenate(np.array(
            [struct.Nonsisters[index]['meanfluorescenceA'][abs_time_start + k * stepsize:abs_time_start + (k + 1) * stepsize] for k in \
             np.arange(0, (len(struct.Nonsisters[index]['meanfluorescenceA'])-abs_time_start) /stepsize, dtype=np.dtype(np.int16))]))

        averaged_non_sis_B = np.concatenate(np.array([struct.Nonsisters[index]['meanfluorescenceB'][abs_time_start + k * stepsize:abs_time_start + (k + 1) * stepsize] for \
                                   k in np.arange(0, (len(struct.Nonsisters[index]['meanfluorescenceB'])-abs_time_start) /stepsize, dtype=np.dtype(
                np.int16))]))

        # non_sis_diff_experiment = stats.pearsonr(struct.Nonsisters[index]['meanfluorescenceA'][abs_time_start:], struct.Nonsisters[index][
        #                                                                                                 'meanfluorescenceB'][abs_time_start:])[0]
        
        non_sis_diff_experiment = stats.pearsonr(averaged_non_sis_A, averaged_non_sis_B)[0]


        non_sis_diff_per_exp.append(non_sis_diff_experiment)

    non_sis_diff = non_sis_diff_per_exp

    # plt.hist(non_sis_diff)
    # plt.show()

    both_diff_per_exp = []
    for indexA, indexB in zip(struct.Control[0], struct.Control[1]):
        # The division times of trace A/B in absolute time
        div_time_A = np.insert(np.array(struct.Sisters[indexA]['timeA'][0] + np.cumsum(np.array(
            struct.A_dict_both['Sister_Trace_A_' + str(indexA)]['generationtime']))), 0,
                               struct.Sisters[indexA]['timeA'][0])
        div_time_B = np.insert(np.array(struct.Nonsisters[indexB]['timeA'][0] + np.cumsum(np.array(
            struct.B_dict_both['Non-sister_Trace_A_' + str(indexB)]['generationtime']))), 0,
                               struct.Nonsisters[indexB]['timeA'][0])

        # indices in the dataframe of the division times
        indices_A = np.concatenate([np.where(round(struct.Sisters[indexA]['timeA'], 2) == round(x, 2)) for x in
                                    div_time_A], axis=1)[0]
        indices_B = np.concatenate([np.where(round(struct.Nonsisters[indexB]['timeA'], 2) == round(x, 2)) for x in
                                    div_time_B], axis=1)[0]

        abs_time_start = max(indices_A[5], indices_B[5])  # start from the sixth generation

        stop_time = min((len(struct.Sisters[indexA]['meanfluorescenceA'])-abs_time_start), len(struct.Nonsisters[indexB]['meanfluorescenceB']))

        stepsize = 10

        averaged_both_A = np.concatenate(np.array(
            [struct.Sisters[indexA]['meanfluorescenceA'][abs_time_start + k * stepsize:abs_time_start + (k + 1) * stepsize] for k in \
             np.arange(0, np.floor((stop_time-abs_time_start) /stepsize), dtype=np.dtype(np.int16))]))

        averaged_both_B = np.concatenate(np.array(
            [struct.Nonsisters[indexB]['meanfluorescenceA'][abs_time_start + k * stepsize:abs_time_start + (k + 1) * stepsize] for \
             k in np.arange(0, np.floor((stop_time-abs_time_start) /stepsize), dtype=np.dtype(np.int16))]))

        # both_diff_experiment = stats.pearsonr(struct.Sisters[indexA]['meanfluorescenceA'][abs_time_start:stop_time], struct.Nonsisters[indexB][
        #                                                                                                        'meanfluorescenceA'][
        #                                                                                                              abs_time_start:stop_time])[0]

        both_diff_experiment = stats.pearsonr(averaged_both_A, averaged_both_B)[0]

        both_diff_per_exp.append(both_diff_experiment)

    both_diff = both_diff_per_exp

    if np.isnan(np.sum(both_diff)):
        both_diff = both_diff[~np.isnan(both_diff)]
        print('both has nans')
    if np.isnan(np.sum(non_sis_diff)):
        non_sis_diff = non_sis_diff[~np.isnan(non_sis_diff)]
        print('non_sis has nans')
    if np.isnan(np.sum(sis_diff)):
        sis_diff = sis_diff[~np.isnan(sis_diff)]
        print('sis has nans')

    # plt.hist(both_diff)
    # plt.show()

    xlabel = 'Non-Abs Difference between pairs'
    Nbins = None
    abs_diff_range = [-1, 1]
    Distribution(sis_diff, non_sis_diff, both_diff, xlabel, Nbins, abs_diff_range)


if __name__ == '__main__':
    main()
