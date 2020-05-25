
from __future__ import print_function

import numpy as np

import matplotlib.pyplot as plt

import pickle

import scipy.stats as stats


# def Distribution(diffs_sis, diffs_non_sis, diffs_both, xlabel, Nbins, abs_diff_range, title, filename):
#     # PoolID == 1
#     sis_label = 'Sister ' + r'$\mu=$' + '{:.2e}'.format(np.mean(diffs_sis)) + r', $\sigma=$' + '{:.2e}'.format(
#         np.std(diffs_sis))
#     non_label = 'Non Sister ' + r'$\mu=$' + '{:.2e}'.format(
#         np.mean(diffs_non_sis)) + r', $\sigma=$' + '{:.2e}'.format(np.std(diffs_non_sis))
#     both_label = 'Control ' + r'$\mu=$' + '{:.2e}'.format(np.mean(diffs_both)) + r', $\sigma=$' + '{:.2e}'.format(
#         np.std(diffs_both))
#
#     arr_sis = plt.hist(x=diffs_sis, label=sis_label, weights=np.ones_like(diffs_sis) / float(len(diffs_sis)), bins=Nbins, range=abs_diff_range)
#     arr_non_sis = plt.hist(x=diffs_non_sis, label=non_label,
#                            weights=np.ones_like(diffs_non_sis) / float(len(diffs_non_sis)), bins=Nbins, range=abs_diff_range)
#     arr_both = plt.hist(x=diffs_both, label=both_label, weights=np.ones_like(diffs_both) / float(len(diffs_both)), bins=Nbins, range=abs_diff_range)
#     plt.close()
#
#     # print('arr_sis[0]:', arr_sis[0])
#     # print('arr_non_sis[0]:', arr_non_sis[0])
#     # print('arr_both[0]:', arr_both[0])
#
#     plt.plot(np.array([(arr_sis[1][l] + arr_sis[1][l + 1]) / 2. for l in range(len(arr_sis[1]) - 1)]), arr_sis[0], label=sis_label, marker='.')
#     plt.plot(np.array([(arr_non_sis[1][l] + arr_non_sis[1][l + 1]) / 2. for l in range(len(arr_non_sis[1]) - 1)]), arr_non_sis[0], label=non_label, marker='.')
#     plt.plot(np.array([(arr_both[1][l] + arr_both[1][l + 1]) / 2. for l in range(len(arr_both[1]) - 1)]), arr_both[0], label=both_label, marker='.')
#     plt.xlabel(xlabel)
#     plt.ylabel('PDF (Weighted Histogram)')
#     plt.title(title)
#     plt.legend()
#     # plt.show()
#     plt.savefig(filename+'.png', dpi=300)


def Distribution(together, diffs_both, xlabel, Nbins, abs_diff_range, title, filename):
    # PoolID == 1
    sis_label = 'Sis and NS ' + r'$\mu=$' + '{:.2e}'.format(np.mean(together)) + r', $\sigma=$' + '{:.2e}'.format(
        np.std(together))
    both_label = 'Control ' + r'$\mu=$' + '{:.2e}'.format(np.mean(diffs_both)) + r', $\sigma=$' + '{:.2e}'.format(
        np.std(diffs_both))

    arr_sis = plt.hist(x=together, label=sis_label, weights=np.ones_like(together) / float(len(together)), bins=Nbins, range=abs_diff_range)
    arr_both = plt.hist(x=diffs_both, label=both_label, weights=np.ones_like(diffs_both) / float(len(diffs_both)), bins=Nbins, range=abs_diff_range)
    plt.close()

    # print('arr_sis[0]:', arr_sis[0])
    # print('arr_non_sis[0]:', arr_non_sis[0])
    # print('arr_both[0]:', arr_both[0])

    plt.plot(np.array([(arr_sis[1][l] + arr_sis[1][l + 1]) / 2. for l in range(len(arr_sis[1]) - 1)]), arr_sis[0], label=sis_label, marker='.')
    plt.plot(np.array([(arr_both[1][l] + arr_both[1][l + 1]) / 2. for l in range(len(arr_both[1]) - 1)]), arr_both[0], label=both_label, marker='.')
    plt.xlabel(xlabel)
    plt.ylabel('PDF (Weighted Histogram)')
    plt.title(title)
    plt.legend()
    plt.show()
    # plt.savefig(filename+'.png', dpi=300)


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

        sis_diff_experiment = np.array(struct.Sisters[index]['meanfluorescenceA'][abs_time_start:] - struct.Sisters[index]['meanfluorescenceB'][
                                                                                           abs_time_start:])

        sis_diff_per_exp.append(sis_diff_experiment)

    print(len(sis_diff_per_exp))
    sis_diff_per_exp = np.array(sis_diff_per_exp)
    print(len(sis_diff_per_exp))
    sis_diff = np.concatenate(sis_diff_per_exp)
    print(len(sis_diff))

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

        non_sis_diff_experiment = np.array(struct.Nonsisters[index]['meanfluorescenceA'][abs_time_start:] - struct.Nonsisters[index]['meanfluorescenceB'][
                                                                                                     abs_time_start:])

        non_sis_diff_per_exp.append(non_sis_diff_experiment)

    print(len(non_sis_diff_per_exp))
    non_sis_diff_per_exp = np.array(non_sis_diff_per_exp)
    print(len(non_sis_diff_per_exp))
    non_sis_diff = np.concatenate(non_sis_diff_per_exp)
    print(len(non_sis_diff))

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

        stop_time = min(len(struct.Sisters[indexA]['meanfluorescenceA']), len(struct.Nonsisters[indexB]['meanfluorescenceB']))

        both_diff_experiment = np.array(struct.Sisters[indexA]['meanfluorescenceA'][abs_time_start:] - struct.Nonsisters[indexB]['meanfluorescenceB'][
                                                                                                     abs_time_start:])

        both_diff_per_exp.append(both_diff_experiment)

    print(len(both_diff_per_exp))
    both_diff_per_exp = np.array(both_diff_per_exp)
    print(len(both_diff_per_exp))
    both_diff = np.concatenate(both_diff_per_exp)
    print(len(both_diff))

    if np.isnan(np.sum(both_diff)):
        both_diff = both_diff[~np.isnan(both_diff)]
        print('both has nans')
    if np.isnan(np.sum(non_sis_diff)):
        non_sis_diff = non_sis_diff[~np.isnan(non_sis_diff)]
        print('non_sis has nans')
    if np.isnan(np.sum(sis_diff)):
        sis_diff = sis_diff[~np.isnan(sis_diff)]
        print('sis has nans')

    print(len(both_diff))

    together = np.append(sis_diff, non_sis_diff)
    print(len(together))

    # plt.hist(both_diff)
    # plt.show()

    xlabel = 'Non-Abs Difference between pairs'
    Nbins = 20
    abs_diff_range = [-50000, 50000]
    # Distribution(sis_diff, non_sis_diff, both_diff, xlabel, Nbins, abs_diff_range, title='Mean Fluorescence', filename='Difference '
    #                                                                                                                    'in MeanFluorescence '
    #                                                                                                                    'Histogram, Env Influence')
    Distribution(together, both_diff, xlabel, Nbins, abs_diff_range, title='Mean Fluorescence', filename='Difference '
                                                                                                                       'in MeanFluorescence '
                                                                                                                       'Histogram, Env Influence')


if __name__ == '__main__':
    main()
