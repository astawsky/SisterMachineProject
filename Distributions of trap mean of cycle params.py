
from __future__ import print_function

import numpy as np

import matplotlib.pyplot as plt

import pickle

import scipy.stats as stats


def Distribution1(diffs_sis, xlabel, abs_range, dataset):
    # PoolID == 1
    sis_label = dataset + r'$\mu=$' + '{:.2e}'.format(np.mean(diffs_sis)) + r', $\sigma=$' + '{:.2e}'.format(
        np.std(diffs_sis))

    plt.hist(x=diffs_sis, label=sis_label, weights=np.ones_like(diffs_sis) / float(len(diffs_sis)), range=abs_range)
    # arr_sis =
    # plt.close()
    #
    # plt.plot(np.array([(arr_sis[1][l] + arr_sis[1][l + 1]) / 2. for l in range(len(arr_sis[1]) - 1)]), arr_sis[0], label=sis_label, marker='.')
    plt.xlabel(xlabel)
    plt.ylabel('PDF (Weighted Histogram)')
    plt.legend()
    # plt.show()
    plt.savefig('mean of '+xlabel+' '+dataset+'distribution.png', dpi=300)
    plt.close()


def main():
    # Import the Refined Data
    pickle_in = open("metastructdata.pickle", "rb")
    struct = pickle.load(pickle_in)
    pickle_in.close()

    for cycle_param, abs_range in zip(['generationtime', 'growth_length', 'length_birth', 'length_final'], [[.1,1],[.5,2],[1.5,4],[3,7.5]]):

        # distribution of the mean of all the experiments (traps)
        dist_of_cycle_param = []
        for val in struct.dict_with_all_sister_traces.values():
            dist_of_cycle_param.append(np.mean(val[cycle_param]))
        dist_of_cycle_param_sis = np.array(dist_of_cycle_param)
        print(dist_of_cycle_param_sis.shape)

        dist_of_cycle_param = []
        for val in struct.dict_with_all_non_sister_traces.values():
            dist_of_cycle_param.append(np.mean(val[cycle_param]))
        dist_of_cycle_param_non_sis = np.array(dist_of_cycle_param)
        print(dist_of_cycle_param_non_sis.shape)

        dist_of_cycle_param_all = np.concatenate([dist_of_cycle_param_sis, dist_of_cycle_param_non_sis], axis=0)
        print(dist_of_cycle_param_all.shape)

        Distribution1(dist_of_cycle_param_sis, cycle_param, None, 'S: ')

        Distribution1(dist_of_cycle_param_non_sis, cycle_param, None, 'NS: ')

        Distribution1(dist_of_cycle_param_all, cycle_param, None, 'S & NS: ')

        # # distribution of the POOLED experiments (traps)
        # dist_of_cycle_param = []
        # for val in struct.dict_with_all_sister_traces.values():
        #     for element in val[cycle_param]:
        #         dist_of_cycle_param.append(element)
        # dist_of_cycle_param_sis = np.array(dist_of_cycle_param)
        # print(dist_of_cycle_param_sis.shape)
        #
        # dist_of_cycle_param = []
        # for val in struct.dict_with_all_non_sister_traces.values():
        #     for element in val[cycle_param]:
        #         dist_of_cycle_param.append(element)
        # dist_of_cycle_param_non_sis = np.array(dist_of_cycle_param)
        # print(dist_of_cycle_param_non_sis.shape)
        #
        # dist_of_cycle_param_all = np.concatenate([dist_of_cycle_param_sis, dist_of_cycle_param_non_sis], axis=0)
        # print(dist_of_cycle_param_all.shape)
        #
        # Distribution1(dist_of_cycle_param_sis, cycle_param, abs_range, 'S: ')
        #
        # Distribution1(dist_of_cycle_param_non_sis, cycle_param, abs_range, 'NS: ')
        #
        # Distribution1(dist_of_cycle_param_all, cycle_param, abs_range, 'S & NS: ')


if __name__ == '__main__':
    main()
